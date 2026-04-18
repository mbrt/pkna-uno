# Fine-Tuning Pipeline Design: Uno as a Social Orchestrator

High-level design for training a small, personality-shaped model via on-policy
distillation. The model should behave like Uno, have strong emotional
intelligence, use tools to search factual knowledge (wiki), and delegate
technical tasks to specialized sub-agents.

## Context

This design draws from:

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) -- Thinking Machines Lab
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) -- Thinking Machines Lab
- [Uno: What I Learned Shaping LLMs into a 90s Comic Book AI](https://blog.mbrt.dev/posts/uno/)
- [Qwen3.5 model collection](https://huggingface.co/collections/Qwen/qwen35)
- [Qwen3.5: Towards Native Multimodal Agents](https://www.alibabacloud.com/blog/602894) -- official benchmarks
- [Qwen3.6 model collection](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) -- drop-in successor to Qwen3.5 MoE (April 2026)
- [Gemma 4](https://deepmind.google/models/gemma/gemma-4/) -- Google DeepMind (April 2026)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) -- architecture and benchmarks
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) -- LoRA config, VRAM, QLoRA warnings
- [Unsloth Qwen3.6 inference & training](https://unsloth.ai/docs/models/qwen3.6) -- GGUF, llama.cpp, Unsloth Studio
- [Unsloth Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train) -- MoE LoRA issues, bug fixes
- [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) -- quantization quality analysis (applies to Qwen3.6, same arch)

## Goals

Train a model that:

1. **Embodies Uno's personality** -- tone, humor, emotional responses, and
   relationship-aware register shifts are baked into the weights, not dependent
   on a system prompt.
2. **Reasons about social situations** -- the model uses internal thinking to
   evaluate the emotional context, the relationship with the interlocutor, and
   the best conversational strategy before responding. This reasoning is hidden
   from the user; only the final in-character response is visible.
3. **Uses tools for knowledge** -- factual information about the PKNA universe
   lives in the wiki and is retrieved via search tools at inference time, not
   memorized during training.
4. **Orchestrates sub-agents** -- the model delegates technical work (coding,
   research, math, etc.) to specialized sub-agents via tool calls, acting as a
   social front-end rather than a generalist.

What we are *not* training:

- A knowledge base. The wiki stays external.
- A generalist assistant. Raw problem-solving is delegated.
- A general-purpose reasoner. The thinking budget is spent on emotional
  grounding and social strategy, not on solving math or logic puzzles.

## Architecture Overview

```
User <-> [Uno model] <-> Tools
                          |-- search_knowledge(query) -> wiki segments
                          |-- read_knowledge(id) -> full article
                          |-- delegate(task, context) -> sub-agent result
                          |-- remember(key, value) -> memory store
                          |-- recall(query) -> memory retrieval
```

The Uno model is the only component that talks to the user. It decides when to
search for facts, when to delegate, and how to phrase everything in character.
Sub-agents are opaque to the user.

Uno always acts as if he is in the Ducklair Tower within the PKNA universe, but
his actual capabilities are limited to his available tools. He does not claim to
operate fictional systems (activate shields, detect Evronians via sensors). In
deployment, users fall into two categories: **casual users** (fans and curious
strangers talking to Uno as a character chatbot) and **roleplay users**
(pretending to be PKNA characters like Paperino, Everett, or Due). In the
roleplay case, the *user* adopts a character identity and Uno treats them
accordingly, informed by his character profile.

## Roadmap

1. Start with **Qwen3.5-4B** student and **self-distillation** (Scenario A).
   This is the cheapest path and validates the pipeline end-to-end.
2. Build the SFT dataset from existing scenes + synthetic conversations.
3. Train SFT, evaluate personality and tool use.
4. Run on-policy distillation to recover any lost capabilities.
5. If tool use is insufficient, switch to **27B teacher** (Scenario B).
6. If personality depth is insufficient, try **9B student** (Scenario C).
   The 9B is the natural next step before MoE -- same training infrastructure,
   just slower.
7. If 9B still falls short on personality nuance, try **Qwen3.6-35B-A3B**
   (Scenario D). This requires a 4xL40S node for training but delivers the
   best benchmarks. Qwen3.6 is the preferred MoE base over Qwen3.5: same
   architecture (training infra carries over), better tool calling (MCPMark
   37.0 vs 27.0), and stronger agentic coding (SWE-bench 73.4 vs 70.0).
8. If 4B self-distill works well, try **2B** for on-device deployment.

## Implementation Status

### Evaluation Pipeline (evals.md)

- [x] **Eval type system** -- `pkna/eval/types.py`: Pydantic models for the
  full 3-stage pipeline (`EvalPrompt`, `EvalTrace`, `JudgeScore`,
  `ScoredTrace`, `SuiteResult`, `EvalReport`). All 6 suites defined.
- [x] **Stage 1: Generate eval prompts** -- `evals/generate_eval_prompts.py`:
  generates per-suite JSONL prompt banks for all 6 suites (personality, social
  reasoning, tool use, memory handling, stability, language).
- [x] **Stage 2: Run inference** -- `evals/run_eval_inference.py`: loads
  prompts, composes context (system prompt + tools + memory bank), calls the
  backend, writes `EvalTrace` JSONL. Supports resume (skip completed IDs).
- [x] **Stage 2: Thinking/tool-call capture** -- `GenerateResult` carries
  `thinking`, `tool_calls`, and `messages`; both Gemini (manual tool loop +
  `ThinkingConfig`) and Anthropic (thinking blocks + tool-call recording)
  backends capture full traces; `run_single_prompt` propagates them into
  `EvalTrace`.
- [x] **Stage 2: Multi-turn stability** -- `datagen/user_simulator.py`: user
  simulator with per-turn adversarial directives (jailbreak, escalate, derail,
  challenge_identity, flatter, continue); `run_multi_turn` in
  `evals/run_eval_inference.py` drives 10-turn conversations; stability prompts
  in `evals/generate_eval_prompts.py` carry `multi_turn`, `turn_count`, and
  `directives` metadata; `run_single_prompt` dispatches automatically.
- [x] **Stage 3: Score traces** -- `evals/score_eval_traces.py`: per-suite
  programmatic scoring (tool-use accuracy, memory triplet pass/fail via
  binary judge calls) and LLM-as-judge scoring against the rubrics defined
  in `evals.md`. Stability suite scores each assistant turn individually.
  Supports resume (skip already-scored prompt IDs).
- [x] **Stage 3: Aggregate report** -- `evals/score_eval_traces.py`:
  `aggregate_report()` computes per-suite `SuiteResult` metrics (mean
  scores, sub-dimension means, programmatic accuracy, triplet pass rate,
  stability turn-level stats) and flags traces scoring <= 2. Writes
  `report.json` as `EvalReport`.

### Tool Infrastructure (dataset-generation-agent.md, architecture)

- [x] **Wiki tools** -- `pkna/inference/tools.py`: in-memory PKNA wiki index with
  `search_knowledge` and `read_knowledge`. Loads markdown from `results/wiki/`.
  Both wiki and memory search use BM25 ranking (`rank-bm25`) for
  production-quality retrieval.
- [x] **Memory bank** -- `pkna/inference/memory.py`: JSONL-backed episodic memory
  with BM25-based `search`, `append`, `recall`, `remember`. Supports eval mode
  (read-only).
- [x] **Memory compaction** -- `pkna/inference/memory_compaction.py`: LLM-driven
  summarization of raw memory banks into concise context for the system prompt.
- [x] **Eval tool registry** -- `pkna/inference/tools.py`: wires tool names from
  `EvalPrompt` to real callables (`search_knowledge`, `read_knowledge`,
  `delegate`, `recall`, `remember`).
- [x] **Delegate stub** -- `pkna/inference/tools.py`: `delegate()` returns a
  placeholder response; real sub-agent routing is not implemented.
- [x] **System prompts** -- `pkna/inference/system_prompts.py`: minimal and full
  templates with per-suite mapping and `render_system_prompt()`.

### Memory Banks (dataset-generation-agent.md, memory consolidation)

- [x] **Seed memory banks on disk** -- `data/memory_banks/`: 3 seed banks
  (`paperino_recent`, `xadhoom_research`, `mixed_irrelevant`), used as input
  to the corpus generator.
- [x] **Memory corpus generator** -- `datagen/generate_memory_corpus.py`:
  ingests seed banks and LLM-generates additional entries (roleplay characters
  and casual users) to produce `output/datagen/memory_corpus.jsonl` (~300-500
  tagged entries). Supersedes `evals/generate_memory_banks.py`.
- [x] **Dynamic memory composition** -- `pkna/datagen/memory.py`:
  `compose_memory()` samples relevant and irrelevant entries from the corpus
  per-trace, producing both a `memory_context` preamble and a `MemoryBank`
  instance wired to the `recall` tool.

### LLM Backends

- [x] **Gemini backend** -- `pkna/llm/backends.py`: Google `genai` with tool
  support and retry logic.
- [x] **Anthropic backend** -- `pkna/llm/backends.py`: Anthropic via Bedrock
  with full tool-use loop.

### Upstream Data

- [x] **Scene extraction** -- `pkna/extract/scenes.py`: structures for
  emotionally annotated comic scenes (panels, dialogue, annotations).
- [x] **Wiki corpus** -- `results/wiki/`: fandom + Wikipedia articles on PKNA
  characters, technology, locations.
- [x] **Character profile** -- `results/uno_soul_document.md`,
  `results/final_ledger.json`, `results/refined_ledger.json`.

### Dataset Generation Agent (dataset-generation-agent.md)

- [x] **Data types** -- `pkna/datagen/types.py`: Pydantic models for the
  datagen pipeline (`DatagenPrompt`, `DatagenTrace`, `QualityScore`,
  `ScoredTrace`).
- [x] **Context composer** -- `datagen/run_datagen.py`:
  `compose_datagen_context()` builds the full system prompt using the full
  template with user summary and memory context slots.
- [x] **Prompt bank** -- `datagen/generate_prompts.py`: 70+ manual prompts
  across 9 categories (emotional, factual, delegation, identity, adversarial,
  register shift, memory, multi-turn, casual); scene-derived prompt extraction
  from `output/extract-emotional/v2/` (365 scenes available); LLM-generated
  prompts from 40 scenario templates (opt-in via `--include-generated`).
- [x] **Execution loop** -- `datagen/run_datagen.py`: runs strong model through
  prompts, records thinking + tool calls + responses as `DatagenTrace` JSONL.
  Supports single-turn and multi-turn (via user simulator). Uses
  `eval_mode=False` so memory tools actually write. Resume support via
  completed ID tracking.
- [x] **User simulator** -- `datagen/user_simulator.py`: generates user
  messages given a conversation history, user profile, and per-turn directive.
  Used by the eval stability suite and the dataset generation execution loop.
- [x] **Quality filtering** -- `datagen/filter_traces.py`: programmatic checks
  (response length 10-500 tokens, language consistency via Italian/English
  heuristic) plus LLM-as-judge scoring (character consistency 1-5, thinking
  quality 1-5, tool correctness pass/fail/na). Outputs scored traces and
  filtered (passing only) traces. Resume support for incremental scoring.
- [ ] **Background chat** -- Tulu3 subset sampling with lightweight thinking
  traces is not implemented.

### SFT Training (training-strategy.md)

- [x] **SFT dataset assembly** -- `pkna/training/sft_dataset.py`:
  `trace_to_messages` converts `DatagenTrace` to Qwen3.5 chat format
  (thinking -> `reasoning_content`, tool calls -> `function` wrapper).
  `training/assemble_sft.py`: loads filtered traces, renders via the
  tokenizer's chat template with `enable_thinking=True`, drops examples
  exceeding `max_seq_length`, saves as HuggingFace `Dataset`.
  Tests in `tests/test_sft_dataset.py`.
- [x] **SFT training script** -- `training/run_sft.py`: Unsloth +
  BF16 LoRA (rank=64, alpha=32, all-linear, dropout=0),
  `train_on_responses_only` loss masking, linear LR schedule with AdamW
  8-bit, gradient checkpointing (`"unsloth"`), MLflow tracking, GGUF
  export via `--export-gguf`. Hyperparameters match `training-strategy.md`.
- [x] **End-to-end smoke test** -- `training/smoke_test.py`: runs all 7
  pipeline stages (prompts, datagen, filter, assemble, train, eval
  inference, eval scoring) with fake backends and canned responses.
  Validates real I/O, serialization, and orchestration. Supports
  `--stage`, `--no-training`, `--all` flags.

### On-Policy Distillation (training-strategy.md)

- [ ] **Distillation loop** -- student sampling, teacher logprob computation,
  reverse KL training is not implemented.
- [ ] **Distillation prompt set** -- the ~600 prompts for distillation
  (character interviews, scenarios, tool-use, general chat) are not produced.

### Deployment (training-strategy.md)

- [ ] **GGUF export** -- quantization and export for llama.cpp is not
  implemented. Note: Qwen3.6 GGUFs currently require llama.cpp-compatible
  backends (no Ollama support yet due to separate mmproj vision files).

## Sub-Documents

| Document | Contents |
|---|---|
| [Model Selection](fine-tuning/model-selection.md) | Student & teacher model analysis, benchmarks, tradeoffs |
| [SFT Dataset](fine-tuning/sft-dataset.md) | Dataset construction, data mix, what goes into vs. stays out of the weights |
| [Dataset Generation Agent](fine-tuning/dataset-generation-agent.md) | Trace capture pipeline: context composer, prompt bank, execution loop, quality filtering |
| [Training Strategy](fine-tuning/training-strategy.md) | Two-stage pipeline (SFT + on-policy distillation), LoRA config, hyperparameters |
| [Evaluation](fine-tuning/evals.md) | Evaluation dimensions, protocol, baselines |
| [Infrastructure & Costs](fine-tuning/infra-costs.md) | Hardware options, cost breakdown per scenario, experimentation budget |
| [Risks & Open Questions](fine-tuning/risks-and-open-questions.md) | Known risks, research questions, future work |
