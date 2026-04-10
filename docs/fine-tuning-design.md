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
- [Gemma 4](https://deepmind.google/models/gemma/gemma-4/) -- Google DeepMind (April 2026)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) -- architecture and benchmarks
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) -- LoRA config, VRAM, QLoRA warnings
- [Unsloth Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train) -- MoE LoRA issues, bug fixes
- [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) -- quantization quality analysis

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
                          |-- search_wiki(query) -> wiki segments
                          |-- read_wiki(id) -> full article
                          |-- delegate(task, agent_type) -> sub-agent result
                          |-- remember(key, value) -> memory store
                          |-- recall(query) -> memory retrieval
```

The Uno model is the only component that talks to the user. It decides when to
search for facts, when to delegate, and how to phrase everything in character.
Sub-agents are opaque to the user.

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
7. If 9B still falls short on personality nuance, try **35B-A3B** (Scenario D).
   This requires a 4xL40S node for training but delivers the best benchmarks.
8. If 4B self-distill works well, try **2B** for on-device deployment.

## Implementation Status

### Evaluation Pipeline (evals.md)

- [x] **Eval type system** -- `pkna/eval_types.py`: Pydantic models for the
  full 3-stage pipeline (`EvalPrompt`, `EvalTrace`, `JudgeScore`,
  `ScoredTrace`, `SuiteResult`, `EvalReport`). All 6 suites defined.
- [x] **Stage 1: Generate eval prompts** -- `evals/generate_eval_prompts.py`:
  generates per-suite JSONL prompt banks for all 6 suites (personality, social
  reasoning, tool use, memory handling, stability, language).
- [x] **Stage 2: Run inference** -- `evals/run_eval_inference.py`: loads
  prompts, composes context (system prompt + tools + memory bank), calls the
  backend, writes `EvalTrace` JSONL. Supports resume (skip completed IDs).
- [ ] **Stage 2: Thinking/tool-call capture** -- the inference harness records
  the visible response but does not yet capture thinking traces or tool call
  details in the `EvalTrace`.
- [ ] **Stage 2: Multi-turn stability** -- the stability suite requires a user
  simulator driving 10-20 turn conversations; this is not yet implemented.
- [ ] **Stage 3: Score traces** -- no scoring script exists. Needs programmatic
  scoring (tool-use accuracy, memory triplet pass/fail) and LLM-as-judge
  scoring against the rubrics defined in `evals.md`.
- [ ] **Stage 3: Aggregate report** -- the `EvalReport` schema is defined but
  no code produces it.

### Tool Infrastructure (dataset-generation-agent.md, architecture)

- [x] **Wiki tools** -- `pkna/wiki_tools.py`: in-memory PKNA wiki index with
  `search_knowledge` and `read_knowledge`. Loads markdown from `results/wiki/`.
- [x] **Memory bank** -- `pkna/memory_bank.py`: JSONL-backed episodic memory
  with `search`, `append`, `recall`, `remember`. Supports eval mode (read-only).
- [x] **Memory compaction** -- `pkna/memory_compaction.py`: LLM-driven
  summarization of raw memory banks into concise context for the system prompt.
- [x] **Eval tool registry** -- `pkna/eval_tools.py`: wires tool names from
  `EvalPrompt` to real callables (`search_wiki`, `read_wiki`, `delegate`,
  `recall`, `remember`).
- [x] **Delegate stub** -- `pkna/eval_tools.py`: `delegate()` returns a
  placeholder response; real sub-agent routing is not implemented.
- [x] **System prompts** -- `pkna/system_prompts.py`: minimal and full
  templates with per-suite mapping and `render_system_prompt()`.

### Memory Banks (dataset-generation-agent.md, memory consolidation)

- [x] **Synthetic memory bank generator** -- `evals/generate_memory_banks.py`:
  LLM-driven generation of JSONL memory banks from scenario descriptions.
- [x] **Generated banks on disk** -- `data/memory_banks/`: 3 banks
  (`paperino_recent`, `xadhoom_research`, `mixed_irrelevant`).
- [ ] **Full bank coverage** -- the design calls for ~50 raw memory sets with
  varied characters, events, and emotional states; only 3 exist.

### LLM Backends

- [x] **Gemini backend** -- `pkna/llm_backends.py`: Google `genai` with tool
  support and retry logic.
- [x] **Anthropic backend** -- `pkna/llm_backends.py`: Anthropic via Bedrock
  with full tool-use loop.

### Upstream Data

- [x] **Scene extraction** -- `pkna/pkna_scenes.py`: structures for
  emotionally annotated comic scenes (panels, dialogue, annotations).
- [x] **Wiki corpus** -- `results/wiki/`: fandom + Wikipedia articles on PKNA
  characters, technology, locations.
- [x] **Character profile** -- `results/uno_soul_document.md`,
  `results/final_ledger.json`, `results/refined_ledger.json`.

### Dataset Generation Agent (dataset-generation-agent.md)

- [ ] **Context composer** -- no standalone implementation; the eval inference
  harness has a `compose_context` function, but the full context composer
  (system prompt + user summary + memory context + tool declarations +
  conversation history) for SFT data generation is not built.
- [ ] **Prompt bank** -- manual prompts (~100-150), generated prompts
  (~400-500), and scene-derived prompts (~229) are not yet produced for
  training. Only eval prompts exist.
- [ ] **Execution loop** -- the trace capture pipeline (run strong model through
  scenarios, record thinking + tool calls + responses as JSONL training
  examples) is not implemented.
- [ ] **User simulator** -- for multi-turn arcs, a separate LLM call plays the
  user role; not implemented.
- [ ] **Quality filtering** -- LLM-as-judge filtering of raw traces (character
  consistency, thinking quality, tool correctness, length, language) is not
  implemented.
- [ ] **Background chat** -- Tulu3 subset sampling with lightweight thinking
  traces is not implemented.

### SFT Training (training-strategy.md)

- [ ] **SFT training script** -- no training code exists (Unsloth + LoRA
  configuration, loss masking, thinking-enabled chat template).
- [ ] **SFT dataset assembly** -- converting filtered traces into the training
  format with proper loss masks (user/tool masked, assistant trained) is not
  implemented.

### On-Policy Distillation (training-strategy.md)

- [ ] **Distillation loop** -- student sampling, teacher logprob computation,
  reverse KL training is not implemented.
- [ ] **Distillation prompt set** -- the ~600 prompts for distillation
  (character interviews, scenarios, tool-use, general chat) are not produced.

### Deployment (training-strategy.md)

- [ ] **GGUF export** -- quantization and export for llama.cpp / Ollama is not
  implemented.

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
