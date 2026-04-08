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
