# Training Strategy

Back to [Fine-Tuning Design](../fine-tuning-design.md).

## Pipeline Overview

Training proceeds in two stages:

1. **Off-Policy SFT** -- Teach the student Uno's personality, tone, and
   interaction patterns using a curated dataset (see [SFT Dataset](sft-dataset.md)).
2. **On-Policy Distillation** -- Recover instruction-following and tool-use
   capabilities degraded by SFT, and sharpen character adherence with dense
   per-token feedback from a teacher.

## Stage 1: Off-Policy SFT (Personality Mid-Training)

**Goal**: Teach the student Uno's personality, tone, and interaction patterns,
including internal social reasoning via thinking traces.

**Loss**: Standard cross-entropy (forward KL), applied to both thinking traces
and visible responses (see loss masking notes below).

See [SFT Dataset](sft-dataset.md) for dataset construction, thinking trace
format, and data mix.

## Stage 2: On-Policy Distillation (Behavior Recovery + Sharpening)

**Goal**: Recover any instruction-following and tool-use capabilities degraded
by SFT, and sharpen character adherence with dense per-token feedback.

**Method**: Sample rollouts from the student, compute teacher logprobs on those
rollouts, train with reverse KL as per-token advantage.

```
for each batch of prompts:
    trajectories = student.sample(prompts)           # student generates
    student_logprobs = trajectories.logprobs          # already computed
    teacher_logprobs = teacher.compute_logprobs(trajectories)  # single forward pass
    advantage = -(student_logprobs - teacher_logprobs)         # negative reverse KL
    student.train(trajectories, advantage)            # policy gradient update
```

### Prompt Sources for Distillation

| Source | Prompts | Purpose |
|---|---|---|
| Character interview questions | ~100 | Test personality consistency |
| Scenario starters (crisis, casual, technical) | ~100 | Test situational adaptation |
| Tool-use prompts (wiki search, delegation) | ~100 | Recover/sharpen tool calling |
| General chat (Tulu3 subset) | ~300 | Recover instruction following |
| **Total** | **~600** | |

**Sampling**: 4 rollouts per prompt, ~150-300 training steps.

### Teacher Choice for This Stage

- Option A: Qwen3.5-27B (strongest signal, requires 4xL40S)
- Option B: Original Qwen3.5-4B pre-SFT (self-distillation, single GPU, cheaper)
- Recommendation: Start with Option B. If tool-use recovery is insufficient,
  switch to Option A.

See [Model Selection](model-selection.md) for details on teacher models and
self-distillation.

## Tooling

**[Unsloth](https://unsloth.ai/)** is the recommended training framework. It
provides ~1.5x faster training with ~50% less VRAM compared to standard FA2
setups, with no accuracy loss. Key features:

- BF16 LoRA and full fine-tuning for both dense and MoE models
- `train_on_responses_only` to mask user turns during SFT (train only on
  assistant outputs)
- GGUF export for llama.cpp / Ollama deployment
- `use_gradient_checkpointing = "unsloth"` for extended context with lower VRAM

References:

- [Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train)
- [Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)

## LoRA Configuration (Dense Models)

| Parameter | Value | Rationale |
|---|---|---|
| Rank | 64 | Sufficient for ~1,500 SFT examples. RL/distillation needs very low capacity per "LoRA Without Regret" |
| Alpha | 32 | Standard; with 1/r scaling, optimal LR is ~independent of rank. Unsloth examples use alpha == r; we use alpha = r/2 per "LoRA Without Regret" |
| Target modules | All layers (MLP + attention) | MLP-only or all-layers >> attention-only per "LoRA Without Regret". Unsloth: `target_modules = "all-linear"` |
| Dropout | 0 | Standard for LoRA |

**QLoRA warning**: Unsloth explicitly recommends against QLoRA (4-bit base
weights) for Qwen3.5 models -- both dense and MoE -- due to "higher than normal
quantization differences." Use BF16 LoRA instead.

VRAM for BF16 LoRA (with Unsloth optimizations):

| Student | LoRA VRAM | LoRA params | % of total |
|---|---|---|---|
| 0.8B | ~3 GB | ~6M | 0.7% |
| 2B | ~5 GB | ~12M | 0.6% |
| 4B | ~10 GB | ~24M | 0.5% |
| 9B | ~22 GB | ~40M | 0.4% |

Source: [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune).

## LoRA Configuration (35B-A3B MoE)

LoRA on MoE models introduces specific challenges:

1. **Per-expert adapters**: "LoRA Without Regret" recommends training a separate
   LoRA on each expert, with the per-expert rank equal to the total rank divided
   by the number of active experts. For 35B-A3B (8 active routed experts + 1
   shared), this means rank 8 per expert at a total effective rank of 64.

2. **Expert routing imbalance**: Recent research (MoE-Sieve, 2026) shows that
   expert activation is highly skewed -- a small subset of "hot" experts handles
   most tokens while many remain "cold". Uniform LoRA across all 256 experts
   wastes capacity. A routing-guided approach (LoRA only on the top-25%
   most-routed experts) can reduce trainable parameters by 70% with no quality
   loss.

3. **VRAM**: BF16 LoRA training on 35B-A3B requires ~74 GB, which means a
   4xL40S node (192 GB) or 2xA100 80GB. QLoRA is not recommended for Qwen3.5
   MoE models (see warning above). Unsloth's MoE kernels provide ~12x faster
   training with >35% less VRAM.

4. **Router stability**: Unsloth disables router-layer fine-tuning by default
   for stability. This is the recommended approach unless there's a specific
   reason to retrain routing.

5. **No multi-tenant serving benefit**: One of LoRA's key advantages -- keeping
   the base model frozen and swapping small adapters -- is less useful for MoE
   because the base model itself is large (70 GB at BF16). The inference
   advantage of MoE comes from activation sparsity, not weight size.

| Parameter | Value (MoE) | Notes |
|---|---|---|
| Total rank | 64 | Same effective capacity as dense |
| Per-expert rank | 8 | 64 / 8 active experts |
| Shared expert rank | 64 | Full rank for the always-active shared expert |
| Target modules | MoE FFN layers + attention | Per "LoRA Without Regret" MoE findings |
| Expert selection | Top-25% most-routed | Per MoE-Sieve; reduces params ~70% |
| Router fine-tuning | Disabled | Default in Unsloth; recommended for stability |

Trainable parameters at rank 64 (MoE):

| Student | LoRA params | % of total | Notes |
|---|---|---|---|
| 35B-A3B (all experts) | ~200M | 0.6% | Expensive, most capacity wasted on cold experts |
| 35B-A3B (top-25% experts) | ~60M | 0.2% | Recommended; comparable to 9B dense |

## Training Hyperparameters

| Parameter | SFT (Stage 1) | Distillation (Stage 2) |
|---|---|---|
| Learning rate | 2e-4 to 3e-4 | 1e-4 |
| LR schedule | Linear (with warmup) | Constant |
| Batch size | 1 (with GA=4) | 32 (4 samples x 8 prompts) |
| Epochs | 3 | N/A (step-based) |
| Max seq length | 8,192 | 4,096 (shorter rollouts suffice) |
| Optimizer | AdamW 8-bit | AdamW |
| Weight decay | 0.01 | 0 |
| Gradient checkpointing | `"unsloth"` | `"unsloth"` |
| Loss masking | `train_on_responses_only` | N/A |

**Notes on learning rate**: The 10x multiplier over FullFT optimal (per "LoRA
Without Regret") gives ~3e-4 for SFT. Unsloth's Qwen3.5 examples use 2e-4 with
a linear schedule. For distillation, a lower LR is appropriate since we're
making smaller behavioral adjustments. For short runs (<100 steps), a 15x
multiplier may be better.

**Notes on batch size**: Keep small. "LoRA Without Regret" shows LoRA pays a
larger penalty for large batch sizes than FullFT, independent of rank. Unsloth
recommends `per_device_train_batch_size = 1` with `gradient_accumulation_steps
= 4` to simulate larger batches without the VRAM cost.

**Notes on loss masking**: Use Unsloth's `train_on_responses_only` to mask user
turns during SFT, training only on assistant outputs. This improves fine-tune
accuracy by not penalizing the model for user-provided text. The loss is
computed on both the thinking trace and the visible response -- the model must
learn *how* to reason about social situations, not just what to say.

Tool result tokens (`"role": "tool"` messages in the
[trace format](dataset-generation-agent.md#trace-format)) must also be masked
during loss computation. These tokens come from the environment (wiki results,
delegation outputs), not from the model -- training on them would teach the
model to memorize tool outputs rather than learn when to call tools. In
summary, the loss mask is:

- User messages: **masked** (not generated by the model)
- Tool results: **masked** (environment responses)
- System prompt / context slots: **masked** (provided context)
- Assistant thinking traces: **trained**
- Assistant visible responses: **trained**
- Assistant tool calls: **trained** (the model must learn to emit these)

**Notes on thinking mode**: Use the thinking-enabled chat template during
training (`"qwen3.5-thinking"` in Unsloth). Qwen3.5 small models (0.8B-9B)
have thinking disabled by default; it must be explicitly enabled. Unsloth
recommends keeping at least 75% reasoning-style examples to preserve reasoning
ability. Our dataset is 100% reasoning-style (all examples include thinking
traces), which is ideal.

**Notes on gradient accumulation**: Unsloth fixed a universal bug where gradient
accumulation inflated loss values. Use Unsloth's corrected implementation.

## Quantization for Deployment

After fine-tuning, export to GGUF for inference via llama.cpp or Ollama.

Unsloth's [Dynamic 2.0 quantization](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
is SOTA for Qwen3.5 GGUFs: important layers are upcasted to 8 or 16-bit while
less critical ones use lower precision. Key findings from their benchmarks:

- **Q4_K_XL** is the recommended quant level for quality/size balance.
- **Sensitive tensors**: `ssm_out` (Mamba/DeltaNet layers) and `ffn_down_exps`
  should not be aggressively quantized -- they cause disproportionate KLD
  increases.
- **Imatrix** calibration helps significantly at lower bit widths. Unsloth uses
  long-context chat and tool-calling examples for calibration rather than
  Wikipedia, which better matches our use case.
- **Perplexity/KLD can be misleading**: Unsloth's IQ2_XXS outperforms other
  providers' IQ3_S on real-world evals (LiveCodeBench, MMLU-Pro) despite worse
  perplexity scores. Always validate with task-specific evals.
