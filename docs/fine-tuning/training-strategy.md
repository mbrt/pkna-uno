# Training Strategy

Back to [Fine-Tuning Design](../fine-tuning-design.md).

## Pipeline Overview

Training proceeds in two stages:

1. **[Off-Policy SFT](sft-training.md)** -- Teach the student Uno's
   personality, tone, and interaction patterns using a curated dataset (see
   [SFT Dataset](sft-dataset.md)).
2. **[On-Policy Distillation](on-policy-distillation.md)** -- Recover
   instruction-following and tool-use capabilities degraded by SFT, and sharpen
   character adherence with dense per-token feedback from a teacher.

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
| Alpha | 64 | Unsloth recommends alpha >= rank (alpha/rank >= 1). alpha == rank is standard; alpha == 2*rank makes learning more aggressive |
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

| Parameter | Value (MoE) | Notes | Implemented |
|---|---|---|---|
| Total rank | 64 | Same effective capacity as dense | Yes |
| Per-expert rank | 8 | 64 / 8 active experts, via PEFT `rank_pattern` | Yes |
| Shared expert rank | 64 | Full rank for the always-active shared expert | Yes (inherits base rank) |
| Target modules | all-linear (MoE FFN + attention) | Per "LoRA Without Regret" MoE findings | Yes |
| rsLoRA | Enabled (`use_rslora=True`) | Auto-scales alpha per rank so a single alpha works | Yes |
| Expert selection | Top-25% most-routed | Per MoE-Sieve; reduces params ~70% | Not yet |
| Router fine-tuning | Disabled | Default in Unsloth; recommended for stability | Yes (Unsloth default) |

Trainable parameters at rank 64 (MoE):

| Student | LoRA params | % of total | Notes |
|---|---|---|---|
| 35B-A3B (all experts) | ~200M | 0.6% | Expensive, most capacity wasted on cold experts |
| 35B-A3B (top-25% experts) | ~60M | 0.2% | Recommended; comparable to 9B dense |

## Per-Model Learning Rates

"LoRA Without Regret" shows that optimal LoRA LR scales with hidden size:
`LR = M_LoRA * (2000 / hidden_size)^model_pow`. Smaller models need lower LR
to avoid oscillating loss. The scripts in `training/` select the default
automatically based on `--model` (see `training/__init__.py`).

| Student | Hidden size | SFT LR | Distillation LR |
|---|---|---|---|
| 0.8B | 1024 | 5e-5 | 3e-5 |
| 4B | 3584 | 2e-4 | 1e-4 |
| 9B | 4096 | 2e-4 | 1e-4 |
| 35B-A3B | ~2560 (active) | 2e-4 | 1e-4 |

The 4B values come from Unsloth's Qwen3.5 examples and the 10x FullFT
multiplier. The 0.8B values are scaled down ~4x to account for its smaller
hidden dimension. The 9B uses the same LR as the 4B: the hidden_size scaling
formula gives `2e-4 * (3584/4096) ~= 1.75e-4`, but the ~12% reduction is
within noise, and Unsloth recommends 2e-4 as a starting point for all Qwen3.5
dense models. The 35B-A3B MoE uses 4B-class LR since its active hidden
size is comparable. All values can be overridden via `--lr`.

**Notes on learning rate**: The 10x multiplier over FullFT optimal (per "LoRA
Without Regret") gives ~3e-4 for SFT on 4B. Unsloth's Qwen3.5 examples use
2e-4 with a linear schedule. For distillation, a lower LR is appropriate since
we're making smaller behavioral adjustments. For short runs (<100 steps), a 15x
multiplier may be better.

**Notes on batch size**: Keep small. "LoRA Without Regret" shows LoRA pays a
larger penalty for large batch sizes than FullFT, independent of rank. Unsloth
recommends `per_device_train_batch_size = 1` with `gradient_accumulation_steps
= 4` to simulate larger batches without the VRAM cost.

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
