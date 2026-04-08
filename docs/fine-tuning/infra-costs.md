# Infrastructure & Costs

Back to [Fine-Tuning Design](../fine-tuning-design.md).

All cost estimates assume [Unsloth](https://unsloth.ai/) as the training
framework, which provides ~1.5x faster training and ~50% less VRAM compared to
standard FA2 setups. See [Training Strategy](training-strategy.md) for details.

## Hardware Options by Student Size

BF16 LoRA VRAM requirements (with Unsloth):

| Student | LoRA VRAM | Training GPU | Teacher GPU (27B) | Teacher GPU (self-distill) |
|---|---|---|---|---|
| 2B | ~5 GB | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 4B | ~10 GB | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 9B | ~22 GB | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 35B-A3B | ~74 GB | 4x L40S (g6e.12xlarge) | 4x L40S (g6e.12xlarge) | 4x L40S |

QLoRA is not recommended for Qwen3.5 (see [Training Strategy](training-strategy.md)).

AWS on-demand pricing (2026):

- g6e.2xlarge (1x L40S, 48GB): **$2.24/hr**
- g6e.12xlarge (4x L40S, 192GB): **$10.50/hr**

The 35B-A3B MoE requires a 4xL40S node for LoRA training (74 GB VRAM in BF16),
but its inference runs on the same hardware as the 4B dense at Q4 quantization
(~8 GB). This creates an asymmetry: training is 5x more expensive than the 4B,
but serving is comparable.

## Cost Breakdown per Training Run

### Scenario A: 4B Self-Distillation (cheapest)

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~30 min | 1x g6e.2xlarge | ~$1 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| Teacher logprobs (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| Student training (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| **Total per run** | | | **~$15** |

### Scenario B: 4B + 27B Teacher

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~30 min | 1x g6e.2xlarge | ~$1 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| **Total per run** | | | **~$34** |

### Scenario C: 9B + 27B Teacher

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| Student sampling (Stage 2) | ~4 hrs | 1x g6e.2xlarge | ~$9 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| **Total per run** | | | **~$42** |

### Scenario D: 35B-A3B MoE + 27B Teacher

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge (Q4) | ~$5 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~3 hrs | 1x g6e.12xlarge | ~$32 |
| **Total per run** | | | **~$84** |

Note: MoE student sampling is fast (~3B active params) and can run on a single
GPU at Q4. But LoRA training requires loading all 35B parameters in BF16,
hence the 4xL40S node.

### Scenario E: 35B-A3B MoE Self-Distillation

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge (Q4) | ~$5 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~3 hrs | 1x g6e.12xlarge | ~$32 |
| **Total per run** | | | **~$84** |

Self-distillation doesn't save much for the MoE because the bottleneck is
training VRAM (loading all 35B params), not teacher size.

## Budget for Experimentation

Expect 5-10 runs for hyperparameter tuning and ablations.

| Scenario | Per run | 10 runs | Notes |
|---|---|---|---|
| A: 4B self-distill | $15 | $150 | Cheapest, good starting point |
| B: 4B + 27B teacher | $34 | $340 | Better if self-distill underperforms |
| C: 9B + 27B teacher | $42 | $420 | Best dense capability |
| D: 35B-A3B + 27B teacher | $84 | $840 | Best benchmarks, highest training cost |
| E: 35B-A3B self-distill | $84 | $840 | Same cost -- MoE training dominates |
