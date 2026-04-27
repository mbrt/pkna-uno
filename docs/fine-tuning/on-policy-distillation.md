# On-Policy Distillation (Stage 2)

Back to [Training Strategy](training-strategy.md).

## Goal

Recover any instruction-following and tool-use capabilities degraded by SFT,
and sharpen character adherence with dense per-token feedback.

## Method

The student generates its own completions on-policy. The teacher computes
logits on those completions, and the loss is the reverse KL divergence over the
full vocabulary at each token position. This is implemented via TRL's
`DistillationTrainer` with `lmbda=1.0` (fully on-policy) and `beta=1.0`
(reverse KL).

Unlike REINFORCE-based implementations (e.g. tinker-cookbook), which estimate
the KL gradient via a single-sample policy gradient with per-token KL as
advantage, TRL computes the exact analytical gradient of the KL divergence.
This means:

- No variance reduction is needed (1 rollout per prompt suffices).
- No group-centered advantages or importance sampling ratios.
- The loss is computed over the full vocabulary (`loss_top_k=0`), not a sparse
  top-k approximation.

```
for each batch of prompts:
    completion = student.generate(prompt)              # on-policy sampling
    student_logits = student.forward(prompt + completion)   # student distribution
    teacher_logits = teacher.forward(prompt + completion)   # teacher distribution
    loss = KL(student || teacher)                      # full-vocabulary reverse KL
    student.backward(loss)                             # exact gradient
```

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | Model-dependent (see [per-model LR](training-strategy.md#per-model-learning-rates)) |
| LR schedule | Constant |
| Batch size | 1 (with GA=4) |
| Max steps | 200 |
| Max seq length | 4,096 (shorter rollouts suffice) |
| Max completion length | 1,024 |
| Optimizer | AdamW |
| Weight decay | 0 |
| Gradient checkpointing | `"unsloth"` |
| lmbda | 1.0 (fully on-policy) |
| beta | 1.0 (reverse KL) |
| loss_top_k | 0 (full vocabulary) |
| temperature | 1.0 |

## Prompt Sources

| Source | Prompts | Purpose |
|---|---|---|
| Character interview questions | ~100 | Test personality consistency |
| Scenario starters (crisis, casual, technical) | ~100 | Test situational adaptation |
| Tool-use prompts (wiki search, delegation) | ~100 | Recover/sharpen tool calling |
| General chat (Tulu3 subset) | ~300 | Recover instruction following |
| **Total** | **~600** | |

**Sampling**: 1 rollout per prompt, ~150-300 training steps. TRL's
`DistillationTrainer` computes exact analytical KL gradients over the full
vocabulary, so multiple rollouts for variance reduction (as in REINFORCE-based
implementations) are unnecessary.

## Teacher Choice

- Option A: Qwen3.5-27B (strongest signal, requires 4xL40S)
- Option B: Original Qwen3.5-4B pre-SFT (self-distillation, single GPU, cheaper)
- Recommendation: Start with Option B. If tool-use recovery is insufficient,
  switch to Option A.

See [Model Selection](model-selection.md) for details on teacher models and
self-distillation.

## Weight Sharing for Self-Distillation

When using self-distillation (Option B), the teacher is the same base model as
the student -- just without the LoRA adapters. Instead of loading a second copy
of the base weights into VRAM, `run_distillation.py` uses PEFT's
`model.disable_adapter()` context manager to temporarily zero out the LoRA
contributions during the teacher forward pass. This produces identical logits
to a standalone base model while sharing all base weights with the student.

VRAM savings from weight sharing:

| Student | Without sharing | With sharing | Saved |
|---|---|---|---|
| 4B BF16 | ~18 GB (2x base + LoRA + optim) | ~10 GB (1x base + LoRA + optim) | ~8 GB |
| 9B BF16 | ~40 GB | ~22 GB | ~18 GB |
| 35B-A3B BF16 | ~144 GB | ~74 GB | ~70 GB |

This optimization only applies to self-distillation (same base model for
teacher and student). If switching to a larger teacher (Option A), a separate
model load is required.
