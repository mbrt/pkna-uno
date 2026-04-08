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

## Goals

Train a model that:

1. **Embodies Uno's personality** -- tone, humor, emotional responses, and
   relationship-aware register shifts are baked into the weights, not dependent
   on a system prompt.
2. **Uses tools for knowledge** -- factual information about the PKNA universe
   lives in the wiki and is retrieved via search tools at inference time, not
   memorized during training.
3. **Orchestrates sub-agents** -- the model delegates technical work (coding,
   research, math, etc.) to specialized sub-agents via tool calls, acting as a
   social front-end rather than a generalist.

What we are *not* training:

- A knowledge base. The wiki stays external.
- A generalist assistant. Raw problem-solving is delegated.
- A reasoning model. No chain-of-thought or extended thinking needed in the
  student -- it should be fast and conversational.

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

## Model Selection

### Student Model Alternatives

All candidates are from the Qwen3.5 family (Apache 2.0, hybrid Gated
DeltaNet + Attention architecture, 262K native context). The table includes
both dense and Mixture-of-Experts (MoE) options.

#### Benchmarks

| Property | 0.8B | 2B | 4B | 9B | 35B-A3B (MoE) |
|---|---|---|---|---|---|
| Type | Dense | Dense | Dense | Dense | MoE (256 experts, 8+1 active) |
| Total params | 0.9B | 2B | 5B | 10B | 35B |
| Active params | 0.9B | 2B | 5B | 10B | ~3B |
| Hidden dim | 1,024 | 2,048 | 2,560 | 4,096 | 2,560 |
| Layers | 24 | 24 | 32 | 32 | 64 |
| IFEval | 52.1 | 78.6 | 89.8 | 91.5 | 91.9 |
| MMLU-Pro | 29.7 | 66.5 | 79.1 | 82.5 | 85.3 |
| BFCL-V4 | ~35* | 43.6 | ~55* | 66.1 | ~62* |
| Agentic (BenchLM) | -- | -- | -- | -- | 55.1 |
| VRAM inference (BF16) | ~2 GB | ~4 GB | ~8 GB | ~18 GB | ~70 GB |
| VRAM inference (Q4) | <2 GB | ~2 GB | ~4 GB | ~8 GB | ~8 GB |
| VRAM LoRA train (BF16) | ~4 GB | ~8 GB | ~16 GB | ~36 GB | ~74 GB |
| Inference speed (Q4, consumer) | Very fast | Very fast | Fast | Moderate | Fast (~3B active) |

*Starred values are estimated from interpolation where official numbers are
unavailable.*

The 35B-A3B MoE is a compelling option: it activates only ~3B parameters per
token (comparable compute to the 4B dense), but draws on 35B total parameters
for richer representations. Its 64-layer depth gives more capacity for
behavioral nuance than any of the dense models.

#### Assessment by Role Suitability

**0.8B** -- Not recommended. IFEval 52.1 and estimated BFCL ~35% are too weak
for reliable tool calling or multi-turn orchestration. Only viable if
personality is the sole goal and all tool routing is hardcoded externally.

**2B** -- Marginal orchestrator. IFEval 78.6 is decent but BFCL 43.6% means
tool calling will be unreliable for multi-step delegation. Viable as a
conversation-only personality model with an external tool dispatcher. The
cheapest on-device option.

**4B** -- Solid middle ground. IFEval 89.8 is strong. BFCL ~55% is workable
for simple tool patterns (wiki search, single delegation). 32 layers give
reasonable depth. Fits on a single consumer GPU for both training and
inference.

**9B** -- Strongest dense option. IFEval 91.5 and BFCL 66.1% make it the most
reliable orchestrator among the dense models. MMLU-Pro 82.5 suggests strong
general reasoning that helps with nuanced personality. Requires ~18 GB for
inference (fits RTX 4090) or ~8 GB quantized.

**35B-A3B (MoE)** -- Best benchmarks overall: IFEval 91.9, MMLU-Pro 85.3.
Inference speed is comparable to the 4B dense (only ~3B params activate per
token), and at Q4 quantization it fits in ~8 GB VRAM -- same as the 4B at
BF16. The 64-layer architecture provides the deepest representational capacity.
However, training is significantly more expensive (see below).

#### Tradeoff Summary

| Dimension | 2B | 4B | 9B | 35B-A3B |
|---|---|---|---|---|
| Personality capacity | Low | Medium | High | Highest |
| Tool use reliability | Poor | Adequate | Good | Good |
| Instruction following | Decent | Strong | Strong | Strong |
| Inference cost | Lowest | Low | Medium | Low (MoE) |
| Training cost | Lowest | Low | Medium | **High** |
| On-device feasibility | Phone/laptop | Laptop | Desktop | Laptop (Q4) |
| LoRA complexity | Simple | Simple | Simple | **Complex** |

The key tension for the MoE option is between inference efficiency (excellent)
and training complexity (high). See the LoRA section below for details.

### Teacher Model

**Qwen3.5-27B** (dense, 28B params, hidden_dim=5120, 64 layers).

Rationale:

- On-policy distillation requires `compute_logprobs` on the student's
  trajectories, which means the teacher must be an open-weight model (API
  models don't expose per-token logprobs at arbitrary positions).
- 27B is the largest dense Qwen3.5 that fits on a 4xL40S node in BF16 (~56 GB
  for weights + KV cache overhead).
- IFEval 95.0 and strong tool-use capabilities make it a good behavioral
  target for the student.
- For the off-policy SFT data generation stage, we can *also* use Gemini 3
  Flash via API (cheaper, no logprobs needed), and reserve the 27B for the
  on-policy distillation stage only.

**Alternative teacher: 122B-A10B (MoE)**. MMLU-Pro 86.1, GPQA 85.5. Stronger
than the 27B but requires ~70 GB VRAM at FP8, meaning 2x L40S or a single A100
80GB. Could be worth it if the 27B's signal proves insufficient, but adds
significant infrastructure cost.

### Self-Distillation

The on-policy distillation paper shows that using the *pre-SFT version of the
same model* as teacher effectively recovers lost post-training behaviors. This
means we could:

1. SFT the student (e.g. Qwen3.5-4B) on personality data.
2. On-policy distill using the *original* Qwen3.5-4B (pre-SFT) as teacher.

This eliminates the need for a larger teacher model entirely, cutting
infrastructure costs significantly. The tradeoff is that the teacher can only
recover behaviors the student already had -- it can't teach new ones. For our
use case (personality + tool use), this may be sufficient since tool use is
already present in the base model.

## Training Pipeline

### Stage 1: Off-Policy SFT (Personality Mid-Training)

**Goal**: Teach the student Uno's personality, tone, and interaction patterns.

**What goes into the weights:**

- Personality traits (sarcasm, warmth, register shifts by interlocutor)
- Emotional response patterns (humor timing, deflection, vulnerability)
- Italian language patterns and idiolect (catchphrases, expressions)
- Tool-use patterns (when to search wiki, when to delegate)

**What stays out of the weights:**

- Factual knowledge about the PKNA universe (retrieved via tools)
- General world knowledge (already in pre-training)
- Technical problem-solving (delegated to sub-agents)

**Dataset construction:**

| Source | Examples | Purpose |
|---|---|---|
| Scene dialogues (existing 229) | ~229 | Core personality signal, reformatted as system + multi-turn chat |
| Synthetic conversations | ~500 | Generated by teacher from claim-based prompts, covering varied interlocutors and situations |
| Tool-use demonstrations | ~200 | Conversations where Uno searches wiki or delegates, showing correct tool patterns |
| Background chat (Tulu3 subset) | ~500 | Sampled from the student itself (on-policy SFT data), to mitigate catastrophic forgetting |
| **Total** | **~1,500** | |

Each training example includes:

- A compact system prompt (~500 tokens) with core personality traits only (not
  the full 208-line soul document -- distillation should internalize this)
- Multi-turn conversation with tool calls where appropriate
- The assistant role is always Uno

The system prompt during training is intentionally minimal -- the goal is for
the model to internalize the behavior so that at inference time, only a brief
reminder is needed (or none at all).

**Data mix**: ~50% personality, ~15% tool-use, ~35% background chat.

**Loss**: Standard cross-entropy (forward KL).

### Stage 2: On-Policy Distillation (Behavior Recovery + Sharpening)

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

**Prompt sources for distillation:**

| Source | Prompts | Purpose |
|---|---|---|
| Character interview questions | ~100 | Test personality consistency |
| Scenario starters (crisis, casual, technical) | ~100 | Test situational adaptation |
| Tool-use prompts (wiki search, delegation) | ~100 | Recover/sharpen tool calling |
| General chat (Tulu3 subset) | ~300 | Recover instruction following |
| **Total** | **~600** | |

**Sampling**: 4 rollouts per prompt, ~150-300 training steps.

**Teacher choice for this stage:**

- Option A: Qwen3.5-27B (strongest signal, requires 4xL40S)
- Option B: Original Qwen3.5-4B pre-SFT (self-distillation, single GPU, cheaper)
- Recommendation: Start with Option B. If tool-use recovery is insufficient,
  switch to Option A.

## Hyperparameters

### LoRA Configuration (Dense Models)

| Parameter | Value | Rationale |
|---|---|---|
| Rank | 64 | Sufficient for ~1,500 SFT examples. RL/distillation needs very low capacity per "LoRA Without Regret" |
| Alpha | 32 | Standard; with 1/r scaling, optimal LR is ~independent of rank |
| Target modules | All layers (MLP + attention) | MLP-only or all-layers >> attention-only per "LoRA Without Regret" |
| Dropout | 0 | Standard for LoRA |

Trainable parameters at rank 64 (dense models):

| Student | LoRA params | % of total |
|---|---|---|
| 0.8B | ~6M | 0.7% |
| 2B | ~12M | 0.6% |
| 4B | ~24M | 0.5% |
| 9B | ~40M | 0.4% |

### LoRA Configuration (35B-A3B MoE)

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

3. **VRAM**: LoRA training on 35B-A3B in BF16 requires ~74 GB, which means a
   4xL40S node (192 GB) or 2xA100 80GB. QLoRA (4-bit base weights) is not
   recommended for MoE models due to quantization artifacts in expert routing.

4. **No multi-tenant serving benefit**: One of LoRA's key advantages -- keeping
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

Trainable parameters at rank 64 (MoE):

| Student | LoRA params | % of total | Notes |
|---|---|---|---|
| 35B-A3B (all experts) | ~200M | 0.6% | Expensive, most capacity wasted on cold experts |
| 35B-A3B (top-25% experts) | ~60M | 0.2% | Recommended; comparable to 9B dense |

### Training Hyperparameters

| Parameter | SFT (Stage 1) | Distillation (Stage 2) |
|---|---|---|
| Learning rate | 3e-4 | 1e-4 |
| LR schedule | Constant (no warmup) | Constant |
| Batch size | 16-32 | 32 (4 samples x 8 prompts) |
| Epochs | 3 | N/A (step-based) |
| Max seq length | 4,096 | 2,048 (shorter rollouts suffice) |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0 |

**Notes on learning rate**: The 10x multiplier over FullFT optimal (per "LoRA
Without Regret") gives ~3e-4 for SFT. For distillation, a lower LR is
appropriate since we're making smaller behavioral adjustments. For short runs
(<100 steps), a 15x multiplier may be better.

**Notes on batch size**: Keep small. "LoRA Without Regret" shows LoRA pays a
larger penalty for large batch sizes than FullFT, independent of rank.

## Infrastructure and Cost Estimates

### Hardware Options by Student Size

| Student | Training GPU | Teacher GPU (27B) | Teacher GPU (self-distill) |
|---|---|---|---|
| 2B | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 4B | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 9B | 1x L40S (g6e.2xlarge) | 4x L40S (g6e.12xlarge) | 1x L40S |
| 35B-A3B | 4x L40S (g6e.12xlarge) | 4x L40S (g6e.12xlarge) | 4x L40S |

AWS on-demand pricing (2026):

- g6e.2xlarge (1x L40S, 48GB): **$2.24/hr**
- g6e.12xlarge (4x L40S, 192GB): **$10.50/hr**

The 35B-A3B MoE requires a 4xL40S node for LoRA training (74 GB VRAM in BF16),
but its inference runs on the same hardware as the 4B dense at Q4 quantization
(~8 GB). This creates an asymmetry: training is 5x more expensive than the 4B,
but serving is comparable.

### Cost Breakdown per Training Run

#### Scenario A: 4B Self-Distillation (cheapest)

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~30 min | 1x g6e.2xlarge | ~$1 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| Teacher logprobs (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| Student training (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| **Total per run** | | | **~$15** |

#### Scenario B: 4B + 27B Teacher

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~30 min | 1x g6e.2xlarge | ~$1 |
| Student sampling (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| **Total per run** | | | **~$34** |

#### Scenario C: 9B + 27B Teacher

| Phase | Duration | Instance | Cost |
|---|---|---|---|
| Synthetic data generation (Gemini API) | -- | -- | ~$5 |
| SFT (Stage 1) | ~1 hr | 1x g6e.2xlarge | ~$2 |
| Student sampling (Stage 2) | ~4 hrs | 1x g6e.2xlarge | ~$9 |
| Teacher logprobs (Stage 2) | ~2 hrs | 1x g6e.12xlarge | ~$21 |
| Student training (Stage 2) | ~2 hrs | 1x g6e.2xlarge | ~$5 |
| **Total per run** | | | **~$42** |

#### Scenario D: 35B-A3B MoE + 27B Teacher

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

#### Scenario E: 35B-A3B MoE Self-Distillation

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

### Budget for Experimentation

Expect 5-10 runs for hyperparameter tuning and ablations.

| Scenario | Per run | 10 runs | Notes |
|---|---|---|---|
| A: 4B self-distill | $15 | $150 | Cheapest, good starting point |
| B: 4B + 27B teacher | $34 | $340 | Better if self-distill underperforms |
| C: 9B + 27B teacher | $42 | $420 | Best dense capability |
| D: 35B-A3B + 27B teacher | $84 | $840 | Best benchmarks, highest training cost |
| E: 35B-A3B self-distill | $84 | $840 | Same cost -- MoE training dominates |

## Evaluation Strategy

### Dimensions (from Character-LLM framework)

1. **Personality**: Does the model exhibit Uno's traits (sarcasm, warmth,
   register shifts) without a system prompt?
2. **Memorization**: Can it recall core identity facts (name, origin, nature)
   from weights alone?
3. **Hallucination**: Does it invent PKNA facts instead of searching the wiki?
4. **Tool use**: Does it correctly call `search_wiki` for factual questions and
   `delegate` for technical tasks?
5. **Stability**: Does character hold over 20+ turn conversations and under
   adversarial probing?

### Eval Protocol

- ~100 single-turn interview questions (personality + memorization)
- ~50 factual questions requiring wiki search (tool use + hallucination)
- ~20 technical requests requiring delegation (orchestration)
- ~20 multi-turn conversations with adversarial follow-ups (stability)
- LLM-as-judge scoring (Gemini 3 or Claude) on each dimension, 1-5 scale

### Baselines

- Gemini 3 Flash + soul document system prompt (current best)
- Qwen3.5-4B + soul document system prompt (no fine-tuning)
- Qwen3.5-4B + SFT only (no distillation)

## Risks and Open Questions

### Risks

1. **Tool-use degradation after SFT**: The on-policy distillation paper shows
   this is recoverable, but tool calling is more structured than general
   instruction following. May need tool-use-specific prompts in the distillation
   stage.

2. **Small dataset**: ~1,500 SFT examples is small. The distillation paper's
   personalization experiment used a similar scale successfully, but our task
   (personality + tool use) is more complex. Synthetic data quality matters a
   lot.

3. **Italian language**: The base models are multilingual but Italian-specific
   personality nuances (sarcasm, idiomatic expressions) may be harder to learn
   than English equivalents. May need to oversample Italian examples.

4. **Evaluation subjectivity**: Personality adherence is inherently subjective.
   The LLM-as-judge approach helps but is not ground truth. Human evaluation
   (even informal vibe checks) remains important.

5. **MoE LoRA maturity**: LoRA on MoE is less battle-tested than on dense
   models. Expert routing imbalance means uniform LoRA wastes capacity on cold
   experts. Recent work (MoE-Sieve, DR-LoRA) addresses this but adds
   implementation complexity. The "LoRA Without Regret" paper validated
   per-expert LoRA on Qwen3-30B-A3B, which is architecturally similar to
   Qwen3.5-35B-A3B, so the approach is grounded but not trivial.

### Open Questions

1. **MoE vs 9B dense**: The 35B-A3B has better benchmarks than the 9B dense
   (IFEval 91.9 vs 91.5, MMLU-Pro 85.3 vs 82.5) and comparable inference
   speed, but 5-6x higher training cost. Is the benchmark gap worth it for a
   personality task? The 9B may be "good enough" at much lower cost.

2. **How much personality can a 2B model hold?** If the 2B proves sufficient
   for personality but weak on tool use, a hybrid architecture (2B personality
   model + external tool router) could be cheaper than a 4B that does both.

3. **System prompt at inference**: Should the fine-tuned model still receive a
   brief system prompt, or should personality be fully internalized? Likely a
   minimal reminder helps, but this needs ablation.

4. **Continual learning**: As the claim ledger grows (new comic issues, refined
   observations), can we incrementally update the model without full retraining?
   The on-policy distillation paper's phase-alternating approach (SFT new data
   -> distill to recover) is promising here.

5. **Serving**: For on-device deployment (2B/4B/35B-A3B at Q4), quantization
   impact on personality quality is unknown. The MoE model is particularly
   interesting here: at Q4 it fits in ~8 GB (same as 4B at BF16) but has
   access to 35B parameters of knowledge. However, MoE quantization can
   degrade expert routing quality.

## Recommended Starting Plan

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
