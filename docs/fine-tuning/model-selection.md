# Model Selection

Back to [Fine-Tuning Design](../fine-tuning-design.md).

## Student Model Families

Two model families are strong candidates: **Qwen3.5/3.6** and **Gemma 4**. Both
are Apache 2.0 licensed, support long context, and offer a range of sizes from
edge-deployable to server-class. This section compares them side-by-side.

### Qwen3.5 / Qwen3.6

Hybrid Gated DeltaNet + Attention architecture, 262K native context (extendable
to 1M via YaRN). Includes both dense and Mixture-of-Experts (MoE) options.

**Qwen3.6** (released April 16, 2026) shares the same architecture as Qwen3.5
and is a drop-in successor for the 35B-A3B MoE variant. It adds native
multimodal support (text + image + video), improved tool calling (MCPMark
37.0 vs 27.0), and stronger agentic performance. Dense sizes (0.8B-9B) are
Qwen3.5 only as of mid-April 2026; Qwen3.6 ships the 35B-A3B MoE only.

#### Qwen3.5 Dense Models

| Property | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| Type | Dense | Dense | Dense | Dense |
| Total params | 0.9B | 2B | 5B | 10B |
| Active params | 0.9B | 2B | 5B | 10B |
| Layers | 24 | 24 | 32 | 32 |
| Context | 262K | 262K | 262K | 262K |
| IFEval | 52.1 | 78.6 | 89.8 | 91.5 |
| MMLU-Pro | 29.7 | 66.5 | 79.1 | 82.5 |
| BFCL-V4 | ~35* | 43.6 | ~55* | 66.1 |
| VRAM inference (BF16) | ~2 GB | ~4 GB | ~8 GB | ~18 GB |
| VRAM inference (Q4) | <2 GB | ~2 GB | ~4 GB | ~8 GB |
| VRAM LoRA train (BF16, Unsloth) | ~3 GB | ~5 GB | ~10 GB | ~22 GB |

#### Qwen 35B-A3B MoE (Qwen3.5 vs Qwen3.6)

| Property | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| Type | MoE (256 experts, 8+1 active) | MoE (256 experts, 8+1 active) |
| Total / active params | 35B / ~3B | 35B / ~3B |
| Layers | 64 | 64 |
| Context | 262K | 262K (1M via YaRN) |
| Multimodal | Text only | Text, image, video |
| IFEval | 91.9 | — (not yet published) |
| MMLU-Pro | 85.3 | ~85%+ (est., same arch) |
| GPQA Diamond | 84.2 | 86.0 |
| BFCL-V4 | ~62* | — (not yet published) |
| MCPMark (tool calling) | 27.0 | 37.0 |
| SWE-bench Verified | 70.0 | 73.4 |
| Terminal-Bench 2.0 | 40.5 | 51.5 |
| MMMU (multimodal) | — | 81.7 |
| VRAM inference (BF16) | ~70 GB | ~70 GB |
| VRAM inference (Q4) | ~8 GB | ~8 GB |
| VRAM LoRA train (BF16, Unsloth) | ~74 GB | ~74 GB (same architecture) |

### Gemma 4

Transformer with hybrid local/global attention, released April 2, 2026. Built
from Gemini 3 research. Includes dense models with Per-Layer Embeddings (PLE)
for the small sizes, and a MoE variant.

| Property | E2B | E4B | 26B-A4B (MoE) | 31B |
|---|---|---|---|---|
| Type | Dense (PLE) | Dense (PLE) | MoE (128 experts, 8+1 active) | Dense |
| Total params | 5.1B (2.3B effective) | 8B (4.5B effective) | 25.2B | 30.7B |
| Active params | 2.3B | 4.5B | 3.8B | 30.7B |
| Layers | 35 | 42 | 30 | 60 |
| Context | 128K | 128K | 256K | 256K |
| MMLU-Pro | 60.0 | 69.4 | 82.6 | 85.2 |
| GPQA Diamond | 43.4 | 58.6 | 82.3 | 84.3 |
| τ2-bench (agentic) | 24.5 | 42.2† | 68.2† | 76.9† |
| IFEval | N/A‡ | N/A‡ | N/A‡ | N/A‡ |
| BFCL | N/A‡ | N/A‡ | N/A‡ | N/A‡ |
| VRAM inference (BF16) | ~2 GB | ~5 GB | ~14 GB | ~24 GB |
| VRAM inference (Q4) | ~1 GB | ~2.4 GB | ~7 GB | ~11 GB |
| Inference speed (Q4)¶ | — | — | **~11 tok/s** | — |
| VRAM LoRA train (BF16) | ~8 GB | ~18 GB | >40 GB | ~48 GB§ |
| VRAM QLoRA train | — | ~10 GB | **Broken**§§ | ~22 GB |

*Starred Qwen values are estimated from interpolation.*
†Gemma τ2-bench is averaged over 3 domains; Qwen's agentic benchmarks use
BenchLM, so these are not directly comparable.
‡Google has not published IFEval or BFCL scores for Gemma 4 as of April 2026.
§Estimated from 2x BF16 inference weight size.
§§Gemma 4 26B-A4B stores expert weights as fused 3D tensors. bitsandbytes
cannot quantize them, so "4-bit" loading still requires ~44 GB -- effectively
broken for QLoRA on consumer hardware. Qwen3.5-35B-A3B stores experts as
separate 2D layers and quantizes normally.
¶Community-reported on RTX 5060 Ti 16GB. Qwen3.5-35B-A3B achieves 60+ tok/s on
the same hardware -- a ~6x speed gap. Multiple users confirm Gemma 4 26B-A4B
is anomalously slow despite similar active parameter counts.

**QLoRA note**: Unsloth recommends against QLoRA for *all* Qwen3.5/3.6 models
(dense and MoE) due to higher-than-normal quantization differences. BF16 LoRA
is the recommended approach. LoRA VRAM numbers above reflect Unsloth's
optimized training (~50% less VRAM than standard FA2 setups). Since Qwen3.6
shares the same architecture as Qwen3.5, all Unsloth training infrastructure
(MoE kernels, LoRA configs) carries over unchanged. See
[Training Strategy](training-strategy.md) for details.

**Ollama note**: As of mid-April 2026, Qwen3.6 GGUFs do not work in Ollama due
to separate mmproj vision files. Use llama.cpp-compatible backends instead.

### Head-to-Head Comparison

#### Small models (~4B active)

| Dimension | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|
| Effective params | 5B | 4.5B (8B total w/ PLE) |
| MMLU-Pro | 79.1 | 69.4 |
| IFEval | 89.8 | N/A |
| Tool calling (BFCL) | ~55%* | N/A (native function calling claimed) |
| Agentic (τ2-bench) | — | 42.2 |
| Context window | 262K | 128K |
| Multimodal | Text only | Text, image, audio |
| VRAM BF16 LoRA (Unsloth) | ~10 GB | ~18 GB |
| QLoRA feasibility | Not recommended | ~10 GB (works for dense Gemma) |

**Assessment**: Qwen3.5-4B leads on text benchmarks (MMLU-Pro +10 points) and
has a longer context window. Gemma 4 E4B brings multimodal capabilities and
audio support, but these are irrelevant for our text-only personality use case.
The lack of published IFEval and BFCL scores for Gemma 4 is a concern --
instruction following and tool calling are critical for our orchestrator role.
Anecdotally, community reports are positive for Gemma 4 E4B on general tasks,
but we have no hard numbers for the dimensions that matter most. On training
VRAM, Qwen3.5-4B with Unsloth BF16 LoRA fits in ~10 GB -- nearly half the
Gemma 4 E4B's ~18 GB for LoRA (though Gemma E4B supports QLoRA at ~10 GB,
which Qwen does not).

#### MoE models (~3-4B active)

| Dimension | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Gemma 4 26B-A4B |
|---|---|---|---|
| Total / active params | 35B / ~3B | 35B / ~3B | 25.2B / 3.8B |
| Experts | 256 (8+1 active) | 256 (8+1 active) | 128 (8+1 active) |
| Layers | 64 | 64 | 30 |
| MMLU-Pro | 85.3 | ~85%+ (est.) | 82.6 |
| GPQA Diamond | 84.2 | 86.0 | 82.3 |
| IFEval | 91.9 | — | N/A |
| MCPMark (tool calling) | 27.0 | 37.0 | 36.3 |
| SWE-bench Verified | 70.0 | 73.4 | 17.4 |
| τ2-bench (agentic) | — | — | 68.2 |
| Multimodal | Text only | Text, image, video | Text, image |
| Context window | 262K | 262K (1M via YaRN) | 256K |
| Inference speed (Q4)† | **60+ tok/s** | ~60+ tok/s (same arch) | **~11 tok/s** |
| VRAM inference (Q4) | ~8 GB | ~8 GB | ~7 GB |
| VRAM BF16 LoRA | ~74 GB | ~74 GB | >40 GB |
| QLoRA feasibility | **Not recommended** | **Not recommended** | **Broken** |

†Inference speed measured on RTX 5060 Ti 16GB, community-reported.

**Assessment**: The MoE comparison has shifted decisively toward Qwen since the
original analysis. Qwen3.6-35B-A3B is now the clear front-runner:

- **Inference speed**: ~6x faster than Gemma 4 26B-A4B on the same hardware
  (60+ vs 11 tok/s). This alone is disqualifying for Gemma in production.
- **Tool calling**: MCPMark 37.0 (Qwen 3.6) vs 36.3 (Gemma 4). Tool calling
  was previously unmeasured for either; now Qwen has a slight edge and Gemma
  still lacks IFEval/BFCL.
- **Agentic coding**: SWE-bench 73.4 vs 17.4 -- a 4x gap.
- **Architecture depth**: 64 vs 30 layers gives Qwen more representational
  capacity for learning personality + reasoning.
- **Training**: Neither supports QLoRA for MoE, but Gemma's fused 3D expert
  tensors make it completely broken while Qwen's separate 2D layers at least
  allow it in principle. Unsloth's MoE kernels (~12x speedup, >35% VRAM
  savings) make Qwen MoE training significantly more practical.
- **Qwen3.6 as free upgrade**: Same architecture as 3.5, so all training
  infrastructure carries over. Better benchmarks across the board.

#### Dense large models

| Dimension | Qwen3.5-27B | Gemma 4 31B |
|---|---|---|
| Total params | 28B | 30.7B |
| MMLU-Pro | 86.1 | 85.2 |
| GPQA Diamond | 85.5 | 84.3 |
| IFEval | 95.0 | N/A |
| τ2-bench (agentic) | — | 76.9 |
| Multilingual (MMMLU) | — | 88.4 |
| VRAM inference (BF16) | ~56 GB | ~24 GB |

These are teacher-class models. Both are strong. Qwen3.5-27B has published
IFEval (95.0) which matters for distillation signal quality. Gemma 4 31B has
better multilingual scores (MMMLU 88.4) which could help with Italian
personality nuances. However, the Qwen 27B is already validated in the
on-policy distillation paper's framework, and switching teacher architecture
introduces unknowns.

### Overall Assessment

**Qwen is the stronger choice for this project**, and the gap has widened since
the initial analysis. The arrival of Qwen3.6-35B-A3B and community data on
Gemma 4 inference speed reinforce the original recommendation:

1. **Inference speed**. Gemma 4 26B-A4B runs at ~11 tok/s vs Qwen's 60+ tok/s
   on the same consumer GPU -- a ~6x gap confirmed by multiple community
   reports. This makes Gemma 4 MoE impractical for interactive deployment.

2. **Tool calling now measured**. Qwen3.6-35B-A3B scores MCPMark 37.0,
   slightly ahead of Gemma 4's 36.3. Combined with Qwen3.5's published
   IFEval (91.9) and BFCL (~62%), Qwen has the stronger tool-use story.
   Gemma 4 still has no published IFEval or BFCL scores (mid-April 2026).

3. **Qwen3.6 as free upgrade**. Same architecture as Qwen3.5, so all Unsloth
   training infrastructure (MoE kernels, LoRA configs, chat templates) carries
   over unchanged. Better benchmarks across the board: GPQA Diamond 86.0
   (up from 84.2), SWE-bench 73.4 (up from 70.0), Terminal-Bench 51.5
   (up from 40.5).

4. **MoE LoRA trainability**. Gemma 4's fused 3D expert tensors remain broken
   for QLoRA with no fix in sight. Qwen's separate 2D layers work with
   Unsloth's MoE kernels (~12x speedup, >35% VRAM savings).

5. **Text benchmark lead at small sizes**. Qwen3.5-4B beats Gemma 4 E4B by
   ~10 points on MMLU-Pro. For the dense student starting point, Qwen remains
   clearly ahead.

6. **Distillation framework alignment**. The on-policy distillation paper
   validated on Qwen models. Using the same family reduces unknowns.

7. **Context window**. 262K vs 128K for the small models. Not critical for
   our short-conversation use case, but a nice margin.

**Where Gemma 4 could win**: Arena AI chat preference scores favor Gemma 4
(~40 Elo points higher), suggesting better subjective response quality in
general chat. Gemma 4 also leads on MMMLU multilingual (86.3 vs 85.2), which
could matter for Italian. However, these advantages are outweighed by the
inference speed gap and the lack of published IFEval/BFCL scores.

**Recommendation**: Proceed with Qwen3.5 for dense students (4B, 9B) and
Qwen3.6-35B-A3B for the MoE student (Scenario D). Gemma 4 is no longer a
realistic fallback for the MoE tier given the inference speed and training
issues.

## Qwen3.5 Student Assessment by Role Suitability

Our model needs to reason about social situations via thinking traces before
responding. This raises the bar: the student must be capable of coherent
internal reasoning, not just surface-level pattern matching. Qwen3.5 small
models (0.8B-9B) ship with thinking disabled by default -- it must be
explicitly enabled during training and inference.

**0.8B** -- Not recommended. IFEval 52.1 and estimated BFCL ~35% are too weak
for reliable tool calling or multi-turn orchestration. At this size, thinking
traces are likely to be incoherent or degenerate into repetition.

**2B** -- Marginal. IFEval 78.6 is decent but BFCL 43.6% means tool calling
will be unreliable. Thinking quality at 2B is an open question -- the model
may produce superficial reasoning that doesn't meaningfully improve responses.
Viable only as a conversation-only personality model with an external tool
dispatcher and no thinking requirement.

**4B** -- Solid starting point. IFEval 89.8 is strong. BFCL ~55% is workable
for simple tool patterns (wiki search, single delegation). 32 layers give
reasonable depth for learning social reasoning patterns. The key question is
whether 4B has enough capacity for *both* personality and coherent thinking.
Fits on a single consumer GPU for both training and inference.

**9B** -- Strongest dense option. IFEval 91.5 and BFCL 66.1% make it the most
reliable orchestrator among the dense models. MMLU-Pro 82.5 suggests strong
general reasoning -- important now that we need the model to reason about
emotional dynamics, not just reproduce personality patterns. Requires ~18 GB
for inference (fits RTX 4090) or ~8 GB quantized.

**35B-A3B (MoE)** -- Best benchmarks overall: IFEval 91.9, MMLU-Pro 85.3.
The 64-layer architecture provides the deepest representational capacity,
which matters more now that the model must learn both personality and social
reasoning. Inference speed is comparable to the 4B dense (only ~3B params
activate per token), and at Q4 quantization it fits in ~8 GB VRAM. However,
training is significantly more expensive (see
[Infrastructure & Costs](infra-costs.md)). **Qwen3.6-35B-A3B** is the
preferred base for this tier: same architecture (so training infra carries
over), with improved tool calling (MCPMark 37.0 vs 27.0), agentic coding
(SWE-bench 73.4 vs 70.0), and native multimodal support.

### Tradeoff Summary

| Dimension | 2B | 4B | 9B | 35B-A3B |
|---|---|---|---|---|
| Personality capacity | Low | Medium | High | Highest |
| Social reasoning capacity | Poor | Uncertain | Good | Best |
| Tool use reliability | Poor | Adequate | Good | Good |
| Instruction following | Decent | Strong | Strong | Strong |
| Inference cost | Lowest | Low | Medium | Low (MoE) |
| Training cost | Lowest | Low | Medium | **High** |
| On-device feasibility | Phone/laptop | Laptop | Desktop | Laptop (Q4) |
| LoRA complexity | Simple | Simple | Simple | **Complex** |

The key tension for the MoE option is between inference efficiency (excellent)
and training complexity (high). See the [Training Strategy](training-strategy.md)
for LoRA details.

## Teacher Model

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

**Alternative teacher: Qwen3.5-122B-A10B (MoE)**. MMLU-Pro 86.1, GPQA 85.5.
Stronger than the 27B but requires ~70 GB VRAM at FP8, meaning 2x L40S or a
single A100 80GB. Could be worth it if the 27B's signal proves insufficient,
but adds significant infrastructure cost.

**Alternative teacher: Gemma 4 31B**. MMLU-Pro 85.2, GPQA 84.3, MMMLU 88.4.
Slightly weaker than Qwen3.5-27B on knowledge benchmarks but stronger on
multilingual tasks. Lower VRAM for inference (~24 GB BF16 vs ~56 GB). Could be
worth exploring if Italian language quality from the Qwen teacher is
insufficient, but cross-family distillation (Gemma teacher -> Qwen student)
introduces tokenizer and architecture mismatches that may degrade logprob
signal quality.

## Self-Distillation

The on-policy distillation paper shows that using the *pre-SFT version of the
same model* as teacher effectively recovers lost post-training behaviors. This
means we could:

1. SFT the student (e.g. Qwen3.5-4B or Qwen3.6-35B-A3B) on personality data.
2. On-policy distill using the *original* pre-SFT checkpoint as teacher.

This eliminates the need for a larger teacher model entirely, cutting
infrastructure costs significantly. The tradeoff is that the teacher can only
recover behaviors the student already had -- it can't teach new ones. For our
use case (personality + tool use), this may be sufficient since tool use is
already present in the base model.
