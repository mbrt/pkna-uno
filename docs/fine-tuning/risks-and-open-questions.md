# Risks & Open Questions

Back to [Fine-Tuning Design](../fine-tuning-design.md).

## Risks

1. **Thinking trace quality at small sizes**: The model must produce coherent
   social reasoning in its thinking traces, not just personality in its visible
   output. At 4B, thinking quality is unproven -- the model may generate
   superficial or repetitive reasoning that doesn't meaningfully improve
   responses. If 4B thinking is poor, we may need to jump to 9B earlier than
   planned.

2. **Tool-use degradation after SFT**: The on-policy distillation paper shows
   this is recoverable, but tool calling is more structured than general
   instruction following. May need tool-use-specific prompts in the distillation
   stage.

3. **Thinking + personality + tool use is a triple burden**: Each capability
   competes for the model's limited capacity. Previous distillation work
   validated personality + tool use, but adding structured thinking traces is a
   new axis. The SFT dataset must balance all three, and the distillation stage
   must recover all three.

4. **Small dataset**: ~1,500 SFT examples is small. The distillation paper's
   personalization experiment used a similar scale successfully, but our task
   (personality + tool use + reasoning) is more complex. Synthetic data quality
   matters a lot -- especially for thinking traces, which must be diverse and
   non-formulaic.

5. **Italian language**: The base models are multilingual but Italian-specific
   personality nuances (sarcasm, idiomatic expressions) may be harder to learn
   than English equivalents. May need to oversample Italian examples.

6. **Evaluation subjectivity**: Personality adherence is inherently subjective.
   The LLM-as-judge approach helps but is not ground truth. Human evaluation
   (even informal vibe checks) remains important. Evaluating thinking trace
   quality adds another subjective dimension.

7. **MoE LoRA maturity**: LoRA on MoE is less battle-tested than on dense
   models. Expert routing imbalance means uniform LoRA wastes capacity on cold
   experts. Recent work (MoE-Sieve, DR-LoRA) addresses this but adds
   implementation complexity. The "LoRA Without Regret" paper validated
   per-expert LoRA on Qwen3-30B-A3B, which is architecturally similar to
   Qwen3.5-35B-A3B, so the approach is grounded but not trivial.

## Open Questions

1. **MoE vs 9B dense**: The 35B-A3B has better benchmarks than the 9B dense
   (IFEval 91.9 vs 91.5, MMLU-Pro 85.3 vs 82.5) and comparable inference
   speed, but 5-6x higher training cost. Is the benchmark gap worth it for a
   personality task? The 9B may be "good enough" at much lower cost.

2. **How much personality + reasoning can a 2B model hold?** If the 2B proves
   sufficient for personality but weak on thinking or tool use, a hybrid
   architecture (2B personality model + external tool router, no thinking) could
   be cheaper than a 4B that does everything. But losing thinking means losing
   the social reasoning that makes responses feel grounded.

3. **System prompt at inference**: Should the fine-tuned model still receive a
   brief system prompt, or should personality be fully internalized? Likely a
   minimal reminder helps, but this needs ablation.

4. **Continual learning**: As the claim ledger grows (new comic issues, refined
   observations), can we incrementally update the model without full retraining?
   The on-policy distillation paper's phase-alternating approach (SFT new data
   -> distill to recover) is promising here.

5. **Thinking trace latency**: Thinking adds tokens before the visible
   response, increasing time-to-first-visible-token. For a conversational
   character, perceived responsiveness matters. How long can the thinking
   trace be before the user experience suffers? Can we train the model to
   keep social reasoning concise (2-3 sentences) without losing quality?

6. **Serving**: For on-device deployment (2B/4B/35B-A3B at Q4), quantization
   impact on personality quality is unknown. The MoE model is particularly
   interesting here: at Q4 it fits in ~8 GB (same as 4B at BF16) but has
   access to 35B parameters of knowledge. However, MoE quantization can
   degrade expert routing quality.
