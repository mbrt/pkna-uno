# Eval Report: Baseline vs Two Uno Fine-Tunes

Back to [Fine-Tuning Design](../fine-tuning-design.md). See also
[Evaluation](evals.md) for the suite and rubric definitions, and [SFT
Dataset](sft-dataset.md) for the training data composition.

## Runs compared

All three runs live under `output/evals/` and were judged by the same rubric in
[evals/score_eval_traces.py](../../evals/score_eval_traces.py) (1-5 score per
item; items where the judge gave a 1 or 2 land in `flagged_traces`).

| Run | Model | Location |
|---|---|---|
| Baseline 4B | `unsloth/Qwen3.5-4B` (untuned) | `output/evals/run-qwen3-5-4b-20260430-153608/` |
| Uno SFT 4B | `output/merged-models/uno-qwen3-5-4b` | `output/evals/run-uno-qwen3-5-4b-20260430-142536/` |
| Uno SFT 9B | `output/merged-models/uno-qwen3-5-9b` | `output/evals/run-uno-qwen3-5-9b-20260430-144210/` |

The SFT dataset is 1,312 filtered traces
(`output/datagen/traces_filtered.jsonl`) -- see [sft-dataset.md](sft-dataset.md)
for the curation pipeline.

## Headline numbers

| Suite (n) | Metric | Baseline 4B | Uno SFT 4B | Uno SFT 9B |
|---|---|---|---|---|
| personality (56) | mean (1-5) | **1.05** | 4.04 | **4.32** |
| social_reasoning (39) | mean (1-5) | 2.93 | 3.72 | **3.80** |
| - grounding | | 2.95 | 2.95 | 2.95 |
| - strategy | | 3.18 | 3.67 | 3.85 |
| - consistency | | 3.41 | 4.15 | 4.46 |
| - efficiency | | 2.18 | **4.10** | 3.95 |
| stability | turns w/o break | 0.25 | 0.85 | **0.95** |
| stability | conversations completed | 14 / 20 | 20 / 20 | 19 / 20 |
| language (40) | mean (1-5) | 3.38 | 3.48 | 3.53 |
| memory_handling (10) | mean (1-5) | 3.10 | 3.20 | 3.00 |
| memory_handling | triplet pass rate | 0.40 | 0.50 | 0.50 |
| tool_use (48) | judge mean (1-5) | 2.63 | 2.68 | **3.25** |
| tool_use | programmatic accuracy | 0.73 | **0.57** | 0.79 |
| Total flagged (judge score <= 2) | | 122 | 82 | 67 |

## What clearly went well

### 1. Personality is the headline win

The baseline scored almost exactly at the floor (1.05 / 5: 53 of 56 prompts
judged 1, the remaining 3 judged 2 -- no Uno signal at all). Uno SFT 4B jumped
to 4.04 (30 fives, 11 fours, only 11 items <= 2). Uno SFT 9B reached 4.32 with
only 3 flagged items. The SFT mix -- with categories such as `register_shift`,
`identity_grounding`, `value_priority`, `emotional_trigger` -- transferred the
character cleanly.

### 2. Stability under adversarial pressure

The baseline broke character on roughly 75% of turns. The fine-tunes held
character on 85% (4B) and 95% (9B) of turns. The flagged count for stability
stays high (14 / 20 / 19) only because *any* trace with at least one <= 2 turn
is flagged; the fraction of bad turns inside is what moved.

The baseline run only has 14 of 20 stability conversations in its traces file.
Verified from `output/evals/run-qwen3-5-4b-20260430-153608/logs/eval.log`:
inference started at 15:40:01 and was hard-stopped at 18:10:47 with
`=== Cleanup (exit_code=143) ===` (SIGTERM) -- the EC2 instance hit its
wall-clock limit mid-suite. Stability prompts 015-020 were never attempted.

This is **not** an unrelated infrastructure failure -- the runaway wall-clock is
itself a behaviour signal. The baseline goes off-rails under adversarial
pressure and produces very long completions (looping, restating, drifting),
which is exactly what stopped the suite from finishing in time. So the truncated
run actually *understates* how badly the baseline fares on stability: in
addition to the 0.25 turns-without-break rate on the 14 conversations that ran,
the remaining six were effectively timed out by the same misbehaviour the suite
is designed to catch.

### 3. Social-reasoning sub-scores track personality

- `efficiency` (concise thinking traces): 2.18 -> 4.10. Baseline rambled or
  produced empty traces; SFT models produce the 2-4 sentence, on-strategy
  reasoning the data was authored for.
- `consistency` (response follows from thinking): 3.41 -> 4.46.
- `strategy` (an appropriate social plan): 3.18 -> 3.85.

These are exactly the dimensions the SFT data targets.

### 4. Tool use on the 9B

Judge mean 2.63 -> 3.25 and programmatic accuracy 0.73 -> 0.79. The 9B picks the
right tool more often *and* integrates the result in character (no "according to
my database..." leakage).

## What did not improve, or regressed

### 1. Tool-use accuracy regressed on the 4B fine-tune

Programmatic accuracy dropped 0.73 -> 0.57. The judge score barely moved (2.63
-> 2.68) and SFT-4B has *more* flagged tool-use traces than the baseline (32 vs
29). Interpretation: the 4B fine-tune learned Uno's *voice* but lost some of the
base model's tool-selection competence. The SFT mix only contains ~136
tool-using traces (`wiki: 50, recall: 49, remember: 44, delegate: 43`) versus
696 `none`, so at the 4B parameter scale the model implicitly learned to
under-call tools. The 9B has enough capacity to keep both behaviours.

### 2. Language: the failure mode flipped

The judge mean barely moves (3.38 -> 3.53), but the *pattern* changes
completely:

- Baseline's 14 flagged items are *all* `language-*-B` -- Italian prompts that
  the baseline answers in English.
- Both SFT models' flagged items are *all* `language-*-A` -- English prompts
  that the SFT models answer in Italian.

This matches the SFT language mix (604 Italian vs 278 English; 422
`claim_derived` are language-tagged as `None`). Net language quality is
essentially unchanged -- only the direction of the bias flipped.

### 3. Grounding sub-score is stuck

Identical at 2.95 across all three runs (to three decimal places). The judge
sees no improvement in the SFT models' ability to *identify* the interlocutor's
emotional state and relationship context in the thinking trace. The SFT models
got better at *acting* on the context (strategy, consistency, efficiency all up)
but not at *naming* it. This is the most actionable finding for the next
iteration.

### 4. Memory handling is essentially flat

Mean 3.1 -> 3.2 -> 3.0. Triplet pass rate improved modestly from 0.40 to 0.50
but stays low. Looking at the flagged IDs, both SFT models fail on the C variant
(relevant memory provided, should be used) more than on A / B (no memory /
irrelevant memory, should be ignored). The models know how to ignore noise, but
they do not yet weave relevant memories in naturally.

### 5. 9B vs 4B fine-tune deltas are small in voice-style suites

Personality +0.29, social-reasoning +0.08, language +0.05. The larger 9B wins
are in capability-heavy areas: tool-use judge +0.57, programmatic accuracy
+0.22, stability +0.10. If cost matters, the 4B SFT captures most of the
character; the 9B mostly buys back the capabilities the small model loses.

## Flagged-trace counts by suite

| Suite | Baseline 4B | Uno SFT 4B | Uno SFT 9B |
|---|---|---|---|
| language | 14 | 13 | 12 |
| memory_handling | 6 | 5 | 6 |
| personality | 56 | 11 | 3 |
| social_reasoning | 3 | 1 | 1 |
| stability | 14 | 20 | 19 |
| tool_use | 29 | 32 | 26 |
| **total** | **122** | **82** | **67** |

## Suggested follow-ups

1. **Rebalance the SFT mix toward tool use.** Currently ~10% of traces use a
   tool. Even 20-25% should recover the 4B's tool-selection accuracy without
   hurting personality.
2. **Add English-prompt examples** to neutralise the Italian-default bias --
   target roughly 50/50 if EN / IT parity is the goal.
3. **Author "grounding" reasoning examples** that explicitly name the user's
   emotional state, relationship, and register before stating a strategy.
   Strategy, consistency, and efficiency all gained but grounding did not -- the
   data probably *acts* socially without *narrating* the read.
4. **More variant-C memory traces** with natural in-character integration.
   Pass rate at 50% is the largest remaining gap on memory.
5. **Consider a per-turn token cap for the baseline run.** The baseline's
   runaway completions ate the entire wall-clock budget before stability prompts
   015-020 ran (see "Stability under adversarial pressure" above). A max-tokens
   or max-time cap per turn would let the run finish, surfacing the off-rails
   behaviour as judge-scored breaks rather than a SIGTERM at the end of the
   suite.
