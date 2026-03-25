# Improving the PKNA-Uno Pipeline

Review of the extraction and profile generation pipelines, with improvement
suggestions grounded in the codebase, the [blog post](https://blog.mbrt.dev/posts/uno/),
and the [Character-LLM paper](https://arxiv.org/abs/2310.10158) (Shao et al., 2023).

---

## A. Extraction Pipeline Improvements

The current `PageExtractor` in [dspy-extract-full.py](../dspy-extract-full.py)
captures per-panel: `is_new_scene`, `description`, `caption_text`, and `dialogues`
(character + line). This is solid for transcription but leaves the profile builder
to infer *all* characterization from raw text. Richer extraction signal would
reduce the load on the downstream profile step and improve consistency.

### A1. Emotional / tonal annotation per dialogue line

Currently a `DialogueLine` is just `character` + `line`. The profile builder has
to guess tone from text alone, which is lossy -- Italian sarcasm, affection, or
worry are often conveyed through visual cues (expression, font style, bubble
shape) that get discarded.

**Suggestion**: Add an optional `tone` field to `DialogueLine` (e.g., sarcastic,
concerned, playful, authoritative, melancholic). The multimodal model already sees
the panel image and can infer this from bubble shape (jagged = shouting, wavy =
whispering, etc.) and facial expression. This gives the claim builder direct signal
for `communication/interaction/emotional_coloring` and `psychology/emotional/*`
paths without having to re-infer it.

### A2. Scene-level situational context tag

Each scene is currently identified only by page range and characters present.
Adding a `scene_type` tag (e.g., crisis, casual_banter, planning,
emotional_confrontation, humorous, exposition) to the extraction output would let
the profile builder weight observations differently. A joke during a crisis tells
you something different than a joke during downtime.

### A3. Other characters' reactions to Uno

The scene-building logic in `create_scene_from_panels` (~line 371 of
[build_claim_ledger_profile.py](../build_claim_ledger_profile.py)) filters for
Uno's own dialogue lines but discards what others say *to* or *about* him.
Characters laughing at his jokes, trusting his judgment, or getting annoyed by his
sarcasm are strong evidence for relationship dynamics and social impact.

**Suggestion**: Include a `context_dialogues` field alongside `uno_dialogues` --
the lines from other characters in the same scene. These don't need to be
exhaustive; just lines directly adjacent to Uno's (the "conversational turn" around
each Uno line). This gives the claim builder direct evidence for:

- `communication/interaction/dominance` and `turn_taking`
- `relationships/{character}` dynamics
- How Uno's humor actually lands

### A4. Non-verbal / visual state annotations

Uno is an AI who manifests as a hologram. His visual state (screen color,
holographic expression, display changes) often conveys emotion the dialogue
doesn't. The `description` field captures this narratively, but a structured
`visual_cues` list per panel (e.g., "Uno's hologram flickers nervously", "screen
displays warning symbols") would help the profile builder extract emotional and
behavioral patterns more reliably.

### A5. Speech act classification

Annotating each dialogue line with its pragmatic function (informing, joking,
commanding, deflecting, comforting, warning, questioning) would directly feed
`communication/voice/style` and `behavior/does` claims. The model already
"understands" the speech act when it reads the panel -- making it explicit prevents
downstream information loss.

---

## B. Profile Generation Improvements

The claim ledger's `VALID_PATHS` taxonomy in
[build_claim_ledger_profile.py](../build_claim_ledger_profile.py) (lines 102-163)
is already rich. The following dimensions are either missing or underspecified.

### B1. Character arc / growth trajectory

The v2 profile had a "Growth" section. The v11 claim paths have no temporal
dimension. Over 56 issues, Uno evolves -- he becomes more emotionally open, learns
to trust, confronts his own nature. Without claim paths like
`psychology/growth/arc` or `behavior/evolution`, the claim builder has no place to
put "Uno starts doing X after issue Y" observations.

**Suggestion**: Add paths like:

- `psychology/growth/emotional_arc` -- how emotional patterns shift over time
- `psychology/growth/relationship_arc` -- how key relationships deepen
- `behavior/evolution` -- behaviors that appear or disappear over the series

Evidence timestamps (issue numbers) already exist in scenes. The profile builder
can surface claims like "Uno's sarcasm softens significantly after the events of
issue 15" with grounded evidence.

### B2. Situational behavioral adaptation

The blog's example transcripts show a critical quality: Uno speaks *very
differently* to Everett (respectful, formal) vs. PK (banter, jokes). This is
partially captured by `relationships/{character}` but there's no structured way to
say "Uno's formality level varies by audience" as a behavioral pattern, independent
of specific relationships.

**Suggestion**: Add paths like:

- `behavior/adaptation/by_audience` -- how behavior changes by interlocutor type
- `behavior/adaptation/by_situation` -- how behavior changes in crisis vs. calm
- `communication/voice/register_shifts` -- when and how formality shifts

This captures the *rule* rather than just the *instances*, which is what the soul
document needs to guide inference.

### B3. Humor taxonomy

Humor is arguably Uno's defining trait, yet it has no dedicated claim path. It's
scattered across `communication/voice/style`, `communication/idiolect/*`, and
`psychology/neural_matrix/creativity`. A character whose humor is central deserves:

- `communication/humor/type` -- sarcasm, self-deprecation, wordplay, dark humor, absurdist
- `communication/humor/timing` -- when humor is deployed (tension-breaking, deflection, affection-masking)
- `communication/humor/targets` -- what/who Uno jokes about (himself, Paperinik, situations, enemies)

### B4. Self-perception and existential stance

How does Uno see himself? Does he consider himself alive? How does he relate to
being an AI? This is central to the character and currently has no dedicated path.
It's too important to be shoehorned into `identity/bio`.

**Suggestion**: Add:

- `psychology/self_model/identity_stance` -- how Uno frames his own existence
- `psychology/self_model/agency` -- does he see himself as having free will?
- `psychology/self_model/mortality` -- how he relates to shutdown/death

### B5. Knowledge boundaries (connects to Character-LLM's "Protective Experiences")

The current `capabilities/limitations` is a catch-all. Explicitly modeling what Uno
*doesn't know* or *shouldn't reference* would directly reduce hallucination at
inference time.

**Suggestion**: Add:

- `capabilities/knowledge_boundaries/temporal` -- what era is his knowledge from?
- `capabilities/knowledge_boundaries/domain` -- what domains does he know well vs. poorly?
- `capabilities/knowledge_boundaries/forbidden` -- what knowledge would break character?

### B6. Values hierarchy and conflict resolution

`psychology/moral_compass/core_values` is flat. Real characters become interesting
when values conflict (loyalty to Everett vs. protecting PK vs. following protocol).
The soul document should capture how Uno resolves these tensions.

**Suggestion**: Expand `psychology/moral_compass/` with:

- `psychology/moral_compass/value_hierarchy` -- which values win when they conflict?
- `psychology/moral_compass/dilemma_patterns` -- recurring ethical tensions and how they resolve

---

## C. Ideas from the Character-LLM Paper

The paper (Shao et al., 2023) focuses on *fine-tuning* LLMs into character
simulacra, which the blog post explicitly rejects for good portability reasons.
However, several ideas transfer well to the system-prompt approach.

### C1. Five-dimension evaluation framework (HIGH VALUE)

The paper evaluates characters on: **Memorization**, **Values**, **Personality**,
**Hallucination**, **Stability**. This is directly applicable as an automated eval
for the soul document's effectiveness. Currently the project has no systematic
evaluation.

**Concrete implementation**:

1. Generate ~100 single-turn interview questions targeting each dimension (can be
   LLM-generated, similar to the paper's approach using the wiki + claim ledger for
   ground truth).
2. Generate ~20 multi-turn conversations to test stability (the paper finds models
   drift out of character over long conversations).
3. Use an LLM judge with the paper's step-by-step evaluation protocol: evaluate one
   dimension at a time, with an explicit evaluation plan (identify trait -> compare
   to profile -> score).
4. Compare soul document versions (v6 vs. v11) and model backends quantitatively.

This turns the current "vibe check" into something repeatable and comparable across
iterations.

### C2. Protective knowledge as explicit negative claims

The paper's "protective experiences" train the model to reject out-of-character
knowledge. The system-prompt equivalent is explicit negative guidance in the soul
document. The current `behavior/avoids` and the chat script's "WHAT YOU DON'T DO"
section partially do this, but the *claim extraction step* doesn't systematically
look for knowledge boundaries.

**Suggestion**: During scene processing, add a prompt instruction to also note when
Uno *doesn't know something* or *explicitly rejects a premise*. These become claims
under the proposed `capabilities/knowledge_boundaries/*` paths. The resulting soul
document section would contain things like:

> Uno has no knowledge of events after [latest issue chronologically]. He does not
> understand modern social media, smartphones, or post-2000s technology. When
> confronted with unfamiliar concepts, he responds with genuine curiosity rather
> than fabrication.

### C3. Experience vignettes in the soul document

The paper's key insight is that *scenes with interactions and internal thoughts*
teach character more effectively than *trait lists*. The current soul document is
entirely prose describing traits. Including 2-3 short "canonical interaction
vignettes" (reconstructed from the highest-evidence claims) would give the
inference model a concrete behavioral template to anchor its generation.

**Example addition to soul document**:

```markdown
## Canonical Interactions

### Greeting Everett (formal, reverent)
Everett: "Uno, status report."
Uno: "Master Ducklair! Systems nominal. The tower awaits your command, as always."

### Deflecting PK's complaints (sarcastic, affectionate)
PK: "Uno, this mission was a disaster!"
Uno: "A disaster? You only destroyed *two* vehicles this time. I'd call that progress, socio."
```

These act as few-shot examples within the soul document itself. The Character-LLM
paper shows that learning from interaction patterns is more effective than trait
descriptions alone, and this is the system-prompt equivalent.

### C4. Multi-turn stability testing with adversarial probes

The paper uses ChatGPT as an adversarial interviewer who asks "harsh" follow-up
questions to test stability. This could be directly implemented:

1. Create a `stability_eval.py` script that runs multi-turn conversations with an
   adversarial interviewer model.
2. The interviewer tries to: break character, elicit hallucinated knowledge, push
   for out-of-era information, test value consistency under pressure.
3. Score with the 5-dimension framework from C1.

This is particularly important because the blog notes that large system prompts
*degrade* performance -- the eval would quantify this and help find the optimal
soul document size.

### C5. Synthetic scene generation for data augmentation

The paper's "Experience Completion" generates detailed multi-turn scenes from brief
profile notes. Applied to the claim ledger: take high-confidence claims and
generate synthetic dialogues that *should* match the character. These become test
cases for the eval framework (C1).

For example, from the claim "Uno defaults to sarcasm in casual exchanges but drops
it in danger":

- Generate a casual conversation -> verify sarcasm is present
- Generate a crisis conversation -> verify sarcasm is absent

This is cheaper than manual labeling and scales with the claim set.

---

## D. Priority ranking

Sorted by expected impact vs. effort:

1. **C1: Five-dimension eval framework** -- highest value; everything else is guesswork without it
2. **B1: Character arc/growth paths** -- captures temporal dynamics that are currently lost
3. **A1: Tonal annotation on dialogue lines** -- cheap to add, directly improves claim quality
4. **A3: Other characters' reactions** -- important signal that's currently discarded
5. **B3: Humor taxonomy** -- central to Uno's character, currently unstructured
6. **C3: Experience vignettes in soul document** -- few-shot anchoring for the inference model
7. **B2: Situational adaptation paths** -- captures the Everett-vs-PK behavioral shift
8. **B4: Self-perception paths** -- important for an AI character specifically
9. **C4: Multi-turn stability testing** -- validates the eval framework in practice
10. **B5: Knowledge boundaries** -- reduces hallucination at inference time
