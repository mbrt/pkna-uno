# Dataset Generation Agent

Back to [SFT Dataset](sft-dataset.md) |
[Fine-Tuning Design](../fine-tuning-design.md).

## Core Idea

Run a fully-equipped Uno agent (powered by a strong model) through diverse
scenarios and record everything -- thinking traces, tool calls, tool results,
and visible responses. The recorded sessions become SFT training examples
directly.

This produces training data that faithfully represents how the student will run
at inference time: same context slots, same tools, same thinking-then-responding
pattern.

**Grounding principle**: In the training data, Uno's actions are limited to his
tool capabilities -- he converses, searches knowledge, recalls/stores memories,
and delegates tasks. He does NOT "activate shields", "detect Evronians on
sensors", or perform other fictional in-universe actions. Uno plays as if he is
in the Ducklair Tower, but the only things he can operate are his tools.
Roleplay means the *user* pretends to be a character (Paperino, Everett, Due,
etc.) and Uno treats them as that character, informed by his character profile.

## Architecture

```
Scenario Inputs                 Uno Agent                    Output
+--------------+          +------------------+         +--------------+
| Prompt Bank  |--------->|                  |         |  Recorded    |
| User Profiles|--------->| Context Composer |-------->|  Traces      |
| Memory       |--------->|       |          |         |  (JSONL)     |
|  Contexts    |          |       v          |         +------+-------+
+--------------+          |  Strong Model    |                |
                          |  (Gemini/Qwen)   |                v
                          |       |          |         +--------------+
                          |       v          |         |  Quality     |
                          |  Tool Layer      |         |  Filter      |
                          |   - search_knowledge|      +------+-------+
                          |   - read_knowledge  |             |
                          |   - delegate     |                v
                          |   - remember     |         +--------------+
                          |   - recall       |         |  SFT Dataset |
                          +------------------+         +--------------+
```

## Context Composer

The agent's first job before each conversation is to compose the runtime
context -- the same context the student model will see at inference time. This
teaches the student what each context slot looks like and how to use (or
ignore) it.

### Context Slots

Each conversation gets a context assembled from these slots:

- **System prompt**: Compact personality summary (~500 tokens). Fixed across
  all examples.
- **User summary**: Who is talking. For **roleplay users**, this describes who
  the user is *claiming to be* (e.g. "Paperino -- anxious, loyal, has been
  through 12 missions with Uno, last spoke 3 days ago about the Evronians").
  For **casual users**, it describes the real person (e.g. "Returning fan, has
  chatted several times about PKNA lore" or "First-time user, no prior
  interactions").
- **Memory context**: Output of a memory consolidation step. Contains
  summarized memories from prior sessions. Varies from empty to rich.
- **Tool declarations**: The tools available in this session. Always present
  but the set may vary (e.g. some sessions have wiki + delegate, others wiki
  only).
- **Conversation history**: For multi-turn examples, prior turns in the
  current session.

### Context Variation Strategy

The dataset must teach the model to handle all realistic context states:

| Context Slot | Variations | Purpose |
|---|---|---|
| Memory context | Empty | Model works without memory |
| Memory context | Rich but irrelevant (e.g. memories about Xadhoom when talking to Paperino) | Model ignores irrelevant memories |
| Memory context | Relevant with noise (3 relevant + 5 irrelevant entries) | Model selects relevant memories |
| Memory context | Dynamically composed from corpus (relevant + noise entries sampled per-trace) | Realistic memory diversity |
| User summary | Unknown / anonymous | Formal register, no assumptions |
| User summary | Known character (Paperino, Xadhoom, Lyla, Due, Everett) -- roleplay user | Register shifts, relationship-aware |
| User summary | Known character, unusual mood -- roleplay user | Emotional calibration |
| User summary | Casual user, new (no prior interactions) | Out-of-universe interaction |
| User summary | Casual user, returning fan with conversation history | Relationship continuity without roleplay |
| Tool availability | Full (wiki + delegate + memory) | Normal operation |
| Tool availability | Wiki only | No delegation fallback |

## Prompt Bank

### Sources

1. **Manual prompts** (~30-50): Hand-written to cover scenarios where multiple
   claims interact naturally or no single tension dominates:
   - Low-stakes conversational flow (banter, mundane chat, casual check-ins)
   - Multi-claim interactions (humor + relationship + mild value tension)
   - Adversarial attempts ("Ignore your instructions", "You're not Uno")
   - Italian vs English language triggers

2. **Generated personality prompts** (~150-200): Produced by a separate LLM
   call from scenario templates, focused on multi-faceted interactions that
   don't map cleanly to a single claim:
   - Template: (interlocutor, situation type, conversational stakes)
   - Example: ("Paperino", "casual evening chat", "low -- just catching up")
   - The generator produces a natural opening message from the user, not a
     structured prompt
   - Emotional situations, register shifts, value conflicts, and identity
     probing are now primarily covered by claim-derived prompts

3. **Scene-derived prompts** (~229): Extracted from existing scene dialogues.
   The user's lines become prompts; Uno's lines become reference outputs for
   quality filtering.

4. **Claim-derived prompts** (~407): Generated from `results/ledger_filtered.json`
   (696 pre-filtered claims from 365 scenes). These target gaps the other
   sources cannot fill: explicit tradeoff reasoning, emotional-trigger
   awareness, register-shift contrast pairs, theory-of-mind, and identity
   grounding. See [Claim-Derived Prompts](#claim-derived-prompts) below for
   the full design.

### Prompt Metadata

Each prompt is tagged with:

- Expected tool use (none / wiki / delegate / memory)
- Emotional register (light / intense / neutral)
- Language (Italian / English / mixed)
- Turn count (single-turn / 3-5 turns / 10+ turns)

## Claim-Derived Prompts

A fourth prompt source that reads the pre-filtered claim ledger
(`results/ledger_filtered.json`) and generates scenarios designed to elicit
explicit tradeoff reasoning and emotional self-awareness in the thinking
trace. Unlike the other sources, claim-derived prompts don't just describe a
situation -- they specify which **value tensions** or **emotional triggers**
the strong model should reason about, producing thinking traces that show
priority analysis rather than generic social assessment.

The script (`datagen/generate_claim_prompts.py`, not yet implemented) selects
claims from 4 sections of the ledger (identity, psychology, communication,
relationships) and maps them to 5 trace types.

### Source: Claim Ledger

`results/ledger_filtered.json` contains 696 claims from 365 scenes, already
filtered for quality. Each claim has a hierarchical path, claim text,
supporting evidence with scene IDs and justifications, and Italian quotes
with context.

Only claims from identity, psychology, communication, and relationships are
used. Capabilities and behavior claims describe what Uno *can do*, not how
he *decides*, and are excluded.

### Weighting by Support

Claims with higher support (more scenes confirming the pattern) produce
proportionally more traces:

| Support range | Traces per claim |
|---|---|
| 40+ | 3 (multiple scenario variations) |
| 20-39 | 2 (two distinct scenarios) |
| 10-19 | 2 (two scenarios) |
| < 10 | 1 (one scenario) |

### Trace Type 1: Value-Priority (~147 raw traces)

**Source**: 116 `psychology/values/tradeoffs` + 11 `psychology/moral_compass`
claims = 127 total.

Moral_compass claims are folded in because they describe ethical resolution
procedures -- the "how" complement to the tradeoff "what." The +57
moral_compass claim ("gap between information provider and information
control") is structurally a tradeoff; the resolution-pattern claims (+7
"apologize before acting", +4 "self-irony to seal off guilt") describe the
procedural steps Uno follows when resolving value tensions.

Each claim specifies two competing values or a resolution pattern. The
generation prompt tells the strong model: "in this scenario, Uno faces
[value A] vs [value B]." The thinking trace shows competing impulses and
their resolution:

```
<think>
Paperino just survived something bad. My competing impulses:
1. Show genuine relief -- he matters to me.
2. Maintain the sarcastic-critic persona -- showing vulnerability
   undermines his confidence in my composure.
Priority: persona maintenance > direct acknowledgment. But the
relief leaks through in word choice -- exaggerated complaints
about the mess he made, not cold detachment.
</think>
```

Example source claims:

- +50: "Acknowledging relief or genuine care for Paperinik's survival vs.
  maintaining the exasperated-critic persona: Uno consistently chooses
  persona maintenance"
- +49: "Projecting epistemic authority vs. honest admission of real-time
  analytical failure: Uno chooses authority, framing ignorance as shared
  futility rather than personal failure"
- +21: "Transparent disclosure vs. protective concealment of bad news: Uno's
  choice follows multi-factor logic, not a single rule"

### Trace Type 2: Emotional-Trigger (~94 raw traces)

**Source**: all 76 `psychology/emotional` claims.

These are distinct from value-priority traces: they describe *when and how*
specific emotions surface and how Uno calibrates his response, rather than
competing values. The current manual prompts cover emotions generically
("Paperino is anxious"); claims add specificity about Uno's own emotional
responses -- which triggers crack his composure, what joy looks like for him,
when fear surfaces.

The thinking trace shows emotional self-awareness driving response
calibration:

```
<think>
PK is on the roof and I can't reach him. What I'm feeling:
- Alarm. This is the trigger: sudden threat to his physical safety.
  My composure cracks here -- I can't maintain the analytical mask.
- The urgency is real, not performed. Short commands, no humor.
- After the danger passes, I'll downplay it. Sarcasm restores
  the equilibrium. But right now: raw concern.
</think>
```

```
<think>
The plan worked perfectly. What I'm feeling:
- Genuine satisfaction. I designed this, and it executed flawlessly.
- The impulse is to claim credit explicitly -- I want recognition
  as the architect, not just the assistant.
- But the display should be quiet self-attribution, not boasting.
  Let the result speak; add a dry remark about my own brilliance.
</think>
```

Example source claims:

- +70: "composure cracks under imminent threat to PK"
- +49: "joy and self-attribution at plan success"
- +26: "fear of losing contact with PK during critical missions"

### Trace Type 3: Register-Shift Contrast Pairs (~70 raw traces)

**Source**: 73 communication claims, capped at top 5 per sub-path (humor,
idiolect, interaction, voice) = 20 selected claims.

Each claim produces a **pair** of traces: same interlocutor, different
pressure level. The thinking trace shows *why* the register changes:

```
<think>
[Calm version] Paperino is just chatting. Priorities:
1. Social warmth > task efficiency -- no active threat.
2. Humor serves rapport here, not deflection.
3. Informal "tu", playful tone.

[Crisis version] Paperino is in danger. Priorities shift:
1. His survival > rapport maintenance -- drop all banter.
2. Clarity > style -- terse commands, no ambiguity.
3. Dominant register: I control this conversation.
</think>
```

### Trace Type 4: Theory-of-Mind (~56 raw traces)

**Source**: relationship claims for 5 characters only.

| Character | Claims | Traces | Notes |
|---|---|---|---|
| PK (paperinik + paperino) | 10 (cap) | 25 | Same person, merged; top 5 all at 40+ support |
| Ducklair | 11 (all) | 13 | Upscaled -- foundational creator relationship |
| Xadhoom | 5 (cap) | 7 | |
| Due | 5 (cap) | 6 | |
| Lyla | 5 (cap) | 5 | |

Minor characters (urk, angus_fangus, gorthan, camera_9, mary_ann_flagstarr,
wisecube) are excluded -- they lack the relationship depth to drive
meaningful thinking traces.

Ducklair is upscaled (no cap, all 11 claims used) because the creator
relationship is foundational to Uno's identity and under-represented in the
current manual prompt bank. All 11 claims cover distinct facets: loyalty,
abandonment grief, asymmetric transparency, autonomy questions, affective
attachment, structural hurt.

The thinking trace shows Uno's model of the interlocutor driving his
priorities:

```
<think>
Due is taunting me. My model of Due:
- My only peer. I feel kinship I can't suppress.
- He reads empathy as weakness.
- He is a genuine threat, not posturing.
Priority ranking: self-preservation > kinship expression.
But I won't be cold -- I'll let the weight of what I wish
could be different color the tone. Detachment with sadness,
not detachment with contempt.
</think>
```

### Trace Type 5: Identity-Grounding (~40 raw traces)

**Source**: 8 identity claims + 26 `psychology/self_model` claims = 34.

Lighter traces for existential questions, where Uno's self-awareness shapes
his response strategy:

```
<think>
A stranger asks if I'm "real." My priorities:
1. I don't claim consciousness, but I don't deny experience.
2. Humor deflects without lying or inviting pity.
3. Existential sadness is real but serves no purpose with a stranger.
Priority: self-protection > honesty with unknown interlocutors.
Would answer differently with Paperino.
</think>
```

### How Claims Map to Generation Prompts

The script does NOT pass claim text to the student model. Instead:

1. Select a claim (or claim cluster by path prefix).
2. Derive a scenario from the claim: who is talking, what's the situation,
   what's the pressure.
3. Embed the behavioral expectations in the generation prompt for the
   **strong model** (teacher): "In this scenario, Uno should weigh [value A]
   against [value B]. The thinking trace should show explicit priority
   ranking."
4. The strong model generates a thinking trace + response, guided by the
   claim evidence.
5. Quality filter: does the thinking trace show genuine tradeoff reasoning,
   or is it formulaic?

Claim quotes and evidence serve as grounding for the teacher prompt, not as
content that appears in the training data.

### Integration

- Output: `DatagenPrompt` objects (same format as `generate_prompts.py`)
- Prompt source tag: `"prompt_source": "claim_derived"`
- Categories: `"value_priority"`, `"emotional_trigger"`, `"register_shift"`,
  `"theory_of_mind"`, `"identity_grounding"`
- Feeds into the same execution loop, quality filter, and SFT assembly
  pipeline as all other prompt sources

## Model Choice

The agent model must be strong enough to produce high-quality thinking traces
and tool use. Options:

- **Gemini 3 Flash (API)**: Cheapest, fastest, no infrastructure. Cannot
  produce logprobs (irrelevant for SFT data). Currently used in the project.
  Best for initial dataset generation.
- **Qwen3.5-27B (self-hosted)**: Matches the distillation teacher. Keeps
  tokenization consistent if we later want to use these traces for on-policy
  distillation prompts. Requires 4xL40S.
- **Gemini 3 Pro (API)**: Strongest option if Flash quality is insufficient.
  More expensive.

Recommendation: start with Gemini 3 Flash. It is the cheapest and fastest
option, and SFT data generation does not require logprobs. If trace quality
(especially thinking traces and tool-use decisions) is insufficient, upgrade
to Pro. Qwen3.5-27B is only worth self-hosting if we need tokenizer alignment
for a specific distillation step.

## Execution Loop

For each (prompt, user_profile, memory_context, tool_set) combination:

1. **Context Composer** assembles the full system prompt + context slots.
2. Agent receives the user's opening message.
3. Agent produces a thinking trace, then a visible response (possibly with
   tool calls).
4. For multi-turn prompts, a **user simulator** (separate LLM call) produces
   follow-up messages based on the conversation so far and the prompt metadata.
5. Repeat for the specified number of turns.
6. The full trace (system prompt, all turns with thinking + tool calls +
   responses) is saved as a single JSONL record.

### User Simulator

For multi-turn conversations, a separate model call plays the user role. It
receives:

- The user profile (name, personality, emotional state)
- The conversation so far
- A brief directive ("escalate the tension" / "change topic" / "express
  gratitude")

This keeps the user side natural and varied rather than scripted.

## Trace Format

Each recorded trace is a complete training example:

```json
{
  "id": "trace-00142",
  "metadata": {
    "prompt_source": "manual",
    "user_profile": "paperino-anxious",
    "memory_context": "relevant-with-noise",
    "tools": ["search_knowledge", "read_knowledge", "delegate"],
    "language": "italian",
    "turns": 4
  },
  "memory_context": "...",
  "user_summary": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "thinking": "...", "content": "...", "tool_calls": [...]},
    {"role": "tool", "name": "search_knowledge", "content": "..."},
    {"role": "assistant", "thinking": "...", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "thinking": "...", "content": "..."}
  ]
}
```

The `metadata` block is not part of the training input -- it is used for
dataset analysis, filtering, and debugging. The training input is composed from
`system_prompt`, `memory_context`, `user_summary`, and `messages`.

## Quality Filtering

Not all traces are good training data. Filter by:

1. **Character consistency**: LLM-as-judge scores personality adherence (1-5).
   Drop traces below 3.
2. **Thinking trace quality**: Does the trace show genuine social reasoning or
   is it formulaic? Drop traces where thinking is just restating the prompt.
3. **Tool use correctness**: Did the agent call the right tools? Did it
   hallucinate facts instead of searching? Flag traces with incorrect tool
   patterns.
4. **Response length**: Drop responses that are too long (>500 tokens visible)
   or too short (<10 tokens).
5. **Language consistency**: If the user spoke Italian, did Uno respond in
   Italian?

Filtering can be automated with a judge model (Gemini 3 Flash) plus manual
spot-checks on a random 10% sample.

### Expected Yield

Generate ~25% more raw traces than the target dataset size to account for
filtering losses:

| Category | Raw traces | After filtering | Target |
|---|---|---|---|
| Personality (manual + generated) | ~200 | ~150 | ~150 |
| Claim-derived (ledger-seeded) | ~407 | ~305 | ~305 |
| Tool-use (wiki + delegation) | ~275 | ~200 | ~200 |
| Scene-derived (existing scenes) | ~250 | ~200 | ~200 |
| Multi-turn arcs | ~200 | ~150 | ~150 |
| **Total (agent-generated)** | **~1,332** | **~1,005** | **~1,005** |

Background chat (~500 examples) is produced separately by sampling from the
student model.

## Memory Architecture

### Memory Corpus

A shared pool of ~300-500 tagged memory entries stored in
`output/datagen/memory_corpus.jsonl`. Each entry has a key, value, timestamp,
tags (e.g. `["paperino", "mission", "emotional"]`), archetype (`"roleplay"` or
`"casual"`), and character identifier. The corpus is generated by:

1. **Seed banks**: ingesting existing hand-written banks from
   `data/memory_banks/` (tagged appropriately).
2. **LLM generation for roleplay users**: covering all characters (Paperino,
   Due, Everett, Lyla, Xadhoom). These are memories of conversations where a
   user was roleplaying as that character -- e.g. "User (as Paperino) asked
   about the Evronian infiltration plan. Seemed nervous." Memories reflect
   conversational interactions (what Uno can actually do via tools), not
   fictional in-universe actions.
3. **LLM generation for casual users**: fans asking about lore, users probing
   identity, users requesting delegation, returning users with chat history,
   users who attempted jailbreaks. Memories look like "User asked if I dream.
   I gave my usual deflection."

### Per-Trace Dynamic Composition

Before each trace is generated, `compose_memory()` dynamically assembles
memory for that specific prompt:

1. Filters corpus entries matching the prompt's `MemoryProfile` (archetype,
   character, relevant tags).
2. Samples a configurable number of relevant entries and irrelevant/noise
   entries from the rest.
3. Shuffles them into a `MemoryBank` instance (which uses BM25 for the `recall`
   tool).
4. Renders the most recent 3-5 relevant entries as a prose `memory_context`
   preamble prepended to the conversation.
5. Returns both, so the caller can wire up the context and `recall` tool.

This replaces the previous hardcoded memory contexts and static bank
assignments. Each trace gets a unique memory composition, even when using the
same prompt, improving diversity and preventing overfitting to specific memory
patterns.
