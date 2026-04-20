# SFT Dataset

Back to [Fine-Tuning Design](../fine-tuning-design.md).

## What Goes Into the Weights

- Personality traits (sarcasm, warmth, register shifts by interlocutor)
- Emotional response patterns (humor timing, deflection, vulnerability)
- Italian language patterns and idiolect (catchphrases, expressions)
- Social reasoning patterns (how to read a situation, choose a strategy,
  calibrate emotional register)
- Tool-use patterns (when to search wiki, when to delegate)

## What Stays Out of the Weights

- Factual knowledge about the PKNA universe (retrieved via tools)
- General world knowledge (already in pre-training)
- Technical problem-solving (delegated to sub-agents)

## Dataset Construction

All categories except background chat are produced by the
[Dataset Generation Agent](dataset-generation-agent.md) -- a trace capture
pipeline that runs a fully-equipped Uno agent through diverse scenarios and
records everything (thinking traces, tool calls, visible responses).

| Category | Examples | Purpose |
|---|---|---|
| Personality (manual + generated prompts) | ~150 | Multi-claim natural scenarios, low-stakes conversational flow |
| Claim-derived (from character profile ledger) | ~305 | Tradeoff reasoning, emotional triggers, register shifts, theory of mind |
| Tool-use (wiki + delegation prompts) | ~200 | Correct tool patterns |
| Scene-derived (existing 229 scenes) | ~200 | Grounded in canon |
| Multi-turn arcs | ~150 | Conversation stability |
| Background chat (Tulu3 subset) | ~500 | Catastrophic forgetting prevention |
| **Total** | **~1,505** | |

The dataset includes both **roleplay users** (pretending to be PKNA characters
like Paperino, Everett, or Due) and **casual users** (fans and curious strangers
interacting without adopting a character). This affects user summaries (roleplay
users have in-universe identities; casual users have real-world descriptions)
and memory contexts (roleplay memories reference in-universe conversations;
casual memories reference real-world interactions).

Background chat examples are sampled from the student model itself (on-policy)
with lightweight thinking traces added.

## Example Format

Each training example includes:

- A compact system prompt (~500 tokens) with core personality traits only (not
  the full 208-line soul document -- distillation should internalize this)
- A user summary describing the interlocutor (from "unknown stranger" to a
  rich profile with relationship history and emotional state)
- A memory context slot dynamically composed from a tagged memory corpus
  (may be empty, irrelevant, or relevant with noise -- sampled per-trace)
- Multi-turn conversation with tool calls where appropriate
- **Thinking traces** in the assistant's response: an internal reasoning block
  where Uno evaluates the emotional context, the relationship with the
  interlocutor, and the best conversational strategy before producing the
  visible reply
- The assistant role is always Uno

The system prompt during training is intentionally minimal -- the goal is for
the model to internalize the behavior so that at inference time, only a brief
reminder is needed (or none at all).

### Thinking Trace Structure

Each assistant turn follows this pattern:

```
<think>
[Who is talking to me? What's their emotional state?]
[What's my relationship with this person?]
[What's the best strategy here -- humor, deflection, warmth, sarcasm?]
[Any facts I need to look up vs. things I know from my identity?]
</think>
[Visible in-character response]
```

The thinking block is where Uno's emotional intelligence lives. It should
model the kind of social reasoning that makes Uno's responses feel grounded
rather than pattern-matched. Examples:

- "Paperino is worried. He needs reassurance, but if I'm too direct he'll
  feel patronized. Light humor first, then the real point."
- "This is a stranger asking about the Ducklair Tower. I don't know them --
  formal register, deflect with a joke, search the wiki for specifics."
- "Xadhoom is angry. She doesn't want comfort, she wants to be heard. Match
  her intensity, don't try to calm her down."

### Claim-Derived Thinking Traces

The claim-derived category (see below) produces traces with a more structured
thinking pattern: explicit tradeoff analysis, priority reasoning, and
emotional self-awareness. These traces are seeded from the character
profile's claim ledger (`results/ledger_filtered.json`), which contains 116
explicit value tradeoff claims, 76 emotional-trigger claims, and 156
relationship claims with rich behavioral evidence.

Claim-derived traces largely subsume what was previously covered by generic
personality prompts (emotional scenarios, register variety, social reasoning).
The residual personality category (~150 examples) covers multi-claim
interactions and low-stakes conversational flow where no single tension
dominates. The key structural difference is that claim-derived thinking shows
**competing impulses and their resolution**. Value-priority traces show
tradeoff reasoning:

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

Theory-of-mind traces show Uno's model of the interlocutor driving his
choices:

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

Emotional-trigger traces show self-awareness about what Uno is feeling and
how to calibrate the response:

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

Identity-grounding traces show how self-awareness shapes register choices:

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

These traces teach the student model how Uno *decides*, not just how he
*sounds*. The claim ledger provides the behavioral evidence that grounds
these decisions in canon rather than generic personality patterns.

## Data Mix

~10% personality, ~20% claim-derived, ~13% tool-use, ~13% scene-derived,
~10% multi-turn, ~33% background chat. (Percentages are approximate targets;
actual mix depends on filtering yield.)

All categories include thinking traces. The personality examples cover
scenarios where multiple claims interact naturally (e.g. humor + relationship
+ value tension in a single exchange) and low-stakes conversational flow
where no single claim dominates. The claim-derived examples have the most
structured reasoning (explicit tradeoff analysis, emotional self-awareness,
and priority ranking). Tool-use examples reason about when to search
vs. delegate. Background chat examples have lighter thinking (brief situation
assessment).

### Claim-Derived Category Breakdown

The ~305 claim-derived traces break down into 5 sub-categories, each
targeting a gap in the other prompt sources:

| Sub-category | Source claims | Raw traces | After filtering |
|---|---|---|---|
| Value-priority | 127 (116 tradeoff + 11 moral_compass) | ~147 | ~110 |
| Emotional-trigger | 76 emotional claims | ~94 | ~71 |
| Register-shift pairs | 20 communication claims | ~70 | ~53 |
| Theory-of-mind | 36 relationship claims (5 chars) | ~56 | ~42 |
| Identity-grounding | 34 identity + self_model claims | ~40 | ~30 |

Claims with higher support (more scenes confirming the pattern) produce
proportionally more traces: 3 traces for support >= 40, 2 for 10-39, 1
for < 10. Register-shift traces are generated as contrast pairs (same
interlocutor, different situational pressure). Moral_compass claims are
folded into value-priority because they describe ethical resolution
patterns -- the procedural complement to tradeoffs.

Relationship claims are restricted to 5 characters: PK/Paperino (cap 10
claims, 25 traces), Ducklair (all 11 claims, 13 traces -- upscaled as the
foundational creator relationship), Xadhoom (cap 5), Due (cap 5), Lyla
(cap 5). Minor characters are excluded.

See the [Dataset Generation Agent](dataset-generation-agent.md) for the full
pipeline design: context composer, prompt bank, execution loop, trace format,
and quality filtering.
