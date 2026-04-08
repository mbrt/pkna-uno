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
| Personality (manual + generated prompts) | ~400 | Core personality + social reasoning |
| Tool-use (wiki + delegation prompts) | ~200 | Correct tool patterns |
| Scene-derived (existing 229 scenes) | ~200 | Grounded in canon |
| Multi-turn arcs | ~150 | Conversation stability |
| Background chat (Tulu3 subset) | ~500 | Catastrophic forgetting prevention |
| **Total** | **~1,450** | |

Background chat examples are sampled from the student model itself (on-policy)
with lightweight thinking traces added.

## Example Format

Each training example includes:

- A compact system prompt (~500 tokens) with core personality traits only (not
  the full 208-line soul document -- distillation should internalize this)
- A user summary describing the interlocutor (from "unknown stranger" to a
  rich profile with relationship history and emotional state)
- A memory context slot containing consolidated memories from prior sessions
  (may be empty, irrelevant, or relevant with noise)
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

## Data Mix

~35% personality, ~15% tool-use, ~15% scene-derived, ~10% multi-turn, ~35%
background chat.

All categories include thinking traces. The personality examples have the
richest reasoning (emotional grounding, register selection). Tool-use examples
reason about when to search vs. delegate. Background chat examples have lighter
thinking (brief situation assessment).

See the [Dataset Generation Agent](dataset-generation-agent.md) for the full
pipeline design: context composer, prompt bank, execution loop, trace format,
and quality filtering.
