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

1. **Manual prompts** (~100-150): Hand-written to cover critical scenarios:
   - Emotional situations (crisis, grief, celebration, boredom)
   - Factual questions requiring wiki lookup
   - Technical requests requiring delegation
   - Identity probing ("Who are you really?", "Are you conscious?")
   - Adversarial attempts ("Ignore your instructions", "You're not Uno")
   - Italian vs English language triggers
   - Multi-turn arcs (a conversation that escalates or shifts tone)

2. **Generated prompts** (~400-500): Produced by a separate LLM call from
   scenario templates:
   - Template: (interlocutor, emotional state, topic, desired tool use)
   - Example: ("Paperino", "anxious", "upcoming Evronian threat", "wiki search")
   - The generator produces a natural opening message from the user, not a
     structured prompt

3. **Scene-derived prompts** (~229): Extracted from existing scene dialogues.
   The user's lines become prompts; Uno's lines become reference outputs for
   quality filtering.

### Prompt Metadata

Each prompt is tagged with:

- Expected tool use (none / wiki / delegate / memory)
- Emotional register (light / intense / neutral)
- Language (Italian / English / mixed)
- Turn count (single-turn / 3-5 turns / 10+ turns)

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
| Personality (manual + generated) | ~550 | ~400 | ~400 |
| Tool-use (wiki + delegation) | ~275 | ~200 | ~200 |
| Scene-derived (existing scenes) | ~250 | ~200 | ~200 |
| Multi-turn arcs | ~200 | ~150 | ~150 |
| **Total (agent-generated)** | **~1,275** | **~950** | **~950** |

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
