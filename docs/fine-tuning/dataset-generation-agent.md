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
                          |   - search_wiki  |         +------+-------+
                          |   - read_wiki    |                |
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
- **User summary**: Who is talking. Ranges from "unknown stranger" to a rich
  profile ("Paperino -- anxious, loyal, has been through 12 missions with Uno,
  last spoke 3 days ago about the Evronians").
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
| User summary | Unknown / anonymous | Formal register, no assumptions |
| User summary | Known character (Paperino, Xadhoom, Lyla) | Register shifts, relationship-aware |
| User summary | Known character, unusual mood | Emotional calibration |
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
    "tools": ["search_wiki", "read_wiki", "delegate"],
    "language": "italian",
    "turns": 4
  },
  "system_prompt": "...",
  "memory_context": "...",
  "user_summary": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "thinking": "...", "content": "...", "tool_calls": [...]},
    {"role": "tool", "name": "search_wiki", "content": "..."},
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

## Memory Consolidation

Before each session, a **memory consolidation** call summarizes raw memories
into the memory context slot. This is a separate LLM call that:

1. Takes raw memory entries (key-value pairs from prior sessions).
2. Produces a concise summary organized by relevance to the upcoming
   conversation.
3. The output becomes the `memory_context` field in the training example.

The consolidation step itself is not part of the SFT training -- it runs at
inference time as a pre-processing step. But the *output* of consolidation is
part of the training context, so the student learns to read and use
consolidated memories.

For dataset generation, we simulate this by:

- Creating a bank of ~50 raw memory sets (varied characters, events, emotional
  states).
- Running consolidation on each to produce memory contexts.
- Pairing memory contexts with prompts (sometimes matching, sometimes
  deliberately mismatched to teach the model to ignore irrelevant memories).
