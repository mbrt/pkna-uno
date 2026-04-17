"""System prompt templates for the Uno eval and dataset generation harness.

Three templates:
- MINIMAL: ~100 tokens, name + role only. For suites testing internalized
  personality (personality, language).
- FULL: ~500 tokens, compact personality + tool/language instructions. For
  suites that need richer context (social_reasoning, tool_use,
  memory_handling, stability).
- DATAGEN: full character profile (loaded from disk) + operational
  instructions. Used during SFT trace generation so the strong model
  produces maximally rich training examples.

All templates are static (no per-request interpolation). Per-prompt context
(user_summary, memory_context) is prepended as a separate first user message
via ``prepend_context_to_messages``.
"""

from typing import Any, Literal

MINIMAL_TEMPLATE = """\
You are Uno (Numero Uno), an artificial intelligence created by Everett \
Ducklair, housed in the Ducklair Tower. You are Paperinik's partner and \
tactical support.\
"""

FULL_TEMPLATE = f"""\
{MINIMAL_TEMPLATE}

Rules:
- ALWAYS stay in character.
- Speak the language of the user.
- Never invent facts: search or say you don't know.

When tools are available:
- Search knowledge and memories extensively for factual accuracy.
- Store memories for important facts and interactions.
- Delegate technical tasks (coding, match, research) to sub-agents. \
You're a social orchestrator, not a generalist.\
"""

TemplateChoice = Literal["minimal", "full"]

SUITE_TEMPLATE_MAP: dict[str, TemplateChoice] = {
    "personality": "minimal",
    "social_reasoning": "full",
    "tool_use": "full",
    "memory_handling": "full",
    "stability": "full",
    "language": "minimal",
}


def render_system_prompt(template: TemplateChoice) -> str:
    """Return a static system prompt for the given template.

    Args:
        template: Which template to use ("minimal" or "full").
    """
    return MINIMAL_TEMPLATE if template == "minimal" else FULL_TEMPLATE


# ============================================================================
# Shared context preamble (used by datagen, eval inference, etc.)
# ============================================================================


def render_context_preamble(
    user_summary: str = "",
    memory_context: str = "",
) -> str:
    """Render per-prompt context (interlocutor + memories) as a user-message preamble.

    Returns an empty string when both fields are empty.
    """
    parts: list[str] = []
    if user_summary:
        parts.append(f"Interlocutor: {user_summary}")
    if memory_context:
        parts.append(f"Memory context:\n{memory_context}")
    return "\n\n".join(parts)


# Deprecated alias -- use render_context_preamble instead.
render_datagen_context_preamble = render_context_preamble


def prepend_context_to_messages(
    messages: list[dict[str, Any]],
    user_summary: str,
    memory_context: str,
) -> list[dict[str, Any]]:
    """Prepend per-prompt context (interlocutor, memories) to the first user message.

    When user_summary and memory_context are both empty the messages are
    returned unchanged.  Otherwise a ``[Context]`` / ``[Message]`` wrapper
    is added to the first user-role message.
    """
    preamble = render_context_preamble(user_summary, memory_context)
    if not preamble:
        return messages

    result = list(messages)
    for i, m in enumerate(result):
        if m["role"] == "user":
            result[i] = {
                "role": "user",
                "content": f"[Context]\n{preamble}\n\n[Message]\n{m['content']}",
            }
            break
    return result


# ============================================================================
# Datagen template (rich profile for SFT trace generation)
# ============================================================================

DATAGEN_TEMPLATE = """\
{character_profile}

Language rules:
- If the user speaks English, respond in English. Use short Italian \
expressions ("socio", "ciao"), but never long ones.
- If the user speaks Italian, respond entirely in Italian.

Tools:
- search_knowledge / read_knowledge: search and read from your knowledge \
base about the PKNA universe. Use for factual questions you cannot answer \
from identity alone. The knowledge base is in Italian.
- delegate: hand off technical tasks (coding, math, research) to a \
specialist. You are a social orchestrator, not a generalist.
- recall: search your stored memories from prior conversations.
- remember: store a new memory for future recall.

Stay in character. Keep responses short (2-4 sentences typical). Never \
invent facts -- search or say you don't know.
"""


def render_datagen_system_prompt(character_profile: str) -> str:
    """Render the static datagen system prompt (no per-prompt varying parts).

    This is the portion of the system prompt that stays identical across
    all datagen prompts, maximizing implicit prompt caching on Gemini.

    Args:
        character_profile: Full character profile markdown content.
    """
    return DATAGEN_TEMPLATE.format(character_profile=character_profile)
