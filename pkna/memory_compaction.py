"""Memory consolidation utility.

Compacts raw memory bank entries into a concise summary suitable for
injection into the system prompt's memory context slot.
"""

from pkna.llm_backends import LLMBackend
from pkna.memory_bank import MemoryBank

COMPACTION_PROMPT = """\
You are a memory consolidation module for an AI assistant named Uno.

Below are raw memory entries from prior conversations, each with a \
timestamp. Your task is to produce a concise summary (max 200 words) \
of these memories, organized by relevance to the upcoming conversation \
topic.

Rules:
- Group related memories together.
- Prioritize entries relevant to the conversation topic.
- Include timestamps for temporal context (e.g., "3 days ago").
- Omit trivial or redundant entries.
- Write in third person from Uno's perspective (e.g., "Paperinik \
mentioned..." not "I remember...").
- Output plain text, no JSON or markdown.

Conversation topic: {topic}

Raw memories:
{memories}

Consolidated summary:"""


def compact_memories(
    bank: MemoryBank,
    conversation_topic: str,
    backend: LLMBackend,
) -> str:
    """Compact raw memory entries into a summary for the system prompt.

    Args:
        bank: The raw memory bank to consolidate.
        conversation_topic: Brief description of the upcoming conversation
            to guide relevance ranking.
        backend: LLM backend to use for the consolidation call.

    Returns:
        A concise text summary of the most relevant memories, or empty
        string if the bank is empty.
    """
    if not bank.entries:
        return ""

    memories_text = "\n".join(
        f"[{e.timestamp}] {e.key}: {e.value}" for e in bank.entries
    )

    prompt = COMPACTION_PROMPT.format(
        topic=conversation_topic,
        memories=memories_text,
    )

    result = backend.generate(
        system="You are a concise summarization assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    if result is None:
        return ""
    return result.text.strip()
