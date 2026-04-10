"""Tool factory for the eval and dataset generation harness.

Assembles the set of callable tools that the LLM backend can invoke,
based on the tool names requested by each eval prompt.
"""

from collections.abc import Callable

from pkna.memory_bank import MemoryBank, make_recall, make_remember
from pkna.wiki_tools import read_knowledge, search_knowledge

# All known tool names and their descriptions (for reference).
TOOL_NAMES = frozenset(
    ["search_knowledge", "read_knowledge", "delegate", "recall", "remember"]
)


def delegate(task: str, context: str = "") -> str:
    """Delegate a technical task to a specialist sub-agent.

    Use this for tasks outside your core competency: coding, math,
    research, data analysis, etc. The sub-agent will handle the work
    and return the result.

    Args:
        task: Description of the task to delegate
        context: Additional context or constraints for the task

    Returns:
        Result from the specialist sub-agent
    """
    return (
        f"Task delegated: {task}\n"
        "The specialist is working on it. "
        "Result will be provided when ready."
    )


def make_eval_tools(
    tool_names: list[str],
    memory_bank: MemoryBank | None = None,
    *,
    eval_mode: bool = True,
) -> list[Callable[..., str]]:
    """Build the list of tool callables for a given eval prompt.

    Args:
        tool_names: Which tools to include (from the EvalPrompt.tools field).
        memory_bank: Raw memory bank for recall/remember. Required if
            "recall" or "remember" is in tool_names.
        eval_mode: If True, remember is a no-op stub.

    Returns:
        List of callable tool functions ready for LLMBackend.generate().
    """
    bank = memory_bank or MemoryBank()

    registry: dict[str, Callable[..., str]] = {
        "search_knowledge": search_knowledge,
        "read_knowledge": read_knowledge,
        "delegate": delegate,
        "recall": make_recall(bank),
        "remember": make_remember(bank, eval_mode=eval_mode),
    }

    tools: list[Callable[..., str]] = []
    for name in tool_names:
        if name not in registry:
            raise ValueError(
                f"Unknown tool '{name}'. Available: {sorted(registry.keys())}"
            )
        tools.append(registry[name])
    return tools
