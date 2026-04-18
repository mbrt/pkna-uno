"""Tool factory for the eval and dataset generation harness.

Assembles the set of callable tools that the LLM backend can invoke,
based on the tool names requested by each eval prompt.

Includes the in-memory wiki knowledge base (search + read) and the
delegate stub, alongside memory recall/remember wiring.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25L

from pkna.inference.memory import MemoryBank, make_recall, make_remember

TOOL_NAMES = frozenset(
    ["search_knowledge", "read_knowledge", "delegate", "recall", "remember"]
)

WIKI_ROOT = Path(__file__).parent.parent.parent / "results" / "wiki"


# ---------------------------------------------------------------------------
# Wiki knowledge base
# ---------------------------------------------------------------------------


@dataclass
class WikiSegment:
    """A segment of wiki content organized by headers."""

    segment_id: str  # e.g. "characters.md::Personaggi::UNO"
    content: str
    file_path: str  # Relative path (e.g. "characters.md")
    section_path: list[str]  # Hierarchical headers
    level: int  # Header level (1=file, 2=##, 3=###)
    token_count: int  # Approximate (~words * 1.3)

    def get_display_path(self) -> str:
        """Returns formatted path like 'characters.md > Personaggi > UNO'."""
        if self.section_path:
            return f"{self.file_path} > {' > '.join(self.section_path)}"
        return self.file_path

    @staticmethod
    def make_id(file_path: str, section_path: list[str]) -> str:
        """Create unique segment ID from file path and section path."""
        if section_path:
            return f"{file_path}::{'::'.join(section_path)}"
        return file_path


class WikiIndex:
    """In-memory index of all wiki content with BM25 search."""

    def __init__(self):
        self.segments: list[WikiSegment] = []
        self.segments_by_id: dict[str, WikiSegment] = {}
        self.total_tokens: int = 0
        self._bm25: BM25L | None = None

    def load_from_directory(self, wiki_root: Path) -> None:
        """Load and parse all markdown files in wiki directory."""
        if not wiki_root.exists():
            return

        for md_file in sorted(wiki_root.rglob("*.md")):
            self._parse_file(md_file, wiki_root)

    def _parse_file(self, file_path: Path, wiki_root: Path) -> None:
        """Parse a single markdown file into segments."""
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError):
            return

        rel_path = str(file_path.relative_to(wiki_root))
        section_stack: list[tuple[int, str]] = []
        current_content: list[str] = []

        for line in lines:
            header_match = re.match(r"^(#{2,3})\s+(.+)$", line)

            if header_match:
                if current_content:
                    self._create_segment(
                        content="".join(current_content),
                        file_path=rel_path,
                        section_stack=section_stack.copy(),
                    )
                    current_content = []

                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()

                section_stack.append((level, header_text))
            else:
                current_content.append(line)

        if current_content:
            self._create_segment(
                content="".join(current_content),
                file_path=rel_path,
                section_stack=section_stack.copy(),
            )

    def _create_segment(
        self, content: str, file_path: str, section_stack: list[tuple[int, str]]
    ) -> None:
        """Create a WikiSegment from accumulated content."""
        section_path = [header_text for _, header_text in section_stack]
        word_count = len(content.split())
        token_count = int(word_count * 1.3)
        segment_id = WikiSegment.make_id(file_path, section_path)
        level = section_stack[-1][0] if section_stack else 1

        segment = WikiSegment(
            segment_id=segment_id,
            content=content.strip(),
            file_path=file_path,
            section_path=section_path,
            level=level,
            token_count=token_count,
        )

        self.segments.append(segment)
        self.segments_by_id[segment_id] = segment
        self.total_tokens += token_count

    def _ensure_bm25(self) -> None:
        """Build BM25 index if not already built."""
        if self._bm25 is not None:
            return
        if not self.segments:
            return
        corpus = [
            (" ".join(seg.section_path) + " " + seg.content).lower().split()
            for seg in self.segments
        ]
        self._bm25 = BM25L(corpus)

    def search(self, keywords: str, max_results: int = 5) -> list[WikiSegment]:
        """BM25 search over segments, return top matches."""
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        tokenized = keywords.lower().split()
        if not tokenized:
            return []
        scores: np.ndarray = self._bm25.get_scores(tokenized)
        ranked_indices = np.argsort(scores)[::-1]
        results: list[WikiSegment] = []
        for idx in ranked_indices[:max_results]:
            if scores[idx] > 0:
                results.append(self.segments[idx])
        return results

    def get_segment(self, segment_id: str) -> WikiSegment | None:
        """Retrieve full segment by ID."""
        return self.segments_by_id.get(segment_id)


_wiki_index: WikiIndex | None = None


def get_wiki_index() -> WikiIndex:
    """Get or create the global wiki index."""
    global _wiki_index
    if _wiki_index is None:
        _wiki_index = WikiIndex()
        _wiki_index.load_from_directory(WIKI_ROOT)
    return _wiki_index


def search_knowledge(keywords: str, max_results: int = 5) -> str:
    """Search the knowledge base for keywords and return relevant segments with snippets.

    Fast in-memory keyword search across all knowledge base content.
    Returns segment IDs and short snippets for browsing results.
    Use read_knowledge() to get full content of a specific segment.

    Args:
        keywords: Keywords to search for
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with segment IDs, paths, and content snippets
    """
    if not keywords or not keywords.strip():
        return "Error: Please provide keywords to search for"

    index = get_wiki_index()
    segments = index.search(keywords.strip(), max_results)

    if not segments:
        return f"No entries found for '{keywords}'"

    formatted = [f"Found {len(segments)} matches for '{keywords}':\n"]
    for i, segment in enumerate(segments, 1):
        snippet = segment.content[:200].strip()
        if len(segment.content) > 200:
            snippet += "..."

        formatted.append(f"{i}. [{segment.segment_id}]")
        formatted.append(f"   Path: {segment.get_display_path()}")
        formatted.append(f"   Snippet: {snippet}")
        formatted.append(f"   (~{segment.token_count} tokens)")
        formatted.append("")

    return "\n".join(formatted)


def read_knowledge(segment_id: str) -> str:
    """Read full content of a specific knowledge base segment.

    Retrieves complete text of a segment identified by its ID.
    Use search_knowledge() first to find relevant segments and their IDs.

    Args:
        segment_id: Unique segment identifier (from search results)

    Returns:
        Full segment content with hierarchical path, or error message
    """
    if not segment_id or not segment_id.strip():
        return "Error: Please provide a segment ID"

    index = get_wiki_index()
    segment = index.get_segment(segment_id.strip())

    if not segment:
        return f"Error: Segment '{segment_id}' not found"

    formatted = [
        f"Segment: {segment.get_display_path()}",
        f"Size: ~{segment.token_count} tokens",
        "",
        segment.content.strip(),
    ]

    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Delegate stub
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


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
