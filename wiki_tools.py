"""
In-memory wiki search and retrieval tools for character generation.

Loads all wiki content into memory on startup, organized hierarchically
by markdown headers. Provides 2 tools for LLM to access wiki knowledge:
1. search_wiki - keyword search returning short snippets with segment IDs
2. read_wiki_segment - retrieve full content of a specific segment by ID
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Wiki root directory
WIKI_ROOT = Path(__file__).parent / "output" / "wiki"


@dataclass
class WikiSegment:
    """A segment of wiki content organized by headers."""

    segment_id: str  # Unique identifier (e.g., "characters.md::Personaggi::UNO")
    content: str  # Full text of segment
    file_path: str  # Relative path (e.g., "characters.md")
    section_path: list[str]  # Hierarchical headers (e.g., ["Personaggi", "UNO"])
    level: int  # Header level (1=file, 2=##, 3=###)
    token_count: int  # Approximate token count (~words * 1.3)

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
    """In-memory index of all wiki content."""

    def __init__(self):
        self.segments: list[WikiSegment] = []
        self.segments_by_id: dict[str, WikiSegment] = {}
        self.total_tokens: int = 0

    def load_from_directory(self, wiki_root: Path) -> None:
        """Load and parse all markdown files in wiki directory."""
        if not wiki_root.exists():
            return

        # Load all .md files
        for md_file in sorted(wiki_root.rglob("*.md")):
            self._parse_file(md_file, wiki_root)

    def _parse_file(self, file_path: Path, wiki_root: Path) -> None:
        """Parse a single markdown file into segments."""
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError):
            # Skip files we can't read
            return

        rel_path = str(file_path.relative_to(wiki_root))
        section_stack: list[tuple[int, str]] = []  # (level, header_text)
        current_content: list[str] = []

        for line in lines:
            # Check if line is a header (## or ###)
            header_match = re.match(r"^(#{2,3})\s+(.+)$", line)

            if header_match:
                # Save previous segment if exists
                if current_content:
                    self._create_segment(
                        content="".join(current_content),
                        file_path=rel_path,
                        section_stack=section_stack.copy(),
                    )
                    current_content = []

                # Update section stack
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Pop stack to current level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()

                # Push new section
                section_stack.append((level, header_text))
            else:
                # Accumulate content
                current_content.append(line)

        # Save final segment
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
        # Extract section path (just the header texts)
        section_path = [header_text for _, header_text in section_stack]

        # Calculate approximate token count
        word_count = len(content.split())
        token_count = int(word_count * 1.3)

        # Create segment ID
        segment_id = WikiSegment.make_id(file_path, section_path)

        # Determine level (deepest header in stack, or 1 if no headers)
        level = section_stack[-1][0] if section_stack else 1

        # Create and store segment
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

    def search(self, keywords: str, max_results: int = 5) -> list[WikiSegment]:
        """Search segments for keywords, return top matches."""
        keywords_lower = keywords.lower()
        keyword_list = keywords_lower.split()

        scored_segments = []
        for segment in self.segments:
            # Calculate relevance score
            content_lower = segment.content.lower()
            section_path_lower = " ".join(segment.section_path).lower()

            # Score: keyword frequency + title matches (weighted 5x higher)
            score = 0
            for kw in keyword_list:
                score += content_lower.count(kw)
                score += section_path_lower.count(kw) * 5

            if score > 0:
                scored_segments.append((score, segment))

        # Sort by score descending, return top N
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        return [seg for _, seg in scored_segments[:max_results]]

    def get_segment(self, segment_id: str) -> WikiSegment | None:
        """Retrieve full segment by ID."""
        return self.segments_by_id.get(segment_id)


# Global wiki index (loaded lazily on first use)
_wiki_index: WikiIndex | None = None


def get_wiki_index() -> WikiIndex:
    """Get or create the global wiki index."""
    global _wiki_index
    if _wiki_index is None:
        _wiki_index = WikiIndex()
        _wiki_index.load_from_directory(WIKI_ROOT)
    return _wiki_index


# Tool functions for LLM


def search_wiki(keywords: str, max_results: int = 5) -> str:
    """Search wiki for keywords and return relevant segments with snippets.

    Fast in-memory keyword search across all wiki content.
    Returns segment IDs and short snippets for browsing results.
    Use read_wiki_segment() to get full content of a specific segment.

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
        return f"No wiki entries found for '{keywords}'"

    # Format results with snippets and IDs
    formatted = [f"Found {len(segments)} matches for '{keywords}':\n"]
    for i, segment in enumerate(segments, 1):
        # Show ~200 char snippet
        snippet = segment.content[:200].strip()
        if len(segment.content) > 200:
            snippet += "..."

        formatted.append(f"{i}. [{segment.segment_id}]")
        formatted.append(f"   Path: {segment.get_display_path()}")
        formatted.append(f"   Snippet: {snippet}")
        formatted.append(f"   (~{segment.token_count} tokens)")
        formatted.append("")

    return "\n".join(formatted)


def read_wiki_segment(segment_id: str) -> str:
    """Read full content of a specific wiki segment.

    Retrieves complete text of a segment identified by its ID.
    Use search_wiki() first to find relevant segments and their IDs.

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

    # Format with path and full content
    formatted = [
        f"Segment: {segment.get_display_path()}",
        f"Size: ~{segment.token_count} tokens",
        "",
        segment.content.strip(),
    ]

    return "\n".join(formatted)
