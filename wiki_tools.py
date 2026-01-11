"""
Wiki search and retrieval tools for character generation.

Provides 4 tools for LLM to access wiki knowledge:
1. search_wiki_content - keyword search across all wiki files
2. list_wiki_categories - list available wiki categories
3. get_wiki_file_summary - get file summary (first pass)
4. read_wiki_file - get full file content (second pass)
"""

import json
import subprocess
from pathlib import Path

# Wiki root directory
WIKI_ROOT = Path(__file__).parent / "output" / "wiki"


def search_wiki_content(keywords: str, max_results: int = 5) -> str:
    """Search wiki files for keywords using ripgrep.

    Fast keyword search across all 479 wiki markdown files.
    Returns file paths with context snippets around matches.

    Args:
        keywords: Keywords to search for in wiki content
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results with file paths and context, or error message
    """
    if not keywords or not keywords.strip():
        return "Error: Please provide keywords to search for"

    try:
        # Use ripgrep with JSON output for structured parsing
        cmd = [
            "rg",
            "--json",  # Structured output
            "--ignore-case",  # Case-insensitive
            "--max-count",
            "2",  # Max 2 matches per file
            "--context",
            "1",  # 1 line context around match
            keywords.strip(),
            str(WIKI_ROOT),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
        )

        # Parse ripgrep JSON output
        matches = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    file_path = Path(data["data"]["path"]["text"])
                    rel_path = file_path.relative_to(WIKI_ROOT)
                    matches.append(
                        {
                            "file": str(rel_path),
                            "line_num": data["data"]["line_number"],
                            "text": data["data"]["lines"]["text"],
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                # Skip malformed lines
                continue

        # Format results for model
        if not matches:
            return f"No wiki entries found for '{keywords}'"

        formatted = [f"Found {len(matches)} matches for '{keywords}':\n"]
        for i, match in enumerate(matches[:max_results], 1):
            formatted.append(
                f"{i}. {match['file']} (line {match['line_num']}): "
                f"{match['text'].strip()}"
            )

        return "\n".join(formatted)

    except subprocess.TimeoutExpired:
        return "Search timed out. Try more specific keywords."
    except FileNotFoundError:
        # ripgrep not available, use Python fallback
        return _python_fallback_search(keywords, max_results)
    except Exception as e:
        return f"Error searching wiki: {str(e)}"


def _python_fallback_search(keywords: str, max_results: int = 5) -> str:
    """Fallback search using Python when ripgrep is unavailable.

    Args:
        keywords: Keywords to search for
        max_results: Maximum number of results

    Returns:
        Search results or error message
    """
    if not WIKI_ROOT.exists():
        return f"Error: Wiki directory not found at {WIKI_ROOT}"

    keywords_lower = keywords.lower()
    matches = []

    try:
        # Search all .md files
        for md_file in WIKI_ROOT.rglob("*.md"):
            if len(matches) >= max_results * 3:  # Stop early if enough matches
                break

            try:
                with open(md_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if keywords_lower in line.lower():
                            rel_path = md_file.relative_to(WIKI_ROOT)
                            matches.append(
                                {
                                    "file": str(rel_path),
                                    "line_num": line_num,
                                    "text": line.strip(),
                                }
                            )
                            if len(matches) >= max_results * 3:
                                break
            except (UnicodeDecodeError, PermissionError):
                # Skip files we can't read
                continue

        if not matches:
            return f"No wiki entries found for '{keywords}'"

        # Format results
        formatted = [f"Found {len(matches)} matches for '{keywords}':\n"]
        for i, match in enumerate(matches[:max_results], 1):
            formatted.append(
                f"{i}. {match['file']} (line {match['line_num']}): {match['text']}"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Error searching wiki: {str(e)}"


def list_wiki_categories() -> str:
    """List all wiki categories and subcategories.

    Returns hierarchical structure of wiki directory:
    - Top-level .md files (characters.md, issues.md, etc.)
    - Subdirectories (fandom/crawl/personaggi/, etc.)

    Returns:
        Formatted list of available wiki categories
    """
    if not WIKI_ROOT.exists():
        return f"Error: Wiki directory not found at {WIKI_ROOT}"

    try:
        categories = []

        # List top-level .md files
        top_level_files = sorted(
            [f.name for f in WIKI_ROOT.glob("*.md")], key=lambda x: x.lower()
        )

        if top_level_files:
            categories.append("**Top-level Wiki Files:**")
            for filename in top_level_files:
                categories.append(f"  - {filename}")

        # List subdirectories
        subdirs = []
        for item in WIKI_ROOT.rglob("*"):
            if item.is_dir():
                rel_path = item.relative_to(WIKI_ROOT)
                # Count .md files in directory
                md_count = len(list(item.glob("*.md")))
                if md_count > 0:
                    subdirs.append((str(rel_path), md_count))

        if subdirs:
            # Sort by path
            subdirs.sort(key=lambda x: x[0])
            categories.append("\n**Wiki Subdirectories:**")
            for subdir, md_count in subdirs:
                categories.append(f"  - {subdir}/ ({md_count} files)")

        if not categories:
            return "No wiki categories found"

        return "\n".join(categories)

    except Exception as e:
        return f"Error listing wiki categories: {str(e)}"


def get_wiki_file_summary(file_path: str) -> str:
    """Extract summary from wiki file (first pass retrieval).

    Returns title, first paragraph, and section headers (~200 tokens max).
    Provides overview without full content for large files.

    Args:
        file_path: Relative path to wiki file (e.g., "uno.md" or "personaggi/xadhoom.md")

    Returns:
        File summary with title, intro, and sections
    """
    # Validate and resolve path
    try:
        wiki_path = (WIKI_ROOT / file_path).resolve()
    except Exception:
        return f"Error: Invalid file path: {file_path}"

    # Security: ensure path is within wiki directory
    try:
        if not wiki_path.is_relative_to(WIKI_ROOT.resolve()):
            return "Error: Invalid file path (outside wiki directory)"
    except Exception:
        return f"Error: Invalid file path: {file_path}"

    if not wiki_path.exists():
        return f"Error: Wiki file not found: {file_path}"

    if not wiki_path.is_file():
        return f"Error: Path is not a file: {file_path}"

    try:
        with open(wiki_path, encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return f"Error: File is empty: {file_path}"

        # Extract components
        title = lines[0].strip() if lines else "Untitled"

        # Find first paragraph (after title, before next header)
        first_para = []
        in_para = False
        for line in lines[1:30]:  # First 30 lines
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith("#"):
                first_para.append(line_stripped)
                in_para = True
            elif in_para and (not line_stripped or line_stripped.startswith("#")):
                break  # End of first paragraph

        # Extract section headers (## headers, not #)
        headers = [
            line.strip()[2:].strip()
            for line in lines[:50]  # First 50 lines
            if line.strip().startswith("##") and not line.strip().startswith("###")
        ]

        # Format summary
        summary_parts = [title]

        if first_para:
            # Limit to 3 sentences or 150 words
            para_text = " ".join(first_para[:3])
            words = para_text.split()
            if len(words) > 150:
                para_text = " ".join(words[:150]) + "..."
            summary_parts.append(f"\n{para_text}")

        if headers:
            summary_parts.append(f"\n\n**Sections:** {', '.join(headers[:8])}")

        return "\n".join(summary_parts)

    except UnicodeDecodeError:
        return f"Error: Cannot decode file (encoding issue): {file_path}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


def read_wiki_file(file_path: str, section: str | None = None) -> str:
    """Read full wiki file content (second pass retrieval).

    Returns complete file content, optionally limited to specific section.
    Truncates very large files (>5000 tokens) with warning.

    Args:
        file_path: Relative path to wiki file
        section: Optional section name to extract (e.g., "Biografia")

    Returns:
        Full file content or specific section
    """
    # Validate and resolve path
    try:
        wiki_path = (WIKI_ROOT / file_path).resolve()
    except Exception:
        return f"Error: Invalid file path: {file_path}"

    # Security: validate path is within wiki directory
    try:
        if not wiki_path.is_relative_to(WIKI_ROOT.resolve()):
            return "Error: Invalid file path (outside wiki directory)"
    except Exception:
        return f"Error: Invalid file path: {file_path}"

    if not wiki_path.exists():
        return f"Error: Wiki file not found: {file_path}"

    if not wiki_path.is_file():
        return f"Error: Path is not a file: {file_path}"

    try:
        with open(wiki_path, encoding="utf-8") as f:
            content = f.read()

        # If section specified, extract only that section
        if section:
            extracted = _extract_section(content, section)
            if not extracted:
                return f"Error: Section '{section}' not found in {file_path}"
            content = extracted

        # Token warning for large files
        approx_tokens = len(content.split()) * 1.3  # Rough estimate
        if approx_tokens > 2000:
            warning = f"[WARNING: Large file ~{int(approx_tokens)} tokens]\n\n"
            # Truncate if too large
            if approx_tokens > 5000:
                content = (
                    content[:10000]
                    + "\n\n[TRUNCATED: File too large, showing first 10000 characters]"
                )
            return warning + content

        return content

    except UnicodeDecodeError:
        return f"Error: Cannot decode file (encoding issue): {file_path}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


def _extract_section(content: str, section_name: str) -> str:
    """Extract specific section from markdown content.

    Args:
        content: Full markdown content
        section_name: Section header to extract (without # symbols)

    Returns:
        Section content, or empty string if not found
    """
    lines = content.split("\n")
    section_lines = []
    in_section = False
    section_level: int | None = None

    for line in lines:
        # Check if this is a header
        if line.startswith("#"):
            header_level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("#").strip()

            if header_text.lower() == section_name.lower():
                # Found target section
                in_section = True
                section_level = header_level
                section_lines.append(line)
            elif (
                in_section
                and section_level is not None
                and header_level <= section_level
            ):
                # Reached next section at same/higher level - stop
                break
        elif in_section:
            section_lines.append(line)

    return "\n".join(section_lines) if section_lines else ""
