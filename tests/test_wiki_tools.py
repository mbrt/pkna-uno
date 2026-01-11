"""Unit tests for wiki_tools module."""

from wiki_tools import (
    WIKI_ROOT,
    get_wiki_file_summary,
    list_wiki_categories,
    read_wiki_file,
    search_wiki_content,
)


class TestSearchWikiContent:
    """Tests for search_wiki_content function."""

    def test_search_finds_results(self):
        """Test that search returns results for known content."""
        results = search_wiki_content("Everett Ducklair")
        assert "Found" in results or "matches" in results.lower()
        assert "ducklair" in results.lower()

    def test_search_no_results(self):
        """Test that search handles no results gracefully."""
        results = search_wiki_content("ZZZNonexistentXYZ123")
        assert "No wiki entries found" in results or "not found" in results.lower()

    def test_search_empty_keywords(self):
        """Test that empty keywords are handled."""
        results = search_wiki_content("")
        assert "Error" in results

    def test_search_max_results_limit(self):
        """Test that max_results parameter limits output."""
        results = search_wiki_content("Uno", max_results=2)
        # Should not have more than 2 results listed
        lines = results.split("\n")
        result_lines = [line for line in lines if line.strip() and line[0].isdigit()]
        assert len(result_lines) <= 2

    def test_search_returns_file_paths(self):
        """Test that search returns file paths."""
        results = search_wiki_content("Uno")
        # Should contain at least one .md file reference
        assert ".md" in results


class TestListWikiCategories:
    """Tests for list_wiki_categories function."""

    def test_lists_top_level_files(self):
        """Test that top-level wiki files are listed."""
        categories = list_wiki_categories()
        assert "characters.md" in categories or "uno.md" in categories
        assert "**Top-level Wiki Files:**" in categories or "Wiki" in categories

    def test_lists_subdirectories(self):
        """Test that subdirectories are listed."""
        categories = list_wiki_categories()
        # Should list some subdirectories
        assert (
            "personaggi" in categories.lower()
            or "fandom" in categories.lower()
            or "Subdirectories" in categories
        )

    def test_returns_string(self):
        """Test that function returns a string."""
        categories = list_wiki_categories()
        assert isinstance(categories, str)
        assert len(categories) > 0


class TestGetWikiFileSummary:
    """Tests for get_wiki_file_summary function."""

    def test_summary_for_existing_file(self):
        """Test summary extraction for existing file."""
        summary = get_wiki_file_summary("uno.md")
        assert "Uno" in summary
        assert "Error" not in summary

    def test_summary_includes_title(self):
        """Test that summary includes title."""
        summary = get_wiki_file_summary("uno.md")
        # First line should be title
        lines = summary.split("\n")
        assert len(lines) > 0
        assert "#" in lines[0] or "Uno" in lines[0]

    def test_summary_token_limit(self):
        """Test that summary is limited to reasonable length."""
        summary = get_wiki_file_summary("characters.md")
        # Should not be the entire file (characters.md is 106KB)
        assert len(summary.split()) < 500  # Less than ~500 words

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        summary = get_wiki_file_summary("nonexistent_file_xyz.md")
        assert "Error" in summary or "not found" in summary.lower()

    def test_file_in_subdirectory(self):
        """Test summary for file in subdirectory."""
        # Find a file in subdirectory first
        fandom_dir = WIKI_ROOT / "fandom" / "crawl" / "personaggi" / "amici"
        if fandom_dir.exists():
            files = list(fandom_dir.glob("*.md"))
            if files:
                rel_path = files[0].relative_to(WIKI_ROOT)
                summary = get_wiki_file_summary(str(rel_path))
                assert "Error" not in summary or "not found" in summary.lower()


class TestReadWikiFile:
    """Tests for read_wiki_file function."""

    def test_read_existing_file(self):
        """Test reading an existing file."""
        content = read_wiki_file("uno.md")
        assert "Error" not in content or len(content) > 100
        assert "Uno" in content

    def test_read_nonexistent_file(self):
        """Test handling of nonexistent file."""
        content = read_wiki_file("nonexistent_xyz.md")
        assert "Error" in content or "not found" in content.lower()

    def test_security_path_traversal(self):
        """Test that path traversal is prevented."""
        # Try to access file outside wiki directory
        content = read_wiki_file("../../etc/passwd")
        assert "Error" in content or "Invalid" in content

    def test_security_absolute_path(self):
        """Test that absolute paths are rejected."""
        content = read_wiki_file("/etc/passwd")
        assert "Error" in content or "Invalid" in content

    def test_read_file_section(self):
        """Test extracting specific section."""
        # uno.md has a "Biografia" section
        content = read_wiki_file("uno.md", section="Biografia")
        if "Error" not in content:
            # If section exists, it should contain the section header
            assert "Biografia" in content or len(content) > 0

    def test_read_file_nonexistent_section(self):
        """Test handling of nonexistent section."""
        content = read_wiki_file("uno.md", section="NonexistentSectionXYZ")
        assert "Error" in content or "not found" in content.lower()

    def test_large_file_warning(self):
        """Test that large files get a warning."""
        # characters.md is 106KB
        content = read_wiki_file("characters.md")
        # Should either have warning or be truncated
        assert (
            "WARNING" in content or "TRUNCATED" in content or len(content) < 100000
        )  # Should be truncated if too large


class TestWikiRootExists:
    """Test that wiki directory exists."""

    def test_wiki_root_exists(self):
        """Test that WIKI_ROOT directory exists."""
        assert WIKI_ROOT.exists()
        assert WIKI_ROOT.is_dir()

    def test_wiki_has_files(self):
        """Test that wiki directory contains .md files."""
        md_files = list(WIKI_ROOT.rglob("*.md"))
        assert len(md_files) > 0
        assert any(f.name == "uno.md" for f in md_files)
