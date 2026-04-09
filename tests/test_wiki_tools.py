"""Unit tests for in-memory wiki tools."""

from pkna.wiki_tools import (
    WIKI_ROOT,
    WikiIndex,
    WikiSegment,
    get_wiki_index,
    read_wiki_segment,
    search_wiki,
)


class TestWikiIndex:
    """Tests for WikiIndex class."""

    def test_load_from_directory(self):
        """Test loading wiki from directory."""
        index = WikiIndex()
        index.load_from_directory(WIKI_ROOT)
        assert len(index.segments) > 0
        assert index.total_tokens > 0

    def test_segments_have_metadata(self):
        """Test that segments have required metadata."""
        index = get_wiki_index()
        for segment in index.segments[:10]:
            assert segment.file_path
            assert isinstance(segment.section_path, list)
            assert segment.level >= 1
            assert segment.token_count >= 0  # Some segments may be empty (headers only)

    def test_hierarchical_structure(self):
        """Test that section_path maintains hierarchy."""
        index = get_wiki_index()
        # Find a segment with multiple levels
        deep_segments = [s for s in index.segments if len(s.section_path) >= 2]
        assert len(deep_segments) > 0


class TestSearch:
    """Tests for search functionality."""

    def test_search_finds_results(self):
        """Test that search returns results for known content."""
        result = search_wiki("Everett Ducklair")
        assert "Found" in result or "matches" in result.lower()
        assert "ducklair" in result.lower()

    def test_search_no_results(self):
        """Test that search handles no results gracefully."""
        result = search_wiki("ZZZNonexistentXYZ123")
        assert "No wiki entries found" in result

    def test_search_empty_keywords(self):
        """Test that empty keywords are handled."""
        result = search_wiki("")
        assert "Error" in result

    def test_search_max_results_limit(self):
        """Test that max_results parameter limits output."""
        index = get_wiki_index()
        segments = index.search("Uno", max_results=2)
        assert len(segments) <= 2

    def test_search_returns_display_paths(self):
        """Test that results include hierarchical paths."""
        result = search_wiki("Uno")
        # Should contain file paths with > separators
        assert ".md >" in result or ">" in result

    def test_search_returns_segment_ids(self):
        """Test that search results include segment IDs."""
        result = search_wiki("Uno")
        # Should contain segment IDs in brackets
        assert "[" in result and "::" in result


class TestReadSegment:
    """Tests for read_wiki_segment functionality."""

    def test_read_segment_success(self):
        """Test reading a valid segment."""
        index = get_wiki_index()

        # Get first segment ID from index
        if index.segments:
            segment_id = index.segments[0].segment_id
            result = read_wiki_segment(segment_id)
            assert "Error" not in result
            assert "Segment:" in result

    def test_read_segment_not_found(self):
        """Test reading nonexistent segment."""
        result = read_wiki_segment("nonexistent::segment::id")
        assert "Error" in result or "not found" in result.lower()

    def test_read_segment_empty_id(self):
        """Test that empty segment ID is handled."""
        result = read_wiki_segment("")
        assert "Error" in result


class TestWikiSegment:
    """Tests for WikiSegment class."""

    def test_get_display_path(self):
        """Test display path formatting."""
        segment = WikiSegment(
            segment_id="characters.md::Personaggi::UNO::Biografia",
            content="Test content",
            file_path="characters.md",
            section_path=["Personaggi", "UNO", "Biografia"],
            level=3,
            token_count=10,
        )
        display = segment.get_display_path()
        assert "characters.md" in display
        assert "Personaggi" in display
        assert ">" in display

    def test_make_id(self):
        """Test segment ID generation."""
        segment_id = WikiSegment.make_id("characters.md", ["Personaggi", "UNO"])
        assert segment_id == "characters.md::Personaggi::UNO"

    def test_make_id_no_sections(self):
        """Test segment ID generation with no sections."""
        segment_id = WikiSegment.make_id("characters.md", [])
        assert segment_id == "characters.md"


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
