"""Unit tests for LineBasedDocument class."""

from build_agentic_character_profile import LineBasedDocument, extract_section_headers


class TestLineBasedDocumentInit:
    """Tests for LineBasedDocument initialization."""

    def test_init_simple(self):
        """Test initialization with simple content."""
        doc = LineBasedDocument("line1\nline2\nline3")
        assert doc.get_content() == "line1\nline2\nline3"

    def test_init_empty(self):
        """Test initialization with empty content."""
        doc = LineBasedDocument("")
        assert doc.get_content() == ""

    def test_init_single_line(self):
        """Test initialization with single line."""
        doc = LineBasedDocument("single line")
        assert doc.get_content() == "single line"


class TestGetLines:
    """Tests for get_lines method."""

    def test_get_all_lines(self):
        """Test getting all lines with numbering."""
        doc = LineBasedDocument("line1\nline2\nline3")
        result = doc.get_lines()
        assert result == "1: line1\n2: line2\n3: line3"

    def test_get_lines_with_offset(self):
        """Test getting lines starting from offset."""
        doc = LineBasedDocument("line1\nline2\nline3\nline4")
        result = doc.get_lines(offset=2)
        assert result == "2: line2\n3: line3\n4: line4"

    def test_get_lines_with_limit(self):
        """Test getting limited number of lines."""
        doc = LineBasedDocument("line1\nline2\nline3\nline4")
        result = doc.get_lines(limit=2)
        assert result == "1: line1\n2: line2"

    def test_get_lines_with_offset_and_limit(self):
        """Test getting lines with both offset and limit."""
        doc = LineBasedDocument("line1\nline2\nline3\nline4\nline5")
        result = doc.get_lines(offset=2, limit=2)
        assert result == "2: line2\n3: line3"

    def test_get_lines_offset_beyond_end(self):
        """Test getting lines with offset beyond document end."""
        doc = LineBasedDocument("line1\nline2")
        result = doc.get_lines(offset=10)
        assert result == ""

    def test_get_lines_offset_zero_treated_as_one(self):
        """Test that offset < 1 is treated as 1."""
        doc = LineBasedDocument("line1\nline2")
        result = doc.get_lines(offset=0)
        assert result == "1: line1\n2: line2"


class TestEdit:
    """Tests for edit method."""

    def test_edit_success(self):
        """Test successful edit with unique match."""
        doc = LineBasedDocument("Hello world\nGoodbye world")
        success, message = doc.edit("Hello world", "Hi there")
        assert success is True
        assert "Edit applied" in message
        assert doc.get_content() == "Hi there\nGoodbye world"

    def test_edit_not_found(self):
        """Test edit with text not found."""
        doc = LineBasedDocument("Hello world")
        success, message = doc.edit("nonexistent", "replacement")
        assert success is False
        assert "not found" in message
        assert doc.get_content() == "Hello world"  # unchanged

    def test_edit_multiple_occurrences(self):
        """Test edit with multiple occurrences."""
        doc = LineBasedDocument("hello hello hello")
        success, message = doc.edit("hello", "hi")
        assert success is False
        assert "Multiple occurrences" in message
        assert "3 matches" in message
        assert doc.get_content() == "hello hello hello"  # unchanged

    def test_edit_multiline_old_text(self):
        """Test edit with multiline old_text."""
        doc = LineBasedDocument("line1\nline2\nline3")
        success, _ = doc.edit("line1\nline2", "replaced")
        assert success is True
        assert doc.get_content() == "replaced\nline3"

    def test_edit_multiline_new_text(self):
        """Test edit with multiline new_text."""
        doc = LineBasedDocument("line1\nline2")
        success, _ = doc.edit("line1", "new1\nnew2\nnew3")
        assert success is True
        assert doc.get_content() == "new1\nnew2\nnew3\nline2"

    def test_edit_empty_replacement(self):
        """Test edit that removes content (empty new_text)."""
        doc = LineBasedDocument("keep this\nremove me\nkeep this too")
        success, _ = doc.edit("\nremove me", "")
        assert success is True
        assert doc.get_content() == "keep this\nkeep this too"

    def test_edit_partial_line_match(self):
        """Test edit that matches part of a line."""
        doc = LineBasedDocument("The quick brown fox")
        success, _ = doc.edit("quick brown", "slow white")
        assert success is True
        assert doc.get_content() == "The slow white fox"

    def test_edit_message_contains_stats(self):
        """Test that success message contains line and token counts."""
        doc = LineBasedDocument("line1\nline2")
        success, message = doc.edit("line1", "replaced")
        assert success is True
        assert "lines" in message
        assert "tokens" in message


class TestEditWithSpecialCases:
    """Tests for edge cases in edit method."""

    def test_edit_empty_document(self):
        """Test edit on empty document."""
        doc = LineBasedDocument("")
        success, message = doc.edit("anything", "replacement")
        assert success is False
        assert "not found" in message

    def test_edit_with_newlines_in_text(self):
        """Test that newlines in old_text are matched correctly."""
        content = "Section A\n\nContent here\n\nSection B"
        doc = LineBasedDocument(content)
        success, _ = doc.edit("Section A\n\nContent here", "New Section")
        assert success is True
        assert doc.get_content() == "New Section\n\nSection B"

    def test_edit_makes_unique_after_partial_edit(self):
        """Test that previously duplicate text becomes unique after edits."""
        doc = LineBasedDocument("duplicate\nduplicate")

        # First attempt fails - multiple matches
        success, _ = doc.edit("duplicate", "changed")
        assert success is False

        # Use more specific match
        success, _ = doc.edit("duplicate\n", "changed\n")
        assert success is True
        assert doc.get_content() == "changed\nduplicate"

        # Now the remaining "duplicate" is unique
        success, _ = doc.edit("duplicate", "also changed")
        assert success is True
        assert doc.get_content() == "changed\nalso changed"


class TestSave:
    """Tests for save method."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates a file with correct content."""
        doc = LineBasedDocument("test content\nline 2")
        output_path = tmp_path / "test.md"
        doc.save(output_path)

        assert output_path.exists()
        assert output_path.read_text() == "test content\nline 2"

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save creates parent directories if needed."""
        doc = LineBasedDocument("content")
        output_path = tmp_path / "nested" / "dirs" / "test.md"
        doc.save(output_path)

        assert output_path.exists()
        assert output_path.read_text() == "content"


class TestExtractSectionHeaders:
    """Tests for extract_section_headers function."""

    def test_extracts_all_levels(self):
        """Test extraction of headers at all levels."""
        content = "# Title\n\n## Section\n\nText\n\n### Subsection\n\nMore text"
        headers = extract_section_headers(content)
        assert headers == {"# Title", "## Section", "### Subsection"}

    def test_empty_content(self):
        """Test extraction from empty content."""
        assert extract_section_headers("") == set()

    def test_no_headers(self):
        """Test content with no headers."""
        content = "Just some text\nwithout any headers"
        assert extract_section_headers(content) == set()


class TestValidateStructure:
    """Tests for validate_structure method."""

    def test_valid_structure(self):
        """Test validation passes when all required sections present."""
        content = "# Title\n\n## Section A\n\nContent\n\n## Section B\n\nMore"
        doc = LineBasedDocument(content)
        required = {"# Title", "## Section A", "## Section B"}

        is_valid, missing = doc.validate_structure(required)

        assert is_valid is True
        assert missing == []

    def test_missing_section(self):
        """Test validation fails when section is missing."""
        content = "# Title\n\n## Section A\n\nContent"
        doc = LineBasedDocument(content)
        required = {"# Title", "## Section A", "## Section B"}

        is_valid, missing = doc.validate_structure(required)

        assert is_valid is False
        assert missing == ["## Section B"]

    def test_missing_multiple_sections(self):
        """Test validation reports all missing sections sorted."""
        content = "# Title\n\nSome content"
        doc = LineBasedDocument(content)
        required = {"# Title", "## Alpha", "## Beta", "### Gamma"}

        is_valid, missing = doc.validate_structure(required)

        assert is_valid is False
        assert missing == ["## Alpha", "## Beta", "### Gamma"]

    def test_extra_sections_allowed(self):
        """Test validation passes even with extra sections not in required set."""
        content = "# Title\n\n## Required\n\n## Extra\n\nContent"
        doc = LineBasedDocument(content)
        required = {"# Title", "## Required"}

        is_valid, missing = doc.validate_structure(required)

        assert is_valid is True
        assert missing == []
