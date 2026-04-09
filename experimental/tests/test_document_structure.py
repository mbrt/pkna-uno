"""Unit tests for document structure classes."""

from build_character_profile import (
    DocumentEdit,
    DocumentManager,
    DocumentStructure,
    EditOperation,
    Line,
    Section,
)


class TestLine:
    """Tests for the Line class."""

    def test_line_creation(self):
        """Test creating a Line."""
        line = Line(content="Test content")
        assert line.content == "Test content"


class TestSection:
    """Tests for the Section class."""

    def test_section_creation(self):
        """Test creating a Section."""
        section = Section(header="## Test", level=2)
        assert section.header == "## Test"
        assert section.level == 2
        assert section.lines == []
        assert section.subsections == []

    def test_section_with_content(self):
        """Test creating a Section with lines and subsections."""
        subsection = Section(header="### Subsection", level=3)
        section = Section(
            header="## Main",
            level=2,
            lines=[Line(content="Line 1"), Line(content="Line 2")],
            subsections=[subsection],
        )
        assert len(section.lines) == 2
        assert len(section.subsections) == 1
        assert section.subsections[0] == subsection


class TestDocumentStructureParsing:
    """Tests for markdown parsing."""

    def test_parse_simple_document(self):
        """Test parsing a simple document."""
        text = """# Title

Content line

## Section 1

Section 1 content
"""
        root = DocumentStructure.parse_markdown(text)
        assert root.level == 0
        assert len(root.subsections) == 1
        assert root.subsections[0].header == "# Title"
        assert len(root.subsections[0].subsections) == 1
        assert root.subsections[0].subsections[0].header == "## Section 1"

    def test_parse_nested_sections(self):
        """Test parsing nested sections."""
        text = """## Level 2

### Level 3

#### Level 4

Content
"""
        root = DocumentStructure.parse_markdown(text)
        level2 = root.subsections[0]
        assert level2.level == 2
        level3 = level2.subsections[0]
        assert level3.level == 3
        level4 = level3.subsections[0]
        assert level4.level == 4
        assert len(level4.lines) > 0

    def test_parse_ignores_empty_lines(self):
        """Test that empty lines within content are ignored."""
        text = """## Section

Line 1

Line 2
"""
        root = DocumentStructure.parse_markdown(text)
        section = root.subsections[0]
        # Should have: "Line 1", "Line 2"
        assert len(section.lines) == 2

    def test_parse_multiple_sections_same_level(self):
        """Test parsing multiple sections at the same level."""
        text = """## Section 1

Content 1

## Section 2

Content 2

## Section 3

Content 3
"""
        root = DocumentStructure.parse_markdown(text)
        assert len(root.subsections) == 3
        assert root.subsections[0].header == "## Section 1"
        assert root.subsections[1].header == "## Section 2"
        assert root.subsections[2].header == "## Section 3"


class TestDocumentStructureSerialization:
    """Tests for markdown serialization."""

    def test_roundtrip_simple(self):
        """Test parse -> serialize roundtrip."""
        text = """# Title

Content

## Section

More multi-line
content

Another paragraph.
"""
        root = DocumentStructure.parse_markdown(text)
        result = DocumentStructure.to_markdown(root)
        # Should be essentially the same (might have trailing newline differences)
        assert result.strip() == text.strip()

    def test_to_markdown_preserves_structure(self):
        """Test that serialization preserves structure."""
        text = """## Section 1

### Subsection 1.1

Content 1.1

### Subsection 1.2

Content 1.2

## Section 2

Content 2
"""
        root = DocumentStructure.parse_markdown(text)
        result = DocumentStructure.to_markdown(root)
        assert "## Section 1" in result
        assert "### Subsection 1.1" in result
        assert "### Subsection 1.2" in result
        assert "## Section 2" in result


class TestDocumentStructureNavigation:
    """Tests for section navigation."""

    def test_find_section_simple(self):
        """Test finding a section by path."""
        text = """## Personality Traits

Content
"""
        root = DocumentStructure.parse_markdown(text)
        section = DocumentStructure.find_section(root, "Personality Traits")
        assert section is not None
        assert section.header == "## Personality Traits"

    def test_find_section_case_insensitive(self):
        """Test that section finding is case-insensitive."""
        text = """## Personality Traits

Content
"""
        root = DocumentStructure.parse_markdown(text)
        section = DocumentStructure.find_section(root, "personality traits")
        assert section is not None
        section = DocumentStructure.find_section(root, "PERSONALITY TRAITS")
        assert section is not None

    def test_find_section_nested(self):
        """Test finding a nested section by name."""
        text = """## Relationships

### With Paperinik

Content
"""
        root = DocumentStructure.parse_markdown(text)
        # Find the subsection directly by name
        section = DocumentStructure.find_section(root, "With Paperinik")
        assert section is not None
        assert section.header == "### With Paperinik"

    def test_find_section_not_found(self):
        """Test that None is returned for missing sections."""
        text = """## Section 1

Content
"""
        root = DocumentStructure.parse_markdown(text)
        section = DocumentStructure.find_section(root, "NonExistent")
        assert section is None

    def test_find_section_empty_path(self):
        """Test that empty path returns root."""
        text = """## Section

Content
"""
        root = DocumentStructure.parse_markdown(text)
        section = DocumentStructure.find_section(root, "")
        assert section == root


class TestDocumentStructureLineSearch:
    """Tests for line searching."""

    def test_find_line_exact_match(self):
        """Test finding a line with exact match."""
        section = Section(
            header="## Test",
            level=2,
            lines=[
                Line(content="To be developed"),
                Line(content="Other content"),
            ],
        )
        line, is_unique = DocumentStructure.find_line_in_section(
            section, "To be developed"
        )
        assert line is not None
        assert is_unique is True
        assert line.content == "To be developed"

    def test_find_line_partial_match(self):
        """Test finding a line with partial match."""
        section = Section(
            header="## Test",
            level=2,
            lines=[
                Line(content="To be developed based on scenes"),
                Line(content="Other content"),
            ],
        )
        line, is_unique = DocumentStructure.find_line_in_section(section, "to be")
        assert line is not None
        assert is_unique is True

    def test_find_line_case_insensitive(self):
        """Test that line search is case-insensitive."""
        section = Section(
            header="## Test",
            level=2,
            lines=[Line(content="To Be Developed")],
        )
        line, is_unique = DocumentStructure.find_line_in_section(section, "to be")
        assert line is not None
        assert is_unique is True

    def test_find_line_not_found(self):
        """Test that None is returned when line not found."""
        section = Section(
            header="## Test",
            level=2,
            lines=[Line(content="Content")],
        )
        line, is_unique = DocumentStructure.find_line_in_section(section, "nonexistent")
        assert line is None
        assert is_unique is False

    def test_find_line_multiple_matches(self):
        """Test handling of multiple matches."""
        section = Section(
            header="## Test",
            level=2,
            lines=[
                Line(content="Content here"),
                Line(content="More content"),
                Line(content="Even more content"),
            ],
        )
        line, is_unique = DocumentStructure.find_line_in_section(section, "content")
        assert line is not None
        assert is_unique is False  # Multiple matches


class TestDocumentManager:
    """Tests for DocumentManager edit operations."""

    def test_add_line_operation(self):
        """Test adding a line to a section."""
        text = """## Section

Existing content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.ADD_LINE,
            section_path="Section",
            new_content="New line",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "Existing content" in result
        assert "New line" in result

    def test_replace_line_operation(self):
        """Test replacing a line."""
        text = """## Section

To be developed
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.REPLACE_LINE,
            section_path="Section",
            search_text="to be developed",
            new_content="Actual content now",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "To be developed" not in result
        assert "Actual content now" in result

    def test_delete_line_operation(self):
        """Test deleting a line."""
        text = """## Section

Line to delete

Other line
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.DELETE_LINE,
            section_path="Section",
            search_text="line to delete",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "Line to delete" not in result
        assert "Other line" in result

    def test_add_subsection_operation(self):
        """Test adding a subsection."""
        text = """## Parent Section

Parent content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.ADD_SUBSECTION,
            section_path="Parent Section",
            subsection_header="New Subsection",
            new_content="Subsection content",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "### New Subsection" in result
        assert "Subsection content" in result

    def test_add_subsection_with_hash_symbols(self):
        """Test adding a subsection with explicit ### symbols."""
        text = """## Parent Section

Content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.ADD_SUBSECTION,
            section_path="Parent Section",
            subsection_header="### Explicit Level",
            new_content="Content",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "### Explicit Level" in result

    def test_section_not_found(self):
        """Test that edit fails when section not found."""
        text = """## Section

Content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.ADD_LINE,
            section_path="NonExistent",
            new_content="New line",
        )
        success = manager.apply_edit(edit)
        assert success is False

    def test_line_not_found(self):
        """Test that replace fails when line not found."""
        text = """## Section

Content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.REPLACE_LINE,
            section_path="Section",
            search_text="nonexistent",
            new_content="New content",
        )
        success = manager.apply_edit(edit)
        assert success is False

    def test_ambiguous_line_match(self):
        """Test that replace fails when multiple lines match."""
        text = """## Section

Content here

More content

Even more content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.REPLACE_LINE,
            section_path="Section",
            search_text="content",  # Matches multiple lines
            new_content="New content",
        )
        success = manager.apply_edit(edit)
        assert success is False

    def test_nested_section_path(self):
        """Test editing in a nested section."""
        text = """## Parent

### Child

Child content
"""
        manager = DocumentManager(text)
        edit = DocumentEdit(
            operation=EditOperation.ADD_LINE,
            section_path="Child",  # Find nested section directly by name
            new_content="New child content",
        )
        success = manager.apply_edit(edit)
        assert success is True
        result = manager.get_content()
        assert "New child content" in result

    def test_multiple_edits(self):
        """Test applying multiple edits sequentially."""
        text = """## Section

To be developed
"""
        manager = DocumentManager(text)

        # Replace placeholder
        edit1 = DocumentEdit(
            operation=EditOperation.REPLACE_LINE,
            section_path="Section",
            search_text="to be developed",
            new_content="Actual content",
        )
        assert manager.apply_edit(edit1) is True

        # Add more content
        edit2 = DocumentEdit(
            operation=EditOperation.ADD_LINE,
            section_path="Section",
            new_content="Additional insight",
        )
        assert manager.apply_edit(edit2) is True

        result = manager.get_content()
        assert "Actual content" in result
        assert "Additional insight" in result
        assert "To be developed" not in result
