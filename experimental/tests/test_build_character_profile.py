"""Unit tests for character_profile_utils.py."""

from pathlib import Path

from build_character_profile import natural_sort_key


class TestNaturalSortKey:
    """Tests for the natural_sort_key function."""

    def test_simple_numeric_sorting(self):
        """Test that numeric parts are sorted numerically, not lexicographically."""
        paths = [Path("pkna-10"), Path("pkna-2"), Path("pkna-1")]
        sorted_paths = sorted(paths, key=natural_sort_key)
        expected = [Path("pkna-1"), Path("pkna-2"), Path("pkna-10")]
        assert sorted_paths == expected

    def test_multi_part_numeric_sorting(self):
        """Test sorting with multiple numeric parts (e.g., pkna-0-2)."""
        paths = [
            Path("pkna-0-3"),
            Path("pkna-0-2"),
            Path("pkna-0-10"),
            Path("pkna-0"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        expected = [
            Path("pkna-0"),
            Path("pkna-0-2"),
            Path("pkna-0-3"),
            Path("pkna-0-10"),
        ]
        assert sorted_paths == expected

    def test_mixed_issues(self):
        """Test the full expected sorting order for PKNA issues."""
        paths = [
            Path("pkna-11"),
            Path("pkna-0-2"),
            Path("pkna-10"),
            Path("pkna-0"),
            Path("pkna-1"),
            Path("pkna-0-3"),
            Path("pkna-2"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        expected = [
            Path("pkna-0"),
            Path("pkna-0-2"),
            Path("pkna-0-3"),
            Path("pkna-1"),
            Path("pkna-2"),
            Path("pkna-10"),
            Path("pkna-11"),
        ]
        assert sorted_paths == expected

    def test_large_numbers(self):
        """Test sorting with larger issue numbers."""
        paths = [
            Path("pkna-100"),
            Path("pkna-20"),
            Path("pkna-3"),
            Path("pkna-99"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        expected = [
            Path("pkna-3"),
            Path("pkna-20"),
            Path("pkna-99"),
            Path("pkna-100"),
        ]
        assert sorted_paths == expected

    def test_return_type(self):
        """Test that the function returns a tuple."""
        result = natural_sort_key(Path("pkna-0-2"))
        assert isinstance(result, tuple)

    def test_key_structure_simple(self):
        """Test the structure of the key for a simple issue."""
        key = natural_sort_key(Path("pkna-5"))
        assert key == ("pkna", 5)

    def test_key_structure_multi_part(self):
        """Test the structure of the key for multi-part issue."""
        key = natural_sort_key(Path("pkna-0-2"))
        assert key == ("pkna", 0, 2)

    def test_non_numeric_parts(self):
        """Test handling of non-numeric parts."""
        paths = [
            Path("issue-beta"),
            Path("issue-alpha"),
            Path("issue-gamma"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        # Alphabetical sorting for non-numeric parts
        expected = [
            Path("issue-alpha"),
            Path("issue-beta"),
            Path("issue-gamma"),
        ]
        assert sorted_paths == expected

    def test_empty_directory_name(self):
        """Test handling of edge case with empty parts (though unlikely)."""
        # This shouldn't happen in practice, but test defensive behavior
        key = natural_sort_key(Path("test"))
        assert key == ("test",)

    def test_preserves_order_for_identical_issues(self):
        """Test that identical issues maintain their relative order."""
        paths = [Path("pkna-1"), Path("pkna-1")]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert len(sorted_paths) == 2
        assert all(p.name == "pkna-1" for p in sorted_paths)
