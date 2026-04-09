"""Unit tests for checkpoint diff generation and cleanup."""

import tempfile
from pathlib import Path

from build_character_profile import (
    SEED_DOCUMENT,
    DocumentManager,
    generate_and_save_diff,
    save_checkpoint_with_diff,
)


class TestDiffGeneration:
    """Tests for diff generation functionality."""

    def test_generate_diff_creates_file(self):
        """Test that diff file is created."""
        old_content = "Line 1\nLine 2\nLine 3\n"
        new_content = "Line 1\nLine 2 modified\nLine 3\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            diff_path = Path(tmpdir) / "test.diff"
            generate_and_save_diff(old_content, new_content, diff_path, 1)

            assert diff_path.exists()
            assert diff_path.stat().st_size > 0

    def test_diff_content_format(self):
        """Test that diff has correct unified diff format."""
        old_content = "Line 1\nLine 2\nLine 3\n"
        new_content = "Line 1\nLine 2 modified\nLine 3\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            diff_path = Path(tmpdir) / "test.diff"
            generate_and_save_diff(old_content, new_content, diff_path, 1)

            diff_content = diff_path.read_text()

            # Should have diff header
            assert "---" in diff_content
            assert "+++" in diff_content
            # Should show the change
            assert "-Line 2" in diff_content
            assert "+Line 2 modified" in diff_content

    def test_diff_with_seed_document(self):
        """Test diff generation from seed document."""
        new_content = SEED_DOCUMENT + "\n## New Section\n\nNew content here.\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            diff_path = Path(tmpdir) / "test.diff"
            generate_and_save_diff(SEED_DOCUMENT, new_content, diff_path, 1)

            assert diff_path.exists()
            diff_content = diff_path.read_text()
            # Should show addition of new section
            assert "+## New Section" in diff_content


class TestCheckpointSaving:
    """Tests for checkpoint saving with diffs."""

    def test_save_first_checkpoint(self):
        """Test saving the first checkpoint (diff from seed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            diffs_dir = Path(tmpdir) / "diffs"

            # Temporarily override globals for testing
            import build_character_profile

            orig_checkpoints = build_character_profile.CHECKPOINTS_DIR
            orig_diffs = build_character_profile.DIFFS_DIR

            try:
                build_character_profile.CHECKPOINTS_DIR = checkpoints_dir
                build_character_profile.DIFFS_DIR = diffs_dir

                doc_manager = DocumentManager(SEED_DOCUMENT)
                previous_content = save_checkpoint_with_diff(doc_manager, 1, None)

                # Check checkpoint file created
                checkpoint_file = checkpoints_dir / "document_v0001.md"
                assert checkpoint_file.exists()

                # Check diff file created
                diff_file = diffs_dir / "document_v0001.diff"
                assert diff_file.exists()

                # Check return value is not None and has content
                assert previous_content is not None
                assert len(previous_content) > 0
                assert "Uno - Character Profile" in previous_content

            finally:
                build_character_profile.CHECKPOINTS_DIR = orig_checkpoints
                build_character_profile.DIFFS_DIR = orig_diffs

    def test_save_subsequent_checkpoint(self):
        """Test saving a checkpoint after the first one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            diffs_dir = Path(tmpdir) / "diffs"

            import build_character_profile

            orig_checkpoints = build_character_profile.CHECKPOINTS_DIR
            orig_diffs = build_character_profile.DIFFS_DIR

            try:
                build_character_profile.CHECKPOINTS_DIR = checkpoints_dir
                build_character_profile.DIFFS_DIR = diffs_dir

                # Save first checkpoint
                doc_manager = DocumentManager(SEED_DOCUMENT)
                prev = save_checkpoint_with_diff(doc_manager, 1, None)

                # Save second checkpoint (no change to document, just testing)
                prev = save_checkpoint_with_diff(doc_manager, 2, prev)

                # Both checkpoints should exist
                assert (checkpoints_dir / "document_v0001.md").exists()
                assert (checkpoints_dir / "document_v0002.md").exists()

                # Both diffs should exist
                assert (diffs_dir / "document_v0001.diff").exists()
                assert (diffs_dir / "document_v0002.diff").exists()

            finally:
                build_character_profile.CHECKPOINTS_DIR = orig_checkpoints
                build_character_profile.DIFFS_DIR = orig_diffs

    def test_rolling_window_cleanup(self):
        """Test that old checkpoints are deleted (keep last 3)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            diffs_dir = Path(tmpdir) / "diffs"

            import build_character_profile

            orig_checkpoints = build_character_profile.CHECKPOINTS_DIR
            orig_diffs = build_character_profile.DIFFS_DIR

            try:
                build_character_profile.CHECKPOINTS_DIR = checkpoints_dir
                build_character_profile.DIFFS_DIR = diffs_dir

                doc_manager = DocumentManager(SEED_DOCUMENT)
                prev = None

                # Save 5 checkpoints
                for i in range(1, 6):
                    prev = save_checkpoint_with_diff(doc_manager, i, prev)

                # Only last 3 checkpoints should exist
                assert not (checkpoints_dir / "document_v0001.md").exists()
                assert not (checkpoints_dir / "document_v0002.md").exists()
                assert (checkpoints_dir / "document_v0003.md").exists()
                assert (checkpoints_dir / "document_v0004.md").exists()
                assert (checkpoints_dir / "document_v0005.md").exists()

                # But ALL diffs should exist
                for i in range(1, 6):
                    assert (diffs_dir / f"document_v{i:04d}.diff").exists()

            finally:
                build_character_profile.CHECKPOINTS_DIR = orig_checkpoints
                build_character_profile.DIFFS_DIR = orig_diffs

    def test_checkpoint_content_matches_manager(self):
        """Test that saved checkpoint matches document manager content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_dir = Path(tmpdir) / "checkpoints"
            diffs_dir = Path(tmpdir) / "diffs"

            import build_character_profile

            orig_checkpoints = build_character_profile.CHECKPOINTS_DIR
            orig_diffs = build_character_profile.DIFFS_DIR

            try:
                build_character_profile.CHECKPOINTS_DIR = checkpoints_dir
                build_character_profile.DIFFS_DIR = diffs_dir

                doc_manager = DocumentManager(SEED_DOCUMENT)
                save_checkpoint_with_diff(doc_manager, 1, None)

                # Read back checkpoint
                checkpoint_file = checkpoints_dir / "document_v0001.md"
                saved_content = checkpoint_file.read_text()

                # Should match document manager's content
                assert saved_content == doc_manager.get_content()

            finally:
                build_character_profile.CHECKPOINTS_DIR = orig_checkpoints
                build_character_profile.DIFFS_DIR = orig_diffs
