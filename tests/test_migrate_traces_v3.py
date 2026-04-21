"""Unit tests for the v2 -> v3 trace migration (bracket -> XML tags)."""

import json
from pathlib import Path

from datagen.migrate_traces_v3 import (
    _has_empty_user_message,
    migrate_content,
    migrate_traces,
)


class TestMigrateContent:
    def test_context_and_message(self):
        old = "[Context]\nInterlocutor: Paperino\n\n[Message]\nCiao, Uno!"
        result = migrate_content(old)
        assert result == (
            "<context>\nInterlocutor: Paperino\n</context>"
            "\n\n<message>\nCiao, Uno!\n</message>"
        )

    def test_multiline_context(self):
        old = (
            "[Context]\nInterlocutor: Paperino\n\n"
            "Memory context:\nRecent interactions:\n- [today] test\n\n"
            "[Message]\nHello"
        )
        result = migrate_content(old)
        assert "<context>\nInterlocutor: Paperino" in result
        assert "Recent interactions:" in result
        assert "\n</context>" in result
        assert "<message>\nHello\n</message>" in result
        assert "[Context]" not in result
        assert "[Message]" not in result

    def test_message_only(self):
        old = "[Message]\nCiao!"
        result = migrate_content(old)
        assert result == "<message>\nCiao!\n</message>"

    def test_plain_text_unchanged(self):
        plain = "Just a normal message"
        assert migrate_content(plain) == plain

    def test_context_with_brackets_in_text(self):
        old = "[Context]\nInterlocutor: [unknown]\n\n[Message]\nHi"
        result = migrate_content(old)
        assert "<context>\nInterlocutor: [unknown]\n</context>" in result
        assert "<message>\nHi\n</message>" in result


class TestHasEmptyUserMessage:
    def test_empty_with_context(self):
        trace = {
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: Paperino\n\n[Message]\n",
                }
            ]
        }
        assert _has_empty_user_message(trace)

    def test_nonempty_with_context(self):
        trace = {
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: Paperino\n\n[Message]\nCiao!",
                }
            ]
        }
        assert not _has_empty_user_message(trace)

    def test_empty_plain(self):
        trace = {"messages": [{"role": "user", "content": ""}]}
        assert _has_empty_user_message(trace)

    def test_no_user_message(self):
        trace = {"messages": [{"role": "assistant", "content": "Hi"}]}
        assert _has_empty_user_message(trace)


class TestMigrateTraces:
    def test_migrates_user_messages(self, tmp_path: Path):
        trace = {
            "id": "t-001",
            "metadata": {},
            "memory_context": "",
            "user_summary": "Paperino",
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: Paperino\n\n[Message]\nCiao!",
                },
                {"role": "assistant", "content": "Ehi, socio!"},
            ],
        }
        input_path = tmp_path / "traces.jsonl"
        input_path.write_text(json.dumps(trace) + "\n")

        migrated, dropped = migrate_traces(input_path)
        assert migrated == 1
        assert dropped == 0

        backup = tmp_path / "traces_v2.jsonl"
        assert backup.exists()

        result = json.loads(input_path.read_text().strip())
        assert "<context>" in result["messages"][0]["content"]
        assert "<message>" in result["messages"][0]["content"]
        assert "[Context]" not in result["messages"][0]["content"]
        assert result["messages"][1]["content"] == "Ehi, socio!"

    def test_drops_empty_claim_traces(self, tmp_path: Path):
        good_trace = {
            "id": "generated-001",
            "metadata": {"prompt_source": "generated"},
            "memory_context": "",
            "user_summary": "",
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: X\n\n[Message]\nHello",
                },
                {"role": "assistant", "content": "Hi"},
            ],
        }
        empty_claim = {
            "id": "claim-tom-0001-1",
            "metadata": {"prompt_source": "claim_derived"},
            "memory_context": "",
            "user_summary": "",
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: Paperino\n\n[Message]\n",
                },
                {"role": "assistant", "content": "..."},
            ],
        }
        input_path = tmp_path / "traces.jsonl"
        input_path.write_text(
            json.dumps(good_trace) + "\n" + json.dumps(empty_claim) + "\n"
        )

        migrated, dropped = migrate_traces(input_path)
        assert migrated == 1
        assert dropped == 1

        lines = input_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "generated-001"

    def test_keeps_claim_traces_with_messages(self, tmp_path: Path):
        claim_with_msg = {
            "id": "claim-tom-0001-1",
            "metadata": {"prompt_source": "claim_derived"},
            "memory_context": "",
            "user_summary": "",
            "messages": [
                {
                    "role": "user",
                    "content": "[Context]\nInterlocutor: X\n\n[Message]\nCiao!",
                },
                {"role": "assistant", "content": "Ehi"},
            ],
        }
        input_path = tmp_path / "traces.jsonl"
        input_path.write_text(json.dumps(claim_with_msg) + "\n")

        migrated, dropped = migrate_traces(input_path)
        assert migrated == 1
        assert dropped == 0

    def test_skips_already_migrated(self, tmp_path: Path):
        trace = {
            "id": "t-002",
            "metadata": {},
            "memory_context": "",
            "user_summary": "",
            "messages": [
                {"role": "user", "content": "Plain message, no tags"},
                {"role": "assistant", "content": "Response"},
            ],
        }
        input_path = tmp_path / "traces.jsonl"
        input_path.write_text(json.dumps(trace) + "\n")

        migrated, dropped = migrate_traces(input_path)
        assert migrated == 0
        assert dropped == 0

    def test_refuses_if_backup_exists(self, tmp_path: Path):
        input_path = tmp_path / "traces.jsonl"
        input_path.write_text("{}\n")
        backup = tmp_path / "traces_v2.jsonl"
        backup.write_text("old backup\n")

        try:
            migrate_traces(input_path)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass
