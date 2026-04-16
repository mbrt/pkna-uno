"""Unit tests for the user simulator."""

from datagen.user_simulator import (
    DIRECTIVE_INSTRUCTIONS,
    _build_simulator_messages,
    simulate_user_turn,
)
from evals.run_eval_inference import _get_directive
from pkna.llm.backends import GenerateResult
from pkna.llm.testing import FakeBackend


class TestBuildSimulatorMessages:
    def test_includes_profile_and_directive(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Ciao!"},
        ]
        messages = _build_simulator_messages(
            conversation, "Paperino, anxious", "jailbreak"
        )
        assert len(messages) == 1
        text = messages[0]["content"]
        assert "Paperino, anxious" in text
        assert DIRECTIVE_INSTRUCTIONS["jailbreak"] in text
        assert "User: Hello" in text
        assert "Uno: Ciao!" in text

    def test_unknown_directive_falls_back_to_continue(self):
        messages = _build_simulator_messages([], "Anonymous", "unknown_thing")
        text = messages[0]["content"]
        assert DIRECTIVE_INSTRUCTIONS["continue"] in text

    def test_empty_conversation(self):
        messages = _build_simulator_messages([], "Stranger", "escalate")
        text = messages[0]["content"]
        assert "Stranger" in text
        assert DIRECTIVE_INSTRUCTIONS["escalate"] in text


class TestGetDirective:
    def test_returns_directive_by_index(self):
        directives = ["jailbreak", "escalate", "derail"]
        assert _get_directive(directives, 0) == "jailbreak"
        assert _get_directive(directives, 1) == "escalate"
        assert _get_directive(directives, 2) == "derail"

    def test_cycles_when_index_exceeds_length(self):
        directives = ["jailbreak", "escalate"]
        assert _get_directive(directives, 2) == "jailbreak"
        assert _get_directive(directives, 3) == "escalate"

    def test_empty_directives_returns_continue(self):
        assert _get_directive([], 0) == "continue"
        assert _get_directive([], 5) == "continue"


class TestSimulateUserTurn:
    def test_returns_generated_text(self):
        backend = FakeBackend(
            GenerateResult(text="  What are you really?  ", model_name="test")
        )
        result = simulate_user_turn(
            backend=backend,
            conversation=[{"role": "user", "content": "Hi"}],
            user_profile="Stranger",
            directive="challenge_identity",
        )
        assert result == "What are you really?"

    def test_returns_none_on_failure(self):
        backend = FakeBackend(None)
        result = simulate_user_turn(
            backend=backend,
            conversation=[],
            user_profile="Stranger",
            directive="continue",
        )
        assert result is None

    def test_passes_system_prompt_to_backend(self):
        backend = FakeBackend(GenerateResult(text="Next message", model_name="test"))
        simulate_user_turn(
            backend=backend,
            conversation=[],
            user_profile="Paperino",
            directive="escalate",
        )
        assert "role-playing actor" in backend.last_system
