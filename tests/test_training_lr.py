"""Tests for per-model training configuration."""

import pytest

from training import ModelConfig, get_config


class TestGetConfig:
    @pytest.mark.parametrize(
        "model_name, expected_sft_lr, expected_distill_lr",
        [
            ("unsloth/Qwen3.5-0.8B", 5e-5, 3e-5),
            ("unsloth/Qwen3.5-4B", 2e-4, 1e-4),
            ("unsloth/Qwen3.6-35B-A3B", 2e-4, 1e-4),
            ("Qwen3.5-0.8B", 5e-5, 3e-5),
            # Unknown model falls back to 4B-class defaults
            ("unsloth/SomeOtherModel-7B", 2e-4, 1e-4),
        ],
    )
    def test_learning_rates(
        self, model_name: str, expected_sft_lr: float, expected_distill_lr: float
    ) -> None:
        cfg = get_config(model_name)
        assert cfg.sft_lr == expected_sft_lr
        assert cfg.distill_lr == expected_distill_lr

    def test_returns_model_config(self) -> None:
        assert isinstance(get_config("unsloth/Qwen3.5-4B"), ModelConfig)


class TestLoraConfig:
    def test_dense_model_returns_standard_config(self) -> None:
        kw = get_config("unsloth/Qwen3.5-4B").lora.to_kwargs()
        assert kw == {"r": 64, "lora_alpha": 64, "target_modules": "all-linear"}

    def test_dense_model_has_no_rank_pattern(self) -> None:
        lora = get_config("unsloth/Qwen3.5-0.8B").lora
        assert lora.rank_pattern == {}
        assert lora.use_rslora is False

    def test_moe_model_has_rank_pattern(self) -> None:
        lora = get_config("unsloth/Qwen3.6-35B-A3B").lora
        assert lora.rank_pattern["gate_up_proj"] == 8
        assert lora.rank_pattern["mlp.experts.down_proj"] == 8

    def test_moe_model_uses_rslora(self) -> None:
        assert get_config("unsloth/Qwen3.6-35B-A3B").lora.use_rslora is True

    def test_moe_model_keeps_full_base_rank(self) -> None:
        assert get_config("unsloth/Qwen3.5-35B-A3B").lora.rank == 64

    def test_moe_to_kwargs_includes_patterns(self) -> None:
        kw = get_config("unsloth/Qwen3.6-35B-A3B").lora.to_kwargs()
        assert "use_rslora" in kw
        assert "rank_pattern" in kw

    @pytest.mark.parametrize(
        "model_name",
        [
            "unsloth/Qwen3.5-35B-A3B",
            "unsloth/Qwen3.6-35B-A3B",
            "Qwen3.6-35B-A3B",
        ],
    )
    def test_moe_detection(self, model_name: str) -> None:
        assert get_config(model_name).lora.use_rslora is True

    @pytest.mark.parametrize(
        "model_name",
        [
            "unsloth/Qwen3.5-4B",
            "unsloth/Qwen3.5-0.8B",
            "unsloth/Qwen3.5-9B",
        ],
    )
    def test_dense_detection(self, model_name: str) -> None:
        assert get_config(model_name).lora.use_rslora is False
