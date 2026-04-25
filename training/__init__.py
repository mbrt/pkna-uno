"""Training utilities and per-model configuration.

Each supported model has a :class:`ModelConfig` that bundles LoRA
parameters, learning rates, and other training knobs.  Use
:func:`get_config` to look up the config for a model by HuggingFace name.

Learning rates follow "LoRA Without Regret" (Schulman et al., 2025):
optimal LoRA LR scales with hidden size and is ~10x the FullFT optimal.
Smaller models need lower LR to avoid oscillating loss.

LoRA alpha follows the Unsloth recommendation of alpha >= rank
(alpha/rank >= 1).  For MoE models, per-expert rank is set via PEFT
``rank_pattern`` and ``use_rslora`` adapts the effective scaling
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def ensure_unsloth() -> None:
    """Import unsloth before transformers to enable its monkey-patches.

    Must be called before any module that imports transformers (e.g.
    assemble_sft, which uses AutoTokenizer). Once transformers is in
    sys.modules, unsloth's patches are incomplete.
    """
    import unsloth  # noqa: F401


def select_device_map() -> str:
    """Return the device_map for FastLanguageModel.from_pretrained.

    On a single GPU, returns "auto" to place the model on the available
    GPU. On multiple GPUs, returns "balanced" to shard the model via
    pipeline parallelism. "auto" is needed because device_map=None
    leaves LoRA adapters on CPU with some Unsloth/PEFT versions.
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        return "balanced"
    return "auto"


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter parameters passed to ``FastLanguageModel.get_peft_model``."""

    rank: int = 64
    alpha: int = 64
    target_modules: str = "all-linear"
    use_rslora: bool = False
    rank_pattern: dict[str, int] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``FastLanguageModel.get_peft_model``."""
        kw: dict[str, Any] = {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": self.target_modules,
        }
        if self.use_rslora:
            kw["use_rslora"] = True
        if self.rank_pattern:
            kw["rank_pattern"] = dict(self.rank_pattern)
        return kw


_DENSE_LORA = LoraConfig()

# MoE: routed expert projections get a reduced rank (64 / 8 active
# experts = 8).  Shared expert and attention layers keep the full rank.
# rsLoRA adapts alpha per layer automatically.
_MOE_LORA = LoraConfig(
    use_rslora=True,
    rank_pattern={
        "gate_up_proj": 8,
        "mlp.experts.down_proj": 8,
    },
)


# ---------------------------------------------------------------------------
# Per-model training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Training hyperparameters for a specific model size."""

    sft_lr: float
    distill_lr: float
    lora: LoraConfig = field(default_factory=lambda: _DENSE_LORA)


_REGISTRY: dict[str, ModelConfig] = {
    "0.8B": ModelConfig(sft_lr=5e-5, distill_lr=3e-5),
    "4B": ModelConfig(sft_lr=2e-4, distill_lr=1e-4),
    "35B-A3B": ModelConfig(sft_lr=2e-4, distill_lr=1e-4, lora=_MOE_LORA),
}

_FALLBACK = ModelConfig(sft_lr=2e-4, distill_lr=1e-4)


def get_config(model_name: str) -> ModelConfig:
    """Look up the :class:`ModelConfig` for *model_name*.

    Matches the last path component (e.g. ``"Qwen3.5-4B"`` from
    ``"unsloth/Qwen3.5-4B"``) against known model-size fragments.
    Returns a sensible default if no match is found.
    """
    key = model_name.rsplit("/", 1)[-1]
    for fragment, cfg in _REGISTRY.items():
        if fragment in key:
            return cfg
    return _FALLBACK
