def ensure_unsloth() -> None:
    """Import unsloth before transformers to enable its monkey-patches.

    Must be called before any module that imports transformers (e.g.
    assemble_sft, which uses AutoTokenizer). Once transformers is in
    sys.modules, unsloth's patches are incomplete.
    """
    import unsloth  # noqa: F401


def select_device_map() -> str | None:
    """Return the device_map for FastLanguageModel.from_pretrained.

    On a single GPU, returns None (Unsloth default). On multiple GPUs,
    returns "balanced" to shard the model across all GPUs via pipeline
    parallelism. This is required for models that exceed a single GPU's
    VRAM (e.g. Qwen3.6-35B-A3B at ~74 GB on 4x L40S @ 48 GB each).
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        return "balanced"
    return None
