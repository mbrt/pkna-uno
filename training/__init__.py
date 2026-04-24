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
