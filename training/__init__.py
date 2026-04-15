def ensure_unsloth() -> None:
    """Import unsloth before transformers to enable its monkey-patches.

    Must be called before any module that imports transformers (e.g.
    assemble_sft, which uses AutoTokenizer). Once transformers is in
    sys.modules, unsloth's patches are incomplete.
    """
    import unsloth  # noqa: F401
