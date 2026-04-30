#!/usr/bin/env python3
"""Download HuggingFace models to a local directory for S3 caching.

Downloads each requested model (and its base model if it is a LoRA adapter)
to a flat local directory tree under ``--cache-dir``. Each model is stored at
``<cache-dir>/<org>/<model-name>/``, mirroring the HuggingFace model ID.

Usage:
    python infra/prefetch_models.py unsloth/Qwen3.5-4B unsloth/Qwen3.5-0.8B
    python infra/prefetch_models.py --cache-dir /tmp/models mbrt/uno-sft-adapter
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def fetch_model(model_id: str, cache_dir: Path, token: str | None) -> list[str]:
    """Download *model_id* and its base model if it is a LoRA adapter.

    Returns a list of local directory paths that were populated.
    """
    local_dir = cache_dir / model_id
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} → {local_dir}")
    snapshot_download(model_id, local_dir=str(local_dir), token=token)
    print(f"Download complete: {local_dir}")

    downloaded = [str(local_dir)]

    cfg_path = local_dir / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            base_model = json.load(f).get("base_model_name_or_path", "")
        if base_model:
            print(f"LoRA adapter detected. Downloading base model: {base_model}")
            base_dirs = fetch_model(base_model, cache_dir, token)
            downloaded.extend(base_dirs)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="HuggingFace model IDs to download")
    parser.add_argument(
        "--cache-dir",
        default="model-cache",
        help="Root directory for downloaded models (default: model-cache)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    cache_dir = Path(args.cache_dir)

    for model_id in args.models:
        fetch_model(model_id, cache_dir, token)

    print("All models downloaded.")


if __name__ == "__main__":
    main()
