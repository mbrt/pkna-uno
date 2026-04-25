#!/usr/bin/env python3
"""Minimal GPU smoke test: CUDA -> PyTorch -> Unsloth -> generate one token.

Verifies the full stack (drivers, CUDA, Triton JIT, Unsloth, LoRA training)
works end-to-end. Uses the smallest available model (0.8B) for speed.

Usage:
    uv run python training/smoke_test.py

    # With Triton logging to see JIT compilation progress:
    TRITON_LOG_LEVEL=info uv run python training/smoke_test.py
"""

import sys
import time
from datetime import datetime, timezone

import torch


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def stage_cuda() -> None:
    print(f"[{now()}] === Stage 1: CUDA ===")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda}")
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(
        f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB"
    )
    print(f"  GPU count:  {torch.cuda.device_count()}")


def stage_triton() -> None:
    print(f"\n[{now()}] === Stage 2: Triton ===")
    import triton

    print(f"  Triton:     {triton.__version__}")


def stage_model_load() -> tuple:  # type: ignore[type-arg]
    print(f"\n[{now()}] === Stage 3: Unsloth model load ===")
    t0 = time.time()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-0.8B",
        max_seq_length=512,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="auto",
    )
    print(f"  [{now()}] Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def stage_inference(model, tokenizer) -> None:  # type: ignore[no-any-explicit]
    from unsloth import FastLanguageModel

    print(f"\n[{now()}] === Stage 4: Inference (Triton JIT happens here) ===")
    FastLanguageModel.for_inference(model)
    enc = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    if enc.pad_token_id is None:
        enc.pad_token_id = enc.eos_token_id

    inputs = enc("Ciao!", return_tensors="pt").to(model.device)

    print(f"  [{now()}] Generating 1 token (may compile Triton kernels)...")
    sys.stdout.flush()
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=enc.pad_token_id)
    print(f"  [{now()}] First token in {time.time() - t0:.1f}s")

    print(f"  [{now()}] Generating 16 tokens...")
    sys.stdout.flush()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, pad_token_id=enc.pad_token_id)
    print(f"  [{now()}] 16 tokens in {time.time() - t0:.1f}s")
    print(f"  Output: {enc.decode(out[0], skip_special_tokens=True)}")


def stage_training() -> None:
    from unsloth import FastLanguageModel

    print(f"\n[{now()}] === Stage 5: LoRA + 1 training step ===")
    sys.stdout.flush()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-0.8B",
        max_seq_length=512,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=512,
    )

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_dict({"text": ["Ciao, sono Uno! " * 20] * 4})
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            max_seq_length=512,  # ty: ignore[unknown-argument]  # added by unsloth
            per_device_train_batch_size=1,
            max_steps=1,
            output_dir="/tmp/smoke_test",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            bf16=True,
        ),
    )
    print(f"  [{now()}] Starting training step...")
    sys.stdout.flush()
    t0 = time.time()
    trainer.train()
    print(f"  [{now()}] 1 training step in {time.time() - t0:.1f}s")


def main() -> None:
    print(f"[{now()}] Starting smoke test")
    stage_cuda()
    stage_triton()
    model, tokenizer = stage_model_load()
    stage_inference(model, tokenizer)
    del model, tokenizer
    stage_training()
    print(f"\n[{now()}] === All stages passed ===")


if __name__ == "__main__":
    main()
