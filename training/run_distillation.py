#!/usr/bin/env python3

"""On-policy self-distillation for behavior recovery after SFT.

Uses the student's own base weights (with LoRA adapters disabled) as the
teacher, avoiding a second copy of the base model in VRAM.  This halves
the model memory required compared to loading a separate teacher.

Uses TRL's DistillationTrainer with reverse KL (beta=1.0) and fully
on-policy generation (lmbda=1.0) to recover general instruction-following
capabilities degraded by personality SFT.

Hyperparameters follow docs/fine-tuning/on-policy-distillation.md.

Usage:
    python training/run_distillation.py \\
        --dataset output/distillation/prompts \\
        --adapter output/sft/lora_adapter \\
        --output output/distillation/lora_adapter \\
        --model unsloth/Qwen3.5-4B

    # With GGUF export:
    python training/run_distillation.py \\
        --dataset output/distillation/prompts \\
        --adapter output/sft/lora_adapter \\
        --output output/distillation/lora_adapter \\
        --export-gguf q4_k_m
"""

# Unsloth must be imported before transformers/trl to monkey-patch
# optimizations. See run_sft.py for details.
from unsloth import FastLanguageModel

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from mlflow import log_params, set_experiment, set_tracking_uri, start_run
from trl.experimental.distillation import (
    DistillationConfig,
    DistillationTrainer,
)

from pkna.logging import setup_logging
from training import get_config, select_device_map

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

console, log = setup_logging()


class _UnslothDistillationTrainer(DistillationTrainer):
    """Self-distillation trainer with Unsloth compatibility and weight sharing.

    Two customizations over the base ``DistillationTrainer``:

    1. **Weight-shared teacher**: Instead of loading a separate teacher
       model, ``_get_teacher_logits`` temporarily disables the LoRA
       adapters on the student via PEFT's ``disable_adapter()`` context
       manager, producing base-model logits from the same weights.
       This halves model VRAM usage for self-distillation.

    2. **Unsloth inference_mode fix**: Unsloth's patched
       ``model.generate()`` runs under ``torch.inference_mode()``,
       producing tensors that cannot participate in autograd.  The
       ``compute_loss`` override clones any such tensors before they
       reach the loss computation.
    """

    def _get_teacher_logits(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> torch.Tensor:
        """Produce base-model logits by disabling LoRA adapters."""
        model = self.model
        assert model is not None
        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter():  # ty: ignore[call-non-callable]
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
        if was_training:
            model.train()
        return logits

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        for key in ("input_ids", "attention_mask", "labels"):
            v = inputs.get(key)
            if isinstance(v, torch.Tensor) and not v.requires_grad:
                inputs[key] = v.clone()
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def _save_peft_adapter(model: Any, tokenizer: Any, output_path: str) -> None:
    """Save a PEFT adapter, working around Qwen3.5's missing vocab_size.

    PEFT's ``save_pretrained`` loads a fresh ``Qwen3_5Config`` via
    ``from_pretrained`` and accesses ``.vocab_size``, which doesn't exist
    on the multimodal composite config.  We monkeypatch the config class
    to expose ``vocab_size`` before saving.
    """
    cfg_cls = model.config.__class__
    original_init = cfg_cls.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not hasattr(self, "vocab_size") and hasattr(self, "text_config"):
            self.vocab_size = self.text_config.vocab_size

    cfg_cls.__init__ = _patched_init  # ty: ignore[invalid-assignment]
    try:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    finally:
        cfg_cls.__init__ = original_init


def run_distillation(
    dataset_path: str,
    adapter_path: str,
    output_path: str,
    model_name: str,
    max_length: int,
    max_completion_length: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    export_gguf: str | None,
) -> None:
    """Run on-policy distillation using TRL's DistillationTrainer."""
    # Load student (SFT adapter) via Unsloth for optimized training
    device_map = select_device_map()
    log.info(
        "Loading student from adapter %s (base: %s, device_map=%s)",
        adapter_path,
        model_name,
        device_map,
    )
    student, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map=device_map,
    )
    # Unsloth returns a Qwen3VLProcessor for Qwen3.5 models, but
    # DistillationTrainer needs a tokenizer with get_vocab(). Extract
    # the inner tokenizer when wrapped in a processor.
    if hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "get_vocab"):
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Qwen3.5's composite config stores vocab_size on text_config, but
    # DistillationTrainer expects model.config.vocab_size directly.
    if not hasattr(student.config, "vocab_size") and hasattr(
        student.config, "text_config"
    ):
        student.config.vocab_size = student.config.text_config.vocab_size

    # No separate teacher model is loaded: self-distillation uses the
    # student's own base weights with LoRA adapters temporarily disabled
    # (see _UnslothDistillationTrainer._get_teacher_logits).

    # Load dataset
    log.info("Loading dataset from %s", dataset_path)
    loaded = load_from_disk(dataset_path)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(loaded).__name__}")
    dataset: Dataset = loaded
    log.info("Loaded %d prompts", len(dataset))

    # DistillationConfig extends TrainingArguments with distillation params.
    # No teacher_model_name_or_path is set: the teacher is the student's
    # own base weights with LoRA disabled (weight sharing).
    config = DistillationConfig(
        output_dir=output_path,
        # Training schedule (on-policy-distillation.md)
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        # Distillation: fully on-policy with full-vocabulary reverse KL.
        # loss_top_k=0 computes exact KL over the entire vocabulary rather
        # than TRL's default sparse top-1 approximation.
        lmbda=1.0,
        beta=1.0,
        loss_top_k=0,
        max_length=max_length,
        max_completion_length=max_completion_length,
        temperature=1.0,
        # Logging
        logging_steps=logging_steps,
        report_to="mlflow",
        save_strategy="no",
        seed=3407,
    )

    trainer = _UnslothDistillationTrainer(
        model=student,
        teacher_model=None,  # ty: ignore[invalid-argument-type]
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # MLflow setup — respect MLFLOW_TRACKING_URI from the environment
    # (e.g. set by aws_train.sh to point at the centralized server)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow_dir = Path(output_path).resolve().parent / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
    set_experiment("uno-distillation")

    log.info(
        "Starting distillation: %d steps, batch=%d, GA=%d, lr=%g",
        max_steps,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
    )

    with start_run(run_name=f"distill-{model_name.split('/')[-1]}"):
        log_params(
            {
                "model": model_name,
                "adapter": adapter_path,
                "max_length": max_length,
                "max_completion_length": max_completion_length,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_steps": max_steps,
                "dataset_size": len(dataset),
                "lmbda": 1.0,
                "beta": 1.0,
            }
        )
        trainer.train()

    # Save the LoRA adapter. PEFT's save_pretrained loads a fresh
    # Qwen3_5Config and accesses .vocab_size which doesn't exist on the
    # multimodal config. Work around by temporarily patching the config
    # class so from_pretrained returns a config with vocab_size.
    log.info("Saving adapter to %s", output_path)
    _save_peft_adapter(student, tokenizer, output_path)

    if export_gguf:
        gguf_dir = f"{output_path}-gguf"
        log.info("Exporting GGUF (%s) to %s", export_gguf, gguf_dir)
        student.save_pretrained_gguf(
            gguf_dir, tokenizer, quantization_method=export_gguf
        )

    console.print("\n[bold green]Distillation complete.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-policy distillation for behavior recovery after SFT"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/distillation/prompts",
        help="Path to HF Dataset directory with distillation prompts",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="output/sft/lora_adapter",
        help="Path to the SFT LoRA adapter (student)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/distillation/lora_adapter",
        help="Output directory for the distilled adapter",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen3.5-4B",
        help="Base model name (teacher for self-distillation)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum total sequence length (prompt + completion)",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Maximum completion length during on-policy generation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: model-aware, see training/__init__.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log every N steps",
    )
    parser.add_argument(
        "--export-gguf",
        type=str,
        default=None,
        help="GGUF quantization method (e.g. q4_k_m). Omit to skip.",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Uno On-Policy Distillation[/bold cyan]\n")

    lr = args.lr if args.lr is not None else get_config(args.model).distill_lr
    log.info("Learning rate: %g (model=%s)", lr, args.model)

    run_distillation(
        dataset_path=args.dataset,
        adapter_path=args.adapter,
        output_path=args.output,
        model_name=args.model,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        learning_rate=lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        export_gguf=args.export_gguf,
    )


if __name__ == "__main__":
    main()
