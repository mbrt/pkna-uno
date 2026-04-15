#!/usr/bin/env python3

"""SFT training script for Uno personality fine-tuning.

Uses Unsloth + LoRA to fine-tune a Qwen3.5 model on the assembled SFT
dataset. Hyperparameters follow docs/fine-tuning/training-strategy.md.

Requires a GPU with sufficient VRAM (see training-strategy.md for estimates).

Usage:
    python training/run_sft.py \
        --dataset output/sft/dataset \
        --output output/sft/lora_adapter \
        --model Qwen/Qwen3.5-4B

    # With GGUF export:
    python training/run_sft.py \
        --dataset output/sft/dataset \
        --output output/sft/lora_adapter \
        --export-gguf q4_k_m
"""

# Unsloth must be imported before transformers/trl/datasets to monkey-patch
# optimizations. If imported later, training runs slower and uses more VRAM.
# Callers that import this module after transformers is already loaded must
# call training.ensure_unsloth() first to guarantee correct import order.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
import logging
from pathlib import Path

from datasets import Dataset, load_from_disk
from mlflow import log_params, set_experiment, set_tracking_uri, start_run
from rich.console import Console
from rich.logging import RichHandler
from trl import SFTConfig, SFTTrainer

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    force=True,
)
log = logging.getLogger(__name__)

# LoRA config from training-strategy.md
LORA_RANK = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0

# Qwen3.5 chat template boundary tokens for loss masking.
# Tool results are wrapped inside <|im_start|>user blocks by the Qwen3.5
# template, so they are automatically masked by the user instruction pattern.
# The system prompt precedes any assistant turn and is masked by default.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def run_sft(
    dataset_path: str,
    output_path: str,
    model_name: str,
    max_seq_length: int,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    weight_decay: float,
    max_steps: int,
    logging_steps: int,
    export_gguf: str | None,
) -> None:
    """Run SFT training."""
    log.info("Loading model %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    log.info(
        "Applying LoRA (rank=%d, alpha=%d, target=all-linear)", LORA_RANK, LORA_ALPHA
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    log.info("Loading dataset from %s", dataset_path)
    loaded = load_from_disk(dataset_path)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(loaded).__name__}")
    dataset: Dataset = loaded

    training_args = SFTConfig(
        max_seq_length=max_seq_length,  # ty: ignore[unknown-argument]  # added by unsloth
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        output_dir=output_path,
        save_strategy="epoch",
        seed=3407,
        bf16=True,
        report_to="mlflow",
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    log.info("Applying loss masking (train on assistant responses only)")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    log.info("Starting training")
    mlflow_dir = Path(output_path).resolve().parent / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
    set_experiment("uno-sft")
    with start_run(run_name=f"sft-{model_name.split('/')[-1]}"):
        log_params(
            {
                "model": model_name,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "max_seq_length": max_seq_length,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "dataset_size": len(dataset),
            }
        )
        trainer.train()

    log.info("Saving LoRA adapter to %s", output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if export_gguf:
        gguf_dir = f"{output_path}-gguf"
        log.info("Exporting GGUF (%s) to %s", export_gguf, gguf_dir)
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=export_gguf)

    console.print("\n[bold green]Training complete.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT training for Uno personality fine-tuning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/sft/dataset",
        help="Path to assembled HF Dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/sft/lora_adapter",
        help="Output directory for the LoRA adapter",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
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
        help="Gradient accumulation steps (effective batch = batch_size * GA)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for epoch-based)",
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
        help="GGUF quantization method (e.g. q4_k_m, q8_0, f16). Omit to skip.",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Uno SFT Training[/bold cyan]\n")

    run_sft(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        export_gguf=args.export_gguf,
    )


if __name__ == "__main__":
    main()
