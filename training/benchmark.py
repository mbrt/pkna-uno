#!/usr/bin/env python3

"""GPU performance benchmark for inference and training.

Detects the available GPU, selects the largest Qwen3.5 model that fits,
and measures inference throughput (tok/s) and training throughput
(steps/s). Results are compared against expectations from the design
docs (docs/fine-tuning/infra-costs.md, docs/fine-tuning/model-selection.md).

Works on both the development laptop (RTX 2000 Ada, 8 GB) and target
cloud hardware (1x or 4x L40S).

Usage:
    python training/benchmark.py                          # auto-detect
    python training/benchmark.py --inference-only
    python training/benchmark.py --training-only
    python training/benchmark.py --model Qwen/Qwen3.5-4B  # override
    python training/benchmark.py --steps 50
"""

from unsloth import FastLanguageModel

import argparse
import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import cast

import torch
from datasets import Dataset
from rich.table import Table
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from pkna.logging import setup_logging

console, log = setup_logging()
logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

VRAM_MB_8GB = 8 * 1024
VRAM_MB_48GB = 48 * 1024
VRAM_MB_192GB = 192 * 1024


class HardwareClass(Enum):
    SMALL_GPU = "small_gpu"
    SINGLE_L40S = "single_l40s"
    QUAD_L40S = "quad_l40s"


@dataclass
class HardwareProfile:
    gpu_name: str
    total_vram_mb: int
    hw_class: HardwareClass


def detect_hardware() -> HardwareProfile:
    """Detect GPU and classify into a hardware tier."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

    if total_vram_mb >= VRAM_MB_192GB * 0.8:
        hw_class = HardwareClass.QUAD_L40S
    elif total_vram_mb >= VRAM_MB_48GB * 0.8:
        hw_class = HardwareClass.SINGLE_L40S
    else:
        hw_class = HardwareClass.SMALL_GPU

    return HardwareProfile(
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
        hw_class=hw_class,
    )


# ---------------------------------------------------------------------------
# Model selection (from docs/fine-tuning/model-selection.md VRAM tables)
# ---------------------------------------------------------------------------

# Maps hardware class to (inference_model, training_model).
# Inference uses BF16 (same as training pipeline).
# Training uses BF16 LoRA via Unsloth.
MODEL_MAP: dict[HardwareClass, tuple[str, str]] = {
    # Laptop 8 GB: 0.8B fits BF16 inference (~2 GB) and LoRA train (~3 GB)
    HardwareClass.SMALL_GPU: ("unsloth/Qwen3.5-0.8B", "unsloth/Qwen3.5-0.8B"),
    # 1xL40S 48 GB: 4B fits BF16 inference (~8 GB) and LoRA train (~10 GB)
    HardwareClass.SINGLE_L40S: ("unsloth/Qwen3.5-4B", "unsloth/Qwen3.5-4B"),
    # 4xL40S 192 GB: 35B-A3B MoE LoRA train (~74 GB)
    HardwareClass.QUAD_L40S: ("unsloth/Qwen3.6-35B-A3B", "unsloth/Qwen3.6-35B-A3B"),
}


def select_models(hw: HardwareProfile, override: str | None) -> tuple[str, str]:
    """Return (inference_model, training_model) for the hardware."""
    if override:
        return override, override
    return MODEL_MAP[hw.hw_class]


# ---------------------------------------------------------------------------
# Expected performance baselines
# ---------------------------------------------------------------------------

# Derived from:
# - infra-costs.md: SFT durations per model/hardware
# - model-selection.md: VRAM tables, community inference speeds
# - training-strategy.md: batch size, seq length, GA settings
#
# Training throughput estimates assume batch_size=1, GA=1, max_seq_length=2048
# (shorter than production 8192 to keep benchmark fast). Values are
# conservative lower bounds.


@dataclass
class ExpectedPerformance:
    """Minimum acceptable performance for a hardware/model combination."""

    # Inference: minimum tokens/second (BF16, greedy decoding)
    min_inference_tok_s: float
    # Training: maximum VRAM in MB during LoRA training
    max_training_vram_mb: int
    # Training: minimum steps/second
    min_training_steps_per_s: float


# Keyed by (HardwareClass, model_name) for specificity.
EXPECTED: dict[tuple[HardwareClass, str], ExpectedPerformance] = {
    # 0.8B on 8 GB laptop: lightweight, should be fast
    (HardwareClass.SMALL_GPU, "unsloth/Qwen3.5-0.8B"): ExpectedPerformance(
        min_inference_tok_s=30.0,
        max_training_vram_mb=6 * 1024,  # ~3 GB model + overhead
        min_training_steps_per_s=0.5,
    ),
    # 4B on 1xL40S
    (HardwareClass.SINGLE_L40S, "unsloth/Qwen3.5-4B"): ExpectedPerformance(
        min_inference_tok_s=20.0,
        max_training_vram_mb=16 * 1024,  # ~10 GB LoRA + overhead
        min_training_steps_per_s=0.3,
    ),
    # 35B-A3B MoE on 4xL40S
    (HardwareClass.QUAD_L40S, "unsloth/Qwen3.6-35B-A3B"): ExpectedPerformance(
        min_inference_tok_s=8.0,
        max_training_vram_mb=80 * 1024,  # ~74 GB LoRA + overhead
        min_training_steps_per_s=0.1,
    ),
}


def get_expected(hw: HardwareProfile, model_name: str) -> ExpectedPerformance | None:
    return EXPECTED.get((hw.hw_class, model_name))


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

BENCH_PROMPT = [
    {"role": "user", "content": "Ciao Uno! Come stai oggi?"},
]

N_GENERATE_TOKENS = 256
N_WARMUP = 2
N_TIMED = 5


@dataclass
class InferenceResult:
    model_name: str
    tok_per_s: float
    ttft_s: float
    peak_vram_mb: int
    generated_tokens: int


def benchmark_inference(
    model_name: str,
    max_seq_length: int = 2048,
) -> InferenceResult:
    """Benchmark inference throughput with Unsloth."""
    log.info("Loading %s for inference", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)

    encoder = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    if encoder.pad_token_id is None:
        encoder.pad_token_id = encoder.eos_token_id

    prompt_text = cast(
        str,
        tokenizer.apply_chat_template(
            BENCH_PROMPT,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ),
    )
    encoded = encoder(prompt_text, return_tensors="pt", return_attention_mask=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pad_token_id": encoder.pad_token_id,
        "do_sample": False,
    }

    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model.generate(**gen_kwargs, max_new_tokens=32)

    # Timed runs
    timings: list[float] = []
    ttfts: list[float] = []
    total_new_tokens = 0

    for _ in range(N_TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model.generate(**gen_kwargs, max_new_tokens=1)
            torch.cuda.synchronize()
            t_first = time.perf_counter()
            ttfts.append(t_first - t0)

            out_full = model.generate(**gen_kwargs, max_new_tokens=N_GENERATE_TOKENS)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        new_tokens = out_full.shape[1] - prompt_len
        total_new_tokens += new_tokens
        timings.append(t1 - t0)

    peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
    avg_new_tokens = total_new_tokens / N_TIMED
    med_time = median(timings)
    tok_per_s = avg_new_tokens / med_time if med_time > 0 else 0.0

    # Cleanup
    del model, tokenizer, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return InferenceResult(
        model_name=model_name,
        tok_per_s=tok_per_s,
        ttft_s=median(ttfts),
        peak_vram_mb=peak_vram_mb,
        generated_tokens=int(avg_new_tokens),
    )


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------

LORA_RANK = 64
LORA_ALPHA = 32

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

SYNTHETIC_EXAMPLES = [
    [
        {"role": "user", "content": f"Domanda di test numero {i}. Puoi aiutarmi?"},
        {
            "role": "assistant",
            "reasoning_content": "L'utente chiede aiuto. Rispondo in modo amichevole.",
            "content": (
                f"Certo, socio! Sono sempre pronto ad aiutarti. "
                f"Questa è la risposta numero {i}, elaborata con cura."
            ),
        },
    ]
    for i in range(32)
]


@dataclass
class TrainingResult:
    model_name: str
    steps_per_s: float
    samples_per_s: float
    peak_vram_mb: int
    total_steps: int
    total_time_s: float


def benchmark_training(
    model_name: str,
    n_steps: int = 20,
    max_seq_length: int = 2048,
) -> TrainingResult:
    """Benchmark LoRA training throughput with Unsloth + SFTTrainer."""
    log.info("Loading %s for training", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    log.info("Applying LoRA (rank=%d, alpha=%d)", LORA_RANK, LORA_ALPHA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    encoder = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    texts: list[str] = []
    for messages in SYNTHETIC_EXAMPLES:
        text = cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            ),
        )
        n_tokens = len(encoder.encode(text))
        if n_tokens <= max_seq_length:
            texts.append(text)

    dataset = Dataset.from_dict({"text": texts})
    log.info("Synthetic dataset: %d examples", len(dataset))

    training_args = SFTConfig(
        max_seq_length=max_seq_length,  # ty: ignore[unknown-argument]
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=n_steps,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_strategy="no",
        output_dir="/tmp/benchmark_training",
        save_strategy="no",
        seed=3407,
        bf16=True,
        report_to="none",
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    trainer.train()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
    steps_per_s = n_steps / total_time if total_time > 0 else 0.0
    samples_per_s = steps_per_s  # batch_size=1, GA=1

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return TrainingResult(
        model_name=model_name,
        steps_per_s=steps_per_s,
        samples_per_s=samples_per_s,
        peak_vram_mb=peak_vram_mb,
        total_steps=n_steps,
        total_time_s=total_time,
    )


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------


def _status(measured: float, threshold: float, higher_is_better: bool) -> str:
    """Return PASS / WARN / FAIL based on how measured compares to threshold."""
    if higher_is_better:
        ratio = measured / threshold if threshold > 0 else float("inf")
        if ratio >= 1.0:
            return "[green]PASS[/green]"
        if ratio >= 0.5:
            return "[yellow]WARN[/yellow]"
        return "[red]FAIL[/red]"
    else:
        ratio = threshold / measured if measured > 0 else float("inf")
        if ratio >= 1.0:
            return "[green]PASS[/green]"
        if ratio >= 0.5:
            return "[yellow]WARN[/yellow]"
        return "[red]FAIL[/red]"


def print_results(
    hw: HardwareProfile,
    infer: InferenceResult | None,
    train: TrainingResult | None,
) -> None:
    """Print benchmark results with comparison to expectations."""
    # Hardware info
    hw_table = Table(title="Hardware")
    hw_table.add_column("Property", style="cyan")
    hw_table.add_column("Value", style="green")
    hw_table.add_row("GPU", hw.gpu_name)
    hw_table.add_row("Total VRAM", f"{hw.total_vram_mb} MB")
    hw_table.add_row("Hardware Class", hw.hw_class.value)
    console.print(hw_table)
    console.print()

    if infer:
        model_name = infer.model_name
    elif train:
        model_name = train.model_name
    else:
        model_name = ""
    expected = get_expected(hw, model_name)

    # Results table
    results = Table(title="Benchmark Results")
    results.add_column("Metric", style="cyan")
    results.add_column("Measured", justify="right")
    results.add_column("Expected", justify="right")
    results.add_column("Status", justify="center")

    if infer:
        exp_tok_s = (
            f">= {expected.min_inference_tok_s:.0f} tok/s" if expected else "N/A"
        )
        status = (
            _status(infer.tok_per_s, expected.min_inference_tok_s, True)
            if expected
            else "N/A"
        )
        results.add_row(
            f"Inference tok/s ({infer.model_name})",
            f"{infer.tok_per_s:.1f} tok/s",
            exp_tok_s,
            status,
        )
        results.add_row(
            "Time to first token",
            f"{infer.ttft_s * 1000:.0f} ms",
            "",
            "",
        )
        results.add_row(
            "Inference peak VRAM",
            f"{infer.peak_vram_mb} MB",
            f"<= {hw.total_vram_mb} MB",
            _status(hw.total_vram_mb, infer.peak_vram_mb, True),
        )
        results.add_row(
            "Generated tokens",
            str(infer.generated_tokens),
            str(N_GENERATE_TOKENS),
            "",
        )

    if train:
        exp_steps_s = (
            f">= {expected.min_training_steps_per_s:.2f}" if expected else "N/A"
        )
        status = (
            _status(train.steps_per_s, expected.min_training_steps_per_s, True)
            if expected
            else "N/A"
        )
        results.add_row(
            f"Training steps/s ({train.model_name})",
            f"{train.steps_per_s:.3f}",
            exp_steps_s,
            status,
        )
        results.add_row(
            "Training time",
            f"{train.total_time_s:.1f}s ({train.total_steps} steps)",
            "",
            "",
        )

        exp_vram = f"<= {expected.max_training_vram_mb} MB" if expected else "N/A"
        vram_status = (
            _status(expected.max_training_vram_mb, train.peak_vram_mb, True)
            if expected
            else "N/A"
        )
        results.add_row(
            "Training peak VRAM",
            f"{train.peak_vram_mb} MB",
            exp_vram,
            vram_status,
        )

    console.print(results)

    if not expected:
        console.print(
            f"\n[yellow]No expected baselines for ({hw.hw_class.value}, {model_name}). "
            "Results are informational only.[/yellow]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU performance benchmark for inference and training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model selection (default: auto-detect based on VRAM)",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference benchmark only",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Run training benchmark only",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of training steps (default: %(default)s)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]GPU Performance Benchmark[/bold cyan]\n")

    hw = detect_hardware()
    log.info(
        "Detected: %s (%d MB VRAM, class=%s)",
        hw.gpu_name,
        hw.total_vram_mb,
        hw.hw_class.value,
    )

    infer_model, train_model = select_models(hw, args.model)

    run_inference = not args.training_only
    run_training = not args.inference_only

    infer_result: InferenceResult | None = None
    train_result: TrainingResult | None = None

    if run_inference:
        console.rule("[bold]Inference Benchmark")
        log.info("Model: %s", infer_model)
        infer_result = benchmark_inference(infer_model)
        log.info(
            "Inference: %.1f tok/s, TTFT %.0f ms, peak %d MB",
            infer_result.tok_per_s,
            infer_result.ttft_s * 1000,
            infer_result.peak_vram_mb,
        )

    if run_training:
        console.rule("[bold]Training Benchmark")
        log.info("Model: %s, steps: %d", train_model, args.steps)
        train_result = benchmark_training(train_model, n_steps=args.steps)
        log.info(
            "Training: %.3f steps/s, peak %d MB, total %.1fs",
            train_result.steps_per_s,
            train_result.peak_vram_mb,
            train_result.total_time_s,
        )

    console.rule("[bold]Summary")
    print_results(hw, infer_result, train_result)
    console.print("\n[bold green]Benchmark complete.[/bold green]")


if __name__ == "__main__":
    main()
