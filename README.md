# PKNA Uno

ML pipeline for extracting structured data from the Italian comic book series
[PKNA](https://en.wikipedia.org/wiki/PKNA) and fine-tuning a small model to
impersonate the AI character [Uno](https://disney.fandom.com/wiki/Uno).

A walkthrough of the project is on [this blog](https://blog.mbrt.dev/posts/uno).

## Structure

| Directory | Contents |
|---|---|
| `pkna/` | Shared library: LLM backends, scene extraction, wiki tools |
| `extract/` | Active pipeline: panel extraction, scene reflection, emotional profile building |
| `datagen/` | Dataset generation: prompt bank, trace capture, quality filtering |
| `training/` | SFT training: dataset assembly, Unsloth training script, smoke test |
| `evals/` | Evaluation: prompt generation, inference, scoring |
| `data/` | Static data for fine-tuning (prompts, rubrics, profiles) |
| `docs/` | Design documents |
| `tests/` | Unit tests |
| `experimental/` | Archived one-shot scripts and notebooks from earlier exploration |
| `results/` | Published outputs (soul document, ledger, wiki) |

## Requirements

> [!NOTE]
> Scripts require comic scans in `./input/pkna`. For copyright reasons these are
> not included.

```sh
uv sync
make test
```

## SFT Training Smoke Test

Validate the full SFT pipeline locally with synthetic data (no upstream
traces or LLM API keys needed):

```sh
# Full pipeline -- generates traces, assembles dataset, trains 10 steps
# Requires a GPU (~3 GB VRAM with Qwen3.5-0.8B)
python training/smoke_test.py

# Dataset assembly only (CPU, no GPU needed)
python training/smoke_test.py --assemble-only
```

Output goes to `output/sft/smoke_test/` (traces, HF dataset, LoRA adapter,
MLflow logs).

## Results

* Uno's [soul document](results/uno_soul_document.md)
* [Ledger](results/final_ledger.json) with structured observations on Uno's
  behavior
* [Refined ledger](results/refined_ledger.json) to compensate for contradictions
* [Rephrased wiki](results/wiki) with factual information on the fictional
  universe of the comic, rephrased _as if it was narrated within it_

## License

Code and soul document are licensed under [Apache 2.0](LICENSE). Rephrased
wiki content follows the original [Fandom license](https://www.fandom.com/licensing)
([CC BY-SA](results/wiki/LICENSE)).
