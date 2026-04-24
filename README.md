# PKNA Uno

ML pipeline for extracting structured data from the Italian comic book series
[PKNA](https://en.wikipedia.org/wiki/PKNA) and fine-tuning a small model to
impersonate the AI character [Uno](https://disney.fandom.com/wiki/Uno).

A walkthrough of the project is on [this blog](https://blog.mbrt.dev/posts/uno).

## Structure

| Directory | Contents |
|---|---|
| `pkna/` | Shared library organized by phase: `llm/` (backends, test doubles), `extract/` (scenes, wiki), `inference/` (prompts, memory, tools), `datagen/` (types), `eval/` (types), `training/` (SFT dataset) |
| `extract/` | Active pipeline: panel extraction, scene reflection, emotional profile building |
| `datagen/` | Dataset generation: prompt bank, trace capture, quality filtering |
| `training/` | SFT training: dataset assembly, Unsloth training script, smoke test |
| `evals/` | Evaluation: prompt generation, inference, scoring |
| `data/` | Static data for fine-tuning (prompts, rubrics, profiles) |
| `docs/` | Design documents |
| `tests/` | Unit tests |
| `infra/` | CloudFormation stack and launcher script for AWS GPU training |
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

## Pipeline Smoke Test

Validate the full pipeline locally with fake LLM backends (no API keys
needed). Stages 1-4 and 6-7 run on CPU; only stage 5 (training) needs a GPU.

```sh
# All stages except training (no GPU needed):
uv run python training/smoke_test.py --no-training

# Full pipeline including training (~3 GB VRAM with Qwen3.5-0.8B):
uv run python training/smoke_test.py --all

# Run a single stage:
uv run python training/smoke_test.py --stage prompts
```

Stages: prompts, datagen, filter, assemble, train, eval.
Output goes to `output/sft/smoke_test/`.

## Training on AWS

Run SFT + on-policy distillation on an AWS GPU instance. The stack provisions an
EC2 instance, trains the model, uploads results to S3, and self-terminates.
Requires the [AWS CLI](https://aws.amazon.com/cli/) and the [Session Manager
plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).

```sh
# Default: Qwen3.5-4B on 1x L40S (~$15, ~3 hours)
./infra/launch-training.sh

# Small model for quick iteration
./infra/launch-training.sh --model unsloth/Qwen3.5-0.8B

# MoE model on 4x L40S (~$84, ~9 hours)
./infra/launch-training.sh --model unsloth/Qwen3.6-35B-A3B --instance-type g6e.12xlarge

# SFT only (skip distillation), export GGUF
./infra/launch-training.sh --sft-only --export-gguf q4_k_m
```

Monitor training live (all commands are printed by the launcher with the
instance ID filled in):

```sh
# Follow training logs
aws ssm start-session --target <instance-id> --region <region>
# then: tail -f /home/ubuntu/training.log

# MLflow UI for loss curves and metrics (port-forwarded through SSM)
aws ssm start-session --target <instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters portNumber=5000,localPortNumber=5000 \
  --region <region>
# Then browse http://localhost:5000
```

## Results

* Uno's [soul document](results/uno_soul_document.md) (condensed for datagen)
* Uno's [full soul document](results/uno_soul_document_full.md)
* [Ledger](results/ledger_raw.json) with structured observations on Uno's
  behavior (unfiltered)
* [Refined ledger](results/ledger_filtered.json) with consolidated observations
  and removed low-support claims
* [Rephrased wiki](results/wiki) with factual information on the fictional
  universe of the comic, rephrased _as if it was narrated within it_

## License

Code and soul document are licensed under [Apache 2.0](LICENSE). Rephrased
wiki content follows the original [Fandom license](https://www.fandom.com/licensing)
([CC BY-SA](results/wiki/LICENSE)).
