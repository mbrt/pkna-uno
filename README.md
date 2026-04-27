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

To automatically upload trained adapters to HuggingFace Hub, store a
[write-access token](https://huggingface.co/settings/tokens) in SSM Parameter
Store. The `--region` must match the region you deploy the stack to. This is
optional -- without it, adapters are still uploaded to S3.

```sh
aws ssm put-parameter \
  --name /pkna-uno/hf-token \
  --type SecureString \
  --value "hf_YOUR_TOKEN" \
  --region us-east-1
```

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

## Evals

Run the eval pipeline against a trained model. Three stages: generate prompts
(deterministic), run inference, score traces with a judge model.

```sh
# Full run with a cloud backend
./scripts/run_evals.sh --backend gemini --model gemini-3-flash

# Local model (Unsloth, slow for multi-turn)
./scripts/run_evals.sh --backend local --model output/sft/lora_adapter --4bit

# Quick sanity check (5 prompts per suite)
./scripts/run_evals.sh --mini --backend vllm --model my-model
```

### vLLM backend (recommended for local models)

The `vllm` backend connects to an external vLLM server, which handles KV cache,
PagedAttention, and continuous batching -- much faster than the `local` backend,
especially for multi-turn suites.

#### Merging the LoRA adapter

Merge the LoRA adapter into the base model first. This avoids the extra memory
overhead of serving LoRA at runtime (which often causes OOM on 8 GB GPUs):

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "output/sft/qwen3-5-4b/output/sft/lora_adapter",
    max_seq_length=4096,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
)
model.save_pretrained_merged(
    "output/sft/qwen3-5-4b-merged",
    tokenizer,
    save_method="merged_16bit",
)
```

#### Starting the server

Start the server with reasoning and tool-call parsing enabled:

```sh
uv tool run --from vllm vllm serve /path/to/merged-model \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto
```

On GPUs with limited memory (e.g. 8 GB), add `--quantization bitsandbytes` for
on-the-fly 4-bit quantization and `--enforce-eager` to disable CUDA graphs
(which reserve significant additional memory):

```sh
uv tool run --from vllm --with 'bitsandbytes>=0.49.2' \
    vllm serve /path/to/merged-model \
    --quantization bitsandbytes \
    --max-model-len 4096 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto
```

> [!NOTE]
> On 8 GB GPUs, serving LoRA adapters on-the-fly with `--enable-lora` typically
> OOMs because the base model + LoRA weights + KV cache don't fit. Merge the
> adapter first (see above) and serve the merged model instead.

Then run evals:

```sh
./scripts/run_evals.sh --backend vllm --model /path/to/merged-model

# Custom server URL
VLLM_BASE_URL=http://gpu-box:8000/v1 \
    ./scripts/run_evals.sh --backend vllm --model my-model
```

### Resuming interrupted runs

```sh
./scripts/run_evals.sh --resume output/evals/run-lora-adapter-20260427-193000 \
    --model output/sft/lora_adapter --backend local
```

Each stage skips already-processed items, so the run continues from where it was
interrupted.

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
