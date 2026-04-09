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
| `finetune/` | Fine-tuning pipeline (WIP): dataset generation, training, evaluation |
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
