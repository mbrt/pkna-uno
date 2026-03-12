# PKNA Uno

An unordered set of experiments to extract structured data from the Italian
comic book series [PKNA](https://en.wikipedia.org/wiki/PKNA), with the goal of
creating an LLM that impersonates the main AI character:
[Uno](https://disney.fandom.com/wiki/Uno).

This is not intended to be run by anyone directly, but just serve as inspiration
for similar projects. Feel free to point your coding agent here and ask it to
generate your version under your constraints (input type, desired character,
etc.). Because of how easy it is to fork and customize anything today, I didn't
really bother generalizing, and focused instead on getting end results.

You can definitely see it in the low code quality and its lack of structure.

## Requirements

> [!NOTE]
> The scripts and notebooks are all mostly self-contained but require scans of
> the comic to be present in `./input/pkna`. For obvious copyright reasons,
> these are not released with this project. I have those because I own all the
> original issues and I just patiently digitalized them.

This project uses [uv](https://astral.sh/uv) to manage the Python environment:

```sh
uv sync
uv run script_x.py
```

## Results

A walkthrough of the code and results is in [this blog](https://blog.mbrt.dev).
Intermediate results are not provided to avoid copyright problems. I did however
include some of the final results, as I believe are different enough from source
material to be considered original work (not a lawyer, so don't quote me on
that).

You can find:

* Uno's [soul document](results/uno_soul_document.md)
* [Ledger](results/final_ledger.json) with structured observations on Uno's
  behavior
* [Refined ledger](results/refined_ledger.json) to compensate for contradictions
* [Rephrased wiki](results/wiki) with factual information on the fictional
  universe of the comic, rephrased _as if it was narrated within it_. The
  original language (Italian) was preserved and not translated to maximize
  fidelity.

## License

Code and soul document are licensed under the [Apache 2.0](LICENSE), and
rephrased content follows the original
[Fandom license](https://www.fandom.com/licensing)
[CC BY-SA](results/wiki/LICENSE).
