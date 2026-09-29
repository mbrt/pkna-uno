#!/usr/bin/env python3
"""Tools that extraction agents run from their workspace.

zoom: crop a region of a page and enlarge it, so balloon outlines and tails
can be inspected. validate: check a written page file against the schema.
validate-cast: check an issue's character resolution against its labels.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from pkna.extract.observations import ObservedPage, page_problems
from pkna.extract.registry import IssueCast, cast_problems

# Long side of a zoomed crop. Larger images are downscaled by the model API.
ZOOM_SIZE = 1568
MAX_ZOOM = 4.0


def zoom(
    page_path: Path, box: tuple[float, float, float, float], out_dir: Path
) -> Path:
    """Save an enlarged crop of the page. The box is in fractions of page size."""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(
            f"Box must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {box}"
        )
    with Image.open(page_path) as page:
        w, h = page.size
        crop = page.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    scale = min(MAX_ZOOM, ZOOM_SIZE / max(crop.size))
    if scale > 1:
        size = (round(crop.width * scale), round(crop.height * scale))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{page_path.stem}_{x0:.2f}_{y0:.2f}_{x1:.2f}_{y1:.2f}.png"
    crop.convert("RGB").save(out)
    return out


def validate(path: Path) -> list[str]:
    """Problems with a page file; empty when it is valid."""
    try:
        page = ObservedPage.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as e:
        return [str(e)]
    return page_problems(page)


def validate_cast(path: Path, mentions_path: Path) -> list[str]:
    """Problems with a character resolution file; empty when it is valid."""
    try:
        cast = IssueCast.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as e:
        return [str(e)]
    mentions = json.loads(mentions_path.read_text(encoding="utf-8"))
    return cast_problems(cast, {m["label"] for m in mentions})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tools for observation extraction agents"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    zoom_parser = sub.add_parser("zoom", help="Enlarge a region of a page")
    zoom_parser.add_argument("page", type=Path)
    zoom_parser.add_argument(
        "box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1")
    )
    zoom_parser.add_argument("--out-dir", type=Path, default=Path("zoom"))
    validate_parser = sub.add_parser("validate", help="Check a page file")
    validate_parser.add_argument("path", type=Path)
    cast_parser = sub.add_parser("validate-cast", help="Check a character resolution")
    cast_parser.add_argument("path", type=Path)
    cast_parser.add_argument("--mentions", type=Path, default=Path("mentions.json"))
    args = parser.parse_args()

    if args.command == "zoom":
        x0, y0, x1, y1 = args.box
        print(zoom(args.page, (x0, y0, x1, y1), args.out_dir))
        return
    if args.command == "validate-cast":
        problems = validate_cast(args.path, args.mentions)
    else:
        problems = validate(args.path)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK: {args.path}")


if __name__ == "__main__":
    main()
