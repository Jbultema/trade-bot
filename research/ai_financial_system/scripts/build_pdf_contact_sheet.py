#!/usr/bin/env python3
"""Build a deterministic multi-page contact sheet from PDF inspection rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

PAGE_WIDTH, PAGE_HEIGHT = LETTER
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_NAME = "AFSContact"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    images: list[Path],
    output: Path,
    *,
    title: str,
    columns: int,
    rows: int,
) -> dict[str, object]:
    if not images:
        raise ValueError("contact sheet requires at least one page image")
    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")
    font_path = Path(FONT_PATH)
    if not font_path.is_file():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(
        str(output),
        pagesize=LETTER,
        invariant=1,
        initialFontName=FONT_NAME,
        initialFontSize=7,
        initialLeading=9,
    )
    canvas.setTitle(title)
    canvas.setAuthor("AI Financial System Research Project")
    canvas.setSubject("Every-page visual inspection contact sheet")

    margin_x = 24.0
    margin_bottom = 24.0
    header = 28.0
    gutter = 12.0
    cell_width = (PAGE_WIDTH - 2 * margin_x - (columns - 1) * gutter) / columns
    cell_height = (
        PAGE_HEIGHT - margin_bottom - header - (rows - 1) * gutter
    ) / rows
    slots = columns * rows

    for index, image_path in enumerate(images):
        slot = index % slots
        if slot == 0:
            canvas.setFont(FONT_NAME, 8)
            canvas.drawString(margin_x, PAGE_HEIGHT - 18, title[:100])
        column = slot % columns
        row = slot // columns
        x = margin_x + column * (cell_width + gutter)
        y_top = PAGE_HEIGHT - header - row * (cell_height + gutter)
        label_height = 12.0
        canvas.setFont(FONT_NAME, 7)
        canvas.drawString(x, y_top - 8, f"Source page {index + 1}")
        canvas.drawImage(
            str(image_path),
            x,
            y_top - cell_height,
            width=cell_width,
            height=cell_height - label_height,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
        canvas.setStrokeColorRGB(0.75, 0.78, 0.82)
        canvas.rect(
            x,
            y_top - cell_height,
            cell_width,
            cell_height - label_height,
            stroke=1,
            fill=0,
        )
        if slot == slots - 1 or index == len(images) - 1:
            canvas.showPage()
    canvas.save()

    return {
        "status": "pass",
        "title": title,
        "output": str(output),
        "output_sha256": sha256(output),
        "source_pages": len(images),
        "contact_sheet_pages": math.ceil(len(images) / slots),
        "columns": columns,
        "rows": rows,
        "sources": [
            {
                "page_number": index,
                "path": str(path),
                "sha256": sha256(path),
            }
            for index, path in enumerate(images, start=1)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--rows", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = sorted(args.image_directory.glob("page_*.png"))
    result = build(
        images,
        args.output,
        title=args.title,
        columns=args.columns,
        rows=args.rows,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
