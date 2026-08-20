#!/usr/bin/env python3
"""Fail-closed structural audit for governed publication PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

EXPECTED_DATE = "D:20000101000000+00'00'"
FONT_FILE_KEYS = ("/FontFile", "/FontFile2", "/FontFile3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_outline(items: list[Any]) -> list[str]:
    titles: list[str] = []
    for item in items:
        if isinstance(item, list):
            titles.extend(flatten_outline(item))
        else:
            titles.append(str(getattr(item, "title", item)))
    return titles


def font_audit(reader: PdfReader) -> list[dict[str, Any]]:
    fonts: dict[tuple[str, str], dict[str, Any]] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        resources = page.get("/Resources", {}).get_object()
        for resource_name, reference in resources.get("/Font", {}).items():
            font = reference.get_object()
            descriptor_reference = font.get("/FontDescriptor")
            embedded_files: list[str] = []
            if descriptor_reference:
                descriptor = descriptor_reference.get_object()
                embedded_files = [key for key in FONT_FILE_KEYS if descriptor.get(key)]
            key = (str(resource_name), str(font.get("/BaseFont", "")))
            record = fonts.setdefault(
                key,
                {
                    "resource_name": key[0],
                    "base_font": key[1],
                    "embedded": bool(embedded_files),
                    "embedded_files": embedded_files,
                    "pages": [],
                },
            )
            record["pages"].append(page_number)
    return sorted(fonts.values(), key=lambda row: (row["resource_name"], row["base_font"]))


def audit(path: Path, *, expected_title: str | None, minimum_pages: int) -> dict[str, Any]:
    reader = PdfReader(path)
    metadata = reader.metadata or {}
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    outline = flatten_outline(reader.outline)
    fonts = font_audit(reader)
    errors: list[str] = []

    if reader.is_encrypted:
        errors.append("PDF is encrypted")
    if len(reader.pages) < minimum_pages:
        errors.append(f"page count {len(reader.pages)} is below required {minimum_pages}")
    if not outline:
        errors.append("PDF has no outline/bookmarks")
    if expected_title is not None and metadata.get("/Title") != expected_title:
        errors.append(
            f"title mismatch: expected {expected_title!r}, observed {metadata.get('/Title')!r}"
        )
    if metadata.get("/CreationDate") != EXPECTED_DATE:
        errors.append(f"nondeterministic creation date: {metadata.get('/CreationDate')!r}")
    if metadata.get("/ModDate") != EXPECTED_DATE:
        errors.append(f"nondeterministic modification date: {metadata.get('/ModDate')!r}")

    blank_pages = [index for index, text in enumerate(page_text, start=1) if not text]
    if blank_pages:
        errors.append(f"pages without searchable text: {blank_pages}")

    unembedded = [
        f"{row['resource_name']} {row['base_font']} pages={row['pages']}"
        for row in fonts
        if not row["embedded"]
    ]
    if unembedded:
        errors.append(f"unembedded fonts: {unembedded}")

    page_sizes: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        width = float(box.width)
        height = float(box.height)
        page_sizes.append({"page": index, "width_points": width, "height_points": height})
        if abs(width - 612.0) > 0.5 or abs(height - 792.0) > 0.5:
            errors.append(f"page {index} is not US Letter: {width} x {height}")

    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "page_count": len(reader.pages),
        "outline_count": len(outline),
        "outline_titles": outline,
        "metadata": {str(key): str(value) for key, value in metadata.items()},
        "searchable_characters_by_page": [len(text) for text in page_text],
        "blank_pages": blank_pages,
        "fonts": fonts,
        "all_fonts_embedded": not unembedded,
        "page_sizes": page_sizes,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-title")
    parser.add_argument("--minimum-pages", type=int, default=1)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    result = audit(
        args.input,
        expected_title=args.expected_title,
        minimum_pages=args.minimum_pages,
    )
    if args.compare is not None:
        if not args.compare.is_file():
            raise FileNotFoundError(args.compare)
        compare_hash = sha256(args.compare)
        result["deterministic_comparison"] = {
            "path": str(args.compare),
            "sha256": compare_hash,
            "byte_identical": compare_hash == result["sha256"],
        }
        if compare_hash != result["sha256"]:
            result["errors"].append("comparison PDF is not byte-identical")
            result["status"] = "fail"
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
