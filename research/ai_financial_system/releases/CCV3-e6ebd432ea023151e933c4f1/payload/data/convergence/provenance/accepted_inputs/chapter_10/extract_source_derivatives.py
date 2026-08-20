#!/usr/bin/env python3
"""Extract deterministic locator-aid text from preserved Chapter 10 bytes."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "source_archive"
DERIVATIVES = HERE / "source_derivatives"


def normalize_text(value: str) -> str:
    lines: list[str] = []
    previous = ""
    for raw in value.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line and line != previous:
            lines.append(line)
            previous = line
    return "\n".join(lines) + ("\n" if lines else "")


def html_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return normalize_text(soup.get_text("\n"))


def pdf_text(path: Path) -> str:
    pages: list[str] = []
    reader = PdfReader(path)
    for number, page in enumerate(reader.pages, start=1):
        extracted = normalize_text(page.extract_text() or "")
        pages.append(f"[[PDF_PAGE_{number:04d}]]\n{extracted}")
    return "\n".join(pages)


def main() -> int:
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    for source in sorted(ARCHIVE.iterdir()):
        if not source.is_file():
            continue
        failure = (
            HERE
            / "response_metadata"
            / f"{source.stem}.failure.txt"
        )
        if failure.exists():
            continue
        target = DERIVATIVES / f"{source.stem}_visible_text.txt"
        text = (
            pdf_text(source)
            if source.suffix.lower() == ".pdf"
            else html_text(source)
        )
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
