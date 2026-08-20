#!/usr/bin/env python3
"""Create deterministic searchable-text derivatives of preserved SEC filings."""

from __future__ import annotations

import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "research/ai_financial_system/sources/sec/private_investment_earnings"
)
SOURCE_MANIFEST = ARCHIVE / "source_archive_manifest.csv"
DERIVED_MANIFEST = ARCHIVE / "derived_text_manifest.csv"
VALIDATION = ARCHIVE / "derived_text_validation.json"
EXTRACTOR_VERSION = "private_investment_sec_visible_text_v2"

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
SKIP_TAGS = {"script", "style", "ix:header"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines: list[str] = []
        for line in raw.splitlines():
            normalized = " ".join(line.split())
            if normalized and (not lines or normalized != lines[-1]):
                lines.append(normalized)
        return "\n".join(lines) + "\n"


def main() -> None:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        sources = list(csv.DictReader(stream))

    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda row: row["source_id"]):
        original = ROOT / source["archived_path"]
        if sha256(original) != source["content_sha256"]:
            raise ValueError(f"original hash mismatch: {source['source_id']}")
        extractor = VisibleTextExtractor()
        extractor.feed(original.read_text(encoding="utf-8", errors="replace"))
        derivative = ARCHIVE / f"{source['source_id']}_visible_text.txt"
        derivative.write_text(extractor.text(), encoding="utf-8", newline="\n")
        text = derivative.read_text(encoding="utf-8")
        identity_ok = (
            source["registrant"].casefold() in text.casefold()
            and source["document_type"] in text
            and source["document_period_end"].casefold() in text.casefold()
        )
        status = (
            "preserved_derived_with_original"
            if identity_ok and derivative.stat().st_size > 10_000
            else "derived_validation_failed"
        )
        rows.append(
            {
                "source_id": source["source_id"],
                "original_archived_path": source["archived_path"],
                "original_content_sha256": source["content_sha256"],
                "derived_text_path": derivative.relative_to(ROOT).as_posix(),
                "derived_text_sha256": sha256(derivative),
                "derived_byte_count": derivative.stat().st_size,
                "extraction_method": EXTRACTOR_VERSION,
                "normalization_rule": (
                    "visible_text_excluding_script_style_ix_hidden;"
                    "whitespace_collapsed;consecutive_duplicate_lines_removed"
                ),
                "preservation_status": status,
                "evidentiary_boundary": (
                    "search_and_locator_aid_only_original_inline_xbrl_governs"
                ),
            }
        )
        checks.append(
            {
                "source_id": source["source_id"],
                "identity_visible": identity_ok,
                "derived_text_nontrivial": derivative.stat().st_size > 10_000,
                "status": "PASS" if status.startswith("preserved_") else "FAIL",
            }
        )

    fields = [
        "source_id",
        "original_archived_path",
        "original_content_sha256",
        "derived_text_path",
        "derived_text_sha256",
        "derived_byte_count",
        "extraction_method",
        "normalization_rule",
        "preservation_status",
        "evidentiary_boundary",
    ]
    with DERIVED_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "source_count": len(checks),
        "passed_source_count": sum(row["status"] == "PASS" for row in checks),
        "failed_source_count": sum(row["status"] != "PASS" for row in checks),
        "extractor_version": EXTRACTOR_VERSION,
        "checks": checks,
    }
    VALIDATION.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        raise SystemExit("SEC visible-text derivative validation failed")


if __name__ == "__main__":
    main()
