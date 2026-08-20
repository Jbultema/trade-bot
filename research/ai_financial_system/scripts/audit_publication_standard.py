from __future__ import annotations

import argparse
import csv
import re
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]

MANDATORY_POST_METHODOLOGY_SECTIONS = (
    ("observed_facts", re.compile(r"\bobserved facts?\b", re.I)),
    ("reconstruction", re.compile(r"\breconstruction\b|\breconstructed architecture\b", re.I)),
    ("interpretation", re.compile(r"\binterpretation\b", re.I)),
    ("remaining_unknowns", re.compile(r"\b(remaining|priority|critical) unknowns?\b|\bwhat remains unknown\b", re.I)),
    ("falsifiers", re.compile(r"\bevidence that would overturn\b|\bfalsif", re.I)),
    ("missing_evidence", re.compile(r"\bevidence still missing\b", re.I)),
    ("confidence_evolution", re.compile(r"\bconfidence evolution\b", re.I)),
    ("economic_importance", re.compile(r"\beconomic importance\b", re.I)),
)


class HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_heading = False
        self.parts: list[str] = []
        self.headings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3"}:
            self.in_heading = True
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.in_heading:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3"} and self.in_heading:
            self.headings.append(" ".join(" ".join(self.parts).split()))
            self.in_heading = False


def audit_chapter(chapter: int) -> dict[str, str | int]:
    path = PROJECT_ROOT / f"chapters/chapter_{chapter:02d}/chapter.html"
    if not path.is_file():
        return {
            "chapter": chapter,
            "source_html": str(path.relative_to(REPO_ROOT)),
            "standard_scope": (
                "methodology" if chapter == 1 else "post_methodology"
            ),
            "heading_count": 0,
            "observed_facts": "source_missing",
            "reconstruction": "source_missing",
            "interpretation": "source_missing",
            "remaining_unknowns": "source_missing",
            "falsifiers": "source_missing",
            "missing_evidence": "source_missing",
            "confidence_evolution": "source_missing",
            "economic_importance": "source_missing",
            "publication_gate": "missing_source",
        }
    parser = HeadingParser()
    parser.feed(path.read_text(encoding="utf-8"))
    headings = parser.headings
    joined = "\n".join(headings)
    row: dict[str, str | int] = {
        "chapter": chapter,
        "source_html": str(path.relative_to(REPO_ROOT)),
        "heading_count": len(headings),
    }
    if chapter == 1:
        confidence_evolution = bool(re.search(r"\bconfidence evolution\b", joined, re.I))
        economic_importance = bool(re.search(r"\beconomic importance\b", joined, re.I))
        falsifiers = bool(re.search(r"\bevidence that would overturn\b", joined, re.I))
        missing_evidence = bool(re.search(r"\bevidence still missing\b", joined, re.I))
        row.update(
            {
                "standard_scope": "methodology",
                "observed_facts": "not_applicable",
                "reconstruction": "not_applicable",
                "interpretation": "not_applicable",
                "remaining_unknowns": "not_applicable",
                "falsifiers": "method_defined" if falsifiers else "missing",
                "missing_evidence": "method_defined" if missing_evidence else "missing",
                "confidence_evolution": "method_defined" if confidence_evolution else "missing",
                "economic_importance": "method_defined" if economic_importance else "missing",
                "publication_gate": "pass"
                if all((falsifiers, missing_evidence, confidence_evolution, economic_importance))
                else "revision_required",
            }
        )
        return row

    found: dict[str, bool] = {}
    for field, pattern in MANDATORY_POST_METHODOLOGY_SECTIONS:
        found[field] = bool(pattern.search(joined))
        row[field] = "present" if found[field] else "missing"
    row["standard_scope"] = "post_methodology"
    row["publication_gate"] = "pass" if all(found.values()) else "revision_required"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, default=19)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "reports/ai_financial_system/publication_standard_audit.csv",
    )
    args = parser.parse_args()
    rows = [audit_chapter(chapter) for chapter in range(1, args.through + 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "chapter",
        "source_html",
        "standard_scope",
        "heading_count",
        "observed_facts",
        "reconstruction",
        "interpretation",
        "remaining_unknowns",
        "falsifiers",
        "missing_evidence",
        "confidence_evolution",
        "economic_importance",
        "publication_gate",
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row["chapter"] for row in rows if row["publication_gate"] != "pass"]
    print(args.output)
    print(f"chapters_audited={len(rows)}")
    print(f"publication_gate_failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
