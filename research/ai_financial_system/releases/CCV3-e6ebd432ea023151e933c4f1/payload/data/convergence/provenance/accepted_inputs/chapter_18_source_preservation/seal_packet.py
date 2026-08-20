#!/usr/bin/env python3
"""Seal the corrected Chapter 18 preservation author packet."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def entry(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(HERE).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        value["rows"] = row_count(path)
    elif path.suffix in {".md", ".py", ".txt"}:
        value["lines"] = line_count(path)
    return value


def main() -> None:
    validation = json.loads(
        (HERE / "validation_results.json").read_text(encoding="utf-8")
    )
    clean = json.loads(
        (HERE / "two_clean_build_results.json").read_text(encoding="utf-8")
    )
    generic = json.loads(
        (HERE / "generic_validation_results.json").read_text(encoding="utf-8")
    )
    ruff = json.loads((HERE / "ruff_results.json").read_text(encoding="utf-8"))
    if (
        validation["status"] != "PASS"
        or clean["status"] != "PASS"
        or not generic["passed"]
        or ruff["status"] != "PASS"
    ):
        raise ValueError("cannot seal a failing packet")
    artifacts = [
        entry(path)
        for path in sorted(HERE.rglob("*"))
        if path.is_file() and path.name != "lane_manifest.json"
    ]
    manifest = {
        "lane": HERE.name,
        "packet_type": "chapter_18_source_preservation_corrected_semantic_rebase",
        "status_date": "2026-07-26",
        "source_author_manifest_sha256": (
            "595fd73bae86430a68e8462109d646845dd5ca4d1e1dc3f10df901c297950f6d"
        ),
        "fresh_review_manifest_sha256": (
            "945938aefbfc97411f2f0c4c9ead68982633fb56657404ff167ae3ccb5ea0de3"
        ),
        "populations": {
            "hold_relationships": 58,
            "ordered_source_edges": 73,
            "unique_sources": 23,
            "source_version_attempts": 25,
            "successful_artifacts": 23,
            "failed_attempts": 2,
            "derivatives": 23,
            "supported": 32,
            "partially_supported": 26,
            "material_patches": 18,
        },
        "author_validation": "PASS_NOT_ACCEPTANCE",
        "two_clean_builds": "PASS",
        "generic_validation": "PASS",
        "ruff": "PASS",
        "metadata_integration": "NOT_AUTHORIZED",
        "canonical_integration": "NOT_AUTHORIZED",
        "chapter_docx_publication": "NOT_AUTHORIZED",
        "coefficient_predictive_causal_comparative_model_use": "BLOCK",
        "files": artifacts,
    }
    (HERE / "lane_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
