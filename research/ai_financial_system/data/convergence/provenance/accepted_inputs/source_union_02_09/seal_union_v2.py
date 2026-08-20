#!/usr/bin/env python3
"""Run generic validation and seal the union-v2 author lane."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_repo(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "research/ai_financial_system"
        ).is_dir():
            return path
    raise RuntimeError("repository root not found")


REPO = find_repo(Path.cwd())
GENERIC = REPO / "research/ai_financial_system/scripts/validate_staging_packet.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def file_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(HERE.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "lane_manifest.json":
            continue
        entry: dict[str, object] = {
            "path": path.name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            entry["rows"] = row_count(path)
        else:
            entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
        entries.append(entry)
    return entries


def seal() -> None:
    author = json.loads((HERE / "validation_results.json").read_text())
    dedicated = json.loads(
        (HERE / "dedicated_validation_results.json").read_text()
    )
    clean = json.loads((HERE / "two_clean_build_results.json").read_text())
    if not author["passed"] or not dedicated["passed"] or not clean["passed"]:
        raise RuntimeError("cannot seal a failing union-v2 packet")
    manifest = {
        "lane": "lane_3_source_preservation_union_v2",
        "status_date": "2026-07-26",
        "scope": "append-only normalized source-preservation union v2 candidate",
        "decision": "READY_FOR_FRESH_NONAUTHOR_REVIEW",
        "source_version_candidate_readiness": (
            "PASS_AUTHOR_AND_DEDICATED_VALIDATION"
        ),
        "canonical_integration_authorized": False,
        "semantic_locator_clearance": (
            "ONLY_FIVE_PREVIOUSLY_ACCEPTED_PRIVATE_CLAIM_REVIEWS"
        ),
        "chapter_prose_authorized": False,
        "publication_authorized": False,
        "snapshot_id": "SPU-V2-20260726-38b990b2",
        "architecture_counts": {
            "source_version_observations": 60,
            "unique_source_ids": 52,
            "preserved_observations": 53,
            "retrieval_failure_observations": 4,
            "metadata_only_rights_boundaries": 2,
            "derivatives": 8,
            "duplicate_blob_reconciliations": 5,
            "private_claim_reviews": 5,
            "claim_locator_states": 167,
            "chapter_02_09_projection_rows": 525,
            "explicit_exception_observations": 6,
            "delta_observations": 3,
        },
        "projection_state_counts": {
            "preserved_bytes_available_locator_review_pending": 326,
            "retrieval_failed": 3,
            "preserved_and_semantically_reviewed": 5,
            "no_matching_preservation_record": 186,
            "internal_P4_reconstruction": 3,
            "metadata_only_rights_boundary": 2,
        },
        "retained_analytical_holds": [
            "S&P 500 aggregate",
            "exact Microsoft component and tax attribution",
            "governed top-20 population",
            "prospective Meta and Tesla realization monitoring",
            "unnamed investee allocation and residual Amazon attribution",
        ],
        "quarantine_policy": (
            "unmanifested_pre_fix_PR_Newswire_response_excluded"
        ),
        "validation": {
            "author_checks": (
                f"{author['checks_passed']}_of_{author['checks_total']}_pass"
            ),
            "dedicated_checks": (
                f"{dedicated['checks_passed']}_of_"
                f"{dedicated['checks_total']}_pass"
            ),
            "clean_builds": 2,
            "deterministic_authored_files": clean["file_count"],
        },
        "no_write_attestation": {
            "v1_modified": False,
            "canonical_or_shared_data_modified": False,
            "chapter_or_publication_files_modified": False,
            "output_only": (
                "research/ai_financial_system/staging/"
                "lane_3_source_preservation_union_v2/"
            ),
        },
        "files": file_entries(),
        "self_hash_policy": "lane_manifest.json excludes its own hash",
    }
    (HERE / "lane_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generic() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(GENERIC), str(HERE), "--json"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    seal()
    first = generic()
    if not first["passed"]:
        raise RuntimeError(json.dumps(first, indent=2))
    (HERE / "generic_validation_results.json").write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal()
    second = generic()
    if not second["passed"]:
        raise RuntimeError(json.dumps(second, indent=2))
    (HERE / "generic_validation_results.json").write_text(
        json.dumps(second, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal()
    reproduced = generic()
    if reproduced != second:
        raise RuntimeError("generic validation did not stabilize")
    print(json.dumps(second, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
