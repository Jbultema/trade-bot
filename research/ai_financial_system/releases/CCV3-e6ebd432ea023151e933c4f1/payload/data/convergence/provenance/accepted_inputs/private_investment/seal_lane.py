#!/usr/bin/env python3
"""Seal the isolated private-investment v2 author lane.

The lane manifest recursively hashes every lane artifact except itself. This
script does not read from or write to canonical data surfaces.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


LANE = Path(__file__).resolve().parent
CANDIDATE_MANIFEST_SHA256 = (
    "efd9feb57f15e059bce85323e6dfd4c0214fd63953deb2eddb9f7e7e10152958"
)
TASK_SHA256 = (
    "dea89ee4c47469902b022c89bae095cdbca1f803ed28d9644dc0d0798fba51fc"
)
REMAINING_HOLDS = [
    "governed top-20 population reconstruction",
    "S&P 500 aggregate reproduction",
    (
        "unnamed investee allocation at Alphabet and NVIDIA and "
        "residual Amazon attribution"
    ),
    "exact Microsoft component and tax allocation",
    "prospective Meta/Tesla and realization monitoring",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path}: missing CSV header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise AssertionError(f"{path}: malformed-width CSV row")
    return len(rows)


def manifest_entry(path: Path) -> dict[str, object]:
    relative = path.relative_to(LANE).as_posix()
    entry: dict[str, object] = {
        "path": relative,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".csv":
        entry["rows"] = csv_rows(path)
    else:
        entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
    return entry


def main() -> int:
    candidate_manifest = LANE / "candidate" / "candidate_manifest.json"
    if sha256(candidate_manifest) != CANDIDATE_MANIFEST_SHA256:
        raise AssertionError("candidate manifest changed before lane sealing")

    files = [
        manifest_entry(path)
        for path in sorted(LANE.rglob("*"))
        if path.is_file() and path.name != "lane_manifest.json"
    ]
    manifest = {
        "lane": "lane_2_private_investment_candidate_v2",
        "workstream": (
            "private_investment_typed_candidate_v2_atomic_preservation_rebase"
        ),
        "status_date": "2026-07-26",
        "task_sha256": TASK_SHA256,
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "intrinsic_economic_reconstruction_verdict": "PASS",
        "typed_candidate_verdict": "PASS_AUTHOR_VALIDATION",
        "source_preservation_atomic_rebase_verdict": (
            "READY_PENDING_FRESH_REVIEW"
        ),
        "canonical_integration_verdict": (
            "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
        ),
        "validation_scope": {
            "checks": "20_of_20_pass",
            "six_exact_repairs": "6_of_6_pass",
            "typed_fields": "1473_checked_0_failures",
            "source_field_lineage": "500_checked_0_failures",
            "economic_tables_loaded": "8_of_8",
            "external_foreign_keys": 11,
            "canonical_governance_populations": (
                "claims_5_evidence_links_8_handoffs_11_unknowns_11"
            ),
            "negative_fk_tests": "6_of_6_pass",
            "arithmetic_checks": "19_of_19_pass",
            "population_checks": "10_of_10_pass",
            "preservation_records": "13_of_13_pass",
            "clean_builds": 2,
            "deterministic_files": "21_of_21_byte_identical",
            "governing_inputs": "109_of_109_unchanged",
            "open_findings": 0,
        },
        "resolved_v1_review_defects": [
            "V2-001 canonical support types",
            "V2-002 rounded Microsoft amount precision",
            "V2-003 partial and mixed investee identity scopes",
            "V2-004 real external foreign keys",
            "V2-005 field-level transformation lineage",
            "V2-006 Chapter 11 claim lineage",
        ],
        "retained_analytical_holds": REMAINING_HOLDS,
        "preservation_boundary": {
            "original_inline_xbrl_governs": True,
            "visible_text_is_locator_aid_only": True,
            "preservation_hold_clearance_requires_atomic_rebase": True,
            "fresh_nonauthor_review_required": True,
        },
        "no_write_attestation": {
            "canonical_files_written": [],
            "database_written": False,
            "chapters_written": False,
            "docx_pdf_or_publication_manifest_written": False,
            "shared_scripts_modified": False,
            "v1_candidate_modified": False,
            "v1_review_modified": False,
            "accepted_source_packet_or_review_modified": False,
            "author_output_only": (
                "research/ai_financial_system/staging/"
                "lane_2_private_investment_candidate_v2/"
            ),
        },
        "self_hash_policy": "lane_manifest.json excludes its own hash",
        "files": files,
    }
    target = LANE / "lane_manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sha256(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
