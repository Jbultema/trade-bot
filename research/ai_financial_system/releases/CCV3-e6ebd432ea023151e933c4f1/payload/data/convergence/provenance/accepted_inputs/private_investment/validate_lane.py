#!/usr/bin/env python3
"""Validate the sealed private-investment v2 author lane read-only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


LANE = Path(__file__).resolve().parent
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "efd9feb57f15e059bce85323e6dfd4c0214fd63953deb2eddb9f7e7e10152958"
)
EXPECTED_TASK_SHA256 = (
    "dea89ee4c47469902b022c89bae095cdbca1f803ed28d9644dc0d0798fba51fc"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path}: missing CSV header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise AssertionError(f"{path}: malformed-width CSV row")
    return rows


def assert_all(
    path: Path,
    *,
    count: int,
    field: str,
    accepted: set[str],
) -> None:
    rows = csv_rows(path)
    assert len(rows) == count, f"{path.name}: {len(rows)} != {count}"
    rejected = [
        row for row in rows if (row.get(field) or "").strip() not in accepted
    ]
    assert not rejected, f"{path.name}: {len(rejected)} nonpassing rows"


def repo_root() -> Path:
    for parent in [LANE, *LANE.parents]:
        if (parent / "pyproject.toml").is_file() and (
            parent / "research" / "ai_financial_system"
        ).is_dir():
            return parent
    raise AssertionError("repository root not found")


def validate_lane_manifest(manifest: dict[str, Any]) -> None:
    assert manifest["task_sha256"] == EXPECTED_TASK_SHA256
    assert (
        manifest["candidate_manifest_sha256"]
        == EXPECTED_CANDIDATE_MANIFEST_SHA256
    )
    assert manifest["intrinsic_economic_reconstruction_verdict"] == "PASS"
    assert manifest["typed_candidate_verdict"] == "PASS_AUTHOR_VALIDATION"
    assert (
        manifest["source_preservation_atomic_rebase_verdict"]
        == "READY_PENDING_FRESH_REVIEW"
    )
    assert (
        manifest["canonical_integration_verdict"]
        == "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
    )
    assert len(manifest["retained_analytical_holds"]) == 5
    attestation = manifest["no_write_attestation"]
    assert attestation["canonical_files_written"] == []
    assert attestation["database_written"] is False
    assert attestation["chapters_written"] is False
    assert attestation["docx_pdf_or_publication_manifest_written"] is False
    assert attestation["shared_scripts_modified"] is False

    expected_paths = {
        path.relative_to(LANE).as_posix()
        for path in LANE.rglob("*")
        if path.is_file() and path.name != "lane_manifest.json"
    }
    entries = manifest["files"]
    recorded_paths = {entry["path"] for entry in entries}
    assert len(entries) == len(recorded_paths), "duplicate lane-manifest paths"
    assert recorded_paths == expected_paths, (
        f"lane-manifest path mismatch: missing={expected_paths-recorded_paths}; "
        f"extra={recorded_paths-expected_paths}"
    )
    for entry in entries:
        path = LANE / entry["path"]
        assert path.stat().st_size == entry["bytes"], f"{entry['path']}: bytes"
        assert sha256(path) == entry["sha256"], f"{entry['path']}: sha256"
        if path.suffix.lower() == ".csv":
            assert len(csv_rows(path)) == entry["rows"], (
                f"{entry['path']}: rows"
            )
        else:
            lines = len(path.read_text(encoding="utf-8").splitlines())
            assert lines == entry["lines"], f"{entry['path']}: lines"


def validate_candidate_manifest() -> None:
    path = LANE / "candidate" / "candidate_manifest.json"
    assert sha256(path) == EXPECTED_CANDIDATE_MANIFEST_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "AUTHOR_V2_CANDIDATE_PENDING_FRESH_REVIEW_NOT_CANONICAL"
    )
    assert payload["canonical_integration"] == (
        "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
    )
    assert len(payload["remaining_holds"]) == 5
    assert len(payload["resolved_repairs"]) == 6
    assert len(payload["files"]) == 20
    for relative, expected in payload["files"].items():
        artifact = path.parent / relative
        assert artifact.is_file(), f"candidate artifact missing: {relative}"
        assert artifact.stat().st_size == expected["bytes"], (
            f"candidate bytes: {relative}"
        )
        assert sha256(artifact) == expected["sha256"], (
            f"candidate sha256: {relative}"
        )
        if artifact.suffix.lower() == ".csv":
            assert len(csv_rows(artifact)) == expected["rows"], (
                f"candidate rows: {relative}"
            )


def validate_author_results() -> None:
    results = json.loads(
        (LANE / "validation_results.json").read_text(encoding="utf-8")
    )
    expected = {
        "status": "PASS",
        "checks_total": 20,
        "checks_passed": 20,
        "checks_failed": 0,
        "repairs_total": 6,
        "repairs_passed": 6,
        "typed_fields_checked": 1473,
        "typed_field_failures": 0,
        "transformation_rows_checked": 500,
        "transformation_lineage_failures": 0,
        "economic_tables_loaded": 8,
        "external_foreign_keys": 11,
        "negative_fk_tests_passed": 6,
        "arithmetic_checks_passed": 19,
        "population_checks_passed": 10,
        "preservation_records_passed": 13,
        "clean_builds_compared": 2,
        "deterministic_files_compared": 21,
        "intrinsic_economic_reconstruction": "PASS",
        "typed_candidate_completeness": "PASS_AUTHOR_VALIDATION",
        "source_preservation_atomic_rebase_readiness": (
            "READY_PENDING_FRESH_REVIEW"
        ),
        "canonical_integration_readiness": (
            "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
        ),
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        assert results.get(key) == value, (
            f"validation_results[{key!r}]={results.get(key)!r}, expected {value!r}"
        )
    assert len(results["remaining_analytical_holds"]) == 5
    assert results["author_canonical_writes"] == []

    assert_all(
        LANE / "checks.csv", count=20, field="result", accepted={"PASS"}
    )
    assert_all(
        LANE / "repair_reperformance.csv",
        count=6,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "typed_semantics_audit.csv",
        count=1473,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "transformation_lineage_validation.csv",
        count=500,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "canonical_constraint_probe.csv",
        count=11,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "negative_fk_tests.csv",
        count=6,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "arithmetic_reconciliation.csv",
        count=19,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "population_reconciliation.csv",
        count=10,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "source_preservation_validation.csv",
        count=13,
        field="result",
        accepted={"PASS"},
    )
    assert_all(
        LANE / "two_clean_build_comparison.csv",
        count=21,
        field="result",
        accepted={"PASS_BYTE_IDENTICAL"},
    )
    assert_all(
        LANE / "input_hash_audit.csv",
        count=109,
        field="result",
        accepted={"PASS_UNCHANGED"},
    )
    assert len(csv_rows(LANE / "findings.csv")) == 0


def validate_live_governing_inputs() -> None:
    root = repo_root()
    for row in csv_rows(LANE / "input_hash_audit.csv"):
        assert row["sha256_before"] == row["sha256_after"]
        path = root / row["path"]
        assert path.is_file(), f"governing input missing: {row['path']}"
        assert sha256(path) == row["sha256_after"], (
            f"governing input changed: {row['path']}"
        )


def main() -> int:
    manifest_path = LANE / "lane_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_lane_manifest(manifest)
    validate_candidate_manifest()
    validate_author_results()
    validate_live_governing_inputs()
    print(
        json.dumps(
            {
                "status": "PASS",
                "lane_manifest_sha256": sha256(manifest_path),
                "manifested_files": len(manifest["files"]),
                "candidate_manifest_sha256": (
                    EXPECTED_CANDIDATE_MANIFEST_SHA256
                ),
                "canonical_integration": (
                    "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
