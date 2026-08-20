#!/usr/bin/env python3
"""Validate the lossless Chapters 11-15 convergence metadata rebase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AI = ROOT / "research/ai_financial_system"
STAGING = AI / "staging"
DATA = AI / "data"
SOURCE_UNION = STAGING / "lane_3_source_preservation_union_v2"

EXPECTED_DATABASE_SHA256 = (
    "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
)
EXPECTED_AUDIT_MANIFEST_SHA256 = (
    "e1e8d8a834edad14f95b5fc8df425e64f0c2d2da8f4be68e20509bfb1209a84c"
)
EXPECTED_SOURCE_UNION_MANIFEST_SHA256 = (
    "437da25ccef0f40d0504d2038f7f32b5d4849fd1fe556739669b91cbedba7406"
)
EXPECTED_SOURCE_UNION_REVIEW_MANIFEST_SHA256 = (
    "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
)
EXPECTED_POPULATIONS = {
    "CH11": {"files": 17, "rows": 484, "fields": 7_894},
    "CH12": {"files": 14, "rows": 121, "fields": 1_776},
    "CH13": {"files": 16, "rows": 195, "fields": 3_323},
    "CH14": {"files": 16, "rows": 259, "fields": 5_457},
    "CH15": {"files": 12, "rows": 100, "fields": 972},
}
CLAIM_PATTERN = re.compile(r"(?<![A-Z0-9-])(?:L[123]-)?CLM-\d+(?![A-Z0-9-])")
SOURCE_PATTERN = re.compile(r"(?<![A-Z0-9-])(?:L[123]-)?SRC-\d+(?![A-Z0-9-])")
ALLOWED_DISPOSITIONS = {
    "canonical_destination",
    "typed_normalized_junction_destination",
    "governed_hold_destination",
    "accepted_duplicate_with_explicit_surviving_identity",
    "explicit_rejection_authorized_by_acceptance_record",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path} has duplicate columns")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{path}:{line_number} exceeds its header width")
    return rows


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        label = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def check_unique(
    errors: list[str],
    rows: list[dict[str, str]],
    key: str,
    label: str,
) -> set[str]:
    values = [row[key] for row in rows]
    if any(not value for value in values):
        errors.append(f"{label} contains a blank {key}")
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        errors.append(f"{label} contains {len(duplicates)} duplicate {key} values")
    return set(values)


def add_check(
    checks: list[dict[str, object]],
    errors: list[str],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"check": name, "passed": passed, "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def canonical_sets() -> tuple[set[str], set[str]]:
    claims = {row["claim_id"] for row in read_csv(DATA / "claims.csv")}
    sources = {row["source_id"] for row in read_csv(DATA / "source_register.csv")}
    return claims, sources


def create_fk_audit(
    files: list[dict[str, str]],
    columns: list[dict[str, str]],
    rows: list[dict[str, str]],
    fields: list[dict[str, str]],
    relationships: list[dict[str, str]],
    version_junction: list[dict[str, str]],
    unmatched_holds: list[dict[str, str]],
    locator_holds: list[dict[str, str]],
) -> tuple[int, str]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE source_files(source_file_id TEXT PRIMARY KEY);
        CREATE TABLE source_columns(
            source_column_id TEXT PRIMARY KEY,
            source_file_id TEXT NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE TABLE source_rows(
            source_row_id TEXT PRIMARY KEY,
            source_file_id TEXT NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE TABLE field_dispositions(
            field_disposition_id TEXT PRIMARY KEY,
            source_row_id TEXT NOT NULL REFERENCES source_rows(source_row_id),
            source_column_id TEXT NOT NULL REFERENCES source_columns(source_column_id)
        );
        CREATE TABLE relationships(
            relationship_id TEXT PRIMARY KEY,
            source_row_id TEXT NOT NULL REFERENCES source_rows(source_row_id)
        );
        CREATE TABLE source_version_junction(
            relationship_source_version_id TEXT PRIMARY KEY,
            relationship_id TEXT NOT NULL REFERENCES relationships(relationship_id)
        );
        CREATE TABLE unmatched_holds(
            hold_id TEXT PRIMARY KEY,
            relationship_id TEXT NOT NULL REFERENCES relationships(relationship_id)
        );
        CREATE TABLE locator_holds(
            hold_id TEXT PRIMARY KEY,
            relationship_id TEXT NOT NULL REFERENCES relationships(relationship_id)
        );
        """
    )
    connection.executemany(
        "INSERT INTO source_files VALUES (?)",
        [(row["source_file_id"],) for row in files],
    )
    connection.executemany(
        "INSERT INTO source_columns VALUES (?, ?)",
        [(row["source_column_id"], row["source_file_id"]) for row in columns],
    )
    connection.executemany(
        "INSERT INTO source_rows VALUES (?, ?)",
        [(row["source_row_id"], row["source_file_id"]) for row in rows],
    )
    connection.executemany(
        "INSERT INTO field_dispositions VALUES (?, ?, ?)",
        [
            (
                row["field_disposition_id"],
                row["source_row_id"],
                row["source_column_id"],
            )
            for row in fields
        ],
    )
    connection.executemany(
        "INSERT INTO relationships VALUES (?, ?)",
        [(row["relationship_id"], row["source_row_id"]) for row in relationships],
    )
    connection.executemany(
        "INSERT INTO source_version_junction VALUES (?, ?)",
        [
            (row["relationship_source_version_id"], row["relationship_id"])
            for row in version_junction
        ],
    )
    connection.executemany(
        "INSERT INTO unmatched_holds VALUES (?, ?)",
        [(row["hold_id"], row["relationship_id"]) for row in unmatched_holds],
    )
    connection.executemany(
        "INSERT INTO locator_holds VALUES (?, ?)",
        [(row["hold_id"], row["relationship_id"]) for row in locator_holds],
    )
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    connection.close()
    return len(violations), integrity[0] if integrity else ""


def source_bearing_row_ids(
    rows: list[dict[str, str]],
) -> set[str]:
    indexed_rows = {
        (row["source_path"], row["primary_key_value"]): row["source_row_id"]
        for row in rows
    }
    expected: set[str] = set()
    for source_path, primary_key in indexed_rows:
        path = ROOT / source_path
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for row in reader:
                if row[header[0]] != primary_key:
                    continue
                claim_tokens = any(
                    CLAIM_PATTERN.search(row[column]) for column in header
                )
                source_tokens = any(
                    SOURCE_PATTERN.search(row[column]) for column in header
                )
                locator_value = any(
                    row[column] != ""
                    for column in header
                    if "locator" in column.lower()
                )
                if claim_tokens or source_tokens or locator_value:
                    expected.add(indexed_rows[(source_path, primary_key)])
                break
    return expected


def validate(target: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: list[dict[str, object]] = []

    required = [
        "acceptance_boundary_bindings.csv",
        "accepted_column_inventory.csv",
        "accepted_file_inventory.csv",
        "accepted_row_inventory.csv",
        "build_counts.json",
        "canonical_baseline_audit.json",
        "canonical_identity_references.csv",
        "canonical_registry_bindings.csv",
        "claim_source_locator_projection.csv",
        "field_disposition_summary.csv",
        "field_dispositions.csv",
        "governing_input_inventory.csv",
        "protected_input_no_write_audit.csv",
        "relationship_source_version_junction.csv",
        "semantic_locator_holds.csv",
        "semantic_reference_dispositions.csv",
        "source_projection_coverage.csv",
        "substantive_limit_controls.csv",
        "unprojected_semantic_references.csv",
        "unmatched_source_version_holds.csv",
    ]
    missing = [name for name in required if not (target / name).is_file()]
    add_check(checks, errors, "required_outputs", not missing, f"missing={missing}")
    if missing:
        return {
            "status": "FAIL",
            "checks": checks,
            "errors": errors,
        }

    files = read_csv(target / "accepted_file_inventory.csv")
    columns = read_csv(target / "accepted_column_inventory.csv")
    rows = read_csv(target / "accepted_row_inventory.csv")
    fields = read_csv(target / "field_dispositions.csv")
    governing = read_csv(target / "governing_input_inventory.csv")
    registry_bindings = read_csv(target / "canonical_registry_bindings.csv")
    identity_references = read_csv(target / "canonical_identity_references.csv")
    acceptances = read_csv(target / "acceptance_boundary_bindings.csv")
    controls = read_csv(target / "substantive_limit_controls.csv")
    relationships = read_csv(target / "claim_source_locator_projection.csv")
    version_junction = read_csv(target / "relationship_source_version_junction.csv")
    unmatched_holds = read_csv(target / "unmatched_source_version_holds.csv")
    locator_holds = read_csv(target / "semantic_locator_holds.csv")
    semantic_dispositions = read_csv(
        target / "semantic_reference_dispositions.csv"
    )
    unprojected = read_csv(target / "unprojected_semantic_references.csv")
    coverage = read_csv(target / "source_projection_coverage.csv")
    no_write = read_csv(target / "protected_input_no_write_audit.csv")

    file_ids = check_unique(errors, files, "source_file_id", "accepted files")
    column_ids = check_unique(errors, columns, "source_column_id", "accepted columns")
    row_ids = check_unique(errors, rows, "source_row_id", "accepted rows")
    field_ids = check_unique(
        errors, fields, "field_disposition_id", "field dispositions"
    )
    relationship_ids = check_unique(
        errors, relationships, "relationship_id", "source projection"
    )
    junction_ids = check_unique(
        errors,
        version_junction,
        "relationship_source_version_id",
        "source-version junction",
    )
    del file_ids, column_ids, row_ids, field_ids, relationship_ids, junction_ids

    chapter_populations: dict[str, dict[str, int]] = {}
    for chapter in EXPECTED_POPULATIONS:
        chapter_populations[chapter] = {
            "files": sum(row["chapter_id"] == chapter for row in files),
            "rows": sum(row["chapter_id"] == chapter for row in rows),
            "fields": sum(row["chapter_id"] == chapter for row in fields),
        }
    population_ok = chapter_populations == EXPECTED_POPULATIONS
    add_check(
        checks,
        errors,
        "exact_frozen_populations",
        population_ok,
        json.dumps(chapter_populations, sort_keys=True),
    )
    add_check(
        checks,
        errors,
        "exact_total_population",
        len(files) == 75 and len(rows) == 1_159 and len(fields) == 19_422,
        f"files={len(files)} rows={len(rows)} fields={len(fields)}",
    )

    current_file_hashes_ok = all(
        row["manifest_declared_sha256"]
        == row["observed_sha256"]
        == sha256(ROOT / row["source_path"])
        and row["byte_identity_status"] == "PASS"
        for row in files
    )
    add_check(
        checks,
        errors,
        "accepted_author_byte_identity",
        current_file_hashes_ok,
        f"files={len(files)}",
    )

    governing_ok = all(
        row["status"] == "PASS"
        and row["expected_sha256"] == row["observed_sha256"]
        and row["observed_sha256"] == sha256(ROOT / row["path"])
        for row in governing
    )
    roles = Counter(row["role"] for row in governing)
    governing_population_ok = (
        roles["author_artifact"] == 75
        and roles["author_manifest"] == 5
        and roles["review_manifest"] == 5
        and roles["review_artifact"] == 5
        and roles["correction_verification_manifest"] == 1
        and roles["correction_verification_artifact"] == 1
        and roles["substantive_acceptance"] == 5
        and roles["governing_metadata_rebase_v1_manifest"] == 1
        and roles["governing_metadata_rebase_v1_artifact"] == 29
        and roles["governing_metadata_rebase_v1_review_manifest"] == 1
        and roles["governing_metadata_rebase_v1_review_artifact"] == 12
    )
    add_check(
        checks,
        errors,
        "governing_input_hashes",
        governing_ok and governing_population_ok,
        f"inputs={len(governing)} roles={dict(sorted(roles.items()))}",
    )

    expected_field_pairs = {
        (row["source_row_id"], row["source_column_id"]) for row in fields
    }
    actual_expected_pairs = {
        (row["source_row_id"], column["source_column_id"])
        for row in rows
        for column in columns
        if column["source_file_id"] == row["source_file_id"]
    }
    one_field_per_cell = (
        len(expected_field_pairs) == len(fields)
        and expected_field_pairs == actual_expected_pairs
    )
    identity_preserved = all(
        row["original_value"] == row["destination_value"]
        and row["original_value_sha256"]
        == sha256_bytes(row["original_value"].encode("utf-8"))
        and row["source_is_populated"]
        == str(row["original_value"] != "").lower()
        and row["destination_identity"] == row["field_disposition_id"]
        and row["transformation_rule"] == "identity_no_transformation"
        and row["disposition"] in ALLOWED_DISPOSITIONS
        for row in fields
    )
    evidence_states_preserved = all(
        row["evidence_state_guard"] != "identity_locked"
        or row["destination_evidence_state"] == row["original_value"]
        for row in fields
    )
    add_check(
        checks,
        errors,
        "lossless_field_dispositions",
        one_field_per_cell and identity_preserved and evidence_states_preserved,
        (
            f"cells={len(fields)} unique_destinations="
            f"{len({row['destination_identity'] for row in fields})}"
        ),
    )

    foreign_key_violations, metadata_integrity = create_fk_audit(
        files,
        columns,
        rows,
        fields,
        relationships,
        version_junction,
        unmatched_holds,
        locator_holds,
    )
    add_check(
        checks,
        errors,
        "stable_identities_and_foreign_keys",
        foreign_key_violations == 0 and metadata_integrity == "ok",
        f"violations={foreign_key_violations} integrity={metadata_integrity}",
    )

    baseline = json.loads(
        (target / "canonical_baseline_audit.json").read_text(encoding="utf-8")
    )
    baseline_ok = (
        baseline["canonical_database_sha256"] == EXPECTED_DATABASE_SHA256
        and baseline["sealed_candidate_database_sha256"] == EXPECTED_DATABASE_SHA256
        and baseline["canonical_candidate_byte_identical"]
        and baseline["canonical_matches_expected"]
        and baseline["candidate_matches_expected"]
        and baseline["canonical_sqlite"]["integrity_check"] == "ok"
        and baseline["candidate_sqlite"]["integrity_check"] == "ok"
        and baseline["canonical_sqlite"]["foreign_key_violations"] == 0
        and baseline["candidate_sqlite"]["foreign_key_violations"] == 0
        and baseline["governing_audit_manifest_sha256"]
        == EXPECTED_AUDIT_MANIFEST_SHA256
        and baseline["accepted_source_union_manifest_sha256"]
        == EXPECTED_SOURCE_UNION_MANIFEST_SHA256
        and baseline["accepted_source_union_review_manifest_sha256"]
        == EXPECTED_SOURCE_UNION_REVIEW_MANIFEST_SHA256
    )
    add_check(
        checks,
        errors,
        "canonical_database_integrity_and_byte_identity",
        baseline_ok,
        f"database_sha256={baseline['canonical_database_sha256']}",
    )

    registry_hashes_ok = all(
        row["canonical_registry_sha256"]
        == sha256(ROOT / row["canonical_registry_path"])
        and row["mutation_authority"] == "none"
        for row in registry_bindings
    )
    reference_registry_pairs = {
        (row["chapter_id"], row["canonical_registry"])
        for row in identity_references
    }
    binding_pairs = {
        (row["chapter_id"], row["canonical_registry"]) for row in registry_bindings
    }
    add_check(
        checks,
        errors,
        "canonical_registry_bindings",
        registry_hashes_ok and reference_registry_pairs <= binding_pairs,
        f"bindings={len(registry_bindings)} references={len(identity_references)}",
    )

    expected_source_rows = source_bearing_row_ids(rows)
    covered_source_rows = {row["source_row_id"] for row in coverage}
    relationships_by_row = Counter(row["source_row_id"] for row in relationships)
    semantic_occurrences_by_row = Counter(
        row["source_row_id"] for row in semantic_dispositions
    )
    explicit_holds_by_row = Counter(row["source_row_id"] for row in unprojected)
    projection_complete = (
        expected_source_rows == covered_source_rows
        and all(row["coverage_status"].startswith("PASS_") for row in coverage)
        and all(
            int(row["relationship_count"]) == relationships_by_row[row["source_row_id"]]
            and int(row["semantic_reference_occurrence_count"])
            == semantic_occurrences_by_row[row["source_row_id"]]
            and int(row["explicit_governed_disposition_count"])
            == explicit_holds_by_row[row["source_row_id"]]
            and (
                int(row["relationship_count"]) > 0
                or row["coverage_status"]
                == "PASS_EXPLICIT_GOVERNED_DISPOSITION"
            )
            for row in coverage
        )
    )
    add_check(
        checks,
        errors,
        "source_locator_projection_completeness",
        projection_complete,
        (
            f"source_rows={len(expected_source_rows)} "
            f"relationships={len(relationships)}"
        ),
    )

    independent = subprocess.run(
        [
            sys.executable,
            str(HERE / "validate_all_cell_semantics.py"),
            "--target",
            str(target),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    independent_results_path = (
        target / "all_cell_semantic_validation_results.json"
    )
    independent_results = (
        json.loads(independent_results_path.read_text(encoding="utf-8"))
        if independent_results_path.is_file()
        else {}
    )
    semantic_coverage_ok = (
        independent.returncode == 0
        and independent_results.get("status") == "PASS"
        and int(
            independent_results.get("semantic_reference_occurrences", -1)
        )
        == len(semantic_dispositions)
        and int(
            independent_results.get("explicit_governed_reference_holds", -1)
        )
        == len(unprojected)
        == 8
        and independent_results.get(
            "reviewed_omissions_exactly_accounted_for"
        )
        is True
    )
    add_check(
        checks,
        errors,
        "independent_all_cell_semantic_reference_completeness",
        semantic_coverage_ok,
        (
            f"occurrences={len(semantic_dispositions)} "
            f"explicit_holds={len(unprojected)} "
            f"returncode={independent.returncode}"
        ),
    )

    canonical_claims, canonical_sources = canonical_sets()
    relationship_junction_counts = Counter(
        row["relationship_id"] for row in version_junction
    )
    unmatched_by_relationship = {
        row["relationship_id"]: row["hold_id"] for row in unmatched_holds
    }
    locator_hold_by_relationship = {
        row["relationship_id"]: row["hold_id"] for row in locator_holds
    }
    separate_state_ok = all(
        row["version_preservation_state"]
        in {
            "referenced_accepted_source_version_snapshot",
            "unmatched_version_hold",
        }
        and row["rights_access_state"] != ""
        and row["semantic_locator_review_state"] != ""
        and row["publication_eligibility"] == "held_no_publication_authority"
        and row["byte_preservation_semantic_boundary"]
        == "byte_preservation_is_not_semantic_support"
        and int(row["source_version_reference_count"])
        == relationship_junction_counts[row["relationship_id"]]
        and (
            (
                int(row["source_version_reference_count"]) > 0
                and row["unmatched_version_hold_id"] == ""
                and row["version_preservation_state"]
                == "referenced_accepted_source_version_snapshot"
            )
            or (
                int(row["source_version_reference_count"]) == 0
                and unmatched_by_relationship.get(row["relationship_id"])
                == row["unmatched_version_hold_id"]
                and row["version_preservation_state"] == "unmatched_version_hold"
            )
        )
        and (
            (
                row["semantic_locator_review_state"]
                == "preserved_and_semantically_reviewed"
                and
                row["semantic_locator_hold_id"] == ""
                and row["semantic_locator_review_reference_id"] != ""
            )
            or (
                row["semantic_locator_review_state"]
                != "preserved_and_semantically_reviewed"
                and locator_hold_by_relationship.get(row["relationship_id"])
                == row["semantic_locator_hold_id"]
            )
        )
        and (
            row["canonical_claim_id"] == ""
            or row["canonical_claim_id"] in canonical_claims
        )
        and (
            row["canonical_source_id"] == ""
            or row["canonical_source_id"] in canonical_sources
        )
        for row in relationships
    )
    add_check(
        checks,
        errors,
        "source_version_rights_locator_publication_separation",
        separate_state_ok,
        (
            f"matched={sum(int(row['source_version_reference_count']) > 0 for row in relationships)} "
            f"unmatched={len(unmatched_holds)} locator_holds={len(locator_holds)}"
        ),
    )

    union_versions = read_csv(SOURCE_UNION / "source_versions.csv")
    union_version_by_id = {row["source_version_id"]: row for row in union_versions}
    version_rows_valid = all(
        row["source_version_id"] in union_version_by_id
        and row["canonical_source_id"]
        == union_version_by_id[row["source_version_id"]]["source_id"]
        and row["source_union_manifest_sha256"]
        == EXPECTED_SOURCE_UNION_MANIFEST_SHA256
        and row["mutable_source_union_status_copied"] == "false"
        for row in version_junction
    )
    add_check(
        checks,
        errors,
        "accepted_source_version_snapshot_references",
        version_rows_valid,
        (
            f"junction_rows={len(version_junction)} "
            f"unique_versions={len({row['source_version_id'] for row in version_junction})}"
        ),
    )

    acceptance_ok = (
        len(acceptances) == 5
        and all(
            row["substantive_acceptance_sha256"]
            == sha256(ROOT / row["substantive_acceptance_path"])
            and row["accepted_author_manifest_sha256"]
            == sha256(ROOT / row["accepted_author_manifest_path"])
            and row["boundary_enforcement"]
            == "all_acceptance_record_boundaries_retained"
            and row["evidence_state_change_authorized"] == "false"
            and row["publication_state"] == "held_no_publication_authority"
            for row in acceptances
        )
    )
    controls_ok = (
        len(controls) == 11
        and all(
            row["control_status"] == "ACTIVE"
            and row["model_use_state"] == "held"
            and row["publication_state"] == "held"
            for row in controls
        )
    )
    held_pairings = all(
        row["pairing_ambiguity"] == "none"
        or row["pairing_ambiguity"].startswith("held_")
        for row in relationships
    )
    add_check(
        checks,
        errors,
        "no_evidence_or_publication_authority_laundering",
        acceptance_ok and controls_ok and held_pairings and evidence_states_preserved,
        f"acceptances={len(acceptances)} controls={len(controls)}",
    )

    no_write_ok = all(
        row["status"] == "PASS"
        and row["before_tree_sha256"] == row["after_tree_sha256"]
        and row["after_tree_sha256"] == tree_hash(ROOT / row["path"])
        for row in no_write
    )
    add_check(
        checks,
        errors,
        "protected_input_no_write",
        no_write_ok,
        f"protected_inputs={len(no_write)}",
    )

    counts = json.loads((target / "build_counts.json").read_text(encoding="utf-8"))
    reported_counts_ok = (
        counts["accepted_author_files"] == len(files)
        and counts["accepted_csv_rows"] == len(rows)
        and counts["accepted_scalar_fields"] == len(fields)
        and counts["claim_source_locator_relationships"] == len(relationships)
        and counts["relationship_source_version_references"] == len(version_junction)
        and counts["unmatched_source_version_holds"] == len(unmatched_holds)
        and counts["semantic_locator_holds"] == len(locator_holds)
        and counts["semantic_reference_occurrences"]
        == len(semantic_dispositions)
        and counts["explicit_semantic_reference_holds"] == len(unprojected)
    )
    add_check(
        checks,
        errors,
        "reported_counts_reperformance",
        reported_counts_ok,
        json.dumps(counts, sort_keys=True),
    )

    return {
        "status": "PASS" if not errors else "FAIL",
        "verdict": (
            "READY_FOR_FRESH_UNINVOLVED_METADATA_ACCEPTANCE_WITH_ALL_HOLDS"
            if not errors
            else "NOT_READY"
        ),
        "checks": checks,
        "checks_passed": sum(bool(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "counts": {
            "accepted_author_files": len(files),
            "accepted_csv_rows": len(rows),
            "accepted_scalar_fields": len(fields),
            "claim_source_locator_relationships": len(relationships),
            "relationship_source_version_references": len(version_junction),
            "unmatched_source_version_holds": len(unmatched_holds),
            "semantic_locator_holds": len(locator_holds),
            "semantic_reference_occurrences": len(semantic_dispositions),
            "explicit_semantic_reference_holds": len(unprojected),
        },
        "errors": errors,
        "canonical_integration_authorized": False,
        "chapter_prose_authorized": False,
        "docx_authorized": False,
        "publication_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=HERE)
    args = parser.parse_args()
    target = args.target.resolve()
    result = validate(target)
    (target / "validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{result['status']}: checks={result.get('checks_passed', 0)}/"
        f"{result.get('checks_total', 0)}"
    )
    for error in result["errors"]:
        print(f"ERROR: {error}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
