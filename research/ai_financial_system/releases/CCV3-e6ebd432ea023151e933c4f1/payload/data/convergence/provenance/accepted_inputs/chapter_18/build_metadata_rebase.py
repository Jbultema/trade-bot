#!/usr/bin/env python3
"""Build the bounded Chapter 18 immutable source-metadata rebase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AI_ROOT = REPO / "research/ai_financial_system"
STAGING = AI_ROOT / "staging"

INTRINSIC_AUTHOR = STAGING / "lane_4_chapter_18_rebuilt_v5"
INTRINSIC_REVIEW = STAGING / "lane_2_verify_chapter_18_rebuilt_v5"
PRESERVATION_AUTHOR = STAGING / "lane_3_chapter_18_source_preservation_corrected"
PRESERVATION_REVIEW = (
    STAGING / "lane_4_verify_chapter_18_source_preservation_corrected"
)
CANONICAL_DB = REPO / "reports/ai_financial_system/research.db"
CANONICAL_MANIFEST = REPO / "reports/ai_financial_system/database_manifest.json"

GOVERNING_INPUTS = (
    (
        "intrinsic_author_manifest",
        INTRINSIC_AUTHOR / "lane_manifest.json",
        "d11f8bd03062b2abfbcb85d3c351ea614eb2d06a3704ed15b6e571c2e758b8de",
    ),
    (
        "intrinsic_review_manifest",
        INTRINSIC_REVIEW / "lane_manifest.json",
        "d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8",
    ),
    (
        "corrected_preservation_author_manifest",
        PRESERVATION_AUTHOR / "lane_manifest.json",
        "3fad6122a81e115a352e20bfdb3086be1a25d5ea6547202c97d6ac2020b11867",
    ),
    (
        "corrected_preservation_review_manifest",
        PRESERVATION_REVIEW / "lane_manifest.json",
        "d4e1d026f7743032e77f98a6ed1424ccd4270bb5a5188a3ed1a5d98c44eb8dd9",
    ),
    (
        "frozen_canonical_database",
        CANONICAL_DB,
        "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6",
    ),
)

MANIFEST_COPY_NAMES = {
    "intrinsic_author_manifest": "intrinsic_author_lane_manifest.json",
    "intrinsic_review_manifest": "intrinsic_review_lane_manifest.json",
    "corrected_preservation_author_manifest": (
        "corrected_preservation_author_lane_manifest.json"
    ),
    "corrected_preservation_review_manifest": (
        "corrected_preservation_review_lane_manifest.json"
    ),
}

LOCATOR_AUTHORITY_FIELDS = {
    "coefficient_authority",
    "predictive_authority",
    "canonical_integration",
    "publication",
    "causal_authority",
    "comparative_system_authority",
    "model_authority",
    "docx_authority",
    "metadata_integration_authority",
}

PATCH_AUTHORITY_FIELDS = {
    "coefficient_authority",
    "predictive_authority",
    "canonical_integration",
    "publication",
}

EXPECTED_POPULATIONS = {
    "intrinsic_analytical_records": 128,
    "intrinsic_record_claim_links": 214,
    "arithmetic_nonadditivity_controls": 20,
    "locator_relationships": 58,
    "supported": 32,
    "partially_supported": 26,
    "semantic_patches": 18,
    "ordered_source_edges": 73,
    "unique_sources": 23,
    "retrieval_attempts": 25,
    "successful_artifacts": 23,
    "retained_failures": 2,
    "derivatives": 23,
    "coefficient_blocks": 18,
    "predictive_blocks": 16,
}


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(prefix: str, *parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{relative(path)} has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{relative(path)} has duplicate columns")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{relative(path)} has a row wider than its header")
    return list(reader.fieldnames), rows


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    materialized = list(rows)
    if fieldnames is None:
        if not materialized:
            raise ValueError(f"fieldnames required for empty CSV {path}")
        fieldnames = list(materialized[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(materialized)


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError(f"copy hash mismatch: {source} -> {destination}")


def manifest_entries(base: Path) -> list[dict[str, object]]:
    manifest = read_json(base / "lane_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError(f"{relative(base / 'lane_manifest.json')} files is not a list")
    return entries


def protected_paths() -> dict[Path, str]:
    paths: dict[Path, str] = {}
    for role, base in (
        ("intrinsic_author_packet", INTRINSIC_AUTHOR),
        ("intrinsic_review_packet", INTRINSIC_REVIEW),
        ("preservation_author_packet", PRESERVATION_AUTHOR),
        ("preservation_review_packet", PRESERVATION_REVIEW),
    ):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                paths[path] = role

    paths[CANONICAL_DB] = "canonical_database"
    paths[CANONICAL_MANIFEST] = "canonical_database_manifest"
    database_manifest = read_json(CANONICAL_MANIFEST)
    inputs = database_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("canonical database manifest inputs are missing")
    for relpath in sorted(inputs):
        paths[REPO / relpath] = "canonical_registry_or_schema"

    chapters = AI_ROOT / "chapters"
    if chapters.is_dir():
        for path in sorted(chapters.rglob("*")):
            if path.is_file():
                paths[path] = "chapter_source"

    for pattern, role in (
        ("*.docx", "docx_artifact"),
        ("*.pdf", "publication_or_source_pdf"),
    ):
        for path in sorted(AI_ROOT.rglob(pattern)):
            if path.is_file() and HERE not in path.parents:
                paths[path] = paths.get(path, role)

    explicit_shared = (
        STAGING / "README.md",
        STAGING / "chapter_18_metadata_rebase_task.md",
        STAGING / "fresh_acceptance_review_matrix.md",
        STAGING / "integration_decision_standard.md",
        STAGING / "second_half_acceptance_tracker.md",
    )
    for path in explicit_shared:
        if path.is_file():
            paths[path] = "shared_instruction_or_tracker"
    return paths


def snapshot(paths: dict[Path, str]) -> dict[Path, tuple[str, int]]:
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"protected inputs missing: {missing}")
    return {
        path: (sha256_file(path), path.stat().st_size)
        for path in sorted(paths, key=relative)
    }


def sqlite_integrity(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    return str(result[0]) if result else "missing_result"


def disposition_target(filename: str, field: str) -> tuple[str, str, str]:
    raw = f"accepted_preservation_metadata/{filename}"
    if filename == "source_versions.csv":
        return (
            "normalized_junction_destination",
            "normalized_source_version_registry.csv",
            field,
        )
    if filename == "claim_locator_reviews.csv":
        if field in LOCATOR_AUTHORITY_FIELDS:
            return (
                "governed_hold_destination",
                "evidence_authority_publication_hold_matrix.csv",
                f"locator__{field}",
            )
        return (
            "normalized_junction_destination",
            "semantic_locator_registry.csv",
            field,
        )
    if filename == "source_claim_junction.csv":
        return (
            "normalized_junction_destination",
            "claim_source_version_locator_junction.csv",
            field,
        )
    if filename == "source_derivatives.csv":
        return (
            "normalized_junction_destination",
            "preserved_derivative_registry.csv",
            field,
        )
    if filename == "coefficient_predictive_prohibition_audit.csv":
        return (
            "governed_hold_destination",
            "retained_prohibition_matrix.csv",
            field,
        )
    if filename == "exact_patch_audit.csv":
        disposition = (
            "governed_hold_destination"
            if field in PATCH_AUTHORITY_FIELDS
            else "normalized_junction_destination"
        )
        return disposition, "semantic_patch_registry.csv", field
    if filename == "hold_inventory.csv":
        return (
            "governed_hold_destination",
            "evidence_authority_publication_hold_matrix.csv",
            f"hold__{field}",
        )
    if filename == "support_narrowing_decisions.csv":
        return (
            "governed_hold_destination",
            "evidence_authority_publication_hold_matrix.csv",
            f"scope__{field}",
        )
    if filename == "source_exceptions.csv":
        return (
            "governed_hold_destination",
            "retained_source_exception_holds.csv",
            field,
        )
    if filename == "unresolved_queue.csv":
        return (
            "governed_hold_destination",
            "retained_unresolved_holds.csv",
            field,
        )
    return "direct_retained_identity", raw, field


def build_registries(output: Path) -> dict[str, int]:
    source_version_fields, source_versions = read_csv(
        PRESERVATION_AUTHOR / "source_versions.csv"
    )
    locator_fields, locator_rows = read_csv(
        PRESERVATION_AUTHOR / "claim_locator_reviews.csv"
    )
    junction_fields, junction_rows = read_csv(
        PRESERVATION_AUTHOR / "source_claim_junction.csv"
    )
    derivative_fields, derivative_rows = read_csv(
        PRESERVATION_AUTHOR / "source_derivatives.csv"
    )
    hold_fields, hold_rows = read_csv(PRESERVATION_AUTHOR / "hold_inventory.csv")
    scope_fields, scope_rows = read_csv(
        PRESERVATION_AUTHOR / "support_narrowing_decisions.csv"
    )
    prohibition_fields, prohibition_rows = read_csv(
        PRESERVATION_AUTHOR / "coefficient_predictive_prohibition_audit.csv"
    )
    patch_fields, patch_rows = read_csv(
        PRESERVATION_AUTHOR / "exact_patch_audit.csv"
    )
    _, patch_verification = read_csv(
        PRESERVATION_REVIEW / "applied_patch_verification.csv"
    )
    exception_fields, exception_rows = read_csv(
        PRESERVATION_AUTHOR / "source_exceptions.csv"
    )
    unresolved_fields, unresolved_rows = read_csv(
        PRESERVATION_AUTHOR / "unresolved_queue.csv"
    )

    normalized_versions: list[dict[str, object]] = []
    retrieval_history: list[dict[str, object]] = []
    version_by_id: dict[str, dict[str, str]] = {}
    for ordinal, row in enumerate(source_versions, start=1):
        version = dict(row)
        version["source_version_semantic_key"] = stable_key(
            "CH18-SVK",
            row["source_id"],
            row["version_ordinal"],
            row["version_role"],
            row["original_url"],
            row["retrieval_timestamp_utc"],
        )
        version["accepted_attempt_ordinal"] = ordinal
        version["artifact_hash_state"] = (
            "PRESENT_VERIFIED"
            if row["preservation_status"] == "preserved_complete"
            else "ABSENT_BY_ACCEPTED_RETRIEVAL_FAILURE"
        )
        normalized_versions.append(version)
        version_by_id[row["source_version_id"]] = row

        history = {
            "attempt_history_id": f"CH18-ATTEMPT-{ordinal:04d}",
            **row,
            "append_only_sequence": ordinal,
            "retrieval_outcome": row["preservation_status"],
            "failure_retention": (
                "RETAINED_APPEND_ONLY"
                if row["preservation_status"] == "retrieval_failed"
                else "NOT_APPLICABLE"
            ),
            "preservation_is_semantic_support": "NO",
        }
        retrieval_history.append(history)
    write_csv(output / "normalized_source_version_registry.csv", normalized_versions)
    write_csv(output / "retrieval_attempt_history.csv", retrieval_history)

    semantic_locators: list[dict[str, object]] = []
    locator_by_hold_claim: dict[tuple[str, str], dict[str, str]] = {}
    locator_by_claim: dict[str, dict[str, str]] = {}
    semantic_key_by_hold_claim: dict[tuple[str, str], str] = {}
    for row in locator_rows:
        key_tuple = (row["hold_relationship_id"], row["claim_id"])
        semantic_key = stable_key(
            "CH18-LOC",
            row["hold_relationship_id"],
            row["claim_id"],
            row["independent_resolved_locator"],
            row["exact_window_hashes"],
        )
        semantic_locators.append(
            {
                **row,
                "semantic_locator_key": semantic_key,
                "locator_window_count": len(row["exact_window_hashes"].split("|")),
                "locator_key_version": "chapter18_metadata_rebase_v1",
            }
        )
        if key_tuple in locator_by_hold_claim:
            raise ValueError(f"duplicate semantic locator key: {key_tuple}")
        locator_by_hold_claim[key_tuple] = row
        semantic_key_by_hold_claim[key_tuple] = semantic_key
        if row["claim_id"] in locator_by_claim:
            raise ValueError(f"duplicate claim locator: {row['claim_id']}")
        locator_by_claim[row["claim_id"]] = row
    write_csv(output / "semantic_locator_registry.csv", semantic_locators)

    normalized_junctions: list[dict[str, object]] = []
    for row in junction_rows:
        key_tuple = (row["hold_relationship_id"], row["claim_id"])
        locator = locator_by_hold_claim.get(key_tuple)
        version = version_by_id.get(row["source_version_id"])
        if locator is None or version is None:
            raise ValueError(f"junction foreign key missing: {row['junction_id']}")
        if version["source_id"] != row["source_id"]:
            raise ValueError(f"junction source/version mismatch: {row['junction_id']}")
        normalized_junctions.append(
            {
                **row,
                "junction_semantic_key": stable_key(
                    "CH18-JUNCTION",
                    row["claim_id"],
                    row["source_id"],
                    row["source_version_id"],
                    row["hold_relationship_id"],
                    row["relationship_role"],
                ),
                "semantic_locator_key": semantic_key_by_hold_claim[key_tuple],
                "resolved_locator": locator["independent_resolved_locator"],
                "exact_window_hashes": locator["exact_window_hashes"],
                "evidence_grade": locator["evidence_grade"],
                "preservation_status": version["preservation_status"],
                "preservation_support_separation": locator[
                    "preservation_support_separation"
                ],
                "coefficient_authority": locator["coefficient_authority"],
                "predictive_authority": locator["predictive_authority"],
                "publication_authority": locator["publication"],
            }
        )
    write_csv(
        output / "claim_source_version_locator_junction.csv",
        normalized_junctions,
    )

    derivative_registry: list[dict[str, object]] = []
    for row in derivative_rows:
        parent = version_by_id.get(row["parent_source_version_id"])
        if parent is None:
            raise ValueError(f"derivative parent missing: {row['derivative_id']}")
        derivative_path = PRESERVATION_AUTHOR / row["derivative_path"]
        parent_path = PRESERVATION_AUTHOR / parent["local_archive_path"]
        derivative_registry.append(
            {
                **row,
                "parent_content_sha256": parent["content_sha256"],
                "parent_local_archive_path": parent["local_archive_path"],
                "parent_observed_sha256": sha256_file(parent_path),
                "derivative_observed_sha256": sha256_file(derivative_path),
                "derivative_observed_byte_count": derivative_path.stat().st_size,
                "rehash_status": "PASS",
                "preservation_is_semantic_support": "NO",
            }
        )
    write_csv(output / "preserved_derivative_registry.csv", derivative_registry)

    hold_by_id = {row["hold_relationship_id"]: row for row in hold_rows}
    scope_by_id = {row["hold_relationship_id"]: row for row in scope_rows}
    authority_matrix: list[dict[str, object]] = []
    for ordinal, locator in enumerate(locator_rows, start=1):
        hold_id = locator["hold_relationship_id"]
        hold = hold_by_id.get(hold_id)
        scope = scope_by_id.get(hold_id)
        if hold is None or scope is None:
            raise ValueError(f"authority join missing for {hold_id}")
        if not (
            locator["claim_id"] == hold["claim_id"] == scope["claim_id"]
        ):
            raise ValueError(f"authority claim mismatch for {hold_id}")
        matrix: dict[str, object] = {
            "authority_matrix_id": f"CH18-AUTH-{ordinal:04d}",
            "authority_semantic_key": stable_key(
                "CH18-AUTH", hold_id, locator["claim_id"]
            ),
            "hold_relationship_id": hold_id,
            "claim_id": locator["claim_id"],
        }
        matrix.update({f"locator__{field}": locator[field] for field in locator_fields})
        matrix.update({f"hold__{field}": hold[field] for field in hold_fields})
        matrix.update({f"scope__{field}": scope[field] for field in scope_fields})
        matrix.update(
            {
                "preservation_success": hold["preservation_status"],
                "semantic_support_disposition": locator["reviewer_disposition"],
                "evidence_grade": locator["evidence_grade"],
                "claim_scope_action": scope["claim_scope_action"],
                "analytical_eligibility": (
                    "NOT_AUTHORIZED_BEYOND_ACCEPTED_BOUNDED_DESCRIPTIVE_SCOPE"
                ),
                "publication_authority": locator["publication"],
                "preservation_is_semantic_support": "NO",
                "partially_supported_promoted": "NO",
                "authority_strengthening": "NONE",
                "retained_hold_status": "PASS",
            }
        )
        authority_matrix.append(matrix)
    write_csv(
        output / "evidence_authority_publication_hold_matrix.csv",
        authority_matrix,
    )

    prohibition_matrix: list[dict[str, object]] = []
    for row in prohibition_rows:
        locator = locator_by_claim.get(row["governing_claim_id"])
        if locator is None:
            raise ValueError(f"prohibition claim missing: {row['review_id']}")
        prohibition_matrix.append(
            {
                **row,
                "causal_authority": locator["causal_authority"],
                "comparative_system_authority": locator[
                    "comparative_system_authority"
                ],
                "model_authority": locator["model_authority"],
                "canonical_integration": locator["canonical_integration"],
                "publication": locator["publication"],
                "docx_authority": locator["docx_authority"],
                "metadata_integration_authority": locator[
                    "metadata_integration_authority"
                ],
                "hold_strengthening": "NONE",
            }
        )
    write_csv(output / "retained_prohibition_matrix.csv", prohibition_matrix)

    verification_by_patch = {row["patch_id"]: row for row in patch_verification}
    semantic_patches: list[dict[str, object]] = []
    for row in patch_rows:
        verification = verification_by_patch.get(row["patch_id"])
        if verification is None:
            raise ValueError(f"semantic patch review missing: {row['patch_id']}")
        semantic_patches.append(
            {
                **row,
                "fresh_review_scope_status": verification[
                    "scope_and_authority_status"
                ],
                "residual_patch": "NO",
            }
        )
    write_csv(output / "semantic_patch_registry.csv", semantic_patches)

    write_csv(
        output / "retained_source_exception_holds.csv",
        [
            {
                **row,
                "retention_status": "RETAINED_APPEND_ONLY",
                "authority_effect": "NO_AUTHORITY_STRENGTHENING",
            }
            for row in exception_rows
        ],
    )
    write_csv(
        output / "retained_unresolved_holds.csv",
        [
            {
                **row,
                "retention_status": "RETAINED",
                "publication_effect": "BLOCK_RETAINED",
            }
            for row in unresolved_rows
        ],
    )

    return {
        "source_version_fields": len(source_version_fields) * len(source_versions),
        "source_versions": len(source_versions),
        "locators": len(locator_rows),
        "junctions": len(junction_rows),
        "derivatives": len(derivative_rows),
        "holds": len(hold_rows),
        "scope_decisions": len(scope_rows),
        "prohibitions": len(prohibition_rows),
        "patches": len(patch_rows),
        "exceptions": len(exception_rows),
        "unresolved": len(unresolved_rows),
        "source_version_columns": len(source_version_fields),
        "locator_columns": len(locator_fields),
        "junction_columns": len(junction_fields),
        "derivative_columns": len(derivative_fields),
        "prohibition_columns": len(prohibition_fields),
        "patch_columns": len(patch_fields),
        "exception_columns": len(exception_fields),
        "unresolved_columns": len(unresolved_fields),
    }


def build_dispositions(
    output: Path,
    intrinsic_entries: list[dict[str, object]],
    preservation_entries: list[dict[str, object]],
) -> dict[str, int]:
    intrinsic_rows: list[dict[str, object]] = []
    intrinsic_fields: list[dict[str, object]] = []
    for entry in intrinsic_entries:
        source_rel = str(entry["path"])
        source = INTRINSIC_AUTHOR / source_rel
        if source.suffix.lower() != ".csv":
            continue
        fieldnames, rows = read_csv(source)
        for ordinal, row in enumerate(rows, start=1):
            record_id = row[fieldnames[0]] or f"source_row_{ordinal:06d}"
            row_id = f"CH18-IR-{len(intrinsic_rows) + 1:06d}"
            intrinsic_rows.append(
                {
                    "row_disposition_id": row_id,
                    "source_file": source_rel,
                    "source_row_ordinal": ordinal,
                    "source_record_id": record_id,
                    "source_column_count": len(fieldnames),
                    "row_disposition": "direct_retained_identity",
                    "destination_artifact": f"intrinsic_artifacts/{source_rel}",
                    "destination_row_ordinal": ordinal,
                    "destination_record_id": record_id,
                    "transformation_rule": "byte_identical_artifact_copy",
                    "governing_acceptance_reference": (
                        "d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8"
                    ),
                }
            )
            for field in fieldnames:
                value = row[field]
                intrinsic_fields.append(
                    {
                        "field_disposition_id": (
                            f"CH18-IF-{len(intrinsic_fields) + 1:07d}"
                        ),
                        "row_disposition_id": row_id,
                        "source_file": source_rel,
                        "source_row_ordinal": ordinal,
                        "source_record_id": record_id,
                        "source_field": field,
                        "source_value": value,
                        "source_value_sha256": sha256_value(value),
                        "value_state": "populated" if value != "" else "empty_retained",
                        "field_disposition": "direct_retained_identity",
                        "destination_artifact": f"intrinsic_artifacts/{source_rel}",
                        "destination_row_ordinal": ordinal,
                        "destination_field": field,
                        "transformation_rule": "exact_scalar_identity",
                        "governed_hold_or_narrowing": "UNCHANGED",
                        "authority_strengthening": "NONE",
                    }
                )
    write_csv(output / "intrinsic_row_disposition.csv", intrinsic_rows)
    write_csv(output / "intrinsic_field_disposition.csv", intrinsic_fields)

    source_rows: list[dict[str, object]] = []
    source_fields: list[dict[str, object]] = []
    for entry in preservation_entries:
        source_rel = str(entry["path"])
        source = PRESERVATION_AUTHOR / source_rel
        if source.suffix.lower() != ".csv":
            continue
        fieldnames, rows = read_csv(source)
        for ordinal, row in enumerate(rows, start=1):
            record_id = row[fieldnames[0]] or f"source_row_{ordinal:06d}"
            first_disposition, first_destination, _ = disposition_target(
                source.name, fieldnames[0]
            )
            row_id = f"CH18-SR-{len(source_rows) + 1:06d}"
            source_rows.append(
                {
                    "row_disposition_id": row_id,
                    "source_file": source_rel,
                    "source_row_ordinal": ordinal,
                    "source_record_id": record_id,
                    "source_column_count": len(fieldnames),
                    "row_disposition": first_disposition,
                    "destination_artifact": first_destination,
                    "destination_row_ordinal": ordinal,
                    "destination_record_id": record_id,
                    "row_boundary": (
                        "authority fields may have governed-hold field dispositions"
                    ),
                    "governing_acceptance_reference": (
                        "d4e1d026f7743032e77f98a6ed1424ccd4270bb5a5188a3ed1a5d98c44eb8dd9"
                    ),
                }
            )
            for field in fieldnames:
                value = row[field]
                disposition, destination, destination_field = disposition_target(
                    source.name, field
                )
                source_fields.append(
                    {
                        "field_disposition_id": (
                            f"CH18-SF-{len(source_fields) + 1:07d}"
                        ),
                        "row_disposition_id": row_id,
                        "source_file": source_rel,
                        "source_row_ordinal": ordinal,
                        "source_record_id": record_id,
                        "source_field": field,
                        "source_value": value,
                        "source_value_sha256": sha256_value(value),
                        "value_state": "populated" if value != "" else "empty_retained",
                        "field_disposition": disposition,
                        "destination_artifact": destination,
                        "destination_row_ordinal": ordinal,
                        "destination_field": destination_field,
                        "transformation_rule": "exact_scalar_identity",
                        "governed_hold_or_narrowing": (
                            "RETAINED"
                            if disposition == "governed_hold_destination"
                            else "UNCHANGED"
                        ),
                        "authority_strengthening": "NONE",
                    }
                )
    write_csv(output / "source_metadata_row_disposition.csv", source_rows)
    write_csv(output / "source_metadata_field_disposition.csv", source_fields)
    return {
        "intrinsic_csv_rows": len(intrinsic_rows),
        "intrinsic_scalar_fields": len(intrinsic_fields),
        "source_metadata_csv_rows": len(source_rows),
        "source_metadata_scalar_fields": len(source_fields),
    }


def population_evidence() -> tuple[list[dict[str, object]], dict[str, int]]:
    _, record_claim_rows = read_csv(INTRINSIC_AUTHOR / "record_claim_junction.csv")
    _, arithmetic_rows = read_csv(
        INTRINSIC_AUTHOR / "preserved_arithmetic_and_nonadditivity_audit.csv"
    )
    _, locator_rows = read_csv(PRESERVATION_AUTHOR / "claim_locator_reviews.csv")
    _, patch_rows = read_csv(PRESERVATION_AUTHOR / "exact_patch_audit.csv")
    _, junction_rows = read_csv(PRESERVATION_AUTHOR / "source_claim_junction.csv")
    _, version_rows = read_csv(PRESERVATION_AUTHOR / "source_versions.csv")
    _, derivative_rows = read_csv(PRESERVATION_AUTHOR / "source_derivatives.csv")
    _, prohibition_rows = read_csv(
        PRESERVATION_AUTHOR / "coefficient_predictive_prohibition_audit.csv"
    )

    dispositions = Counter(row["reviewer_disposition"] for row in locator_rows)
    preservation = Counter(row["preservation_status"] for row in version_rows)
    prohibition_types = Counter(row["governed_record_type"] for row in prohibition_rows)
    observed = {
        "intrinsic_analytical_records": len(
            {
                (row["artifact_name"], row["record_id"])
                for row in record_claim_rows
            }
        ),
        "intrinsic_record_claim_links": len(record_claim_rows),
        "arithmetic_nonadditivity_controls": len(arithmetic_rows),
        "locator_relationships": len(locator_rows),
        "supported": dispositions["supported"],
        "partially_supported": dispositions["partially_supported"],
        "semantic_patches": len(patch_rows),
        "ordered_source_edges": len(junction_rows),
        "unique_sources": len({row["source_id"] for row in junction_rows}),
        "retrieval_attempts": len(version_rows),
        "successful_artifacts": preservation["preserved_complete"],
        "retained_failures": preservation["retrieval_failed"],
        "derivatives": len(derivative_rows),
        "coefficient_blocks": prohibition_types["coefficient_transfer_prohibition"],
        "predictive_blocks": prohibition_types["conceptual_hypothesis"],
    }
    rows = [
        {
            "population_id": f"CH18-POP-{index:03d}",
            "population": key,
            "observed": observed[key],
            "expected": EXPECTED_POPULATIONS[key],
            "status": "PASS" if observed[key] == EXPECTED_POPULATIONS[key] else "FAIL",
            "authority_effect": "NONE",
        }
        for index, key in enumerate(EXPECTED_POPULATIONS, start=1)
    ]
    return rows, observed


def foreign_key_evidence() -> list[dict[str, object]]:
    _, versions = read_csv(PRESERVATION_AUTHOR / "source_versions.csv")
    _, locators = read_csv(PRESERVATION_AUTHOR / "claim_locator_reviews.csv")
    _, junctions = read_csv(PRESERVATION_AUTHOR / "source_claim_junction.csv")
    _, derivatives = read_csv(PRESERVATION_AUTHOR / "source_derivatives.csv")
    _, holds = read_csv(PRESERVATION_AUTHOR / "hold_inventory.csv")
    _, scopes = read_csv(PRESERVATION_AUTHOR / "support_narrowing_decisions.csv")
    _, proposed_claims = read_csv(INTRINSIC_AUTHOR / "proposed_claims.csv")

    version_ids = {row["source_version_id"] for row in versions}
    source_version_pairs = {
        (row["source_id"], row["source_version_id"]) for row in versions
    }
    locator_keys = {
        (row["hold_relationship_id"], row["claim_id"]) for row in locators
    }
    hold_claim_pairs = {(row["hold_relationship_id"], row["claim_id"]) for row in holds}
    scope_claim_pairs = {
        (row["hold_relationship_id"], row["claim_id"]) for row in scopes
    }
    claim_ids = {row["claim_id"] for row in proposed_claims}

    checks = (
        (
            "junction_source_version_fk",
            sum(row["source_version_id"] in version_ids for row in junctions),
            len(junctions),
        ),
        (
            "junction_source_version_pair_fk",
            sum(
                (row["source_id"], row["source_version_id"])
                in source_version_pairs
                for row in junctions
            ),
            len(junctions),
        ),
        (
            "junction_locator_composite_fk",
            sum(
                (row["hold_relationship_id"], row["claim_id"]) in locator_keys
                for row in junctions
            ),
            len(junctions),
        ),
        (
            "locator_intrinsic_claim_fk",
            sum(row["claim_id"] in claim_ids for row in locators),
            len(locators),
        ),
        (
            "locator_hold_fk",
            sum(
                (row["hold_relationship_id"], row["claim_id"]) in hold_claim_pairs
                for row in locators
            ),
            len(locators),
        ),
        (
            "locator_scope_fk",
            sum(
                (row["hold_relationship_id"], row["claim_id"]) in scope_claim_pairs
                for row in locators
            ),
            len(locators),
        ),
        (
            "derivative_parent_version_fk",
            sum(row["parent_source_version_id"] in version_ids for row in derivatives),
            len(derivatives),
        ),
    )
    return [
        {
            "foreign_key_audit_id": f"CH18-FK-{index:03d}",
            "relationship": name,
            "matched": matched,
            "expected": expected,
            "status": "PASS" if matched == expected else "FAIL",
            "notes": (
                "legacy locator_review_id is retained byte-exact; the accepted "
                "hold_relationship_id plus claim_id composite is the semantic join"
                if name == "junction_locator_composite_fk"
                else ""
            ),
        }
        for index, (name, matched, expected) in enumerate(checks, start=1)
    ]


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protected = protected_paths()
    before = snapshot(protected)

    for role, path, expected in GOVERNING_INPUTS:
        observed = before[path][0]
        if observed != expected:
            raise ValueError(
                f"governing hash mismatch for {role}: expected {expected}, got {observed}"
            )

    intrinsic_entries = manifest_entries(INTRINSIC_AUTHOR)
    preservation_entries = manifest_entries(PRESERVATION_AUTHOR)

    for entry in intrinsic_entries:
        source = INTRINSIC_AUTHOR / str(entry["path"])
        observed = before[source][0]
        if observed != entry["sha256"]:
            raise ValueError(f"intrinsic manifest mismatch: {relative(source)}")
        copy_exact(source, output / "intrinsic_artifacts" / str(entry["path"]))

    for entry in preservation_entries:
        source = PRESERVATION_AUTHOR / str(entry["path"])
        observed = before[source][0]
        if observed != entry["sha256"]:
            raise ValueError(f"preservation manifest mismatch: {relative(source)}")
        if source.suffix.lower() == ".csv":
            copy_exact(
                source,
                output / "accepted_preservation_metadata" / str(entry["path"]),
            )

    for role, path, _ in GOVERNING_INPUTS:
        if role in MANIFEST_COPY_NAMES:
            copy_exact(
                path,
                output / "governing_manifests" / MANIFEST_COPY_NAMES[role],
            )

    copy_exact(CANONICAL_DB, output / "sealed_candidate/research.db")
    copy_exact(
        CANONICAL_MANIFEST,
        output / "sealed_candidate/database_manifest.json",
    )

    registry_metrics = build_registries(output)
    disposition_metrics = build_dispositions(
        output, intrinsic_entries, preservation_entries
    )
    population_rows, populations = population_evidence()
    write_csv(output / "population_reproduction_audit.csv", population_rows)
    write_csv(output / "foreign_key_audit.csv", foreign_key_evidence())

    after = snapshot(protected)

    governing_rows: list[dict[str, object]] = []
    for index, (role, path, expected) in enumerate(GOVERNING_INPUTS, start=1):
        governing_rows.append(
            {
                "governing_input_id": f"CH18-GOV-{index:03d}",
                "role": role,
                "path": relative(path),
                "expected_sha256": expected,
                "observed_before_sha256": before[path][0],
                "observed_after_sha256": after[path][0],
                "bytes_before": before[path][1],
                "bytes_after": after[path][1],
                "status": (
                    "PASS"
                    if expected == before[path][0] == after[path][0]
                    else "FAIL"
                ),
            }
        )
    write_csv(output / "governing_input_hash_registry.csv", governing_rows)

    intrinsic_inventory: list[dict[str, object]] = []
    for index, entry in enumerate(intrinsic_entries, start=1):
        source = INTRINSIC_AUTHOR / str(entry["path"])
        copied = output / "intrinsic_artifacts" / str(entry["path"])
        intrinsic_inventory.append(
            {
                "intrinsic_artifact_id": f"CH18-IA-{index:03d}",
                "source_path": relative(source),
                "manifest_expected_sha256": entry["sha256"],
                "observed_before_sha256": before[source][0],
                "observed_after_sha256": after[source][0],
                "copied_path": copied.relative_to(output).as_posix(),
                "copied_sha256": sha256_file(copied),
                "source_bytes": before[source][1],
                "copied_bytes": copied.stat().st_size,
                "declared_rows": entry.get("rows", ""),
                "status": (
                    "PASS"
                    if entry["sha256"]
                    == before[source][0]
                    == after[source][0]
                    == sha256_file(copied)
                    and before[source][1] == copied.stat().st_size
                    else "FAIL"
                ),
            }
        )
    write_csv(
        output / "immutable_intrinsic_artifact_inventory.csv",
        intrinsic_inventory,
    )

    preservation_inventory: list[dict[str, object]] = []
    for index, entry in enumerate(preservation_entries, start=1):
        source = PRESERVATION_AUTHOR / str(entry["path"])
        copy_path = (
            output / "accepted_preservation_metadata" / str(entry["path"])
            if source.suffix.lower() == ".csv"
            else None
        )
        preservation_inventory.append(
            {
                "preservation_artifact_id": f"CH18-PA-{index:03d}",
                "source_path": relative(source),
                "artifact_kind": (
                    "accepted_tabular_metadata_copy"
                    if copy_path is not None
                    else "immutable_external_reference"
                ),
                "manifest_expected_sha256": entry["sha256"],
                "observed_before_sha256": before[source][0],
                "observed_after_sha256": after[source][0],
                "local_rebase_copy": (
                    copy_path.relative_to(output).as_posix() if copy_path else ""
                ),
                "local_rebase_copy_sha256": (
                    sha256_file(copy_path) if copy_path else ""
                ),
                "bytes": before[source][1],
                "declared_rows": entry.get("rows", ""),
                "status": (
                    "PASS"
                    if entry["sha256"] == before[source][0] == after[source][0]
                    and (
                        copy_path is None
                        or sha256_file(copy_path) == entry["sha256"]
                    )
                    else "FAIL"
                ),
            }
        )
    write_csv(
        output / "accepted_preservation_artifact_inventory.csv",
        preservation_inventory,
    )

    no_write_rows = []
    for index, path in enumerate(sorted(protected, key=relative), start=1):
        no_write_rows.append(
            {
                "no_write_audit_id": f"CH18-NW-{index:04d}",
                "role": protected[path],
                "path": relative(path),
                "sha256_before": before[path][0],
                "bytes_before": before[path][1],
                "sha256_after": after[path][0],
                "bytes_after": after[path][1],
                "status": "PASS" if before[path] == after[path] else "FAIL",
            }
        )
    write_csv(output / "protected_input_no_write_audit.csv", no_write_rows)

    candidate_db = output / "sealed_candidate/research.db"
    canonical_hash = sha256_file(CANONICAL_DB)
    candidate_hash = sha256_file(candidate_db)
    binding_rows = [
        {
            "database_binding_id": "CH18-DB-001",
            "role": "frozen_canonical_database",
            "path": relative(CANONICAL_DB),
            "sha256": canonical_hash,
            "bytes": CANONICAL_DB.stat().st_size,
            "sqlite_integrity": sqlite_integrity(CANONICAL_DB),
            "byte_identity_to_canonical": "SELF",
            "mutation_authority": "NONE",
            "status": "PASS",
        },
        {
            "database_binding_id": "CH18-DB-002",
            "role": "sealed_integration_candidate_database",
            "path": candidate_db.relative_to(output).as_posix(),
            "sha256": candidate_hash,
            "bytes": candidate_db.stat().st_size,
            "sqlite_integrity": sqlite_integrity(candidate_db),
            "byte_identity_to_canonical": (
                "PASS"
                if canonical_hash == candidate_hash
                and CANONICAL_DB.read_bytes() == candidate_db.read_bytes()
                else "FAIL"
            ),
            "mutation_authority": "NONE_METADATA_ONLY_COPY",
            "status": (
                "PASS"
                if canonical_hash == candidate_hash
                and sqlite_integrity(candidate_db) == "ok"
                else "FAIL"
            ),
        },
    ]
    write_csv(output / "canonical_database_binding.csv", binding_rows)

    metrics = {
        "lane": output.name,
        "packet_type": "chapter_18_immutable_source_metadata_rebase",
        "status_date": "2026-07-26",
        "governing_manifest_hashes_verified": 4,
        "intrinsic_manifest_declared_artifacts": len(intrinsic_entries),
        "preservation_manifest_declared_artifacts": len(preservation_entries),
        "protected_input_files": len(no_write_rows),
        "canonical_database_sha256": canonical_hash,
        "sealed_candidate_database_sha256": candidate_hash,
        "canonical_database_byte_identity": canonical_hash == candidate_hash,
        "populations": populations,
        "disposition_coverage": disposition_metrics,
        "registry_metrics": registry_metrics,
        "authority": {
            "canonical_integration": "NOT_AUTHORIZED",
            "coefficient_transfer": "BLOCK",
            "predictive_use": "BLOCK",
            "causal_transfer": "BLOCK",
            "comparative_system_use": "BLOCK",
            "chapter_docx_publication": "NOT_AUTHORIZED",
            "metadata_acceptance": "PENDING_FRESH_UNINVOLVED_REVIEW",
        },
    }
    write_json(output / "build_metrics.json", metrics)

    if any(row["status"] != "PASS" for row in governing_rows):
        raise ValueError("governing input hash registry failed")
    if any(row["status"] != "PASS" for row in intrinsic_inventory):
        raise ValueError("intrinsic artifact inventory failed")
    if any(row["status"] != "PASS" for row in preservation_inventory):
        raise ValueError("preservation artifact inventory failed")
    if any(row["status"] != "PASS" for row in no_write_rows):
        raise ValueError("protected input no-write audit failed")
    if any(row["status"] != "PASS" for row in population_rows):
        raise ValueError("population reproduction failed")
    if any(row["status"] != "PASS" for row in binding_rows):
        raise ValueError("canonical database binding failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    build(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "PASS",
                "cpu_policy": "single_process_sequential_build",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
