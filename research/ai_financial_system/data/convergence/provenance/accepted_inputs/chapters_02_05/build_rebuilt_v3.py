#!/usr/bin/env python3
"""Build the metadata-only Chapters 2-5 rebuilt-v3 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_NAMES = {
    "build_rebuilt_v3.py",
    "validate_rebuilt_v3.py",
    "verify_two_builds.py",
    "finalize_rebuilt_v3.py",
}
SNAPSHOT = "SPU-V2-20260726-38b990b2"
CHAPTERS = {"CH02": 243, "CH03": 40, "CH04": 47, "CH05": 31}


def find_repo(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "research/ai_financial_system"
        ).is_dir():
            return path
    raise RuntimeError("repository root not found")


HERE = Path(__file__).resolve().parent
REPO = find_repo(Path.cwd())
AI = REPO / "research/ai_financial_system"
STAGING = AI / "staging"
V2 = STAGING / "lane_3_chapters_02_05_rebuilt_v2"
V2_REVIEW = STAGING / "lane_4_verify_chapters_02_05_rebuilt_v2"
UNION = STAGING / "lane_3_source_preservation_union_v2"
UNION_REVIEW = STAGING / "lane_2_verify_source_preservation_union_v2"
TASK = STAGING / "chapters_02_05_rebuilt_v3_task.md"
DOWNSTREAM = {
    "CH16": STAGING / "lane_2_verify_chapter_16_rebuilt_v3",
    "CH17": STAGING / "lane_1_verify_chapter_17_atomic_rebase",
    "CH18": STAGING / "lane_2_verify_chapter_18_rebuilt_v5",
    "CH19": STAGING / "lane_2_verify_chapter_19_rebuilt_v3",
}
COPY_ROOT = [
    "arithmetic_nonadditivity_audit.csv",
    "chapter_html_projection_audit.csv",
    "chapter_html_repair_audit.csv",
    "chapter_revision_summary.csv",
    "claim_chronology_repair_audit.csv",
    "confidence_repair_history.csv",
    "contradiction_repair_audit.csv",
    "economic_importance_repair_audit.csv",
    "exact_patch_disposition.csv",
    "falsifier_audit.csv",
    "fresh_review_exact_row_patches.csv",
    "fresh_review_findings.csv",
    "fresh_review_writing_boundary_audit.csv",
    "handoff_blocked_field_audit.csv",
    "identity_overlap_audit.csv",
    "source_join_repair_audit.csv",
    "unknown_population_audit.csv",
    "writing_gate_audit.csv",
]
STALE_FILES = [
    (
        "claim_source_support_junction.csv",
        "source_preservation_state;author_validation_state",
        396,
        "intrinsic_claim_source_support_junction.csv and normalized_preservation_projection_references.csv",
    ),
    (
        "complete_claim_source_audit.csv",
        "author_semantic_state;external_source_preservation",
        278,
        "intrinsic_complete_claim_source_audit.csv and normalized_preservation_projection_references.csv",
    ),
    (
        "claim_source_locator_audit.csv",
        "all_source_urls_canonical_or_archived;external_source_persistence",
        278,
        "intrinsic_claim_source_locator_audit.csv and normalized_preservation_projection_references.csv",
    ),
    (
        "external_holds_audit.csv",
        "current_condition",
        6,
        "current_governance_holds.csv and downstream_dependency_audit.csv",
    ),
    (
        "external_primary_source_sample_audit.csv",
        "immutable_local_snapshot",
        10,
        "normalized_preservation_projection_references.csv",
    ),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"bad CSV header: {path}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"malformed CSV row: {path}")
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def governing_paths() -> list[Path]:
    paths = {
        TASK,
        AI / "source_preservation_standard.md",
        REPO / "reports/ai_financial_system/research.db",
    }
    for folder in (V2, V2_REVIEW, UNION, UNION_REVIEW, *DOWNSTREAM.values()):
        paths.update(path for path in folder.rglob("*") if path.is_file())
    return sorted(paths)


def protected_paths() -> list[Path]:
    paths = {REPO / "reports/ai_financial_system/research.db"}
    for folder in (
        V2,
        V2_REVIEW,
        UNION,
        UNION_REVIEW,
        *DOWNSTREAM.values(),
        AI / "data",
        AI / "chapters",
        AI / "sources",
    ):
        paths.update(path for path in folder.rglob("*") if path.is_file())
    return sorted(paths)


def file_state(paths: list[Path]) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(REPO).as_posix(): (digest(path), path.stat().st_size)
        for path in paths
    }


def copy_intrinsic(output: Path) -> list[dict[str, Any]]:
    inventory = []
    intrinsic = output / "accepted_intrinsic_v2"
    for name in COPY_ROOT:
        source = V2 / name
        destination = intrinsic / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        inventory.append(
            {
                "artifact_id": f"INTR-0205-{len(inventory)+1:03d}",
                "relative_path": destination.relative_to(output).as_posix(),
                "v2_sha256": digest(source),
                "v3_sha256": digest(destination),
                "byte_count": destination.stat().st_size,
                "treatment": "byte_identical_accepted_intrinsic",
            }
        )
    for chapter in range(2, 6):
        source_dir = V2 / f"chapter_{chapter:02d}"
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.name == "validation_report.md":
                continue
            destination = intrinsic / f"chapter_{chapter:02d}" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            inventory.append(
                {
                    "artifact_id": f"INTR-0205-{len(inventory)+1:03d}",
                    "relative_path": destination.relative_to(output).as_posix(),
                    "v2_sha256": digest(source),
                    "v3_sha256": digest(destination),
                    "byte_count": destination.stat().st_size,
                    "treatment": "byte_identical_accepted_intrinsic",
                }
            )
    return inventory


def transform_intrinsic_tables(output: Path) -> list[dict[str, Any]]:
    transformations = []

    source = V2 / "claim_source_support_junction.csv"
    rows = read_csv(source)
    fields = [
        field
        for field in rows[0]
        if field not in {"source_preservation_state", "author_validation_state"}
    ]
    fields += ["intrinsic_review_state", "canonical_integration_status"]
    transformed = [
        {
            **{field: row[field] for field in fields if field in row},
            "intrinsic_review_state": "ACCEPTED_INTRINSIC_BY_FRESH_V2_REVIEW",
            "canonical_integration_status": "NOT_AUTHORIZED",
        }
        for row in rows
    ]
    target = output / "intrinsic_claim_source_support_junction.csv"
    write_csv(target, transformed, fields)
    transformations.append(
        {
            "transformation_id": "META-0205-001",
            "source_file": source.name,
            "source_rows": len(rows),
            "removed_fields": "source_preservation_state;author_validation_state",
            "current_file": target.name,
            "current_rows": len(transformed),
            "intrinsic_identity_preserved": "yes",
        }
    )

    source = V2 / "complete_claim_source_audit.csv"
    rows = read_csv(source)
    fields = [
        field
        for field in rows[0]
        if field not in {"external_source_preservation", "author_semantic_state"}
    ]
    fields += ["intrinsic_review_state", "canonical_integration_status"]
    transformed = [
        {
            **{field: row[field] for field in fields if field in row},
            "intrinsic_review_state": "ACCEPTED_INTRINSIC_BY_FRESH_V2_REVIEW",
            "canonical_integration_status": "NOT_AUTHORIZED",
        }
        for row in rows
    ]
    target = output / "intrinsic_complete_claim_source_audit.csv"
    write_csv(target, transformed, fields)
    transformations.append(
        {
            "transformation_id": "META-0205-002",
            "source_file": source.name,
            "source_rows": len(rows),
            "removed_fields": "external_source_preservation;author_semantic_state",
            "current_file": target.name,
            "current_rows": len(transformed),
            "intrinsic_identity_preserved": "yes",
        }
    )

    source = V2 / "claim_source_locator_audit.csv"
    rows = read_csv(source)
    fields = [
        field
        for field in rows[0]
        if field
        not in {"all_source_urls_canonical_or_archived", "external_source_persistence"}
    ]
    fields += ["intrinsic_review_state", "canonical_integration_status"]
    transformed = [
        {
            **{field: row[field] for field in fields if field in row},
            "intrinsic_review_state": "ACCEPTED_INTRINSIC_BY_FRESH_V2_REVIEW",
            "canonical_integration_status": "NOT_AUTHORIZED",
        }
        for row in rows
    ]
    target = output / "intrinsic_claim_source_locator_audit.csv"
    write_csv(target, transformed, fields)
    transformations.append(
        {
            "transformation_id": "META-0205-003",
            "source_file": source.name,
            "source_rows": len(rows),
            "removed_fields": (
                "all_source_urls_canonical_or_archived;external_source_persistence"
            ),
            "current_file": target.name,
            "current_rows": len(transformed),
            "intrinsic_identity_preserved": "yes",
        }
    )
    return transformations


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in list(output.iterdir()):
        if path.is_file() and path.name in SCRIPT_NAMES:
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    accepted_union_manifest = UNION_REVIEW / "lane_manifest.json"
    if digest(accepted_union_manifest) != (
        "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
    ):
        raise RuntimeError("accepted union-v2 review manifest drifted")
    accepted = json.loads(accepted_union_manifest.read_text())
    if not accepted["validation"]["passed"]:
        raise RuntimeError("union-v2 lacks fresh acceptance")

    before = file_state(protected_paths())
    inputs = [
        {
            "input_id": f"INPUT-0205V3-{index:04d}",
            "path": path.relative_to(REPO).as_posix(),
            "sha256": digest(path),
            "byte_count": path.stat().st_size,
        }
        for index, path in enumerate(governing_paths(), 1)
    ]
    write_csv(output / "sealed_input_inventory.csv", inputs, list(inputs[0]))

    intrinsic_inventory = copy_intrinsic(output)
    write_csv(
        output / "intrinsic_artifact_inventory.csv",
        intrinsic_inventory,
        list(intrinsic_inventory[0]),
    )
    transformations = transform_intrinsic_tables(output)
    write_csv(
        output / "metadata_transformation_audit.csv",
        transformations,
        list(transformations[0]),
    )

    projection = [
        row
        for row in read_csv(UNION / "chapter_02_09_support_state_projection.csv")
        if row["chapter_id"] in CHAPTERS
    ]
    write_csv(
        output / "normalized_preservation_projection_references.csv",
        projection,
        list(projection[0]),
    )

    superseded = []
    for index, (name, fields, row_count, replacement) in enumerate(STALE_FILES, 1):
        source = V2 / name
        superseded.append(
            {
                "superseded_metadata_id": f"SUPMETA-0205-{index:03d}",
                "historical_file": name,
                "historical_file_sha256": digest(source),
                "historical_rows": row_count,
                "superseded_fields": fields,
                "reason": "copied mutable preservation or review metadata was accepted intrinsically but rejected as stale current state",
                "current_replacement": replacement,
                "historical_rows_copied_as_current": "no",
            }
        )
    write_csv(
        output / "superseded_metadata_audit.csv",
        superseded,
        list(superseded[0]),
    )

    stale_source_ids = [
        "SRC-0010",
        "SRC-0011",
        "SRC-0012",
        "SRC-0013",
        "SRC-0014",
        "SRC-0018",
        "SRC-0064",
        "SRC-0078",
    ]
    change_rows = []
    projection_by_source = {}
    for row in projection:
        projection_by_source.setdefault(row["source_id"], []).append(row)
    for source_id in stale_source_ids:
        rows = projection_by_source.get(source_id, [])
        change_rows.append(
            {
                "change_id": f"PRES-0205-{len(change_rows)+1:03d}",
                "source_id": source_id,
                "historical_v2_condition": "copied_external_content_hash_hold_or_sample_flag",
                "union_v2_projection_rows": len(rows),
                "current_dispositions": "|".join(
                    sorted({row["source_state_disposition"] for row in rows})
                ),
                "snapshot_id": SNAPSHOT,
                "semantic_clearance": "none",
                "publication_authority": "none",
            }
        )
    write_csv(
        output / "eight_source_preservation_change_audit.csv",
        change_rows,
        list(change_rows[0]),
    )

    internal = [row for row in projection if row["source_id"] == "SRC-0094"]
    internal_audit = [
        {
            "classification_audit_id": f"SRC0094-0205-{index:03d}",
            "projection_row_id": row["projection_row_id"],
            "claim_id": row["claim_id"],
            "chapter_id": row["chapter_id"],
            "source_version_id": row["source_version_id"],
            "source_state_disposition": row["source_state_disposition"],
            "external_source_hold": "no",
            "result": (
                "PASS_INTERNAL_P4"
                if row["source_state_disposition"] == "internal_P4_reconstruction"
                else "FAIL"
            ),
        }
        for index, row in enumerate(internal, 1)
    ]
    write_csv(
        output / "src_0094_internal_classification_audit.csv",
        internal_audit,
        list(internal_audit[0]),
    )

    dependency_specs = {
        "CH16": {
            "intrinsic_state": "ACCEPT_INTRINSIC_PACKET",
            "bounded_state": "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS",
            "holds": "297 role holds;59 equivalence holds;427 free-text claim holds;topology-only weights;no first-loss inference",
        },
        "CH17": {
            "intrinsic_state": "ACCEPT_INTRINSIC_PACKET",
            "bounded_state": "ACCEPT_NORMALIZED_CROSSWALK_WITH_EXPLICIT_EXTERNAL_HOLDS",
            "holds": "78 rows remain prose-ineligible and integration-ineligible;predictive probability and expected-loss use blocked",
        },
        "CH18": {
            "intrinsic_state": "ACCEPT_INTRINSIC_WITH_EXPLICIT_EXTERNAL_HOLDS",
            "bounded_state": "ACCEPT_BOUNDED_DESCRIPTIVE_STAGING_USE",
            "holds": "58 external source-content holds;coefficient transfer blocked;predictive use blocked;comparative publication blocked",
        },
        "CH19": {
            "intrinsic_state": "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE",
            "bounded_state": "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS",
            "holds": "operational gates open;resolved validation findings not authorized;publication as findings blocked",
        },
    }
    dependencies = []
    for chapter, folder in DOWNSTREAM.items():
        report_name = (
            "verification_report.md"
            if (folder / "verification_report.md").exists()
            else "acceptance_report.md"
        )
        dependencies.append(
            {
                "dependency_id": f"DEP-0205-{chapter}",
                "downstream_chapter_id": chapter,
                **dependency_specs[chapter],
                "review_packet": folder.name,
                "review_manifest_sha256": digest(folder / "lane_manifest.json"),
                "review_report_sha256": digest(folder / report_name),
                "canonical_integration": "NOT_AUTHORIZED",
                "publication": "BLOCKED",
                "handoff_readiness_change": "none",
            }
        )
    write_csv(
        output / "downstream_dependency_audit.csv",
        dependencies,
        list(dependencies[0]),
    )

    holds = [
        {
            "hold_id": "HOLD-0205V3-001",
            "scope": "source preservation",
            "condition": "union-v2 preservation states are references, not semantic locator clearance",
            "release_action": "fresh claim-locator review and primary atomic integration",
            "status": "OPEN",
        },
        {
            "hold_id": "HOLD-0205V3-002",
            "scope": "canonical integration",
            "condition": "intrinsic v2 and metadata v3 are staging candidates",
            "release_action": "fresh nonauthor v3 review followed by primary integration decision",
            "status": "OPEN",
        },
        {
            "hold_id": "HOLD-0205V3-003",
            "scope": "DOCX and publication",
            "condition": "no rendered artifact or publication packet reviewed",
            "release_action": "canonical rebase, DOCX rebuild, render review, and publication validation",
            "status": "OPEN",
        },
        {
            "hold_id": "HOLD-0205V3-004",
            "scope": "downstream handoffs",
            "condition": "intrinsic downstream acceptance does not make blocked fields ready",
            "release_action": "resolve each field-level missing-evidence condition",
            "status": "OPEN",
        },
        {
            "hold_id": "HOLD-0205V3-005",
            "scope": "model and analytical use",
            "condition": "Chapter 18 coefficient and predictive blocks and Chapter 19 operational gates remain",
            "release_action": "separate governed empirical validation",
            "status": "OPEN",
        },
    ]
    write_csv(output / "current_governance_holds.csv", holds, list(holds[0]))

    authority = [
        {
            "authority_id": "AUTH-0205V3-001",
            "layer": "source_preservation",
            "current_state": "REFERENCED_ACCEPTED_UNION_V2_CANDIDATE",
            "does_not_imply": "semantic support or publication",
        },
        {
            "authority_id": "AUTH-0205V3-002",
            "layer": "semantic_support",
            "current_state": "278 claims and nine intrinsic repairs accepted in v2 review",
            "does_not_imply": "canonical integration",
        },
        {
            "authority_id": "AUTH-0205V3-003",
            "layer": "canonical_integration",
            "current_state": "NOT_AUTHORIZED",
            "does_not_imply": "DOCX replacement",
        },
        {
            "authority_id": "AUTH-0205V3-004",
            "layer": "publication",
            "current_state": "BLOCKED",
            "does_not_imply": "none",
        },
    ]
    write_csv(output / "authority_separation_audit.csv", authority, list(authority[0]))

    after = file_state(protected_paths())
    no_write = [
        {
            "path": path,
            "before_sha256": before.get(path, ("", 0))[0],
            "after_sha256": after.get(path, ("", 0))[0],
            "result": (
                "PASS_UNCHANGED"
                if before.get(path) == after.get(path)
                else "FAIL_CHANGED"
            ),
        }
        for path in sorted(set(before) | set(after))
    ]
    write_csv(output / "no_write_audit.csv", no_write, list(no_write[0]))

    metrics = {
        "status_date": "2026-07-26",
        "author_verdict": "READY_FOR_FRESH_NONAUTHOR_METADATA_REBASE_REVIEW",
        "canonical_integration": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "BLOCKED",
        "claims": len(read_csv(output / "intrinsic_complete_claim_source_audit.csv")),
        "intrinsic_support_links": len(
            read_csv(output / "intrinsic_claim_source_support_junction.csv")
        ),
        "chapter_projections": len(
            read_csv(output / "accepted_intrinsic_v2/chapter_html_projection_audit.csv")
        ),
        "unique_claim_appearances": len(
            {
                row["claim_id"]
                for row in read_csv(
                    output / "accepted_intrinsic_v2/chapter_html_projection_audit.csv"
                )
            }
        ),
        "repairs": len(
            read_csv(output / "accepted_intrinsic_v2/exact_patch_disposition.csv")
        ),
        "arithmetic_controls": sum(
            row["audit_type"] == "arithmetic_reperformance"
            for row in read_csv(
                output / "accepted_intrinsic_v2/arithmetic_nonadditivity_audit.csv"
            )
        ),
        "nonadditivity_controls": sum(
            row["audit_type"] != "arithmetic_reperformance"
            for row in read_csv(
                output / "accepted_intrinsic_v2/arithmetic_nonadditivity_audit.csv"
            )
        ),
        "unknowns": len(read_csv(output / "accepted_intrinsic_v2/unknown_population_audit.csv")),
        "falsifiers": len(read_csv(output / "accepted_intrinsic_v2/falsifier_audit.csv")),
        "handoffs": len(
            read_csv(output / "accepted_intrinsic_v2/handoff_blocked_field_audit.csv")
        ),
        "preservation_projection_rows": len(projection),
        "preservation_projection_counts": dict(
            sorted(Counter(row["source_state_disposition"] for row in projection).items())
        ),
        "union_snapshot": SNAPSHOT,
        "union_review_manifest_sha256": digest(accepted_union_manifest),
        "intrinsic_artifacts_copied": len(intrinsic_inventory),
    }
    write_json(output / "author_validation_results.json", metrics)
    (output / "author_report.md").write_text(
        """# Chapters 2–5 rebuilt-v3 metadata-rebase candidate

This packet preserves the freshly accepted v2 intrinsic claim, repair, chapter,
unknown, falsifier, and handoff results. It removes copied mutable preservation
status from current governed rows and references the accepted normalized
source-preservation snapshot `SPU-V2-20260726-38b990b2`.

The 361 Chapters 2–5 preservation references contain 172 preserved-byte rows
whose locators remain pending, 186 rows without a matching preservation
record, and three internal P4 reconstruction rows. `SRC-0094` is internal and
has no external-source hold.

Downstream Chapters 16–19 are recorded at their current accepted intrinsic
boundaries. No handoff becomes ready, and every canonical, model, DOCX, and
publication hold remains open. Fresh nonauthor review is required.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
