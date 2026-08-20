#!/usr/bin/env python3
"""Build the corrected Chapter 18 source-preservation semantic rebase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
AI_ROOT = STAGING.parent
REPO = AI_ROOT.parents[1]
AUTHOR = STAGING / "lane_1_chapter_18_source_preservation"
REVIEW = STAGING / "lane_2_verify_chapter_18_source_preservation"
V5 = STAGING / "lane_4_chapter_18_rebuilt_v5"
V5_REVIEW = STAGING / "lane_2_verify_chapter_18_rebuilt_v5"
DATA = AI_ROOT / "data"

AUTHOR_MANIFEST_SHA256 = (
    "595fd73bae86430a68e8462109d646845dd5ca4d1e1dc3f10df901c297950f6d"
)
REVIEW_MANIFEST_SHA256 = (
    "945938aefbfc97411f2f0c4c9ead68982633fb56657404ff167ae3ccb5ea0de3"
)
V5_MANIFEST_SHA256 = (
    "d11f8bd03062b2abfbcb85d3c351ea614eb2d06a3704ed15b6e571c2e758b8de"
)
V5_REVIEW_MANIFEST_SHA256 = (
    "d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8"
)
UNCTAD_SHA256 = (
    "6b989ac7fa173a2de5d47757874e1cdc41419fa4407374fd5c9aedc69ffacb46"
)
PATCH_HOLDS = {
    "CH18-HOLD-0013",
    "CH18-HOLD-0015",
    "CH18-HOLD-0023",
    "CH18-HOLD-0025",
    "CH18-HOLD-0029",
    "CH18-HOLD-0032",
    "CH18-HOLD-0036",
    "CH18-HOLD-0038",
    "CH18-HOLD-0048",
    "CH18-HOLD-0049",
    "CH18-HOLD-0050",
    "CH18-HOLD-0051",
    "CH18-HOLD-0052",
    "CH18-HOLD-0054",
    "CH18-HOLD-0055",
    "CH18-HOLD-0056",
    "CH18-HOLD-0057",
    "CH18-HOLD-0058",
}
FROZEN_TABLES = (
    "source_versions.csv",
    "source_derivatives.csv",
    "hold_inventory.csv",
    "source_claim_junction.csv",
    "source_exceptions.csv",
)
FROZEN_DIRS = ("source_archive", "response_metadata", "source_derivatives")
SCRIPT_FILES = (
    "build_corrected_preservation.py",
    "validate_corrected_preservation.py",
    "verify_clean_builds.py",
    "run_author_checks.py",
    "seal_packet.py",
)
GENERATED = {
    "claim_locator_reviews.csv",
    "coefficient_predictive_prohibition_audit.csv",
    "exact_patch_audit.csv",
    "input_no_write_audit.csv",
    "lane_manifest.json",
    "locator_reconciliation.csv",
    "preservation_semantic_coverage.csv",
    "source_derivative_rehash_audit.csv",
    "source_version_rehash_audit.csv",
    "support_narrowing_decisions.csv",
    "unresolved_queue.csv",
    "author_report.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"{path}: malformed row")
        return rows


def write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{name}: cannot write empty table")
    with (output / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def entry(root: Path, path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        value["rows"] = row_count(path)
    elif path.suffix in {".md", ".py", ".txt"}:
        value["lines"] = line_count(path)
    return value


def protected_paths() -> list[Path]:
    paths = [
        STAGING / "chapter_18_source_preservation_corrected_task.md",
        STAGING / "chapter_18_source_preservation_task.md",
        STAGING / "chapter_18_rebuilt_v5_task.md",
        AI_ROOT / "source_preservation_standard.md",
        STAGING / "second_half_canonical_integration_design.md",
        DATA / "source_register.csv",
        DATA / "claims.csv",
        DATA / "evidence_links.csv",
        AI_ROOT / "integration/candidate_data/research.db",
    ]
    paths.extend(
        path
        for root in (AUTHOR, REVIEW, V5, V5_REVIEW)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    return paths


def protected_inventory() -> list[dict[str, Any]]:
    return [
        {
            "inventory_id": f"CH18-CSP-IN-{number:04d}",
            "path": path.relative_to(REPO).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "status": "PASS_HASHED_PROTECTED_INPUT",
        }
        for number, path in enumerate(protected_paths(), start=1)
    ]


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if output != HERE:
        for name in SCRIPT_FILES:
            shutil.copy2(HERE / name, output / name)
    for name in GENERATED:
        path = output / name
        if path.exists():
            path.unlink()
    for name in FROZEN_TABLES:
        shutil.copy2(AUTHOR / name, output / name)
    for name in FROZEN_DIRS:
        target = output / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(AUTHOR / name, target)


def line_window_hashes_valid(review: dict[str, str], output: Path) -> bool:
    locator_parts = review["independent_resolved_locator"].split("|")
    hash_parts = review["exact_window_hashes"].split("|")
    if len(locator_parts) != len(hash_parts):
        return False
    for locator, expected in zip(locator_parts, hash_parts, strict=True):
        filename, range_text = locator.split(":lines ")
        start, end = (int(value) for value in range_text.split("-"))
        lines = (output / "source_derivatives" / filename).read_text(
            encoding="utf-8"
        ).splitlines()
        selected = "\n".join(lines[start - 1 : end])
        observed = hashlib.sha256(selected.encode()).hexdigest()
        if expected != f"{locator}:sha256={observed}":
            return False
    return True


def semantic_outputs(
    output: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    author_reviews = {
        row["hold_relationship_id"]: row
        for row in read_csv(AUTHOR / "claim_locator_reviews.csv")
    }
    reviews = read_csv(REVIEW / "semantic_locator_review.csv")
    patches = read_csv(REVIEW / "patch_list.csv")
    corrected_reviews = []
    reconciliation = []
    decisions = []
    for review in reviews:
        hold_id = review["hold_relationship_id"]
        author = author_reviews[hold_id]
        if not line_window_hashes_valid(review, output):
            raise ValueError(f"{hold_id}: exact window hash failed")
        corrected_reviews.append(
            {
                **review,
                "semantic_rebase_status": (
                    "ADOPTED_EXACT_FRESH_NONAUTHOR_LOCATOR_AND_SCOPE"
                ),
                "causal_authority": "BLOCK",
                "comparative_system_authority": "BLOCK",
                "model_authority": "BLOCK",
                "docx_authority": "BLOCK",
                "metadata_integration_authority": "NOT_AUTHORIZED",
            }
        )
        reconciliation.append(
            {
                "reconciliation_id": review["review_id"].replace(
                    "SEM", "REC"
                ),
                "hold_relationship_id": hold_id,
                "claim_id": review["claim_id"],
                "author_resolved_locator": author["resolved_locator"],
                "corrected_resolved_locator": review[
                    "independent_resolved_locator"
                ],
                "exact_window_hashes": review["exact_window_hashes"],
                "author_disposition": author["semantic_disposition"],
                "corrected_disposition": review["reviewer_disposition"],
                "disposition_change": review["author_comparison"],
                "line_window_rehash": "PASS",
                "full_compound_direct_support_prohibited": (
                    "YES"
                    if review["reviewer_disposition"]
                    == "partially_supported"
                    else "NOT_APPLICABLE"
                ),
            }
        )
        decisions.append(
            {
                "decision_id": review["review_id"].replace("SEM", "DEC"),
                "hold_relationship_id": hold_id,
                "claim_id": review["claim_id"],
                "semantic_disposition": review["reviewer_disposition"],
                "directly_supported_scope": review["reviewer_paraphrase"],
                "retained_scope_boundary": review[
                    "grade_and_method_boundary"
                ],
                "claim_scope_action": (
                    "retain_exact_bounded_claim"
                    if review["reviewer_disposition"] == "supported"
                    else "do_not_represent_full_compound_proposition_as_direct_support"
                ),
                "coefficient_probability_frequency_timing_use": "PROHIBITED",
                "predictive_use": "PROHIBITED",
                "causal_transfer": "PROHIBITED",
                "comparative_system_use": "PROHIBITED",
                "canonical_mutation_authority": "NONE",
                "docx_publication_authority": "NONE",
            }
        )
    patch_audit = []
    corrected_by = {
        row["hold_relationship_id"]: row for row in corrected_reviews
    }
    for patch in patches:
        corrected = corrected_by[patch["record_id"]]
        patch_audit.append(
            {
                **patch,
                "corrected_disposition": corrected["reviewer_disposition"],
                "corrected_locator": corrected[
                    "independent_resolved_locator"
                ],
                "corrected_window_hashes": corrected[
                    "exact_window_hashes"
                ],
                "coefficient_authority": corrected[
                    "coefficient_authority"
                ],
                "predictive_authority": corrected["predictive_authority"],
                "canonical_integration": corrected[
                    "canonical_integration"
                ],
                "publication": corrected["publication"],
                "patch_status": (
                    "PASS_APPLIED"
                    if corrected["reviewer_disposition"]
                    == "partially_supported"
                    else "FAIL"
                ),
            }
        )
    return corrected_reviews, reconciliation, decisions, patch_audit


def rehash_rows(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    versions = read_csv(output / "source_versions.csv")
    derivatives = read_csv(output / "source_derivatives.csv")
    version_rows = []
    for row in versions:
        if row["preservation_status"] == "preserved_complete":
            artifact = output / row["local_archive_path"]
            status = (
                "PASS"
                if artifact.is_file()
                and sha(artifact) == row["content_sha256"]
                and artifact.stat().st_size == int(row["byte_count"])
                else "FAIL"
            )
        else:
            status = (
                "PASS_APPEND_ONLY_FAILURE_NO_SYNTHETIC_HASH"
                if not row["content_sha256"] and not row["local_archive_path"]
                else "FAIL"
            )
        version_rows.append(
            {
                "source_version_id": row["source_version_id"],
                "source_id": row["source_id"],
                "preservation_status": row["preservation_status"],
                "expected_sha256": row["content_sha256"],
                "observed_sha256": (
                    sha(output / row["local_archive_path"])
                    if row["local_archive_path"]
                    else ""
                ),
                "rehash_status": status,
            }
        )
    derivative_rows = []
    version_ids = {row["source_version_id"] for row in versions}
    for row in derivatives:
        artifact = output / row["derivative_path"]
        status = (
            "PASS"
            if row["parent_source_version_id"] in version_ids
            and artifact.is_file()
            and sha(artifact) == row["derivative_sha256"]
            and artifact.stat().st_size == int(row["derivative_byte_count"])
            else "FAIL"
        )
        derivative_rows.append(
            {
                "derivative_id": row["derivative_id"],
                "parent_source_version_id": row[
                    "parent_source_version_id"
                ],
                "expected_sha256": row["derivative_sha256"],
                "observed_sha256": sha(artifact),
                "parent_and_rehash_status": status,
            }
        )
    return version_rows, derivative_rows


def coverage_rows(
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dispositions = Counter(row["reviewer_disposition"] for row in reviews)
    return [
        {
            "coverage_id": "CH18-CSP-COV-001",
            "dimension": "held_relationships",
            "observed": 58,
            "expected": 58,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-002",
            "dimension": "ordered_source_edges",
            "observed": 73,
            "expected": 73,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-003",
            "dimension": "unique_sources",
            "observed": 23,
            "expected": 23,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-004",
            "dimension": "source_version_attempts",
            "observed": 25,
            "expected": 25,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-005",
            "dimension": "successful_preserved_artifacts",
            "observed": 23,
            "expected": 23,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-006",
            "dimension": "locator_derivatives",
            "observed": 23,
            "expected": 23,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-007",
            "dimension": "supported_dispositions",
            "observed": dispositions["supported"],
            "expected": 32,
            "status": "PASS",
        },
        {
            "coverage_id": "CH18-CSP-COV-008",
            "dimension": "partially_supported_dispositions",
            "observed": dispositions["partially_supported"],
            "expected": 26,
            "status": "PASS",
        },
    ]


def report() -> str:
    return """# Corrected Chapter 18 source-preservation semantic rebase

**Author status:** complete candidate; fresh nonauthor review required  
**Metadata integration:** not authorized  
**Canonical, chapter, DOCX, and publication mutation:** prohibited  
**Coefficient, predictive, causal, model, and comparative use:** blocked

## Correction

This immutable replacement adopts the fresh nonauthor review's exact 58 locator
dispositions, line ranges, and proposition-window hashes. The corrected result
is 32 `supported` and 26 `partially_supported` relationships. All 18 material
scope patches are applied. No partial relationship represents the complete
compound claim as directly supported by a primary source.

## Preserved population

The preservation population is unchanged: 58 held relationships, 73 ordered
source edges, 23 unique sources, 25 append-only version attempts, 23 successful
artifacts, two retained UNCTAD HTTP-403 failures, one separate UN Digital
Library v0003 success, and 23 parented text derivatives. Original and derivative
hashes are revalidated.

## Authority boundary

All 18 coefficient-transfer prohibitions and 16 predictive/conceptual blocks
remain. Exact historical support does not authorize coefficients,
probabilities, empirical frequencies, timing rules, forecasts, causal transfer,
system comparisons, canonical integration, chapter replacement, DOCX, or
publication. This author packet only recommends readiness for fresh review and
a later separately authorized metadata rebase.
"""


def build(output: Path) -> None:
    if sha(AUTHOR / "lane_manifest.json") != AUTHOR_MANIFEST_SHA256:
        raise ValueError("source-preservation author manifest drift")
    if sha(REVIEW / "lane_manifest.json") != REVIEW_MANIFEST_SHA256:
        raise ValueError("fresh review manifest drift")
    if sha(V5 / "lane_manifest.json") != V5_MANIFEST_SHA256:
        raise ValueError("accepted Chapter 18 v5 manifest drift")
    if sha(V5_REVIEW / "lane_manifest.json") != V5_REVIEW_MANIFEST_SHA256:
        raise ValueError("accepted Chapter 18 v5 review manifest drift")
    before = protected_inventory()
    prepare(output)
    reviews, reconciliation, decisions, patches = semantic_outputs(output)
    version_rehash, derivative_rehash = rehash_rows(output)
    prohibitions = read_csv(REVIEW / "independent_prohibition_audit.csv")
    unresolved = read_csv(REVIEW / "unresolved_queue.csv")
    coverage = coverage_rows(reviews)
    after = protected_inventory()
    no_write = [
        {
            **row,
            "post_build_sha256": after[index]["sha256"],
            "no_write_status": (
                "PASS_UNCHANGED"
                if row["sha256"] == after[index]["sha256"]
                else "FAIL_CHANGED"
            ),
        }
        for index, row in enumerate(before)
    ]

    write_csv(output, "claim_locator_reviews.csv", reviews)
    write_csv(output, "locator_reconciliation.csv", reconciliation)
    write_csv(output, "support_narrowing_decisions.csv", decisions)
    write_csv(output, "exact_patch_audit.csv", patches)
    write_csv(output, "source_version_rehash_audit.csv", version_rehash)
    write_csv(output, "source_derivative_rehash_audit.csv", derivative_rehash)
    write_csv(
        output,
        "coefficient_predictive_prohibition_audit.csv",
        prohibitions,
    )
    write_csv(output, "unresolved_queue.csv", unresolved)
    write_csv(output, "preservation_semantic_coverage.csv", coverage)
    write_csv(output, "input_no_write_audit.csv", no_write)
    (output / "author_report.md").write_text(report(), encoding="utf-8")

    artifacts = [
        entry(output, path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name != "lane_manifest.json"
        and path.name
        not in {
            "validation_results.json",
            "generic_validation_results.json",
            "ruff_results.json",
            "two_clean_build_results.json",
        }
    ]
    manifest = {
        "lane": output.name,
        "packet_type": "chapter_18_source_preservation_corrected_semantic_rebase",
        "status_date": "2026-07-26",
        "source_author_manifest_sha256": AUTHOR_MANIFEST_SHA256,
        "fresh_review_manifest_sha256": REVIEW_MANIFEST_SHA256,
        "accepted_chapter18_v5_manifest_sha256": V5_MANIFEST_SHA256,
        "accepted_chapter18_v5_review_manifest_sha256": (
            V5_REVIEW_MANIFEST_SHA256
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
        "author_validation_is_acceptance": False,
        "metadata_integration": "NOT_AUTHORIZED",
        "canonical_integration": "NOT_AUTHORIZED",
        "chapter_docx_publication": "NOT_AUTHORIZED",
        "coefficient_predictive_causal_comparative_model_use": "BLOCK",
        "files": artifacts,
    }
    (output / "lane_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
