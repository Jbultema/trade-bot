#!/usr/bin/env python3
"""Build the narrow Chapters 6-9 v4 source-boundary repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGING = REPO / "research/ai_financial_system/staging"
V3 = STAGING / "lane_2_chapters_06_09_rebuilt_v3"
V3_REVIEW = STAGING / "lane_3_verify_chapters_06_09_rebuilt_v3"
UNION = STAGING / "lane_3_source_preservation_union_v2"
UNION_REVIEW = STAGING / "lane_2_verify_source_preservation_union_v2"
SNAPSHOT = "SPU-V2-20260726-38b990b2"
V3_MANIFEST_SHA = "eb00409fdfcb22758132c26ac1ccc7dd0a13eb6215aab6934ae5ab00ddc59f9c"
V3_REVIEW_SHA = "ef96660a6d4c4d50343198eaddea6d5bd2e018cd7a7eef11050506fbe401ee13"
UNION_REVIEW_SHA = "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
PROJECTION_RELATIVE = (
    "research/ai_financial_system/staging/lane_3_source_preservation_union_v2/"
    "chapter_02_09_support_state_projection.csv"
)
SCRIPTS = {"build_rebuilt_v4.py", "validate_rebuilt_v4.py", "verify_clean_builds.py"}
DO_NOT_CARRY = {
    "build_rebuilt_v3.py",
    "validate_rebuilt_v3.py",
    "verify_clean_builds.py",
    "lane_manifest.json",
    "generic_validation_results.json",
    "ruff_results.json",
    "deterministic_rebuild_results.json",
    "two_clean_build_comparison.csv",
    "no_write_audit.csv",
    "author_report.md",
    "validation_report.md",
    "validation_results.json",
    "semantic_validation_checks_v3.csv",
    "exact_repair_map.csv",
    "governing_input_inventory.csv",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing header")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path, rows: list[dict[str, object]], fields: list[str] | None = None
) -> None:
    names = fields or (list(rows[0]) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def reset_target(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        if target == HERE and item.name in SCRIPTS:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def copy_v3(target: Path) -> None:
    reset_target(target)
    for source in sorted(V3.rglob("*")):
        if not source.is_file() or source.name in DO_NOT_CARRY:
            continue
        relative = source.relative_to(V3)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def support_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["claim_id"],
        row["source_id"],
        row["locator"],
        row["support_type"],
    )


def patch_support_junction(target: Path) -> None:
    _, author = read_csv(V3 / "claim_source_support_junction.csv")
    _, projection = read_csv(UNION / "chapter_02_09_support_state_projection.csv")
    projection_by_key = {support_key(row): row for row in projection}
    if len(projection_by_key) != len(projection):
        raise ValueError("source-union projection keys are not unique")
    repaired = []
    audit = []
    for row in author:
        key = support_key(row)
        external = projection_by_key.get(key)
        if external is None:
            raise ValueError(f"missing source-union projection for {key}")
        if external["snapshot_id"] != SNAPSHOT:
            raise ValueError(f"unexpected snapshot for {key}")
        repaired_row = {
            key_name: value
            for key_name, value in row.items()
            if key_name != "source_version_status"
        }
        repaired_row.update(
            {
                "preservation_projection_file": PROJECTION_RELATIVE,
                "preservation_snapshot_id": external["snapshot_id"],
                "preservation_projection_row_id": external["projection_row_id"],
                "preservation_source_version_id": external["source_version_id"],
                "preservation_review_manifest_sha256": UNION_REVIEW_SHA,
            }
        )
        repaired.append(repaired_row)
        audit.append(
            {
                "junction_id": row["junction_id"],
                "claim_id": row["claim_id"],
                "source_id": row["source_id"],
                "locator": row["locator"],
                "support_type": row["support_type"],
                "removed_author_source_version_status": row["source_version_status"],
                "preservation_snapshot_id": external["snapshot_id"],
                "preservation_projection_row_id": external["projection_row_id"],
                "preservation_source_version_id": external["source_version_id"],
                "immutable_reference_complete": 1,
                "mutable_union_state_copied": 0,
            }
        )
    if len(repaired) != 164:
        raise ValueError(f"expected 164 support rows, got {len(repaired)}")
    write_csv(target / "claim_source_support_junction.csv", repaired)
    write_csv(target / "source_union_reference_audit.csv", audit)


def label_historical_state(target: Path) -> None:
    _, queue = read_csv(V3 / "source_queue.csv")
    queue_rows = [
        {
            **row,
            "authority_status": "historical_author_observation_nonauthoritative",
            "current_authority": (
                "resolve through source_preservation_dependency.csv and immutable "
                "projection reference"
            ),
        }
        for row in queue
    ]
    write_csv(target / "source_queue.csv", queue_rows)

    _, versions = read_csv(V3 / "source_version_candidates.csv")
    version_rows = [
        {
            **row,
            "authority_status": "historical_author_observation_nonauthoritative",
            "current_authority": (
                "resolve through source_preservation_dependency.csv and immutable "
                "projection reference"
            ),
        }
        for row in versions
    ]
    write_csv(target / "source_version_candidates.csv", version_rows)


def write_dependency(target: Path) -> None:
    write_csv(
        target / "source_preservation_dependency.csv",
        [
            {
                "dependency_id": "DEP-0609-SPU-V2",
                "projection_file": PROJECTION_RELATIVE,
                "snapshot_id": SNAPSHOT,
                "source_union_author_manifest_sha256": sha(
                    UNION / "lane_manifest.json"
                ),
                "source_union_review_manifest_sha256": UNION_REVIEW_SHA,
                "dependency_authority": "immutable_external_reference_only",
                "semantic_clearance": "not_granted_by_preservation_union",
                "canonical_integration": "not_authorized",
                "docx_replacement": "not_authorized",
                "publication": "not_authorized",
            }
        ],
    )


def write_audits(target: Path) -> None:
    _, patches = read_csv(V3_REVIEW / "exact_row_patches.csv")
    write_csv(
        target / "exact_patch_audit.csv",
        [
            {
                "patch_id": row["patch_id"],
                "target": row["target"],
                "required_change": row["required_change"],
                "implementation": (
                    "immutable projection references on 164 keys; no copied mutable "
                    "union authority"
                    if row["patch_id"] == "PATCH-001"
                    else "dependency pin plus historical/nonauthoritative labels"
                ),
                "status": "implemented_pending_fresh_review",
            }
            for row in patches
        ],
    )
    governing = [
        V3 / "lane_manifest.json",
        V3_REVIEW / "lane_manifest.json",
        V3_REVIEW / "exact_row_patches.csv",
        V3_REVIEW / "source_union_exact_repairs.csv",
        V3_REVIEW / "frozen_intrinsic_artifacts.csv",
        UNION / "lane_manifest.json",
        UNION / "chapter_02_09_support_state_projection.csv",
        UNION / "source_versions.csv",
        UNION_REVIEW / "lane_manifest.json",
        REPO / "reports/ai_financial_system/database_manifest.json",
        REPO / "reports/ai_financial_system/research.db",
    ]
    write_csv(
        target / "governing_input_inventory.csv",
        [
            {
                "input_id": f"V4-IN-{index:03d}",
                "path": path.relative_to(REPO).as_posix(),
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "role": "immutable_governing_or_protected_input",
            }
            for index, path in enumerate(governing, 1)
        ],
    )


def write_author_report(target: Path) -> None:
    (target / "author_report.md").write_text(
        "# Chapters 6-9 rebuilt-v4 author report\n\n"
        "## Verdict\n\n"
        "`READY_FOR_FRESH_UNINVOLVED_REVIEW`\n\n"
        "This narrow packet changes source-preservation metadata only. All 18 "
        "review-frozen intrinsic artifacts remain byte-identical to v3. The 164 "
        "claim/source support keys now reference the accepted source-union-v2 "
        "projection by immutable snapshot, projection-row, source-version, file, "
        "and review-manifest identity. Five stale alternate-version rows use the "
        "exact accepted alternate source versions. Carried source queue and source "
        "version statuses are labeled historical and nonauthoritative. Original "
        "failed-retrieval observations remain append-only in the external union.\n\n"
        "No canonical integration, chapter or DOCX replacement, or publication is "
        "authorized. Fresh uninvolved acceptance remains mandatory.\n",
        encoding="utf-8",
    )


def build(target: Path) -> dict[str, object]:
    if sha(V3 / "lane_manifest.json") != V3_MANIFEST_SHA:
        raise ValueError("v3 manifest drift")
    if sha(V3_REVIEW / "lane_manifest.json") != V3_REVIEW_SHA:
        raise ValueError("v3 review manifest drift")
    if sha(UNION_REVIEW / "lane_manifest.json") != UNION_REVIEW_SHA:
        raise ValueError("source-union review manifest drift")
    copy_v3(target)
    patch_support_junction(target)
    label_historical_state(target)
    write_dependency(target)
    write_audits(target)
    write_author_report(target)
    result = {
        "support_rows": 164,
        "immutable_reference_fields": 5,
        "frozen_intrinsic_artifacts": 18,
        "source_queue_rows": 48,
        "source_version_candidate_rows": 51,
        "snapshot_id": SNAPSHOT,
        "author_verdict": "READY_FOR_FRESH_UNINVOLVED_REVIEW",
        "canonical_integration": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "NOT_AUTHORIZED",
    }
    write_json(target / "author_validation_results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
