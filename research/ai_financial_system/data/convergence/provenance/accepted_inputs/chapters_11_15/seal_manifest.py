#!/usr/bin/env python3
"""Seal the stable Chapters 11-15 metadata-rebase manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
EXPECTED_GOVERNING_AUTHOR_V1_MANIFEST_SHA256 = (
    "134d597b1a0a723b2bb18e846f36cc3185fe95d15c41638a5939598a37651649"
)
EXPECTED_GOVERNING_REVIEW_V1_MANIFEST_SHA256 = (
    "6823b16cdb75c6f5ccf23f0f2fb3543bfbcebb6c84e194867932bceedcd74242"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return len(list(csv.DictReader(handle)))


def main() -> int:
    counts = json.loads((HERE / "build_counts.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (HERE / "validation_results.json").read_text(encoding="utf-8")
    )
    clean_builds = json.loads(
        (HERE / "two_clean_build_results.json").read_text(encoding="utf-8")
    )
    generic = json.loads(
        (HERE / "generic_validation_results.json").read_text(encoding="utf-8")
    )
    ruff = json.loads((HERE / "ruff_results.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise ValueError("dedicated validation has not passed")
    if clean_builds["status"] != "PASS" or not clean_builds["byte_identical"]:
        raise ValueError("two clean builds have not passed")
    if not generic["passed"]:
        raise ValueError("generic validation has not passed")
    if ruff["status"] != "PASS":
        raise ValueError("Ruff has not passed")

    files: list[dict[str, object]] = []
    for path in sorted(
        item
        for item in HERE.iterdir()
        if item.is_file() and item.name != "lane_manifest.json"
    ):
        entry: dict[str, object] = {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            entry["rows"] = csv_row_count(path)
        else:
            entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
        files.append(entry)

    manifest = {
        "lane": HERE.name,
        "workstream": "chapters_11_15_lossless_convergence_metadata_rebase",
        "chapter_scope": ["CH11", "CH12", "CH13", "CH14", "CH15"],
        "status_date": "2026-07-26",
        "verdict": (
            "READY_FOR_FRESH_UNINVOLVED_METADATA_ACCEPTANCE_WITH_ALL_"
            "PUBLICATION_AND_MODEL_USE_HOLDS"
        ),
        "governing_audit_manifest_sha256": EXPECTED_AUDIT_MANIFEST_SHA256,
        "canonical_database_sha256": EXPECTED_DATABASE_SHA256,
        "sealed_candidate_database_sha256": EXPECTED_DATABASE_SHA256,
        "canonical_candidate_byte_identical": True,
        "accepted_source_union_manifest_sha256": (
            EXPECTED_SOURCE_UNION_MANIFEST_SHA256
        ),
        "accepted_source_union_review_manifest_sha256": (
            EXPECTED_SOURCE_UNION_REVIEW_MANIFEST_SHA256
        ),
        "governing_metadata_rebase_v1_manifest_sha256": (
            EXPECTED_GOVERNING_AUTHOR_V1_MANIFEST_SHA256
        ),
        "governing_metadata_rebase_v1_review_manifest_sha256": (
            EXPECTED_GOVERNING_REVIEW_V1_MANIFEST_SHA256
        ),
        "repair_scope": (
            "all_cell_semantic_reference_completeness_and_explicit_"
            "non_support_dispositions"
        ),
        "populations": counts,
        "validation": {
            "dedicated": validation,
            "two_clean_builds": clean_builds,
            "generic": generic,
            "ruff": ruff,
        },
        "canonical_integration_authorized": False,
        "chapter_prose_authorized": False,
        "docx_authorized": False,
        "publication_authorized": False,
        "source_acquisition_performed": False,
        "mutable_source_union_status_copied": False,
        "byte_preservation_treated_as_semantic_support": False,
        "self_hash_policy": "lane_manifest.json excludes its own hash",
        "files": files,
    }
    (HERE / "lane_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sha256(HERE / "lane_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
