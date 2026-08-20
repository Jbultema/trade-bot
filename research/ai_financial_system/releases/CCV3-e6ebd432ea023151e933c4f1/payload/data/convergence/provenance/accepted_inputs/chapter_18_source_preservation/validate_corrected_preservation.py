#!/usr/bin/env python3
"""Dedicated author validation for corrected Chapter 18 preservation metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
AUTHOR = STAGING / "lane_1_chapter_18_source_preservation"
REVIEW = STAGING / "lane_2_verify_chapter_18_source_preservation"
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header")
        result = list(reader)
        if any(None in row for row in result):
            raise ValueError(f"{path}: malformed row")
        return result


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> int:
    manifest = json.loads(
        (HERE / "lane_manifest.json").read_text(encoding="utf-8")
    )
    errors = []
    for item in manifest["files"]:
        path = HERE / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            errors.append(f"manifest mismatch: {item['path']}")
        if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
            errors.append(f"byte mismatch: {item['path']}")
        if item.get("rows") is not None and row_count(path) != item["rows"]:
            errors.append(f"row mismatch: {item['path']}")
        if item.get("lines") is not None and line_count(path) != item["lines"]:
            errors.append(f"line mismatch: {item['path']}")

    reviews = rows(HERE / "claim_locator_reviews.csv")
    source_reviews = rows(REVIEW / "semantic_locator_review.csv")
    reconciliations = rows(HERE / "locator_reconciliation.csv")
    decisions = rows(HERE / "support_narrowing_decisions.csv")
    patches = rows(HERE / "exact_patch_audit.csv")
    versions = rows(HERE / "source_versions.csv")
    derivatives = rows(HERE / "source_derivatives.csv")
    version_rehash = rows(HERE / "source_version_rehash_audit.csv")
    derivative_rehash = rows(HERE / "source_derivative_rehash_audit.csv")
    junction = rows(HERE / "source_claim_junction.csv")
    prohibitions = rows(
        HERE / "coefficient_predictive_prohibition_audit.csv"
    )
    no_write = rows(HERE / "input_no_write_audit.csv")

    review_base_fields = list(source_reviews[0])
    exact_review_adoption = len(reviews) == len(source_reviews) == 58 and all(
        all(observed[field] == expected[field] for field in review_base_fields)
        for observed, expected in zip(reviews, source_reviews, strict=True)
    )
    dispositions = Counter(row["reviewer_disposition"] for row in reviews)
    partial_holds = {
        row["hold_relationship_id"]
        for row in reviews
        if row["reviewer_disposition"] == "partially_supported"
    }
    success_versions = [
        row
        for row in versions
        if row["preservation_status"] == "preserved_complete"
    ]
    failed_versions = [
        row
        for row in versions
        if row["preservation_status"] == "retrieval_failed"
    ]
    checks = {
        "manifest_integrity": not errors,
        "frozen_source_tables": all(
            sha(HERE / name) == sha(AUTHOR / name)
            for name in (
                "source_versions.csv",
                "source_derivatives.csv",
                "hold_inventory.csv",
                "source_claim_junction.csv",
                "source_exceptions.csv",
            )
        ),
        "frozen_source_artifacts": all(
            directory_hashes(HERE / name) == directory_hashes(AUTHOR / name)
            for name in (
                "source_archive",
                "response_metadata",
                "source_derivatives",
            )
        ),
        "exact_58_locator_adoption": exact_review_adoption
        and len(reconciliations) == len(decisions) == 58,
        "correct_32_26_dispositions": dispositions
        == {"supported": 32, "partially_supported": 26},
        "exact_18_material_patches": len(patches) == 18
        and {row["record_id"] for row in patches} == PATCH_HOLDS
        and all(
            row["patch_status"] == "PASS_APPLIED"
            and row["corrected_disposition"] == "partially_supported"
            for row in patches
        )
        and PATCH_HOLDS.issubset(partial_holds),
        "partial_scope_not_laundered": all(
            row["full_compound_direct_support_prohibited"] == "YES"
            for row in reconciliations
            if row["corrected_disposition"] == "partially_supported"
        )
        and all(
            row["claim_scope_action"]
            == "do_not_represent_full_compound_proposition_as_direct_support"
            for row in decisions
            if row["semantic_disposition"] == "partially_supported"
        ),
        "population_58_73_23_25_23": len(reviews) == 58
        and len(junction) == 73
        and len(versions) == 25
        and len(success_versions) == 23
        and len(failed_versions) == 2
        and len({row["source_id"] for row in success_versions}) == 23
        and len(derivatives) == 23,
        "two_unctad_failures_preserved": all(
            row["source_id"] == "L1-SRC18-0016"
            and row["version_ordinal"] in {"v0001", "v0002"}
            and not row["content_sha256"]
            and not row["local_archive_path"]
            for row in failed_versions
        ),
        "unctad_v0003_identity": any(
            row["source_id"] == "L1-SRC18-0016"
            and row["version_ordinal"] == "v0003"
            and row["content_sha256"]
            == "6b989ac7fa173a2de5d47757874e1cdc41419fa4407374fd5c9aedc69ffacb46"
            for row in success_versions
        ),
        "source_and_derivative_rehashes": len(version_rehash) == 25
        and len(derivative_rehash) == 23
        and all(
            row["rehash_status"]
            in {"PASS", "PASS_APPEND_ONLY_FAILURE_NO_SYNTHETIC_HASH"}
            for row in version_rehash
        )
        and all(
            row["parent_and_rehash_status"] == "PASS"
            for row in derivative_rehash
        ),
        "all_34_prohibitions_retained": len(prohibitions) == 34
        and all(
            row["coefficient_probability_frequency_timing_authority"]
            == "BLOCK"
            and row["predictive_authority"] == "BLOCK"
            and row["review_status"] == "PASS_RETAINED"
            for row in prohibitions
        ),
        "all_authority_holds": all(
            row["coefficient_probability_frequency_timing_use"] == "PROHIBITED"
            and row["predictive_use"] == "PROHIBITED"
            and row["causal_transfer"] == "PROHIBITED"
            and row["comparative_system_use"] == "PROHIBITED"
            and row["canonical_mutation_authority"] == "NONE"
            and row["docx_publication_authority"] == "NONE"
            for row in decisions
        )
        and manifest["metadata_integration"] == "NOT_AUTHORIZED"
        and manifest["canonical_integration"] == "NOT_AUTHORIZED"
        and manifest["chapter_docx_publication"] == "NOT_AUTHORIZED",
        "protected_inputs_no_write": len(no_write) >= 200
        and all(row["no_write_status"] == "PASS_UNCHANGED" for row in no_write),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": [
            {"check_id": key, "status": "PASS" if value else "FAIL"}
            for key, value in checks.items()
        ],
        "errors": errors,
        "counts": {
            "relationships": len(reviews),
            "edges": len(junction),
            "sources": len({row["source_id"] for row in success_versions}),
            "versions": len(versions),
            "successful_artifacts": len(success_versions),
            "failed_attempts": len(failed_versions),
            "derivatives": len(derivatives),
            "supported": dispositions["supported"],
            "partially_supported": dispositions["partially_supported"],
            "patches": len(patches),
        },
        "author_validation_is_acceptance": False,
        "metadata_integration": "NOT_AUTHORIZED",
        "canonical_integration": "NOT_AUTHORIZED",
        "chapter_docx_publication": "NOT_AUTHORIZED",
        "coefficient_predictive_causal_comparative_model_use": "BLOCK",
    }
    (HERE / "validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
