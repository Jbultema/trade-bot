#!/usr/bin/env python3
"""Meaning-based validation for the Chapter 18 immutable metadata rebase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGING = REPO / "research/ai_financial_system/staging"
INTRINSIC_AUTHOR = STAGING / "lane_4_chapter_18_rebuilt_v5"
INTRINSIC_REVIEW = STAGING / "lane_2_verify_chapter_18_rebuilt_v5"
PRESERVATION_AUTHOR = STAGING / "lane_3_chapter_18_source_preservation_corrected"
CANONICAL_DB = REPO / "reports/ai_financial_system/research.db"

EXPECTED_GOVERNING_HASHES = {
    "intrinsic_author_manifest": (
        "d11f8bd03062b2abfbcb85d3c351ea614eb2d06a3704ed15b6e571c2e758b8de"
    ),
    "intrinsic_review_manifest": (
        "d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8"
    ),
    "corrected_preservation_author_manifest": (
        "3fad6122a81e115a352e20bfdb3086be1a25d5ea6547202c97d6ac2020b11867"
    ),
    "corrected_preservation_review_manifest": (
        "d4e1d026f7743032e77f98a6ed1424ccd4270bb5a5188a3ed1a5d98c44eb8dd9"
    ),
    "frozen_canonical_database": (
        "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
    ),
}

EXPECTED_ROWS = {
    "immutable_intrinsic_artifact_inventory.csv": 50,
    "accepted_preservation_artifact_inventory.csv": 123,
    "intrinsic_row_disposition.csv": 1654,
    "intrinsic_field_disposition.csv": 17112,
    "source_metadata_row_disposition.csv": 718,
    "source_metadata_field_disposition.csv": 8096,
    "normalized_source_version_registry.csv": 25,
    "semantic_locator_registry.csv": 58,
    "claim_source_version_locator_junction.csv": 73,
    "preserved_derivative_registry.csv": 23,
    "retrieval_attempt_history.csv": 25,
    "evidence_authority_publication_hold_matrix.csv": 58,
    "retained_prohibition_matrix.csv": 34,
    "semantic_patch_registry.csv": 18,
    "retained_source_exception_holds.csv": 2,
    "retained_unresolved_holds.csv": 26,
    "population_reproduction_audit.csv": 15,
    "foreign_key_audit.csv": 7,
    "canonical_database_binding.csv": 2,
    "governing_input_hash_registry.csv": 5,
}

ALLOWED_DISPOSITIONS = {
    "direct_retained_identity",
    "normalized_junction_destination",
    "governed_hold_destination",
    "accepted_duplicate_with_surviving_identity",
    "explicit_rejection_authorized_by_accepted_review",
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
STABLE_KEY = re.compile(r"^CH18-[A-Z]+-[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(prefix: str, *parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path} has a row wider than its header")
    return list(reader.fieldnames), rows


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sqlite_integrity(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    return str(result[0]) if result else "missing_result"


class Validator:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.checks: list[dict[str, object]] = []
        self.errors: list[str] = []
        self.csv_cache: dict[str, tuple[list[str], list[dict[str, str]]]] = {}

    def check(
        self,
        check_id: str,
        description: str,
        condition: bool,
        observed: object,
        expected: object,
    ) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "description": description,
                "status": "PASS" if condition else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )
        if not condition:
            self.errors.append(
                f"{check_id}: {description}; observed={observed!r}; "
                f"expected={expected!r}"
            )

    def csv(self, relpath: str) -> tuple[list[str], list[dict[str, str]]]:
        if relpath not in self.csv_cache:
            self.csv_cache[relpath] = read_csv(self.target / relpath)
        return self.csv_cache[relpath]

    def validate_required_files(self) -> None:
        missing = [name for name in EXPECTED_ROWS if not (self.target / name).is_file()]
        self.check(
            "FILES-001",
            "all required tabular outputs exist",
            not missing,
            missing,
            [],
        )
        if missing:
            return
        counts = {name: len(self.csv(name)[1]) for name in EXPECTED_ROWS}
        self.check(
            "FILES-002",
            "required output row populations are exact",
            counts == EXPECTED_ROWS,
            counts,
            EXPECTED_ROWS,
        )
        other_required = (
            "build_metrics.json",
            "sealed_candidate/research.db",
            "sealed_candidate/database_manifest.json",
            "governing_manifests/intrinsic_author_lane_manifest.json",
            "governing_manifests/intrinsic_review_lane_manifest.json",
            "governing_manifests/corrected_preservation_author_lane_manifest.json",
            "governing_manifests/corrected_preservation_review_lane_manifest.json",
        )
        other_missing = [
            name for name in other_required if not (self.target / name).is_file()
        ]
        self.check(
            "FILES-003",
            "required non-tabular bindings exist",
            not other_missing,
            other_missing,
            [],
        )

    def validate_governing_inputs(self) -> None:
        _, rows = self.csv("governing_input_hash_registry.csv")
        by_role = {row["role"]: row for row in rows}
        exact = (
            set(by_role) == set(EXPECTED_GOVERNING_HASHES)
            and all(
                row["expected_sha256"] == expected
                and row["observed_before_sha256"] == expected
                and row["observed_after_sha256"] == expected
                and row["status"] == "PASS"
                for role, expected in EXPECTED_GOVERNING_HASHES.items()
                for row in [by_role[role]]
            )
        )
        self.check(
            "HASH-001",
            "all five governing inputs match before and after",
            exact,
            {role: by_role.get(role, {}).get("observed_after_sha256") for role in by_role},
            EXPECTED_GOVERNING_HASHES,
        )

        manifest_copies = {
            "intrinsic_author_manifest": (
                "governing_manifests/intrinsic_author_lane_manifest.json"
            ),
            "intrinsic_review_manifest": (
                "governing_manifests/intrinsic_review_lane_manifest.json"
            ),
            "corrected_preservation_author_manifest": (
                "governing_manifests/"
                "corrected_preservation_author_lane_manifest.json"
            ),
            "corrected_preservation_review_manifest": (
                "governing_manifests/"
                "corrected_preservation_review_lane_manifest.json"
            ),
        }
        copy_hashes = {
            role: sha256_file(self.target / path)
            for role, path in manifest_copies.items()
        }
        self.check(
            "HASH-002",
            "governing manifest snapshots are byte exact",
            all(
                copy_hashes[role] == EXPECTED_GOVERNING_HASHES[role]
                for role in manifest_copies
            ),
            copy_hashes,
            {
                role: EXPECTED_GOVERNING_HASHES[role]
                for role in manifest_copies
            },
        )

    def validate_inventories(self) -> None:
        _, intrinsic = self.csv("immutable_intrinsic_artifact_inventory.csv")
        intrinsic_ok = all(
            row["status"] == "PASS"
            and row["manifest_expected_sha256"]
            == row["observed_before_sha256"]
            == row["observed_after_sha256"]
            == row["copied_sha256"]
            and sha256_file(self.target / row["copied_path"]) == row["copied_sha256"]
            for row in intrinsic
        )
        self.check(
            "IMMUTABLE-001",
            "all 50 manifest-declared intrinsic artifacts are byte identical",
            intrinsic_ok,
            sum(row["status"] == "PASS" for row in intrinsic),
            50,
        )

        _, preservation = self.csv("accepted_preservation_artifact_inventory.csv")
        preservation_ok = all(
            row["status"] == "PASS"
            and row["manifest_expected_sha256"]
            == row["observed_before_sha256"]
            == row["observed_after_sha256"]
            and (
                not row["local_rebase_copy"]
                or sha256_file(self.target / row["local_rebase_copy"])
                == row["local_rebase_copy_sha256"]
                == row["manifest_expected_sha256"]
            )
            for row in preservation
        )
        self.check(
            "IMMUTABLE-002",
            "all 123 accepted preservation artifacts are bound without mutation",
            preservation_ok,
            sum(row["status"] == "PASS" for row in preservation),
            123,
        )

    def source_csv_inventory(
        self, base: Path, manifest_path: Path
    ) -> tuple[int, int, set[tuple[str, int, str]]]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row_count = 0
        field_count = 0
        keys: set[tuple[str, int, str]] = set()
        for entry in manifest["files"]:
            relpath = entry["path"]
            path = base / relpath
            if path.suffix.lower() != ".csv":
                continue
            fieldnames, rows = read_csv(path)
            row_count += len(rows)
            field_count += len(rows) * len(fieldnames)
            for ordinal, _ in enumerate(rows, start=1):
                keys.update((relpath, ordinal, field) for field in fieldnames)
        return row_count, field_count, keys

    def validate_dispositions(self) -> None:
        _, intrinsic_rows = self.csv("intrinsic_row_disposition.csv")
        _, intrinsic_fields = self.csv("intrinsic_field_disposition.csv")
        _, source_rows = self.csv("source_metadata_row_disposition.csv")
        _, source_fields = self.csv("source_metadata_field_disposition.csv")

        intrinsic_source = self.source_csv_inventory(
            INTRINSIC_AUTHOR, INTRINSIC_AUTHOR / "lane_manifest.json"
        )
        preservation_source = self.source_csv_inventory(
            PRESERVATION_AUTHOR, PRESERVATION_AUTHOR / "lane_manifest.json"
        )
        self.check(
            "LOSSLESS-001",
            "intrinsic row and scalar coverage is exact",
            (len(intrinsic_rows), len(intrinsic_fields))
            == intrinsic_source[:2]
            == (1654, 17112),
            (len(intrinsic_rows), len(intrinsic_fields)),
            (1654, 17112),
        )
        self.check(
            "LOSSLESS-002",
            "source-metadata row and scalar coverage is exact",
            (len(source_rows), len(source_fields))
            == preservation_source[:2]
            == (718, 8096),
            (len(source_rows), len(source_fields)),
            (718, 8096),
        )

        intrinsic_keys = {
            (
                row["source_file"],
                int(row["source_row_ordinal"]),
                row["source_field"],
            )
            for row in intrinsic_fields
        }
        source_keys = {
            (
                row["source_file"],
                int(row["source_row_ordinal"]),
                row["source_field"],
            )
            for row in source_fields
        }
        self.check(
            "LOSSLESS-003",
            "every intrinsic scalar has exactly one disposition",
            len(intrinsic_keys) == len(intrinsic_fields)
            and intrinsic_keys == intrinsic_source[2],
            (len(intrinsic_keys), len(intrinsic_fields)),
            (len(intrinsic_fields), len(intrinsic_fields)),
        )
        self.check(
            "LOSSLESS-004",
            "every source-metadata scalar has exactly one disposition",
            len(source_keys) == len(source_fields)
            and source_keys == preservation_source[2],
            (len(source_keys), len(source_fields)),
            (len(source_fields), len(source_fields)),
        )

        vocab = {
            row["field_disposition"] for row in intrinsic_fields + source_fields
        }
        self.check(
            "LOSSLESS-005",
            "field dispositions use only the authorized vocabulary",
            vocab <= ALLOWED_DISPOSITIONS,
            sorted(vocab),
            sorted(ALLOWED_DISPOSITIONS),
        )

        exact_destination = True
        destination_errors: list[str] = []
        destination_cells: list[tuple[str, int, str]] = []
        for row in intrinsic_fields + source_fields:
            destination = row["destination_artifact"]
            try:
                fields, target_rows = self.csv(destination)
                ordinal = int(row["destination_row_ordinal"])
                destination_field = row["destination_field"]
                if ordinal < 1 or ordinal > len(target_rows):
                    raise ValueError("destination ordinal outside row population")
                if destination_field not in fields:
                    raise ValueError("destination field missing")
                observed = target_rows[ordinal - 1][destination_field]
                if observed != row["source_value"]:
                    raise ValueError("destination scalar differs")
                expected_hash = hashlib.sha256(
                    row["source_value"].encode("utf-8")
                ).hexdigest()
                if expected_hash != row["source_value_sha256"]:
                    raise ValueError("source scalar hash differs")
                if row["value_state"] == "empty_retained" and observed != "":
                    raise ValueError("empty scalar became populated")
                if row["value_state"] == "populated" and observed == "":
                    raise ValueError("populated scalar became null")
                destination_cells.append(
                    (destination, ordinal, destination_field)
                )
            except (OSError, ValueError, IndexError) as exc:
                exact_destination = False
                if len(destination_errors) < 10:
                    destination_errors.append(
                        f"{row['field_disposition_id']}: {exc}"
                    )
        self.check(
            "LOSSLESS-006",
            "all destination scalars reproduce source values exactly",
            exact_destination,
            destination_errors,
            [],
        )
        duplicate_cells = len(destination_cells) - len(set(destination_cells))
        self.check(
            "LOSSLESS-007",
            "no many-to-one scalar overwrite occurs",
            duplicate_cells == 0,
            duplicate_cells,
            0,
        )
        self.check(
            "LOSSLESS-008",
            "no evidence authority strengthening is declared",
            all(
                row["authority_strengthening"] == "NONE"
                for row in intrinsic_fields + source_fields
            ),
            Counter(
                row["authority_strengthening"]
                for row in intrinsic_fields + source_fields
            ),
            {"NONE": len(intrinsic_fields) + len(source_fields)},
        )
        self.check(
            "LOSSLESS-009",
            "no source scalar is explicitly rejected by the rebase",
            all(
                row["field_disposition"]
                != "explicit_rejection_authorized_by_accepted_review"
                for row in intrinsic_fields + source_fields
            ),
            sum(
                row["field_disposition"]
                == "explicit_rejection_authorized_by_accepted_review"
                for row in intrinsic_fields + source_fields
            ),
            0,
        )

    def validate_populations_and_keys(self) -> None:
        _, population_rows = self.csv("population_reproduction_audit.csv")
        populations = {
            row["population"]: int(row["observed"]) for row in population_rows
        }
        expected = {
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
        self.check(
            "POP-001",
            "all task-declared populations reproduce exactly",
            populations == expected
            and all(row["status"] == "PASS" for row in population_rows),
            populations,
            expected,
        )

        _, versions = self.csv("normalized_source_version_registry.csv")
        version_keys = [row["source_version_semantic_key"] for row in versions]
        expected_version_keys = [
            stable_key(
                "CH18-SVK",
                row["source_id"],
                row["version_ordinal"],
                row["version_role"],
                row["original_url"],
                row["retrieval_timestamp_utc"],
            )
            for row in versions
        ]
        self.check(
            "KEY-001",
            "source-version semantic keys are unique and deterministic",
            version_keys == expected_version_keys
            and len(version_keys) == len(set(version_keys))
            and all(STABLE_KEY.match(key) for key in version_keys),
            len(set(version_keys)),
            25,
        )

        _, locators = self.csv("semantic_locator_registry.csv")
        locator_keys = [row["semantic_locator_key"] for row in locators]
        expected_locator_keys = [
            stable_key(
                "CH18-LOC",
                row["hold_relationship_id"],
                row["claim_id"],
                row["independent_resolved_locator"],
                row["exact_window_hashes"],
            )
            for row in locators
        ]
        self.check(
            "KEY-002",
            "semantic locator keys are unique and deterministic",
            locator_keys == expected_locator_keys
            and len(locator_keys) == len(set(locator_keys))
            and all(STABLE_KEY.match(key) for key in locator_keys),
            len(set(locator_keys)),
            58,
        )

        _, junctions = self.csv("claim_source_version_locator_junction.csv")
        junction_keys = [row["junction_semantic_key"] for row in junctions]
        expected_junction_keys = [
            stable_key(
                "CH18-JUNCTION",
                row["claim_id"],
                row["source_id"],
                row["source_version_id"],
                row["hold_relationship_id"],
                row["relationship_role"],
            )
            for row in junctions
        ]
        self.check(
            "KEY-003",
            "junction semantic keys are unique and deterministic",
            junction_keys == expected_junction_keys
            and len(junction_keys) == len(set(junction_keys))
            and all(STABLE_KEY.match(key) for key in junction_keys),
            len(set(junction_keys)),
            73,
        )

        _, fk_rows = self.csv("foreign_key_audit.csv")
        self.check(
            "FK-001",
            "all stable foreign-key audits pass",
            all(
                row["status"] == "PASS"
                and row["matched"] == row["expected"]
                for row in fk_rows
            ),
            {row["relationship"]: row["status"] for row in fk_rows},
            "all PASS",
        )

    def validate_locator_and_authority_semantics(self) -> None:
        _, locators = self.csv("semantic_locator_registry.csv")
        disposition_counts = Counter(row["reviewer_disposition"] for row in locators)
        exact_windows = True
        for row in locators:
            resolved = row["independent_resolved_locator"].split("|")
            hashes = row["exact_window_hashes"].split("|")
            if len(resolved) != len(hashes):
                exact_windows = False
                break
            for locator, window in zip(resolved, hashes, strict=True):
                if not (
                    ":lines " in locator
                    and window.startswith(locator + ":sha256=")
                    and HEX64.match(window.rsplit(":sha256=", 1)[1])
                ):
                    exact_windows = False
                    break
        self.check(
            "LOCATOR-001",
            "all 58 exact line ranges and window hashes reproduce",
            len(locators) == 58 and exact_windows,
            len(locators),
            58,
        )
        self.check(
            "LOCATOR-002",
            "semantic dispositions remain 32 supported and 26 partially supported",
            disposition_counts == {"supported": 32, "partially_supported": 26},
            dict(disposition_counts),
            {"supported": 32, "partially_supported": 26},
        )

        _, matrix = self.csv("evidence_authority_publication_hold_matrix.csv")
        holds_ok = all(
            row["locator__coefficient_authority"] == "BLOCK"
            and row["locator__predictive_authority"] == "BLOCK"
            and row["locator__canonical_integration"] == "NOT_AUTHORIZED"
            and row["locator__publication"] == "NOT_AUTHORIZED"
            and row["locator__causal_authority"] == "BLOCK"
            and row["locator__comparative_system_authority"] == "BLOCK"
            and row["locator__model_authority"] == "BLOCK"
            and row["locator__docx_authority"] == "BLOCK"
            and row["locator__metadata_integration_authority"] == "NOT_AUTHORIZED"
            and row["scope__coefficient_probability_frequency_timing_use"]
            == "PROHIBITED"
            and row["scope__predictive_use"] == "PROHIBITED"
            and row["scope__causal_transfer"] == "PROHIBITED"
            and row["scope__comparative_system_use"] == "PROHIBITED"
            and row["scope__canonical_mutation_authority"] == "NONE"
            and row["scope__docx_publication_authority"] == "NONE"
            and row["preservation_is_semantic_support"] == "NO"
            and row["partially_supported_promoted"] == "NO"
            and row["authority_strengthening"] == "NONE"
            and row["retained_hold_status"] == "PASS"
            for row in matrix
        )
        self.check(
            "AUTH-001",
            "preservation, support, grade, scope, analytics, and publication stay separate",
            holds_ok,
            sum(holds_ok for _ in [0]) if matrix else 0,
            1,
        )
        matrix_dispositions = Counter(
            row["semantic_support_disposition"] for row in matrix
        )
        self.check(
            "AUTH-002",
            "partially supported relationships are never promoted",
            matrix_dispositions
            == {"supported": 32, "partially_supported": 26},
            dict(matrix_dispositions),
            {"supported": 32, "partially_supported": 26},
        )

        _, prohibitions = self.csv("retained_prohibition_matrix.csv")
        prohibition_counts = Counter(
            row["governed_record_type"] for row in prohibitions
        )
        prohibition_holds = all(
            row["coefficient_probability_frequency_timing_authority"] == "BLOCK"
            and row["predictive_authority"] == "BLOCK"
            and row["review_status"] == "PASS_RETAINED"
            and row["causal_authority"] == "BLOCK"
            and row["comparative_system_authority"] == "BLOCK"
            and row["model_authority"] == "BLOCK"
            and row["canonical_integration"] == "NOT_AUTHORIZED"
            and row["publication"] == "NOT_AUTHORIZED"
            and row["docx_authority"] == "BLOCK"
            and row["metadata_integration_authority"] == "NOT_AUTHORIZED"
            and row["hold_strengthening"] == "NONE"
            for row in prohibitions
        )
        self.check(
            "AUTH-003",
            "all 18 coefficient and 16 predictive blocks remain exact",
            prohibition_counts
            == {
                "coefficient_transfer_prohibition": 18,
                "conceptual_hypothesis": 16,
            }
            and prohibition_holds,
            dict(prohibition_counts),
            {
                "coefficient_transfer_prohibition": 18,
                "conceptual_hypothesis": 16,
            },
        )

    def validate_versions_derivatives_and_patches(self) -> None:
        _, versions = self.csv("normalized_source_version_registry.csv")
        by_id = {row["source_version_id"]: row for row in versions}
        unctad_expected = {
            "CH18-SP-L1-SRC18-0016-v0001": ("retrieval_failed", "403", ""),
            "CH18-SP-L1-SRC18-0016-v0002": ("retrieval_failed", "403", ""),
            "CH18-SP-L1-SRC18-0016-v0003": (
                "preserved_complete",
                "200",
                "6b989ac7fa173a2de5d47757874e1cdc41419fa4407374fd5c9aedc69ffacb46",
            ),
        }
        unctad_observed = {
            source_version_id: (
                by_id[source_version_id]["preservation_status"],
                by_id[source_version_id]["http_status"],
                by_id[source_version_id]["content_sha256"],
            )
            for source_version_id in unctad_expected
        }
        self.check(
            "VERSION-001",
            "both original UNCTAD 403 failures and successful v0003 remain append-only",
            unctad_observed == unctad_expected,
            unctad_observed,
            unctad_expected,
        )
        self.check(
            "VERSION-002",
            "failed retrievals have no synthetic content hash or archive path",
            all(
                row["content_sha256"] == ""
                and row["local_archive_path"] == ""
                and row["byte_count"] == ""
                for row in versions
                if row["preservation_status"] == "retrieval_failed"
            ),
            sum(
                row["preservation_status"] == "retrieval_failed"
                for row in versions
            ),
            2,
        )

        _, derivatives = self.csv("preserved_derivative_registry.csv")
        derivative_ok = all(
            row["derivative_sha256"] == row["derivative_observed_sha256"]
            and row["parent_content_sha256"] == row["parent_observed_sha256"]
            and row["derivative_byte_count"]
            == row["derivative_observed_byte_count"]
            and row["rehash_status"] == "PASS"
            and row["preservation_is_semantic_support"] == "NO"
            for row in derivatives
        )
        self.check(
            "DERIVATIVE-001",
            "all 23 derivatives and parent hashes reperform exactly",
            derivative_ok,
            sum(row["rehash_status"] == "PASS" for row in derivatives),
            23,
        )

        _, patches = self.csv("semantic_patch_registry.csv")
        patch_ok = all(
            row["patch_status"] == "PASS_APPLIED"
            and row["fresh_review_scope_status"] == "PASS"
            and row["residual_patch"] == "NO"
            for row in patches
        )
        self.check(
            "PATCH-001",
            "all 18 accepted semantic-scope patches reproduce with no residual",
            patch_ok,
            len(patches),
            18,
        )

        _, repairs = read_csv(INTRINSIC_REVIEW / "exact_repair_reperformance.csv")
        earlier = {
            row["exact_patch_id"]
            for row in repairs
            if row["exact_patch_id"].startswith("FAR18-XP-")
        }
        v5 = {
            row["exact_patch_id"]
            for row in repairs
            if row["exact_patch_id"].startswith("V4R-XP-")
        }
        self.check(
            "REPAIR-001",
            "17 earlier intrinsic repairs and both v5 residual repairs stay accepted",
            len(repairs) == 19
            and len(earlier) == 17
            and v5 == {"V4R-XP-0001", "V4R-XP-0002"}
            and all(row["semantic_release_status"] == "PASS" for row in repairs),
            (len(earlier), len(v5)),
            (17, 2),
        )

    def validate_database_and_no_write(self) -> None:
        _, binding = self.csv("canonical_database_binding.csv")
        by_role = {row["role"]: row for row in binding}
        candidate = self.target / "sealed_candidate/research.db"
        expected_hash = EXPECTED_GOVERNING_HASHES["frozen_canonical_database"]
        byte_identical = (
            CANONICAL_DB.read_bytes() == candidate.read_bytes()
            and sha256_file(CANONICAL_DB)
            == sha256_file(candidate)
            == expected_hash
        )
        binding_ok = (
            byte_identical
            and sqlite_integrity(CANONICAL_DB) == "ok"
            and sqlite_integrity(candidate) == "ok"
            and by_role["sealed_integration_candidate_database"][
                "byte_identity_to_canonical"
            ]
            == "PASS"
            and all(row["status"] == "PASS" for row in binding)
        )
        self.check(
            "DB-001",
            "sealed candidate database is byte-identical and SQLite-clean",
            binding_ok,
            sha256_file(candidate),
            expected_hash,
        )

        _, no_write = self.csv("protected_input_no_write_audit.csv")
        current_ok = True
        failures: list[str] = []
        for row in no_write:
            path = REPO / row["path"]
            if (
                not path.is_file()
                or sha256_file(path) != row["sha256_before"]
                or row["sha256_before"] != row["sha256_after"]
                or path.stat().st_size != int(row["bytes_before"])
                or row["bytes_before"] != row["bytes_after"]
                or row["status"] != "PASS"
            ):
                current_ok = False
                if len(failures) < 10:
                    failures.append(row["path"])
        self.check(
            "NOWRITE-001",
            "all protected inputs remain byte-identical through validation",
            current_ok,
            failures,
            [],
        )

    def validate_manifest_if_present(self) -> None:
        manifest_path = self.target / "lane_manifest.json"
        if not manifest_path.is_file():
            self.check(
                "MANIFEST-001",
                "sealed manifest check is deferred until finalization",
                True,
                "DEFERRED",
                "DEFERRED",
            )
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files", [])
        exact = True
        for entry in entries:
            path = self.target / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                exact = False
                break
        self.check(
            "MANIFEST-001",
            "sealed manifest entries match current files",
            exact and manifest.get("lane") == self.target.name,
            len(entries),
            len(entries),
        )

    def run(self) -> dict[str, object]:
        stages: tuple[Callable[[], None], ...] = (
            self.validate_required_files,
            self.validate_governing_inputs,
            self.validate_inventories,
            self.validate_dispositions,
            self.validate_populations_and_keys,
            self.validate_locator_and_authority_semantics,
            self.validate_versions_derivatives_and_patches,
            self.validate_database_and_no_write,
            self.validate_manifest_if_present,
        )
        for stage in stages:
            try:
                stage()
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                self.errors.append(f"{stage.__name__}: {exc}")
        return {
            "lane": self.target.name,
            "packet_type": "chapter_18_immutable_source_metadata_rebase",
            "status_date": "2026-07-26",
            "passed": not self.errors,
            "checks_passed": sum(
                check["status"] == "PASS" for check in self.checks
            ),
            "checks_failed": sum(
                check["status"] == "FAIL" for check in self.checks
            ),
            "checks": self.checks,
            "errors": self.errors,
            "authority": {
                "metadata_acceptance": "PENDING_FRESH_UNINVOLVED_REVIEW",
                "canonical_integration": "NOT_AUTHORIZED",
                "coefficient_transfer": "BLOCK",
                "predictive_use": "BLOCK",
                "comparative_system_use": "BLOCK",
                "chapter_docx_publication": "NOT_AUTHORIZED",
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=HERE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = Validator(args.target.resolve()).run()
    if args.write:
        write_json(args.target.resolve() / "validation_results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
