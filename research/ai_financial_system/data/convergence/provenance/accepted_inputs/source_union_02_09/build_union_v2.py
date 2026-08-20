#!/usr/bin/env python3
"""Build the append-only normalized source-preservation union v2 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

LANE = "lane_3_source_preservation_union_v2"
SNAPSHOT = "SPU-V2-20260726-38b990b2"
SCRIPT_NAMES = {
    "build_union_v2.py",
    "validate_union_v2.py",
    "verify_two_clean_builds.py",
    "seal_union_v2.py",
}


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
V1 = STAGING / "lane_4_source_preservation_union_candidate"
V1_REVIEW = STAGING / "lane_1_verify_source_preservation_union_candidate"
DELTA = STAGING / "source_preservation_alternate_candidate"
DELTA_INVENTORY = STAGING / "source_preservation_alternate_candidates.csv"
DELTA_MANIFEST = DELTA / "source_archive_manifest.json"
DELTA_VALIDATION = DELTA / "source_archive_validation.json"
DB = REPO / "reports/ai_financial_system/research.db"
TASK = STAGING / "source_preservation_union_v2_task.md"
V1_TASK = STAGING / "source_preservation_union_candidate_task.md"
STANDARD = AI / "source_preservation_standard.md"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def source_version_id(source_id: str, timestamp: str) -> str:
    compact = "".join(character for character in timestamp if character.isdigit())[
        :14
    ]
    return f"SV-ALT-{source_id}-{compact}"


def governing_inputs() -> list[Path]:
    paths: set[Path] = {
        DB,
        TASK,
        V1_TASK,
        STANDARD,
        DELTA_INVENTORY,
        DELTA / "README.md",
        DELTA_MANIFEST,
        DELTA_VALIDATION,
        DELTA / "semantic_identity_review.csv",
    }
    for folder in (V1, V1_REVIEW):
        paths.update(path for path in folder.rglob("*") if path.is_file())
    delta_manifest = json.loads(DELTA_MANIFEST.read_text(encoding="utf-8"))
    for record in delta_manifest["records"]:
        for field in ("local_archive_path", "response_metadata_path"):
            value = record.get(field) or ""
            if value:
                path = REPO / value
                if "quarantine/unmanifested_pre_fix" in path.as_posix():
                    raise RuntimeError("quarantined pre-fix response entered manifest")
                paths.add(path)
    return sorted(paths)


def schema_text() -> str:
    return """-- Append-only source-preservation union v2 candidate; not canonical.
PRAGMA foreign_keys=ON;
CREATE TABLE source_versions (
  source_version_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  provenance_population TEXT NOT NULL, original_url TEXT,
  final_resolved_url TEXT, retrieval_timestamp_utc TEXT,
  publication_date TEXT, preservation_class TEXT NOT NULL,
  preservation_status TEXT NOT NULL, mime_type TEXT, byte_count INTEGER,
  content_sha256 TEXT, local_archive_path TEXT, preservation_method TEXT,
  response_metadata_state TEXT, access_licensing_boundary TEXT,
  artifact_role TEXT, cited_locator_relationship TEXT,
  exception_reason TEXT, candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_derivatives (
  derivative_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  parent_source_version_id TEXT NOT NULL REFERENCES source_versions,
  derivation_method_version TEXT NOT NULL, normalization_rule TEXT,
  derivative_sha256 TEXT NOT NULL, derivative_byte_count INTEGER NOT NULL,
  derivative_path TEXT NOT NULL, locator_aid_status TEXT NOT NULL,
  governing_boundary TEXT NOT NULL, candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_exceptions (
  exception_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  source_version_id TEXT NOT NULL REFERENCES source_versions,
  exception_type TEXT NOT NULL, exception_reason TEXT NOT NULL,
  content_sha256 TEXT, local_archive_path TEXT,
  resolution_state TEXT NOT NULL, publication_effect TEXT NOT NULL
);
CREATE TABLE duplicate_blob_reconciliation (
  source_id TEXT PRIMARY KEY,
  private_source_version_id TEXT NOT NULL REFERENCES source_versions,
  chapters_06_09_source_version_id TEXT NOT NULL REFERENCES source_versions,
  private_sha256 TEXT NOT NULL, chapters_06_09_sha256 TEXT NOT NULL,
  private_byte_count INTEGER NOT NULL, chapters_06_09_byte_count INTEGER NOT NULL,
  reconciliation_result TEXT NOT NULL, version_policy TEXT NOT NULL
);
CREATE TABLE carried_private_locator_reviews (
  claim_locator_review_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
  evidence_grade TEXT NOT NULL, source_ids TEXT NOT NULL, locator TEXT NOT NULL,
  archived_support TEXT NOT NULL, semantic_verdict TEXT NOT NULL,
  remaining_boundary TEXT NOT NULL, review_provenance TEXT NOT NULL,
  candidate_status TEXT NOT NULL
);
CREATE TABLE claim_locator_states (
  claim_locator_state_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
  chapter_id TEXT NOT NULL, source_id TEXT NOT NULL, locator TEXT NOT NULL,
  support_type TEXT NOT NULL, semantic_review_state TEXT NOT NULL,
  semantic_verdict TEXT, remaining_boundary TEXT NOT NULL,
  review_provenance TEXT NOT NULL
);
CREATE TABLE chapter_support_state_projection (
  projection_row_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL,
  chapter_id TEXT NOT NULL, claim_id TEXT NOT NULL, source_id TEXT NOT NULL,
  locator TEXT NOT NULL, support_type TEXT NOT NULL, support_origin TEXT NOT NULL,
  source_version_id TEXT REFERENCES source_versions,
  source_state_disposition TEXT NOT NULL, semantic_review_authority TEXT NOT NULL,
  candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_version_delta_audit (
  delta_audit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  historical_source_version_id TEXT NOT NULL REFERENCES source_versions,
  appended_source_version_id TEXT NOT NULL REFERENCES source_versions,
  historical_status TEXT NOT NULL, appended_status TEXT NOT NULL,
  alternate_venue_relationship TEXT NOT NULL, current_projection_version_id TEXT
    NOT NULL REFERENCES source_versions,
  semantic_locator_state TEXT NOT NULL, quarantine_excluded INTEGER NOT NULL,
  result TEXT NOT NULL
);
"""


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in list(output.iterdir()):
        if path.is_file() and path.name not in SCRIPT_NAMES:
            path.unlink()

    input_rows = [
        {
            "path": path.relative_to(REPO).as_posix(),
            "sha256": digest(path),
            "byte_count": path.stat().st_size,
        }
        for path in governing_inputs()
    ]
    write_csv(
        output / "sealed_input_inventory.csv",
        input_rows,
        ["path", "sha256", "byte_count"],
    )

    accepted_review = json.loads(
        (V1_REVIEW / "validation_results.json").read_text(encoding="utf-8")
    )
    if (
        not accepted_review["passed"]
        or accepted_review["verdict"]
        != "ACCEPT_FROZEN_V1_INTRINSICALLY_APPEND_ONLY_UNION_V2_REQUIRED"
    ):
        raise RuntimeError("v1 does not have the required fresh acceptance state")
    delta_validation = json.loads(DELTA_VALIDATION.read_text(encoding="utf-8"))
    if delta_validation.get("status") != "PASS":
        raise RuntimeError("alternate-source delta validation is not PASS")

    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    canonical_source_ids = {
        row[0] for row in connection.execute("SELECT source_id FROM sources")
    }
    connection.close()

    versions = read_csv(V1 / "source_versions.csv")
    if len(versions) != 57:
        raise RuntimeError("frozen v1 source-version population changed")
    delta_records = json.loads(DELTA_MANIFEST.read_text(encoding="utf-8"))["records"]
    appended: list[dict[str, str]] = []
    for record in delta_records:
        preserved = record["status"] == "preserved_complete"
        sid = record["source_id"]
        version_id = source_version_id(sid, record["retrieval_completed_utc"])
        appended.append(
            {
                "source_version_id": version_id,
                "source_id": sid,
                "provenance_population": "alternate_source_preservation_delta",
                "original_url": record["original_url"],
                "final_resolved_url": record.get("final_resolved_url", ""),
                "retrieval_timestamp_utc": record["retrieval_completed_utc"],
                "publication_date": record.get("publication_date", ""),
                "preservation_class": record["preservation_class"],
                "preservation_status": record["status"],
                "mime_type": record.get("mime_type", "") if preserved else "",
                "byte_count": str(record.get("byte_length", "")) if preserved else "",
                "content_sha256": record.get("content_sha256", "") if preserved else "",
                "local_archive_path": record.get("local_archive_path", "")
                if preserved
                else "",
                "preservation_method": record["preservation_method"],
                "response_metadata_state": (
                    "preserved_separate_response_json"
                    if record.get("response_metadata_path")
                    else "not_preserved"
                ),
                "access_licensing_boundary": (
                    record.get("access_or_licensing_restriction")
                    or "public_primary_alternate_venue"
                ),
                "artifact_role": "original_governing_bytes" if preserved else "none",
                "cited_locator_relationship": record["locator_relationship"],
                "exception_reason": record.get("exception_reason", ""),
                "candidate_integration_status": (
                    "candidate_alternate_source_version_ready_pending_atomic_rebase"
                    if preserved
                    else "candidate_append_only_failure_observation_retained"
                ),
            }
        )
    versions.extend(appended)
    versions.sort(
        key=lambda row: (
            row["source_id"],
            row["provenance_population"],
            row["retrieval_timestamp_utc"],
            row["source_version_id"],
        )
    )
    if len(versions) != 60 or len({row["source_version_id"] for row in versions}) != 60:
        raise RuntimeError("v2 source-version identity population is not 60 unique rows")
    if not {row["source_id"] for row in versions} <= canonical_source_ids:
        raise RuntimeError("v2 contains an unregistered canonical source")
    write_csv(output / "source_versions.csv", versions, list(versions[0]))

    for name in (
        "source_derivatives.csv",
        "duplicate_blob_reconciliation.csv",
        "carried_private_locator_reviews.csv",
        "claim_locator_states.csv",
    ):
        shutil.copyfile(V1 / name, output / name)

    exceptions = read_csv(V1 / "source_exceptions.csv")
    new_failure = next(
        row
        for row in appended
        if row["source_id"] == "SRC-0113"
        and row["preservation_status"] == "retrieval_failed"
    )
    exceptions.append(
        {
            "exception_id": (
                "EXC-ALT-SRC-0113-"
                + "".join(
                    character
                    for character in new_failure["retrieval_timestamp_utc"]
                    if character.isdigit()
                )[:14]
            ),
            "source_id": "SRC-0113",
            "source_version_id": new_failure["source_version_id"],
            "exception_type": "retrieval_failed",
            "exception_reason": new_failure["exception_reason"],
            "content_sha256": "",
            "local_archive_path": "",
            "resolution_state": "open_explicit_hold",
            "publication_effect": "claim_level_support_remains_held_pending_resolution",
        }
    )
    exceptions.sort(key=lambda row: row["exception_id"])
    write_csv(output / "source_exceptions.csv", exceptions, list(exceptions[0]))

    appended_by_source = {row["source_id"]: row for row in appended}
    original_failures = {
        row["source_id"]: row
        for row in read_csv(V1 / "source_versions.csv")
        if row["source_id"] in {"SRC-0112", "SRC-0113", "SRC-0132"}
        and row["preservation_status"] == "retrieval_failed"
    }
    identity_review = {
        row["source_id"]: row
        for row in read_csv(DELTA / "semantic_identity_review.csv")
    }
    delta_audit = []
    for sid in ("SRC-0112", "SRC-0113", "SRC-0132"):
        old = original_failures[sid]
        new = appended_by_source[sid]
        delta_audit.append(
            {
                "delta_audit_id": f"DELTA-V2-{sid}",
                "source_id": sid,
                "historical_source_version_id": old["source_version_id"],
                "appended_source_version_id": new["source_version_id"],
                "historical_status": old["preservation_status"],
                "appended_status": new["preservation_status"],
                "alternate_venue_relationship": identity_review[sid][
                    "review_status"
                ],
                "current_projection_version_id": new["source_version_id"],
                "semantic_locator_state": "pending_semantic_locator_review",
                "quarantine_excluded": 1,
                "result": "PASS_APPEND_ONLY_HISTORY_PRESERVED",
            }
        )
    write_csv(
        output / "source_version_delta_audit.csv",
        delta_audit,
        list(delta_audit[0]),
    )

    projection = read_csv(V1 / "chapter_02_09_support_state_projection.csv")
    for row in projection:
        row["snapshot_id"] = SNAPSHOT
        sid = row["source_id"]
        if sid in {"SRC-0112", "SRC-0132"}:
            row["source_version_id"] = appended_by_source[sid]["source_version_id"]
            row["source_state_disposition"] = (
                "preserved_bytes_available_locator_review_pending"
            )
            row["semantic_review_authority"] = (
                "not_semantically_cleared_by_union_candidate"
            )
        elif sid == "SRC-0113":
            row["source_version_id"] = appended_by_source[sid]["source_version_id"]
            row["source_state_disposition"] = "retrieval_failed"
            row["semantic_review_authority"] = (
                "not_semantically_cleared_by_union_candidate"
            )
        row["candidate_integration_status"] = (
            "derived_projection_v2_ready_pending_primary_atomic_rebase"
        )
    write_csv(
        output / "chapter_02_09_support_state_projection.csv",
        projection,
        list(projection[0]),
    )

    (output / "candidate_schema.sql").write_text(schema_text(), encoding="utf-8")
    counts = Counter(row["source_state_disposition"] for row in projection)
    summary = f"""# Normalized source-preservation union v2 candidate

## Result

The append-only v2 candidate retains all **57** accepted v1 observations and
appends all **3** validated delta observations. It therefore contains **60
source-version observations across 52 canonical source identities**, including
53 preserved observations, four retrieval-failure observations, and two
metadata-only P3 rights-boundary observations.

The original `SRC-0112`, `SRC-0113`, and `SRC-0132` failures are unchanged.
Alternate versions do not replace original venues. The later `SRC-0113` timeout
is a separate failure observation. The quarantined unmanifested PR Newswire
response is excluded.

## Current Chapters 2–9 support projection

{chr(10).join(f"- `{key}`: **{value}**" for key, value in sorted(counts.items()))}

The two newly preserved alternate versions change five relationships from
`retrieval_failed` to preserved-bytes/pending-locator status. Three `SRC-0113`
relationships remain failed. No new semantic locator has been cleared.

## Authority boundary

This isolated author packet may recommend normalized source-version v2 candidate
readiness only. It does not authorize canonical mutation, semantic locator
clearance, chapter prose, or publication. Fresh nonauthor review and primary
atomic integration remain required. The five private-investment analytical holds
remain unchanged.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")

    report = """# Append-only union v2 author report

## Rebase logic

The accepted frozen v1 tables are the immutable base. This build appends the
three manifest-controlled alternate retrieval observations and changes only the
derived current-state projection. It does not reinterpret the accepted private
locator reviews or the 159 pending Chapters 6–9 locator relationships.

`SRC-0112` now has both its issuer-endpoint failure and a later SEC-hosted
preserved version. `SRC-0132` has both its investor-site failure and a later
company-provided syndication version. `SRC-0113` has two separate issuer-hosted
failure observations. The delta identity review establishes alternate-document
identity/provenance only; it does not establish claim scope.

## Retained holds

- S&P 500 aggregate
- exact Microsoft component and tax attribution
- governed top-20 population
- prospective Meta and Tesla realization monitoring
- unnamed investee allocation and residual Amazon attribution

The packet retains all source-level, semantic, canonical-integration, chapter,
and publication gates.
"""
    (output / "author_report.md").write_text(report, encoding="utf-8")

    validation = validate(output)
    write_json(output / "validation_results.json", validation)
    if not validation["passed"]:
        raise RuntimeError(json.dumps(validation, indent=2))


def load_schema_check(output: Path) -> tuple[bool, str]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    tables = [
        ("source_versions", "source_versions.csv"),
        ("source_derivatives", "source_derivatives.csv"),
        ("source_exceptions", "source_exceptions.csv"),
        ("duplicate_blob_reconciliation", "duplicate_blob_reconciliation.csv"),
        ("carried_private_locator_reviews", "carried_private_locator_reviews.csv"),
        ("claim_locator_states", "claim_locator_states.csv"),
        (
            "chapter_support_state_projection",
            "chapter_02_09_support_state_projection.csv",
        ),
        ("source_version_delta_audit", "source_version_delta_audit.csv"),
    ]
    try:
        connection.executescript((output / "candidate_schema.sql").read_text())
        for table, filename in tables:
            rows = read_csv(output / filename)
            columns = list(rows[0])
            placeholders = ",".join("?" for _ in columns)
            sql = (
                f"INSERT INTO {table} ({','.join(columns)}) "
                f"VALUES ({placeholders})"
            )
            values = [
                [
                    None
                    if column == "source_version_id" and row[column] == ""
                    else row[column]
                    for column in columns
                ]
                for row in rows
            ]
            connection.executemany(sql, values)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table, _ in tables
        }
        expected = {
            "source_versions": 60,
            "source_derivatives": 8,
            "source_exceptions": 6,
            "duplicate_blob_reconciliation": 5,
            "carried_private_locator_reviews": 5,
            "claim_locator_states": 167,
            "chapter_support_state_projection": 525,
            "source_version_delta_audit": 3,
        }
        passed = integrity == "ok" and not foreign_keys and counts == expected
        return passed, json.dumps(
            {
                "integrity": integrity,
                "foreign_key_errors": len(foreign_keys),
                "counts": counts,
            },
            sort_keys=True,
        )
    except sqlite3.Error as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        connection.close()


def validate(output: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    versions = read_csv(output / "source_versions.csv")
    v1_versions = read_csv(V1 / "source_versions.csv")
    appended = [
        row
        for row in versions
        if row["provenance_population"] == "alternate_source_preservation_delta"
    ]
    check(
        "population_60_versions_52_sources",
        len(versions) == 60
        and len({row["source_version_id"] for row in versions}) == 60
        and len({row["source_id"] for row in versions}) == 52,
        f"{len(versions)}|{len({r['source_id'] for r in versions})}",
    )
    v1_by_id = {row["source_version_id"]: row for row in v1_versions}
    v2_by_id = {row["source_version_id"]: row for row in versions}
    check(
        "all_57_v1_rows_unchanged",
        all(v2_by_id.get(key) == row for key, row in v1_by_id.items()),
        str(sum(v2_by_id.get(key) == row for key, row in v1_by_id.items())),
    )
    preserved = [
        row for row in versions if row["preservation_status"] == "preserved_complete"
    ]
    failures = [
        row for row in versions if row["preservation_status"] == "retrieval_failed"
    ]
    rights = [
        row
        for row in versions
        if row["preservation_status"] == "metadata_only_rights_boundary"
    ]
    check(
        "status_population",
        len(preserved) == 53 and len(failures) == 4 and len(rights) == 2,
        f"preserved={len(preserved)} failures={len(failures)} rights={len(rights)}",
    )
    byte_ok = True
    for row in preserved:
        path = REPO / row["local_archive_path"]
        byte_ok &= (
            path.is_file()
            and digest(path) == row["content_sha256"]
            and path.stat().st_size == int(row["byte_count"])
        )
    check("all_53_preserved_bytes_rehashed", byte_ok, str(len(preserved)))
    check(
        "three_delta_rows_exact_statuses",
        len(appended) == 3
        and {
            row["source_id"]: row["preservation_status"] for row in appended
        }
        == {
            "SRC-0112": "preserved_complete",
            "SRC-0113": "retrieval_failed",
            "SRC-0132": "preserved_complete",
        },
        str(len(appended)),
    )
    new_failure = next(row for row in appended if row["source_id"] == "SRC-0113")
    check(
        "new_failure_has_no_synthetic_content",
        not new_failure["content_sha256"]
        and not new_failure["local_archive_path"]
        and "ReadTimeout" in new_failure["exception_reason"],
        new_failure["exception_reason"],
    )
    check(
        "three_historical_failures_unchanged",
        all(
            v2_by_id[row["source_version_id"]] == row
            for row in v1_versions
            if row["source_id"] in {"SRC-0112", "SRC-0113", "SRC-0132"}
            and row["preservation_status"] == "retrieval_failed"
        ),
        "3",
    )
    check(
        "private_reviews_and_derivatives_unchanged",
        digest(output / "carried_private_locator_reviews.csv")
        == digest(V1 / "carried_private_locator_reviews.csv")
        and digest(output / "source_derivatives.csv")
        == digest(V1 / "source_derivatives.csv"),
        "5 reviews|8 derivatives",
    )
    check(
        "locator_states_unchanged",
        len(read_csv(output / "claim_locator_states.csv")) == 167
        and digest(output / "claim_locator_states.csv")
        == digest(V1 / "claim_locator_states.csv"),
        "167",
    )
    projection = read_csv(output / "chapter_02_09_support_state_projection.csv")
    state_counts = Counter(row["source_state_disposition"] for row in projection)
    expected_counts = {
        "preserved_bytes_available_locator_review_pending": 326,
        "retrieval_failed": 3,
        "preserved_and_semantically_reviewed": 5,
        "no_matching_preservation_record": 186,
        "internal_P4_reconstruction": 3,
        "metadata_only_rights_boundary": 2,
    }
    check(
        "projection_525_exact_state_counts",
        len(projection) == 525
        and len({row["projection_row_id"] for row in projection}) == 525
        and dict(state_counts) == expected_counts
        and all(row["snapshot_id"] == SNAPSHOT for row in projection),
        json.dumps(dict(sorted(state_counts.items())), sort_keys=True),
    )
    check(
        "no_semantic_status_laundering",
        sum(
            row["source_state_disposition"]
            == "preserved_and_semantically_reviewed"
            for row in projection
        )
        == 5
        and sum(
            row["semantic_review_authority"]
            == "accepted_private_v2_nonauthor_review"
            for row in projection
        )
        == 5
        and all(
            row["semantic_review_authority"]
            == "not_semantically_cleared_by_union_candidate"
            for row in projection
            if row["source_id"] in {"SRC-0112", "SRC-0113", "SRC-0132"}
        ),
        "5 accepted; delta remains pending",
    )
    quarantine_paths = [
        row["local_archive_path"] for row in versions if row["local_archive_path"]
    ]
    check(
        "quarantine_excluded",
        all("quarantine/unmanifested_pre_fix" not in path for path in quarantine_paths),
        str(len(quarantine_paths)),
    )
    schema_ok, schema_detail = load_schema_check(output)
    check("isolated_sqlite_load_and_foreign_keys", schema_ok, schema_detail)
    check(
        "delta_audit_three_rows",
        len(read_csv(output / "source_version_delta_audit.csv")) == 3
        and all(
            row["result"] == "PASS_APPEND_ONLY_HISTORY_PRESERVED"
            for row in read_csv(output / "source_version_delta_audit.csv")
        ),
        "3",
    )
    passed = all(row["passed"] for row in checks)
    return {
        "validator": "source_preservation_union_v2_author_validator",
        "passed": passed,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "boundary": (
            "Author validation is not fresh acceptance, canonical integration, "
            "semantic locator clearance, chapter authority, or publication authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    build(output)
    result = json.loads((output / "validation_results.json").read_text())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
