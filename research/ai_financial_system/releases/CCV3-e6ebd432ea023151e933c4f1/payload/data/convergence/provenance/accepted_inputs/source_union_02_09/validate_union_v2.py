#!/usr/bin/env python3
"""Dedicated validator for the append-only source-preservation union v2."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


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
DELTA = STAGING / "source_preservation_alternate_candidate"
DB = REPO / "reports/ai_financial_system/research.db"
SNAPSHOT = "SPU-V2-20260726-38b990b2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def protected_paths() -> list[Path]:
    paths: set[Path] = {DB}
    for folder in (
        V1,
        STAGING / "lane_1_verify_source_preservation_union_candidate",
        DELTA,
        AI / "data",
        AI / "chapters",
        AI / "sources",
    ):
        paths.update(path for path in folder.rglob("*") if path.is_file())
    return sorted(paths)


def state(paths: list[Path]) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(REPO).as_posix(): (digest(path), path.stat().st_size)
        for path in paths
    }


def sqlite_load() -> tuple[bool, dict[str, Any]]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    mapping = [
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
    try:
        connection.executescript((HERE / "candidate_schema.sql").read_text())
        for table, filename in mapping:
            rows = read_csv(HERE / filename)
            columns = list(rows[0])
            statement = (
                f"INSERT INTO {table} ({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})"
            )
            connection.executemany(
                statement,
                [
                    [
                        None
                        if column == "source_version_id" and row[column] == ""
                        else row[column]
                        for column in columns
                    ]
                    for row in rows
                ],
            )
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table, _ in mapping
        }
        result = {
            "integrity_check": connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0],
            "foreign_key_violations": len(
                list(connection.execute("PRAGMA foreign_key_check"))
            ),
            "table_counts": counts,
        }
        return (
            result["integrity_check"] == "ok"
            and result["foreign_key_violations"] == 0
            and counts == expected,
            result,
        )
    except sqlite3.Error as error:
        return False, {"error": f"{type(error).__name__}: {error}"}
    finally:
        connection.close()


def validate() -> dict[str, Any]:
    before = state(protected_paths())
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    versions = read_csv(HERE / "source_versions.csv")
    version_by_id = {row["source_version_id"]: row for row in versions}
    v1_versions = read_csv(V1 / "source_versions.csv")
    v1_audit = []
    for row in v1_versions:
        actual = version_by_id.get(row["source_version_id"])
        ok = actual == row
        v1_audit.append(
            {
                "source_version_id": row["source_version_id"],
                "expected_row_sha256": row_digest(row),
                "actual_row_sha256": row_digest(actual) if actual else "",
                "result": "PASS_EXACT" if ok else "FAIL",
            }
        )
    check(
        "v1_append_only_retention",
        len(v1_audit) == 57
        and all(row["result"] == "PASS_EXACT" for row in v1_audit),
        f"{sum(r['result'] == 'PASS_EXACT' for r in v1_audit)}/57",
    )
    write_csv(
        HERE / "v1_retention_audit.csv",
        v1_audit,
        list(v1_audit[0]),
    )

    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    canonical = {row[0] for row in connection.execute("SELECT source_id FROM sources")}
    connection.close()
    check(
        "version_identity_and_canonical_sources",
        len(versions) == 60
        and len(version_by_id) == 60
        and len({row["source_id"] for row in versions}) == 52
        and {row["source_id"] for row in versions} <= canonical,
        f"{len(versions)} versions|{len({r['source_id'] for r in versions})} sources",
    )

    preserved = [
        row for row in versions if row["preservation_status"] == "preserved_complete"
    ]
    byte_audit = []
    for row in preserved:
        path = REPO / row["local_archive_path"]
        ok = (
            path.is_file()
            and digest(path) == row["content_sha256"]
            and path.stat().st_size == int(row["byte_count"])
            and bool(row["retrieval_timestamp_utc"])
            and bool(row["final_resolved_url"])
            and bool(row["mime_type"])
            and bool(row["preservation_method"])
            and bool(row["provenance_population"])
        )
        byte_audit.append(
            {
                "source_version_id": row["source_version_id"],
                "source_id": row["source_id"],
                "declared_sha256": row["content_sha256"],
                "actual_sha256": digest(path) if path.is_file() else "",
                "declared_bytes": row["byte_count"],
                "actual_bytes": path.stat().st_size if path.is_file() else "",
                "result": "PASS" if ok else "FAIL",
            }
        )
    check(
        "all_preserved_bytes_and_provenance",
        len(byte_audit) == 53 and all(row["result"] == "PASS" for row in byte_audit),
        f"{sum(r['result'] == 'PASS' for r in byte_audit)}/53",
    )
    write_csv(
        HERE / "preserved_byte_reperformance.csv",
        byte_audit,
        list(byte_audit[0]),
    )

    manifest_records = {
        row["source_id"]: row
        for row in json.loads(
            (DELTA / "source_archive_manifest.json").read_text()
        )["records"]
    }
    appended = {
        row["source_id"]: row
        for row in versions
        if row["provenance_population"] == "alternate_source_preservation_delta"
    }
    delta_audit = []
    delta_ok = set(appended) == set(manifest_records) == {
        "SRC-0112",
        "SRC-0113",
        "SRC-0132",
    }
    for sid in sorted(manifest_records):
        source = manifest_records[sid]
        actual = appended.get(sid, {})
        preserved_source = source["status"] == "preserved_complete"
        ok = (
            actual.get("original_url") == source["original_url"]
            and actual.get("final_resolved_url")
            == source.get("final_resolved_url", "")
            and actual.get("retrieval_timestamp_utc")
            == source["retrieval_completed_utc"]
            and actual.get("publication_date") == source["publication_date"]
            and actual.get("preservation_status") == source["status"]
            and actual.get("preservation_class") == source["preservation_class"]
            and actual.get("preservation_method") == source["preservation_method"]
            and actual.get("exception_reason") == source.get("exception_reason", "")
            and actual.get("content_sha256")
            == (source.get("content_sha256", "") if preserved_source else "")
            and actual.get("local_archive_path")
            == (source.get("local_archive_path", "") if preserved_source else "")
        )
        delta_ok &= ok
        delta_audit.append(
            {
                "delta_audit_id": f"VALIDATE-{sid}",
                "source_id": sid,
                "manifest_status": source["status"],
                "candidate_status": actual.get("preservation_status", ""),
                "manifest_sha256": source.get("content_sha256", ""),
                "candidate_sha256": actual.get("content_sha256", ""),
                "result": "PASS" if ok else "FAIL",
            }
        )
    check(
        "delta_manifest_exact_reperformance",
        delta_ok,
        f"{sum(r['result'] == 'PASS' for r in delta_audit)}/3",
    )
    write_csv(
        HERE / "delta_manifest_reperformance.csv",
        delta_audit,
        list(delta_audit[0]),
    )

    failures = [
        row for row in versions if row["preservation_status"] == "retrieval_failed"
    ]
    rights = [
        row
        for row in versions
        if row["preservation_status"] == "metadata_only_rights_boundary"
    ]
    check(
        "failure_and_rights_population_fail_closed",
        len(failures) == 4
        and len(rights) == 2
        and all(
            not row["content_sha256"] and not row["local_archive_path"]
            for row in [*failures, *rights]
        ),
        f"failures={len(failures)} rights={len(rights)}",
    )
    exceptions = read_csv(HERE / "source_exceptions.csv")
    check(
        "six_explicit_exception_observations",
        len(exceptions) == 6
        and len({row["exception_id"] for row in exceptions}) == 6
        and all(
            row["source_version_id"] in version_by_id
            and not row["content_sha256"]
            and not row["local_archive_path"]
            for row in exceptions
        ),
        str(len(exceptions)),
    )

    unchanged_names = (
        "source_derivatives.csv",
        "duplicate_blob_reconciliation.csv",
        "carried_private_locator_reviews.csv",
        "claim_locator_states.csv",
    )
    unchanged = {
        name: digest(HERE / name) == digest(V1 / name) for name in unchanged_names
    }
    check(
        "accepted_v1_semantic_and_blob_tables_unchanged",
        all(unchanged.values())
        and len(read_csv(HERE / "source_derivatives.csv")) == 8
        and len(read_csv(HERE / "duplicate_blob_reconciliation.csv")) == 5
        and len(read_csv(HERE / "carried_private_locator_reviews.csv")) == 5
        and len(read_csv(HERE / "claim_locator_states.csv")) == 167,
        json.dumps(unchanged, sort_keys=True),
    )

    projection = read_csv(HERE / "chapter_02_09_support_state_projection.csv")
    v1_projection = {
        row["projection_row_id"]: row
        for row in read_csv(V1 / "chapter_02_09_support_state_projection.csv")
    }
    delta_current = {
        sid: row["source_version_id"] for sid, row in appended.items()
    }
    projection_audit = []
    projection_ok = len(projection) == 525
    for row in projection:
        prior = v1_projection.get(row["projection_row_id"])
        expected_state = prior["source_state_disposition"] if prior else ""
        expected_version = prior["source_version_id"] if prior else ""
        if row["source_id"] in {"SRC-0112", "SRC-0132"}:
            expected_state = "preserved_bytes_available_locator_review_pending"
            expected_version = delta_current[row["source_id"]]
        elif row["source_id"] == "SRC-0113":
            expected_state = "retrieval_failed"
            expected_version = delta_current[row["source_id"]]
        ok = (
            prior is not None
            and row["snapshot_id"] == SNAPSHOT
            and row["source_state_disposition"] == expected_state
            and row["source_version_id"] == expected_version
            and row["semantic_review_authority"]
            == (
                "accepted_private_v2_nonauthor_review"
                if expected_state == "preserved_and_semantically_reviewed"
                else "not_semantically_cleared_by_union_candidate"
            )
            and all(
                row[field] == prior[field]
                for field in (
                    "projection_row_id",
                    "chapter_id",
                    "claim_id",
                    "source_id",
                    "locator",
                    "support_type",
                    "support_origin",
                )
            )
        )
        projection_ok &= ok
        projection_audit.append(
            {
                "projection_row_id": row["projection_row_id"],
                "source_id": row["source_id"],
                "expected_state": expected_state,
                "actual_state": row["source_state_disposition"],
                "expected_source_version_id": expected_version,
                "actual_source_version_id": row["source_version_id"],
                "result": "PASS" if ok else "FAIL",
            }
        )
    counts = Counter(row["source_state_disposition"] for row in projection)
    expected_counts = {
        "preserved_bytes_available_locator_review_pending": 326,
        "retrieval_failed": 3,
        "preserved_and_semantically_reviewed": 5,
        "no_matching_preservation_record": 186,
        "internal_P4_reconstruction": 3,
        "metadata_only_rights_boundary": 2,
    }
    projection_ok &= dict(counts) == expected_counts
    check(
        "projection_exact_reperformance_and_counts",
        projection_ok
        and all(row["result"] == "PASS" for row in projection_audit),
        f"{sum(r['result'] == 'PASS' for r in projection_audit)}/525 "
        f"{json.dumps(dict(sorted(counts.items())))}",
    )
    write_csv(
        HERE / "projection_reperformance.csv",
        projection_audit,
        list(projection_audit[0]),
    )

    quarantine_ok = all(
        "quarantine/unmanifested_pre_fix" not in row["local_archive_path"]
        for row in versions
    )
    check("quarantine_excluded", quarantine_ok, "all 60 rows")

    sqlite_ok, sqlite_detail = sqlite_load()
    check(
        "schema_load_integrity_and_foreign_keys",
        sqlite_ok,
        json.dumps(sqlite_detail, sort_keys=True),
    )
    (HERE / "sqlite_load_results.json").write_text(
        json.dumps(
            {"passed": sqlite_ok, **sqlite_detail}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    after = state(protected_paths())
    no_write_rows = [
        {
            "path": path,
            "before_sha256": before.get(path, ("", 0))[0],
            "after_sha256": after.get(path, ("", 0))[0],
            "result": "PASS_UNCHANGED"
            if before.get(path) == after.get(path)
            else "FAIL_CHANGED",
        }
        for path in sorted(set(before) | set(after))
    ]
    no_write_ok = all(row["result"] == "PASS_UNCHANGED" for row in no_write_rows)
    write_csv(
        HERE / "no_write_audit.csv",
        no_write_rows,
        list(no_write_rows[0]),
    )
    check(
        "protected_inputs_unchanged",
        no_write_ok,
        f"{sum(r['result'] == 'PASS_UNCHANGED' for r in no_write_rows)}/"
        f"{len(no_write_rows)}",
    )

    passed = all(row["passed"] for row in checks)
    result = {
        "validator": "source_preservation_union_v2_dedicated_validator",
        "passed": passed,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "canonical_integration_authorized": False,
        "semantic_locator_clearance": "none_beyond_accepted_v1_five_claim_reviews",
        "chapter_prose_authorized": False,
        "publication_authorized": False,
    }
    (HERE / "dedicated_validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
