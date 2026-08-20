#!/usr/bin/env python3
"""Validate the narrow Chapters 6-9 v4 source-boundary repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
REPO = HERE.parents[3]
STAGING = REPO / "research/ai_financial_system/staging"
V3 = STAGING / "lane_2_chapters_06_09_rebuilt_v3"
REVIEW = STAGING / "lane_3_verify_chapters_06_09_rebuilt_v3"
UNION = STAGING / "lane_3_source_preservation_union_v2"
SNAPSHOT = "SPU-V2-20260726-38b990b2"
UNION_REVIEW_SHA = "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
MUTABLE_UNION_FIELDS = {
    "source_state_disposition",
    "semantic_review_authority",
    "candidate_integration_status",
    "source_version_status",
}
COUNT_EXPECTATIONS = {
    "proposed_atomic_claims.csv": 117,
    "claim_source_support_junction.csv": 164,
    "source_version_candidates.csv": 51,
    "proposed_unknowns.csv": 81,
    "proposed_handoffs.csv": 21,
    "proposed_confidence_history.csv": 117,
    "proposed_economic_importance.csv": 117,
    "typed_row_dispositions.csv": 403,
    "typed_field_dispositions.csv": 7078,
    "private_row_lineage_audit.csv": 61,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def support_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["claim_id"],
        row["source_id"],
        row["locator"],
        row["support_type"],
    )


def check(
    rows: list[dict[str, object]],
    check_id: str,
    observed: object,
    expected: object,
    passed: bool,
    evidence: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "observed": observed,
            "expected": expected,
            "result": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
    )


def validate(target: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    _, frozen = read_csv(REVIEW / "frozen_intrinsic_artifacts.csv")
    frozen_audit = []
    for row in frozen:
        path = target / row["path"]
        observed_hash = sha(path) if path.exists() else ""
        observed_bytes = path.stat().st_size if path.exists() else -1
        passed = observed_hash == row["sha256"] and observed_bytes == int(row["bytes"])
        frozen_audit.append(
            {
                "path": row["path"],
                "expected_sha256": row["sha256"],
                "observed_sha256": observed_hash,
                "expected_bytes": row["bytes"],
                "observed_bytes": observed_bytes,
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(target / "frozen_intrinsic_reperformance.csv", frozen_audit)
    check(
        checks,
        "FROZEN-18",
        f"rows={len(frozen_audit)};failures={sum(row['result'] == 'FAIL' for row in frozen_audit)}",
        "18 byte-identical intrinsic artifacts",
        len(frozen_audit) == 18
        and all(row["result"] == "PASS" for row in frozen_audit),
        "frozen_intrinsic_reperformance.csv",
    )

    for name, expected in COUNT_EXPECTATIONS.items():
        _, rows = read_csv(target / name)
        check(checks, f"COUNT-{name}", len(rows), expected, len(rows) == expected, name)

    v3_fields, v3_support = read_csv(V3 / "claim_source_support_junction.csv")
    fields, support = read_csv(target / "claim_source_support_junction.csv")
    _, projection = read_csv(UNION / "chapter_02_09_support_state_projection.csv")
    projection_by_key = {support_key(row): row for row in projection}
    author_keys = {support_key(row) for row in support}
    v3_keys = {support_key(row) for row in v3_support}
    required_fields = {
        "preservation_projection_file",
        "preservation_snapshot_id",
        "preservation_projection_row_id",
        "preservation_source_version_id",
        "preservation_review_manifest_sha256",
    }
    reference_errors = []
    reference_audit = []
    for row in support:
        key = support_key(row)
        external = projection_by_key.get(key)
        passed = bool(
            external
            and row["preservation_projection_file"]
            == (
                "research/ai_financial_system/staging/"
                "lane_3_source_preservation_union_v2/"
                "chapter_02_09_support_state_projection.csv"
            )
            and row["preservation_snapshot_id"] == SNAPSHOT
            and row["preservation_projection_row_id"] == external["projection_row_id"]
            and row["preservation_source_version_id"] == external["source_version_id"]
            and row["preservation_review_manifest_sha256"] == UNION_REVIEW_SHA
        )
        if not passed:
            reference_errors.append(key)
        reference_audit.append(
            {
                "reference_audit_id": f"SURR-{len(reference_audit) + 1:04d}",
                "claim_id": row["claim_id"],
                "source_id": row["source_id"],
                "locator": row["locator"],
                "support_type": row["support_type"],
                "projection_row_id": row["preservation_projection_row_id"],
                "source_version_id": row["preservation_source_version_id"],
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(target / "source_union_reference_reperformance.csv", reference_audit)
    check(
        checks,
        "SUPPORT-KEYS",
        f"v3={len(v3_keys)};v4={len(author_keys)};difference={len(v3_keys ^ author_keys)}",
        "164 exact unchanged semantic keys",
        len(author_keys) == len(v3_keys) == 164 and author_keys == v3_keys,
        "claim_source_support_junction.csv",
    )
    check(
        checks,
        "IMMUTABLE-REFERENCES",
        f"rows={len(support)};errors={len(reference_errors)}",
        "164 complete exact external references",
        len(support) == 164 and not reference_errors,
        "source_union_reference_reperformance.csv",
    )
    check(
        checks,
        "NO-MUTABLE-AUTHORITY",
        sorted(MUTABLE_UNION_FIELDS & set(fields)),
        "no copied mutable preservation authority fields",
        not (MUTABLE_UNION_FIELDS & set(fields))
        and required_fields.issubset(fields)
        and "source_version_status" in v3_fields,
        "claim_source_support_junction.csv header",
    )

    _, exact_repairs = read_csv(REVIEW / "source_union_exact_repairs.csv")
    support_by_key = {support_key(row): row for row in support}
    alternate_audit = []
    for expected in exact_repairs:
        key = support_key(expected)
        row = support_by_key.get(key)
        passed = bool(
            row
            and row["preservation_snapshot_id"] == expected["required_snapshot_id"]
            and row["preservation_projection_row_id"]
            == expected["required_projection_row_id"]
            and row["preservation_source_version_id"]
            == expected["required_source_version_id"]
        )
        alternate_audit.append(
            {
                "claim_id": expected["claim_id"],
                "source_id": expected["source_id"],
                "required_projection_row_id": expected["required_projection_row_id"],
                "observed_projection_row_id": (
                    row["preservation_projection_row_id"] if row else ""
                ),
                "required_source_version_id": expected["required_source_version_id"],
                "observed_source_version_id": (
                    row["preservation_source_version_id"] if row else ""
                ),
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(target / "alternate_version_reperformance.csv", alternate_audit)
    check(
        checks,
        "ALTERNATE-5",
        f"rows={len(alternate_audit)};failures={sum(row['result'] == 'FAIL' for row in alternate_audit)}",
        "5 exact accepted alternate-version references",
        len(alternate_audit) == 5
        and all(row["result"] == "PASS" for row in alternate_audit),
        "alternate_version_reperformance.csv",
    )

    _, union_versions = read_csv(UNION / "source_versions.csv")
    failures_retained = all(
        any(
            row["source_id"] == source_id
            and row["preservation_status"] == "retrieval_failed"
            for row in union_versions
        )
        for source_id in ("SRC-0112", "SRC-0132")
    )
    alternates_retained = all(
        any(
            row["source_version_id"] == version_id
            and row["preservation_status"] == "preserved_complete"
            for row in union_versions
        )
        for version_id in (
            "SV-ALT-SRC-0112-20260726180523",
            "SV-ALT-SRC-0132-20260726180826",
        )
    )
    check(
        checks,
        "APPEND-ONLY-FAILURES",
        f"failures_retained={failures_retained};alternates_retained={alternates_retained}",
        "original failures and alternate versions both retained externally",
        failures_retained and alternates_retained,
        "accepted source-union-v2 source_versions.csv",
    )

    _, dependency = read_csv(target / "source_preservation_dependency.csv")
    dependency_row = dependency[0]
    check(
        checks,
        "DEPENDENCY-PIN",
        dependency_row["source_union_review_manifest_sha256"],
        UNION_REVIEW_SHA,
        len(dependency) == 1
        and dependency_row["snapshot_id"] == SNAPSHOT
        and dependency_row["source_union_review_manifest_sha256"]
        == UNION_REVIEW_SHA
        and dependency_row["dependency_authority"]
        == "immutable_external_reference_only",
        "source_preservation_dependency.csv",
    )
    for name, expected in (("source_queue.csv", 48), ("source_version_candidates.csv", 51)):
        _, rows = read_csv(target / name)
        passed = len(rows) == expected and all(
            row["authority_status"]
            == "historical_author_observation_nonauthoritative"
            for row in rows
        )
        check(
            checks,
            f"HISTORICAL-{name}",
            f"rows={len(rows)};labeled={sum(row.get('authority_status') == 'historical_author_observation_nonauthoritative' for row in rows)}",
            f"{expected} historical/nonauthoritative rows",
            passed,
            name,
        )

    # Every non-targeted hold and intrinsic boundary remains exact.
    hold_files = [
        "external_holds.csv",
        "private_investment_dependency_boundary.csv",
        "chapter18_dependency_rebase.csv",
        "proposed_falsifiers.csv",
    ]
    hold_audit = [
        {
            "path": name,
            "v3_sha256": sha(V3 / name),
            "v4_sha256": sha(target / name),
            "result": "PASS" if sha(V3 / name) == sha(target / name) else "FAIL",
        }
        for name in hold_files
    ]
    write_csv(target / "hold_boundary_reperformance.csv", hold_audit)
    check(
        checks,
        "HOLDS-FROZEN",
        f"files={len(hold_audit)};failures={sum(row['result'] == 'FAIL' for row in hold_audit)}",
        "all hold/boundary files byte-identical",
        all(row["result"] == "PASS" for row in hold_audit),
        "hold_boundary_reperformance.csv",
    )

    _, inventory = read_csv(target / "governing_input_inventory.csv")
    no_write = [
        {
            "path": row["path"],
            "sealed_sha256": row["sha256"],
            "current_sha256": sha(REPO / row["path"]),
            "sealed_bytes": row["bytes"],
            "current_bytes": (REPO / row["path"]).stat().st_size,
            "result": (
                "PASS"
                if row["sha256"] == sha(REPO / row["path"])
                and int(row["bytes"]) == (REPO / row["path"]).stat().st_size
                else "FAIL"
            ),
        }
        for row in inventory
    ]
    write_csv(target / "no_write_audit.csv", no_write)
    check(
        checks,
        "NO-WRITE",
        f"rows={len(no_write)};failures={sum(row['result'] == 'FAIL' for row in no_write)}",
        "all protected/governing inputs unchanged",
        all(row["result"] == "PASS" for row in no_write),
        "no_write_audit.csv",
    )

    failures = [row for row in checks if row["result"] == "FAIL"]
    write_csv(target / "semantic_validation_checks_v4.csv", checks)
    result = {
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    write_json(target / "validation_results.json", result)
    (target / "validation_report.md").write_text(
        "# Chapters 6-9 rebuilt-v4 author validation\n\n"
        f"Dedicated checks: {result['passed']}/{result['checks']} pass; "
        f"failures: {result['failed']}. This narrow metadata repair is ready only "
        "for fresh uninvolved review. Canonical integration, chapter or DOCX "
        "replacement, and publication remain unauthorized.\n",
        encoding="utf-8",
    )

    artifacts = []
    for path in sorted(
        item
        for item in target.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.name not in {"lane_manifest.json", "generic_validation_results.json"}
    ):
        entry: dict[str, object] = {
            "path": path.relative_to(target).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            _, rows = read_csv(path)
            entry["rows"] = len(rows)
        else:
            entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
        artifacts.append(entry)
    manifest = {
        "lane": target.name,
        "workstream": "chapters_06_09_rebuilt_v4_source_boundary_repair",
        "status_date": "2026-07-26",
        "author_verdict": (
            "READY_FOR_FRESH_UNINVOLVED_REVIEW"
            if result["status"] == "PASS"
            else "NOT_READY"
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "chapter_replacement": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "NOT_AUTHORIZED",
        "validation": result,
        "files": artifacts,
        "self_hash_policy": (
            "lane_manifest.json and generic_validation_results.json are excluded "
            "from manifest entries"
        ),
    }
    write_json(target / "lane_manifest.json", manifest)
    process = subprocess.run(
        [
            sys.executable,
            str(REPO / "research/ai_financial_system/scripts/validate_staging_packet.py"),
            str(target),
            "--json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        generic = json.loads(process.stdout)
    except json.JSONDecodeError:
        generic = {
            "passed": False,
            "errors": [process.stdout, process.stderr],
            "returncode": process.returncode,
        }
    write_json(target / "generic_validation_results.json", generic)
    if not generic.get("passed"):
        result["status"] = "FAIL"
        result["generic_validation_errors"] = generic.get("errors", [])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=HERE)
    args = parser.parse_args()
    result = validate(args.target.resolve())
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
