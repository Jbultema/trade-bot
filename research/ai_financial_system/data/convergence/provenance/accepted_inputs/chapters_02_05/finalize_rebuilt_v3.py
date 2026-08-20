#!/usr/bin/env python3
"""Run final controls and seal the Chapters 2-5 rebuilt-v3 author packet."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AI = REPO / "research/ai_financial_system"
GENERIC = AI / "scripts/validate_staging_packet.py"
SCRIPTS = [
    HERE / "build_rebuilt_v3.py",
    HERE / "validate_rebuilt_v3.py",
    HERE / "verify_two_builds.py",
    HERE / "finalize_rebuilt_v3.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def seal(validation: dict[str, object]) -> None:
    files = []
    for path in sorted(p for p in HERE.rglob("*") if p.is_file() and p.name != "lane_manifest.json"):
        entry = {
            "path": path.relative_to(HERE).as_posix(),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            entry["rows"] = csv_rows(path)
        else:
            entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
        files.append(entry)
    manifest = {
        "lane": HERE.name,
        "workstream": "chapters_02_05_metadata_only_normalized_preservation_rebase_v3",
        "status_date": "2026-07-26",
        "author_verdict": validation["author_verdict"],
        "union_snapshot": "SPU-V2-20260726-38b990b2",
        "union_acceptance_manifest_sha256": (
            "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "BLOCKED",
        "validation": validation,
        "self_hash_policy": "lane_manifest.json excludes its own hash",
        "files": files,
    }
    write_json(HERE / "lane_manifest.json", manifest)


def main() -> int:
    dedicated = json.loads((HERE / "dedicated_validation_results.json").read_text())
    clean = json.loads((HERE / "two_clean_build_results.json").read_text())
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    ruff = subprocess.run(
        [str(REPO / ".venv/bin/ruff"), "check", *map(str, SCRIPTS)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    ruff_result = {
        "passed": ruff.returncode == 0,
        "returncode": ruff.returncode,
        "files_checked": len(SCRIPTS),
        "stdout": ruff.stdout,
        "stderr": ruff.stderr,
    }
    write_json(HERE / "ruff_results.json", ruff_result)
    checks = [
        {
            "check": "dedicated_metadata_rebase_validation",
            "passed": dedicated["passed"],
            "detail": f"{dedicated['checks_passed']}/{dedicated['checks_total']}",
        },
        {
            "check": "two_clean_builds",
            "passed": clean["passed"],
            "detail": f"{clean['files_compared']} files",
        },
        {
            "check": "ruff",
            "passed": ruff_result["passed"],
            "detail": f"{ruff_result['files_checked']} scripts",
        },
    ]
    passed = all(row["passed"] for row in checks)
    validation: dict[str, object] = {
        "validator": "chapters_02_05_rebuilt_v3_final_author_validator",
        "passed": passed,
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": sum(row["passed"] for row in checks),
        "author_verdict": (
            "READY_FOR_FRESH_NONAUTHOR_METADATA_REBASE_REVIEW"
            if passed
            else "REVISE_METADATA_REBASE"
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "BLOCKED",
    }
    write_json(HERE / "validation_results.json", validation)
    seal(validation)
    generic = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(GENERIC), str(HERE), "--json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        generic_result = json.loads(generic.stdout)
    except json.JSONDecodeError:
        generic_result = {
            "passed": False,
            "errors": [generic.stdout, generic.stderr],
        }
    write_json(HERE / "generic_validation_results.json", generic_result)
    checks.append(
        {
            "check": "generic_staging_validation",
            "passed": bool(generic_result.get("passed")),
            "detail": (
                f"files={generic_result.get('files_checked', 0)} "
                f"csv_rows={generic_result.get('csv_rows_checked', 0)}"
            ),
        }
    )
    validation["checks"] = checks
    validation["checks_total"] = len(checks)
    validation["checks_passed"] = sum(row["passed"] for row in checks)
    validation["passed"] = all(row["passed"] for row in checks)
    if not validation["passed"]:
        validation["author_verdict"] = "REVISE_METADATA_REBASE"
    write_json(HERE / "validation_results.json", validation)
    seal(validation)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
