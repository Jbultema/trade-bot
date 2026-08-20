#!/usr/bin/env python3
"""Build, validate, compare, and seal the Chapters 11-15 metadata rebase."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GENERIC_VALIDATOR = (
    ROOT / "research/ai_financial_system/scripts/validate_staging_packet.py"
)
SCRIPT_NAMES = [
    "build_metadata_rebase.py",
    "run_author_checks.py",
    "seal_manifest.py",
    "validate_all_cell_semantics.py",
    "validate_metadata_rebase.py",
    "verify_two_clean_builds.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generic_validation() -> dict[str, object]:
    result = run(
        [
            sys.executable,
            str(GENERIC_VALIDATOR),
            str(HERE),
            "--json",
        ]
    )
    parsed = json.loads(result.stdout)
    if not parsed["passed"]:
        raise RuntimeError(f"generic staging validation failed: {parsed['errors']}")
    return parsed


def main() -> int:
    for name in [
        "generic_validation_results.json",
        "lane_manifest.json",
        "ruff_results.json",
        "two_clean_build_comparison.csv",
        "two_clean_build_results.json",
        "all_cell_semantic_validation_results.json",
        "validation_results.json",
    ]:
        (HERE / name).unlink(missing_ok=True)

    run([sys.executable, str(HERE / "build_metadata_rebase.py"), "--output", str(HERE)])
    run([sys.executable, str(HERE / "validate_metadata_rebase.py"), "--target", str(HERE)])
    run([sys.executable, str(HERE / "verify_two_clean_builds.py")])

    ruff_command = [
        "ruff",
        "check",
        "--no-cache",
        *[str(HERE / name) for name in SCRIPT_NAMES],
    ]
    ruff_result = run(ruff_command)
    write_json(
        HERE / "ruff_results.json",
        {
            "status": "PASS",
            "command": "ruff check --no-cache " + " ".join(SCRIPT_NAMES),
            "files_checked": len(SCRIPT_NAMES),
            "stdout": ruff_result.stdout.strip(),
        },
    )

    write_json(
        HERE / "generic_validation_results.json",
        {
            "checks": [],
            "csv_rows_checked": 0,
            "errors": [],
            "files_checked": 0,
            "lane": HERE.name,
            "passed": True,
            "warnings": [],
        },
    )
    run([sys.executable, str(HERE / "seal_manifest.py")])
    first_generic = generic_validation()
    write_json(HERE / "generic_validation_results.json", first_generic)
    run([sys.executable, str(HERE / "seal_manifest.py")])

    generic_hash_before = sha256(HERE / "generic_validation_results.json")
    second_generic = generic_validation()
    write_json(HERE / "generic_validation_results.json", second_generic)
    generic_hash_after = sha256(HERE / "generic_validation_results.json")
    if first_generic != second_generic or generic_hash_before != generic_hash_after:
        raise RuntimeError("generic validation was not stable after final reseal")

    first_manifest_hash = sha256(HERE / "lane_manifest.json")
    run([sys.executable, str(HERE / "seal_manifest.py")])
    second_manifest_hash = sha256(HERE / "lane_manifest.json")
    if first_manifest_hash != second_manifest_hash:
        raise RuntimeError("lane manifest was not stable across consecutive seals")

    final_generic = generic_validation()
    if final_generic != second_generic:
        raise RuntimeError("final sealed packet did not reproduce generic validation")
    print(f"PASS: manifest_sha256={second_manifest_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
