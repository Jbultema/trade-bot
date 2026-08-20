#!/usr/bin/env python3
"""Run two clean v3 builds and prove protected inputs receive no writes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILDER = HERE / "build_packet.py"
VALIDATOR = HERE / "validate_packet.py"

RUNTIME_ARTIFACTS = {
    "deterministic_build_results.json",
    "generic_validation_results.json",
    "lane_manifest.json",
    "manifest_verification.json",
    "no_write_audit.csv",
    "ruff_results.json",
    "two_clean_build_comparison.csv",
    "validation_results.json",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: sha(path)}
    return {
        item.relative_to(path).as_posix(): sha(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
        and item.name not in RUNTIME_ARTIFACTS
    }


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for relative, value in inventory(path).items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    protected = [
        ROOT / "research/ai_financial_system/data",
        ROOT / "research/ai_financial_system/chapters/chapter_10",
        ROOT / "reports/ai_financial_system/chapters/chapter_10",
        ROOT / "reports/ai_financial_system/research.db",
        ROOT / "reports/ai_financial_system/database_manifest.json",
        ROOT
        / "research/ai_financial_system/scripts/"
        "build_chapter_10_public_capital_data.py",
        ROOT
        / "research/ai_financial_system/staging/"
        "lane_1_chapter_10_rebuilt_v1",
        ROOT
        / "research/ai_financial_system/staging/"
        "lane_2_verify_chapter_10_rebuilt_v1",
        ROOT
        / "research/ai_financial_system/staging/"
        "lane_4_audit_chapters_10_15_convergence_inputs",
        ROOT
        / "research/ai_financial_system/staging/"
        "lane_3_chapter_10_rebuilt_v2",
        ROOT
        / "research/ai_financial_system/staging/"
        "lane_4_verify_chapter_10_rebuilt_v2",
    ]
    if not all(path.exists() for path in protected):
        missing = [path.relative_to(ROOT).as_posix() for path in protected if not path.exists()]
        raise FileNotFoundError(f"protected targets missing: {missing}")
    before = {path.relative_to(ROOT).as_posix(): tree_hash(path) for path in protected}

    with (
        tempfile.TemporaryDirectory(prefix="ch10-v3-a-") as first,
        tempfile.TemporaryDirectory(prefix="ch10-v3-b-") as second,
    ):
        build_roots = [Path(first) / "packet", Path(second) / "packet"]
        run_results: list[dict[str, Any]] = []
        for build_index, output in enumerate(build_roots, start=1):
            build = run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--output",
                    str(output),
                ]
            )
            validate = run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--target",
                    str(output),
                    "--no-write",
                ]
            )
            run_results.append(
                {
                    "build_index": build_index,
                    "build_returncode": build.returncode,
                    "validation_returncode": validate.returncode,
                    "build_stdout": build.stdout.strip(),
                    "build_stderr": build.stderr.strip(),
                    "validation_status": (
                        json.loads(validate.stdout).get("status")
                        if validate.stdout.strip()
                        else "NO_OUTPUT"
                    ),
                    "validation_stderr": validate.stderr.strip(),
                }
            )
        inventories = [inventory(path) for path in build_roots]

    root_inventory = inventory(HERE)
    paths = sorted(set(inventories[0]) | set(inventories[1]))
    comparison = []
    for relative in paths:
        first_hash = inventories[0].get(relative, "")
        second_hash = inventories[1].get(relative, "")
        root_hash = root_inventory.get(relative, "")
        passed = bool(first_hash) and first_hash == second_hash == root_hash
        comparison.append(
            {
                "path": relative,
                "build_a_sha256": first_hash,
                "build_b_sha256": second_hash,
                "sealed_root_sha256": root_hash,
                "status": "PASS" if passed else "FAIL",
            }
        )
    write_csv(HERE / "two_clean_build_comparison.csv", comparison)

    after = {path.relative_to(ROOT).as_posix(): tree_hash(path) for path in protected}
    no_write = [
        {
            "path": path,
            "before_sha256": before[path],
            "after_sha256": after[path],
            "status": "PASS" if before[path] == after[path] else "FAIL",
        }
        for path in before
    ]
    write_csv(HERE / "no_write_audit.csv", no_write)

    builds_pass = all(
        row["build_returncode"] == 0
        and row["validation_returncode"] == 0
        and row["validation_status"] == "PASS"
        for row in run_results
    )
    comparison_pass = all(row["status"] == "PASS" for row in comparison)
    no_write_pass = all(row["status"] == "PASS" for row in no_write)
    payload = {
        "status": (
            "PASS"
            if builds_pass and comparison_pass and no_write_pass
            else "FAIL"
        ),
        "clean_builds": run_results,
        "builder_output_file_count": len(comparison),
        "byte_identical_builds_and_root": comparison_pass,
        "protected_targets": len(no_write),
        "protected_no_write": no_write_pass,
        "boundary": "clean build proof does not create substantive or publication authority",
    }
    (HERE / "deterministic_build_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
