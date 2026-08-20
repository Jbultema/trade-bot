#!/usr/bin/env python3
"""Run and compare two clean deterministic metadata-rebase builds."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILDER = HERE / "build_metadata_rebase.py"
VALIDATOR = HERE / "validate_metadata_rebase.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def aggregate_hash(items: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for path, metadata in sorted(items.items()):
        digest.update(path.encode("utf-8"))
        digest.update(bytes.fromhex(str(metadata["sha256"])))
    return digest.hexdigest()


def main() -> int:
    with (
        tempfile.TemporaryDirectory(prefix="ch1115-metadata-a-") as first_tmp,
        tempfile.TemporaryDirectory(prefix="ch1115-metadata-b-") as second_tmp,
    ):
        build_roots = [
            Path(first_tmp) / "packet",
            Path(second_tmp) / "packet",
        ]
        run_results: list[dict[str, object]] = []
        for build_number, build_root in enumerate(build_roots, start=1):
            build = subprocess.run(
                [sys.executable, str(BUILDER), "--output", str(build_root)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            validate = subprocess.run(
                [sys.executable, str(VALIDATOR), "--target", str(build_root)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            run_results.append(
                {
                    "build_number": build_number,
                    "builder_returncode": build.returncode,
                    "validator_returncode": validate.returncode,
                    "builder_stderr_empty": not build.stderr.strip(),
                    "validator_status": (
                        "PASS" if validate.returncode == 0 else "FAIL"
                    ),
                }
            )
        first_inventory = inventory(build_roots[0])
        second_inventory = inventory(build_roots[1])

    all_paths = sorted(set(first_inventory) | set(second_inventory))
    comparison = []
    for path in all_paths:
        first = first_inventory.get(path, {})
        second = second_inventory.get(path, {})
        status = (
            "PASS"
            if first
            and second
            and first["sha256"] == second["sha256"]
            and first["bytes"] == second["bytes"]
            else "FAIL"
        )
        comparison.append(
            {
                "path": path,
                "build_a_sha256": first.get("sha256", ""),
                "build_b_sha256": second.get("sha256", ""),
                "build_a_bytes": first.get("bytes", ""),
                "build_b_bytes": second.get("bytes", ""),
                "status": status,
            }
        )

    with (HERE / "two_clean_build_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "build_a_sha256",
                "build_b_sha256",
                "build_a_bytes",
                "build_b_bytes",
                "status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison)

    run_status_ok = all(
        row["builder_returncode"] == 0 and row["validator_returncode"] == 0
        for row in run_results
    )
    comparison_ok = bool(comparison) and all(
        row["status"] == "PASS" for row in comparison
    )
    first_aggregate = aggregate_hash(first_inventory)
    second_aggregate = aggregate_hash(second_inventory)
    result = {
        "status": "PASS" if run_status_ok and comparison_ok else "FAIL",
        "builds_run_sequentially": True,
        "build_a_aggregate_sha256": first_aggregate,
        "build_b_aggregate_sha256": second_aggregate,
        "byte_identical": first_aggregate == second_aggregate,
        "files_compared": len(comparison),
        "run_results": run_results,
    }
    (HERE / "two_clean_build_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{result['status']}: files={len(comparison)} "
        f"aggregate={first_aggregate}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
