#!/usr/bin/env python3
"""Prove two clean Chapters 2-5 v3 builds are byte-identical."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BUILDER = HERE / "build_rebuilt_v3.py"
EXCLUDED = {
    "build_rebuilt_v3.py",
    "validate_rebuilt_v3.py",
    "verify_two_builds.py",
    "finalize_rebuilt_v3.py",
    "dedicated_validation_results.json",
    "dedicated_validation_checks.csv",
    "two_clean_build_results.json",
    "two_clean_build_comparison.csv",
    "ruff_results.json",
    "generic_validation_results.json",
    "validation_results.json",
    "lane_manifest.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): digest(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name not in EXCLUDED
    }


def run(folder: Path) -> None:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(folder)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chapters_0205_v3_a_") as a_name, (
        tempfile.TemporaryDirectory(prefix="chapters_0205_v3_b_")
    ) as b_name:
        a, b = Path(a_name), Path(b_name)
        run(a)
        run(b)
        a_hashes, b_hashes, retained = hashes(a), hashes(b), hashes(HERE)
        rows = []
        for name in sorted(set(a_hashes) | set(b_hashes) | set(retained)):
            passed = (
                name in a_hashes
                and a_hashes[name] == b_hashes.get(name) == retained.get(name)
            )
            rows.append(
                {
                    "path": name,
                    "build_a_sha256": a_hashes.get(name, ""),
                    "build_b_sha256": b_hashes.get(name, ""),
                    "retained_sha256": retained.get(name, ""),
                    "result": "PASS_IDENTICAL" if passed else "FAIL_MISMATCH",
                }
            )
    with (HERE / "two_clean_build_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["result"] == "PASS_IDENTICAL" for row in rows)
    result = {
        "passed": passed,
        "clean_builds": 2,
        "files_compared": len(rows),
        "result": (
            "two_clean_builds_and_retained_outputs_byte_identical"
            if passed
            else "clean_build_mismatch"
        ),
        "thread_limits": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }
    (HERE / "two_clean_build_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
