#!/usr/bin/env python3
"""Compare two empty-directory Chapters 6-9 v4 builds with the sealed lane."""

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
BUILDER = HERE / "build_rebuilt_v4.py"
VALIDATOR = HERE / "validate_rebuilt_v4.py"
EXCLUDED = {
    "build_rebuilt_v4.py",
    "validate_rebuilt_v4.py",
    "verify_clean_builds.py",
    "lane_manifest.json",
    "generic_validation_results.json",
    "ruff_results.json",
    "deterministic_rebuild_results.json",
    "two_clean_build_comparison.csv",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name not in EXCLUDED
    }


def main() -> int:
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    with (
        tempfile.TemporaryDirectory(prefix="ch0609-v4-a-") as first_dir,
        tempfile.TemporaryDirectory(prefix="ch0609-v4-b-") as second_dir,
    ):
        roots = [Path(first_dir) / "packet", Path(second_dir) / "packet"]
        for root in roots:
            subprocess.run(
                [sys.executable, str(BUILDER), "--output", str(root)],
                cwd=REPO,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(VALIDATOR), "--target", str(root)],
                cwd=REPO,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        sealed, first, second = (
            inventory(HERE),
            inventory(roots[0]),
            inventory(roots[1]),
        )
    names = sorted(set(sealed) | set(first) | set(second))
    comparison = [
        {
            "path": name,
            "sealed_sha256": sealed.get(name, ""),
            "build_a_sha256": first.get(name, ""),
            "build_b_sha256": second.get(name, ""),
            "result": (
                "PASS"
                if sealed.get(name)
                and len({sealed[name], first.get(name), second.get(name)}) == 1
                else "FAIL"
            ),
        }
        for name in names
    ]
    with (HERE / "two_clean_build_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    failures = [row for row in comparison if row["result"] == "FAIL"]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "clean_builds": 2,
        "files_compared": len(comparison),
        "byte_identical_files": len(comparison) - len(failures),
        "mismatches": failures,
        "thread_limits": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    (HERE / "deterministic_rebuild_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
