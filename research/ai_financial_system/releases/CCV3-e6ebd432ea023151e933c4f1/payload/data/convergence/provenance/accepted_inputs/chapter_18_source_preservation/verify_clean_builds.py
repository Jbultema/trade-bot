#!/usr/bin/env python3
"""Run two clean corrected-preservation builds and compare all core bytes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXCLUDED = {
    "lane_manifest.json",
    "validation_results.json",
    "generic_validation_results.json",
    "ruff_results.json",
    "two_clean_build_results.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED
    }


def main() -> int:
    env = dict(os.environ)
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="ch18-corrected-preserve-a-"
    ) as left, tempfile.TemporaryDirectory(
        prefix="ch18-corrected-preserve-b-"
    ) as right:
        outputs = [Path(left) / "packet", Path(right) / "packet"]
        for output in outputs:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "build_corrected_preservation.py"),
                    "--output",
                    str(output),
                ],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        source = inventory(HERE)
        first = inventory(outputs[0])
        second = inventory(outputs[1])
    paths = sorted(set(source) | set(first) | set(second))
    mismatches = [
        path
        for path in paths
        if len({source.get(path), first.get(path), second.get(path)}) != 1
    ]
    result = {
        "status": "PASS" if not mismatches else "FAIL",
        "clean_builds": 2,
        "files_compared": len(paths),
        "byte_identical_files": len(paths) - len(mismatches),
        "mismatches": mismatches,
        "thread_limits": {
            key: env[key]
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    (HERE / "two_clean_build_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
