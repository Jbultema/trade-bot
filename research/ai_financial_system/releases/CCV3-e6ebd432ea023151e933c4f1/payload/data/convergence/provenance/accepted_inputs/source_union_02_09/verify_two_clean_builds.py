#!/usr/bin/env python3
"""Require two empty-directory union-v2 builds to be byte-identical."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_union_v2.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder: Path) -> dict[str, str]:
    return {
        path.name: digest(path)
        for path in sorted(folder.iterdir())
        if path.is_file()
    }


def build(folder: Path) -> None:
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(folder)],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="source_union_v2_a_") as a_name, (
        tempfile.TemporaryDirectory(prefix="source_union_v2_b_")
    ) as b_name:
        a = Path(a_name)
        b = Path(b_name)
        build(a)
        build(b)
        a_hashes = files(a)
        b_hashes = files(b)
        names = sorted(set(a_hashes) | set(b_hashes))
        rows = []
        passed = True
        for name in names:
            retained_hash = digest(HERE / name) if (HERE / name).is_file() else ""
            ok = (
                name in a_hashes
                and a_hashes[name] == b_hashes.get(name) == retained_hash
            )
            passed &= ok
            rows.append(
                {
                    "path": name,
                    "build_a_sha256": a_hashes.get(name, ""),
                    "build_b_sha256": b_hashes.get(name, ""),
                    "retained_sha256": retained_hash,
                    "result": "PASS_IDENTICAL" if ok else "FAIL",
                }
            )
    with (HERE / "two_clean_build_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "build_a_sha256",
                "build_b_sha256",
                "retained_sha256",
                "result",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "passed": passed,
        "result": (
            "two_empty_builds_and_retained_authored_outputs_byte_identical"
            if passed
            else "clean_build_mismatch"
        ),
        "file_count": len(rows),
        "scope": "all deterministic outputs authored by build_union_v2.py",
    }
    (HERE / "two_clean_build_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
