#!/usr/bin/env python3
"""Verify two sequential byte-identical Chapter 18 metadata-rebase builds."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_metadata_rebase.py"
VALIDATOR = HERE / "validate_metadata_rebase.py"
LANE_NAME = "lane_1_chapter_18_metadata_rebase"
EXPECTED_POST_SEAL_VARIANCE = {"validation_results.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_map(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_hash(files: dict[str, tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for relpath, (file_hash, byte_count) in sorted(files.items()):
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(byte_count).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_build(destination: Path) -> dict[str, tuple[str, int]]:
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(destination)],
        check=True,
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--target",
            str(destination),
            "--write",
        ],
        check=True,
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return file_map(destination)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("clean-build comparison cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with (
        tempfile.TemporaryDirectory(
            prefix="ch18_metadata_rebase_build_a_"
        ) as temp_a,
        tempfile.TemporaryDirectory(
            prefix="ch18_metadata_rebase_build_b_"
        ) as temp_b,
    ):
        build_a = Path(temp_a) / LANE_NAME
        build_b = Path(temp_b) / LANE_NAME
        files_a = run_build(build_a)
        files_b = run_build(build_b)

    author_files = file_map(HERE)
    generated_paths = sorted(set(files_a) | set(files_b))
    comparison: list[dict[str, object]] = []
    for relpath in generated_paths:
        hash_a, bytes_a = files_a.get(relpath, ("MISSING", -1))
        hash_b, bytes_b = files_b.get(relpath, ("MISSING", -1))
        author_hash, author_bytes = author_files.get(relpath, ("MISSING", -1))
        clean_build_match = hash_a == hash_b and bytes_a == bytes_b
        author_match = hash_a == author_hash and bytes_a == author_bytes
        expected_finalization_variance = (
            relpath in EXPECTED_POST_SEAL_VARIANCE
            and clean_build_match
            and author_hash != "MISSING"
        )
        comparison.append(
            {
                "path": relpath,
                "build_a_sha256": hash_a,
                "build_b_sha256": hash_b,
                "author_lane_sha256": author_hash,
                "build_a_bytes": bytes_a,
                "build_b_bytes": bytes_b,
                "author_lane_bytes": author_bytes,
                "clean_build_status": "PASS" if clean_build_match else "FAIL",
                "author_match_status": (
                    "PASS"
                    if author_match
                    else (
                        "EXPECTED_POST_SEAL_VARIANCE"
                        if expected_finalization_variance
                        else "FAIL"
                    )
                ),
                "status": (
                    "PASS"
                    if clean_build_match
                    and (author_match or expected_finalization_variance)
                    else "FAIL"
                ),
            }
        )

    same_paths = set(files_a) == set(files_b)
    byte_identical = files_a == files_b
    exact_author_match = all(
        row["author_match_status"] == "PASS" for row in comparison
    )
    author_core_match = all(row["status"] == "PASS" for row in comparison)
    result = {
        "lane": LANE_NAME,
        "status_date": "2026-07-26",
        "passed": same_paths and byte_identical and author_core_match,
        "build_mode": "sequential_single_process_per_build",
        "cpu_policy": "no_parallel_jobs",
        "files_compared": len(generated_paths),
        "build_a_tree_sha256": tree_hash(files_a),
        "build_b_tree_sha256": tree_hash(files_b),
        "path_sets_identical": same_paths,
        "byte_identical": byte_identical,
        "author_lane_matches_clean_build": exact_author_match,
        "author_lane_core_matches_clean_build": author_core_match,
        "expected_post_seal_variance": sorted(EXPECTED_POST_SEAL_VARIANCE),
        "failed_paths": [
            row["path"] for row in comparison if row["status"] != "PASS"
        ],
    }
    write_csv(HERE / "two_clean_build_comparison.csv", comparison)
    write_json(HERE / "two_clean_build_results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
