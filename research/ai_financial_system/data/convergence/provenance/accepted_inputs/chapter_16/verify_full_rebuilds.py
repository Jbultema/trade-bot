#!/usr/bin/env python3
"""Compare two clean full Chapter 16 rebuilt-v3 packets byte-for-byte."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_2_chapter_16_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(HERE)
STAGING = REPO / "research/ai_financial_system/staging"
SCRIPTS = (
    "build_rebuilt_v3.py",
    "validate_rebuilt_v3.py",
    "verify_full_rebuilds.py",
)
LOGICAL_LANE = "lane_3_chapter_16_rebuilt_v3"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(target: Path) -> tuple[dict[str, object], str]:
    for name in SCRIPTS:
        shutil.copyfile(HERE / name, target / name)
    subprocess.run(
        [sys.executable, str(target / "build_rebuilt_v3.py")],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    manifest_path = target / "lane_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = target / artifact["path"]
        if not path.exists() or digest(path) != artifact["sha256"]:
            raise ValueError(f"manifest mismatch in clean build: {path}")
    return manifest, digest(manifest_path)


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix=".ch16v3_full_a_", dir=str(STAGING)
    ) as left_name:
        with tempfile.TemporaryDirectory(
            prefix=".ch16v3_full_b_", dir=str(STAGING)
        ) as right_name:
            left_root = Path(left_name) / LOGICAL_LANE
            right_root = Path(right_name) / LOGICAL_LANE
            left_root.mkdir()
            right_root.mkdir()
            left, left_manifest_hash = build(left_root)
            right, right_manifest_hash = build(right_root)
            left_hashes = {
                row["path"]: row["sha256"] for row in left["files"]
            }
            right_hashes = {
                row["path"]: row["sha256"] for row in right["files"]
            }
            paths_match = set(left_hashes) == set(right_hashes)
            artifact_hashes_match = left_hashes == right_hashes
            manifest_hashes_match = left_manifest_hash == right_manifest_hash
            status = (
                "PASS"
                if paths_match
                and artifact_hashes_match
                and manifest_hashes_match
                else "FAIL"
            )
            result = {
                "status": status,
                "clean_full_builds_compared": 2,
                "manifested_artifacts_compared": len(left_hashes),
                "manifest_included_in_comparison": True,
                "manifest_paths_match": paths_match,
                "artifact_hashes_match": artifact_hashes_match,
                "manifest_hashes_match": manifest_hashes_match,
                "shared_manifest_sha256": (
                    left_manifest_hash if manifest_hashes_match else ""
                ),
                "logical_lane": left["lane"],
                "comparative_publication": left["publication_boundary"][
                    "comparative_ordering"
                ],
                "canonical_write": left["publication_boundary"][
                    "canonical_database_write"
                ],
            }
            print(json.dumps(result, sort_keys=True))
            return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
