#!/usr/bin/env python3
"""Compare two clean core builds of the Chapter 17 atomic rebase packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


OUT = Path(__file__).resolve().parent
LOGICAL_LANE = "lane_3_chapter_17_atomic_rebase"
SCRIPTS = (
    "build_atomic_rebase.py",
    "validate_atomic_rebase.py",
    "verify_full_rebuilds.py",
)


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_3_chapter_17_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
STAGING = REPO / "research/ai_financial_system/staging"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_clean(parent: Path) -> tuple[dict[str, object], str]:
    lane = parent / LOGICAL_LANE
    lane.mkdir()
    for script in SCRIPTS:
        shutil.copyfile(OUT / script, lane / script)
    subprocess.run(
        [sys.executable, str(lane / "build_atomic_rebase.py"), "--core-only"],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest_path = lane / "lane_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = lane / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"clean build manifest mismatch: {path}")
    return manifest, sha256(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(
        prefix=".ch17_atomic_a_", dir=str(STAGING)
    ) as left_parent:
        with tempfile.TemporaryDirectory(
            prefix=".ch17_atomic_b_", dir=str(STAGING)
        ) as right_parent:
            left, left_manifest_hash = build_clean(Path(left_parent))
            right, right_manifest_hash = build_clean(Path(right_parent))
            left_hashes = {
                row["path"]: row["sha256"] for row in left["files"]
            }
            right_hashes = {
                row["path"]: row["sha256"] for row in right["files"]
            }
            passed = (
                left_hashes == right_hashes
                and left_manifest_hash == right_manifest_hash
            )
            payload = {
                "status": "PASS" if passed else "FAIL",
                "logical_lane": LOGICAL_LANE,
                "clean_builds_compared": 2,
                "manifested_artifacts_compared": len(left_hashes),
                "manifest_included_in_comparison": True,
                "shared_core_manifest_sha256": (
                    left_manifest_hash if passed else ""
                ),
                "byte_identical": passed,
                "core_build_mode": True,
                "canonical_integration": "NOT_AUTHORIZED",
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
