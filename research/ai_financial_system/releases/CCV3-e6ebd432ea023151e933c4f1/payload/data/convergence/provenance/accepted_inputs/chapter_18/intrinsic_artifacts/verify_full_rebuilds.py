#!/usr/bin/env python3
"""Compare two clean Chapter 18 rebuilt-v5 core packets byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
SCRIPTS = (
    "build_rebuilt_v5.py",
    "validate_rebuilt_v5.py",
    "verify_full_rebuilds.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build(root: Path) -> tuple[dict[str, str], str]:
    target = root / "lane_4_chapter_18_rebuilt_v5"
    target.mkdir()
    for name in SCRIPTS:
        shutil.copyfile(HERE / name, target / name)
    result = subprocess.run(
        [
            sys.executable,
            str(target / "build_rebuilt_v5.py"),
            "--core-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    manifest_path = target / "lane_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    for relpath, expected in hashes.items():
        if sha(target / relpath) != expected:
            raise ValueError(f"manifest mismatch: {relpath}")
    return hashes, sha(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with (
        tempfile.TemporaryDirectory(
            prefix=".chapter18v5_a_", dir=STAGING
        ) as left_name,
        tempfile.TemporaryDirectory(
            prefix=".chapter18v5_b_", dir=STAGING
        ) as right_name,
    ):
        left, left_manifest = build(Path(left_name))
        right, right_manifest = build(Path(right_name))
    result = {
        "status": (
            "PASS"
            if left == right and left_manifest == right_manifest
            else "FAIL"
        ),
        "clean_core_builds_compared": 2,
        "artifact_paths_and_hashes_match": left == right,
        "manifest_hashes_match": left_manifest == right_manifest,
        "manifested_artifacts_compared": len(left),
        "shared_manifest_sha256": (
            left_manifest if left_manifest == right_manifest else ""
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "comparative_publication": "BLOCKED",
    }
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
