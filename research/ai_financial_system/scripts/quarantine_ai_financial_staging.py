#!/usr/bin/env python3
"""Plan or atomically quarantine superseded AI-financial-system staging lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STAGING = ROOT / "research/ai_financial_system/staging"
RELEASE_TOKEN = "AI-FINANCIAL-SYSTEM-RELEASE-1.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def tree_facts(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"staging candidate cannot be a symlink: {path}")
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    digest = hashlib.sha256()
    bytes_total = 0
    for item in files:
        if item.is_symlink():
            raise ValueError(f"candidate tree cannot contain symlinks: {item}")
        relative = item.relative_to(path.parent).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = sha256(item)
        digest.update(bytes.fromhex(file_hash))
        bytes_total += item.stat().st_size
    return {
        "path": path.name,
        "type": "directory" if path.is_dir() else "file",
        "file_count": len(files),
        "bytes": bytes_total,
        "tree_sha256": digest.hexdigest(),
    }


def validate_release(release: Path) -> dict[str, Any]:
    if release.is_symlink() or not release.is_dir():
        raise ValueError("release must be a real directory")
    receipt = json.loads((release / "release_receipt.json").read_text(encoding="utf-8"))
    inventory_path = release / "release_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS_READY_FOR_DURABLE_PROMOTION"
        or inventory.get("status") != "SEALED_SELF_CONTAINED_RELEASE"
        or receipt.get("release_inventory_sha256") != sha256(inventory_path)
        or receipt.get("external_staging_runtime_dependencies") != 0
        or inventory.get("chapter_docx_count") != 19
        or inventory.get("appendix_docx_count") != 2
    ):
        raise ValueError("durable release receipt or inventory does not pass")
    for row in inventory.get("files", []):
        target = release / row["path"]
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != row["bytes"]
            or sha256(target) != row["sha256"]
        ):
            raise ValueError(f"release inventory drift: {row['path']}")
    required = [
        release / "final_documents/AI_Financial_System_Executive_Summary.pdf",
        release
        / "final_documents/The_Modern_AI_Financial_System_Forensic_Monograph.pdf",
        release
        / "final_documents/One_Build_Many_Balance_Sheets.pdf",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("reader-facing final PDF set is incomplete")
    return {
        "release_inventory_sha256": sha256(inventory_path),
        "release_receipt_sha256": sha256(release / "release_receipt.json"),
        "release_tree_sha256": inventory["tree_sha256"],
    }


def enumerate_candidates() -> list[dict[str, Any]]:
    if STAGING.is_symlink() or not STAGING.is_dir():
        raise ValueError("staging root is missing or unsafe")
    return [tree_facts(path) for path in sorted(STAGING.iterdir())]


def plan(release: Path, output: Path) -> None:
    release_facts = validate_release(release)
    candidates = enumerate_candidates()
    manifest = {
        "status": "DRY_RUN_REVIEW_REQUIRED",
        "release_token": RELEASE_TOKEN,
        "release_path": release.relative_to(ROOT).as_posix(),
        **release_facts,
        "staging_root": STAGING.relative_to(ROOT).as_posix(),
        "candidate_count": len(candidates),
        "candidate_file_count": sum(row["file_count"] for row in candidates),
        "candidate_bytes": sum(row["bytes"] for row in candidates),
        "candidates": candidates,
        "execution_contract": (
            "Every direct staging child must retain the exact recorded tree hash. "
            "Execution uses same-filesystem atomic renames and restores every "
            "completed move if any later move or validation fails."
        ),
    }
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": output.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256(output),
                "candidate_count": manifest["candidate_count"],
                "candidate_bytes": manifest["candidate_bytes"],
            },
            sort_keys=True,
        )
    )


def execute(
    release: Path,
    candidate_manifest: Path,
    manifest_sha256: str,
    quarantine: Path,
    approval_token: str,
    receipt: Path,
) -> None:
    if approval_token != RELEASE_TOKEN:
        raise ValueError("approval token does not match the sealed release")
    if sha256(candidate_manifest) != manifest_sha256:
        raise ValueError("candidate manifest hash mismatch")
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "DRY_RUN_REVIEW_REQUIRED":
        raise ValueError("candidate manifest is not an executable dry run")
    release_facts = validate_release(release)
    for key, value in release_facts.items():
        if manifest.get(key) != value:
            raise ValueError(f"release drift since dry run: {key}")
    current = enumerate_candidates()
    if current != manifest.get("candidates"):
        raise ValueError("staging candidates changed since dry run")

    destination_root = quarantine / RELEASE_TOKEN / "staging"
    if quarantine.is_symlink():
        raise ValueError("quarantine root cannot be a symlink")
    quarantine.mkdir(parents=True, exist_ok=True)
    if os.stat(STAGING).st_dev != os.stat(quarantine).st_dev:
        raise ValueError("quarantine must be on the same filesystem")
    if destination_root.exists():
        raise FileExistsError(destination_root)
    destination_root.mkdir(parents=True)

    moved: list[tuple[Path, Path]] = []
    try:
        for row in current:
            source = STAGING / row["path"]
            destination = destination_root / row["path"]
            os.rename(source, destination)
            moved.append((source, destination))
        if any(STAGING.iterdir()):
            raise RuntimeError("staging root is not empty after quarantine")
        for row in current:
            if tree_facts(destination_root / row["path"]) != row:
                raise RuntimeError(f"quarantine verification failed: {row['path']}")
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                os.rename(destination, source)
        raise

    result = {
        "status": "PASS_QUARANTINED_REVERSIBLY",
        "release_token": RELEASE_TOKEN,
        "candidate_manifest": candidate_manifest.relative_to(ROOT).as_posix(),
        "candidate_manifest_sha256": manifest_sha256,
        "release_path": release.relative_to(ROOT).as_posix(),
        **release_facts,
        "original_root": STAGING.relative_to(ROOT).as_posix(),
        "quarantine_root": destination_root.relative_to(ROOT).as_posix(),
        "moved_count": len(moved),
        "moved_file_count": manifest["candidate_file_count"],
        "moved_bytes": manifest["candidate_bytes"],
        "recovery": (
            "Restore by atomically renaming each direct child from quarantine_root "
            "back into original_root after verifying the recorded tree hash."
        ),
        "purged": False,
    }
    write_json(receipt, result)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--quarantine-dir", type=Path)
    parser.add_argument("--approval-token")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    release = args.release.resolve()
    if not args.execute:
        if args.output_manifest is None:
            parser.error("--output-manifest is required in dry-run mode")
        plan(release, args.output_manifest.resolve())
        return
    required = (
        args.candidate_manifest,
        args.manifest_sha256,
        args.quarantine_dir,
        args.approval_token,
        args.receipt,
    )
    if any(value is None for value in required):
        parser.error(
            "--execute requires --candidate-manifest, --manifest-sha256, "
            "--quarantine-dir, --approval-token, and --receipt"
        )
    execute(
        release,
        args.candidate_manifest.resolve(),
        args.manifest_sha256,
        args.quarantine_dir.resolve(),
        args.approval_token,
        args.receipt.resolve(),
    )


if __name__ == "__main__":
    main()
