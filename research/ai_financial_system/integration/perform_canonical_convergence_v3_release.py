#!/usr/bin/env python3
"""Perform the governed, reversible canonical-convergence-v3 release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research/ai_financial_system"
REPORTS = ROOT / "reports/ai_financial_system"
CANDIDATE = (
    RESEARCH
    / "staging/canonical_convergence_v3_design/canonical_release_candidate"
)
REVIEW = RESEARCH / "staging/review_canonical_convergence_v3_release"
RELEASE_ID = "CCV3-e6ebd432ea023151e933c4f1"
EXPECTED = {
    "release_manifest": (
        "b07fc85ff4b443c6e3b4f05bbc09de8098ac337f471b221193be7eeedc41b1d3"
    ),
    "database_manifest": (
        "6a3a7445dcdfa7647938a12a79db00744ab2ba21254086d61f7eb7d6e458f019"
    ),
    "database": (
        "14c5b3473cdde8dee51653b812597a507610efb7686bd30402c1acd16b75cc65"
    ),
    "review_manifest": (
        "3e5e5a0d317765f848382e6a171d396b1c1e979344a8f18f64a6e4a5d3494eb1"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def inventory_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_exclusive(path: Path, value: object) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_candidate() -> dict[str, Any]:
    release_manifest_path = CANDIDATE / "release_manifest.json"
    database_manifest_path = CANDIDATE / "reports/database_manifest.json"
    database_path = CANDIDATE / "reports/research.db"
    review_manifest_path = REVIEW / "lane_manifest.json"
    observed = {
        "release_manifest": sha256(release_manifest_path),
        "database_manifest": sha256(database_manifest_path),
        "database": sha256(database_path),
        "review_manifest": sha256(review_manifest_path),
    }
    if observed != EXPECTED:
        raise RuntimeError(
            f"frozen candidate or review drift: {observed!r} != {EXPECTED!r}"
        )
    review_manifest = json.loads(review_manifest_path.read_text())
    review_report = (REVIEW / "review_report.md").read_text(encoding="utf-8")
    if "**ACCEPT**" not in review_report:
        raise RuntimeError("independent review does not carry an ACCEPT verdict")
    release_manifest = json.loads(release_manifest_path.read_text())
    for row in release_manifest["files"]:
        path = CANDIDATE / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"candidate manifest mismatch: {row['path']}")
    if len(release_manifest["files"]) != release_manifest["file_count"]:
        raise RuntimeError("candidate file-count mismatch")
    return {
        "observed_hashes": observed,
        "release_manifest": release_manifest,
        "review_manifest_status": review_manifest.get("status"),
    }


def database_audit(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        views = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'view'
            ORDER BY name
            """
        ).fetchall()
    finally:
        connection.close()
    return {
        "integrity_check": integrity,
        "foreign_key_violations": len(foreign_keys),
        "table_count": len(tables),
        "view_count": len(views),
        "select_star_view_count": sum(
            "select *" in (sql or "").lower() for _, sql in views
        ),
    }


def main() -> int:
    verification = verify_candidate()
    release_home = RESEARCH / "releases" / RELEASE_ID
    payload = release_home / "payload"
    rollback = release_home / "rollback"
    transaction = release_home / "transaction"
    receipt_path = release_home / "release_receipt.json"
    if release_home.exists():
        raise FileExistsError(f"release target already exists: {release_home}")
    release_home.parent.mkdir(parents=True, exist_ok=True)
    release_home.mkdir()
    rollback.mkdir()
    transaction.mkdir()

    # Preserve the complete accepted payload, not merely the live projections.
    shutil.copytree(CANDIDATE, payload)
    if sha256(payload / "release_manifest.json") != EXPECTED["release_manifest"]:
        raise RuntimeError("preserved release payload drift")

    live_targets = {
        "data": RESEARCH / "data",
        "schema": RESEARCH / "schema",
        "builder": RESEARCH / "scripts/build_released_database.py",
        "database": REPORTS / "research.db",
        "database_manifest": REPORTS / "database_manifest.json",
        "release_manifest": REPORTS / "canonical_release_manifest.json",
    }
    prior: dict[str, dict[str, Any]] = {}
    for key, target in live_targets.items():
        if target.is_dir():
            rows = inventory(target)
            prior[key] = {
                "exists": True,
                "kind": "directory",
                "inventory": rows,
                "inventory_sha256": inventory_hash(rows),
            }
            shutil.copytree(target, rollback / key)
        elif target.is_file():
            prior[key] = {
                "exists": True,
                "kind": "file",
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
            shutil.copy2(target, rollback / key)
        else:
            prior[key] = {"exists": False, "kind": "absent"}

    # Prove the rollback copy before touching live state.
    for key, state in prior.items():
        if state["kind"] == "directory":
            copied = inventory(rollback / key)
            if inventory_hash(copied) != state["inventory_sha256"]:
                raise RuntimeError(f"rollback directory mismatch: {key}")
        elif state["kind"] == "file":
            if sha256(rollback / key) != state["sha256"]:
                raise RuntimeError(f"rollback file mismatch: {key}")

    stage = transaction / "stage"
    stage.mkdir()
    shutil.copytree(payload / "data", stage / "data")
    shutil.copytree(payload / "schema", stage / "schema")
    shutil.copy2(
        payload / "scripts/build_released_database.py",
        stage / "build_released_database.py",
    )
    shutil.copy2(payload / "reports/research.db", stage / "research.db")
    shutil.copy2(
        payload / "reports/database_manifest.json",
        stage / "database_manifest.json",
    )
    shutil.copy2(
        payload / "release_manifest.json",
        stage / "canonical_release_manifest.json",
    )

    installed: list[str] = []
    displaced: dict[str, Path] = {}
    try:
        for key in ("data", "schema"):
            target = live_targets[key]
            if target.exists():
                old = transaction / f"displaced_{key}"
                os.rename(target, old)
                displaced[key] = old
            os.rename(stage / key, target)
            installed.append(key)

        for key, staged_name in (
            ("builder", "build_released_database.py"),
            ("database", "research.db"),
            ("database_manifest", "database_manifest.json"),
            ("release_manifest", "canonical_release_manifest.json"),
        ):
            target = live_targets[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                old = transaction / f"displaced_{key}"
                os.replace(target, old)
                displaced[key] = old
            os.replace(stage / staged_name, target)
            installed.append(key)

        rebuild = transaction / "independent_rebuild.db"
        subprocess.run(
            [
                sys.executable,
                str(live_targets["builder"]),
                "--release-root",
                str(RESEARCH),
                "--output",
                str(rebuild),
            ],
            check=True,
            cwd=ROOT,
        )
        live_database_hash = sha256(live_targets["database"])
        rebuilt_hash = sha256(rebuild)
        if live_database_hash != EXPECTED["database"] or rebuilt_hash != (
            EXPECTED["database"]
        ):
            raise RuntimeError("post-release rebuild is not byte-identical")
        audit = database_audit(live_targets["database"])
        if audit != {
            "integrity_check": "ok",
            "foreign_key_violations": 0,
            "table_count": 81,
            "view_count": 2,
            "select_star_view_count": 0,
        }:
            raise RuntimeError(f"post-release database audit failed: {audit}")

        receipt = {
            "release_id": RELEASE_ID,
            "status": "ACCEPTED_CANONICAL_RELEASE",
            "released_at": datetime.now(UTC).isoformat(),
            "publication_authorized": False,
            "publication_boundary": (
                "canonical research integration accepted; claim and chapter "
                "publication clearance remains separately governed"
            ),
            "candidate_release_manifest": {
                "path": str(
                    (payload / "release_manifest.json").relative_to(ROOT)
                ),
                "sha256": EXPECTED["release_manifest"],
            },
            "independent_review_manifest": {
                "path": str((REVIEW / "lane_manifest.json").relative_to(ROOT)),
                "sha256": EXPECTED["review_manifest"],
            },
            "canonical_database": {
                "path": str(live_targets["database"].relative_to(ROOT)),
                "sha256": live_database_hash,
                "independent_rebuild_sha256": rebuilt_hash,
                "audit": audit,
            },
            "canonical_database_manifest_sha256": sha256(
                live_targets["database_manifest"]
            ),
            "canonical_release_manifest_sha256": sha256(
                live_targets["release_manifest"]
            ),
            "prior_state": prior,
            "rollback_package": str(rollback.relative_to(ROOT)),
            "retained_boundaries": {
                "chapter_support_projections_not_cleared": 520,
                "source_exceptions_on_hold": 6,
                "archived_artifacts_not_promoted": 894,
                "private_investment_claims_with_analytical_holds": 5,
            },
            "verification": verification,
        }
        write_json_exclusive(receipt_path, receipt)
        shutil.copy2(
            receipt_path,
            REPORTS / "canonical_release_receipt.json",
        )
    except BaseException:
        for key in reversed(installed):
            target = live_targets[key]
            failed = transaction / f"failed_{key}"
            if target.exists():
                os.rename(target, failed)
            old = displaced.get(key)
            if old is not None and old.exists():
                os.rename(old, target)
        raise

    print(
        json.dumps(
            {
                "status": "ACCEPTED_CANONICAL_RELEASE",
                "release_id": RELEASE_ID,
                "database_sha256": EXPECTED["database"],
                "receipt": str(receipt_path.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
