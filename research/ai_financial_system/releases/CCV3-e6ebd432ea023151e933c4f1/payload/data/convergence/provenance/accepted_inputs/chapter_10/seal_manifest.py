#!/usr/bin/env python3
"""Seal the narrow Chapter 10 rebuilt-v3 author manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GENERIC = ROOT / "research/ai_financial_system/scripts/validate_staging_packet.py"
MANIFEST = HERE / "lane_manifest.json"
GENERIC_ARTIFACT = HERE / "generic_validation_results.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(*, include_generic: bool) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in HERE.rglob("*") if item.is_file()):
        if path == MANIFEST:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path == GENERIC_ARTIFACT and not include_generic:
            continue
        relative = path.relative_to(HERE).as_posix()
        row: dict[str, Any] = {
            "path": relative,
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                row["rows"] = len(list(csv.DictReader(handle)))
        elif path.suffix.lower() in {
            ".html",
            ".json",
            ".md",
            ".py",
            ".sh",
            ".txt",
        }:
            row["lines"] = len(
                path.read_text(encoding="utf-8").splitlines()
            )
        files.append(row)
    return files


def write_manifest(
    files: list[dict[str, Any]],
    generic_final: dict[str, Any] | None,
) -> None:
    validation = json.loads(
        (HERE / "validation_results.json").read_text(encoding="utf-8")
    )
    ruff = json.loads((HERE / "ruff_results.json").read_text(encoding="utf-8"))
    deterministic = json.loads(
        (HERE / "deterministic_build_results.json").read_text(encoding="utf-8")
    )
    counts = json.loads((HERE / "build_counts.json").read_text(encoding="utf-8"))
    payload = {
        "lane": HERE.name,
        "chapter": "CH10",
        "packet_type": "narrow_locator_corrected_rebuilt_v3_author_packet",
        "author_verdict": (
            "READY_FOR_FRESH_UNINVOLVED_ACCEPTANCE_WITH_EXPLICIT_"
            "VERSION_LOCATOR_HOLDS"
        ),
        "prior_author_manifest_sha256": (
            "ab0108513e9ceb9d9fa6e532133405cb4f8b2fcad96b212b44209c28347207f0"
        ),
        "required_correction_review_manifest_sha256": (
            "d5d9bc7ae7cf28da974699adff6ed7297b627b868b8440f16af4cae02951c890"
        ),
        "narrow_patch": "CH10-V2-P001",
        "canonical_database_sha256": (
            "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
        ),
        "canonical_parent_claims": counts["canonical_parent_claims"],
        "canonical_evidence_links": counts["canonical_evidence_links"],
        "canonical_source_identities": counts["canonical_source_identities"],
        "lossless_typed_rows": counts["legacy_source_inventory_rows"],
        "lossless_typed_scalars": counts["legacy_source_inventory_scalars"],
        "source_version_attempts": counts["normalized_version_attempts"],
        "selected_preserved_versions": counts["selected_preserved_versions"],
        "selected_version_holds": counts["selected_version_holds"],
        "semantic_verdicts": counts["semantic_verdicts"],
        "atomic_children": counts["atomic_children"],
        "timeline_events": counts["timeline_events"],
        "patch_status": validation["patch_status"],
        "dedicated_validation": validation["status"],
        "ruff_validation": ruff["status"],
        "deterministic_clean_build_validation": deterministic["status"],
        "generic_validation_final": generic_final,
        "authority": {
            "fresh_uninvolved_acceptance": False,
            "substantive_acceptance": False,
            "canonical_integration": False,
            "chapter_replacement": False,
            "docx_replacement": False,
            "publication": False,
        },
        "retained_holds": [
            (
                "SRC-0145 Commerce cited report and lawful official alternate "
                "returned access or validity failures"
            ),
            "SRC-0151 PIF cited page and official alternate remained access-blocked",
            (
                "SRC-0152 Mubadala cited page and official alternate remained "
                "access-blocked"
            ),
            "all author semantic verdicts remain pending fresh uninvolved acceptance",
        ],
        "self_hash_policy": (
            "lane_manifest.json is excluded from its own file inventory; "
            "the delivered manifest SHA-256 is computed externally"
        ),
        "files": files,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_generic() -> tuple[int, dict[str, Any], str]:
    process = subprocess.run(
        [
            sys.executable,
            str(GENERIC),
            str(HERE),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = (
        json.loads(process.stdout)
        if process.stdout.strip()
        else {"passed": False, "errors": ["generic validator emitted no JSON"]}
    )
    return process.returncode, payload, process.stderr.strip()


def main() -> int:
    write_manifest(file_inventory(include_generic=False), generic_final=None)
    first_code, first_payload, first_stderr = run_generic()
    first_artifact = {
        "status": (
            "PASS"
            if first_code == 0 and first_payload.get("passed")
            else "FAIL"
        ),
        "phase": "pre_final_manifest_inventory_generic_validation",
        "command": (
            "python research/ai_financial_system/scripts/"
            "validate_staging_packet.py "
            "research/ai_financial_system/staging/"
            "lane_2_chapter_10_rebuilt_v3 --json"
        ),
        "returncode": first_code,
        "result": first_payload,
        "stderr": first_stderr,
    }
    GENERIC_ARTIFACT.write_text(
        json.dumps(first_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_manifest(file_inventory(include_generic=True), generic_final=None)
    final_code, final_payload, final_stderr = run_generic()
    final_record = {
        "status": (
            "PASS"
            if final_code == 0 and final_payload.get("passed")
            else "FAIL"
        ),
        "returncode": final_code,
        "result": final_payload,
        "stderr": final_stderr,
        "phase": "final_file_inventory",
    }
    write_manifest(
        file_inventory(include_generic=True),
        generic_final=final_record,
    )
    verify_code, verify_payload, verify_stderr = run_generic()
    passed = (
        first_artifact["status"] == "PASS"
        and final_record["status"] == "PASS"
        and verify_code == 0
        and bool(verify_payload.get("passed"))
        and not verify_stderr
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "manifest_sha256": sha(MANIFEST),
        "file_count": len(file_inventory(include_generic=True)),
        "final_generic_reverification": verify_payload,
        "boundary": (
            "sealed author packet only; no substantive, canonical, DOCX, or "
            "publication authority"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
