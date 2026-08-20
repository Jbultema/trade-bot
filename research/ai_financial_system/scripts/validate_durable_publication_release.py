#!/usr/bin/env python3
"""Fail-closed validation for the durable AI-financial publication release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PUBLICATIONS = {
    "executive_summary": {
        "pdf": "AI_Financial_System_Executive_Summary.pdf",
        "docx": "AI_Financial_System_Executive_Summary.docx",
    },
    "full_monograph": {
        "pdf": "The_Modern_AI_Financial_System_Forensic_Monograph.pdf",
        "docx": "The_Modern_AI_Financial_System_Forensic_Monograph.docx",
    },
    "burning_up_essay": {
        "pdf": "One_Build_Many_Balance_Sheets.pdf",
        "docx": "One_Build_Many_Balance_Sheets.docx",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def validate(release: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not release.is_dir() or release.is_symlink():
        raise ValueError("release is missing or is a symlink")
    inventory_path = release / "release_inventory.json"
    receipt_path = release / "release_receipt.json"
    sums_path = release / "SHA256SUMS.txt"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "SEALED_SELF_CONTAINED_RELEASE":
        fail("release inventory status is not sealed", errors)
    if receipt.get("status") != "PASS_READY_FOR_DURABLE_PROMOTION":
        fail("release receipt status is not passing", errors)
    if receipt.get("release_inventory_sha256") != sha256(inventory_path):
        fail("release inventory hash does not match receipt", errors)
    if receipt.get("sha256sums_sha256") != sha256(sums_path):
        fail("SHA256SUMS hash does not match receipt", errors)

    declared: dict[str, dict[str, Any]] = {}
    for row in inventory.get("files", []):
        relative = row.get("path")
        if not isinstance(relative, str) or relative in declared:
            fail(f"invalid or duplicate inventory path: {relative}", errors)
            continue
        declared[relative] = row
        path = release / relative
        if path.is_symlink() or not path.is_file():
            fail(f"inventory artifact missing or unsafe: {relative}", errors)
            continue
        if path.stat().st_size != row.get("bytes") or sha256(path) != row.get(
            "sha256"
        ):
            fail(f"inventory artifact drift: {relative}", errors)

    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file()
    }
    governed_extras = {
        "release_inventory.json",
        "release_receipt.json",
        "SHA256SUMS.txt",
    }
    if actual != set(declared) | governed_extras:
        fail(
            "whole-tree closure mismatch: "
            f"extra={sorted(actual - set(declared) - governed_extras)} "
            f"missing={sorted(set(declared) - actual)}",
            errors,
        )

    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in sums:
            fail(f"invalid SHA256SUMS row: {line}", errors)
            continue
        sums[relative] = digest
        path = release / relative
        if not path.is_file() or sha256(path) != digest:
            fail(f"SHA256SUMS mismatch: {relative}", errors)
    if set(sums) != set(declared) | {"release_inventory.json"}:
        fail("SHA256SUMS coverage mismatch", errors)

    sealed_manifest_path = release / "governance/final_package_manifest.json"
    sealed_manifest = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    if sealed_manifest.get("status") != "SEALED_ALL_DELIVERABLE_GATES_ACCEPTED":
        fail("copied publication manifest is not sealed", errors)
    for key, names in PUBLICATIONS.items():
        pdf = release / "final_documents" / names["pdf"]
        docx = release / "final_documents" / names["docx"]
        inspection_path = release / "acceptance/docx" / f"{key}.inspection.json"
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        if sha256(pdf) != sealed_manifest["publications"][key]["pdf_sha256"]:
            fail(f"{key}: PDF does not match sealed publication hash", errors)
        if not docx.is_file() or inspection.get("status") != "PASS":
            fail(f"{key}: DOCX or exact-source inspection does not pass", errors)
        result = inventory.get("docx_results", {}).get(key, {})
        if (
            result.get("docx_sha256") != sha256(docx)
            or result.get("inspection_sha256") != sha256(inspection_path)
            or result.get("deterministic_double_render") is not True
        ):
            fail(f"{key}: DOCX release binding mismatch", errors)

    chapters = sorted((release / "chapter_details/chapters").glob("*.docx"))
    appendices = sorted((release / "chapter_details/appendices").glob("*.docx"))
    inspections = sorted((release / "chapter_details/inspections").glob("*.json"))
    if (len(chapters), len(appendices), len(inspections)) != (19, 2, 21):
        fail(
            "standalone chapter package count mismatch: "
            f"{len(chapters)}/{len(appendices)}/{len(inspections)}",
            errors,
        )
    for inspection_path in inspections:
        if json.loads(inspection_path.read_text(encoding="utf-8")).get(
            "status"
        ) != "PASS":
            fail(f"chapter inspection is not PASS: {inspection_path.name}", errors)

    chapter_review_path = (
        release / "chapter_details/independent_review/lane_manifest.json"
    )
    chapter_review = json.loads(chapter_review_path.read_text(encoding="utf-8"))
    chapter_review_state = " ".join(
        str(chapter_review.get(field, ""))
        for field in ("decision", "status")
    ).upper()
    if "ACCEPT" not in chapter_review_state and "APPROVE" not in chapter_review_state:
        fail("independent standalone-chapter review is not accepted", errors)

    db = release / "evidence/research.db"
    if sha256(db) != sealed_manifest.get("canonical_database_sha256"):
        fail("canonical database does not match sealed package", errors)

    # Historical staging paths are permitted only in immutable governance and
    # provenance records. Reader documents, exact sources, chapter sources,
    # framework instructions, and acceptance results must be staging-independent.
    disallowed_staging_refs: list[str] = []
    for path in sorted(item for item in release.rglob("*") if item.is_file()):
        relative = path.relative_to(release).as_posix()
        if (
            relative.startswith("governance/")
            or relative.startswith("acceptance/")
            or relative.startswith("chapter_details/manifests/")
            or relative.startswith("chapter_details/independent_review/")
            or relative.startswith("chapter_details/inspections/")
            or relative.startswith("evidence/")
            or relative == "provenance_relocation_map.json"
        ):
            continue
        if path.suffix.lower() not in {
            ".md",
            ".json",
            ".csv",
            ".html",
            ".py",
            ".txt",
            ".sql",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "research/ai_financial_system/staging/" in text:
            disallowed_staging_refs.append(relative)
    if disallowed_staging_refs:
        fail(
            "runtime surfaces retain staging references: "
            + ", ".join(disallowed_staging_refs),
            errors,
        )

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "release": str(release),
        "inventory_files": len(declared),
        "actual_files": len(actual),
        "final_documents": 6,
        "chapter_docx": len(chapters),
        "appendix_docx": len(appendices),
        "chapter_inspections": len(inspections),
        "staging_runtime_dependencies": len(disallowed_staging_refs),
        "canonical_database_sha256": sha256(db),
        "sealed_manifest_sha256": sha256(sealed_manifest_path),
        "release_inventory_sha256": sha256(inventory_path),
        "release_receipt_sha256": sha256(receipt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.release.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
