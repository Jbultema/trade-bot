from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_ROOT = REPO_ROOT / "reports/ai_financial_system/publication"

PUBLICATIONS = {
    "executive_summary": "AI_Financial_System_Executive_Summary.pdf",
    "full_monograph": "The_Modern_AI_Financial_System_Forensic_Monograph.pdf",
    "burning_up_essay": "The_AI_Financial_System_Burning_Up_Essay.pdf",
}

REQUIRED_PUBLICATION_MANIFEST_FIELDS = {
    "document_id",
    "title",
    "version",
    "evidence_cutoff",
    "canonical_database_sha256",
    "source_tree_sha256",
    "renderer",
    "renderer_version",
    "fonts",
    "pdf_file",
    "pdf_sha256",
    "page_count",
    "build_timestamp",
    "acceptance_review",
    "visual_inspection_file",
    "visual_contact_sheet_file",
    "pdf_audit_file",
    "structural_pdf_audit_file",
    "deterministic_twin_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(sha256(path))
        digest.update(file_hash)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_visual_inspection(
    path: Path, expected_pages: int, errors: list[str]
) -> None:
    if not path.is_file():
        errors.append(f"missing visual inspection: {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"page_number", "review_status", "reviewer", "reviewed_at", "notes"}
    if not rows:
        errors.append(f"{path}: visual inspection has no page rows")
        return
    missing = required - set(rows[0])
    if missing:
        errors.append(f"{path}: missing columns {sorted(missing)}")
        return
    page_numbers: list[int] = []
    for row in rows:
        try:
            page_numbers.append(int(row["page_number"]))
        except (TypeError, ValueError):
            errors.append(f"{path}: invalid page_number {row.get('page_number')!r}")
        if row["review_status"] != "accepted":
            errors.append(
                f"{path}: page {row.get('page_number')} is not accepted "
                f"({row['review_status']!r})"
            )
        if not nonempty_text(row["reviewer"]) or not nonempty_text(row["reviewed_at"]):
            errors.append(
                f"{path}: page {row.get('page_number')} lacks reviewer or timestamp"
            )
    expected = list(range(1, expected_pages + 1))
    if sorted(page_numbers) != expected:
        errors.append(
            f"{path}: page coverage {sorted(page_numbers)} does not equal {expected}"
        )


def validate_pdf_audit(
    path: Path,
    expected_pages: int,
    publication_key: str,
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"missing PDF audit: {path}")
        return
    audit = load_json(path)
    if audit.get("page_count") != expected_pages:
        errors.append(
            f"{path}: audit page_count {audit.get('page_count')} != {expected_pages}"
        )
    if audit.get("is_encrypted") or audit.get("is_locked"):
        errors.append(f"{path}: publication must be readable without encryption")
    if audit.get("blank_candidate_pages"):
        errors.append(
            f"{path}: unresolved blank candidates {audit['blank_candidate_pages']}"
        )
    if int(audit.get("extracted_document_characters") or 0) < expected_pages * 80:
        errors.append(f"{path}: insufficient searchable text for page count")
    pages = audit.get("pages")
    if not isinstance(pages, list) or len(pages) != expected_pages:
        errors.append(f"{path}: page audit inventory is incomplete")
    if (
        publication_key == "full_monograph"
        and int(audit.get("outline_item_count") or 0) < 19
    ):
        errors.append(f"{path}: full monograph lacks 19 chapter outline entries")


def validate_structural_pdf_audit(
    path: Path,
    expected_pdf: Path,
    expected_pages: int,
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"missing structural PDF audit: {path}")
        return
    audit = load_json(path)
    if audit.get("status") != "pass":
        errors.append(f"{path}: structural audit did not pass")
    if audit.get("sha256") != sha256(expected_pdf):
        errors.append(f"{path}: structural audit PDF hash mismatch")
    if audit.get("page_count") != expected_pages:
        errors.append(
            f"{path}: structural page_count {audit.get('page_count')} != {expected_pages}"
        )
    if audit.get("blank_pages"):
        errors.append(f"{path}: structural audit reports blank pages")
    if audit.get("all_fonts_embedded") is not True:
        errors.append(f"{path}: not all PDF fonts are embedded")
    comparison = audit.get("deterministic_comparison")
    if not isinstance(comparison, dict) or comparison.get("byte_identical") is not True:
        errors.append(f"{path}: deterministic twin comparison did not pass")
    if audit.get("errors"):
        errors.append(f"{path}: structural audit retains errors {audit['errors']}")


def validate_publication(
    root: Path,
    publication_key: str,
    expected_pdf_name: str,
    canonical_hash: str,
    errors: list[str],
) -> dict[str, Any] | None:
    directory = root / publication_key
    manifest_path = directory / "publication_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing publication manifest: {manifest_path}")
        return None
    manifest = load_json(manifest_path)
    missing_fields = REQUIRED_PUBLICATION_MANIFEST_FIELDS - set(manifest)
    if missing_fields:
        errors.append(f"{manifest_path}: missing fields {sorted(missing_fields)}")
    for field in REQUIRED_PUBLICATION_MANIFEST_FIELDS - {"fonts", "page_count"}:
        if field in manifest and not nonempty_text(manifest[field]):
            errors.append(f"{manifest_path}: blank required field {field}")
    if not isinstance(manifest.get("fonts"), list) or not manifest.get("fonts"):
        errors.append(f"{manifest_path}: fonts must be a nonempty list")
    try:
        page_count = int(manifest.get("page_count"))
    except (TypeError, ValueError):
        page_count = 0
        errors.append(f"{manifest_path}: invalid page_count")
    if page_count <= 0:
        errors.append(f"{manifest_path}: page_count must be positive")

    if manifest.get("pdf_file") != expected_pdf_name:
        errors.append(
            f"{manifest_path}: pdf_file {manifest.get('pdf_file')!r} "
            f"!= {expected_pdf_name!r}"
        )
    pdf_path = directory / expected_pdf_name
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        errors.append(f"missing or empty PDF: {pdf_path}")
    elif manifest.get("pdf_sha256") != sha256(pdf_path):
        errors.append(f"{manifest_path}: PDF hash mismatch")
    if manifest.get("deterministic_twin_sha256") != manifest.get("pdf_sha256"):
        errors.append(f"{manifest_path}: deterministic twin hash does not match PDF")
    if manifest.get("canonical_database_sha256") != canonical_hash:
        errors.append(f"{manifest_path}: canonical database hash mismatch")

    source_directory = directory / "source"
    if not source_directory.is_dir() or not any(source_directory.iterdir()):
        errors.append(f"missing publication source tree: {source_directory}")
    elif manifest.get("source_tree_sha256") != tree_sha256(source_directory):
        errors.append(f"{manifest_path}: publication source tree hash mismatch")

    visual_path = directory / str(manifest.get("visual_inspection_file", ""))
    contact_sheet_path = directory / str(
        manifest.get("visual_contact_sheet_file", "")
    )
    audit_path = directory / str(manifest.get("pdf_audit_file", ""))
    structural_audit_path = directory / str(
        manifest.get("structural_pdf_audit_file", "")
    )
    validate_visual_inspection(visual_path, page_count, errors)
    if not contact_sheet_path.is_file() or contact_sheet_path.stat().st_size == 0:
        errors.append(f"missing or empty visual contact sheet: {contact_sheet_path}")
    validate_pdf_audit(audit_path, page_count, publication_key, errors)
    if pdf_path.is_file():
        validate_structural_pdf_audit(
            structural_audit_path,
            pdf_path,
            page_count,
            errors,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.publication_root.resolve()
    errors: list[str] = []

    final_manifest_path = root / "final_package_manifest.json"
    if not final_manifest_path.is_file():
        errors.append(f"missing final package manifest: {final_manifest_path}")
        final_manifest: dict[str, Any] = {}
    else:
        final_manifest = load_json(final_manifest_path)

    canonical_hash = str(final_manifest.get("canonical_database_sha256", ""))
    if len(canonical_hash) != 64:
        errors.append("final package manifest lacks a valid canonical database hash")

    publication_manifests: dict[str, dict[str, Any]] = {}
    for key, pdf_name in PUBLICATIONS.items():
        manifest = validate_publication(
            root, key, pdf_name, canonical_hash, errors
        )
        if manifest is not None:
            publication_manifests[key] = manifest

    declared = final_manifest.get("publications")
    if not isinstance(declared, dict) or set(declared) != set(PUBLICATIONS):
        errors.append(
            "final package manifest must declare exactly the three publications"
        )
    else:
        for key, manifest in publication_manifests.items():
            declaration = declared.get(key)
            if not isinstance(declaration, dict):
                errors.append(f"final package declaration missing object for {key}")
                continue
            if declaration.get("pdf_sha256") != manifest.get("pdf_sha256"):
                errors.append(f"final package hash does not match {key} manifest")
            manifest_path = root / key / "publication_manifest.json"
            if declaration.get("manifest_sha256") != sha256(manifest_path):
                errors.append(f"final package manifest hash mismatch for {key}")

    evidence_cutoffs = {
        manifest.get("evidence_cutoff") for manifest in publication_manifests.values()
    }
    database_hashes = {
        manifest.get("canonical_database_sha256")
        for manifest in publication_manifests.values()
    }
    if len(evidence_cutoffs) > 1:
        errors.append(f"mixed evidence cutoffs across publications: {evidence_cutoffs}")
    if len(database_hashes) > 1:
        errors.append(f"mixed canonical database hashes: {database_hashes}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "publication_root": str(root),
        "publications_expected": len(PUBLICATIONS),
        "publications_validated": len(publication_manifests),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
