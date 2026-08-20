#!/usr/bin/env python3
"""Assemble the sealed AI-financial-system publication into a durable release.

The durable release deliberately separates reader-facing documents, standalone
chapter details, the research framework, evidence, and compact governance
records. It copies rather than references staging artifacts, regenerates the
three editable DOCX editions from the exact sealed HTML sources, validates those
DOCX files, and writes a complete hash inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RENDER_DOCX = ROOT / "research/ai_financial_system/scripts/render_publication_docx.py"
INSPECT_DOCX = ROOT / "research/ai_financial_system/scripts/inspect_publication_docx.py"
ASSEMBLY_ROOT: Path | None = None

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
        "sealed_pdf": "The_AI_Financial_System_Burning_Up_Essay.pdf",
        "pdf": "One_Build_Many_Balance_Sheets.pdf",
        "docx": "One_Build_Many_Balance_Sheets.docx",
    },
}

FRAMEWORK_FILES = (
    "research/ai_financial_system/README.md",
    "research/ai_financial_system/editorial_and_investigative_standard.md",
    "research/ai_financial_system/source_preservation_standard.md",
    "research/ai_financial_system/synthesis_deliverables_standard.md",
    "research/ai_financial_system/publication_pdf_standard.md",
    "research/ai_financial_system/chapter_docx_publication_standard.md",
    "research/ai_financial_system/monograph_plan.md",
    "research/ai_financial_system/schema.sql",
    "research/ai_financial_system/schema/canonical_core.sql",
    "research/ai_financial_system/schema/convergence_extensions.sql",
    "research/ai_financial_system/model_specifications/chapter_12_14_research_program.md",
    "research/ai_financial_system/model_specifications/economic_model_governance.md",
    "research/ai_financial_system/scripts/render_publication_docx.py",
    "research/ai_financial_system/scripts/inspect_publication_docx.py",
    "research/ai_financial_system/scripts/render_publication_pdf.py",
    "research/ai_financial_system/scripts/audit_publication_pdf.py",
    "research/ai_financial_system/scripts/validate_publication_package.py",
    "research/ai_financial_system/scripts/build_relationship_overlap_audit.py",
)

EVIDENCE_FILES = (
    "reports/ai_financial_system/research.db",
    "reports/ai_financial_system/database_manifest.json",
    "reports/ai_financial_system/canonical_release_manifest.json",
    "reports/ai_financial_system/canonical_release_receipt.json",
    "reports/ai_financial_system/relationship_overlap_audit/relationship_overlap_summary.json",
    "reports/ai_financial_system/relationship_overlap_audit/multi_ledger_entity_pairs.csv",
    "reports/ai_financial_system/relationship_overlap_audit/multi_role_entities.csv",
)


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


def copy_bound(
    source: Path,
    destination: Path,
    provenance: list[dict[str, Any]],
) -> None:
    if ASSEMBLY_ROOT is None:
        raise RuntimeError("assembly root is not initialized")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
    provenance.append(
        {
            "origin": source.relative_to(ROOT).as_posix(),
            "release_path": destination.relative_to(ASSEMBLY_ROOT).as_posix(),
            "sha256": destination_hash,
            "bytes": destination.stat().st_size,
        }
    )


def render_docx(source: Path, destination: Path, inspection: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    inspection.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="afs_release_docx_") as temp:
        temporary = Path(temp)
        first = temporary / "first.docx"
        second = temporary / "second.docx"
        for target in (first, second):
            subprocess.run(
                [
                    sys.executable,
                    str(RENDER_DOCX),
                    "--source",
                    str(source),
                    "--output",
                    str(target),
                ],
                cwd=ROOT,
                check=True,
            )
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"nondeterministic DOCX render: {source}")
        shutil.copyfile(first, destination)
    subprocess.run(
        [
            sys.executable,
            str(INSPECT_DOCX),
            str(destination),
            "--source",
            str(source),
            "--output",
            str(inspection),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    result = json.loads(inspection.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"DOCX inspection failed: {destination}")
    return {
        "source_sha256": sha256(source),
        "docx_sha256": sha256(destination),
        "inspection_sha256": sha256(inspection),
        "deterministic_double_render": True,
        "inspection_status": "PASS",
    }


def regular_inventory(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"release cannot contain symlinks: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def release_tree_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = row["path"].encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def copy_framework(temporary: Path, provenance: list[dict[str, Any]]) -> None:
    for configured in FRAMEWORK_FILES:
        source = ROOT / configured
        relative = Path(configured).relative_to("research/ai_financial_system")
        copy_bound(source, temporary / "framework" / relative, provenance)


def copy_governance(
    sealed: Path,
    temporary: Path,
    provenance: list[dict[str, Any]],
) -> None:
    compact = temporary / "governance"
    for name in ("final_package_manifest.json", "package_seal.json"):
        copy_bound(sealed / name, compact / name, provenance)
    for key in PUBLICATIONS:
        source_directory = sealed / key
        destination = compact / "publication_reviews" / key
        for name in (
            "publication_manifest.json",
            "visual_inspection.csv",
            "grayscale_print_inspection.csv",
            "editorial_review.csv",
            "citation_reconciliation_review.csv",
            "primary_source_signoff.csv",
        ):
            copy_bound(source_directory / name, destination / name, provenance)

    convergence = (
        ROOT
        / "research/ai_financial_system/staging/primary_publication_governance_v5"
        / "accepted_convergence_manifest.json"
    )
    copy_bound(
        convergence,
        compact / "accepted_convergence_manifest.json",
        provenance,
    )
    governance_review = (
        ROOT
        / "research/ai_financial_system/staging/review_primary_publication_governance_v5"
        / "lane_manifest.json"
    )
    copy_bound(
        governance_review,
        compact / "governance_review_manifest.json",
        provenance,
    )
    chapter_manifests = (
        ROOT
        / "research/ai_financial_system/staging/primary_publication_governance_v1"
        / "chapter_publication_manifests"
    )
    for source in sorted(chapter_manifests.glob("CH*.json")):
        copy_bound(
            source,
            compact / "chapter_publication_manifests" / source.name,
            provenance,
        )
    cleanup_plan = (
        ROOT / "research/ai_financial_system/staging/cleanup_plan_v1"
    )
    for source in sorted(item for item in cleanup_plan.rglob("*") if item.is_file()):
        copy_bound(
            source,
            compact / "cleanup_plan" / source.relative_to(cleanup_plan),
            provenance,
        )


def main() -> None:
    global ASSEMBLY_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-package", type=Path, required=True)
    parser.add_argument("--chapter-package", type=Path, required=True)
    parser.add_argument("--chapter-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sealed = args.sealed_package.resolve()
    chapter_package = args.chapter_package.resolve()
    chapter_review = args.chapter_review.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite release: {output}")
    sealed_manifest = json.loads(
        (sealed / "final_package_manifest.json").read_text(encoding="utf-8")
    )
    seal = json.loads((sealed / "package_seal.json").read_text(encoding="utf-8"))
    if (
        sealed_manifest.get("status") != "SEALED_ALL_DELIVERABLE_GATES_ACCEPTED"
        or seal.get("status") != "PASS"
    ):
        raise ValueError("input publication package is not sealed and passing")

    chapter_manifest_path = chapter_package / "lane_manifest.json"
    chapter_manifest = json.loads(chapter_manifest_path.read_text(encoding="utf-8"))
    if (
        chapter_manifest.get("chapter_docx_count") != 19
        or chapter_manifest.get("appendix_docx_count") != 2
        or chapter_manifest.get("inspection_pass_count") != 21
    ):
        raise ValueError("chapter package is incomplete or not fully inspected")
    chapter_review_manifest_path = chapter_review / "lane_manifest.json"
    chapter_review_manifest = json.loads(
        chapter_review_manifest_path.read_text(encoding="utf-8")
    )
    chapter_review_state = " ".join(
        str(chapter_review_manifest.get(field, ""))
        for field in ("decision", "status")
    ).upper()
    if "ACCEPT" not in chapter_review_state and "APPROVE" not in chapter_review_state:
        raise ValueError("independent chapter-package review is not accepted")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.assembling-",
        dir=output.parent,
    ) as temp:
        temporary = Path(temp)
        ASSEMBLY_ROOT = temporary
        provenance: list[dict[str, Any]] = []
        docx_results: dict[str, Any] = {}

        for key, names in PUBLICATIONS.items():
            entry = sealed_manifest["publications"][key]
            pdf_source = sealed / key / names.get("sealed_pdf", names["pdf"])
            if sha256(pdf_source) != entry["pdf_sha256"]:
                raise ValueError(f"{key}: PDF does not match sealed manifest")
            copy_bound(
                pdf_source,
                temporary / "final_documents" / names["pdf"],
                provenance,
            )
            packaged_sources = sorted((sealed / key / "source").glob("*.html"))
            if len(packaged_sources) != 1:
                raise ValueError(f"{key}: expected exactly one packaged source")
            source = packaged_sources[0]
            copy_bound(
                source,
                temporary / "publication_sources" / source.name,
                provenance,
            )
            docx_results[key] = render_docx(
                source,
                temporary / "final_documents" / names["docx"],
                temporary / "acceptance" / "docx" / f"{key}.inspection.json",
            )

        chapter_files = sorted((chapter_package / "chapters").glob("*.docx"))
        appendix_files = sorted((chapter_package / "appendices").glob("*.docx"))
        if len(chapter_files) != 19 or len(appendix_files) != 2:
            raise ValueError("chapter package file count mismatch")
        for source in chapter_files:
            copy_bound(source, temporary / "chapter_details" / "chapters" / source.name, provenance)
        for source in appendix_files:
            copy_bound(source, temporary / "chapter_details" / "appendices" / source.name, provenance)
        for subdirectory in ("sources", "inspections"):
            source_root = chapter_package / subdirectory
            if not source_root.is_dir():
                raise FileNotFoundError(source_root)
            for source in sorted(item for item in source_root.rglob("*") if item.is_file()):
                relative = source.relative_to(source_root)
                copy_bound(
                    source,
                    temporary / "chapter_details" / subdirectory / relative,
                    provenance,
                )
        for name in (
            "docx_package_inventory.csv",
            "cleanup_issue_ledger.csv",
            "banned_label_scan.csv",
            "source_bindings.csv",
            "build_final_chapter_docx_release.py",
            "README.md",
            "lane_manifest.json",
        ):
            source = chapter_package / name
            if source.is_file():
                copy_bound(
                    source,
                    temporary / "chapter_details" / "manifests" / name,
                    provenance,
                )
        for source in sorted(item for item in chapter_review.rglob("*") if item.is_file()):
            copy_bound(
                source,
                temporary
                / "chapter_details"
                / "independent_review"
                / source.relative_to(chapter_review),
                provenance,
            )

        copy_framework(temporary, provenance)
        for configured in EVIDENCE_FILES:
            copy_bound(
                ROOT / configured,
                temporary / "evidence" / Path(configured).name,
                provenance,
            )
        copy_governance(sealed, temporary, provenance)

        readme = temporary / "README.md"
        readme.write_text(
            f"""# The Modern AI Financial System — {sealed_manifest["version"]}

This is the permanent, self-contained publication release dated 26 July 2026.

## Start here

- `final_documents/` contains the executive summary, full 19-chapter monograph,
  and investigative article in both PDF and editable DOCX.
- `chapter_details/chapters/` contains nineteen standalone chapter DOCX files.
- `chapter_details/appendices/` contains the two research appendices.
- `framework/` contains the methodology, evidence, editorial, rendering, and
  reproducibility standards.
- `evidence/` contains the canonical research database and release receipts.
- `governance/` contains the immutable package seal, manifests, and independent
  review decisions.

The sealed PDFs are the publication-of-record visual editions. The DOCX editions
are deterministic renders of the exact accepted publication sources. Historical
paths inside copied provenance manifests are identifiers of the original review
transaction, not runtime dependencies of this release.
""",
            encoding="utf-8",
        )

        copied_origins = {
            row["origin"]: row["release_path"] for row in provenance
        }
        write_json(
            temporary / "provenance_relocation_map.json",
            {
                "status": "PASS_ALL_RELEASE_INPUTS_COPIED_AND_HASH_VERIFIED",
                "entries": provenance,
                "origin_to_release_path": copied_origins,
            },
        )

        excluded = {
            "release_inventory.json",
            "release_receipt.json",
            "SHA256SUMS.txt",
        }
        inventory = regular_inventory(temporary, excluded=excluded)
        tree_hash = release_tree_hash(inventory)
        write_json(
            temporary / "release_inventory.json",
            {
                "status": "SEALED_SELF_CONTAINED_RELEASE",
                "version": sealed_manifest["version"],
                "evidence_cutoff": sealed_manifest["evidence_cutoff"],
                "sealed_package_manifest_sha256": sha256(
                    sealed / "final_package_manifest.json"
                ),
                "package_seal_sha256": sha256(sealed / "package_seal.json"),
                "canonical_database_sha256": sealed_manifest[
                    "canonical_database_sha256"
                ],
                "chapter_package_manifest_sha256": sha256(chapter_manifest_path),
                "chapter_review_manifest_sha256": sha256(
                    chapter_review_manifest_path
                ),
                "final_document_count": 6,
                "chapter_docx_count": 19,
                "appendix_docx_count": 2,
                "docx_results": docx_results,
                "file_count": len(inventory),
                "tree_sha256": tree_hash,
                "files": inventory,
            },
        )
        inventory_with_manifest = regular_inventory(
            temporary,
            excluded={"release_receipt.json", "SHA256SUMS.txt"},
        )
        sums = temporary / "SHA256SUMS.txt"
        sums.write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory_with_manifest),
            encoding="utf-8",
        )
        write_json(
            temporary / "release_receipt.json",
            {
                "status": "PASS_READY_FOR_DURABLE_PROMOTION",
                "release_inventory_sha256": sha256(
                    temporary / "release_inventory.json"
                ),
                "sha256sums_sha256": sha256(sums),
                "content_tree_sha256": tree_hash,
                "final_document_count": 6,
                "chapter_docx_count": 19,
                "appendix_docx_count": 2,
                "docx_inspection_pass_count": 3,
                "external_staging_runtime_dependencies": 0,
                "note": (
                    "Copied provenance manifests retain historical staging path "
                    "strings; all reader artifacts and required release evidence "
                    "are local and hash-bound."
                ),
            },
        )

        final_inventory = regular_inventory(temporary)
        if any(row["bytes"] == 0 for row in final_inventory):
            raise ValueError("release contains an empty file")
        os.rename(temporary, output)

    print(
        json.dumps(
            {
                "status": "PASS_RELEASE_ASSEMBLED",
                "output": output.relative_to(ROOT).as_posix(),
                "release_receipt_sha256": sha256(output / "release_receipt.json"),
                "files": len(regular_inventory(output)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
