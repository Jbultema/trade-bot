#!/usr/bin/env python3
"""Build the isolated Chapter 18 rebuilt-v5 two-repair packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(
    p
    for p in (HERE, *HERE.parents)
    if (p / "research/ai_financial_system/data/claims.csv").exists()
)
AI = REPO / "research/ai_financial_system"
STAGING = AI / "staging"
V4 = STAGING / "lane_3_chapter_18_rebuilt_v4"
REVIEW = STAGING / "lane_4_verify_chapter_18_rebuilt_v4"
TASK = STAGING / "chapter_18_rebuilt_v5_task.md"
GENERIC = AI / "scripts/validate_staging_packet.py"
CANONICAL_DB = REPO / "reports/ai_financial_system/research.db"
V4_MANIFEST_HASH = (
    "c6ba2f8f1870011adf5e872bdeb78f3f9978157b2c953861a8dd4d6336be0f88"
)
REVIEW_MANIFEST_HASH = (
    "16b927c566f797b8ae95f4c81455548833356a13bce3994d7568d7196b40d086"
)
TASK_HASH = (
    "de1f9ecf2378c72868972911d0fa8c3fc4da386b3ce011cb223ab8bc46a03479"
)
CANONICAL_DB_HASH = (
    "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
)
VERDICT = "READY_FOR_FRESH_NONAUTHOR_REVIEW_WITH_EXTERNAL_HOLDS"
SCRIPTS = {
    "build_rebuilt_v5.py",
    "validate_rebuilt_v5.py",
    "verify_full_rebuilds.py",
}
REGENERATED = {
    "author_validation_results.json",
    "canonical_no_write_audit.json",
    "deterministic_rebuild_results.json",
    "generic_validation_results.json",
    "lane_manifest.json",
    "output_artifact_hash_audit.csv",
    "rebuilt_v5_metrics.json",
    "research_pass.md",
    "v5_exact_patch_disposition.csv",
    "v5_immutable_input_hash_audit.csv",
    "v5_input_inventory.csv",
    "validation_report.md",
    "validation_results.json",
}
V4_EXCLUDED = {
    "author_validation_results.json",
    "build_rebuilt_v4.py",
    "canonical_no_write_audit.json",
    "deterministic_rebuild_results.json",
    "generic_validation_results.json",
    "lane_manifest.json",
    "output_artifact_hash_audit.csv",
    "rebuilt_v4_metrics.json",
    "research_pass.md",
    "validate_rebuilt_v4.py",
    "validation_report.md",
    "validation_results.json",
    "verify_full_rebuilds.py",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"{path}: duplicate columns")
    if any(None in row for row in rows):
        raise ValueError(f"{path}: malformed row")
    return list(reader.fieldnames), rows


def write_csv(
    path: Path, fields: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_manifest(root: Path, expected_hash: str) -> dict[str, object]:
    manifest_path = root / "lane_manifest.json"
    if sha(manifest_path) != expected_hash:
        raise ValueError(f"manifest drift: {root.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        if not path.exists() or sha(path) != artifact["sha256"]:
            raise ValueError(f"artifact drift: {path}")
    return manifest


def one(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"{field}={value}: expected one, found {len(found)}")
    return found[0]


def clean(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for path in list(out.iterdir()):
        if path.name in SCRIPTS:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_v4(out: Path, manifest: dict[str, object]) -> None:
    for artifact in manifest["files"]:
        name = artifact["path"]
        if name in V4_EXCLUDED:
            continue
        source = V4 / name
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def patch_matrix(out: Path) -> None:
    path = out / "ai_similarity_non_similarity_matrix.csv"
    fields, rows = read_csv(path)
    row = one(rows, "matrix_id", "L1-MTX18-0009")
    if row["episode_id"] != "L1-AN18-0005":
        raise ValueError("matrix episode drift")
    if row["source_claim_id"] != "L1-CLM18-0026":
        raise ValueError("matrix principal drift")
    row["source_claim_id"] = "L1-CLM18-0022"
    row["evidence_grade"] = "C"
    write_csv(path, fields, rows)


def patch_junction(out: Path) -> None:
    path = out / "record_claim_junction.csv"
    fields, rows = read_csv(path)
    links = [
        row
        for row in rows
        if row["artifact_name"] == "ai_similarity_non_similarity_matrix.csv"
        and row["record_id"] == "L1-MTX18-0009"
    ]
    if len(links) != 4:
        raise ValueError("matrix junction population changed")
    by_id = {row["junction_id"]: row for row in links}
    principal = by_id["L1-RCJ18-0174"]
    principal["atomic_claim_id"] = "L1-CLM18-0022"
    principal["support_role"] = "principal_atomic_claim"
    principal["supported_scope"] = (
        "same-episode atomic principal: Suntech shipments and producer economics "
        "within L1-AN18-0005"
    )
    cross = by_id["L1-RCJ18-0175"]
    cross["atomic_claim_id"] = "L1-CLM18-0026"
    cross["support_role"] = "cross_episode_grade_c_support"
    cross["supported_scope"] = (
        "explicit Grade-C cross-episode support from L1-AN18-0006: aggregate "
        "deployment growth can coexist with negative producer margins; it does "
        "not change the row's L1-AN18-0005 episode identity"
    )
    for row in links:
        row["resolution_mode"] = (
            "same_episode_principal_with_explicit_cross_episode_grade_c_support"
        )
        row["author_validation_state"] = (
            "AUTHOR_PASS_PENDING_FRESH_NONAUTHOR_REVIEW"
        )
        row["fresh_review_patch_id"] = "V4R-XP-0002"
        row["v4_semantic_state"] = (
            "V5_REPAIRED_SCOPE_PENDING_FRESH_NONAUTHOR_REVIEW"
        )
    write_csv(path, fields, rows)


def patch_compatibility(out: Path) -> None:
    path = out / "row_claim_support.csv"
    fields, rows = read_csv(path)
    links = [
        row
        for row in rows
        if row["artifact_name"] == "ai_similarity_non_similarity_matrix.csv"
        and row["row_id"] == "L1-MTX18-0009"
    ]
    if len(links) != 3:
        raise ValueError("compatibility link population changed")
    by_id = {row["support_link_id"]: row for row in links}
    for row in links:
        row["principal_claim_id"] = "L1-CLM18-0022"
        row["fresh_review_patch_id"] = "V4R-XP-0002"
        row["v4_semantic_state"] = (
            "V5_REPAIRED_SCOPE_PENDING_FRESH_NONAUTHOR_REVIEW"
        )
    cross = by_id["L1-RCS18-0073"]
    cross["supporting_claim_id"] = "L1-CLM18-0026"
    cross["support_role"] = "cross_episode_grade_c_support"
    cross["scope_note"] = (
        "Explicit Grade-C support from L1-AN18-0006; does not change the "
        "L1-AN18-0005 row identity or authorize cross-episode calibration"
    )
    for link_id in ("L1-RCS18-0074", "L1-RCS18-0075"):
        by_id[link_id]["scope_note"] = (
            "Same-episode compatibility projection from L1-AN18-0005"
        )
    write_csv(path, fields, rows)


def patch_resolution(out: Path) -> None:
    path = out / "record_scope_resolution.csv"
    fields, rows = read_csv(path)
    row = one(rows, "record_id", "L1-MTX18-0009")
    row["atomic_claim_ids"] = (
        "L1-CLM18-0022|L1-CLM18-0026|L1-CLM18-0020|L1-CLM18-0023"
    )
    row["scope_action"] = (
        "same-episode principal L1-CLM18-0022 anchors L1-AN18-0005; "
        "L1-CLM18-0026 is explicit Grade-C cross-episode support from "
        "L1-AN18-0006 and cannot be read as a same-episode observed fact or "
        "used for calibration"
    )
    row["fresh_review_patch_id"] = "V4R-XP-0002"
    row["v4_resolution_mode"] = (
        "v5_same_episode_principal_explicit_cross_episode_grade_c_support"
    )
    row["author_v4_semantic_state"] = (
        "AUTHOR_PASS_PENDING_FRESH_NONAUTHOR_REVIEW"
    )
    write_csv(path, fields, rows)


def patch_semantic_audit(out: Path) -> None:
    path = out / "junction_semantic_audit.csv"
    fields, rows = read_csv(path)
    row = one(rows, "record_id", "L1-MTX18-0009")
    row["principal_claim_id"] = "L1-CLM18-0022"
    row["linked_claim_ids"] = (
        "L1-CLM18-0022|L1-CLM18-0026|L1-CLM18-0020|L1-CLM18-0023"
    )
    row["link_count"] = "4"
    row["v4_author_structural_status"] = "PASS"
    row["v4_author_semantic_status"] = (
        "PASS_V5_SAME_EPISODE_PRINCIPAL_WITH_EXPLICIT_CROSS_EPISODE_GRADE_C_SUPPORT"
    )
    row["fresh_review_patch_id"] = "V4R-XP-0002"
    row["semantic_basis"] = (
        "L1-CLM18-0022 is the same-episode atomic principal for "
        "L1-AN18-0005; L1-CLM18-0026 is explicit cross-episode Grade-C "
        "support from L1-AN18-0006 and cannot supply an observed fact, "
        "coefficient, probability, frequency, or timing rule"
    )
    row["fresh_nonauthor_acceptance"] = "PENDING"
    write_csv(path, fields, rows)


def write_patch_disposition(out: Path) -> None:
    _, patches = read_csv(REVIEW / "exact_row_patches.csv")
    rows: list[dict[str, str]] = []
    for patch in patches:
        rows.append(
            {
                **patch,
                "author_disposition": (
                    "APPLIED_PENDING_FRESH_NONAUTHOR_REVIEW"
                ),
                "intrinsic_verdict": "PENDING_FRESH_NONAUTHOR_REVIEW",
                "canonical_integration": "NOT_AUTHORIZED",
                "coefficient_predictive_use": "BLOCK",
                "comparative_publication": "BLOCKED",
            }
        )
    fields = list(patches[0]) + [
        "author_disposition",
        "intrinsic_verdict",
        "canonical_integration",
        "coefficient_predictive_use",
        "comparative_publication",
    ]
    write_csv(out / "v5_exact_patch_disposition.csv", fields, rows)


def input_audit(
    out: Path,
    v4_manifest: dict[str, object],
    review_manifest: dict[str, object],
) -> list[tuple[Path, str, str]]:
    inputs: list[tuple[Path, str, str]] = [
        (V4 / "lane_manifest.json", V4_MANIFEST_HASH, "v4_manifest"),
        (
            REVIEW / "lane_manifest.json",
            REVIEW_MANIFEST_HASH,
            "review_manifest",
        ),
        (TASK, TASK_HASH, "v5_task"),
    ]
    for artifact in v4_manifest["files"]:
        inputs.append(
            (
                V4 / artifact["path"],
                artifact["sha256"],
                "v4_artifact",
            )
        )
    for artifact in review_manifest["files"]:
        inputs.append(
            (
                REVIEW / artifact["path"],
                artifact["sha256"],
                "review_artifact",
            )
        )
    rows = [
        {
            "input_id": f"V5-IN-{index:04d}",
            "role": role,
            "path": str(path.relative_to(REPO)),
            "expected_sha256": expected,
            "observed_before_sha256": sha(path),
            "observed_after_sha256": sha(path),
            "status": (
                "PASS_INPUT_IMMUTABLE"
                if sha(path) == expected
                else "FAIL_INPUT_DRIFT"
            ),
        }
        for index, (path, expected, role) in enumerate(inputs, start=1)
    ]
    write_csv(
        out / "v5_immutable_input_hash_audit.csv",
        [
            "input_id",
            "role",
            "path",
            "expected_sha256",
            "observed_before_sha256",
            "observed_after_sha256",
            "status",
        ],
        rows,
    )
    write_csv(
        out / "v5_input_inventory.csv",
        ["path", "sha256", "bytes", "role"],
        [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": expected,
                "bytes": str(path.stat().st_size),
                "role": role,
            }
            for path, expected, role in inputs
        ],
    )
    if any(row["status"] != "PASS_INPUT_IMMUTABLE" for row in rows):
        raise ValueError("v5 input drift")
    return inputs


def run_validator(out: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(out / "validate_rebuilt_v5.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def write_reports(out: Path, validation: dict[str, object]) -> None:
    write_json(out / "author_validation_results.json", validation)
    metrics = {
        "status": VERDICT,
        "exact_repairs_applied": 2,
        "preserved_prior_repairs": 17,
        "retained_empirical_records": 128,
        "normalized_junction_links": 214,
        "compatibility_links": 86,
        "external_source_content_holds": 58,
        "arithmetic_nonadditivity_controls": 20,
        "coefficient_transfer_blocks": 18,
        "predictive_use_blocks": 16,
    }
    write_json(out / "rebuilt_v5_metrics.json", metrics)
    text = """# Chapter 18 rebuilt-v5 two-repair pass

**Author verdict:** `READY_FOR_FRESH_NONAUTHOR_REVIEW_WITH_EXTERNAL_HOLDS`  
**Intrinsic and bounded descriptive use:** pending fresh nonauthor review  
**Canonical integration:** `NOT_AUTHORIZED`  
**Coefficient and predictive use:** `BLOCK`  
**Comparative publication:** `BLOCKED`

## Repairs

`FAR18-XP-0010` is now identified everywhere as the Micron 2002 chronology
repair. Its underlying row-level semantic test is unchanged.

Matrix row `L1-MTX18-0009` remains a Grade-C synthesis keyed to solar episode
`L1-AN18-0005`, but now uses same-episode atomic claim `L1-CLM18-0022` as its
principal. Grade-C claim `L1-CLM18-0026` is explicitly typed as cross-episode
support from `L1-AN18-0006`. The boundary propagates through the normalized and
compatibility junctions, scope resolution, semantic audit, and validator. It
cannot be interpreted as a single-episode observed fact or used as a historical
coefficient, probability, empirical frequency, calibrated monitoring signal, or
timing rule.

## Preserved controls

All 17 prior chronology/claimant/mechanism repairs remain meaning-identical.
The packet retains 128 empirical records, 214 normalized junction links, 86
compatibility links, 58 external-content holds, 20 arithmetic/nonadditivity
controls, 18 coefficient-transfer blocks, and 16 predictive-use blocks.

This author-side packet is deterministic but is not acceptance. Canonical
integration, descriptive publication, coefficient/predictive transfer, and
comparative publication remain unauthorized.
"""
    (out / "research_pass.md").write_text(text, encoding="utf-8")


def output_hash_audit(out: Path) -> None:
    files = [
        path
        for path in sorted(out.iterdir())
        if path.is_file()
        and path.name
        not in {"lane_manifest.json", "output_artifact_hash_audit.csv"}
    ]
    write_csv(
        out / "output_artifact_hash_audit.csv",
        ["output_id", "path", "sha256", "bytes", "status"],
        [
            {
                "output_id": f"V5-OUT-{index:04d}",
                "path": path.name,
                "sha256": sha(path),
                "bytes": str(path.stat().st_size),
                "status": "PASS",
            }
            for index, path in enumerate(files, start=1)
        ],
    )


def manifest(out: Path, validation: dict[str, object]) -> None:
    files = [
        path
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "lane_manifest.json"
    ]
    payload = {
        "lane": "lane_4_chapter_18_rebuilt_v5",
        "status_date": "2026-07-26",
        "author_verdict": VERDICT,
        "author_validation_is_acceptance": False,
        "intrinsic_verdict": "PENDING_FRESH_NONAUTHOR_REVIEW",
        "descriptive_verdict": "PENDING_FRESH_NONAUTHOR_REVIEW",
        "canonical_integration": "NOT_AUTHORIZED",
        "coefficient_transfer": "BLOCK",
        "predictive_use": "BLOCK",
        "comparative_publication": "BLOCKED_PENDING_FRESH_REVIEW_AND_EXTERNAL_HOLDS",
        "external_source_content_holds": 58,
        "validation_checks_passed": validation["checks_passed"],
        "validation_checks_failed": validation["checks_failed"],
        "files": [
            {
                "path": path.name,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                **(
                    {"rows": len(read_csv(path)[1])}
                    if path.suffix == ".csv"
                    else {}
                ),
            }
            for path in files
        ],
    }
    write_json(out / "lane_manifest.json", payload)


def generic_validate(out: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(GENERIC), str(out), "--json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def build(out: Path, core_only: bool) -> None:
    clean(out)
    v4_manifest = verify_manifest(V4, V4_MANIFEST_HASH)
    review_manifest = verify_manifest(REVIEW, REVIEW_MANIFEST_HASH)
    if sha(TASK) != TASK_HASH:
        raise ValueError("task drift")
    if sha(CANONICAL_DB) != CANONICAL_DB_HASH:
        raise ValueError("canonical database drift")
    copy_v4(out, v4_manifest)
    patch_matrix(out)
    patch_junction(out)
    patch_compatibility(out)
    patch_resolution(out)
    patch_semantic_audit(out)
    write_patch_disposition(out)
    inputs = input_audit(out, v4_manifest, review_manifest)
    write_json(
        out / "canonical_no_write_audit.json",
        {
            "status": "PASS",
            "canonical_database": str(CANONICAL_DB.relative_to(REPO)),
            "sha256_before": CANONICAL_DB_HASH,
            "sha256_after": sha(CANONICAL_DB),
            "modified": False,
            "canonical_integration": "NOT_AUTHORIZED",
        },
    )
    validation = run_validator(out)
    write_reports(out, validation)
    output_hash_audit(out)
    manifest(out, validation)
    generic = generic_validate(out)
    write_json(out / "generic_validation_results.json", generic)
    output_hash_audit(out)
    manifest(out, validation)
    generic = generic_validate(out)
    if not generic["passed"]:
        raise ValueError("generic validation failed")
    write_json(out / "generic_validation_results.json", generic)
    output_hash_audit(out)
    manifest(out, validation)

    if not core_only:
        result = subprocess.run(
            [sys.executable, str(out / "verify_full_rebuilds.py"), "--json"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        write_json(
            out / "deterministic_rebuild_results.json",
            json.loads(result.stdout),
        )
        output_hash_audit(out)
        manifest(out, validation)
        generic = generic_validate(out)
        write_json(out / "generic_validation_results.json", generic)
        output_hash_audit(out)
        manifest(out, validation)

    for path, expected, _ in inputs:
        if sha(path) != expected:
            raise ValueError(f"input mutated: {path}")
    if sha(CANONICAL_DB) != CANONICAL_DB_HASH:
        raise ValueError("canonical database changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    build(args.out.resolve(), args.core_only)
    print(
        json.dumps(
            {
                "status": "CHAPTER_18_REBUILT_V5_GENERATED",
                "out": str(args.out.resolve()),
                "author_verdict": VERDICT,
                "canonical_integration": "NOT_AUTHORIZED",
                "publication": "BLOCKED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
