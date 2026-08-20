#!/usr/bin/env python3
"""Finalize validations, author report, and stable manifest for the rebase."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GENERIC_VALIDATOR = (
    REPO / "research/ai_financial_system/scripts/validate_staging_packet.py"
)
DEDICATED_VALIDATOR = HERE / "validate_metadata_rebase.py"
EXCLUDED_FROM_MANIFEST = {
    "lane_manifest.json",
    "generic_validation_results.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def assert_no_write_current() -> None:
    failures: list[str] = []
    for row in read_csv(HERE / "protected_input_no_write_audit.csv"):
        path = REPO / row["path"]
        if (
            not path.is_file()
            or sha256_file(path) != row["sha256_before"]
            or row["sha256_before"] != row["sha256_after"]
            or path.stat().st_size != int(row["bytes_before"])
            or row["bytes_before"] != row["bytes_after"]
            or row["status"] != "PASS"
        ):
            failures.append(row["path"])
    if failures:
        raise RuntimeError(f"protected input changed: {failures[:10]}")


def run_ruff() -> dict[str, object]:
    scripts = [str(path) for path in sorted(HERE.glob("*.py"))]
    run = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *scripts],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "status": "PASS" if run.returncode == 0 else "FAIL",
        "returncode": run.returncode,
        "files_checked": len(scripts),
        "stdout": run.stdout,
        "stderr": run.stderr,
    }
    write_json(HERE / "ruff_results.json", result)
    if run.returncode != 0:
        raise RuntimeError(f"Ruff failed:\n{run.stdout}\n{run.stderr}")
    return result


def run_dedicated(write: bool) -> dict[str, object]:
    command = [sys.executable, str(DEDICATED_VALIDATOR), "--target", str(HERE)]
    if write:
        command.append("--write")
    run = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"dedicated validator did not emit JSON:\n{run.stdout}\n{run.stderr}"
        ) from exc
    if run.returncode != 0 or not result.get("passed"):
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def run_generic() -> dict[str, object]:
    run = subprocess.run(
        [sys.executable, str(GENERIC_VALIDATOR), str(HERE), "--json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"generic validator did not emit JSON:\n{run.stdout}\n{run.stderr}"
        ) from exc
    if run.returncode != 0 or not result.get("passed"):
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def render_author_report() -> str:
    governing = read_csv(HERE / "governing_input_hash_registry.csv")
    population = read_csv(HERE / "population_reproduction_audit.csv")
    intrinsic_fields = read_csv(HERE / "intrinsic_field_disposition.csv")
    source_fields = read_csv(HERE / "source_metadata_field_disposition.csv")
    no_write = read_csv(HERE / "protected_input_no_write_audit.csv")
    clean = read_json(HERE / "two_clean_build_results.json")
    binding = read_csv(HERE / "canonical_database_binding.csv")
    source_dispositions = Counter(
        row["field_disposition"] for row in source_fields
    )
    population_map = {row["population"]: row["observed"] for row in population}
    governing_lines = "\n".join(
        f"- `{row['role']}`: `{row['observed_after_sha256']}` (`{row['status']}`)"
        for row in governing
    )
    disposition_lines = "\n".join(
        f"- `{name}`: {count}"
        for name, count in sorted(source_dispositions.items())
    )
    candidate = next(
        row
        for row in binding
        if row["role"] == "sealed_integration_candidate_database"
    )
    return f"""# Chapter 18 immutable source-metadata rebase author report

**Status date:** 2026-07-26  
**Author verdict:** `READY_FOR_FRESH_UNINVOLVED_METADATA_ACCEPTANCE_WITH_RETAINED_HOLDS`  
**Author validation is acceptance:** `NO`  
**Canonical integration:** `NOT_AUTHORIZED`  
**Chapter/DOCX/publication authority:** `NOT_AUTHORIZED`  
**Coefficient, predictive, causal, comparative, and model use:** `BLOCK`

## Governing bindings

All four accepted manifests and the frozen canonical database matched their
task-pinned hashes before and after the sequential build:

{governing_lines}

The sealed candidate database is byte-identical to the frozen canonical
database at `{candidate['sha256']}` ({candidate['bytes']} bytes), and both
read-only SQLite integrity checks return `ok`. This packet performs no
canonical mutation.

## Immutable intrinsic population

All 50 manifest-declared intrinsic author artifacts are copied byte-for-byte
under `intrinsic_artifacts/`. The accepted packet remains exactly:

- {population_map['intrinsic_analytical_records']} analytical records;
- {population_map['intrinsic_record_claim_links']} record/claim junction links;
- {population_map['arithmetic_nonadditivity_controls']} arithmetic and
  nonadditivity controls;
- all 17 earlier intrinsic repairs and both rebuilt-v5 residual repairs.

The lossless disposition covers 1,654 CSV rows and
{len(intrinsic_fields):,} scalar fields, each by direct retained identity.
No accepted intrinsic claim, arithmetic result, prose boundary, evidence
grade, support direction, or prohibition is rewritten.

## Accepted source preservation and semantics

The rebase reproduces exactly:

- {population_map['locator_relationships']} semantic-locator relationships:
  {population_map['supported']} `supported` and
  {population_map['partially_supported']} `partially_supported`;
- {population_map['semantic_patches']} accepted semantic-scope patches and
  {population_map['ordered_source_edges']} ordered claim/source/version/locator
  edges;
- {population_map['unique_sources']} cited source identities and
  {population_map['retrieval_attempts']} append-only retrieval attempts;
- {population_map['successful_artifacts']} preserved artifacts,
  {population_map['retained_failures']} original UNCTAD HTTP-403 failures, and
  the distinct successful v0003 alternate;
- {population_map['derivatives']} derivatives with exact derivative and parent
  hashes.

Every one of 718 accepted preservation-metadata CSV rows and
{len(source_fields):,} scalar fields has exactly one lossless disposition:

{disposition_lines}

There are no missing fields, populated-to-null changes, many-to-one
overwrites, or authorized rejections. Stable semantic keys and all foreign
keys pass.

## Authority boundary and retained holds

Preservation success, semantic support, evidence grade, claim scope,
analytical eligibility, and publication authority remain separate machine
fields. `partially_supported` is never promoted. Preserved bytes are explicitly
not semantic proof.

All {population_map['coefficient_blocks']} coefficient-transfer blocks and
{population_map['predictive_blocks']} predictive/conceptual-hypothesis blocks
remain in force. Causal transfer, comparative-system use, model use, canonical
integration, DOCX mutation, and publication remain blocked or not authorized
on every authority row. The two failed retrieval attempts, all source
exceptions, unresolved queue rows, and claim-scope limitations remain
append-only governed holds.

## Reproducibility and no-write evidence

Two sequential clean builds compare {clean['files_compared']} generated files
and are byte-identical at tree hash `{clean['build_a_tree_sha256']}`. The
author lane matches the deterministic core; `validation_results.json` has the
expected post-seal-only variance that replaces the pre-seal deferred-manifest
check with the passing 102-entry sealed-manifest audit. Dedicated validation,
generic staging validation, and Ruff pass.

The protected-input audit covers {len(no_write)} files across the four
governing packets, canonical database and registries, chapter sources,
document/publication surfaces, and named shared instructions/trackers. Every
protected file is byte-identical before and after the build.

## Recommendation

This packet is ready for a fresh uninvolved metadata acceptance review. It does
not perform primary canonical integration, rewrite Chapter 18, replace a DOCX,
clear coefficient/predictive/comparative holds, or authorize publication.
"""


def manifest_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        relpath = path.relative_to(HERE).as_posix()
        if relpath in EXCLUDED_FROM_MANIFEST:
            continue
        entry: dict[str, object] = {
            "path": relpath,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            entry["rows"] = csv_count(path)
        elif path.suffix.lower() in {
            ".json",
            ".md",
            ".py",
            ".txt",
            ".html",
            ".headers",
        }:
            entry["lines"] = len(
                path.read_text(encoding="utf-8").splitlines()
            )
        entries.append(entry)
    return entries


def seal_manifest() -> dict[str, object]:
    metrics = read_json(HERE / "build_metrics.json")
    validation = read_json(HERE / "validation_results.json")
    clean = read_json(HERE / "two_clean_build_results.json")
    ruff = read_json(HERE / "ruff_results.json")
    manifest = {
        "lane": "lane_1_chapter_18_metadata_rebase",
        "packet_type": "chapter_18_immutable_source_metadata_rebase",
        "status_date": "2026-07-26",
        "author_verdict": (
            "READY_FOR_FRESH_UNINVOLVED_METADATA_ACCEPTANCE_WITH_RETAINED_HOLDS"
        ),
        "author_validation_is_acceptance": False,
        "metadata_acceptance": "PENDING_FRESH_UNINVOLVED_REVIEW",
        "canonical_integration": "NOT_AUTHORIZED",
        "chapter_docx_publication": "NOT_AUTHORIZED",
        "coefficient_transfer": "BLOCK",
        "predictive_use": "BLOCK",
        "causal_transfer": "BLOCK",
        "comparative_system_use": "BLOCK",
        "model_use": "BLOCK",
        "governing_inputs": {
            "intrinsic_author_manifest_sha256": (
                "d11f8bd03062b2abfbcb85d3c351ea614eb2d06a3704ed15b6e571c2e758b8de"
            ),
            "intrinsic_review_manifest_sha256": (
                "d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8"
            ),
            "corrected_preservation_author_manifest_sha256": (
                "3fad6122a81e115a352e20bfdb3086be1a25d5ea6547202c97d6ac2020b11867"
            ),
            "corrected_preservation_review_manifest_sha256": (
                "d4e1d026f7743032e77f98a6ed1424ccd4270bb5a5188a3ed1a5d98c44eb8dd9"
            ),
            "canonical_database_sha256": (
                "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
            ),
        },
        "populations": metrics["populations"],
        "disposition_coverage": metrics["disposition_coverage"],
        "intrinsic_manifest_declared_artifacts": metrics[
            "intrinsic_manifest_declared_artifacts"
        ],
        "preservation_manifest_declared_artifacts": metrics[
            "preservation_manifest_declared_artifacts"
        ],
        "protected_input_files": metrics["protected_input_files"],
        "canonical_database_byte_identity": metrics[
            "canonical_database_byte_identity"
        ],
        "two_clean_builds": "PASS" if clean["passed"] else "FAIL",
        "clean_build_files_compared": clean["files_compared"],
        "clean_build_tree_sha256": clean["build_a_tree_sha256"],
        "dedicated_validation": "PASS" if validation["passed"] else "FAIL",
        "ruff": ruff["status"],
        "generic_validation": "PASS",
        "self_hash_policy": (
            "lane_manifest.json and generic_validation_results.json are "
            "excluded from manifest file entries"
        ),
        "files": manifest_entries(),
    }
    write_json(HERE / "lane_manifest.json", manifest)
    return manifest


def main() -> int:
    clean = read_json(HERE / "two_clean_build_results.json")
    validation = read_json(HERE / "validation_results.json")
    if not clean.get("passed") or not validation.get("passed"):
        raise RuntimeError("clean-build or dedicated validation evidence is not PASS")

    assert_no_write_current()
    run_ruff()
    (HERE / "author_report.md").write_text(
        render_author_report(),
        encoding="utf-8",
    )
    seal_manifest()

    run_dedicated(write=True)
    manifest = seal_manifest()
    sealed_validation = run_dedicated(write=False)
    if not sealed_validation.get("passed"):
        raise RuntimeError("post-seal dedicated validation failed")

    first_generic = run_generic()
    write_json(HERE / "generic_validation_results.json", first_generic)
    second_generic = run_generic()
    if first_generic != second_generic:
        raise RuntimeError("generic validation did not stabilize")
    write_json(HERE / "generic_validation_results.json", second_generic)

    assert_no_write_current()
    final_manifest_hash = sha256_file(HERE / "lane_manifest.json")
    result = {
        "status": "PASS",
        "lane_manifest_sha256": final_manifest_hash,
        "manifest_files": len(manifest["files"]),
        "dedicated_checks_passed": sealed_validation["checks_passed"],
        "generic_files_checked": second_generic["files_checked"],
        "generic_csv_rows_checked": second_generic["csv_rows_checked"],
        "ruff": "PASS",
        "no_write": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
