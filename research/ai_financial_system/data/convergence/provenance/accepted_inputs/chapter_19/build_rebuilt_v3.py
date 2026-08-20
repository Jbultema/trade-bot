#!/usr/bin/env python3
"""Build the isolated Chapter 19 rebuilt-v3 correction packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


OUT = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_2_chapter_19_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
STAGING = REPO / "research/ai_financial_system/staging"
SOURCE = STAGING / "lane_2_chapter_19_rebuilt_v2"
REVIEW = STAGING / "lane_1_verify_chapter_19_rebuilt_v2"
LOGICAL_LANE = "lane_3_chapter_19_rebuilt_v3"
STATUS_DATE = "2026-07-26"
CHANGED_SOURCE_CSVS = {
    "correction_map.csv",
    "evidence_system_coverage.csv",
    "finding_resolution.csv",
    "independent_review_resolution.csv",
    "review_patch_application.csv",
    "writing_gate_audit.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate columns")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"{path}: malformed-width row")
        return rows


def write_csv(
    path: Path, rows: list[dict[str, object]], fields: list[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"{path}: refusing empty CSV")
    names = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "lane_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        if not path.exists() or digest(path) != artifact["sha256"]:
            raise ValueError(f"manifest mismatch: {path}")
    return manifest


def find(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"expected one {field}={value}, found {len(found)}")
    return found[0]


def seed_source_csvs(source_manifest: dict[str, object]) -> None:
    for artifact in source_manifest["files"]:
        path = SOURCE / artifact["path"]
        if path.suffix == ".csv":
            shutil.copyfile(path, OUT / path.name)


def apply_review_patch_dispositions() -> None:
    applications = read_csv(OUT / "review_patch_application.csv")
    reperformance = {
        row["review_patch_id"]: row
        for row in read_csv(REVIEW / "review_patch_reperformance.csv")
    }
    if len(applications) != 2247 or len(reperformance) != 2247:
        raise ValueError("review patch population mismatch")
    if {row["review_patch_id"] for row in applications} != set(reperformance):
        raise ValueError("review patch IDs do not reconcile")
    fields = list(applications[0])
    for field in (
        "exact_requested_work_completed",
        "safe_unresolved_architecture_effect",
        "independent_disposition_evidence",
    ):
        if field not in fields:
            fields.append(field)
    for row in applications:
        review = reperformance[row["review_patch_id"]]
        row["application_status"] = review["independent_verdict"]
        row["exact_requested_work_completed"] = review[
            "exact_requested_work_completed"
        ]
        row["safe_unresolved_architecture_effect"] = review[
            "safe_unresolved_architecture_effect"
        ]
        row["independent_disposition_evidence"] = review[
            "independent_evidence"
        ]
        row["implementation_rule"] = (
            review["required_replacement_or_rule"]
            if review["independent_verdict"]
            != "DIRECT_OR_SUPERSEDING_CORRECTION_VERIFIED"
            else row["implementation_rule"]
        )
        row["validation_test"] = review["release_test"]
        row["canonical_integration_status"] = "HOLD"
    write_csv(OUT / "review_patch_application.csv", applications, fields)


def patch_correction_map() -> None:
    rows = read_csv(OUT / "correction_map.csv")
    original = {
        row["patch_id"]: row
        for row in read_csv(REVIEW / "original_patch_reperformance.csv")
    }
    if len(rows) != 101 or len(original) != 101:
        raise ValueError("original correction population mismatch")

    old_fields = list(rows[0])
    fields: list[str] = []
    for field in old_fields:
        if field == "implemented_artifacts":
            fields.extend(
                [
                    "corrected_packet_implemented_artifacts",
                    "rebuilt_v2_implemented_artifacts",
                    "corrected_packet_artifact_scope",
                    "rebuilt_v2_artifact_scope",
                ]
            )
        else:
            fields.append(field)
    fields.append("literal_or_exact_completion")

    status_map = {
        "PASS": "DIRECT_OR_SUPERSEDING_CORRECTION_VERIFIED",
        "PASS_WITH_EXPLICIT_HOLD": (
            "DIRECT_CORRECTION_VERIFIED_WITH_EXPLICIT_OPERATIONAL_HOLD"
        ),
        "PASS_DESCRIPTIVE_ONLY": "DESCRIPTIVE_ONLY_NOT_POLICY_VALIDATION",
        "SAFE_ALTERNATIVE_NOT_LITERAL_COMPLETION": (
            "SAFE_ALTERNATIVE_PENDING_SEMANTIC_DECOMPOSITION"
        ),
        "PARTIAL_PROVENANCE_MISMATCH": (
            "LIVE_SEMANTICS_FIXED_IN_REBUILT_V2_HISTORICAL_PROVENANCE_REPAIRED"
        ),
    }
    for row in rows:
        patch_id = row["patch_id"]
        review = original[patch_id]
        prior_artifacts = row.pop("implemented_artifacts")
        artifacts = [item for item in prior_artifacts.split("|") if item]
        row["corrected_packet_implemented_artifacts"] = prior_artifacts
        row["rebuilt_v2_implemented_artifacts"] = "|".join(
            item for item in artifacts if (SOURCE / item).exists()
        )
        row["corrected_packet_artifact_scope"] = (
            "lane_2_chapter_19_corrected_external_historical_context"
        )
        row["rebuilt_v2_artifact_scope"] = (
            "lane_2_chapter_19_rebuilt_v2_current_source_context"
        )
        row["literal_or_exact_completion"] = review[
            "literal_or_exact_completion"
        ]
        row["rebuilt_v2_resolution_status"] = status_map[
            review["independent_verdict"]
        ]
        row["rebuilt_v2_evidence"] = review["independent_evidence"]
        if patch_id == "PATCH-H19-0028":
            row["correction_status"] = (
                "not_applied_in_corrected_staging_rebuilt_v2_fixed"
            )
            row["rebuilt_v2_resolution_method"] = (
                "live_semantics_fixed_in_rebuilt_v2_and_historical_provenance_repaired"
            )
        elif "PATCH-H19-0029" <= patch_id <= "PATCH-H19-0080":
            row["rebuilt_v2_resolution_status"] = (
                "SAFE_ALTERNATIVE_PENDING_SEMANTIC_DECOMPOSITION"
            )
            row["rebuilt_v2_resolution_method"] = (
                "safe_alternative_parent_restored_child_route_retired_"
                "semantic_decomposition_pending"
            )
    write_csv(OUT / "correction_map.csv", rows, fields)


FINDING_RESOLUTIONS = {
    "H19-01": (
        "Generated blocker hypotheses are inactive; the 138 preserved references "
        "remain unmapped and owner-unaccepted, so no dependency discovery is claimed."
    ),
    "H19-02": (
        "Numeric scores and ordinal ranks are removed; materiality strata are "
        "descriptive only and do not validate a monitoring policy."
    ),
    "H19-03": (
        "Only five independently authored tests remain populated; classification "
        "of 247 missing or blocked tests is not execution of those tests."
    ),
    "H19-04": (
        "All 119 reporting-path audits remain unperformed and zero epistemic "
        "states are accepted; temporary-versus-structural findings remain blocked."
    ),
    "H19-05": (
        "Fifty-two full parent questions are restored and 114 false-atomic child "
        "proposals are inactive; parent restoration is not semantic decomposition."
    ),
    "H19-06": (
        "Document preservation and venue monitoring are separated, while external "
        "archive and content-hash coverage remains zero of 155."
    ),
    "H19-07": (
        "Token-aware supersession and row-level record-type corrections remove the "
        "known substring and residual-ambiguity classification defects."
    ),
    "H19-08": (
        "All 73 later-language flags are disabled and 17 effective-date inversions "
        "remain visible; version order remains unknown without timestamps."
    ),
    "H19-09": (
        "All ten falsifier rules remain proposed and unvalidated, with sampling "
        "and acceptance prerequisites explicit."
    ),
    "H19-10": (
        "The three missing chapter-level proposal links are structurally present, "
        "but link owner acceptance remains pending."
    ),
    "H19-11": (
        "The 393 confidence rows are current snapshots with blank priors, history "
        "IDs, and integration timestamps; no evolution is claimed."
    ),
    "H19-12": (
        "Evolution now requires an actual grade or confidence change or a linked "
        "evidence-change event; row count alone is insufficient."
    ),
    "H19-13": (
        "The internal dataset manifest is separated from external preservation; "
        "external archive coverage remains zero of 155."
    ),
    "H19-14": (
        "Hash, lineage, row-count, semantic, generic, and deterministic controls "
        "are structural validation only, not evidence acquisition."
    ),
}


def finding_status(verdict: str) -> str:
    return {
        "PASS": "DIRECT_OR_SUPERSEDING_CORRECTION_VERIFIED",
        "PASS_DESCRIPTIVE_ONLY": "DESCRIPTIVE_ONLY_NOT_POLICY_VALIDATION",
        "PASS_STRUCTURAL_ONLY": "STRUCTURAL_VALIDATION_ONLY",
        "PASS_WITH_EXPLICIT_HOLD": (
            "SAFE_UNRESOLVED_ARCHITECTURE_WITH_EXPLICIT_HOLD"
        ),
    }[verdict]


def patch_finding_resolutions() -> None:
    reperformance = {
        row["finding_id"]: row
        for row in read_csv(REVIEW / "finding_reperformance.csv")
    }
    independent = read_csv(OUT / "independent_review_resolution.csv")
    independent_fields = list(independent[0])
    if "residual_operational_gate" not in independent_fields:
        independent_fields.insert(
            independent_fields.index("canonical_integration_status"),
            "residual_operational_gate",
        )
    for row in independent:
        review = reperformance[row["finding_id"]]
        row["rebuilt_v2_status"] = finding_status(
            review["independent_reperformance_verdict"]
        )
        row["resolution"] = FINDING_RESOLUTIONS[row["finding_id"]]
        row["residual_operational_gate"] = review[
            "residual_operational_gate"
        ]
        row["canonical_integration_status"] = "HOLD"
    write_csv(
        OUT / "independent_review_resolution.csv",
        independent,
        independent_fields,
    )

    findings = read_csv(OUT / "finding_resolution.csv")
    for row in findings:
        review = reperformance[row["finding_id"]]
        row["rebuilt_v2_status"] = finding_status(
            review["independent_reperformance_verdict"]
        )
        row["rebuilt_v2_resolution"] = FINDING_RESOLUTIONS[row["finding_id"]]
        row["residual_operational_gate"] = review[
            "residual_operational_gate"
        ]
        row["canonical_integration_status"] = "HOLD"
    write_csv(OUT / "finding_resolution.csv", findings)


def patch_metrics_and_gates() -> None:
    coverage = read_csv(OUT / "evidence_system_coverage.csv")
    metric = find(coverage, "metric_id", "ECM-R2-0012")
    metric["metric"] = "review_patch_dispositions_accounted"
    metric["value"] = "2247"
    metric["denominator_or_context"] = (
        "396 direct_or_superseding_corrections;985 deferred_holds;"
        "471 retired_child_routes;394 parent_restorations;1 partial_provenance_mismatch"
    )
    metric["governance_status"] = "accounting_only_not_work_completion"
    coverage.extend(
        [
            {
                "metric_id": f"ECM-R3-{index:04d}",
                "metric": name,
                "value": str(value),
                "denominator_or_context": "2247 reconciled review dispositions",
                "governance_status": governance,
            }
            for index, (name, value, governance) in enumerate(
                [
                    (
                        "direct_or_superseding_corrections_verified",
                        396,
                        "independently_verified_completion",
                    ),
                    ("deferred_explicit_holds", 985, "unresolved_work"),
                    (
                        "safe_alternative_child_routes_retired",
                        471,
                        "safe_disposition_not_literal_completion",
                    ),
                    (
                        "safe_alternative_parents_restored",
                        394,
                        "safe_disposition_not_literal_completion",
                    ),
                    (
                        "partial_provenance_mismatches",
                        1,
                        "historical_provenance_repaired_in_v3",
                    ),
                ],
                start=16,
            )
        ]
    )
    write_csv(OUT / "evidence_system_coverage.csv", coverage)

    gates = read_csv(OUT / "writing_gate_audit.csv")
    patch_gate = find(gates, "gate_id", "WG-R2-0011")
    patch_gate["status"] = "pass_accounting_only"
    patch_gate["evidence"] = (
        "All 2,247 rows have explicit dispositions; only 396 independently "
        "verified direct or superseding corrections count as completed work. "
        "The remainder are 985 holds, 471 child retirements, 394 parent "
        "restorations, and 1 repaired provenance mismatch."
    )
    gates.extend(
        [
            {
                "gate_id": "WG-R3-0013",
                "gate": "operational_results_publication",
                "status": "hold",
                "evidence": (
                    "A missing-test classification is not execution; parent "
                    "restoration is not resolution; child retirement is not "
                    "evidence; inactive links are not accepted dependencies."
                ),
                "as_of_date": STATUS_DATE,
            },
            {
                "gate_id": "WG-R3-0014",
                "gate": "fresh_nonauthor_acceptance",
                "status": "hold",
                "evidence": (
                    "Author validation cannot accept rebuilt-v3; a fresh "
                    "nonauthor agent must independently review the sealed lane."
                ),
                "as_of_date": STATUS_DATE,
            },
        ]
    )
    write_csv(OUT / "writing_gate_audit.csv", gates)


def write_exact_patch_audit(review_manifest: dict[str, object]) -> None:
    patches = read_csv(REVIEW / "exact_patches.csv")
    if [row["review_fix_id"] for row in patches] != [
        f"V19R2-{number:03d}" for number in range(1, 11)
    ]:
        raise ValueError("governed 10-patch sequence mismatch")
    rows: list[dict[str, object]] = []
    for ordinal, patch in enumerate(patches, start=1):
        rows.append(
            {
                "application_id": f"APP-V19V3-{ordinal:03d}",
                "review_fix_id": patch["review_fix_id"],
                "review_manifest_sha256": digest(REVIEW / "lane_manifest.json"),
                "severity": patch["severity"],
                "target": patch["target"],
                "row_scope": patch["row_scope"],
                "field": patch["field"],
                "before_state": patch["current_value_or_claim"],
                "after_state": patch["required_replacement_or_rule"],
                "author_application_status": (
                    "APPLIED_PENDING_FRESH_NONAUTHOR_ACCEPTANCE"
                ),
                "verification_rule": f"V19V3-PATCH-{ordinal:03d}",
                "canonical_integration_effect": "HOLD",
            }
        )
    write_csv(OUT / "v3_exact_patch_disposition_audit.csv", rows)


def snapshot_input_hashes(
    source_manifest: dict[str, object], review_manifest: dict[str, object]
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for role, root, manifest in (
        ("source", SOURCE, source_manifest),
        ("review", REVIEW, review_manifest),
    ):
        result[(role, "lane_manifest.json")] = digest(root / "lane_manifest.json")
        for artifact in manifest["files"]:
            result[(role, artifact["path"])] = digest(root / artifact["path"])
    return result


def write_input_hash_audit(
    before: dict[tuple[str, str], str],
    source_manifest: dict[str, object],
    review_manifest: dict[str, object],
) -> None:
    rows: list[dict[str, object]] = []
    roots = {"source": SOURCE, "review": REVIEW}
    for role, artifact in sorted(before):
        after = digest(roots[role] / artifact)
        rows.append(
            {
                "input_role": role,
                "artifact": artifact,
                "before_sha256": before[(role, artifact)],
                "after_sha256": after,
                "status": (
                    "PASS_UNCHANGED"
                    if before[(role, artifact)] == after
                    else "FAIL_CHANGED"
                ),
            }
        )
    write_csv(OUT / "v3_input_hash_audit.csv", rows)


def write_preservation_audit(source_manifest: dict[str, object]) -> None:
    source_hashes = {
        row["path"]: row["sha256"] for row in source_manifest["files"]
    }
    rows: list[dict[str, object]] = []
    for name in sorted(
        row["path"]
        for row in source_manifest["files"]
        if row["path"].endswith(".csv")
    ):
        actual = digest(OUT / name)
        intended = name in CHANGED_SOURCE_CSVS
        same = actual == source_hashes[name]
        rows.append(
            {
                "artifact": name,
                "source_sha256": source_hashes[name],
                "v3_sha256": actual,
                "intentionally_changed": str(intended).lower(),
                "status": (
                    "PASS_INTENTIONAL_CHANGE"
                    if intended and not same
                    else "PASS_BYTE_PRESERVED"
                    if not intended and same
                    else "FAIL"
                ),
            }
        )
    write_csv(OUT / "preserved_artifact_hash_audit.csv", rows)


def write_reports() -> None:
    write_text(
        OUT / "research_pass.md",
        """# Chapter 19 rebuilt-v3 research pass

## Author decision

**SAFE ONLY AS UNRESOLVED ARCHITECTURE.** This isolated staging packet is not
canonical, does not authorize publication of resolved validation findings, and
requires a fresh nonauthor acceptance review.

## Patch-disposition accounting

All 2,247 review rows are accounted for, but they are not 2,247 completed
repairs:

- 396 direct or superseding corrections are independently verified;
- 985 requests remain deferred explicit holds;
- 471 unsafe child routes were retired as safe alternatives;
- 394 governed parents were restored as safe alternatives; and
- 1 historical provenance mismatch was repaired in rebuilt-v3.

A safe alternative or retirement is not literal completion of the originating
request. Classification of a missing test is not execution of that test.
Restoration of a parent question is not resolution or semantic decomposition.
Retirement of a false child is not acquisition of evidence. An inactive
generated link is not an accepted dependency route.

## Preserved unresolved architecture

The packet retains 252 governed questions, 52 active compound parents, and 114
inactive false-atomic child proposals. Only five tests were independently
authored; 247 remain blank or blocked. All 119 reporting audits remain
unperformed, all 138 preserved blocker references remain unmapped and
unaccepted, and all 750 generated links remain inactive.

There are no pseudo-scores or ordinal ranks. All 994 routes remain unnormalized,
unsearched, owner-unaccepted, and excluded from priority. The 73 chronology rows
have no unsafe later-evidence flag, with 17 date inversions visible. The 393
confidence snapshots do not claim evolution. External archival coverage remains
zero of 155, and all ten falsifiers remain proposed and unexecuted.

## Publication boundary

Operational gates must be completed before their results can enter the
monograph, executive summary, or standalone essay as findings. This author pass
does not accept the packet and does not authorize canonical integration.
""",
    )
    write_json(
        OUT / "semantic_validation_results.json",
        {
            "lane": LOGICAL_LANE,
            "chapter": "CH19",
            "status_date": STATUS_DATE,
            "author_verdict": "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE",
            "fresh_nonauthor_acceptance": "REQUIRED",
            "canonical_integration": "NOT_AUTHORIZED",
            "resolved_validation_findings_publication": "NOT_AUTHORIZED",
            "review_patch_dispositions": {
                "total_accounted": 2247,
                "direct_or_superseding_corrections_verified": 396,
                "deferred_explicit_holds": 985,
                "safe_alternative_child_routes_retired": 471,
                "safe_alternative_parents_restored": 394,
                "historical_provenance_mismatch_repaired_in_v3": 1,
            },
            "preserved_architecture": {
                "governed_questions": 252,
                "compound_parents": 52,
                "inactive_child_proposals": 114,
                "executed_or_independently_authored_tests": 5,
                "blank_or_blocked_tests": 247,
                "reporting_audits": 119,
                "accepted_reporting_states": 0,
                "preserved_blocker_references": 138,
                "mapped_or_accepted_blockers": 0,
                "inactive_generated_links": 750,
                "pseudo_scores_or_ranks": 0,
                "source_routes": 994,
                "normalized_searched_or_accepted_routes": 0,
                "chronology_rows": 73,
                "unsafe_later_evidence_flags": 0,
                "effective_date_inversions": 17,
                "current_confidence_snapshots": 393,
                "fabricated_confidence_evolution": 0,
                "external_archives": 0,
                "external_archive_population": 155,
                "proposed_falsifiers": 10,
                "executed_falsifiers": 0,
            },
            "fixed_logical_packet_identifier": True,
        },
    )
    write_text(
        OUT / "validation_report.md",
        """# Chapter 19 rebuilt-v3 author validation report

**Author verdict: SAFE ONLY AS UNRESOLVED ARCHITECTURE.**

The ten exact accounting and provenance repairs are applied. The author
validator meaning-tests those repairs and recomputes the negative-control
architecture. It does not accept the packet; fresh nonauthor review is required.

Disposition accounting is now explicit: 396 independently verified direct or
superseding corrections, 985 deferred holds, 471 child-route retirements, 394
parent-restoration alternatives, and one repaired historical provenance
mismatch. The 1,851 non-completion dispositions are not presented as completed
research work.

All evidence-production gates remain operational holds. No missing test was
executed, no restored parent was resolved, no retired child acquired evidence,
and no inactive generated link became an accepted dependency route.

Two clean full builds must reproduce every manifested artifact and the manifest
byte-for-byte through `verify_full_rebuilds.py`. Canonical writes and publication
of resolved validation findings remain unauthorized.
""",
    )


def run_validator() -> None:
    subprocess.run(
        [sys.executable, str(OUT / "validate_rebuilt_v3.py")],
        cwd=REPO,
        check=True,
    )


def internal_clean_build_comparison() -> dict[str, object]:
    scripts = (
        "build_rebuilt_v3.py",
        "validate_rebuilt_v3.py",
        "verify_full_rebuilds.py",
    )

    def build(target: Path) -> tuple[dict[str, object], str]:
        for name in scripts:
            shutil.copyfile(OUT / name, target / name)
        subprocess.run(
            [sys.executable, str(target / "build_rebuilt_v3.py"), "--core-only"],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest_path = target / "lane_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest, digest(manifest_path)

    with tempfile.TemporaryDirectory(
        prefix=".ch19v3_core_a_", dir=str(STAGING)
    ) as left_parent:
        with tempfile.TemporaryDirectory(
            prefix=".ch19v3_core_b_", dir=str(STAGING)
        ) as right_parent:
            left_root = Path(left_parent) / LOGICAL_LANE
            right_root = Path(right_parent) / LOGICAL_LANE
            left_root.mkdir()
            right_root.mkdir()
            left, left_manifest_hash = build(left_root)
            right, right_manifest_hash = build(right_root)
            left_hashes = {row["path"]: row["sha256"] for row in left["files"]}
            right_hashes = {
                row["path"]: row["sha256"] for row in right["files"]
            }
            if (
                left_hashes != right_hashes
                or left_manifest_hash != right_manifest_hash
            ):
                raise ValueError("clean core builds are not byte-identical")
            return {
                "status": "PASS",
                "logical_lane": LOGICAL_LANE,
                "clean_builds_compared": 2,
                "manifested_artifacts_compared": len(left_hashes),
                "manifest_included_in_comparison": True,
                "shared_core_manifest_sha256": left_manifest_hash,
                "external_full_packet_verifier": "verify_full_rebuilds.py",
                "external_full_packet_comparison_required": True,
            }


def build(core_only: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = verify_manifest(SOURCE)
    review_manifest = verify_manifest(REVIEW)
    before = snapshot_input_hashes(source_manifest, review_manifest)
    seed_source_csvs(source_manifest)
    apply_review_patch_dispositions()
    patch_correction_map()
    patch_finding_resolutions()
    patch_metrics_and_gates()
    write_exact_patch_audit(review_manifest)
    write_reports()
    write_input_hash_audit(before, source_manifest, review_manifest)
    write_preservation_audit(source_manifest)
    counts = Counter(
        row["application_status"]
        for row in read_csv(OUT / "review_patch_application.csv")
    )
    write_json(
        OUT / "patch_application_results.json",
        {
            "lane": LOGICAL_LANE,
            "status": "PASS_10_OF_10_APPLIED_PENDING_NONAUTHOR_ACCEPTANCE",
            "exact_fixes_expected": 10,
            "exact_fixes_applied": 10,
            "review_disposition_counts": dict(sorted(counts.items())),
            "source_manifest_sha256": digest(SOURCE / "lane_manifest.json"),
            "review_manifest_sha256": digest(REVIEW / "lane_manifest.json"),
            "canonical_integration": "NOT_AUTHORIZED",
            "author_acceptance": "NOT_ACCEPTED_FRESH_NONAUTHOR_REVIEW_REQUIRED",
        },
    )
    run_validator()
    if core_only:
        return
    write_json(
        OUT / "deterministic_rebuild_results.json",
        internal_clean_build_comparison(),
    )
    run_validator()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    build(args.core_only)
    print(
        json.dumps(
            {
                "status": "CH19_REBUILT_V3_GENERATED",
                "logical_lane": LOGICAL_LANE,
                "core_only": args.core_only,
                "author_verdict": "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE",
                "fresh_nonauthor_acceptance": "REQUIRED",
                "canonical_integration": "NOT_AUTHORIZED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
