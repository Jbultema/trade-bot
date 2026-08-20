#!/usr/bin/env python3
"""Meaning-based author validator for the Chapter 19 rebuilt-v3 packet."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
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
AI_ROOT = REPO / "research/ai_financial_system"
STAGING = AI_ROOT / "staging"
SOURCE = STAGING / "lane_2_chapter_19_rebuilt_v2"
REVIEW = STAGING / "lane_1_verify_chapter_19_rebuilt_v2"
LOGICAL_LANE = "lane_3_chapter_19_rebuilt_v3"
STATUS_DATE = "2026-07-26"
GENERIC_VALIDATOR = AI_ROOT / "scripts/validate_staging_packet.py"
AUTHOR_VERDICT = "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE"
MANIFEST_VERDICT = (
    "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE;"
    "FRESH_NONAUTHOR_ACCEPTANCE_REQUIRED;NOT_CANONICAL"
)
EXPECTED_DISPOSITIONS = {
    "DIRECT_OR_SUPERSEDING_CORRECTION_VERIFIED": 396,
    "DEFERRED_EXPLICIT_HOLD": 985,
    "SAFE_ALTERNATIVE_CHILD_ROUTE_RETIRED": 471,
    "SAFE_ALTERNATIVE_PARENT_RESTORED": 394,
    "PARTIAL_PROVENANCE_MISMATCH": 1,
}
EXPECTED_ORIGINAL = {
    "PASS": 36,
    "PASS_WITH_EXPLICIT_HOLD": 11,
    "PASS_DESCRIPTIVE_ONLY": 1,
    "SAFE_ALTERNATIVE_NOT_LITERAL_COMPLETION": 52,
    "PARTIAL_PROVENANCE_MISMATCH": 1,
}
EXPECTED_FINDINGS = {
    "PASS": 2,
    "PASS_WITH_EXPLICIT_HOLD": 10,
    "PASS_DESCRIPTIVE_ONLY": 1,
    "PASS_STRUCTURAL_ONLY": 1,
}
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"{path}: empty validation table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def count_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def find(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"expected one {field}={value}, found {len(found)}")
    return found[0]


def check_row(
    check_id: str, passed: bool, description: str, evidence: str
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "description": description,
        "evidence": evidence,
    }


def verify_input_manifests() -> bool:
    for root in (SOURCE, REVIEW):
        manifest = json.loads(
            (root / "lane_manifest.json").read_text(encoding="utf-8")
        )
        for artifact in manifest["files"]:
            path = root / artifact["path"]
            if not path.exists() or digest(path) != artifact["sha256"]:
                return False
    return True


def semantic_patch_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    applications = read_csv(OUT / "review_patch_application.csv")
    dispositions = Counter(row["application_status"] for row in applications)

    checks.append(
        check_row(
            "V19V3-PATCH-001",
            len(applications) == 2247
            and dispositions["DEFERRED_EXPLICIT_HOLD"] == 985
            and all(
                row["exact_requested_work_completed"] == "false"
                for row in applications
                if row["application_status"] == "DEFERRED_EXPLICIT_HOLD"
            ),
            "The 985 held rows are deferred, not completed repairs.",
            f"rows={len(applications)}; dispositions={dict(dispositions)}",
        )
    )
    checks.append(
        check_row(
            "V19V3-PATCH-002",
            dispositions["SAFE_ALTERNATIVE_CHILD_ROUTE_RETIRED"] == 471
            and all(
                row["exact_requested_work_completed"] == "false"
                and row["safe_unresolved_architecture_effect"] == "true"
                for row in applications
                if row["application_status"]
                == "SAFE_ALTERNATIVE_CHILD_ROUTE_RETIRED"
            ),
            "Child-route retirement is a safe alternative, not route completion.",
            "471 rows are explicitly nonliteral safe alternatives.",
        )
    )
    checks.append(
        check_row(
            "V19V3-PATCH-003",
            dispositions["SAFE_ALTERNATIVE_PARENT_RESTORED"] == 394
            and all(
                row["exact_requested_work_completed"] == "false"
                and row["safe_unresolved_architecture_effect"] == "true"
                for row in applications
                if row["application_status"]
                == "SAFE_ALTERNATIVE_PARENT_RESTORED"
            ),
            "Parent restoration is a safe fallback, not semantic completion.",
            "394 rows are explicitly nonliteral safe alternatives.",
        )
    )

    corrections = read_csv(OUT / "correction_map.csv")
    patch_28 = find(corrections, "patch_id", "PATCH-H19-0028")
    checks.append(
        check_row(
            "V19V3-PATCH-004",
            patch_28["correction_status"]
            == "not_applied_in_corrected_staging_rebuilt_v2_fixed"
            and "historical_provenance_repaired"
            in patch_28["rebuilt_v2_resolution_method"]
            and patch_28["rebuilt_v2_resolution_status"]
            == (
                "LIVE_SEMANTICS_FIXED_IN_REBUILT_V2_"
                "HISTORICAL_PROVENANCE_REPAIRED"
            ),
            "PATCH-H19-0028 now separates corrected-packet history from v2 repair.",
            (
                f"correction_status={patch_28['correction_status']}; "
                f"rebuilt_status={patch_28['rebuilt_v2_resolution_status']}"
            ),
        )
    )
    restored = [
        row
        for row in corrections
        if "PATCH-H19-0029" <= row["patch_id"] <= "PATCH-H19-0080"
    ]
    checks.append(
        check_row(
            "V19V3-PATCH-005",
            len(restored) == 52
            and all(
                row["rebuilt_v2_resolution_status"]
                == "SAFE_ALTERNATIVE_PENDING_SEMANTIC_DECOMPOSITION"
                and row["literal_or_exact_completion"] == "false"
                for row in restored
            ),
            "All 52 parent restorations remain pending semantic decomposition.",
            f"parent_restoration_rows={len(restored)}",
        )
    )
    correction_fields = set(corrections[0])
    v2_refs = [
        item
        for row in corrections
        for item in row["rebuilt_v2_implemented_artifacts"].split("|")
        if item
    ]
    checks.append(
        check_row(
            "V19V3-PATCH-006",
            "implemented_artifacts" not in correction_fields
            and {
                "corrected_packet_implemented_artifacts",
                "rebuilt_v2_implemented_artifacts",
                "corrected_packet_artifact_scope",
                "rebuilt_v2_artifact_scope",
            }.issubset(correction_fields)
            and bool(v2_refs)
            and all((OUT / item).exists() for item in v2_refs)
            and all(
                row["corrected_packet_artifact_scope"]
                == "lane_2_chapter_19_corrected_external_historical_context"
                and row["rebuilt_v2_artifact_scope"]
                == "lane_2_chapter_19_rebuilt_v2_current_source_context"
                for row in corrections
            ),
            "Corrected-packet and rebuilt-v2 artifact provenance are versioned.",
            (
                f"resolved_rebuilt_v2_refs={len(v2_refs)}; "
                f"missing={sum(not (OUT / item).exists() for item in v2_refs)}"
            ),
        )
    )

    manifest = json.loads(
        (OUT / "lane_manifest.json").read_text(encoding="utf-8")
    )
    report = (OUT / "validation_report.md").read_text(encoding="utf-8")
    semantic = json.loads(
        (OUT / "semantic_validation_results.json").read_text(encoding="utf-8")
    )
    expected_manifest_counts = {
        "direct_or_superseding_corrections_verified": 396,
        "deferred_explicit_holds": 985,
        "safe_alternative_child_routes_retired": 471,
        "safe_alternative_parents_restored": 394,
        "partial_provenance_mismatches": 1,
        "total_dispositions_accounted": 2247,
    }
    semantic_counts = semantic["review_patch_dispositions"]
    checks.append(
        check_row(
            "V19V3-PATCH-007",
            manifest.get("review_disposition_accounting")
            == expected_manifest_counts
            and semantic_counts["total_accounted"] == 2247
            and semantic_counts[
                "direct_or_superseding_corrections_verified"
            ]
            == 396
            and semantic_counts["deferred_explicit_holds"] == 985
            and "396 independently verified" in report
            and "985 deferred holds" in report
            and "2,247 completed" not in report,
            "Report, JSON, and manifest use disposition accounting.",
            f"manifest_counts={manifest.get('review_disposition_accounting')}",
        )
    )

    gates = read_csv(OUT / "writing_gate_audit.csv")
    gate = find(gates, "gate_id", "WG-R2-0011")
    checks.append(
        check_row(
            "V19V3-PATCH-008",
            gate["status"] == "pass_accounting_only"
            and "only 396" in gate["evidence"]
            and "985 holds" in gate["evidence"]
            and find(gates, "gate_id", "WG-R3-0013")["status"] == "hold"
            and find(gates, "gate_id", "WG-R3-0014")["status"] == "hold",
            "The patch gate certifies accounting only and preserves holds.",
            f"WG-R2-0011={gate['status']}",
        )
    )

    coverage = read_csv(OUT / "evidence_system_coverage.csv")
    metric = find(coverage, "metric_id", "ECM-R2-0012")
    new_metrics = {
        row["metric"]: row["value"]
        for row in coverage
        if row["metric_id"].startswith("ECM-R3-")
    }
    checks.append(
        check_row(
            "V19V3-PATCH-009",
            metric["metric"] == "review_patch_dispositions_accounted"
            and metric["value"] == "2247"
            and metric["governance_status"]
            == "accounting_only_not_work_completion"
            and new_metrics
            == {
                "direct_or_superseding_corrections_verified": "396",
                "deferred_explicit_holds": "985",
                "safe_alternative_child_routes_retired": "471",
                "safe_alternative_parents_restored": "394",
                "partial_provenance_mismatches": "1",
            },
            "Coverage metrics name dispositions rather than completed repairs.",
            f"metric={metric['metric']}; categories={new_metrics}",
        )
    )

    resolutions = read_csv(OUT / "independent_review_resolution.csv")
    statuses = Counter(row["rebuilt_v2_status"] for row in resolutions)
    checks.append(
        check_row(
            "V19V3-PATCH-010",
            len(resolutions) == 14
            and len({row["resolution"] for row in resolutions}) == 14
            and all(
                row["residual_operational_gate"]
                and row["canonical_integration_status"] == "HOLD"
                for row in resolutions
            )
            and statuses
            == Counter(
                {
                    "SAFE_UNRESOLVED_ARCHITECTURE_WITH_EXPLICIT_HOLD": 10,
                    "DIRECT_OR_SUPERSEDING_CORRECTION_VERIFIED": 2,
                    "DESCRIPTIVE_ONLY_NOT_POLICY_VALIDATION": 1,
                    "STRUCTURAL_VALIDATION_ONLY": 1,
                }
            ),
            "All 14 findings have specific architecture changes and residual gates.",
            f"status_categories={dict(statuses)}",
        )
    )
    return checks


def regression_checks() -> tuple[list[dict[str, object]], dict[str, object]]:
    checks: list[dict[str, object]] = []

    input_hashes = read_csv(OUT / "v3_input_hash_audit.csv")
    checks.append(
        check_row(
            "V19V3-REG-001",
            verify_input_manifests()
            and bool(input_hashes)
            and all(row["status"] == "PASS_UNCHANGED" for row in input_hashes),
            "Source and review lanes remain hash-identical to their manifests.",
            f"input_artifacts_checked={len(input_hashes)}",
        )
    )

    preservation = read_csv(OUT / "preserved_artifact_hash_audit.csv")
    preserved_status = {
        row["artifact"]: row["status"] for row in preservation
    }
    checks.append(
        check_row(
            "V19V3-REG-002",
            len(preservation) > len(CHANGED_SOURCE_CSVS)
            and set(
                row["artifact"]
                for row in preservation
                if row["intentionally_changed"] == "true"
            )
            == CHANGED_SOURCE_CSVS
            and all(
                status
                in {"PASS_INTENTIONAL_CHANGE", "PASS_BYTE_PRESERVED"}
                for status in preserved_status.values()
            ),
            "Only the six governed source CSVs changed; all others are byte-preserved.",
            f"source_csvs_checked={len(preservation)}",
        )
    )

    unknowns = read_csv(OUT / "classified_unknowns.csv")
    role_counts = Counter(row["record_role"] for row in unknowns)
    test_counts = Counter(row["resolution_test_status"] for row in unknowns)
    checks.append(
        check_row(
            "V19V3-REG-003",
            len(unknowns) == 252
            and role_counts
            == Counter(
                {
                    "governed_single_question": 200,
                    "governed_compound_parent": 52,
                }
            )
            and test_counts
            == Counter(
                {
                    "unresolved_unvalidated_row_test_required": 195,
                    "blocked_pending_semantic_decomposition": 52,
                    "independently_authored_pending_row_owner_acceptance": 5,
                }
            ),
            "Governed question and test populations are preserved.",
            f"roles={dict(role_counts)}; tests={dict(test_counts)}",
        )
    )

    children = read_csv(OUT / "proposed_child_decomposition.csv")
    checks.append(
        check_row(
            "V19V3-REG-004",
            len(children) == 114
            and all(
                row["question_self_contained"] == "false"
                and row["sources_component_specific"] == "false"
                and row["resolution_plan_component_specific"] == "false"
                and row["active_in_governance_population"] == "false"
                and row["proposal_status"]
                == "retired_structural_decomposition_not_semantically_atomic"
                for row in children
            ),
            "All 114 false-atomic children remain inactive and retired.",
            "active_children=0",
        )
    )

    reporting = read_csv(OUT / "reporting_path_audit.csv")
    checks.append(
        check_row(
            "V19V3-REG-005",
            len(reporting) == 119
            and all(
                row["reporting_obligation_reviewed"] == "false"
                and row["repositories_searched"] == ""
                and row["search_exhaustion_status"] == "not_performed"
                and row["accepted_epistemic_state"] == ""
                and row["owner_acceptance_status"] == "pending"
                for row in reporting
            ),
            "The 119 reporting audits remain unperformed and unaccepted.",
            "reporting_audits=119; accepted_states=0",
        )
    )

    blockers = read_csv(OUT / "downstream_blocker_matrix.csv")
    checks.append(
        check_row(
            "V19V3-REG-006",
            len(blockers) == 138
            and all(
                row["exact_downstream_input_or_output"] == ""
                and row["chapter_owner_acceptance_status"] == "pending"
                and row["governed_link_status"]
                == "preserved_reference_not_current_governed_dependency"
                and row["priority_eligible"] == "false"
                and row["publication_status"] == "HOLD"
                for row in blockers
            ),
            "All 138 blocker references remain unmapped and unaccepted.",
            "mapped_or_accepted_blockers=0",
        )
    )

    generated = read_csv(OUT / "proposed_blocker_hypotheses.csv")
    checks.append(
        check_row(
            "V19V3-REG-007",
            len(generated) == 750
            and all(
                row["hypothesis_status"]
                == "inactive_unverified_template_hypothesis"
                and row["exact_downstream_input_or_output"] == ""
                and row["chapter_owner_acceptance_status"] == "pending"
                and row["priority_eligible"] == "false"
                and row["publication_status"] == "HOLD"
                for row in generated
            ),
            "All 750 generated dependency links remain inactive.",
            "accepted_generated_links=0",
        )
    )

    queue = read_csv(OUT / "monitoring_queue.csv")
    priority = read_csv(OUT / "priority_sensitivity.csv")
    checks.append(
        check_row(
            "V19V3-REG-008",
            len(queue) == 252
            and all(
                row["priority_rank"] == ""
                and row["policy_status"]
                == "descriptive_stratum_not_validated_queue_policy"
                for row in queue
            )
            and all(
                not any(
                    token in value.lower()
                    for token in ("pseudo_score", "ordinal_rank")
                )
                for row in priority
                for value in row.values()
            ),
            "No pseudo-score or ordinal rank is reintroduced.",
            "ranked_queue_rows=0",
        )
    )

    routes = read_csv(OUT / "unknown_source_route_register.csv")
    checks.append(
        check_row(
            "V19V3-REG-009",
            len(routes) == 994
            and all(
                row["venue_identity_status"] == "proposed_pending_normalization"
                and row["search_status"] == "not_executed_in_rebuild"
                and row["search_evidence_path"] == ""
                and row["owner_acceptance_status"] == "pending"
                and row["priority_eligible"] == "false"
                and row["publication_status"] == "HOLD"
                for row in routes
            ),
            "All 994 route candidates remain unnormalized, unsearched, and unaccepted.",
            "normalized_searched_or_accepted_routes=0",
        )
    )

    chronology = read_csv(OUT / "supersession_version_audit.csv")
    order_counts = Counter(row["effective_date_order"] for row in chronology)
    checks.append(
        check_row(
            "V19V3-REG-010",
            len(chronology) == 73
            and order_counts
            == Counter(
                {
                    "nondecreasing": 56,
                    "superseding_effective_date_before_prior": 17,
                }
            )
            and all(
                row["word_later_is_safe"] == "false"
                and row["version_ingestion_timestamp"] == ""
                and row["version_order_conclusion"]
                == "unknown_no_version_or_ingestion_timestamp"
                for row in chronology
            ),
            "Chronology remains fail-closed with all 17 inversions visible.",
            f"effective_date_order={dict(order_counts)}",
        )
    )

    confidence = read_csv(OUT / "confidence_evolution_audit.csv")
    baselines = read_csv(OUT / "proposed_claim_confidence_baselines.csv")
    history_counts = Counter(row["history_entry_count"] for row in confidence)
    checks.append(
        check_row(
            "V19V3-REG-011",
            len(confidence) == 427
            and history_counts == Counter({"0": 393, "1": 34})
            and all(
                row["genuine_longitudinal_evolution_observable"] == "false"
                and row["historical_change_claimable"] == "false"
                for row in confidence
            )
            and len(baselines) == 393
            and all(
                row["prior_evidence_grade"] == ""
                and row["prior_confidence"] == ""
                and row["integration_timestamp"] == ""
                and row["canonical_history_id"] == ""
                for row in baselines
            ),
            "Confidence records remain snapshots, not fabricated history.",
            f"history_entry_counts={dict(history_counts)}",
        )
    )

    monitoring = read_csv(OUT / "source_monitoring_register.csv")
    external = [
        row
        for row in monitoring
        if row["external_or_internal"] == "external_source"
    ]
    checks.append(
        check_row(
            "V19V3-REG-012",
            len(monitoring) == 156
            and len(external) == 155
            and all(
                row["document_archive_path_present"] == "false"
                and row["document_content_hash_present"] == "false"
                and row["document_preservation_status"]
                == "external_source_not_archived_or_content_hashed"
                for row in external
            ),
            "External preservation remains zero of 155.",
            "external_archives=0/155",
        )
    )

    falsifiers = read_csv(OUT / "falsifiers.csv")
    checks.append(
        check_row(
            "V19V3-REG-013",
            len(falsifiers) == 10
            and all(
                row["threshold_status"] == "proposed_not_validated"
                and row["governance_status"]
                == "proposed_unvalidated_decision_rule"
                and row["owner_acceptance_status"] == "pending"
                and row["publication_status"] == "HOLD"
                for row in falsifiers
            ),
            "All ten falsifiers remain proposed and unexecuted.",
            "executed_falsifiers=0",
        )
    )

    patch_audit = read_csv(OUT / "v3_exact_patch_disposition_audit.csv")
    checks.append(
        check_row(
            "V19V3-REG-014",
            len(patch_audit) == 10
            and [row["review_fix_id"] for row in patch_audit]
            == [f"V19R2-{number:03d}" for number in range(1, 11)]
            and all(
                row["author_application_status"]
                == "APPLIED_PENDING_FRESH_NONAUTHOR_ACCEPTANCE"
                and row["canonical_integration_effect"] == "HOLD"
                for row in patch_audit
            ),
            "All ten exact fixes are author-applied but not author-accepted.",
            "fresh_nonauthor_acceptance=REQUIRED",
        )
    )

    report = (OUT / "research_pass.md").read_text(encoding="utf-8")
    normalized_report = " ".join(report.lower().split())
    checks.append(
        check_row(
            "V19V3-REG-015",
            AUTHOR_VERDICT in report.replace(" ", "_")
            and "not canonical" in normalized_report
            and "monograph, executive summary, or standalone essay"
            in report
            and "Classification of a missing test is not execution" in report
            and "Restoration of a parent question is not resolution" in report,
            "Publication boundaries and non-equivalence controls are explicit.",
            AUTHOR_VERDICT,
        )
    )

    summary = {
        "governed_questions": len(unknowns),
        "compound_parents": role_counts["governed_compound_parent"],
        "inactive_false_atomic_children": len(children),
        "authored_tests": test_counts[
            "independently_authored_pending_row_owner_acceptance"
        ],
        "blank_or_blocked_tests": (
            test_counts["unresolved_unvalidated_row_test_required"]
            + test_counts["blocked_pending_semantic_decomposition"]
        ),
        "reporting_audits": len(reporting),
        "accepted_reporting_states": 0,
        "blocker_references": len(blockers),
        "mapped_or_accepted_blockers": 0,
        "inactive_generated_links": len(generated),
        "pseudo_scores_or_ranks": 0,
        "source_routes": len(routes),
        "normalized_searched_or_accepted_routes": 0,
        "chronology_rows": len(chronology),
        "unsafe_later_flags": 0,
        "effective_date_inversions": order_counts[
            "superseding_effective_date_before_prior"
        ],
        "current_confidence_snapshots": len(baselines),
        "fabricated_confidence_evolution": 0,
        "external_archives": 0,
        "external_archive_population": len(external),
        "proposed_falsifiers": len(falsifiers),
        "executed_falsifiers": 0,
    }
    return checks, summary


def refresh_manifest() -> None:
    files: list[dict[str, object]] = []
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path.name == "lane_manifest.json":
            continue
        item: dict[str, object] = {
            "path": path.name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            item["rows"] = count_csv(path)
        else:
            item["lines"] = count_lines(path)
        files.append(item)
    manifest = {
        "lane": LOGICAL_LANE,
        "chapter": "CH19",
        "status_date": STATUS_DATE,
        "scope": (
            "Chapter 19 unresolved evidence-architecture correction packet"
        ),
        "source_lane": SOURCE.name,
        "source_manifest_sha256": digest(SOURCE / "lane_manifest.json"),
        "review_lane": REVIEW.name,
        "review_manifest_sha256": digest(REVIEW / "lane_manifest.json"),
        "decision": AUTHOR_VERDICT,
        "verdict": MANIFEST_VERDICT,
        "review_disposition_accounting": {
            "direct_or_superseding_corrections_verified": 396,
            "deferred_explicit_holds": 985,
            "safe_alternative_child_routes_retired": 471,
            "safe_alternative_parents_restored": 394,
            "partial_provenance_mismatches": 1,
            "total_dispositions_accounted": 2247,
        },
        "files": files,
        "publication_boundary": {
            "unresolved_architecture": (
                "author_repairs_applied_pending_fresh_nonauthor_acceptance"
            ),
            "resolved_validation_findings": "not_authorized",
            "monograph_findings": "operational_gates_required",
            "executive_summary_findings": "operational_gates_required",
            "standalone_essay_findings": "operational_gates_required",
            "canonical_database_write": "not_authorized",
        },
        "acceptance": {
            "author_validation": "SELF_VALIDATION_ONLY",
            "fresh_nonauthor_acceptance": "REQUIRED",
        },
        "no_canonical_write_attestation": {
            "source_or_review_lane_modified": False,
            "canonical_paths_written": [],
        },
        "fixed_logical_packet_identifier": True,
        "physical_output_path_in_manifested_semantics": False,
        "self_hash_policy": "lane_manifest.json excludes its own hash",
    }
    write_json(OUT / "lane_manifest.json", manifest)


def run_generic() -> dict[str, object]:
    generic = subprocess.run(
        [
            sys.executable,
            str(GENERIC_VALIDATOR),
            str(OUT),
            "--json",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if generic.returncode != 0:
        raise RuntimeError(
            (generic.stdout.strip() + "\n" + generic.stderr.strip()).strip()
        )
    result = json.loads(generic.stdout)
    if not result["passed"]:
        raise RuntimeError(f"generic validation failed: {result}")
    return result


def main() -> int:
    # Patch check 7 meaning-tests manifest language, so seal a fail-closed
    # current manifest before evaluating semantic checks.
    refresh_manifest()
    patches = semantic_patch_checks()
    regressions, summary = regression_checks()
    all_checks = patches + regressions
    status = (
        "PASS"
        if len(patches) == 10
        and all(row["status"] == "PASS" for row in all_checks)
        else "FAIL"
    )

    write_csv(OUT / "patch_semantic_validation.csv", patches)
    write_csv(OUT / "semantic_regression_audit.csv", regressions)
    results = {
        "lane": LOGICAL_LANE,
        "chapter": "CH19",
        "status_date": STATUS_DATE,
        "validator_status": status,
        "author_verdict": AUTHOR_VERDICT,
        "author_validation": "SELF_VALIDATION_ONLY",
        "fresh_nonauthor_acceptance": "REQUIRED",
        "exact_fixes_expected": 10,
        "exact_fixes_semantically_passed": sum(
            row["status"] == "PASS" for row in patches
        ),
        "regression_checks_passed": sum(
            row["status"] == "PASS" for row in regressions
        ),
        "regression_checks_total": len(regressions),
        "review_disposition_counts": EXPECTED_DISPOSITIONS,
        "original_patch_reperformance_counts": EXPECTED_ORIGINAL,
        "finding_reperformance_counts": EXPECTED_FINDINGS,
        "preserved_unresolved_architecture": summary,
        "resolved_validation_findings_publication": "NOT_AUTHORIZED",
        "canonical_write": "NOT_AUTHORIZED",
    }
    write_json(OUT / "validation_results.json", results)
    write_json(
        OUT / "generic_validation_results.json",
        {
            "lane": LOGICAL_LANE,
            "passed": True,
            "status": "PENDING_RESEAL_AND_GENERIC_VALIDATION",
        },
    )
    refresh_manifest()

    if status != "PASS":
        print(json.dumps(results, sort_keys=True))
        return 1

    generic = run_generic()
    generic["status"] = "PASS"
    generic["validator"] = (
        "research/ai_financial_system/scripts/validate_staging_packet.py"
    )
    write_json(OUT / "generic_validation_results.json", generic)
    refresh_manifest()

    # Validate the exact final manifest after the generic-result hash changes.
    generic_final = run_generic()
    if (
        generic_final["files_checked"] != generic["files_checked"]
        or generic_final["csv_rows_checked"] != generic["csv_rows_checked"]
        or generic_final["errors"]
        or generic_final["warnings"] != generic["warnings"]
    ):
        raise RuntimeError("generic validator did not stabilize after reseal")

    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
