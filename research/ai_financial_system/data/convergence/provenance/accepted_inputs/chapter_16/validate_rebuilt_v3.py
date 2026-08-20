#!/usr/bin/env python3
"""Meaning-based semantic and regression validator for Chapter 16 rebuilt-v3."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
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
            "lane_2_chapter_16_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
AI_ROOT = REPO / "research/ai_financial_system"
STAGING = AI_ROOT / "staging"
SOURCE = STAGING / "lane_2_chapter_16_rebuilt_v2"
REVIEW = STAGING / "lane_1_verify_chapter_16_rebuilt_v2"
LOGICAL_LANE = "lane_3_chapter_16_rebuilt_v3"
STATUS_DATE = "2026-07-26"
GENERIC_VALIDATOR = AI_ROOT / "scripts/validate_staging_packet.py"


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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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


def load_review_module():
    path = REVIEW / "run_verification.py"
    spec = importlib.util.spec_from_file_location("chapter16_v2_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PACKET = OUT
    return module


def semantic_patch_checks() -> list[dict[str, object]]:
    checks: list[tuple[str, bool, str]] = []
    parties = read_csv(OUT / "project_party_registry.csv")
    blackstone = find(parties, "project_party_registry_id", "PPARTY-C16-003")
    saline = find(parties, "project_party_registry_id", "PPARTY-C16-022")
    checks.append(
        (
            "V3-PATCH-001",
            blackstone["party_id"] == "ENT-BLACKSTONE"
            and blackstone["party_type"] == "canonical_proxy_or_group"
            and blackstone["multiplicity"] == "identity_not_one_to_one"
            and blackstone["identity_status"]
            == "manager_and_affiliated_fund_group"
            and "exact sponsor" in blackstone["notes"],
            "Blackstone manager/affiliate-group identity is explicit and unresolved.",
        )
    )
    checks.append(
        (
            "V3-PATCH-002",
            saline["party_id"] == "ENT-SALINE-DC"
            and saline["party_type"] == "canonical_research_project_construct"
            and saline["multiplicity"] == "identity_not_one_to_one"
            and saline["identity_status"]
            == "project_research_construct_exact_legal_entity_unresolved"
            and all(
                token in saline["notes"]
                for token in ("borrower", "owner", "vehicle", "unresolved")
            ),
            "Saline is a research construct; borrower/owner/vehicle are unresolved.",
        )
    )

    roles = read_csv(OUT / "project_role_edges.csv")
    blackstone_role = find(roles, "project_edge_id", "PROLE-160008")
    saline_role = find(roles, "project_edge_id", "PROLE-160006")
    checks.append(
        (
            "V3-PATCH-003",
            blackstone_role["role"]
            == "equity_sponsor_affiliate_group_exact_fund_and_vehicle_unresolved",
            "Blackstone identity qualification propagates into the role edge.",
        )
    )
    checks.append(
        (
            "V3-PATCH-004",
            saline_role["role"]
            == "project_research_construct_exact_legal_entity_unresolved",
            "Saline project-construct qualification propagates into the role edge.",
        )
    )

    snapshot = find(
        read_csv(OUT / "project_centrality_snapshot.csv"),
        "snapshot_id",
        "PCS-160001",
    )
    checks.append(
        (
            "V3-PATCH-005",
            "equity_sponsor_affiliate_group" in snapshot["role_types"]
            and "project_research_construct" in snapshot["role_types"]
            and all(
                token in snapshot["identity_state"]
                for token in ("Blackstone", "Saline", "unresolved")
            ),
            "Project snapshot carries both repaired meanings without count changes.",
        )
    )

    matrix = read_csv(OUT / "counterparty_role_matrix.csv")
    google = find(matrix, "role_matrix_id", "CRM-160005")
    dte = find(matrix, "role_matrix_id", "CRM-160009")
    checks.append(
        (
            "V3-PATCH-006",
            "Alphabet_parent_proxy" in google["legal_obligor_role"]
            and "exact_Google" in google["legal_obligor_role"]
            and "unresolved" in google["guarantor_role"]
            and "parent_proxy" in google["role_boundary"],
            "Alphabet parent-proxy and unresolved Google legal-party semantics propagate.",
        )
    )
    checks.append(
        (
            "V3-PATCH-007",
            all(
                "unresolved" in dte[field]
                for field in (
                    "legal_obligor_role",
                    "customer_role",
                    "supplier_role",
                    "manager_role",
                    "role_boundary",
                )
            )
            and "DTE_parent_proxy" in dte["role_boundary"],
            "DTE parent-proxy and unresolved operating-utility semantics propagate.",
        )
    )

    ares = find(
        read_csv(OUT / "single_point_failure_register.csv"),
        "failure_id",
        "SPF-160006",
    )
    expected_clock = (
        "default_at_day_5_unless_satisfactory_cure_evidence_delivered_"
        "within_first_5_days_conditionally_extends_to_day_10"
    )
    checks.append(
        (
            "V3-PATCH-008",
            ares["documented_lag_or_trigger"] == expected_clock,
            "Ares day-five/day-ten cure clock preserves the conditional branch.",
        )
    )

    gate = find(
        read_csv(OUT / "writing_gate_audit.csv"),
        "gate_id",
        "WGA-160015",
    )
    checks.append(
        (
            "V3-PATCH-009",
            gate["decision"].startswith("block_")
            and "nonauthor" in gate["required_evidence"]
            and "001_through_008" in gate["required_evidence"],
            "Bounded identity prose remains blocked pending semantics and nonauthor review.",
        )
    )

    semantics = json.loads(
        (OUT / "semantic_validation_results.json").read_text(encoding="utf-8")
    )
    report = (OUT / "validation_report.md").read_text(encoding="utf-8")
    checks.append(
        (
            "V3-PATCH-010",
            semantics["packet_decision"]
            == "REVIEW_REQUIRED_FRESH_NONAUTHOR_ACCEPTANCE"
            and semantics["comparative_publication"] == "BLOCKED"
            and semantics["canonical_database_write"]
            == "NOT_PERFORMED_NOT_AUTHORIZED"
            and "REVIEW REQUIRED" in report,
            "Packet remains revision/review-required with both release blocks.",
        )
    )

    research = (OUT / "research_pass.md").read_text(encoding="utf-8")
    normalized_research = " ".join(research.split())
    prior = find(
        read_csv(OUT / "applied_exact_patch_audit.csv"),
        "review_patch_id",
        "PATCH-V16R-0044",
    )
    checks.append(
        (
            "V3-PATCH-011",
            all(
                phrase in normalized_research
                for phrase in (
                    "574 mechanically enumerated structured",
                    "44 adjudication records",
                    "618-row combined",
                    "356 remain unadjudicated holds",
                )
            )
            and "574 structured candidates" in prior["notes"]
            and "44 separately reviewed adjudication" in prior["notes"],
            "Research pass distinguishes structured candidates and adjudications.",
        )
    )

    checks.append(
        (
            "V3-PATCH-012",
            semantics["lane"] == LOGICAL_LANE
            and semantics["fixed_logical_packet_identifier"] is True
            and str(OUT) not in json.dumps(semantics),
            "Manifested semantic results use a fixed logical packet identity.",
        )
    )

    return [
        {
            "check_id": check_id,
            "patch_id": f"PATCH-V16V2-{ordinal:03d}",
            "status": "PASS" if condition else "FAIL",
            "meaning_test": detail,
        }
        for ordinal, (check_id, condition, detail) in enumerate(checks, start=1)
    ]


def regression_checks() -> tuple[list[dict[str, object]], dict[str, object]]:
    audit = load_review_module()
    reconstruction, freeze, state = audit.candidate_reconstruction()
    equivalence, free = audit.equivalence_and_free_text(state["candidates"])
    edges, graph = audit.graph_audits(state["candidates"])
    identities, project_roles = audit.identity_and_project_audits()
    concentration, failures = audit.concentration_and_failure_audits()
    lineage = audit.lineage_audit()
    schemas, typed = audit.schema_roundtrip_audit()
    prior_review = read_csv(REVIEW / "prior_patch_outcome_audit.csv")
    prior_unchanged_pass = all(
        row["independent_status"] == "PASS"
        for row in prior_review
        if row["patch_id"] != "PATCH-V16R-0044"
    )
    research = (OUT / "research_pass.md").read_text(encoding="utf-8")
    prior_0044_pass = all(
        phrase in research
        for phrase in ("574", "356", "holds", "adjudication records", "618-row")
    )

    counts = state["counts"]
    rows: list[dict[str, object]] = []

    def add(check_id: str, condition: bool, detail: str) -> None:
        rows.append(
            {
                "check_id": check_id,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    add(
        "V3-REG-001",
        len(freeze) == 17
        and all(
            row["hash_status"] == row["row_status"] == "PASS" for row in freeze
        ),
        "17 frozen structured source tables retain exact hashes and rows.",
    )
    add(
        "V3-REG-002",
        len(reconstruction) == 574
        and all(row["result"] == "PASS" for row in reconstruction),
        "574 structured candidates regenerate exactly.",
    )
    add(
        "V3-REG-003",
        len(state["candidates"]) == 618
        and counts["include_role_edge"] == 44
        and counts["hold_pending_role_adjudication"] == 297
        and counts[
            "hold_pending_relationship_equivalence_adjudication"
        ]
        == 59
        and counts["exclude_from_bounded_graph"] == 218,
        "44/297/59/218 partition is unchanged across 618 combined rows.",
    )
    add(
        "V3-REG-004",
        len(equivalence) == 59
        and all(row["result"] == "PASS" for row in equivalence),
        "59 equivalence rows remain holds rather than silent suppressions.",
    )
    add(
        "V3-REG-005",
        len(free) == 5 and all(row["status"] == "PASS" for row in free),
        "427-claim free-text incompleteness remains an explicit release block.",
    )
    add(
        "V3-REG-006",
        len(edges) == 44 and all(row["result"] == "PASS" for row in edges),
        "44 selected topology-only edges recompute exactly.",
    )
    add(
        "V3-REG-007",
        len(graph) == 86 and all(row["result"] == "PASS" for row in graph),
        "86 entity/project graph metrics recompute exactly.",
    )
    add(
        "V3-REG-008",
        len(identities) == 78
        and all(row["verdict"] == "PASS" for row in identities),
        "All 78 graph/project/role identity meaning checks now pass.",
    )
    add(
        "V3-REG-009",
        len(project_roles) == 26
        and all(row["verdict"] == "PASS" for row in project_roles),
        "All 26 project-role joins and qualifications pass.",
    )
    add(
        "V3-REG-010",
        len(concentration) == 8
        and all(row["verdict"] == "PASS" for row in concentration),
        "Concentration, topology nonadditivity, and first-loss boundaries pass.",
    )
    add(
        "V3-REG-011",
        len(failures) == 11
        and all(row["verdict"] == "PASS" for row in failures),
        "Ten failure-party roles and the corrected Ares clock pass.",
    )
    add(
        "V3-REG-012",
        len(lineage) == 1726
        and all(row["result"] == "PASS" for row in lineage),
        "1,726 atomic record-claim links recompute exactly.",
    )
    add(
        "V3-REG-013",
        len(schemas) == 19
        and all(row["result"] == "PASS" for row in schemas),
        "All 19 proposed schema/loader surfaces round-trip losslessly.",
    )
    add(
        "V3-REG-014",
        len(typed) == 3 and all(row["result"] == "PASS" for row in typed),
        "Borrower, private-credit, and project-finance joins remain typed.",
    )
    add(
        "V3-REG-015",
        len(prior_review) == 46 and prior_unchanged_pass and prior_0044_pass,
        "All 46 prior repairs now pass the independent-review regression.",
    )
    add(
        "V3-REG-016",
        all(
            row["first_loss_bearer"] == "not_determined"
            for row in read_csv(OUT / "single_point_failure_register.csv")
        ),
        "No first-loss party is inferred.",
    )
    add(
        "V3-REG-017",
        all(
            row["edge_weight"] == "1"
            for row in read_csv(OUT / "transaction_network_edges.csv")
        ),
        "All selected graph weights remain topology-only unit weights.",
    )
    add(
        "V3-REG-018",
        all(
            row["status"].startswith("PASS")
            for row in read_csv(OUT / "preserved_artifact_hash_audit.csv")
        ),
        "Every source CSV is either byte-preserved or an enumerated patch target.",
    )
    add(
        "V3-REG-019",
        all(
            row["decision"].startswith("block")
            or row["decision"] == "publish_with_scope"
            or row["decision"] == "publish_with_conditions"
            for row in read_csv(OUT / "writing_gate_audit.csv")
        )
        and find(
            read_csv(OUT / "writing_gate_audit.csv"),
            "gate_id",
            "WGA-160013",
        )["decision"]
        == "block",
        "Existing writing gates remain intact and the census claim remains blocked.",
    )

    summary = {
        "structured_candidates": len(reconstruction),
        "combined_candidate_and_adjudication_rows": len(state["candidates"]),
        "selected_edges": len(edges),
        "graph_metrics": len(graph),
        "identity_checks": len(identities),
        "project_role_checks": len(project_roles),
        "record_claim_links": len(lineage),
        "schema_roundtrips": len(schemas),
        "prior_repairs": len(prior_review),
    }
    return rows, summary


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
        "chapter": "CH16",
        "status_date": STATUS_DATE,
        "scope": "bounded documented Chapter 16 subgraph correction packet",
        "source_lane": SOURCE.name,
        "source_manifest_sha256": digest(SOURCE / "lane_manifest.json"),
        "review_lane": REVIEW.name,
        "review_manifest_sha256": digest(REVIEW / "lane_manifest.json"),
        "verdict": (
            "REVIEW_REQUIRED_FRESH_NONAUTHOR_ACCEPTANCE;"
            "BLOCK_COMPARATIVE_PUBLICATION_AND_CANONICAL_WRITE"
        ),
        "files": files,
        "publication_boundary": {
            "bounded_documented_subgraph": (
                "repairs_applied_pending_fresh_nonauthor_acceptance"
            ),
            "comparative_ordering": "blocked",
            "free_text_completeness": "blocked",
            "canonical_database_write": "not_authorized",
        },
        "no_canonical_write_attestation": {
            "source_or_review_lane_modified": False,
            "canonical_paths_written": [],
        },
        "physical_output_path_in_manifested_semantics": False,
        "self_hash_policy": "lane_manifest.json excludes its own hash",
    }
    write_json(OUT / "lane_manifest.json", manifest)


def main() -> int:
    # The inherited prior-patch regression includes the packet verdict as one
    # control, so provide a current fail-closed manifest before evaluating it.
    refresh_manifest()
    patch_checks = semantic_patch_checks()
    regression, summary = regression_checks()
    accounting = read_csv(OUT / "v3_patch_accounting.csv")
    accounting_ok = (
        len(accounting) == 12
        and [row["patch_id"] for row in accounting]
        == [f"PATCH-V16V2-{number:03d}" for number in range(1, 13)]
        and all(row["implementation_status"].startswith("APPLIED") for row in accounting)
    )
    all_checks = patch_checks + regression
    status = (
        "PASS"
        if accounting_ok and all(row["status"] == "PASS" for row in all_checks)
        else "FAIL"
    )

    write_csv(OUT / "patch_semantic_validation.csv", patch_checks)
    write_csv(OUT / "semantic_regression_audit.csv", regression)
    results = {
        "lane": LOGICAL_LANE,
        "chapter": "CH16",
        "status_date": STATUS_DATE,
        "validator_status": status,
        "builder_acceptance": "SELF_VALIDATION_ONLY",
        "fresh_nonauthor_acceptance": "REQUIRED",
        "patches_expected": 12,
        "patches_semantically_passed": sum(
            row["status"] == "PASS" for row in patch_checks
        ),
        "regression_checks_passed": sum(
            row["status"] == "PASS" for row in regression
        ),
        "regression_checks_total": len(regression),
        "patch_accounting": "PASS" if accounting_ok else "FAIL",
        "summary": summary,
        "comparative_publication": "BLOCKED",
        "canonical_write": "NOT_AUTHORIZED",
    }
    write_json(OUT / "validation_results.json", results)
    write_json(
        OUT / "generic_validation_results.json",
        {
            "lane": LOGICAL_LANE,
            "status": "PASS",
            "validator": (
                "research/ai_financial_system/scripts/validate_staging_packet.py"
            ),
            "manifest_hash_and_count_validation": "PASS",
            "csv_shape_and_provisional_id_uniqueness": "PASS",
            "canonical_source_id_and_url_consistency": "PASS",
            "staged_source_claim_referential_integrity": "PASS",
        },
    )
    refresh_manifest()
    generic = subprocess.run(
        [sys.executable, str(GENERIC_VALIDATOR), str(OUT)],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if generic.returncode != 0:
        results["validator_status"] = "FAIL"
        results["generic_validation_error"] = (
            generic.stdout.strip() + "\n" + generic.stderr.strip()
        ).strip()
        write_json(OUT / "validation_results.json", results)
        write_json(
            OUT / "generic_validation_results.json",
            {
                "lane": LOGICAL_LANE,
                "status": "FAIL",
                "validator": (
                    "research/ai_financial_system/scripts/"
                    "validate_staging_packet.py"
                ),
            },
        )
        refresh_manifest()
        print(json.dumps(results, sort_keys=True))
        return 1

    # Re-seal and run the generic validator against the exact final manifest.
    refresh_manifest()
    generic_final = subprocess.run(
        [sys.executable, str(GENERIC_VALIDATOR), str(OUT)],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if generic_final.returncode != 0:
        raise RuntimeError(generic_final.stdout + generic_final.stderr)

    print(json.dumps(results, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
