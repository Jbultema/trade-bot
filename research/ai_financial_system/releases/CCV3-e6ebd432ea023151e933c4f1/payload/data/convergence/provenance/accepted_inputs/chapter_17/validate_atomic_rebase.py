#!/usr/bin/env python3
"""Meaning-based validation for the Chapter 17 atomic rebase packet."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path


OUT = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_3_chapter_17_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
AI_ROOT = REPO / "research/ai_financial_system"
STAGING = AI_ROOT / "staging"
SOURCE = STAGING / "lane_3_chapter_17_rebuilt_v2"
CH16 = STAGING / "lane_3_chapter_16_rebuilt_v3"
CH16_REVIEW = STAGING / "lane_2_verify_chapter_16_rebuilt_v3"
CANONICAL_DB = REPO / "reports/ai_financial_system/research.db"
EXPECTED_MANIFEST_HASHES = {
    SOURCE / "lane_manifest.json": (
        "53601a0fcb02daab119b898d227172fd893513674ad86df510b6c66625c34c21"
    ),
    CH16 / "lane_manifest.json": (
        "e506d400f9dc317097c11cf88d52920417b3af2cb04488b77ba485488285c1a4"
    ),
    CH16_REVIEW / "lane_manifest.json": (
        "8149151d11375bc7151e9361e811035f9dfada906fdd152050eb120769c88266"
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"{path}: malformed-width row")
        return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"expected one {field}={value}, found {len(found)}")
    return found[0]


def scalar_foreign_keys_are_atomic(
    artifacts: list[tuple[str, list[dict[str, str]]]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for artifact, rows in artifacts:
        for line, row in enumerate(rows, start=2):
            for field, value in row.items():
                is_fk_shape = (
                    field.endswith("_id")
                    or field.endswith("_reference")
                    or field in {
                        "dependent_reference_id",
                        "counterparty_id",
                        "scenario_id",
                        "ch17_object_id",
                        "failure_id",
                        "claim_id",
                        "source_id",
                    }
                )
                if is_fk_shape and "|" in value:
                    failures.append(f"{artifact}:{line}:{field}")
    return not failures, failures


def main() -> int:
    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    crosswalk = read_csv(OUT / "chapter16_stress_crosswalk.csv")
    counterparties = read_csv(OUT / "crosswalk_counterparty_junction.csv")
    scenarios_xw = read_csv(OUT / "crosswalk_scenario_junction.csv")
    objects = read_csv(OUT / "crosswalk_propagation_object_junction.csv")
    failures_xw = read_csv(OUT / "crosswalk_failure_mechanism_junction.csv")
    claims_xw = read_csv(OUT / "crosswalk_claim_junction.csv")
    sources_xw = read_csv(OUT / "crosswalk_claim_source_junction.csv")
    project_identity = read_csv(OUT / "atomic_project_identity.csv")
    row_map = read_csv(OUT / "original_to_normalized_row_map.csv")
    identity_audit = read_csv(OUT / "identity_resolution_audit.csv")
    field_audit = read_csv(OUT / "field_loss_audit.csv")
    lineage_audit = read_csv(OUT / "claim_source_lineage_audit.csv")
    intrinsic_audit = read_csv(OUT / "intrinsic_regression_audit.csv")
    hold_audit = read_csv(OUT / "external_hold_propagation_audit.csv")

    scenarios = read_csv(OUT / "scenario_catalog.csv")
    capex = read_csv(OUT / "capex_stress_arithmetic.csv")
    mechanics = read_csv(OUT / "observed_mechanics.csv")
    edges = read_csv(OUT / "propagation_edges.csv")
    loops = read_csv(OUT / "feedback_loops.csv")
    timeline = read_csv(OUT / "sequence_of_loss_timeline.csv")
    contradictions = read_csv(OUT / "contradictions.csv")
    gates = read_csv(OUT / "writing_gate_audit.csv")
    evidence = read_csv(OUT / "normalized_evidence_lineage.csv")
    provenance = read_csv(OUT / "normalized_provenance.csv")
    patch_audit = read_csv(OUT / "v2_patch_audit.csv")

    check(
        "immutable input manifests",
        all(sha256(path) == expected for path, expected in EXPECTED_MANIFEST_HASHES.items()),
        "Chapter 17 source, Chapter 16 author, and fresh Chapter 16 acceptance manifests match sealed hashes",
    )
    review = json.loads(
        (CH16_REVIEW / "lane_manifest.json").read_text(encoding="utf-8")
    )
    check(
        "Chapter 16 acceptance boundary",
        review["intrinsic_packet"] == "ACCEPT_INTRINSIC_PACKET"
        and review["verdict"] == "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS"
        and review["comparative_publication"] == "BLOCKED"
        and review["canonical_integration"] == "NOT_AUTHORIZED",
        "intrinsic acceptance is not promoted into comparative or canonical clearance",
    )

    check(
        "normalized population counts",
        (
            len(crosswalk),
            len(counterparties),
            len(scenarios_xw),
            len(objects),
            len(failures_xw),
            len(claims_xw),
            len(sources_xw),
            len(project_identity),
        )
        == (8, 10, 9, 16, 8, 13, 13, 1),
        "8 core; 10 parties; 9 scenarios; 16 objects; 8 failures; 13 claims; 13 sources; 1 project identity",
    )
    atomic_ok, atomic_failures = scalar_foreign_keys_are_atomic(
        [
            ("chapter16_stress_crosswalk.csv", crosswalk),
            ("crosswalk_counterparty_junction.csv", counterparties),
            ("crosswalk_scenario_junction.csv", scenarios_xw),
            ("crosswalk_propagation_object_junction.csv", objects),
            ("crosswalk_failure_mechanism_junction.csv", failures_xw),
            ("crosswalk_claim_junction.csv", claims_xw),
            ("crosswalk_claim_source_junction.csv", sources_xw),
            ("atomic_project_identity.csv", project_identity),
        ]
    )
    check(
        "transport composites removed from scalar foreign keys",
        atomic_ok,
        "failures=" + ",".join(atomic_failures) if atomic_failures else "no pipe-delimited scalar foreign keys",
    )

    original = {
        row["crosswalk_id"]: row
        for row in read_csv(SOURCE / "chapter16_stress_crosswalk.csv")
    }
    failures = {
        row["failure_id"]: row
        for row in read_csv(CH16 / "single_point_failure_register.csv")
    }
    accepted_parties = read_csv(CH16 / "single_point_failure_parties.csv")
    check(
        "eight scalar core joins",
        {row["crosswalk_id"] for row in crosswalk} == set(original)
        and all(row["ch16_failure_id"] in failures for row in crosswalk)
        and all(
            row["dependent_reference_id"] != "PPRJ-160001"
            for row in crosswalk
        ),
        "all original crosswalk identities retained and former PPRJ-160001 link replaced",
    )
    expected_party_pairs = {
        (row["failure_id"], row["party_id"]) for row in accepted_parties
    }
    actual_party_pairs = {
        (row["failure_id"], row["counterparty_id"]) for row in counterparties
    }
    check(
        "counterparty identities reperform accepted Chapter 16 parties",
        expected_party_pairs == actual_party_pairs
        and sum(row["exact_legal_entity_resolved"] == "0" for row in counterparties)
        == 5
        and sum(row["exact_legal_entity_resolved"] == "1" for row in counterparties)
        == 5,
        "five exact/canonical records and five explicitly unresolved proxy, group, customer, or placeholder records",
    )

    scenario_ids = {row["scenario_id"] for row in scenarios}
    expected_scenario_pairs = {
        (crosswalk_id, scenario_id)
        for crosswalk_id, row in original.items()
        for scenario_id in row["ch17_scenario_id"].split("|")
    }
    actual_scenario_pairs = {
        (row["crosswalk_id"], row["scenario_id"]) for row in scenarios_xw
    }
    check(
        "scenario junction exactness",
        expected_scenario_pairs == actual_scenario_pairs
        and all(row["scenario_id"] in scenario_ids for row in scenarios_xw),
        "all nine scenario links reconcile to existing Chapter 17 scenarios",
    )

    edge_ids = {row["edge_id"] for row in edges}
    mechanic_ids = {row["mechanism_id"] for row in mechanics}
    resolved_objects = [row for row in objects if row["ch17_object_id"]]
    unresolved_objects = [row for row in objects if not row["ch17_object_id"]]
    check(
        "propagation object junction exactness",
        len(resolved_objects) == 15
        and len(unresolved_objects) == 1
        and all(
            (
                row["object_type"] == "propagation_edge"
                and row["ch17_object_id"] in edge_ids
            )
            or (
                row["object_type"] == "observed_mechanism"
                and row["ch17_object_id"] in mechanic_ids
            )
            for row in resolved_objects
        )
        and unresolved_objects[0]["crosswalk_id"] == "L3-XW17-0002"
        and unresolved_objects[0]["unresolved_object_reference"]
        == "L3-UNRES-XWOBJ17-0001",
        "15 exact Chapter 17 objects and one typed Anthropic trigger unresolved without a guessed foreign key",
    )

    project = project_identity[0]
    xw4 = find_one(crosswalk, "crosswalk_id", "L3-XW17-0004")
    finance_case = find_one(
        read_csv(AI_ROOT / "data/project_finance_cases.csv"),
        "project_finance_case_id",
        "PFC-040004",
    )
    check(
        "CCAC VIII DDTL 4.0 atomic identity",
        project["rebase_project_id"] == "PPRJ-17R-0001"
        and project["prior_provisional_project_id"] == "PPRJ-160001"
        and project["canonical_project_finance_case_id"] == "PFC-040004"
        and project["canonical_project_id"] == ""
        and finance_case["project_id"] == ""
        and xw4["dependent_reference_id"] == "PPRJ-17R-0001"
        and project["integration_eligible"] == "0",
        "one multi-site program identity exists while canonical project assignment remains null and blocked",
    )

    claim_ids = {row["claim_id"] for row in read_csv(AI_ROOT / "data/claims.csv")}
    evidence_pairs = {
        (row["claim_id"], row["source_id"])
        for row in read_csv(AI_ROOT / "data/evidence_links.csv")
    }
    source_rows = {
        row["source_id"]: row
        for row in read_csv(AI_ROOT / "data/source_register.csv")
    }
    claim_pairs = {(row["crosswalk_id"], row["claim_id"]) for row in claims_xw}
    source_pairs = {(row["claim_id"], row["source_id"]) for row in sources_xw}
    check(
        "claim and source lineage",
        len(claim_pairs) == 13
        and len(source_pairs) == 13
        and all(row["claim_id"] in claim_ids for row in claims_xw)
        and source_pairs.issubset(evidence_pairs)
        and all(
            row["source_id"] in source_rows
            and row["source_url"] == source_rows[row["source_id"]]["url"]
            for row in sources_xw
        ),
        "13 accepted failure claims each resolve through one exact canonical evidence link",
    )

    check(
        "original-to-normalized map",
        len(row_map) == 78
        and {row["original_crosswalk_id"] for row in row_map} == set(original)
        and all(row["information_preserved"] == "1" for row in row_map),
        "77 normalized core/junction rows plus one CCAC identity mapping are traceable",
    )
    check(
        "field-loss audit",
        len(field_audit) == 16
        and all(
            row["values_preserved"] == "1"
            and row["semantic_loss"] == "0"
            and row["audit_result"] == "PASS"
            for row in field_audit
        ),
        "all sixteen original crosswalk fields have explicit lossless destinations",
    )
    check(
        "identity and claim-source audits",
        len(identity_audit) == 19
        and len(lineage_audit) == 13
        and all(row["audit_result"] == "PASS" for row in identity_audit)
        and all(row["audit_result"] == "PASS" for row in lineage_audit),
        "dependent, counterparty, project, claim, and source identities are independently enumerated",
    )

    fail_closed_tables = [
        crosswalk,
        counterparties,
        scenarios_xw,
        objects,
        failures_xw,
        claims_xw,
        sources_xw,
        project_identity,
    ]
    check(
        "all rebased rows fail closed",
        all(
            row["prose_eligible"] == "0"
            and row["integration_eligible"] == "0"
            for table in fail_closed_tables
            for row in table
        ),
        "all 78 rebase rows remain ineligible for prose and integration",
    )
    hold_ids = {row["hold_id"] for row in hold_audit}
    check(
        "external holds propagated",
        len(hold_audit) == 56
        and len(hold_ids) == 7
        and all(
            row["propagated"] == "1"
            and row["prose_eligible"] == "0"
            and row["integration_eligible"] == "0"
            and row["audit_result"] == "PASS"
            for row in hold_audit
        ),
        "seven Chapter 16 or targeted-review holds attach to every crosswalk",
    )

    source_manifest = json.loads(
        (SOURCE / "lane_manifest.json").read_text(encoding="utf-8")
    )
    copied_csv_entries = [
        artifact
        for artifact in source_manifest["files"]
        if artifact["path"].endswith(".csv")
        and artifact["path"] != "chapter16_stress_crosswalk.csv"
    ]
    check(
        "Chapter 17 intrinsic artifacts byte preserved",
        all(
            sha256(OUT / artifact["path"]) == artifact["sha256"]
            for artifact in copied_csv_entries
        ),
        f"byte-identical intrinsic CSVs={len(copied_csv_entries)}",
    )
    check(
        "intrinsic surface populations",
        (
            len(scenarios),
            len(capex),
            len(mechanics),
            len(edges),
            len(loops),
            len(timeline),
            len(evidence),
            len(provenance),
        )
        == (20, 5, 27, 54, 9, 50, 358, 61)
        and {
            f"L3-EDGE17-{index:04d}" for index in range(1, 53)
        }.issubset(edge_ids)
        and {"L3-EDGE17-0053", "L3-EDGE17-0054"}.issubset(edge_ids)
        and {
            f"L3-TIME17-{index:04d}" for index in range(1, 50)
        }.issubset({row["timeline_id"] for row in timeline})
        and "L3-TIME17-0050" in {
            row["timeline_id"] for row in timeline
        },
        "20 scenarios; 5 capex; 27 mechanics; 52 directed edges plus 2 repair bridges; 9 loops; 49 phases plus 1 blocked phase; 358 lineage; 61 provenance",
    )
    check(
        "all rebuilt-v2 repairs remain applied",
        len(patch_audit) == 35
        and all(row["application_status"] == "APPLIED_V2" for row in patch_audit),
        "35 of 35 accepted Chapter 17 repairs remain present",
    )

    ares_values = (
        find_one(scenarios, "scenario_id", "L3-SCN17-0016")[
            "structured_lag"
        ],
        find_one(mechanics, "mechanism_id", "L3-MECH17-0014")["lag_unit"],
        find_one(edges, "edge_id", "L3-EDGE17-0033")["lag_unit"],
        find_one(edges, "edge_id", "L3-EDGE17-0033")["trigger"],
        find_one(timeline, "timeline_id", "L3-TIME17-0028")[
            "timing_unit"
        ],
        find_one(loops, "loop_id", "L3-LOOP17-0004")["observed_anchor"],
        find_one(
            contradictions, "contradiction_id", "L3-CON17-0008"
        )["proposition_a"],
        find_one(crosswalk, "crosswalk_id", "L3-XW17-0006")[
            "ch17_disposition"
        ],
    )
    check(
        "conditional Ares cure preserved",
        all(
            "day_5" in value
            and "day_10" in value
            and "cure_evidence" in value
            for value in ares_values
        ),
        "day 10 applies only through the timely satisfactory cure-evidence branch on all eight surfaces",
    )

    rb_phase7 = find_one(timeline, "timeline_id", "L3-TIME17-0050")
    rb_edge7 = find_one(edges, "edge_id", "L3-EDGE17-0053")
    check(
        "River Bend blocked phase 7 preserved",
        rb_phase7["phase_order"] == "7"
        and rb_phase7["status"] == "blocked_execution_unobserved"
        and rb_edge7["sequence_order"] == "7"
        and rb_edge7["edge_status"] == "blocked_execution_unobserved"
        and "actual_note_reduction" in rb_phase7["cash_or_balance_sheet_effect"]
        and find_one(timeline, "timeline_id", "L3-TIME17-0012")[
            "phase_order"
        ]
        == "8",
        "offer execution, tender settlement, actual note reduction, and residual loss remain separate",
    )

    helios_scenario = find_one(
        scenarios, "scenario_id", "L3-SCN17-0011"
    )
    helios_mechanic = find_one(
        mechanics, "mechanism_id", "L3-MECH17-0015"
    )
    helios_edge = find_one(edges, "edge_id", "L3-EDGE17-0031")
    check(
        "Helios maturity remains separate from response lag",
        helios_scenario["structured_lag"] == "unknown"
        and "2028-08-15" in helios_scenario["native_period_end"]
        and (
            helios_mechanic["lag_unit"],
            helios_mechanic["lag_evidence_state"],
        )
        == ("not_a_response_lag", "not_applicable")
        and (
            helios_edge["lag_unit"],
            helios_edge["lag_evidence_state"],
        )
        == ("unknown", "unresolved"),
        "2028 maturity is a contract endpoint, not an estimated transmission lag",
    )

    first_loss_tables = (
        scenarios,
        mechanics,
        edges,
        loops,
        timeline,
        crosswalk,
        capex,
    )
    first_loss_values = [
        row.get("evidenced_first_loss", row.get("first_loss_bearer", ""))
        for table in first_loss_tables
        for row in table
    ]
    accounting_holders = [
        row.get("accounting_loss_holder", "")
        for table in (scenarios, mechanics, edges, timeline)
        for row in table
        if row.get("accounting_loss_holder")
    ]
    check(
        "NVIDIA accounting charge and economic first loss remain separate",
        len(first_loss_values) == 173
        and set(first_loss_values) == {"not_determined"}
        and accounting_holders == ["NVIDIA"] * 4,
        "four NVIDIA accounting-holder rows coexist with 173 undetermined economic first-loss fields",
    )

    capex_ok = all(
        Decimal(row["shocked_value_usd"])
        == Decimal(row["base_value_usd"]) * Decimal("0.8")
        and Decimal(row["delta_value_usd"])
        == Decimal(row["shocked_value_usd"])
        - Decimal(row["base_value_usd"])
        and row["additive_eligible"] == "0"
        for row in capex
    )
    check(
        "company-native capex arithmetic remains nonadditive",
        capex_ok and len({row["overlap_identity"] for row in capex}) == 5,
        "five exact 20 percent counterfactuals retain native periods and cannot be summed",
    )

    loop_ok = True
    for loop in loops:
        signs = loop["edge_sign_sequence"].split(";")
        if loop["loop_closure_state"].startswith("closed"):
            product = 1
            for sign in signs:
                product *= 1 if sign == "positive" else -1
            expected = "reinforcing" if product > 0 else "balancing"
            loop_ok &= loop["loop_polarity"] == expected
        else:
            loop_ok &= loop["loop_polarity"] == "undetermined"
    check(
        "feedback polarity and empirical boundary preserved",
        loop_ok
        and all(row["probability"] == "not_assigned" for row in loops)
        and all(row["model_status"] == "blocked" for row in loops),
        "closed sign products recompute; open chains remain undetermined; probabilities and empirical gains remain unassigned",
    )

    unresolved_lags_ok = all(
        not row.get("lag_low") and not row.get("lag_high")
        for table in (mechanics, edges, loops)
        for row in table
        if row.get("lag_evidence_state") in {"unresolved", "not_applicable"}
    )
    no_probability_gate = find_one(
        gates, "gate_id", "L3-WG17-0008"
    )
    check(
        "unsupported probabilities expected losses and lags remain unassigned",
        unresolved_lags_ok
        and no_probability_gate["status"] == "pass"
        and "none_or_not_assigned"
        in no_probability_gate["evidence_or_required_revision"]
        and all(
            "expected_loss" not in field
            for table in (scenarios, edges, loops)
            for row in table
            for field in row
        ),
        "no calibrated probability, expected loss, empirical loop gain, or unsupported lag is introduced",
    )

    check(
        "audit artifacts pass",
        len(intrinsic_audit) == 15
        and all(row["audit_result"] == "PASS" for row in intrinsic_audit),
        "eight byte/count controls and seven meaning controls pass",
    )
    no_write = json.loads(
        (OUT / "canonical_no_write_audit.json").read_text(encoding="utf-8")
    )
    check(
        "canonical database unchanged",
        no_write["status"] == "PASS"
        and not no_write["modified"]
        and no_write["sha256_before"] == no_write["sha256_after"]
        == sha256(CANONICAL_DB),
        f"research.db sha256={sha256(CANONICAL_DB)}",
    )

    failures_out = [
        item["check"] for item in checks if item["status"] != "PASS"
    ]
    payload = {
        "validator": "validate_atomic_rebase.py",
        "status": "PASS" if not failures_out else "FAIL",
        "checks": checks,
        "failures": failures_out,
        "intrinsic_chapter17_verdict": (
            "PRESERVED" if not failures_out else "REVISE"
        ),
        "crosswalk_author_verdict": (
            "STRUCTURAL_REBASE_COMPLETE_PENDING_FRESH_NONAUTHOR_REVIEW"
            if not failures_out
            else "REVISE"
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "publication": "BLOCKED",
        "author_self_acceptance": False,
    }
    (OUT / "validation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Chapter 17 atomic rebase validation",
        "",
        f"**Status:** {payload['status']}",
        "**Intrinsic Chapter 17:** preserved",
        "**Crosswalk:** pending fresh nonauthor review",
        "**Canonical integration:** not authorized",
        "**Publication:** blocked",
        "",
        "## Meaning-based controls",
        "",
    ]
    report_lines.extend(
        f"- **{item['status']} — {item['check']}:** {item['detail']}"
        for item in checks
    )
    report_lines.extend(
        [
            "",
            "The author validator does not constitute acceptance. A fresh reviewer must",
            "independently reconstruct all eight joins and reperform the intrinsic controls.",
        ]
    )
    (OUT / "validation_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    if failures_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(checks),
                "crosswalk": "PENDING_FRESH_NONAUTHOR_REVIEW",
                "canonical_integration": "NOT_AUTHORIZED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
