from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import argparse
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
AUTHOR = (
    PROJECT_ROOT
    / "staging/lane_1_private_investment_earnings_corrected"
)
REVIEW = (
    PROJECT_ROOT
    / "staging/lane_3_verify_private_investment_earnings_corrected"
)
OUTPUT = PROJECT_ROOT / "integration/private_investment_candidate"

EXPECTED_AUTHOR_MANIFEST_SHA = (
    "609108aca41f6d717cddce064d8130bebce592e7cfd559ceb157d415e76d258b"
)
EXPECTED_REVIEW_MANIFEST_SHA = (
    "8e48b9470b2ed1ea7e9315027b5d17fa3cf38e83724a16ca40970d801a98ed48"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV width mismatch: {path}")
    return rows


def write_csv(
    path: Path, columns: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column)
                    for column in columns
                }
            )


def parse_amount(value: str) -> tuple[str, str, str]:
    value = value.strip()
    if not value:
        return "", "unavailable", ""
    precision = "approximate" if value.startswith("approximately_") else "exact"
    match = re.match(r"^(?:approximately_)?(-?\d+(?:\.\d+)?)", value)
    if match is None:
        return "", "not_numeric", value
    number = match.group(1)
    suffix = value[match.end() :].lstrip("_")
    return number, precision, suffix


PERIODS = {
    "2026Q1": ("2026-01-01", "2026-03-31", "calendar_quarter_exact"),
    "FY2027Q1": ("", "2026-04-26", "issuer_quarter_end_exact_start_unresolved"),
    "9M_ended_2026_03_31": (
        "2025-07-01",
        "2026-03-31",
        "issuer_nine_month_period_exact",
    ),
    "3M_ended_2026_03_31": (
        "2026-01-01",
        "2026-03-31",
        "issuer_quarter_exact",
    ),
    "9M_ended_2026_02_28": (
        "",
        "2026-02-28",
        "issuer_nine_month_end_exact_start_unresolved",
    ),
    "2026_02_28": ("", "2026-02-28", "point_in_time"),
    "9M_ended_2025_09_30": (
        "2025-01-01",
        "2025-09-30",
        "calendar_nine_month_period_exact",
    ),
}

METHODS = {
    "measurement_alternative_observable_price_change": "PIM-01",
    "available_for_sale_fair_value_through_oci": "PIM-02",
    "available_for_sale_reclassification_to_earnings": "PIM-03",
    "equity_method_dilution_gain": "PIM-04",
    "equity_method_share_of_income_loss": "PIM-05",
    "fair_value_through_earnings": "PIM-06",
    "impairment": "PIM-07",
    "consolidated_subsidiary": "PIM-08",
    "equity_method_HLBV_income_loss": "PIM-09",
    "realized_private_investment_disposal_gain": "PIM-10",
    "ASC825_fair_value_option_equity_method_investment": "PIM-11",
}

MIXED_METHODS = {
    "analytical_sum_of_PIE_0001_and_PIE_0002": (
        "",
        "analytical_sum_PIM-03_and_PIM-01_not_separate_earnings_event",
    ),
    "net_equity_method_OpenAI_result_primarily_recapitalization_dilution": (
        "",
        "mixed_net_result_primarily_PIM-04_with_PIM-09_components_unallocated",
    ),
    "net_equity_method_OpenAI_result_current_quarter": (
        "",
        "mixed_net_equity_method_result_components_unallocated",
    ),
    "equity_method_and_other_nonmarketable": (
        "",
        "mixed_equity_method_and_measurement_alternative",
    ),
    "measurement_alternative_and_equity_method": (
        "",
        "mixed_PIM-01_and_equity_method",
    ),
    "AFS_reclassification_plus_measurement_alternative_mark": (
        "",
        "mixed_PIM-03_and_PIM-01",
    ),
    "net_OpenAI_result_primarily_recapitalization_dilution": (
        "",
        "mixed_net_result_primarily_PIM-04_with_PIM-09_components_unallocated",
    ),
    "various": ("", "mixed_unallocated_methods"),
}


def period_fields(label: str) -> dict[str, str]:
    start, end, status = PERIODS.get(
        label, ("", "", "unmapped_period_requires_resolution")
    )
    return {
        "reporting_period_label": label,
        "reporting_period_start": start,
        "reporting_period_end": end,
        "reporting_period_date_status": status,
    }


def investee_fields(scope: str) -> dict[str, str]:
    if scope == "ENT-ANTHROPIC":
        entity = "ENT-ANTHROPIC"
    elif "ENT-ANTHROPIC" in scope or "Anthropic" in scope:
        entity = "ENT-ANTHROPIC"
    elif scope == "ENT-OAI-GROUP":
        entity = "ENT-OAI-GROUP"
    elif scope == "ENT-SPACEX":
        entity = "ENT-SPACEX"
    elif scope == "primarily_TikTok_USDS_JV":
        entity = "ENT-TIKTOK-USDS"
    else:
        entity = ""
    return {
        "investee_entity_id": entity,
        "investee_scope_description": scope,
    }


def method_fields(label: str) -> dict[str, str]:
    if label in METHODS:
        return {
            "accounting_method_id": METHODS[label],
            "accounting_method_scope": label,
        }
    method_id, scope = MIXED_METHODS.get(
        label, ("", f"unmapped_method:{label}")
    )
    return {
        "accounting_method_id": method_id,
        "accounting_method_scope": scope,
    }


def attribution_state(value: str) -> str:
    mapping = {
        "named": "exact_named",
        "primarily_named": "primarily_named",
        "primarily_named_not_fully_allocated": "primarily_named_with_residual",
        "portfolio_aggregate_only": "portfolio_aggregate",
        "portfolio_aggregate": "portfolio_aggregate",
        "mixed_portfolio": "mixed_portfolio_unallocated",
        "Scale_cost_named_portfolio_result_unallocated": (
            "named_cost_portfolio_result_unallocated"
        ),
        "named_components_not_fully_disclosed": "named_components_unallocated",
    }
    return mapping.get(value, value)


def require_acceptance() -> dict[str, Any]:
    author_manifest = AUTHOR / "lane_manifest.json"
    review_manifest = REVIEW / "lane_manifest.json"
    if sha256(author_manifest) != EXPECTED_AUTHOR_MANIFEST_SHA:
        raise ValueError("accepted author manifest hash changed")
    if sha256(review_manifest) != EXPECTED_REVIEW_MANIFEST_SHA:
        raise ValueError("accepted review manifest hash changed")
    results = json.loads((REVIEW / "validation_results.json").read_text())
    expected = {
        "status": "PASS",
        "verdict": "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS",
        "canonical_integration_authorization": (
            "AUTHORIZED_FOR_PRIMARY_AGENT_BOUNDED_REBASE_ONLY_WITH_RETAINED_HOLDS"
        ),
        "new_exact_patches": 0,
    }
    for key, value in expected.items():
        if results.get(key) != value:
            raise ValueError(f"review acceptance mismatch for {key}")
    return results


def build_methods() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "accounting_method_taxonomy.csv"):
        rows.append(
            {
                "method_id": source["method_id"],
                "accounting_method": source["accounting_method"],
                "governing_basis": (
                    "ASC_825_fair_value_option"
                    if source["method_id"] == "PIM-11"
                    else "accounting_method_registry_primary_filing_based"
                ),
                "recognition_trigger": source["trigger"],
                "income_statement_or_oci_location": source[
                    "income_statement_location"
                ],
                "cash_at_recognition_default": source["cash_at_recognition"],
                "carrying_value_effect": source["carrying_value_effect"],
                "required_separation": source["required_separation"],
                "example": source["example"],
                "provenance_class": "model_specification_from_accepted_issuer_cases",
                "notes": (
                    "Example illustrates the method and is not part of its definition."
                ),
            }
        )
    return rows


def build_earnings() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "private_investment_earnings_ledger.csv"):
        amount, precision, amount_suffix = parse_amount(
            source["pretax_gain_loss_usd_m"]
        )
        pretax, pretax_precision, _ = parse_amount(
            source["reported_pretax_income_usd_m"]
        )
        ratio, ratio_precision, _ = parse_amount(
            source["gain_share_of_pretax_income_pct"]
        )
        cash = source["cash_received_at_recognition"]
        cash_amount = ""
        cash_precision = "unavailable"
        cash_state = cash
        cash_match = re.match(r"^yes_(\d+)_proceeds$", cash)
        if cash_match:
            cash_amount = cash_match.group(1)
            cash_precision = "approximate"
            cash_state = "cash_proceeds_observed"
        row = {
            "private_investment_earnings_id": source["record_id"],
            "investor_entity_id": source["investor_entity_id"],
            **period_fields(source["period"]),
            **investee_fields(source["investee_scope"]),
            "instrument_scope": source["instrument_scope"],
            **method_fields(source["accounting_method"]),
            "recognition_location": source["recognized_location"],
            "reported_line_item": source["recognized_location"],
            "pretax_gain_loss_amount": amount,
            "currency": "USD",
            "unit_scale": "millions",
            "amount_precision": precision,
            "amount_unparsed_suffix": amount_suffix,
            "recognition_state": source["realized_status"],
            "cash_received_amount": cash_amount,
            "cash_received_state": cash_state,
            "cash_amount_precision": cash_precision,
            "reported_pretax_income_amount": pretax,
            "reported_pretax_income_precision": pretax_precision,
            "gain_share_of_pretax_income_pct": ratio,
            "ratio_precision": ratio_precision,
            "investee_attribution_state": attribution_state(
                source["investee_attribution"]
            ),
            "evidence_grade": source["evidence_grade"],
            "source_id": source["source_id"],
            "source_locator": source["source_locator"],
            "source_claim_id": source["canonical_claim_id"],
            "nonadditive_overlap_group": (
                "AMZN_2026Q1_SELECTED_PRIVATE_GAIN"
                if source["record_id"] in {"PIE-0001", "PIE-0002", "PIE-0003"}
                else ""
            ),
            "additive_eligible": (
                0 if source["record_id"] == "PIE-0003" else 1
            ),
            "analyst_derived": (
                1 if source["status"] == "derived_reconciliation" else 0
            ),
            "integration_status": (
                "accepted_bounded_source_archive_hold"
            ),
            "notes": source["notes"],
        }
        rows.append(row)
    for sequence, child in enumerate(("PIE-0001", "PIE-0002"), start=1):
        components.append(
            {
                "parent_earnings_id": "PIE-0003",
                "component_earnings_id": child,
                "component_order": sequence,
                "relationship": "analytical_sum_component",
                "additive_within_parent": 1,
                "notes": (
                    "Parent is a reconciliation only and is not an additional gain."
                ),
            }
        )
    return rows, components


def build_rollforwards() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    headers: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    component_columns = {
        "additions_usd_m": "cash_or_unclassified_addition",
        "unrealized_gain_usd_m": "unrealized_gain_or_stock_adjustment",
        "reclassification_usd_m": "reclassification",
        "impairment_or_loss_usd_m": "impairment_or_loss",
        "other_or_unreconciled_usd_m": "other_or_unresolved_residual",
    }
    for source in read_csv(AUTHOR / "carrying_value_rollforward.csv"):
        beginning, begin_precision, _ = parse_amount(
            source["beginning_carrying_value_usd_m"]
        )
        ending, end_precision, _ = parse_amount(
            source["ending_carrying_value_usd_m"]
        )
        headers.append(
            {
                "rollforward_id": source["rollforward_id"],
                "investor_entity_id": source["investor_entity_id"],
                **period_fields(source["period"]),
                **investee_fields(source["investee_scope"]),
                "instrument_scope": source["instrument_scope"],
                "beginning_carrying_value_amount": beginning,
                "ending_carrying_value_amount": ending,
                "currency": "USD",
                "unit_scale": "millions",
                "beginning_amount_precision": begin_precision,
                "ending_amount_precision": end_precision,
                "reconciliation_state": source["reconciliation_status"],
                "source_id": source["source_id"],
                "source_locator": source["source_locator"],
                "integration_status": "accepted_bounded_source_archive_hold",
                "notes": source["notes"],
            }
        )
        component_number = 0
        for staging_field, component_type in component_columns.items():
            if not source[staging_field]:
                continue
            component_number += 1
            amount, precision, suffix = parse_amount(source[staging_field])
            components.append(
                {
                    "rollforward_component_id": (
                        f"{source['rollforward_id']}-C{component_number:02d}"
                    ),
                    "rollforward_id": source["rollforward_id"],
                    "component_type": component_type,
                    "amount": amount,
                    "currency": "USD",
                    "unit_scale": "millions",
                    "amount_precision": precision,
                    "amount_unparsed_suffix": suffix,
                    "cash_state": (
                        "not_determined"
                        if staging_field == "additions_usd_m"
                        else "noncash_or_accounting_stock"
                    ),
                    "income_aoci_balance_sheet_class": (
                        "AOCI_net_of_tax"
                        if source["rollforward_id"] == "PIR-0003"
                        else "carrying_value_stock_component"
                    ),
                    "additive_eligible": 1,
                    "overlap_group": source["rollforward_id"],
                    "notes": source["notes"],
                }
            )
    return headers, components


def build_bridges() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bridges: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    selected_links = {
        "EQB-0001": ["PIE-0001", "PIE-0002"],
        "EQB-0002": ["PIE-0004"],
        "EQB-0003": ["PIE-0005"],
        "EQB-0004": ["PIE-0007"],
    }
    for source in read_csv(AUTHOR / "earnings_quality_bridge.csv"):
        parsed: dict[str, tuple[str, str, str]] = {}
        for field in (
            "reported_operating_income_usd_m",
            "reported_pretax_income_usd_m",
            "identified_private_investment_gain_usd_m",
            "arithmetic_pretax_residual_after_selected_gain_usd_m",
            "private_gain_share_pretax_pct",
            "cash_flow_reversal_usd_m",
        ):
            parsed[field] = parse_amount(source[field])
        cash_amount, cash_precision, cash_suffix = parsed[
            "cash_flow_reversal_usd_m"
        ]
        bridges.append(
            {
                "bridge_id": source["bridge_id"],
                "investor_entity_id": source["investor_entity_id"],
                **period_fields(source["period"]),
                "reported_operating_income_amount": parsed[
                    "reported_operating_income_usd_m"
                ][0],
                "reported_pretax_income_amount": parsed[
                    "reported_pretax_income_usd_m"
                ][0],
                "identified_private_investment_gain_amount": parsed[
                    "identified_private_investment_gain_usd_m"
                ][0],
                "arithmetic_pretax_residual_amount": parsed[
                    "arithmetic_pretax_residual_after_selected_gain_usd_m"
                ][0],
                "private_gain_share_pretax_pct": parsed[
                    "private_gain_share_pretax_pct"
                ][0],
                "cash_flow_reversal_amount": cash_amount,
                "cash_flow_reversal_precision": cash_precision,
                "cash_flow_reversal_perimeter": cash_suffix,
                "currency": "USD",
                "unit_scale": "millions",
                "selected_gain_precision": parsed[
                    "identified_private_investment_gain_usd_m"
                ][1],
                "residual_precision": parsed[
                    "arithmetic_pretax_residual_after_selected_gain_usd_m"
                ][1],
                "ratio_precision": parsed[
                    "private_gain_share_pretax_pct"
                ][1],
                "tax_effect_state": source["tax_effect_status"],
                "analyst_derived_state": source["model_status"],
                "integration_status": "accepted_bounded_source_archive_hold",
                "notes": source["notes"],
            }
        )
        for earnings_id in selected_links[source["bridge_id"]]:
            links.append(
                {
                    "bridge_id": source["bridge_id"],
                    "private_investment_earnings_id": earnings_id,
                    "component_role": "selected_gain_component",
                }
            )
    return bridges, links


def build_population() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "population_evidence.csv"):
        values = {}
        for field in (
            "carrying_value_usd_m",
            "cash_proceeds_usd_m",
            "cumulative_upward_adjustments_usd_m",
            "current_period_private_gain_loss_usd_m",
            "gain_share_of_pretax_pct",
            "named_position_cost_usd_m",
            "reported_pretax_income_usd_m",
        ):
            values[field] = parse_amount(source[field])
        rows.append(
            {
                "population_observation_id": source["population_id"],
                "investor_entity_id": source["investor_entity_id"],
                "issuer_label": source["issuer"],
                **period_fields(source["period"]),
                **investee_fields(source["investee_scope"]),
                **method_fields(source["accounting_method"]),
                "observation_class": source["observation_class"],
                "material_private_gain_state": source[
                    "material_current_private_gain_identified"
                ],
                "operating_nonoperating_state": source[
                    "operating_nonoperating_state"
                ],
                "carrying_value_amount": values["carrying_value_usd_m"][0],
                "cash_proceeds_amount": values["cash_proceeds_usd_m"][0],
                "cumulative_upward_adjustments_amount": values[
                    "cumulative_upward_adjustments_usd_m"
                ][0],
                "current_period_private_gain_loss_amount": values[
                    "current_period_private_gain_loss_usd_m"
                ][0],
                "gain_share_of_pretax_pct": values[
                    "gain_share_of_pretax_pct"
                ][0],
                "named_position_cost_amount": values[
                    "named_position_cost_usd_m"
                ][0],
                "reported_pretax_income_amount": values[
                    "reported_pretax_income_usd_m"
                ][0],
                "currency": "USD",
                "unit_scale": "millions",
                "amount_precision_state": source["amount_precision"],
                "investee_attribution_state": attribution_state(
                    source["attribution_state"]
                ),
                "source_id": source["source_id"],
                "source_url": source["source_url"],
                "integration_status": "accepted_bounded_source_archive_hold",
                "notes": source["notes"],
            }
        )
    return rows


CLAIM_IDS = {
    "PCLM-PIE-001": "CLM-070035",
    "PCLM-PIE-002": "CLM-070036",
    "PCLM-PIE-003": "CLM-070037",
    "PCLM-PIE-004": "CLM-110003",
    "PCLM-PIE-005": "CLM-070038",
}


def build_claims() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "proposed_claims.csv"):
        claim_id = CLAIM_IDS[source["proposed_claim_id"]]
        claims.append(
            {
                "claim_id": claim_id,
                "chapter_id": source["chapter_id"],
                "claim_text": source["claim_text"],
                "evidence_grade": source["evidence_grade"],
                "confidence": source["confidence"],
                "claim_type": source["claim_type"],
                "accounting_status": source["accounting_status"],
                "economic_status": source["economic_status"],
                "quantification_status": source["quantification_status"],
                "effective_date": source["effective_date"],
                "supersedes_claim_id": "",
                "review_status": "accepted_bounded_source_archive_hold",
                "reviewer_notes": (
                    f"Staged as {source['proposed_claim_id']}; fresh review permits "
                    "bounded rebase, but final publication requires source archives."
                ),
            }
        )
        for source_id in source["source_id"].split("|"):
            links.append(
                {
                    "claim_id": claim_id,
                    "source_id": source_id,
                    "locator": source["locator"],
                    "support_type": (
                        "direct"
                        if source["evidence_grade"] == "A"
                        else "cross_source_reconstruction"
                    ),
                    "excerpt_note": source["claim_text"],
                }
            )
    return claims, links


UNKNOWN_CLASSES = {
    "UNK-PIE-001": "investee_allocation_gap",
    "UNK-PIE-002": "investee_allocation_gap",
    "UNK-PIE-003": "tax_allocation_gap",
    "UNK-PIE-004": "component_and_tax_allocation_gap",
    "UNK-PIE-005": "population_reproduction_gap",
    "UNK-PIE-006": "consolidation_perimeter_gap",
    "UNK-PIE-007": "valuation_trigger_gap",
    "UNK-PIE-008": "cash_realization_gap",
    "UNK-PIE-009": "conversion_carrying_value_gap",
    "UNK-PIE-010": "period_gain_attribution_gap",
    "UNK-PIE-011": "prospective_monitoring_gap",
}


def build_unknowns() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "proposed_unknowns.csv"):
        rows.append(
            {
                "unknown_id": source["unknown_id"],
                "chapter_id": "CH07",
                "question": source["question"],
                "materiality": source["materiality"],
                "unknown_class": UNKNOWN_CLASSES[source["unknown_id"]],
                "downstream_blocking_chapters": source["blocked_outputs"],
                "resolution_trigger": source["most_likely_resolution_source"],
                "suspected_sources": source["most_likely_resolution_source"],
                "resolution_plan": (
                    "Acquire, archive and reconcile the named primary disclosure "
                    "against the accepted private-investment ledgers."
                ),
                "status": source["status"],
                "last_reviewed": "2026-07-26",
            }
        )
    return rows


HANDOFF_STATUS = {
    "HOF-PIE-001": "bounded",
    "HOF-PIE-002": "bounded",
    "HOF-PIE-003": "bounded",
    "HOF-PIE-004": "bounded",
    "HOF-PIE-005": "bounded",
    "HOF-PIE-006": "bounded",
    "HOF-PIE-007": "bounded",
    "HOF-PIE-008": "bounded",
    "HOF-PIE-009": "blocked",
    "HOF-PIE-010": "bounded",
    "HOF-PIE-011": "bounded",
}


def build_handoffs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(AUTHOR / "proposed_handoffs.csv"):
        status = HANDOFF_STATUS[source["handoff_id"]]
        rows.append(
            {
                "handoff_id": source["handoff_id"],
                "upstream_chapter_id": "CH07",
                "downstream_chapter_id": source["to_chapter"],
                "dataset_or_ledger": "private_investment_earnings",
                "analytical_input": source["field_or_conclusion"],
                "required_fields": source["required_use"],
                "readiness_status": status,
                "missing_evidence": source["blocking_unknowns"],
                "source_claim_id": (
                    "CLM-070038"
                    if source["handoff_id"] in {"HOF-PIE-003", "HOF-PIE-011"}
                    else ""
                ),
                "as_of_date": "2026-07-26",
                "notes": (
                    "Fresh review permits bounded mechanism and issuer-level "
                    "materiality use. Final publication requires archived source "
                    "documents; top-20, S&P-wide, frequency, typicality and complete "
                    "investee allocation remain blocked."
                ),
            }
        )
    return rows


def build_transform_audit(
    source_file: str,
    source_rows: list[dict[str, str]],
    id_field: str,
    canonical_table: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence = 0
    for source in source_rows:
        for field, value in source.items():
            sequence += 1
            rows.append(
                {
                    "audit_id": f"PIE-XFM-{sequence:05d}-{source_file[:3].upper()}",
                    "packet_id": "private_investment_earnings_corrected",
                    "staging_artifact": source_file,
                    "staging_record_id": source[id_field],
                    "staging_field": field,
                    "staging_value": value,
                    "canonical_table": canonical_table,
                    "canonical_record_id": source[id_field],
                    "canonical_field": (
                        "see_typed_transformation_builder"
                        if field
                        not in {
                            id_field,
                            "investor_entity_id",
                            "notes",
                            "source_id",
                        }
                        else field
                    ),
                    "canonical_value": (
                        value
                        if field
                        in {
                            id_field,
                            "investor_entity_id",
                            "notes",
                            "source_id",
                        }
                        else "transformed_in_candidate_tables"
                    ),
                    "transformation_rule": (
                        "typed_split_or_normalization"
                        if field
                        not in {
                            id_field,
                            "investor_entity_id",
                            "notes",
                            "source_id",
                        }
                        else "identity_copy"
                    ),
                    "lossless_status": "preserved_with_source_value",
                    "review_status": "accepted_bounded_rebase",
                    "notes": (
                        "The exact staging value is retained here even when split "
                        "across typed canonical fields."
                    ),
                }
            )
    return rows


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE private_investment_accounting_methods (
    method_id TEXT PRIMARY KEY,
    accounting_method TEXT NOT NULL UNIQUE,
    governing_basis TEXT NOT NULL,
    recognition_trigger TEXT NOT NULL,
    income_statement_or_oci_location TEXT NOT NULL,
    cash_at_recognition_default TEXT NOT NULL,
    carrying_value_effect TEXT NOT NULL,
    required_separation TEXT NOT NULL,
    example TEXT NOT NULL,
    provenance_class TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings (
    private_investment_earnings_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL,
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT,
    investee_scope_description TEXT NOT NULL,
    instrument_scope TEXT NOT NULL,
    accounting_method_id TEXT REFERENCES private_investment_accounting_methods(method_id),
    accounting_method_scope TEXT NOT NULL,
    recognition_location TEXT NOT NULL,
    reported_line_item TEXT NOT NULL,
    pretax_gain_loss_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision TEXT NOT NULL,
    amount_unparsed_suffix TEXT NOT NULL,
    recognition_state TEXT NOT NULL,
    cash_received_amount REAL,
    cash_received_state TEXT NOT NULL,
    cash_amount_precision TEXT NOT NULL,
    reported_pretax_income_amount REAL,
    reported_pretax_income_precision TEXT NOT NULL,
    gain_share_of_pretax_income_pct REAL,
    ratio_precision TEXT NOT NULL,
    investee_attribution_state TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_id TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    source_claim_id TEXT,
    nonadditive_overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0,1)),
    analyst_derived INTEGER NOT NULL CHECK (analyst_derived IN (0,1)),
    integration_status TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings_components (
    parent_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_order INTEGER NOT NULL,
    relationship TEXT NOT NULL,
    additive_within_parent INTEGER NOT NULL CHECK (additive_within_parent IN (0,1)),
    notes TEXT NOT NULL,
    PRIMARY KEY (parent_earnings_id, component_earnings_id)
);

CREATE TABLE private_investment_carrying_value_rollforwards (
    rollforward_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL,
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT,
    investee_scope_description TEXT NOT NULL,
    instrument_scope TEXT NOT NULL,
    beginning_carrying_value_amount REAL,
    ending_carrying_value_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    beginning_amount_precision TEXT NOT NULL,
    ending_amount_precision TEXT NOT NULL,
    reconciliation_state TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    integration_status TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_rollforward_components (
    rollforward_component_id TEXT PRIMARY KEY,
    rollforward_id TEXT NOT NULL REFERENCES private_investment_carrying_value_rollforwards(rollforward_id),
    component_type TEXT NOT NULL,
    amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision TEXT NOT NULL,
    amount_unparsed_suffix TEXT NOT NULL,
    cash_state TEXT NOT NULL,
    income_aoci_balance_sheet_class TEXT NOT NULL,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0,1)),
    overlap_group TEXT,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings_bridges (
    bridge_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL,
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    reported_operating_income_amount REAL,
    reported_pretax_income_amount REAL,
    identified_private_investment_gain_amount REAL,
    arithmetic_pretax_residual_amount REAL,
    private_gain_share_pretax_pct REAL,
    cash_flow_reversal_amount REAL,
    cash_flow_reversal_precision TEXT NOT NULL,
    cash_flow_reversal_perimeter TEXT NOT NULL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    selected_gain_precision TEXT NOT NULL,
    residual_precision TEXT NOT NULL,
    ratio_precision TEXT NOT NULL,
    tax_effect_state TEXT NOT NULL,
    analyst_derived_state TEXT NOT NULL,
    integration_status TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_bridge_components (
    bridge_id TEXT NOT NULL REFERENCES private_investment_earnings_bridges(bridge_id),
    private_investment_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_role TEXT NOT NULL,
    PRIMARY KEY (bridge_id, private_investment_earnings_id)
);

CREATE TABLE private_investment_population_observations (
    population_observation_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL,
    issuer_label TEXT NOT NULL,
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT,
    investee_scope_description TEXT NOT NULL,
    accounting_method_id TEXT REFERENCES private_investment_accounting_methods(method_id),
    accounting_method_scope TEXT NOT NULL,
    observation_class TEXT NOT NULL,
    material_private_gain_state TEXT NOT NULL,
    operating_nonoperating_state TEXT NOT NULL,
    carrying_value_amount REAL,
    cash_proceeds_amount REAL,
    cumulative_upward_adjustments_amount REAL,
    current_period_private_gain_loss_amount REAL,
    gain_share_of_pretax_pct REAL,
    named_position_cost_amount REAL,
    reported_pretax_income_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision_state TEXT NOT NULL,
    investee_attribution_state TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    integration_status TEXT NOT NULL,
    notes TEXT NOT NULL
);
""".strip() + "\n"


def build(output: Path = OUTPUT) -> None:
    review_results = require_acceptance()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    methods = build_methods()
    earnings, earnings_components = build_earnings()
    rollforwards, rollforward_components = build_rollforwards()
    bridges, bridge_components = build_bridges()
    population = build_population()
    claims, evidence_links = build_claims()
    unknowns = build_unknowns()
    handoffs = build_handoffs()

    datasets: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "private_investment_accounting_methods.csv": (
            list(methods[0]),
            methods,
        ),
        "private_investment_earnings.csv": (list(earnings[0]), earnings),
        "private_investment_earnings_components.csv": (
            list(earnings_components[0]),
            earnings_components,
        ),
        "private_investment_carrying_value_rollforwards.csv": (
            list(rollforwards[0]),
            rollforwards,
        ),
        "private_investment_rollforward_components.csv": (
            list(rollforward_components[0]),
            rollforward_components,
        ),
        "private_investment_earnings_bridges.csv": (
            list(bridges[0]),
            bridges,
        ),
        "private_investment_bridge_components.csv": (
            list(bridge_components[0]),
            bridge_components,
        ),
        "private_investment_population_observations.csv": (
            list(population[0]),
            population,
        ),
        "candidate_claims.csv": (list(claims[0]), claims),
        "candidate_evidence_links.csv": (
            list(evidence_links[0]),
            evidence_links,
        ),
        "candidate_unknowns.csv": (list(unknowns[0]), unknowns),
        "candidate_handoffs.csv": (list(handoffs[0]), handoffs),
    }
    for file_name, (columns, rows) in datasets.items():
        write_csv(output / file_name, columns, rows)

    audit: list[dict[str, Any]] = []
    audit.extend(
        build_transform_audit(
            "private_investment_earnings_ledger.csv",
            read_csv(AUTHOR / "private_investment_earnings_ledger.csv"),
            "record_id",
            "private_investment_earnings",
        )
    )
    audit.extend(
        build_transform_audit(
            "carrying_value_rollforward.csv",
            read_csv(AUTHOR / "carrying_value_rollforward.csv"),
            "rollforward_id",
            "private_investment_carrying_value_rollforwards",
        )
    )
    audit.extend(
        build_transform_audit(
            "earnings_quality_bridge.csv",
            read_csv(AUTHOR / "earnings_quality_bridge.csv"),
            "bridge_id",
            "private_investment_earnings_bridges",
        )
    )
    audit.extend(
        build_transform_audit(
            "population_evidence.csv",
            read_csv(AUTHOR / "population_evidence.csv"),
            "population_id",
            "private_investment_population_observations",
        )
    )
    write_csv(output / "transformation_audit.csv", list(audit[0]), audit)
    (output / "candidate_schema.sql").write_text(SCHEMA, encoding="utf-8")

    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "status": "BOUNDED_INTEGRATION_CANDIDATE_NOT_CANONICAL",
        "generated_from_author_manifest_sha256": EXPECTED_AUTHOR_MANIFEST_SHA,
        "generated_from_review_manifest_sha256": EXPECTED_REVIEW_MANIFEST_SHA,
        "review_verdict": review_results["verdict"],
        "canonical_authorization": review_results[
            "canonical_integration_authorization"
        ],
        "retained_holds": review_results["external_holds"],
        "files": {
            path.name: {
                "sha256": sha256(path),
                "rows": (
                    len(read_csv(path)) if path.suffix == ".csv" else None
                ),
            }
            for path in files
        },
        "limits": [
            "This is a typed candidate rebase, not canonical data.",
            "Final publication requires archived and content-hashed primary sources.",
            "Top-20, S&P-wide, frequency, typicality and complete investee allocation remain blocked.",
            "Candidate claims and handoffs require atomic merge with the final Chapters 16-19 state.",
        ],
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    build(arguments.output.resolve())
