from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports/ai_financial_system/research.db"
SEED_TABLES = {
    "sources": PROJECT_ROOT / "data/source_register.csv",
    "entities": PROJECT_ROOT / "data/entities.csv",
    "claims": PROJECT_ROOT / "data/claims.csv",
    "evidence_links": PROJECT_ROOT / "data/evidence_links.csv",
    "financial_flows": PROJECT_ROOT / "data/financial_flows.csv",
    "capital_observations": PROJECT_ROOT / "data/capital_observations.csv",
    "capital_bridge": PROJECT_ROOT / "data/capital_bridge.csv",
    "contracts": PROJECT_ROOT / "data/contracts.csv",
    "debt_instruments": PROJECT_ROOT / "data/debt_instruments.csv",
    "leases": PROJECT_ROOT / "data/leases.csv",
    "guarantees": PROJECT_ROOT / "data/guarantees.csv",
    "investments": PROJECT_ROOT / "data/investments.csv",
    "funding_timeline": PROJECT_ROOT / "data/funding_timeline.csv",
    "projects": PROJECT_ROOT / "data/projects.csv",
    "project_finance_cases": PROJECT_ROOT / "data/project_finance_cases.csv",
    "project_cash_waterfalls": PROJECT_ROOT / "data/project_cash_waterfalls.csv",
    "project_delivery_milestones": PROJECT_ROOT / "data/project_delivery_milestones.csv",
    "power_agreements": PROJECT_ROOT / "data/power_agreements.csv",
    "power_system_cases": PROJECT_ROOT / "data/power_system_cases.csv",
    "power_infrastructure_assets": PROJECT_ROOT / "data/power_infrastructure_assets.csv",
    "power_cost_waterfalls": PROJECT_ROOT / "data/power_cost_waterfalls.csv",
    "laboratory_control": PROJECT_ROOT / "data/laboratory_control.csv",
    "laboratory_operating_observations": PROJECT_ROOT / "data/laboratory_operating_observations.csv",
    "laboratory_counterparty_flows": PROJECT_ROOT / "data/laboratory_counterparty_flows.csv",
    "hyperscaler_capex_observations": PROJECT_ROOT / "data/hyperscaler_capex_observations.csv",
    "hyperscaler_obligations": PROJECT_ROOT / "data/hyperscaler_obligations.csv",
    "hyperscaler_revenue_funding": PROJECT_ROOT / "data/hyperscaler_revenue_funding.csv",
    "compute_supply": PROJECT_ROOT / "data/compute_supply.csv",
    "compute_commitments": PROJECT_ROOT / "data/compute_commitments.csv",
    "compute_asset_finance": PROJECT_ROOT / "data/compute_asset_finance.csv",
    "ownership": PROJECT_ROOT / "data/ownership.csv",
    "dependencies": PROJECT_ROOT / "data/dependencies.csv",
    "risk_pathways": PROJECT_ROOT / "data/risk_pathways.csv",
    "unknowns": PROJECT_ROOT / "data/unknowns.csv",
    "semiconductor_physical_dependencies": PROJECT_ROOT / "data/semiconductor_physical_dependencies.csv",
    "semiconductor_capacity_finance": PROJECT_ROOT / "data/semiconductor_capacity_finance.csv",
    "semiconductor_loss_episodes": PROJECT_ROOT / "data/semiconductor_loss_episodes.csv",
    "semiconductor_substitution_constraints": PROJECT_ROOT / "data/semiconductor_substitution_constraints.csv",
    "semiconductor_regulatory_controls": PROJECT_ROOT / "data/semiconductor_regulatory_controls.csv",
    "semiconductor_measurement_reconciliations": PROJECT_ROOT / "data/semiconductor_measurement_reconciliations.csv",
    "private_credit_exposures": PROJECT_ROOT / "data/private_credit_exposures.csv",
    "private_credit_fund_liquidity": PROJECT_ROOT / "data/private_credit_fund_liquidity.csv",
    "private_credit_warehouse_controls": PROJECT_ROOT / "data/private_credit_warehouse_controls.csv",
    "private_credit_valuation_clocks": PROJECT_ROOT / "data/private_credit_valuation_clocks.csv",
    "private_credit_measurement_reconciliations": PROJECT_ROOT / "data/private_credit_measurement_reconciliations.csv",
    "public_capital_awards": PROJECT_ROOT / "data/public_capital_awards.csv",
    "public_tax_incentives": PROJECT_ROOT / "data/public_tax_incentives.csv",
    "public_equity_positions": PROJECT_ROOT / "data/public_equity_positions.csv",
    "public_infrastructure_cost_allocation": PROJECT_ROOT / "data/public_infrastructure_cost_allocation.csv",
    "sovereign_capital_cases": PROJECT_ROOT / "data/sovereign_capital_cases.csv",
    "public_capital_measurement_reconciliations": PROJECT_ROOT / "data/public_capital_measurement_reconciliations.csv",
    "claim_confidence_history": PROJECT_ROOT / "data/claim_confidence_history.csv",
    "claim_economic_importance": PROJECT_ROOT / "data/claim_economic_importance.csv",
    "economic_exposures": PROJECT_ROOT / "data/economic_exposures.csv",
    "exposure_components": PROJECT_ROOT / "data/exposure_components.csv",
    "utilization_observations": PROJECT_ROOT / "data/utilization_observations.csv",
    "chapter_handoffs": PROJECT_ROOT / "data/chapter_handoffs.csv",
}
NULLABLE_COLUMNS = {
    "url",
    "publication_date",
    "archived_path",
    "content_sha256",
    "jurisdiction",
    "public_private_status",
    "cik",
    "lei",
    "parent_entity_id",
    "effective_date",
    "supersedes_claim_id",
    "amount_low",
    "amount_high",
    "instrument",
    "amount",
    "announced_date",
    "committed_date",
    "funded_date",
    "recognized_period",
    "recourse",
    "seniority",
    "overlap_group",
    "announcement_date",
    "execution_date",
    "expiry_date",
    "stated_value",
    "currency",
    "minimum_commitment",
    "lender_or_holder_entity_id",
    "principal",
    "issue_date",
    "maturity_date",
    "guarantor_entity_id",
    "drawn_amount",
    "lessor_entity_id",
    "commencement_date",
    "disclosed_commitment",
    "recognized_liability",
    "announced_amount",
    "committed_amount",
    "funded_amount",
    "investor_entity_id",
    "closing_date",
    "ownership_percentage",
    "pre_money_valuation",
    "post_money_valuation",
    "supersedes_event_id",
    "financing_source",
    "owner_entity_id",
    "operator_entity_id",
    "project_finance_case_id",
    "borrower_entity_id",
    "sponsor_entity_id",
    "asset_owner_entity_id",
    "developer_entity_id",
    "tenant_entity_id",
    "end_user_entity_id",
    "project_cost_low",
    "project_cost_high",
    "debt_committed",
    "debt_funded",
    "sponsor_equity_disclosed",
    "reimbursed_equity",
    "initial_delivery_date",
    "base_contract_revenue",
    "optional_contract_revenue",
    "average_annual_noi",
    "trigger_or_timing",
    "restriction_or_remedy",
    "milestone_date",
    "responsible_or_controlling_party_entity_id",
    "location",
    "announced_capacity",
    "contracted_capacity",
    "operational_capacity",
    "announced_cost",
    "funded_cost",
    "seller_entity_id",
    "project_id",
    "capacity_mw",
    "energy_mwh",
    "term_years",
    "regulatory_docket",
    "cost_recovery_party",
    "utility_entity_id",
    "customer_entity_id",
    "parent_support_entity_id",
    "announced_load_mw",
    "contracted_load_mw",
    "minimum_billing_demand_percent",
    "generation_nameplate_mw",
    "storage_power_mw",
    "storage_energy_mwh",
    "transmission_miles",
    "base_service_term_years",
    "owner_entity_id",
    "nameplate_capacity",
    "capacity_unit",
    "accredited_capacity",
    "storage_duration_hours",
    "line_miles_low",
    "line_miles_high",
    "voltage_kv",
    "announced_date",
    "approval_date",
    "construction_start",
    "target_in_service_date",
    "actual_in_service_date",
    "disclosed_cost",
    "provider_entity_id",
    "prepaid_amount",
    "minimum_spend",
    "intermediary_entity_id",
    "asset_owner_entity_id",
    "payer_entity_id",
    "quantity_text",
    "capacity_value",
    "capacity_unit",
    "contract_value",
    "prepaid_amount",
    "service_start",
    "termination_rights",
    "renewal_rights",
    "remedies_or_credits",
    "accounting_treatment",
    "compute_commitment_id",
    "legal_owner_entity_id",
    "end_user_entity_id",
    "financing_provider_entity_id",
    "headline_amount",
    "funded_or_drawn_amount",
    "recognized_asset_amount",
    "recognized_liability_amount",
    "collateral_or_security",
    "recognition_status",
    "beneficiary_entity_id",
    "supported_entity_id",
    "maximum_exposure",
    "expiry_date",
    "economic_percentage",
    "voting_percentage",
    "as_of_date",
    "controlled_or_related_entity_id",
    "controller_or_counterparty_entity_id",
    "economic_percentage",
    "voting_percentage",
    "effective_date",
    "metric_value",
    "metric_unit",
    "period_start",
    "period_end",
    "measurement_date",
    "recognized_amount",
    "undiscounted_amount",
    "commitment_amount",
    "maturity_or_term",
    "counterparty_entity_id",
    "overlap_group",
    "counterparty_entity_id",
    "announced_amount",
    "committed_amount",
    "funded_amount",
    "currency",
    "milestone_or_condition",
    "start_date",
    "end_date",
    "system_controller_entity_id",
    "provider_name",
    "geographic_locus",
    "headline_amount",
    "funded_or_recognized_amount",
    "measurement_or_announcement_date",
    "output_milestone",
    "recourse_or_condition",
    "trigger_date",
    "first_loss_bearing_entity_id",
    "recognized_charge_amount",
    "recovery_or_mitigation",
    "incumbent_provider_entity_id",
    "estimated_time_to_replace",
    "performance_or_yield_penalty",
    "linked_unknown_id",
    "announcement_or_effective_date",
    "documented_financial_effect",
    "corroborating_claim_ids",
    "borrower_entity_id",
    "lien_or_seniority",
    "rate_or_coupon",
    "principal_or_face",
    "cost_basis",
    "fair_value",
    "unfunded_commitment",
    "valuation_level",
    "overlap_group",
    "net_assets_or_nav",
    "gross_assets",
    "indebtedness",
    "period_repurchase_amount",
    "distribution_per_share",
    "return_of_capital_per_share",
    "lender_entity_id",
    "commitment_or_capacity",
    "drawn_amount",
    "rate_or_spread",
    "borrowing_base_or_ratio_trigger",
    "cure_period",
    "preliminary_amount",
    "final_award_ceiling",
    "disbursed_amount",
    "recognized_amount",
    "preliminary_date",
    "final_award_date",
    "qualified_basis",
    "recognized_credit",
    "cash_collected_or_offset",
    "taxpayer_entity_id",
    "cash_consideration",
    "shares_issued_or_released",
    "shares_in_escrow",
    "warrant_shares",
    "stated_price_per_share",
    "beneficial_ownership_percentage",
    "customer_entity_id",
    "disclosed_cost",
    "sovereign_or_public_controller_entity_id",
    "investee_or_project_entity_id",
    "headline_or_target_amount",
    "prior_evidence_grade",
    "prior_confidence",
    "legal_obligor_entity_id",
    "economic_beneficiary_entity_id",
    "first_loss_bearer_entity_id",
    "guarantee_id",
    "amount_low",
    "amount_high",
    "maturity_date",
    "overlap_group",
    "project_id",
    "supplier_entity_id",
    "customer_entity_id",
    "period_start",
    "period_end",
    "installed_quantity",
    "available_quantity",
    "announced_quantity",
    "binding_base_quantity",
    "optional_expansion_quantity",
    "delivered_quantity",
    "energized_quantity",
    "accepted_quantity",
    "reserved_quantity",
    "consumed_quantity",
    "billable_quantity",
    "productive_quantity",
    "realized_revenue",
    "revenue_currency",
    "realized_unit_price",
    "source_claim_id",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help=(
            "Directory containing the canonical CSV filenames. This permits a "
            "candidate data tree to be validated without mutating canonical data."
        ),
    )
    args = parser.parse_args()
    build_database(args.output, data_dir=args.data_dir)


def build_database(output: Path, data_dir: Path | None = None) -> None:
    selected_data_dir = (data_dir or PROJECT_ROOT / "data").resolve()
    seed_tables = {
        table: selected_data_dir / canonical_path.name
        for table, canonical_path in SEED_TABLES.items()
    }
    missing_inputs = [path for path in seed_tables.values() if not path.is_file()]
    if missing_inputs:
        missing = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Missing database seed CSVs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.db")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript((PROJECT_ROOT / "schema.sql").read_text(encoding="utf-8"))
        for table, path in seed_tables.items():
            _load_csv(connection, table, path)
        _create_views(connection)
        _validate(connection)
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, output)
    _write_manifest(output, seed_tables)


def _load_csv(connection: sqlite3.Connection, table: str, path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    columns = list(rows[0])
    placeholders = ",".join("?" for _ in columns)
    statement = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    values = []
    for row in rows:
        cleaned = []
        for column in columns:
            value: object = row[column]
            if value == "" and column in NULLABLE_COLUMNS:
                value = None
            if column == "primary_document" and value is not None:
                value = int(value)
            cleaned.append(value)
        values.append(cleaned)
    connection.executemany(statement, values)


def _create_views(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE VIEW claim_evidence AS
        SELECT
            c.claim_id,
            c.chapter_id,
            c.claim_text,
            c.evidence_grade AS claim_grade,
            c.confidence,
            s.source_id,
            s.evidence_grade AS source_grade,
            s.title,
            s.publisher,
            s.url,
            e.locator,
            e.support_type
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id = c.claim_id
        LEFT JOIN sources s ON s.source_id = e.source_id;

        CREATE VIEW unresolved_high_materiality AS
        SELECT *
        FROM unknowns
        WHERE status <> 'resolved' AND materiality = 'high';
        """
    )


def _validate(connection: sqlite3.Connection) -> None:
    orphan_claims = connection.execute(
        """
        SELECT c.claim_id
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id = c.claim_id
        WHERE e.claim_id IS NULL
        """
    ).fetchall()
    if orphan_claims:
        raise ValueError(f"Claims without evidence links: {orphan_claims}")
    weak_primary_claims = connection.execute(
        """
        SELECT DISTINCT c.claim_id
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id = c.claim_id
        LEFT JOIN sources s ON s.source_id = e.source_id
        WHERE c.evidence_grade = 'A'
        GROUP BY c.claim_id
        HAVING SUM(CASE WHEN s.evidence_grade = 'A' THEN 1 ELSE 0 END) = 0
        """
    ).fetchall()
    if weak_primary_claims:
        raise ValueError(f"Grade-A claims without Grade-A sources: {weak_primary_claims}")


def _write_manifest(database: Path, seed_tables: dict[str, Path]) -> None:
    connection = sqlite3.connect(database)
    try:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in seed_tables
        }
    finally:
        connection.close()
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "database": str(database),
        "database_sha256": _sha256(database),
        "row_counts": counts,
        "inputs": {
            _manifest_path(path): _sha256(path)
            for path in [PROJECT_ROOT / "schema.sql", *seed_tables.values()]
        },
    }
    path = database.with_name("database_manifest.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
