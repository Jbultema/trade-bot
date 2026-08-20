from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DATABASE = REPO_ROOT / "reports/ai_financial_system/research.db"
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_04/data"


RISK_ALLOCATION = [
    {
        "project_finance_case_id": "PFC-040001",
        "case_name": "River Bend",
        "construction_and_completion": "Hut 8 DC project perimeter; Jacobs performance support is separate from noteholder collateral",
        "customer_acceptance_and_revenue": "Fluidstack lease; Google backstop supports base rent and specified pass-through obligations",
        "power_and_operating_cost": "Triple-net tenant/pass-through structure, subject to private contract allocation",
        "residual_value_and_retenanting": "Issuer and noteholders rely on termination fee, reserve liquidity, replacement tenant, or project asset value",
        "refinancing_or_maturity": "Fully amortizing notes to 2042 reduce but do not eliminate lease-tail and operating risk",
        "parent_or_vendor_recourse": "Nonrecourse to Hut 8 parent except issuer covenants; Google support is contractual rather than project ownership",
        "evidence_basis": "A with private-document gaps",
        "source_claim_ids": "CLM-040001;CLM-040003;CLM-040004;CLM-040005",
    },
    {
        "project_finance_case_id": "PFC-040002",
        "case_name": "Saline Township",
        "construction_and_completion": "Related/Blackstone project perimeter; exact borrower and completion support undisclosed",
        "customer_acceptance_and_revenue": "Green Chile is facility lessee/operator and utility customer; campus rent is undisclosed",
        "power_and_operating_cost": "Green Chile pays DTE tariff and separate storage revenue requirements; Oracle guaranty and possible LC support utility obligations",
        "residual_value_and_retenanting": "Campus residual allocation undisclosed; DTE retains storage and bears residual unrecovered utility cost under the Commission order",
        "refinancing_or_maturity": "PIMCO-anchored fixed-rate project debt announced but principal maturity and takeout terms undisclosed",
        "parent_or_vendor_recourse": "Oracle supports Green Chile utility obligations; no public evidence assigns the 16 billion dollar campus cost to Oracle",
        "evidence_basis": "A/B with campus loan and lease gaps",
        "source_claim_ids": "CLM-040008;CLM-040009;CLM-040011;CLM-040013;CLM-040016",
    },
    {
        "project_finance_case_id": "PFC-040003",
        "case_name": "Meta Hyperion",
        "construction_and_completion": "Meta develops and manages; Meta and Blue Owl fund pro rata; private project debt terms are undisclosed",
        "customer_acceptance_and_revenue": "Meta becomes tenant in 2029 under 12.31 billion dollars of initial four-year leases",
        "power_and_operating_cost": "Project perimeter and Meta management roles bear operating execution; detailed utility allocation is deferred to Chapter 5",
        "residual_value_and_retenanting": "Blue Owl side controls remarketing and property sales; Meta covers qualifying fair-value shortfall below a declining threshold",
        "refinancing_or_maturity": "PIMCO and selected investor private debt exists but principal maturity amortization and refinancing exposure are unknown",
        "parent_or_vendor_recourse": "Meta retains future funding rent and conditional residual-value support despite nonconsolidation",
        "evidence_basis": "A/B with private-debt gaps",
        "source_claim_ids": "CLM-040017;CLM-040018;CLM-040019;CLM-040020;CLM-040024;CLM-040025",
    },
    {
        "project_finance_case_id": "PFC-040004",
        "case_name": "CoreWeave CCAC VIII DDTL 4.0",
        "construction_and_completion": "CCAC VIII draws against eligible GPU acquisition installation and site conditions; delayed negative-NOI sites can force repayment",
        "customer_acceptance_and_revenue": "Redacted customer acceptance performance and MSA cash flow determine debt sizing and cash trapping",
        "power_and_operating_cost": "Borrower funds operating costs and an excess-power reserve before distributions",
        "residual_value_and_retenanting": "Lenders hold all-asset and equity collateral; recovery depends on GPU and contract value after customer impairment",
        "refinancing_or_maturity": "Draws end in 2027 and facility matures in 2032 with undisclosed principal schedule",
        "parent_or_vendor_recourse": "CoreWeave recourse is limited to defined bad acts rather than ordinary project underperformance",
        "evidence_basis": "A with redacted customer and site schedules",
        "source_claim_ids": "CLM-040026;CLM-040027;CLM-040029;CLM-040030;CLM-040031;CLM-040032;CLM-040033",
    },
    {
        "project_finance_case_id": "PFC-040005",
        "case_name": "Broadcom–Apollo AI racks",
        "construction_and_completion": "Broadcom supplies custom racks; deployment and customer acceptance schedule are undisclosed",
        "customer_acceptance_and_revenue": "Unidentified customer pays five-year leases to an Apollo-side vehicle",
        "power_and_operating_cost": "Not disclosed at the rack-finance level",
        "residual_value_and_retenanting": "Broadcom may assume leases or cause rack sales; first-loss priority and customized-rack resale value are unknown",
        "refinancing_or_maturity": "Apollo vehicle debt equity and maturity structure are undisclosed",
        "parent_or_vendor_recourse": "Broadcom retains a deployment-linked backstop capped at 29 billion dollars despite transferring rack title",
        "evidence_basis": "A/C with fund SPV customer and capital-stack gaps",
        "source_claim_ids": "CLM-040034;CLM-040035;CLM-040036;CLM-040037;CLM-040038",
    },
]

CONTROL_AND_CONSOLIDATION = [
    {
        "project_finance_case_id": "PFC-040001",
        "case_name": "River Bend",
        "legal_asset_owner": "Hut 8 DC LLC",
        "accounting_conclusion": "Wholly owned Hut 8 project subsidiary within the consolidated company perimeter",
        "economic_control": "Issuer covenants and noteholder collateral constrain cash and asset actions; Hut 8 develops and operates",
        "retained_parent_exposure": "Prior equity development execution and operating dependence remain although notes are nonrecourse to Hut 8 Corp.",
        "confidence": "high",
        "source_claim_ids": "CLM-040001;CLM-040002;CLM-040004",
    },
    {
        "project_finance_case_id": "PFC-040002",
        "case_name": "Saline Township",
        "legal_asset_owner": "Exact Saline project borrower and owner unresolved",
        "accounting_conclusion": "Sponsor consolidation and project-SPV perimeter unresolved; Green Chile is the Oracle utility customer and planned facility lessee/operator",
        "economic_control": "Related develops; Green Chile plans to lease operate cause construction and commission; DTE controls utility-owned storage",
        "retained_parent_exposure": "Oracle supports Green Chile utility obligations; Related Blackstone and PIMCO exposure cannot be allocated from public terms",
        "confidence": "medium",
        "source_claim_ids": "CLM-040008;CLM-040011;CLM-040016",
    },
    {
        "project_finance_case_id": "PFC-040003",
        "case_name": "Meta Hyperion",
        "legal_asset_owner": "Meta-Blue Owl Hyperion venture",
        "accounting_conclusion": "Unconsolidated VIE accounted for under the equity method because Meta lacks power over remarketing and property-sale activities",
        "economic_control": "Blue Owl side controls terminal remarketing and sales while Meta develops manages funds and occupies the campus",
        "retained_parent_exposure": "Meta retains investment future funding initial rent and conditional residual-value support",
        "confidence": "verified",
        "source_claim_ids": "CLM-040017;CLM-040018;CLM-040019;CLM-040020;CLM-040024",
    },
    {
        "project_finance_case_id": "PFC-040004",
        "case_name": "CoreWeave CCAC VIII DDTL 4.0",
        "legal_asset_owner": "CCAC VIII and pledged subsidiaries",
        "accounting_conclusion": "Indirect CoreWeave subsidiary included in consolidated indebtedness while the facility is legally nonrecourse except bad-act carve-outs",
        "economic_control": "Lenders control pledged assets equity and project accounts; CoreWeave operates the platform and customer contract",
        "retained_parent_exposure": "Operational and reputational dependence plus defined bad acts remain without ordinary parent debt-service recourse",
        "confidence": "verified",
        "source_claim_ids": "CLM-040026;CLM-040029;CLM-040031",
    },
    {
        "project_finance_case_id": "PFC-040005",
        "case_name": "Broadcom-Apollo AI racks",
        "legal_asset_owner": "Unidentified Apollo transaction affiliate",
        "accounting_conclusion": "Racks and customer leases transfer outside Broadcom; Apollo fund and vehicle consolidation are unresolved",
        "economic_control": "Customer controls rack use; Apollo vehicle holds title; Broadcom retains default step-in or sale rights",
        "retained_parent_exposure": "Broadcom backstop rises with deployment and declines with customer payments up to a 29 billion dollar ceiling",
        "confidence": "high",
        "source_claim_ids": "CLM-040034;CLM-040035;CLM-040036;CLM-040038",
    },
]

TIER2_CONTROLS = [
    {
        "control_case": "Oracle uncommenced data-center leases",
        "headline_measure": "260.0 billion dollars of substantially data-center lease payments not yet commenced",
        "embedded_or_related_finance": "3.3 billion dollar guarantee of one lessor's borrowing maturing September 2026",
        "structural_test": "Future rent and lessor debt support can coexist before lease recognition",
        "measurement_boundary": "The 3.3 billion dollar guarantee is embedded in rather than additive to the 260 billion dollar lease population",
        "source_claim_ids": "CLM-020097;CLM-020098;CLM-020106",
    },
    {
        "control_case": "Alphabet third-party infrastructure backstops",
        "headline_measure": "28.4 billion dollars credit derivatives plus 9.0 billion dollars financial guarantees at March 2026",
        "embedded_or_related_finance": "33.3 billion dollar future framework with 15.3 billion dollars executed in April against that framework",
        "structural_test": "A hyperscaler can leave title with providers while retaining assumption sublease or asset-transfer remedies",
        "measurement_boundary": "Maximum potential payments are not project cost funded capex or additive to later executions against the same framework",
        "source_claim_ids": "CLM-020082;CLM-020083",
    },
    {
        "control_case": "SpaceX-xAI Valor compute equipment",
        "headline_measure": "20.206 billion dollars aggregate undiscounted payments across three related-party equipment structures",
        "embedded_or_related_finance": "9.041 billion dollars recognized debt from two failed sale-leasebacks; SpaceX-side payment and performance guarantees",
        "structural_test": "Movable compute assets can be legally leased yet accounted for as financing when sale criteria fail",
        "measurement_boundary": "Undiscounted payments recognized debt and guarantees describe overlapping equipment obligations and cannot be summed",
        "source_claim_ids": "CLM-020144;CLM-020145;CLM-020150",
    },
    {
        "control_case": "Cohere-CoreWeave Cambridge",
        "headline_measure": "725 million Canadian dollar compute project",
        "embedded_or_related_finance": "Up to 240 million Canadian dollars of public contribution",
        "structural_test": "Public capital supports a CoreWeave-built and operated facility while Cohere anchors demand",
        "measurement_boundary": "Public contribution project cost facility ownership and customer payments are distinct and private funding remains unresolved",
        "source_claim_ids": "CLM-020162;CLM-020164",
    },
    {
        "control_case": "CoreWeave Kenilworth venture",
        "headline_measure": "57 million dollars of CoreWeave-contributed net assets and a twenty-percent interest",
        "embedded_or_related_finance": "Forward-starting lease and development-management agreement with the unconsolidated venture",
        "structural_test": "Minority ownership can coexist with development control and future tenant dependence",
        "measurement_boundary": "The contribution is not total project cost and future rent and third-party capital are undisclosed",
        "source_claim_ids": "CLM-020167",
    },
    {
        "control_case": "GPU vendor partner-facility guarantees",
        "headline_measure": "3.5 billion dollars NVIDIA and 4.1 billion dollars AMD maximum facility-related support",
        "embedded_or_related_finance": "Partner defaults can trigger payment while warrants or escrow provide separate economics",
        "structural_test": "Semiconductor vendors can support downstream data-center credit while remaining upstream suppliers",
        "measurement_boundary": "Maximum guarantees are not GPU sales customer commitments project cost or expected loss",
        "source_claim_ids": "CLM-020034;CLM-020038;CLM-020185;CLM-020186",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query(connection: sqlite3.Connection, sql: str) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(sql)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    capital = query(
        connection,
        """
        SELECT
            p.project_finance_case_id,
            COALESCE(pr.project_name,
                CASE p.project_finance_case_id
                    WHEN 'PFC-040004' THEN 'CoreWeave CCAC VIII DDTL 4.0'
                    WHEN 'PFC-040005' THEN 'Broadcom-Apollo AI racks'
                END) AS case_name,
            b.canonical_name AS borrower,
            s.canonical_name AS sponsor,
            o.canonical_name AS asset_owner,
            d.canonical_name AS developer,
            t.canonical_name AS tenant,
            u.canonical_name AS end_user,
            g.canonical_name AS guarantor,
            p.project_cost_low,
            p.project_cost_high,
            p.debt_committed,
            p.debt_funded,
            p.sponsor_equity_disclosed,
            p.reimbursed_equity,
            p.currency,
            p.debt_type,
            p.revenue_model,
            p.collateral_or_security,
            p.recourse,
            p.coverage_or_distribution_test,
            p.overlap_group,
            p.source_claim_id,
            p.notes
        FROM project_finance_cases p
        LEFT JOIN projects pr ON pr.project_id = p.project_id
        LEFT JOIN entities b ON b.entity_id = p.borrower_entity_id
        LEFT JOIN entities s ON s.entity_id = p.sponsor_entity_id
        LEFT JOIN entities o ON o.entity_id = p.asset_owner_entity_id
        LEFT JOIN entities d ON d.entity_id = p.developer_entity_id
        LEFT JOIN entities t ON t.entity_id = p.tenant_entity_id
        LEFT JOIN entities u ON u.entity_id = p.end_user_entity_id
        LEFT JOIN entities g ON g.entity_id = p.guarantor_entity_id
        ORDER BY p.project_finance_case_id
        """,
    )
    milestones = query(
        connection,
        """
        SELECT
            m.*,
            e.canonical_name AS responsible_or_controlling_party
        FROM project_delivery_milestones m
        LEFT JOIN entities e
          ON e.entity_id = m.responsible_or_controlling_party_entity_id
        ORDER BY m.project_finance_case_id, m.milestone_order
        """,
    )
    waterfalls = query(
        connection,
        """
        SELECT *
        FROM project_cash_waterfalls
        ORDER BY project_finance_case_id, stage_order
        """,
    )
    maturities = query(
        connection,
        """
        SELECT
            project_finance_case_id,
            issue_date,
            maturity_date,
            amortization_start,
            debt_committed,
            debt_funded,
            currency,
            debt_type,
            coupon_or_rate,
            construction_status,
            initial_delivery_date,
            CASE
              WHEN maturity_date IS NULL THEN 'maturity_undisclosed'
              WHEN amortization_start IS NULL THEN 'bullet_or_schedule_undisclosed'
              ELSE 'amortization_documented'
            END AS maturity_profile,
            source_claim_id
        FROM project_finance_cases
        ORDER BY maturity_date IS NULL, maturity_date
        """,
    )
    measurement_bridge = query(
        connection,
        """
        SELECT
            project_finance_case_id,
            project_cost_low,
            project_cost_high,
            debt_committed,
            debt_funded,
            sponsor_equity_disclosed,
            reimbursed_equity,
            base_contract_revenue,
            optional_contract_revenue,
            average_annual_noi,
            overlap_group,
            additive_eligible,
            source_claim_id,
            notes
        FROM project_finance_cases
        ORDER BY project_finance_case_id
        """,
    )
    connection.close()

    outputs = {
        "project_capital_stack.csv": capital,
        "project_delivery_timeline.csv": milestones,
        "project_cash_waterfall.csv": waterfalls,
        "project_maturity_map.csv": maturities,
        "project_measurement_bridge.csv": measurement_bridge,
        "project_risk_allocation_matrix.csv": RISK_ALLOCATION,
        "project_control_and_consolidation.csv": CONTROL_AND_CONSOLIDATION,
        "tier2_structure_controls.csv": TIER2_CONTROLS,
    }
    for filename, rows in outputs.items():
        write_csv(OUTPUT / filename, rows)

    snapshot = OUTPUT / "research_snapshot.db"
    shutil.copyfile(DATABASE, snapshot)
    artifacts = [OUTPUT / name for name in outputs] + [snapshot]
    manifest = {
        "database": str(DATABASE),
        "database_sha256": sha256(DATABASE),
        "database_snapshot": snapshot.name,
        "case_records": len(capital),
        "milestone_records": len(milestones),
        "waterfall_records": len(waterfalls),
        "risk_allocation_records": len(RISK_ALLOCATION),
        "control_and_consolidation_records": len(CONTROL_AND_CONSOLIDATION),
        "tier2_control_records": len(TIER2_CONTROLS),
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "Blank capital fields are unresolved and must not be interpreted as zero.",
            "Derived dates are analytical boundaries rather than contractual dates.",
            "Project cost, debt, rent, NOI, and contingent support use different measurement classes and are not additive by default.",
            "Risk allocation identifies legal and economic channels but does not estimate probability of default or loss given default.",
        ],
    }
    (OUTPUT / "chapter_04_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
