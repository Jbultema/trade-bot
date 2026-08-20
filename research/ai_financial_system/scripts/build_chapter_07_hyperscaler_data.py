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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_07/data"

HYPERSCALER_IDS = (
    "ENT-MSFT",
    "ENT-AMZN",
    "ENT-GOOGL",
    "ENT-META",
    "ENT-ORCL",
)


FILING_CURRENCY_AUDIT = [
    {
        "company_entity_id": "ENT-MSFT",
        "company": "Microsoft Corporation",
        "form": "10-Q",
        "period_end": "2026-03-31",
        "filing_date": "2026-04-30",
        "accession": "0001193125-26-191507",
        "source_id": "SRC-0010",
        "audit_date": "2026-07-25",
        "latest_periodic_filing_status": "latest_available",
        "notes": "June fiscal year had ended but the fiscal 2026 Form 10-K was not yet filed at the freeze date.",
    },
    {
        "company_entity_id": "ENT-AMZN",
        "company": "Amazon.com Inc.",
        "form": "10-Q",
        "period_end": "2026-03-31",
        "filing_date": "2026-04-30",
        "accession": "0001018724-26-000014",
        "source_id": "SRC-0013",
        "audit_date": "2026-07-25",
        "latest_periodic_filing_status": "latest_available",
        "notes": "No June-quarter Form 10-Q was filed by the freeze date.",
    },
    {
        "company_entity_id": "ENT-GOOGL",
        "company": "Alphabet Inc.",
        "form": "10-Q",
        "period_end": "2026-03-31",
        "filing_date": "2026-04-30",
        "accession": "0001652044-26-000048",
        "source_id": "SRC-0012",
        "audit_date": "2026-07-25",
        "latest_periodic_filing_status": "latest_available",
        "notes": "No June-quarter Form 10-Q was filed by the freeze date.",
    },
    {
        "company_entity_id": "ENT-META",
        "company": "Meta Platforms Inc.",
        "form": "10-Q",
        "period_end": "2026-03-31",
        "filing_date": "2026-04-30",
        "accession": "0001628280-26-028526",
        "source_id": "SRC-0011",
        "audit_date": "2026-07-25",
        "latest_periodic_filing_status": "latest_available",
        "notes": "No June-quarter Form 10-Q was filed by the freeze date.",
    },
    {
        "company_entity_id": "ENT-ORCL",
        "company": "Oracle Corporation",
        "form": "10-K",
        "period_end": "2026-05-31",
        "filing_date": "2026-06-22",
        "accession": "0001193125-26-277521",
        "source_id": "SRC-0034",
        "audit_date": "2026-07-25",
        "latest_periodic_filing_status": "latest_available",
        "notes": "Latest fiscal-year filing and latest periodic filing at the freeze date.",
    },
]


CAPITAL_CASH_BRIDGE = [
    {
        "company_entity_id": "ENT-MSFT",
        "company": "Microsoft Corporation",
        "period": "nine months ended 2026-03-31",
        "operating_cash_flow_usd_m": 127494,
        "cash_capex_usd_m": 80146,
        "company_capex_usd_m": "",
        "cash_capex_to_ocf_percent": 62.86,
        "noncash_or_supplier_channel_usd_m": 19486,
        "noncash_channel": "finance-lease ROU assets obtained",
        "ending_ppe_payable_usd_m": 22600,
        "ending_net_ppe_usd_m": 283228,
        "construction_or_predeployment_usd_m": "",
        "source_claim_ids": "CLM-070001;CLM-070002",
        "comparison_limit": "Nine-month period; payable is an ending stock and lease additions are noncash.",
    },
    {
        "company_entity_id": "ENT-AMZN",
        "company": "Amazon.com Inc.",
        "period": "trailing twelve months ended 2026-03-31",
        "operating_cash_flow_usd_m": 148531,
        "cash_capex_usd_m": 147299,
        "company_capex_usd_m": "",
        "cash_capex_to_ocf_percent": 99.17,
        "noncash_or_supplier_channel_usd_m": "",
        "noncash_channel": "Quarterly lease and acquired-not-paid channels disclosed outside TTM FCF bridge.",
        "ending_ppe_payable_usd_m": "",
        "ending_net_ppe_usd_m": 397458,
        "construction_or_predeployment_usd_m": "",
        "source_claim_ids": "CLM-070006;CLM-070007;CLM-070009",
        "comparison_limit": "TTM cash measures; net PP&E is quarter-end and includes non-AWS assets.",
    },
    {
        "company_entity_id": "ENT-GOOGL",
        "company": "Alphabet Inc.",
        "period": "three months ended 2026-03-31",
        "operating_cash_flow_usd_m": 45800,
        "cash_capex_usd_m": 35674,
        "company_capex_usd_m": "",
        "cash_capex_to_ocf_percent": 77.89,
        "noncash_or_supplier_channel_usd_m": "",
        "noncash_channel": "Accrued PP&E is separately visible as an ending stock.",
        "ending_ppe_payable_usd_m": 24131,
        "ending_net_ppe_usd_m": 281020,
        "construction_or_predeployment_usd_m": 108597,
        "source_claim_ids": "CLM-070013",
        "comparison_limit": "Quarterly period; assets not in service and payable are ending stocks.",
    },
    {
        "company_entity_id": "ENT-META",
        "company": "Meta Platforms Inc.",
        "period": "three months ended 2026-03-31",
        "operating_cash_flow_usd_m": 32226,
        "cash_capex_usd_m": 18997,
        "company_capex_usd_m": 19840,
        "cash_capex_to_ocf_percent": 58.95,
        "noncash_or_supplier_channel_usd_m": 843,
        "noncash_channel": "Arithmetic difference from finance-lease principal included in company capex.",
        "ending_ppe_payable_usd_m": 16040,
        "ending_net_ppe_usd_m": 194776,
        "construction_or_predeployment_usd_m": 61017,
        "source_claim_ids": "CLM-070017;CLM-070019",
        "comparison_limit": "Quarterly period; finance-lease principal residual is derived.",
    },
    {
        "company_entity_id": "ENT-ORCL",
        "company": "Oracle Corporation",
        "period": "fiscal year ended 2026-05-31",
        "operating_cash_flow_usd_m": 31977,
        "cash_capex_usd_m": 55663,
        "company_capex_usd_m": 47726,
        "cash_capex_to_ocf_percent": 174.07,
        "noncash_or_supplier_channel_usd_m": 7937,
        "noncash_channel": "Manufacturer financing plus significant-financing-component customer prepayments.",
        "ending_ppe_payable_usd_m": "",
        "ending_net_ppe_usd_m": 99957,
        "construction_or_predeployment_usd_m": 39973,
        "source_claim_ids": "CLM-070022;CLM-070023",
        "comparison_limit": "Fiscal-year period; net cash outlay is Oracle supplemental non-GAAP measure.",
    },
]


RISK_TRANSMISSION = [
    {
        "company": "Microsoft",
        "trigger": "supplier or major AI customer fails before data-center delivery",
        "first_order_effect": "server-component receivables or restricted supplier assets weaken",
        "second_order_effect": "uncommenced leases convert into capacity without matched demand",
        "external_transmission": "supplier credit landlords and project lenders absorb delayed cash flow",
        "mitigation_visible": "large commercial RPO and diversified corporate cash flow",
        "primary_unknown": "customer-level RPO cash and supplier recourse",
        "source_claim_ids": "CLM-070003;CLM-070004;CLM-070005",
    },
    {
        "company": "Amazon",
        "trigger": "frontier-laboratory demand or credit weakens",
        "first_order_effect": "AWS backlog conversion and utilization slow",
        "second_order_effect": "leases purchases and financing obligations remain while FCF is already compressed",
        "external_transmission": "landlords utilities suppliers and strategic-investment exposure weaken together",
        "mitigation_visible": "mixed-use AWS fleet and company-wide cash generation",
        "primary_unknown": "OpenAI and Anthropic minimums prepayments and RPO allocation",
        "source_claim_ids": "CLM-070008;CLM-070009;CLM-070010",
    },
    {
        "company": "Alphabet",
        "trigger": "supported infrastructure counterparty defaults or Cloud demand arrives late",
        "first_order_effect": "Alphabet may assume or sublease leases or make termination payments",
        "second_order_effect": "the assets-not-in-service queue begins depreciation without planned utilization",
        "external_transmission": "VIE sponsors landlords lenders and energy suppliers shift exposure toward Alphabet",
        "mitigation_visible": "Cloud RPO and broad internal workload base",
        "primary_unknown": "project-level overlap among commitments backstops and RPO",
        "source_claim_ids": "CLM-070013;CLM-070015;CLM-070016",
    },
    {
        "company": "Meta",
        "trigger": "AI or core-service infrastructure demand underperforms",
        "first_order_effect": "large cloud lease and equipment commitments outlast workload need",
        "second_order_effect": "owned assets migrate to codevelopment or incur impairment and utilization drag",
        "external_transmission": "cloud providers codevelopers suppliers and landlords bear different portions",
        "mitigation_visible": "advertising cash flow and provider resale reduction in one contingent cloud arrangement",
        "primary_unknown": "overlap between lease and commitment populations and codeveloper retained risk",
        "source_claim_ids": "CLM-070018;CLM-070020;CLM-070021",
    },
    {
        "company": "Oracle",
        "trigger": "large AI customer delays delivery acceptance does not renew or cannot perform",
        "first_order_effect": "RPO and customer funding fail to produce expected service cash flow",
        "second_order_effect": "long-duration leases power purchases debt and owned assets remain",
        "external_transmission": "lessors project lenders manufacturers power suppliers and Oracle capital providers weaken",
        "mitigation_visible": "customer prepayments customer-supplied GPUs and contracted backlog",
        "primary_unknown": "customer-by-project duration match cash split and equipment title",
        "source_claim_ids": "CLM-070022;CLM-070024;CLM-070026;CLM-070027",
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


def query(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(sql, parameters)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    placeholders = ",".join("?" for _ in HYPERSCALER_IDS)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    capex = query(
        connection,
        f"""
        SELECT h.*, e.canonical_name AS company_name
        FROM hyperscaler_capex_observations h
        JOIN entities e ON e.entity_id = h.company_entity_id
        WHERE h.company_entity_id IN ({placeholders})
        ORDER BY h.company_entity_id, h.period_end, h.hyperscaler_capex_id
        """,
        HYPERSCALER_IDS,
    )
    obligations = query(
        connection,
        f"""
        SELECT h.*, e.canonical_name AS company_name
        FROM hyperscaler_obligations h
        JOIN entities e ON e.entity_id = h.company_entity_id
        WHERE h.company_entity_id IN ({placeholders})
        ORDER BY h.company_entity_id, h.measurement_date, h.hyperscaler_obligation_id
        """,
        HYPERSCALER_IDS,
    )
    revenue = query(
        connection,
        f"""
        SELECT
            h.*,
            e.canonical_name AS company_name,
            cp.canonical_name AS counterparty_name
        FROM hyperscaler_revenue_funding h
        JOIN entities e ON e.entity_id = h.company_entity_id
        LEFT JOIN entities cp ON cp.entity_id = h.counterparty_entity_id
        WHERE h.company_entity_id IN ({placeholders})
        ORDER BY h.company_entity_id, h.measurement_date, h.hyperscaler_revenue_funding_id
        """,
        HYPERSCALER_IDS,
    )
    shadow_leases = query(
        connection,
        f"""
        SELECT
            h.company_entity_id,
            e.canonical_name AS company_name,
            h.measurement_date,
            h.obligation_type,
            h.commitment_amount,
            h.currency,
            h.maturity_or_term,
            h.ai_attribution,
            h.source_claim_id,
            h.notes
        FROM hyperscaler_obligations h
        JOIN entities e ON e.entity_id = h.company_entity_id
        WHERE h.company_entity_id IN ({placeholders})
          AND h.commencement_status = 'not_yet_commenced'
        ORDER BY h.commitment_amount DESC
        """,
        HYPERSCALER_IDS,
    )
    recognized_debt = query(
        connection,
        f"""
        SELECT
            h.company_entity_id,
            e.canonical_name AS company_name,
            h.measurement_date,
            h.recognized_amount,
            h.undiscounted_amount,
            h.currency,
            h.maturity_or_term,
            h.ai_attribution,
            h.source_claim_id,
            h.notes
        FROM hyperscaler_obligations h
        JOIN entities e ON e.entity_id = h.company_entity_id
        WHERE h.company_entity_id IN ({placeholders})
          AND h.obligation_type = 'recognized_corporate_debt'
        ORDER BY h.recognized_amount DESC
        """,
        HYPERSCALER_IDS,
    )
    claims = query(
        connection,
        """
        SELECT
            c.claim_id,
            c.claim_text,
            c.evidence_grade,
            c.confidence,
            c.claim_type,
            c.accounting_status,
            c.economic_status,
            c.quantification_status,
            c.effective_date,
            GROUP_CONCAT(e.source_id, ';') AS source_ids
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id = c.claim_id
        WHERE c.chapter_id = 'CH07'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(
        connection,
        """
        SELECT *
        FROM unknowns
        WHERE chapter_id = 'CH07'
        ORDER BY unknown_id
        """,
    )
    connection.close()

    shadow_total = sum(float(row["commitment_amount"]) for row in shadow_leases)
    if round(shadow_total, 3) != 821427.0:
        raise ValueError(f"Unexpected uncommenced-lease total: {shadow_total}")
    recognized_debt_total = sum(float(row["recognized_amount"]) for row in recognized_debt)
    if round(recognized_debt_total, 3) != 429956.0:
        raise ValueError(f"Unexpected recognized-debt total: {recognized_debt_total}")

    outputs = {
        "hyperscaler_capex_observations.csv": capex,
        "hyperscaler_obligations.csv": obligations,
        "hyperscaler_revenue_funding.csv": revenue,
        "cross_company_capital_cash_bridge.csv": CAPITAL_CASH_BRIDGE,
        "cross_company_uncommenced_leases.csv": shadow_leases,
        "cross_company_recognized_debt.csv": recognized_debt,
        "cross_company_risk_transmission.csv": RISK_TRANSMISSION,
        "filing_currency_audit.csv": FILING_CURRENCY_AUDIT,
        "chapter_07_claims.csv": claims,
        "chapter_07_unknowns.csv": unknowns,
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
        "capex_observations": len(capex),
        "obligation_observations": len(obligations),
        "revenue_funding_observations": len(revenue),
        "capital_cash_bridge_rows": len(CAPITAL_CASH_BRIDGE),
        "uncommenced_lease_rows": len(shadow_leases),
        "uncommenced_lease_arithmetic_total_usd_m": shadow_total,
        "recognized_debt_rows": len(recognized_debt),
        "recognized_debt_arithmetic_total_usd_m": recognized_debt_total,
        "risk_transmission_pathways": len(RISK_TRANSMISSION),
        "filing_audit_rows": len(FILING_CURRENCY_AUDIT),
        "chapter_claims": len(claims),
        "chapter_unknowns": len(unknowns),
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "Company cash-flow periods differ and are not annualized in the frozen comparison.",
            "Cash capex, company-defined capex, PP&E additions, construction in progress, ending payables and lease commencements are different measurement classes.",
            "Uncommenced lease amounts are not harmonized present values and include non-AI or non-data-center uses in some companies.",
            "RPO is contracted future revenue rather than cash, collateral, profit or an asset-level utilization guarantee.",
            "Purchase commitments, leases, guarantees, derivatives and project backstops can overlap and are not summed into a loss estimate.",
            "Customer-level allocation of backlog, prepayments, supplied equipment and supported infrastructure remains materially undisclosed.",
        ],
    }
    (OUTPUT / "chapter_07_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
