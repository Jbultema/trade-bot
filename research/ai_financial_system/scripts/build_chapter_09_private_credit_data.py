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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_09/data"


CAPITAL_CHAIN_MAP = [
    {
        "sequence": 1,
        "capital_or_risk_holder": "Nontraded fund shareholder",
        "legal_claim": "Fund or REIT share measured at periodic NAV",
        "cash_in": "Subscription proceeds",
        "cash_out_or_control": "Distributions and discretionary quarterly repurchases",
        "next_layer": "Ares Core Infrastructure Fund or Blue Owl Digital Infrastructure Trust",
        "early_stress_indicator": "NAV change repurchase proration suspension or distribution composition",
        "source_claim_ids": "CLM-090014;CLM-090015",
    },
    {
        "sequence": 2,
        "capital_or_risk_holder": "Insurer or retirement-services balance sheet",
        "legal_claim": "Policy funding agreement or other insurance liability",
        "cash_in": "Premiums and funding-agreement proceeds",
        "cash_out_or_control": "Investment allocation and liability payments",
        "next_layer": "Apollo-originated managed or affiliated assets",
        "early_stress_indicator": "Investment commitment draw ratings capital charge or liquidity need",
        "source_claim_ids": "CLM-090018",
    },
    {
        "sequence": 3,
        "capital_or_risk_holder": "Bank warehouse lender",
        "legal_claim": "Senior secured revolving or warehouse loan to fund SPV",
        "cash_in": "Interest fees and principal repayment",
        "cash_out_or_control": "Advance rate borrowing base cash control and collateral remedies",
        "next_layer": "Eligible private loans or fund collateral",
        "early_stress_indicator": "Advance-ratio breach eligibility change cash trap or servicer action",
        "source_claim_ids": "CLM-090005;CLM-090006",
    },
    {
        "sequence": 4,
        "capital_or_risk_holder": "Private fund or nontraded REIT",
        "legal_claim": "Loan property or CMBS investment",
        "cash_in": "Interest rent principal and asset-sale proceeds",
        "cash_out_or_control": "Delayed draws property costs debt service fees and distributions",
        "next_layer": "Project borrower property owner or ABS issuer",
        "early_stress_indicator": "Level 3 mark unfunded draw leverage or net investment income change",
        "source_claim_ids": "CLM-090001;CLM-090002;CLM-090011;CLM-090013",
    },
    {
        "sequence": 5,
        "capital_or_risk_holder": "ABS or project lender",
        "legal_claim": "Secured note or project loan",
        "cash_in": "Project or lease debt service",
        "cash_out_or_control": "Construction funding reserves and enforcement",
        "next_layer": "Data-center project SPV",
        "early_stress_indicator": "Construction draw reserve use DSCR loan-to-cost or refinancing spread",
        "source_claim_ids": "CLM-090007;CLM-090008;CLM-090010;CLM-090012",
    },
    {
        "sequence": 6,
        "capital_or_risk_holder": "Project sponsor and borrower",
        "legal_claim": "Residual equity and ownership of project assets",
        "cash_in": "Tenant rent and residual value",
        "cash_out_or_control": "Construction equipment power operating cost and debt service",
        "next_layer": "Tenant or AI compute customer",
        "early_stress_indicator": "Cost overrun delayed acceptance lower utilization or sponsor equity call",
        "source_claim_ids": "CLM-090008;CLM-090009",
    },
    {
        "sequence": 7,
        "capital_or_risk_holder": "Tenant or AI compute customer",
        "legal_claim": "Capacity lease or service right",
        "cash_in": "End-customer compute revenue or internal workload value",
        "cash_out_or_control": "Rent capacity payment and deployment commitments",
        "next_layer": "External AI demand and enterprise or consumer cash flow",
        "early_stress_indicator": "Deployment deferral renegotiation utilization pricing or credit weakness",
        "source_claim_ids": "CLM-090008;CLM-090017",
    },
]


FILING_CURRENCY_AUDIT = [
    {
        "entity_id": "ENT-ARES-CIF",
        "entity": "Ares Core Infrastructure Fund",
        "current_source_id": "SRC-0127",
        "period_end": "2026-03-31",
        "publication_date": "2026-05-11",
        "research_cutoff": "2026-07-25",
        "status": "current_periodic_filing",
        "notes": "June warehouse amendment is separately registered as a subsequent contractual source.",
    },
    {
        "entity_id": "ENT-ARES-ACI-LIQUID-SPV",
        "entity": "ACI Liquid Aggregator SPV LLC",
        "current_source_id": "SRC-0128",
        "period_end": "2026-06-26",
        "publication_date": "2026-06-26",
        "research_cutoff": "2026-07-25",
        "status": "current_file_contract",
        "notes": "Executed asset-specific DC Addenda remain unavailable.",
    },
    {
        "entity_id": "ENT-BODIT",
        "entity": "Blue Owl Digital Infrastructure Trust",
        "current_source_id": "SRC-0129",
        "period_end": "2026-03-31",
        "publication_date": "2026-05-14",
        "research_cutoff": "2026-07-25",
        "status": "current_periodic_filing",
        "notes": "April revolver filing is separately registered.",
    },
    {
        "entity_id": "ENT-GALAXY",
        "entity": "Galaxy Digital Inc.",
        "current_source_id": "SRC-0131",
        "period_end": "2026-03-31",
        "publication_date": "2026-05-08",
        "research_cutoff": "2026-07-25",
        "status": "current_periodic_plus_pricing_announcement",
        "notes": "July 23 note pricing had an announced closing after the cutoff.",
    },
    {
        "entity_id": "ENT-APOLLO",
        "entity": "Apollo Global Management Inc.",
        "current_source_id": "SRC-0134",
        "period_end": "2026-03-31",
        "publication_date": "2026-05-08",
        "research_cutoff": "2026-07-25",
        "status": "current_periodic_plus_transaction_announcement",
        "notes": "No instrument-level AI XPV closing package was located.",
    },
    {
        "entity_id": "ENT-BXINFRA",
        "entity": "Blackstone Infrastructure Strategies LP",
        "current_source_id": "SRC-0135",
        "period_end": "2026-03-31",
        "publication_date": "2026-04-30",
        "research_cutoff": "2026-07-25",
        "status": "current_periodic_filing",
        "notes": "Warehouse amount remains non-AI-attributed.",
    },
]


WRITING_GATE_AUDIT = [
    {"gate_id": "WG-0901", "gate": "Named Ares loans reconcile to filed fair value", "status": "pass", "evidence": "Six rows sum to 405.076 million dollars."},
    {"gate_id": "WG-0902", "gate": "Ares unfunded commitments reconcile", "status": "pass", "evidence": "Six rows sum to 301.583 million dollars."},
    {"gate_id": "WG-0903", "gate": "Warehouse capacity is nonadditive to collateral", "status": "pass", "evidence": "Overlap groups and additive flags are explicit."},
    {"gate_id": "WG-0904", "gate": "Warehouse control waterfall is frozen", "status": "pass", "evidence": "Advance ratio cure cash control servicer and disposition rights are separated."},
    {"gate_id": "WG-0905", "gate": "Blue Owl assets debt CMBS and liquidity reconcile", "status": "pass", "evidence": "Balance-sheet and investor-liquidity measures occupy separate ledgers."},
    {"gate_id": "WG-0906", "gate": "Apollo and Broadcom headlines are nonadditive", "status": "pass", "evidence": "Measurement reconciliation preserves instrument and status differences."},
    {"gate_id": "WG-0907", "gate": "Insurer perimeter is separated from AI attribution", "status": "pass", "evidence": "Athene amount is not assigned to any named AI asset."},
    {"gate_id": "WG-0908", "gate": "Mark liquidity nonaccrual and default clocks are distinct", "status": "pass", "evidence": "Seven-stage valuation-clock ledger and five risk pathways are frozen."},
    {"gate_id": "WG-0909", "gate": "Material unknowns remain visible", "status": "pass", "evidence": "Twelve open Chapter 9 unknowns are retained."},
]


def query(connection: sqlite3.Connection, sql: str) -> list[dict[str, object]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(sql).fetchall()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    exposures = query(connection, "SELECT * FROM private_credit_exposures ORDER BY exposure_id")
    funds = query(connection, "SELECT * FROM private_credit_fund_liquidity ORDER BY fund_liquidity_id")
    warehouses = query(connection, "SELECT * FROM private_credit_warehouse_controls ORDER BY warehouse_control_id")
    clocks = query(connection, "SELECT * FROM private_credit_valuation_clocks ORDER BY sequence_order")
    reconciliations = query(connection, "SELECT * FROM private_credit_measurement_reconciliations ORDER BY reconciliation_id")
    claims = query(
        connection,
        """
        SELECT c.claim_id,c.claim_text,c.evidence_grade,c.confidence,c.claim_type,
               c.accounting_status,c.economic_status,c.quantification_status,c.effective_date,
               GROUP_CONCAT(e.source_id,';') AS source_ids
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id=c.claim_id
        WHERE c.chapter_id='CH09'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(connection, "SELECT * FROM unknowns WHERE chapter_id='CH09' ORDER BY unknown_id")
    pathways = query(connection, "SELECT * FROM risk_pathways WHERE pathway_id LIKE 'RISK-09%' ORDER BY pathway_id")
    source_mismatches = query(
        connection,
        """
        SELECT c.claim_id,c.evidence_grade AS claim_grade,MAX(s.evidence_grade) AS source_grade
        FROM claims c
        JOIN evidence_links e ON e.claim_id=c.claim_id
        JOIN sources s ON s.source_id=e.source_id
        WHERE c.chapter_id='CH09' AND c.evidence_grade='A'
        GROUP BY c.claim_id
        HAVING SUM(CASE WHEN s.evidence_grade='A' THEN 1 ELSE 0 END)=0
        """,
    )
    connection.close()

    if source_mismatches:
        raise ValueError(f"Grade-A claims without Grade-A sources: {source_mismatches}")
    expected = {
        "exposures": 7,
        "funds": 2,
        "warehouses": 2,
        "clocks": 7,
        "reconciliations": 6,
        "claims": 20,
        "unknowns": 12,
        "pathways": 5,
        "capital_chain": 7,
        "currency_audit": 6,
        "writing_gate": 9,
    }
    actual = {
        "exposures": len(exposures),
        "funds": len(funds),
        "warehouses": len(warehouses),
        "clocks": len(clocks),
        "reconciliations": len(reconciliations),
        "claims": len(claims),
        "unknowns": len(unknowns),
        "pathways": len(pathways),
        "capital_chain": len(CAPITAL_CHAIN_MAP),
        "currency_audit": len(FILING_CURRENCY_AUDIT),
        "writing_gate": len(WRITING_GATE_AUDIT),
    }
    if actual != expected:
        raise ValueError(f"Unexpected Chapter 9 counts: {actual}")

    ares = [row for row in exposures if row["holder_entity_id"] == "ENT-ARES-CIF"]
    fair_value = sum(float(row["fair_value"] or 0) for row in ares)
    unfunded = sum(float(row["unfunded_commitment"] or 0) for row in ares)
    if fair_value != 405_076_000:
        raise ValueError(f"Ares fair-value reconciliation failed: {fair_value}")
    if unfunded != 301_583_000:
        raise ValueError(f"Ares unfunded reconciliation failed: {unfunded}")
    if any(int(row["additive_eligible"]) for row in warehouses):
        raise ValueError("Warehouse facilities must not be additive to underlying credit exposures")
    if any(int(row["additive_eligible"]) for row in reconciliations):
        raise ValueError("Measurement reconciliations must be nonadditive")
    if any(row["status"] != "pass" for row in WRITING_GATE_AUDIT):
        raise ValueError("Chapter 9 writing gate has not passed")

    outputs = {
        "private_credit_exposures.csv": exposures,
        "private_credit_fund_liquidity.csv": funds,
        "private_credit_warehouse_controls.csv": warehouses,
        "private_credit_valuation_clocks.csv": clocks,
        "private_credit_measurement_reconciliations.csv": reconciliations,
        "risk_pathways.csv": pathways,
        "capital_chain_map.csv": CAPITAL_CHAIN_MAP,
        "filing_currency_audit.csv": FILING_CURRENCY_AUDIT,
        "writing_gate_audit.csv": WRITING_GATE_AUDIT,
        "chapter_09_claims.csv": claims,
        "chapter_09_unknowns.csv": unknowns,
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
        **actual,
        "ares_data_center_fair_value_usd": fair_value,
        "ares_data_center_unfunded_commitments_usd": unfunded,
        "ares_data_center_total_contractual_perimeter_usd": fair_value + unfunded,
        "total_contractual_perimeter_status": "analytic_sum_of_different_financial_states_not_funded_exposure",
        "cross_system_private_credit_total": None,
        "cross_system_total_status": "intentionally_not_calculated_due_to_overlap_and_missing_lookthrough",
        "writing_gate_status": "passed",
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "The Ares warehouse agreement does not disclose executed asset-specific DC Addenda or advance rates.",
            "The Blue Owl ABS and CMBS collateral pools do not have complete public tenant and property look-through.",
            "Apollo's announced capital solution is not allocated among funds banks debt equity insurers or funded assets.",
            "Athene is not assigned an AI exposure without asset-level evidence.",
            "Galaxy Helios II notes are recorded as priced rather than funded because the announced closing follows the research cutoff.",
            "Fair value unfunded commitments warehouse capacity project debt and contingent support are not summed into a system capital total.",
            "Risk pathways document contractual mechanisms and do not assert current distress at any named vehicle.",
        ],
    }
    (OUTPUT / "chapter_09_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
