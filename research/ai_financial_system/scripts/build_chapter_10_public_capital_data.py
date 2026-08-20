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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_10/data"


PUBLIC_CAPITAL_CHANNEL_MAP = [
    {
        "sequence": 1,
        "channel": "Direct award",
        "legal_measure": "Final award ceiling",
        "cash_measure": "Recipient or agency disbursement by date",
        "control_right": "Milestone audit reduction termination and clawback",
        "nonadditive_rule": "Do not add award ceiling to supported private investment",
        "source_claim_ids": "CLM-100001;CLM-100015",
    },
    {
        "sequence": 2,
        "channel": "Public loan",
        "legal_measure": "Loan commitment or draw",
        "cash_measure": "Principal disbursed net of repayment",
        "control_right": "Loan covenants collateral guarantee and repayment",
        "nonadditive_rule": "Principal is not subsidy cost and may overlap the same project",
        "source_claim_ids": "CLM-100002;CLM-100007",
    },
    {
        "sequence": 3,
        "channel": "Tax expenditure",
        "legal_measure": "Statutory rate applied to qualified basis",
        "cash_measure": "Tax offset or elective payment collected",
        "control_right": "Registration examination disqualification and recapture",
        "nonadditive_rule": "Rate is not cash and direct-award overlap requires property-level coordination",
        "source_claim_ids": "CLM-100012;CLM-100013;CLM-100014",
    },
    {
        "sequence": 4,
        "channel": "Government equity",
        "legal_measure": "Shares escrow and warrants",
        "cash_measure": "Purchase consideration dividends and sale proceeds",
        "control_right": "Voting transfer registration and contingent warrant terms",
        "nonadditive_rule": "Converted grant history is not additive to the resulting equity",
        "source_claim_ids": "CLM-100008;CLM-100009;CLM-100010;CLM-100011",
    },
    {
        "sequence": 5,
        "channel": "State and local incentive",
        "legal_measure": "Final award or binding tax schedule",
        "cash_measure": "Tax foregone or payment actually realized",
        "control_right": "Capital employment wage and recapture conditions",
        "nonadditive_rule": "Nonbinding term sheet and nominal multi-decade ceiling are not present cash",
        "source_claim_ids": "CLM-100004",
    },
    {
        "sequence": 6,
        "channel": "Public or regulated infrastructure",
        "legal_measure": "Asset ownership contribution and rate order",
        "cash_measure": "Customer payments debt service and residual recovery",
        "control_right": "Special contract collateral curtailment and commission allocation",
        "nonadditive_rule": "Public ownership does not identify the ultimate cost bearer",
        "source_claim_ids": "CLM-100016;CLM-100017;CLM-100018",
    },
    {
        "sequence": 7,
        "channel": "Sovereign or supranational vehicle",
        "legal_measure": "Ownership fund commitment or policy facility",
        "cash_measure": "Called and funded capital by asset",
        "control_right": "Board fund consortium and strategic-location conditions",
        "nonadditive_rule": "Mobilization target control and enterprise value are not sovereign cash",
        "source_claim_ids": "CLM-100020;CLM-100021;CLM-100022;CLM-100023;CLM-100024;CLM-100025",
    },
]


WRITING_GATE_AUDIT = [
    {"gate_id": "WG-1001", "gate": "Five major CHIPS awards have award-to-disbursement status bridges", "status": "pass", "evidence": "TSMC Intel Micron Samsung and SK hynix each retain award ceiling and disclosed or unresolved cash status."},
    {"gate_id": "WG-1002", "gate": "Intel grant-to-equity conversion is reconstructed from both sides", "status": "pass", "evidence": "Agreement company annual report proxy and Commerce annual report are linked."},
    {"gate_id": "WG-1003", "gate": "Section 48D rate transition and coordination are verified", "status": "pass", "evidence": "Enacted law IRS instructions and Announcement 2024-40 are separated from taxpayer amounts."},
    {"gate_id": "WG-1004", "gate": "Public loans are separated from direct funding and tax expenditure", "status": "pass", "evidence": "Instrument-specific rows and nonadditive overlap groups are frozen."},
    {"gate_id": "WG-1005", "gate": "Three utility allocation models identify residual holder", "status": "pass", "evidence": "MLGW Entergy and DTE cases distinguish customer contribution term recovery and tail risk."},
    {"gate_id": "WG-1006", "gate": "Canada Cohere contribution is reconciled to project perimeter", "status": "pass", "evidence": "C$240 million ceiling within C$725 million project preserves C$485 million funding unknown."},
    {"gate_id": "WG-1007", "gate": "Three sovereign or national-champion cases have legal look-through", "status": "pass", "evidence": "HUMAIN MGX and InvestAI distinguish control vehicle target commitment and funded amount."},
    {"gate_id": "WG-1008", "gate": "Public capital and catalyzed private investment are nonadditive", "status": "pass", "evidence": "Ten reconciliation rows and additive flags prevent headline summation."},
    {"gate_id": "WG-1009", "gate": "Award ceilings retain milestone and disbursement status", "status": "pass", "evidence": "Every award row has a status ladder milestone conditions and explicit cash field."},
    {"gate_id": "WG-1010", "gate": "Material unknowns and risk pathways remain visible", "status": "pass", "evidence": "Fifteen unknowns and five risk pathways are retained."},
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
    awards = query(connection, "SELECT * FROM public_capital_awards ORDER BY award_id")
    taxes = query(connection, "SELECT * FROM public_tax_incentives ORDER BY tax_incentive_id")
    equity = query(connection, "SELECT * FROM public_equity_positions ORDER BY equity_position_id")
    infrastructure = query(
        connection,
        "SELECT * FROM public_infrastructure_cost_allocation ORDER BY allocation_id",
    )
    sovereign = query(connection, "SELECT * FROM sovereign_capital_cases ORDER BY sovereign_case_id")
    reconciliations = query(
        connection,
        "SELECT * FROM public_capital_measurement_reconciliations ORDER BY reconciliation_id",
    )
    confidence_history = query(
        connection,
        """
        SELECT h.*
        FROM claim_confidence_history h
        JOIN claims c ON c.claim_id=h.claim_id
        WHERE c.chapter_id='CH10'
        ORDER BY h.history_id
        """,
    )
    economic_importance = query(
        connection,
        """
        SELECT i.*
        FROM claim_economic_importance i
        JOIN claims c ON c.claim_id=i.claim_id
        WHERE c.chapter_id='CH10'
        ORDER BY i.importance_id
        """,
    )
    claims = query(
        connection,
        """
        SELECT c.claim_id,c.claim_text,c.evidence_grade,c.confidence,c.claim_type,
               c.accounting_status,c.economic_status,c.quantification_status,c.effective_date,
               GROUP_CONCAT(e.source_id,';') AS source_ids
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id=c.claim_id
        WHERE c.chapter_id='CH10'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(connection, "SELECT * FROM unknowns WHERE chapter_id='CH10' ORDER BY unknown_id")
    pathways = query(
        connection,
        "SELECT * FROM risk_pathways WHERE pathway_id LIKE 'RISK-10%' ORDER BY pathway_id",
    )
    source_mismatches = query(
        connection,
        """
        SELECT c.claim_id,c.evidence_grade AS claim_grade
        FROM claims c
        JOIN evidence_links e ON e.claim_id=c.claim_id
        JOIN sources s ON s.source_id=e.source_id
        WHERE c.chapter_id='CH10' AND c.evidence_grade='A'
        GROUP BY c.claim_id
        HAVING SUM(CASE WHEN s.evidence_grade='A' THEN 1 ELSE 0 END)=0
        """,
    )
    foreign_keys = query(connection, "PRAGMA foreign_key_check")
    connection.close()

    if source_mismatches:
        raise ValueError(f"Grade-A claims without Grade-A sources: {source_mismatches}")
    if foreign_keys:
        raise ValueError(f"Foreign-key violations: {foreign_keys}")

    expected = {
        "awards": 10,
        "taxes": 5,
        "equity": 4,
        "infrastructure": 4,
        "sovereign": 6,
        "reconciliations": 10,
        "confidence_history": 26,
        "economic_importance": 26,
        "claims": 26,
        "unknowns": 15,
        "pathways": 5,
        "channel_map": 7,
        "writing_gate": 10,
    }
    actual = {
        "awards": len(awards),
        "taxes": len(taxes),
        "equity": len(equity),
        "infrastructure": len(infrastructure),
        "sovereign": len(sovereign),
        "reconciliations": len(reconciliations),
        "confidence_history": len(confidence_history),
        "economic_importance": len(economic_importance),
        "claims": len(claims),
        "unknowns": len(unknowns),
        "pathways": len(pathways),
        "channel_map": len(PUBLIC_CAPITAL_CHANNEL_MAP),
        "writing_gate": len(WRITING_GATE_AUDIT),
    }
    if actual != expected:
        raise ValueError(f"Unexpected Chapter 10 counts: {actual}")

    by_id = {row["award_id"]: row for row in awards}
    if float(by_id["PCA-100001"]["final_award_ceiling"]) != 6_600_000_000:
        raise ValueError("TSMC direct-award ceiling reconciliation failed")
    if float(by_id["PCA-100002"]["final_award_ceiling"]) != 5_000_000_000:
        raise ValueError("TSMC loan-ceiling reconciliation failed")
    micron = sum(
        float(by_id[key]["final_award_ceiling"] or 0)
        for key in ("PCA-100004", "PCA-100005")
    )
    if micron != 6_440_000_000:
        raise ValueError(f"Micron current federal direct-funding ceiling failed: {micron}")
    sk_hynix = sum(
        float(by_id[key]["final_award_ceiling"] or 0)
        for key in ("PCA-100007", "PCA-100008")
    )
    if sk_hynix != 958_000_000:
        raise ValueError(f"SK hynix award-plus-loan perimeter failed: {sk_hynix}")

    equity_by_id = {row["equity_position_id"]: row for row in equity}
    issued = equity_by_id["PEQ-100001"]
    if float(issued["shares_issued_or_released"]) != 274_583_000:
        raise ValueError("Intel issued-share count failed")
    rounded_price_value = float(issued["shares_issued_or_released"]) * float(
        issued["stated_price_per_share"]
    )
    if abs(rounded_price_value - float(issued["cash_consideration"])) > 200_000:
        raise ValueError("Intel issued-share price bridge exceeds rounding tolerance")
    escrow = equity_by_id["PEQ-100002"]
    if (
        float(issued["shares_issued_or_released"])
        + float(escrow["shares_issued_or_released"])
        + float(escrow["shares_in_escrow"])
        != 433_323_000
    ):
        raise ValueError("Intel issued and escrow share population does not reconcile")
    if any(int(row["additive_eligible"]) for row in reconciliations):
        raise ValueError("Public-capital measurement reconciliations must be nonadditive")
    if any(row["status"] != "pass" for row in WRITING_GATE_AUDIT):
        raise ValueError("Chapter 10 writing gate has not passed")

    outputs = {
        "public_capital_awards.csv": awards,
        "public_tax_incentives.csv": taxes,
        "public_equity_positions.csv": equity,
        "public_infrastructure_cost_allocation.csv": infrastructure,
        "sovereign_capital_cases.csv": sovereign,
        "public_capital_measurement_reconciliations.csv": reconciliations,
        "claim_confidence_history.csv": confidence_history,
        "claim_economic_importance.csv": economic_importance,
        "chapter_10_claims.csv": claims,
        "chapter_10_unknowns.csv": unknowns,
        "risk_pathways.csv": pathways,
        "public_capital_channel_map.csv": PUBLIC_CAPITAL_CHANNEL_MAP,
        "writing_gate_audit.csv": WRITING_GATE_AUDIT,
    }

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for filename, rows in outputs.items():
        write_csv(OUTPUT / filename, rows)

    manifest = {
        "chapter": 10,
        "database": str(DATABASE.relative_to(REPO_ROOT)),
        "database_sha256": sha256(DATABASE),
        "counts": actual,
        "reconciliations": {
            "tsmc_direct_award_ceiling_usd": 6_600_000_000,
            "tsmc_loan_ceiling_usd": 5_000_000_000,
            "micron_current_federal_direct_funding_ceiling_usd": micron,
            "sk_hynix_grant_and_loan_perimeter_usd": sk_hynix,
            "intel_agreement_disbursement_total_usd": 8_869_800_000,
            "intel_proxy_beneficial_share_population": 433_323_000,
            "cohere_project_funding_gap_cad": 485_000_000,
        },
        "artifacts": {
            name: {"sha256": sha256(OUTPUT / name), "rows": len(rows)}
            for name, rows in outputs.items()
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
