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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_05/data"


DELIVERY_TIMELINE = [
    {
        "timeline_id": "PTL-050001",
        "power_system_case_id": "PSC-050001",
        "event_date": "2024-12-04",
        "event_type": "announced",
        "event": "Meta and Entergy announce original Richland plan",
        "physical_status": "three-plant plan",
        "contract_status": "service framework announced",
        "source_claim_id": "CLM-050001",
        "notes": "Later plant and portfolio descriptions may supersede or extend this plan.",
    },
    {
        "timeline_id": "PTL-050002",
        "power_system_case_id": "PSC-050001",
        "event_date": "2025-08-07",
        "event_type": "construction",
        "event": "Two approved combined-cycle plants enter construction",
        "physical_status": "approximately 1500 MW under construction",
        "contract_status": "approved",
        "source_claim_id": "CLM-050002",
        "notes": "The original third plant was not included in the cited construction milestone.",
    },
    {
        "timeline_id": "PTL-050003",
        "power_system_case_id": "PSC-050001",
        "event_date": "2028-12",
        "event_type": "target_service",
        "event": "Target service for first two Richland plants",
        "physical_status": "scheduled",
        "contract_status": "future",
        "source_claim_id": "CLM-050002",
        "notes": "Scheduled service is not guaranteed COD.",
    },
    {
        "timeline_id": "PTL-050004",
        "power_system_case_id": "PSC-050002",
        "event_date": "2031-12-01",
        "event_type": "target_service",
        "event": "West Fork Creek to St Landry system line target",
        "physical_status": "proposed schedule",
        "contract_status": "ESA recovery proposed",
        "source_claim_id": "CLM-050007",
        "notes": "The line can retain regulated residual value after the original ESA term.",
    },
    {
        "timeline_id": "PTL-050005",
        "power_system_case_id": "PSC-050003",
        "event_date": "2025",
        "event_type": "operational",
        "event": "Lowry first 150 MW of grid delivery documented",
        "physical_status": "operational",
        "contract_status": "industrial tariff and demand response",
        "source_claim_id": "CLM-050010",
        "notes": "Second 150 MW was requested and must be tracked separately.",
    },
    {
        "timeline_id": "PTL-050006",
        "power_system_case_id": "PSC-050004",
        "event_date": "2025-07-22",
        "event_type": "point_in_time",
        "event": "MLGW reports about 0.5 MW general service at Tulane Road",
        "physical_status": "minimal municipal supply",
        "contract_status": "no MLGW supercomputer supply",
        "source_claim_id": "CLM-050011",
        "notes": "Later self-generation disclosure does not invalidate this dated observation.",
    },
    {
        "timeline_id": "PTL-050007",
        "power_system_case_id": "PSC-050004",
        "event_date": "2026-03-31",
        "event_type": "company_reported_operation",
        "event": "SpaceX reports gigawatt-scale behind-the-meter capability",
        "physical_status": "company-reported operational capability",
        "contract_status": "owned or leased equipment",
        "source_claim_id": "CLM-050012",
        "notes": "No synchronized turbine dispatch and site-load record was disclosed.",
    },
    {
        "timeline_id": "PTL-050008",
        "power_system_case_id": "PSC-050005",
        "event_date": "2025-12-18",
        "event_type": "regulatory_approval",
        "event": "MPSC conditionally approves Saline special contracts",
        "physical_status": "contract framework approved",
        "contract_status": "1383 MW maximum and 80 percent minimum billing demand",
        "source_claim_id": "CLM-050017",
        "notes": "Contract approval is not energization.",
    },
    {
        "timeline_id": "PTL-050009",
        "power_system_case_id": "PSC-050005",
        "event_date": "2026-03-27",
        "event_type": "project_approval",
        "event": "First 332 MW of Saline storage approved",
        "physical_status": "approved not proven operational",
        "contract_status": "fifteen-year project revenue requirements",
        "source_claim_id": "CLM-050019",
        "notes": "The planned portfolio ceiling remains 1383 MW.",
    },
    {
        "timeline_id": "PTL-050010",
        "power_system_case_id": "PSC-050005",
        "event_date": "2027-12",
        "event_type": "contract_ramp",
        "event": "Saline maximum load ramp",
        "physical_status": "scheduled",
        "contract_status": "maximum may be delayed no more than twelve months",
        "source_claim_id": "CLM-050017",
        "notes": "A contractual date is not a load forecast.",
    },
    {
        "timeline_id": "PTL-050011",
        "power_system_case_id": "PSC-050006",
        "event_date": "2026-03-10",
        "event_type": "control_change",
        "event": "Intersect closes into Alphabet and IPX carve-out launches",
        "physical_status": "mixed development and operating assets",
        "contract_status": "acquired and excluded perimeters split",
        "source_claim_id": "CLM-050024",
        "notes": "Excluded 4.4 GW solar and 8.8 GWh storage remain outside Alphabet.",
    },
    {
        "timeline_id": "PTL-050012",
        "power_system_case_id": "PSC-050008",
        "event_date": "2026-06",
        "event_type": "power_operation",
        "event": "Meitner Quantum power project begins operation",
        "physical_status": "power operating",
        "contract_status": "project economics undisclosed",
        "source_claim_id": "CLM-050026",
        "notes": "The Google data center remained under construction.",
    },
]


CONTROL_RISK = [
    {
        "case_name": "Meta Evest",
        "control_holder": "Entergy owns system assets; Meta controls customer load through Evest and Laidley",
        "first_loss_or_payment_source": "CIAC, ESA revenue, true-ups, termination fee and Meta support",
        "tail_risk_holder": "Other customers can bear the documented 16 percent NPV remainder after the original term",
        "operational_failure_channel": "Correlated loss of both Meta loads can overload a separate 500-kV element",
        "evidence_grade": "A",
        "source_claim_ids": "CLM-050007;CLM-050008;CLM-050009",
    },
    {
        "case_name": "xAI Memphis",
        "control_holder": "MLGW and TVA control grid delivery; SpaceX group controls self-generation and batteries",
        "first_loss_or_payment_source": "Customer-funded grid upgrades, industrial tariff and corporate turbine capital",
        "tail_risk_holder": "SpaceX group retains fuel, permit, uptime and turbine residual-value risk",
        "operational_failure_channel": "Grid curtailment or turbine and permitting constraint reaches compute availability",
        "evidence_grade": "A/C",
        "source_claim_ids": "CLM-050010;CLM-050014;CLM-050015;CLM-050016",
    },
    {
        "case_name": "Oracle Saline",
        "control_holder": "DTE controls utility-owned or procured storage; Green Chile controls facility demand",
        "first_loss_or_payment_source": "Minimum bill, storage revenue requirement, Oracle support and collateral",
        "tail_risk_holder": "DTE equity after customer and collateral recovery",
        "operational_failure_channel": "Data center receives earlier emergency curtailment priority than other firm load",
        "evidence_grade": "A",
        "source_claim_ids": "CLM-050017;CLM-050018;CLM-050019",
    },
    {
        "case_name": "Alphabet Intersect",
        "control_holder": "Alphabet consolidates acquired Intersect; IPX controls explicitly excluded assets",
        "first_loss_or_payment_source": "Alphabet capital for owned assets and take-or-pay or guarantees for external projects",
        "tail_risk_holder": "Alphabet for consolidated economics and contingent guarantees; external investors for excluded assets",
        "operational_failure_channel": "Power can reach operation before compute acceptance and utilization",
        "evidence_grade": "A",
        "source_claim_ids": "CLM-050020;CLM-050021;CLM-050024;CLM-050025;CLM-050026",
    },
    {
        "case_name": "Amazon energy contracts",
        "control_holder": "Project counterparties control assets; Amazon controls consumption and contract elections",
        "first_loss_or_payment_source": "Energy settlement and fixed termination penalties",
        "tail_risk_holder": "Amazon retains price and volume exposure while counterparties retain asset risk subject to remedies",
        "operational_failure_channel": "Consumption variance converts expected physical energy into net cash settlement",
        "evidence_grade": "A",
        "source_claim_ids": "CLM-050022;CLM-050023;CLM-050027",
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

    cases = query(
        connection,
        """
        SELECT
            p.*,
            u.canonical_name AS utility_name,
            c.canonical_name AS customer_name,
            s.canonical_name AS parent_support_name,
            pr.project_name
        FROM power_system_cases p
        LEFT JOIN entities u ON u.entity_id = p.utility_entity_id
        LEFT JOIN entities c ON c.entity_id = p.customer_entity_id
        LEFT JOIN entities s ON s.entity_id = p.parent_support_entity_id
        LEFT JOIN projects pr ON pr.project_id = p.project_id
        ORDER BY p.power_system_case_id
        """,
    )
    assets = query(
        connection,
        """
        SELECT
            a.*,
            e.canonical_name AS owner_name,
            p.case_name
        FROM power_infrastructure_assets a
        LEFT JOIN entities e ON e.entity_id = a.owner_entity_id
        JOIN power_system_cases p
          ON p.power_system_case_id = a.power_system_case_id
        ORDER BY a.power_system_case_id, a.power_asset_id
        """,
    )
    waterfalls = query(
        connection,
        """
        SELECT w.*, p.case_name
        FROM power_cost_waterfalls w
        JOIN power_system_cases p
          ON p.power_system_case_id = w.power_system_case_id
        ORDER BY w.power_system_case_id, w.stage_order
        """,
    )
    measurement_bridge = query(
        connection,
        """
        SELECT
            power_system_case_id,
            case_name,
            announced_load_mw,
            contracted_load_mw,
            minimum_billing_demand_percent,
            generation_nameplate_mw,
            storage_power_mw,
            storage_energy_mwh,
            transmission_miles,
            base_service_term_years,
            overlap_group,
            additive_eligible,
            source_claim_id,
            notes
        FROM power_system_cases
        ORDER BY power_system_case_id
        """,
    )
    unknowns = query(
        connection,
        """
        SELECT *
        FROM unknowns
        WHERE chapter_id = 'CH05'
        ORDER BY unknown_id
        """,
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
        WHERE c.chapter_id = 'CH05'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    connection.close()

    outputs = {
        "power_system_case_ledger.csv": cases,
        "power_infrastructure_asset_ledger.csv": assets,
        "power_cost_waterfall.csv": waterfalls,
        "power_measurement_bridge.csv": measurement_bridge,
        "power_delivery_timeline.csv": DELIVERY_TIMELINE,
        "power_control_and_risk_matrix.csv": CONTROL_RISK,
        "chapter_05_claims.csv": claims,
        "chapter_05_unknowns.csv": unknowns,
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
        "power_system_cases": len(cases),
        "power_assets": len(assets),
        "cost_waterfall_stages": len(waterfalls),
        "timeline_events": len(DELIVERY_TIMELINE),
        "control_risk_cases": len(CONTROL_RISK),
        "chapter_claims": len(claims),
        "chapter_unknowns": len(unknowns),
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "Blank capacity fields are unresolved and must not be interpreted as zero.",
            "Load MW, generation MW, storage MW, storage MWh, contracted MWh and monetary values are distinct measurement classes.",
            "Company-reported capacity and project status are not independently verified dispatch or utilization.",
            "Maximum guarantees are contingent exposure and are not funded capex, expected loss or owned generation.",
            "Timeline dates distinguish announcement, approval, construction, operation and contract boundaries.",
        ],
    }
    (OUTPUT / "chapter_05_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
