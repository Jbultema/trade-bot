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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_06/data"

LAB_ENTITY_IDS = (
    "ENT-OAI-GROUP",
    "ENT-OAI-OPCO",
    "ENT-OAI-GLOBAL",
    "ENT-ANTHROPIC",
    "ENT-XAI",
)


CAPITAL_STATUS_BRIDGE = [
    {
        "laboratory": "OpenAI",
        "measurement_date": "2026-07-01",
        "capital_program": "2026 equity round",
        "announced_or_committed_usd_m": 122000,
        "minimum_traceable_funded_usd_m": 35000,
        "undrawn_or_scheduled_usd_m": 45000,
        "unresolved_usd_m": 42000,
        "do_not_add_usd_m": 0,
        "interpretation": (
            "The 122 billion dollars is committed capital, not closing cash. At least "
            "15 billion Amazon and 20 billion SoftBank were traceably funded by July 1. "
            "Amazon's remaining 35 billion and SoftBank's scheduled 10 billion were not "
            "then funded; NVIDIA and residual-investor cash timing remains unresolved."
        ),
        "source_claim_ids": "CLM-020109;CLM-020071;CLM-020111;CLM-020115;CLM-020120",
    },
    {
        "laboratory": "OpenAI",
        "measurement_date": "2026-03-31",
        "capital_program": "revolving credit facility",
        "announced_or_committed_usd_m": 4700,
        "minimum_traceable_funded_usd_m": 0,
        "undrawn_or_scheduled_usd_m": 4700,
        "unresolved_usd_m": 0,
        "do_not_add_usd_m": 0,
        "interpretation": "Committed debt capacity was undrawn and is not equity-round funding.",
        "source_claim_ids": "CLM-020113",
    },
    {
        "laboratory": "Anthropic",
        "measurement_date": "2026-05-28",
        "capital_program": "Series H",
        "announced_or_committed_usd_m": 65000,
        "minimum_traceable_funded_usd_m": 5000,
        "undrawn_or_scheduled_usd_m": 0,
        "unresolved_usd_m": 60000,
        "do_not_add_usd_m": 15000,
        "interpretation": (
            "The round was described as raised and includes 15 billion dollars of "
            "previously committed hyperscaler investments. Only Amazon's separately "
            "documented 5 billion dollars is traceable funded cash in the reviewed "
            "evidence; the remaining closing schedule is unresolved. Embedded "
            "components must not be added again."
        ),
        "source_claim_ids": "CLM-020127;CLM-020128;CLM-020130;CLM-020133",
    },
    {
        "laboratory": "Anthropic",
        "measurement_date": "2026-04-24",
        "capital_program": "future hyperscaler financing facilities",
        "announced_or_committed_usd_m": 50000,
        "minimum_traceable_funded_usd_m": 0,
        "undrawn_or_scheduled_usd_m": 50000,
        "unresolved_usd_m": 0,
        "do_not_add_usd_m": 5000,
        "interpretation": (
            "Amazon's 20 billion facility initially has zero availability and grows with "
            "compute delivery; its overlapping 5 billion option reduces that facility. "
            "Alphabet's 30 billion path is milestone contingent."
        ),
        "source_claim_ids": "CLM-020074;CLM-020075;CLM-020086",
    },
    {
        "laboratory": "xAI",
        "measurement_date": "2026-01-06",
        "capital_program": "Series E",
        "announced_or_committed_usd_m": 20000,
        "minimum_traceable_funded_usd_m": 20000,
        "undrawn_or_scheduled_usd_m": 0,
        "unresolved_usd_m": 0,
        "do_not_add_usd_m": 2000,
        "interpretation": (
            "The aggregate includes Tesla's 2 billion dollar component, which closed "
            "after the merger as SpaceX common stock and is not additive."
        ),
        "source_claim_ids": "CLM-020138;CLM-020139",
    },
    {
        "laboratory": "xAI",
        "measurement_date": "2026-03-31",
        "capital_program": "secured debt and parent bridge",
        "announced_or_committed_usd_m": 25000,
        "minimum_traceable_funded_usd_m": 25000,
        "undrawn_or_scheduled_usd_m": 0,
        "unresolved_usd_m": 20000,
        "do_not_add_usd_m": 5000,
        "interpretation": (
            "The 20 billion dollar SpaceX bridge refinanced legacy X and xAI debt and "
            "served broader purposes. The prior 5 billion xAI package is a refinanced "
            "component, not incremental debt outstanding; bridge allocation is unknown."
        ),
        "source_claim_ids": "CLM-020142;CLM-020143",
    },
    {
        "laboratory": "xAI",
        "measurement_date": "2026-02-02",
        "capital_program": "SpaceX common-control combination cash",
        "announced_or_committed_usd_m": 2947,
        "minimum_traceable_funded_usd_m": 0,
        "undrawn_or_scheduled_usd_m": 0,
        "unresolved_usd_m": 0,
        "do_not_add_usd_m": 2947,
        "interpretation": (
            "Cash was consideration paid to xAI security holders and did not become "
            "laboratory operating liquidity."
        ),
        "source_claim_ids": "CLM-020140",
    },
]


RISK_TRANSMISSION = [
    {
        "pathway_id": "LRISK-060001",
        "laboratory": "OpenAI",
        "trigger": "capital tranche conditions fail or cloud capacity arrives late",
        "first_order_effect": "less deployable cash or compute than headline commitments imply",
        "second_order_effect": "training and inference deployment slows while fixed purchase commitments persist",
        "external_transmission": "cloud-provider backlog, data-center utilization and financing assumptions weaken",
        "control_or_mitigation": "multi-provider portfolio and undrawn revolver",
        "primary_unknown": "instrument-level closing cash and minimum-spend schedules",
        "source_claim_ids": "CLM-060003;CLM-060006",
    },
    {
        "pathway_id": "LRISK-060002",
        "laboratory": "OpenAI",
        "trigger": "governance conflict between nonprofit controller and capital providers",
        "first_order_effect": "strategic action may not maximize investor economic value",
        "second_order_effect": "recapitalization, financing or supplier negotiations become slower or contested",
        "external_transmission": "Microsoft, Amazon and other investors bear economic exposure without voting control",
        "control_or_mitigation": "defined special voting rights and amended commercial agreements",
        "primary_unknown": "complete reserved-matter, removal and enforcement terms",
        "source_claim_ids": "CLM-060001;CLM-060005",
    },
    {
        "pathway_id": "LRISK-060003",
        "laboratory": "Anthropic",
        "trigger": "Amazon compute delivery milestones are missed",
        "first_order_effect": "facility availability remains below the 20 billion dollar ceiling",
        "second_order_effect": "Anthropic must fund compute purchases from other capital or reduce deployment",
        "external_transmission": "AWS revenue and financing exposure deteriorate together",
        "control_or_mitigation": "Google, Azure and SpaceX capacity diversification",
        "primary_unknown": "delivery milestones, draw elections and remedies",
        "source_claim_ids": "CLM-060008;CLM-060009",
    },
    {
        "pathway_id": "LRISK-060004",
        "laboratory": "Anthropic",
        "trigger": "revenue run rate fails to convert into durable recognized revenue and cash",
        "first_order_effect": "compute obligations outgrow internally generated liquidity",
        "second_order_effect": "new equity or contingent strategic funding becomes more important",
        "external_transmission": "hyperscalers, project tenants and Colossus inherit utilization risk",
        "control_or_mitigation": "rapid reported customer growth and multiple capital sources",
        "primary_unknown": "gross margin, collections, retention and customer concentration",
        "source_claim_ids": "CLM-060012;CLM-060013;CLM-060014",
    },
    {
        "pathway_id": "LRISK-060005",
        "laboratory": "xAI",
        "trigger": "Grok or external compute demand underutilizes Colossus",
        "first_order_effect": "depreciation, energy, external cloud and equipment-finance costs remain",
        "second_order_effect": "combined AI-segment losses and refinancing needs increase",
        "external_transmission": "Valor, Tesla, cloud suppliers, lenders and parent SpaceX absorb different channels",
        "control_or_mitigation": "prospective Anthropic capacity purchase and parent financing access",
        "primary_unknown": "stand-alone utilization, external cloud commitments and asset title",
        "source_claim_ids": "CLM-060018;CLM-060019;CLM-060020;CLM-060022;CLM-060023",
    },
    {
        "pathway_id": "LRISK-060006",
        "laboratory": "xAI",
        "trigger": "related-party asset value or refinancing assumptions prove optimistic",
        "first_order_effect": "recognized debt and payment obligations exceed recoverable equipment value",
        "second_order_effect": "common-control parent must supply liquidity or restructure obligations",
        "external_transmission": "SpaceX creditors and minority holders become exposed to AI-segment economics",
        "control_or_mitigation": "consolidated parent liquidity and cross-segment asset base",
        "primary_unknown": "Valor agreements, collateral, valuation and governance process",
        "source_claim_ids": "CLM-060015;CLM-060020;CLM-060021",
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


def query(connection: sqlite3.Connection, sql: str, parameters: tuple = ()) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(sql, parameters)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    placeholders = ",".join("?" for _ in LAB_ENTITY_IDS)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    control = query(
        connection,
        """
        SELECT
            l.*,
            lab.canonical_name AS laboratory_name,
            related.canonical_name AS controlled_or_related_name,
            controller.canonical_name AS controller_or_counterparty_name
        FROM laboratory_control l
        JOIN entities lab ON lab.entity_id = l.laboratory_entity_id
        JOIN entities related ON related.entity_id = l.controlled_or_related_entity_id
        LEFT JOIN entities controller
          ON controller.entity_id = l.controller_or_counterparty_entity_id
        ORDER BY l.laboratory_entity_id, l.lab_control_id
        """,
    )
    observations = query(
        connection,
        """
        SELECT o.*, e.canonical_name AS laboratory_name
        FROM laboratory_operating_observations o
        JOIN entities e ON e.entity_id = o.laboratory_entity_id
        ORDER BY o.laboratory_entity_id, o.observation_date, o.lab_observation_id
        """,
    )
    flows = query(
        connection,
        """
        SELECT
            f.*,
            lab.canonical_name AS laboratory_name,
            cp.canonical_name AS counterparty_name
        FROM laboratory_counterparty_flows f
        JOIN entities lab ON lab.entity_id = f.laboratory_entity_id
        JOIN entities cp ON cp.entity_id = f.counterparty_entity_id
        ORDER BY f.laboratory_entity_id, f.start_date, f.lab_flow_id
        """,
    )
    funding = query(
        connection,
        f"""
        SELECT
            f.*,
            recipient.canonical_name AS recipient_name,
            investor.canonical_name AS investor_name
        FROM funding_timeline f
        JOIN entities recipient ON recipient.entity_id = f.recipient_entity_id
        LEFT JOIN entities investor ON investor.entity_id = f.investor_entity_id
        WHERE f.recipient_entity_id IN ({placeholders})
        ORDER BY f.recipient_entity_id, f.event_date, f.funding_event_id
        """,
        LAB_ENTITY_IDS,
    )
    compute = query(
        connection,
        f"""
        SELECT
            c.*,
            supplier.canonical_name AS supplier_name,
            customer.canonical_name AS customer_name,
            intermediary.canonical_name AS intermediary_name,
            owner.canonical_name AS asset_owner_name,
            operator.canonical_name AS operator_name,
            payer.canonical_name AS payer_name
        FROM compute_commitments c
        JOIN entities supplier ON supplier.entity_id = c.supplier_entity_id
        JOIN entities customer ON customer.entity_id = c.customer_entity_id
        LEFT JOIN entities intermediary ON intermediary.entity_id = c.intermediary_entity_id
        LEFT JOIN entities owner ON owner.entity_id = c.asset_owner_entity_id
        LEFT JOIN entities operator ON operator.entity_id = c.operator_entity_id
        LEFT JOIN entities payer ON payer.entity_id = c.payer_entity_id
        WHERE c.customer_entity_id IN ({placeholders})
           OR c.payer_entity_id IN ({placeholders})
        ORDER BY c.customer_entity_id, c.announcement_date, c.compute_commitment_id
        """,
        LAB_ENTITY_IDS + LAB_ENTITY_IDS,
    )
    projects = query(
        connection,
        """
        SELECT
            pfc.*,
            p.project_name,
            end_user.canonical_name AS end_user_name,
            tenant.canonical_name AS tenant_name,
            guarantor.canonical_name AS guarantor_name,
            borrower.canonical_name AS borrower_name
        FROM project_finance_cases pfc
        LEFT JOIN projects p ON p.project_id = pfc.project_id
        LEFT JOIN entities end_user ON end_user.entity_id = pfc.end_user_entity_id
        LEFT JOIN entities tenant ON tenant.entity_id = pfc.tenant_entity_id
        LEFT JOIN entities guarantor ON guarantor.entity_id = pfc.guarantor_entity_id
        LEFT JOIN entities borrower ON borrower.entity_id = pfc.borrower_entity_id
        WHERE pfc.end_user_entity_id IN ('ENT-OAI-GROUP', 'ENT-OAI-OPCO', 'ENT-ANTHROPIC', 'ENT-XAI')
        ORDER BY pfc.project_finance_case_id
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
        WHERE c.chapter_id = 'CH06'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(
        connection,
        """
        SELECT *
        FROM unknowns
        WHERE chapter_id = 'CH06'
        ORDER BY unknown_id
        """,
    )
    connection.close()

    outputs = {
        "laboratory_control_map.csv": control,
        "laboratory_operating_observations.csv": observations,
        "laboratory_counterparty_flows.csv": flows,
        "laboratory_funding_timeline.csv": funding,
        "laboratory_compute_commitments.csv": compute,
        "laboratory_project_dependencies.csv": projects,
        "laboratory_capital_status_bridge.csv": CAPITAL_STATUS_BRIDGE,
        "laboratory_risk_transmission.csv": RISK_TRANSMISSION,
        "chapter_06_claims.csv": claims,
        "chapter_06_unknowns.csv": unknowns,
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
        "control_relationships": len(control),
        "operating_observations": len(observations),
        "counterparty_flows": len(flows),
        "funding_events": len(funding),
        "compute_commitments": len(compute),
        "project_dependencies": len(projects),
        "capital_status_rows": len(CAPITAL_STATUS_BRIDGE),
        "risk_transmission_pathways": len(RISK_TRANSMISSION),
        "chapter_claims": len(claims),
        "chapter_unknowns": len(unknowns),
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "OpenAI, Anthropic and xAI are private; public evidence does not supply comparable audited stand-alone financial statements.",
            "Run-rate revenue, recognized revenue, funding, valuation, purchase commitments, capacity and project cost are different measurement classes.",
            "Aggregate funding rounds and investor components are linked by overlap groups and must not be added together.",
            "xAI observations use a combined X and xAI AI-segment perimeter unless explicitly stated otherwise.",
            "Compute values are heterogeneous ceilings, commitments, contract values, monthly prices and letters of intent.",
            "Project debt belongs to its legal borrower and is not silently assigned to the downstream laboratory.",
        ],
    }
    (OUTPUT / "chapter_06_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
