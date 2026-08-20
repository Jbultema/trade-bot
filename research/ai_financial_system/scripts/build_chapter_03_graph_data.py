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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_03/data"


STATUS = {
    "CMP-030001": ("yes", "yes_amendment_uncertain", "unknown", "not_applicable", "yes_general_service", "unknown", "Azure purchase is contracted, but the 2026 amendment does not restate the 250 billion dollar amount and no dedicated physical tranche is disclosed."),
    "CMP-030002": ("yes", "yes", "unknown", "partial", "partial", "unknown", "The relationship combines an existing 38 billion dollar path and a 100 billion dollar expansion; AWS said all capacity was targeted before year-end 2026."),
    "CMP-030003": ("yes", "yes", "unknown", "partial", "partial", "unknown", "Anthropic described nearly one gigawatt by end-2026 and a path to five gigawatts, but utilization and payment history are not public."),
    "CMP-030004": ("yes", "yes_conditional_consumption", "unknown", "no", "no", "no", "Broadcom described approximately 3.5 gigawatts from 2027 and tied consumption to Anthropic commercial success."),
    "CMP-030005": ("yes", "partial_initial_1GW", "unknown", "no", "no", "no", "The first gigawatt is the binding base; scale to six gigawatts and warrant vesting are contingent and schedules are redacted."),
    "CMP-030006": ("yes", "partial_initial_1GW", "unknown", "no", "no", "no", "The first gigawatt-equivalent is the binding base; later tranches and warrant vesting are contingent."),
    "CMP-030007": ("yes", "no_LOI", "unknown", "no", "no", "no", "The ten-gigawatt NVIDIA/OpenAI document is a letter of intent rather than a filed definitive supply contract."),
    "CMP-030008": ("yes", "yes_terminable", "not_applicable_provider_assets", "yes", "yes", "partial_ramp_reported", "Colossus had approximately 320,000 enumerated operational GPUs against a roughly 325,000-GPU service scope; either party can terminate after the initial period on notice."),
    "CMP-030009": ("yes", "yes_conditional", "not_applicable_cloud_service", "unknown", "unknown", "unknown_residual", "NVIDIA buys only qualifying residual unsold CoreWeave capacity subject to delivery, availability, and termination conditions."),
    "CMP-030010": ("yes", "yes_upper_value", "unknown", "partial", "partial", "unknown", "The OpenAI order is executed, but the public record does not allocate delivered capacity or minimum service revenue."),
    "CMP-030011": ("yes", "yes_upper_value", "unknown", "partial", "partial", "unknown", "The Meta order is executed, but the public record does not allocate delivered capacity or minimum service revenue."),
    "CMP-030012": ("yes", "not_applicable_internal_platform", "unknown", "partial", "partial", "partial_external_customers_reported", "Mistral reported initial external customers but did not disclose installed megawatts or utilization against the 200-megawatt target."),
    "CMP-030013": ("yes", "yes", "yes_project_notes", "no_under_construction", "no", "no", "The 245-megawatt lease and project bond are executed, but initial delivery is scheduled for 2027."),
    "CMP-030014": ("yes", "yes", "unknown", "no_future_generations", "no", "no", "Broadcom and Google have long-term supply agreements, but no fixed quantities, project financing, or delivered next-generation scope is public."),
    "CMP-030015": ("yes", "not_applicable_installed_asset", "not_disclosed", "yes", "yes", "yes_operational", "SpaceX reports the approximately 100,000-H100 cluster as operational."),
    "CMP-030016": ("yes", "not_applicable_installed_asset", "not_disclosed", "yes", "yes", "yes_operational", "SpaceX reports the approximately 110,000-GB200 cluster as operational."),
    "CMP-030017": ("yes", "not_applicable_installed_asset", "not_disclosed", "yes", "yes", "yes_operational", "SpaceX reports the approximately 110,000-GB300 cluster as operational."),
    "CMP-030018": ("yes", "no_planned", "unknown", "no", "no", "no", "The next Colossus II phase is planned rather than installed."),
    "CMP-030019": ("yes", "unknown", "unknown", "no", "no", "no", "OpenAI's later five-gigawatt Vera Rubin statement does not disclose a definitive contract or reconcile to the earlier ten-gigawatt LOI."),
    "CMP-030020": ("yes_aggregate_measure", "unknown_customer_contracts", "partial_customer_funding", "unknown", "unknown", "unknown", "The 75 billion dollar Oracle measure mixes customer prepayments and customer-supplied GPUs and is not a physical-capacity or single-contract measure."),
    "CMP-030021": ("yes", "no_project_announcement", "partial_financing_announced", "no", "no", "no", "The 16 billion dollar Saline project financing was announced subject to closing and the campus remained prospective."),
    "CMP-030022": ("yes", "yes_later_scope", "unknown", "no", "no", "no", "Anthropic later described signed five-gigawatt Google/Broadcom agreements, but did not reconcile the earlier 3.5-gigawatt scope or report delivery."),
    "CMP-030023": ("yes", "partial_245MW_base_only", "partial_base_only", "no_under_development", "no", "no", "Only the 245-megawatt River Bend base is executed and financed; the remaining 2,050 megawatts are rights or prospective development."),
    "CMP-030024": ("yes", "yes", "unknown", "no", "no", "no", "Anthropic committed 30 billion dollars to Azure and up to one gigawatt of additional NVIDIA-based capacity, but delivery and utilization are undisclosed."),
}


RECONCILIATION = [
    ("REC-030001", "Microsoft–OpenAI Azure", 250.0, "purchase_commitment", None, None, None, "The 2026 amendment may modify terms; no customer-level funded cash or Microsoft RPO allocation is disclosed."),
    ("REC-030002", "Amazon–OpenAI AWS", 138.0, "commercial_contract_value", None, None, None, "Usage and AWS performance affect timing; the amount is not proven to be an unconditional take-or-pay floor."),
    ("REC-030003", "Amazon–Anthropic AWS", 100.0, "lower_bound_commercial_commitment", None, None, None, "The separate up-to-20-billion-dollar Amazon investment facility unlocks with compute delivery and is capital, not a reduction of cloud spend."),
    ("REC-030004", "CoreWeave–NVIDIA residual order", 6.3, "conditional_service_order", None, None, None, "Only qualifying residual unsold capacity is purchased; the value is not funding for all CoreWeave assets."),
    ("REC-030005", "CoreWeave–OpenAI order", 6.5, "upper_contract_value", None, None, None, "Redacted order forms prevent isolation of minimum revenue, delivered service, and the debt-financed asset pool serving the order."),
    ("REC-030006", "CoreWeave–Meta order", 14.2, "upper_contract_value", None, None, None, "Remaining fees, refunds, and resale offsets depend on breach and service delivery; asset financing is pooled."),
    ("REC-030007", "River Bend base", 7.0, "undiscounted_lease_revenue", 3.25, 3.25, None, "Project-note principal is funded capital; lease value is future rent; Google's maximum backstop exceeds or differs from disclosed base rent and is unquantified."),
    ("REC-030008", "Broadcom investor-owned racks", None, "undisclosed_rack_cost_and_customer_rent", None, None, 29.0, "The 29-billion-dollar figure is Broadcom's maximum customer-lease backstop, not rack purchase cost or funded debt."),
    ("REC-030009", "Oracle customer-funded GPU pool", 75.0, "mixed_customer_cash_and_hardware", 4.592, 4.592, None, "The 4.592-billion-dollar deferred-revenue increase is a fiscal-year accounting change, not a disclosed subset that can be subtracted from the 75-billion-dollar mixed aggregate."),
    ("REC-030010", "Oracle uncommenced data-center leases", 260.0, "undiscounted_future_lease_payments", None, None, 3.3, "The 3.3-billion-dollar lessor-borrowing guarantee is embedded in the 260-billion-dollar lease population and is not additive."),
    ("REC-030011", "SpaceX/xAI Valor I and II", 13.619, "undiscounted_equipment_payments", 9.041, 9.041, 13.619, "Failed-sale-leaseback debt, undiscounted payments, and the group guarantee describe the same financed equipment and cannot be summed."),
    ("REC-030012", "SpaceX/xAI Valor III", 6.587, "undiscounted_equipment_lease_payments", None, None, 6.587, "The lease is executed and guaranteed, but post-commencement accounting and recognized liability are not yet public."),
    ("REC-030013", "Meta Hyperion venture", 12.31, "initial_future_lease_commitment", None, None, 45.99, "Maximum exposure contains carrying value, future funding, lease, and guarantee elements; PIMCO project-debt principal remains undisclosed."),
    ("REC-030014", "Saline Township Oracle campus", 16.0, "total_project_cost", None, None, None, "The amount is neither Oracle contract value nor PIMCO debt principal; sponsor-equity and debt splits were not disclosed."),
    ("REC-030015", "Alphabet existing infrastructure backstops", None, "third_party_project_cost_undisclosed", None, None, 37.4, "The figure is maximum potential credit support for third-party infrastructure, not funded capex."),
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    connection.row_factory = sqlite3.Row
    commitments = {
        row["compute_commitment_id"]: dict(row)
        for row in connection.execute(
            """
            SELECT c.*, s.canonical_name AS supplier_name, k.canonical_name AS customer_name
            FROM compute_commitments c
            JOIN entities s ON s.entity_id = c.supplier_entity_id
            JOIN entities k ON k.entity_id = c.customer_entity_id
            ORDER BY c.compute_commitment_id
            """
        )
    }
    finance = [
        dict(row)
        for row in connection.execute(
            """
            SELECT f.*, o.canonical_name AS owner_name, p.canonical_name AS operator_name,
                   u.canonical_name AS end_user_name, n.canonical_name AS financier_name
            FROM compute_asset_finance f
            LEFT JOIN entities o ON o.entity_id = f.legal_owner_entity_id
            LEFT JOIN entities p ON p.entity_id = f.operator_entity_id
            LEFT JOIN entities u ON u.entity_id = f.end_user_entity_id
            LEFT JOIN entities n ON n.entity_id = f.financing_provider_entity_id
            ORDER BY f.asset_finance_id
            """
        )
    ]
    connection.close()

    missing = sorted(set(commitments) - set(STATUS))
    extra = sorted(set(STATUS) - set(commitments))
    if missing or extra:
        raise ValueError(f"Status map mismatch: missing={missing}, extra={extra}")

    status_rows = []
    for commitment_id, commitment in commitments.items():
        announced, executable, financed, delivered, available, utilized, basis = STATUS[commitment_id]
        status_rows.append(
            {
                "compute_commitment_id": commitment_id,
                "supplier": commitment["supplier_name"],
                "customer": commitment["customer_name"],
                "product_or_service": commitment["product_or_service"],
                "announced_scope": announced,
                "executable_contract": executable,
                "financing_specific": financed,
                "physical_delivery": delivered,
                "service_available": available,
                "utilized_or_paid": utilized,
                "status_basis": basis,
            }
        )
    write_csv(
        OUTPUT / "compute_status_waterfall.csv",
        list(status_rows[0]),
        status_rows,
    )

    edges = []
    for commitment in commitments.values():
        edges.append(
            {
                "edge_id": f"EDGE-{commitment['compute_commitment_id']}",
                "from_node": commitment["supplier_name"],
                "to_node": commitment["customer_name"],
                "edge_type": "compute_or_cloud_commitment",
                "value_or_capacity": (
                    f"{commitment['contract_value']} {commitment['currency']}"
                    if commitment["contract_value"] is not None
                    else f"{commitment['capacity_value']} {commitment['capacity_unit']}"
                ),
                "legal_status": commitment["status_ladder"],
                "source_claim_id": commitment["source_claim_id"],
            }
        )
    for row in finance:
        if row["financier_name"] and row["owner_name"]:
            edges.append(
                {
                    "edge_id": f"EDGE-{row['asset_finance_id']}-FINANCE",
                    "from_node": row["financier_name"],
                    "to_node": row["owner_name"],
                    "edge_type": row["instrument_type"],
                    "value_or_capacity": (
                        f"{row['headline_amount']} {row['currency']}"
                        if row["headline_amount"] is not None
                        else "amount undisclosed"
                    ),
                    "legal_status": row["recognition_status"],
                    "source_claim_id": row["source_claim_id"],
                }
            )
        if row["owner_name"] and row["operator_name"] and row["owner_name"] != row["operator_name"]:
            edges.append(
                {
                    "edge_id": f"EDGE-{row['asset_finance_id']}-CONTROL",
                    "from_node": row["owner_name"],
                    "to_node": row["operator_name"],
                    "edge_type": "asset_title_to_operating_control",
                    "value_or_capacity": row["project_or_asset"],
                    "legal_status": row["title_or_control_status"],
                    "source_claim_id": row["source_claim_id"],
                }
            )
        if row["operator_name"] and row["end_user_name"] and row["operator_name"] != row["end_user_name"]:
            edges.append(
                {
                    "edge_id": f"EDGE-{row['asset_finance_id']}-USE",
                    "from_node": row["operator_name"],
                    "to_node": row["end_user_name"],
                    "edge_type": "operating_control_to_end_use",
                    "value_or_capacity": row["project_or_asset"],
                    "legal_status": row["recognition_status"],
                    "source_claim_id": row["source_claim_id"],
                }
            )
    write_csv(OUTPUT / "compute_dependency_edges.csv", list(edges[0]), edges)

    recon_fields = [
        "reconciliation_id",
        "relationship",
        "headline_value_bn",
        "headline_measure_class",
        "funded_or_drawn_bn",
        "recognized_liability_bn",
        "contingent_maximum_bn",
        "why_not_add_or_net",
    ]
    write_csv(
        OUTPUT / "contract_asset_finance_reconciliation.csv",
        recon_fields,
        [dict(zip(recon_fields, row, strict=True)) for row in RECONCILIATION],
    )

    database_snapshot = OUTPUT / "research_snapshot.db"
    shutil.copyfile(DATABASE, database_snapshot)
    artifacts = [
        OUTPUT / "compute_status_waterfall.csv",
        OUTPUT / "compute_dependency_edges.csv",
        OUTPUT / "contract_asset_finance_reconciliation.csv",
        database_snapshot,
    ]
    manifest = {
        "database": str(DATABASE),
        "database_sha256": sha256(DATABASE),
        "database_snapshot": database_snapshot.name,
        "commitment_records": len(commitments),
        "asset_finance_records": len(finance),
        "dependency_edges": len(edges),
        "reconciliation_records": len(RECONCILIATION),
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "The status waterfall counts relationships, not homogeneous gigawatts, GPUs, or dollars.",
            "Unknown is not zero and partial is not converted into a numeric completion estimate.",
            "Headline, funded, recognized, and contingent amounts are not summed because measurement classes and overlap differ.",
        ],
    }
    (OUTPUT / "chapter_03_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
