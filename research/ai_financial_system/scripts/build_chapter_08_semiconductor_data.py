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
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_08/data"


FILING_CURRENCY_AUDIT = [
    {
        "company_entity_id": "ENT-NVDA",
        "company": "NVIDIA Corporation",
        "document_type": "Form 10-Q",
        "period_end": "2026-04-26",
        "publication_or_filing_date": "2026-05-20",
        "source_id": "SRC-0018",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "The fiscal 2026 Form 10-K remains the current named-supplier source; the April-quarter Form 10-Q is the current commitment and H20-loss source.",
    },
    {
        "company_entity_id": "ENT-AMD",
        "company": "Advanced Micro Devices Inc.",
        "document_type": "Form 10-Q",
        "period_end": "2026-03-28",
        "publication_or_filing_date": "2026-05-05",
        "source_id": "SRC-0019",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "The 2025 Form 10-K remains the current detailed named-foundry and backend-dependency source.",
    },
    {
        "company_entity_id": "ENT-AVGO",
        "company": "Broadcom Inc.",
        "document_type": "Form 10-Q",
        "period_end": "2026-05-03",
        "publication_or_filing_date": "2026-06-09",
        "source_id": "SRC-0022",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "Current source for TSMC concentration, purchase commitments, RPO and subsequent Apollo-linked rack financing.",
    },
    {
        "company_entity_id": "ENT-TSMC",
        "company": "Taiwan Semiconductor Manufacturing Company Limited",
        "document_type": "Q2 financial statements and management report",
        "period_end": "2026-06-30",
        "publication_or_filing_date": "2026-07-16",
        "source_id": "SRC-0110",
        "audit_date": "2026-07-25",
        "currency_status": "latest_official_quarterly_financials_available",
        "notes": "Unaudited quarterly package is newer than the 2025 Form 20-F; the official transcript is retained separately for capacity-planning and packaging statements.",
    },
    {
        "company_entity_id": "ENT-MU",
        "company": "Micron Technology Inc.",
        "document_type": "Form 10-Q",
        "period_end": "2026-05-28",
        "publication_or_filing_date": "2026-06-25",
        "source_id": "SRC-0112",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "Prepared remarks dated one day earlier are retained only for the broader strategic-customer-agreement aggregate.",
    },
    {
        "company_entity_id": "ENT-SKHYNIX",
        "company": "SK hynix Inc.",
        "document_type": "Final ADS prospectus",
        "period_end": "2026-03-31",
        "publication_or_filing_date": "2026-07-09",
        "source_id": "SRC-0117",
        "audit_date": "2026-07-25",
        "currency_status": "latest_comprehensive_filed_package_available",
        "notes": "Second-quarter results were scheduled for July 29 and were not available at the freeze date; the prospectus is not a substitute for that future periodic release.",
    },
    {
        "company_entity_id": "ENT-SAMSUNG",
        "company": "Samsung Electronics Co. Ltd.",
        "document_type": "Q1 interim business report",
        "period_end": "2026-03-31",
        "publication_or_filing_date": "2026-05-15",
        "source_id": "SRC-0115",
        "audit_date": "2026-07-25",
        "currency_status": "latest_detailed_interim_report_available",
        "notes": "Second-quarter full results were scheduled for July 30; June investment guidance is newer but is a plan disclosure rather than a periodic financial report.",
    },
    {
        "company_entity_id": "ENT-ASML",
        "company": "ASML Holding NV",
        "document_type": "Form 6-K and statutory interim report",
        "period_end": "2026-06-28",
        "publication_or_filing_date": "2026-07-15",
        "source_id": "SRC-0114",
        "audit_date": "2026-07-25",
        "currency_status": "latest_interim_financial_package_available",
        "notes": "Current source for quarterly system and installed-base-management economics and current export-control and supply-chain risks.",
    },
    {
        "company_entity_id": "ENT-ANET",
        "company": "Arista Networks Inc.",
        "document_type": "Form 10-Q",
        "period_end": "2026-03-31",
        "publication_or_filing_date": "2026-05-06",
        "source_id": "SRC-0120",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "Current source for AI-network customer concentration, purchase commitments, supplier deposits, cancellation asymmetry and qualification risk.",
    },
    {
        "company_entity_id": "ENT-MRVL",
        "company": "Marvell Technology Inc.",
        "document_type": "Form 10-Q",
        "period_end": "2026-05-02",
        "publication_or_filing_date": "2026-05-28",
        "source_id": "SRC-0121",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "Current source for data-center revenue, optical and switching acquisitions, NVIDIA preferred equity, and subsequent wafer and substrate deposits.",
    },
    {
        "company_entity_id": "ENT-COHR",
        "company": "Coherent Corp.",
        "document_type": "Form 10-Q",
        "period_end": "2026-03-31",
        "publication_or_filing_date": "2026-05-06",
        "source_id": "SRC-0122",
        "audit_date": "2026-07-25",
        "currency_status": "latest_periodic_filing_available",
        "notes": "Current source for the NVIDIA equity and capacity relationship, data-center and communications revenue, and indium-phosphide expansion.",
    },
]


SERIAL_CONTROL_MAP = [
    {
        "sequence": 1,
        "control_point": "architecture_and_software",
        "primary_controller": "NVIDIA or another accelerator designer",
        "required_output": "qualified design software stack and customer workload compatibility",
        "documented_substitute": "not established in the Chapter 8 evidence package",
        "stop_condition": "a different physical accelerator is not an economic substitute if software and workloads cannot move",
        "source_claim_ids": "CLM-080013",
    },
    {
        "sequence": 2,
        "control_point": "logic_wafer",
        "primary_controller": "TSMC for disclosed AMD and Broadcom concentration; TSMC and Samsung named by NVIDIA",
        "required_output": "yielding product-specific dies on a qualified process",
        "documented_substitute": "no equivalent-volume alternate is documented for a named current product",
        "stop_condition": "no qualified die exists for downstream package assembly",
        "source_claim_ids": "CLM-080001;CLM-080007;CLM-080008",
    },
    {
        "sequence": 3,
        "control_point": "high_bandwidth_memory",
        "primary_controller": "SK hynix Micron and Samsung named in NVIDIA supplier perimeter",
        "required_output": "customer-qualified HBM generation and stack configuration",
        "documented_substitute": "supplier names are documented but architecture shares and qualified spare volume are not",
        "stop_condition": "logic die cannot become the intended accelerator package without qualified memory",
        "source_claim_ids": "CLM-080001;CLM-080011;CLM-080015",
    },
    {
        "sequence": 4,
        "control_point": "advanced_packaging",
        "primary_controller": "TSMC CoWoS ecosystem for the documented NVIDIA path",
        "required_output": "yielding logic-memory package with required bandwidth thermals and interconnect",
        "documented_substitute": "not documented at equivalent package type yield or commercial volume",
        "stop_condition": "completed frontend wafers and memory remain unshippable as a finished accelerator",
        "source_claim_ids": "CLM-080005;CLM-080013",
    },
    {
        "sequence": 5,
        "control_point": "substrates_and_package_components",
        "primary_controller": "counterparty allocation undisclosed",
        "required_output": "qualified substrate interposer power and package components",
        "documented_substitute": "not reconstructed",
        "stop_condition": "missing component prevents package or board completion",
        "source_claim_ids": "CLM-080013",
    },
    {
        "sequence": 6,
        "control_point": "assembly_test_and_system_integration",
        "primary_controller": "Hon Hai Wistron Fabrinet and other providers named or undisclosed by product",
        "required_output": "tested boards systems and racks",
        "documented_substitute": "supplier plurality exists but alternate product and site ramp is undisclosed",
        "stop_condition": "completed semiconductors remain inventory rather than accepted systems",
        "source_claim_ids": "CLM-080001;CLM-080007",
    },
    {
        "sequence": 7,
        "control_point": "network_fabric_and_optics",
        "primary_controller": "NVIDIA native networking plus aligned suppliers including Marvell and Coherent; Arista controls a separate integrated Ethernet platform",
        "required_output": "qualified switches NICs transceivers and fabric software",
        "documented_substitute": "strategic compatibility and nonexclusive supply are documented but equivalent qualified volume is not",
        "stop_condition": "accelerators cannot operate at intended cluster scale or utilization",
        "source_claim_ids": "CLM-080013;CLM-080019;CLM-080022;CLM-080024",
    },
    {
        "sequence": 8,
        "control_point": "rack_power_cooling_and_commissioning",
        "primary_controller": "rack integrators data-center operators utilities and customers",
        "required_output": "powered cooled commissioned and customer-accepted capacity",
        "documented_substitute": "site and project specific",
        "stop_condition": "finished systems cannot enter revenue-producing service",
        "source_claim_ids": "CLM-080013",
    },
]


WRITING_GATE_AUDIT = [
    {
        "gate_id": "WG-0801",
        "requirement": "Latest-source currency is verified for every core public company.",
        "status": "pass",
        "evidence": "Eleven-row filing currency audit covering NVIDIA AMD Broadcom TSMC Micron SK hynix Samsung ASML Arista Marvell and Coherent.",
        "retained_limitation": "Scheduled future periodic releases are not treated as available at the July 25 freeze date.",
    },
    {
        "gate_id": "WG-0802",
        "requirement": "At least one physical bill-of-dependencies map is frozen.",
        "status": "pass",
        "evidence": "Eight-stage serial map plus twenty-four provider and layer observations from architecture through powered deployment.",
        "retained_limitation": "Product-level bills of materials and supplier shares remain undisclosed.",
    },
    {
        "gate_id": "WG-0803",
        "requirement": "Manufacturing commitments are reconciled separately from inventory and capex.",
        "status": "pass",
        "evidence": "Nineteen-row capacity-finance ledger and eight nonadditive measurement reconciliations preserve commitments cash capex assets payables deposits RPO equity and visions separately.",
        "retained_limitation": "Supplier and project subledgers needed for a complete sources-and-uses bridge are not public.",
    },
    {
        "gate_id": "WG-0804",
        "requirement": "Foundry packaging and HBM control points each have primary evidence and explicit unknowns.",
        "status": "pass",
        "evidence": "TSMC AMD Broadcom NVIDIA Micron SK hynix and Samsung evidence linked to UNK-0802 through UNK-0808.",
        "retained_limitation": "Qualified spare capacity package-specific yields and time-to-substitute are not disclosed.",
    },
    {
        "gate_id": "WG-0805",
        "requirement": "NVIDIA AMD and Broadcom financing or incentive channels are separated from product revenue.",
        "status": "pass",
        "evidence": "NVIDIA supplier equity and capacity relationships are separated in Chapter 8; AMD warrants are CLM-020039 and CLM-020040; Broadcom Apollo rack ownership and backstop are CLM-020046 and CLM-040034 through CLM-040037.",
        "retained_limitation": "Customer-level purchase pricing and accounting allocation remain redacted or undisclosed.",
    },
    {
        "gate_id": "WG-0806",
        "requirement": "Geographic and export-control pathways are tied to dated rules and products.",
        "status": "pass",
        "evidence": "Three-row regulatory ledger distinguishes the April 2025 H20 episode the January 2026 H200 and MI325X policy and May 2026 D:5 and Macau headquarters guidance.",
        "retained_limitation": "Individual classifications applications license decisions and shipment quantities remain unresolved in UNK-0817.",
    },
    {
        "gate_id": "WG-0807",
        "requirement": "Supplier concentration is not inferred from unnamed commitments.",
        "status": "pass",
        "evidence": "Named supplier shares appear only where filed; unnamed NVIDIA Marvell Arista and other aggregates retain unallocated counterparties.",
        "retained_limitation": "The resulting maps contain explicit unallocated residuals rather than estimated market-share assignments.",
    },
    {
        "gate_id": "WG-0808",
        "requirement": "At least three demand-downside pathways identify who holds inventory capacity credit or residual-value exposure.",
        "status": "pass",
        "evidence": "Five-row loss ledger includes one quantified H20 episode three conditional semiconductor pathways and Arista's historically observed unquantified supplier-liability mechanism.",
        "retained_limitation": "Only the H20 episode has a quantified realized charge.",
    },
    {
        "gate_id": "WG-0809",
        "requirement": "Material statements are atomic and source-linked while contradictions and disclosure gaps remain visible.",
        "status": "pass",
        "evidence": "Thirty-one claims have evidence links; database validation reports no orphan claims and no Grade-A claim without a Grade-A source; seventeen unknowns and eight reconciliations remain frozen.",
        "retained_limitation": "A passed writing gate authorizes evidence-led drafting and does not resolve the retained unknowns or elevate Grade-B and Grade-C claims.",
    },
]


def query(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(sql, parameters)]


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
    connection.row_factory = sqlite3.Row
    dependencies = query(
        connection,
        """
        SELECT
            s.*,
            controller.canonical_name AS system_controller_name,
            dependent.canonical_name AS dependent_name,
            provider.canonical_name AS provider_canonical_name
        FROM semiconductor_physical_dependencies s
        LEFT JOIN entities controller
            ON controller.entity_id = s.system_controller_entity_id
        JOIN entities dependent
            ON dependent.entity_id = s.dependent_entity_id
        LEFT JOIN entities provider
            ON provider.entity_id = s.provider_entity_id
        ORDER BY s.semiconductor_dependency_id
        """,
    )
    finance = query(
        connection,
        """
        SELECT
            s.*,
            recipient.canonical_name AS recipient_name,
            provider.canonical_name AS provider_canonical_name
        FROM semiconductor_capacity_finance s
        JOIN entities recipient
            ON recipient.entity_id = s.recipient_entity_id
        LEFT JOIN entities provider
            ON provider.entity_id = s.provider_entity_id
        ORDER BY s.capacity_finance_id
        """,
    )
    losses = query(
        connection,
        """
        SELECT
            s.*,
            entity.canonical_name AS first_loss_bearing_entity_name
        FROM semiconductor_loss_episodes s
        LEFT JOIN entities entity
            ON entity.entity_id = s.first_loss_bearing_entity_id
        ORDER BY s.loss_episode_id
        """,
    )
    substitution = query(
        connection,
        """
        SELECT
            s.*,
            dependent.canonical_name AS dependent_name,
            incumbent.canonical_name AS incumbent_provider_canonical_name,
            u.question AS linked_unknown_question
        FROM semiconductor_substitution_constraints s
        JOIN entities dependent
            ON dependent.entity_id = s.dependent_entity_id
        LEFT JOIN entities incumbent
            ON incumbent.entity_id = s.incumbent_provider_entity_id
        LEFT JOIN unknowns u
            ON u.unknown_id = s.linked_unknown_id
        ORDER BY s.substitution_constraint_id
        """,
    )
    regulatory = query(
        connection,
        """
        SELECT *
        FROM semiconductor_regulatory_controls
        ORDER BY announcement_or_effective_date, regulatory_control_id
        """,
    )
    reconciliations = query(
        connection,
        """
        SELECT *
        FROM semiconductor_measurement_reconciliations
        ORDER BY reconciliation_id
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
        WHERE c.chapter_id = 'CH08'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(
        connection,
        """
        SELECT *
        FROM unknowns
        WHERE chapter_id = 'CH08'
        ORDER BY unknown_id
        """,
    )
    source_grade_mismatches = query(
        connection,
        """
        SELECT DISTINCT
            table_name,
            row_id,
            evidence_grade,
            claim_grade
        FROM (
            SELECT
                'semiconductor_physical_dependencies' AS table_name,
                s.semiconductor_dependency_id AS row_id,
                s.evidence_grade,
                c.evidence_grade AS claim_grade
            FROM semiconductor_physical_dependencies s
            JOIN claims c ON c.claim_id = s.source_claim_id
            UNION ALL
            SELECT
                'semiconductor_capacity_finance',
                s.capacity_finance_id,
                s.evidence_grade,
                c.evidence_grade
            FROM semiconductor_capacity_finance s
            JOIN claims c ON c.claim_id = s.source_claim_id
            UNION ALL
            SELECT
                'semiconductor_loss_episodes',
                s.loss_episode_id,
                s.evidence_grade,
                c.evidence_grade
            FROM semiconductor_loss_episodes s
            JOIN claims c ON c.claim_id = s.source_claim_id
            UNION ALL
            SELECT
                'semiconductor_substitution_constraints',
                s.substitution_constraint_id,
                s.evidence_grade,
                c.evidence_grade
            FROM semiconductor_substitution_constraints s
            JOIN claims c ON c.claim_id = s.source_claim_id
            UNION ALL
            SELECT
                'semiconductor_regulatory_controls',
                s.regulatory_control_id,
                s.evidence_grade,
                c.evidence_grade
            FROM semiconductor_regulatory_controls s
            JOIN claims c ON c.claim_id = s.source_claim_id
            UNION ALL
            SELECT
                'semiconductor_measurement_reconciliations',
                s.reconciliation_id,
                s.evidence_grade,
                c.evidence_grade
            FROM semiconductor_measurement_reconciliations s
            JOIN claims c ON c.claim_id = s.primary_claim_id
        )
        WHERE
            (evidence_grade = 'A' AND claim_grade <> 'A')
            OR (evidence_grade = 'B' AND claim_grade IN ('C', 'D'))
            OR (evidence_grade = 'C' AND claim_grade = 'D')
        ORDER BY table_name, row_id
        """,
    )
    connection.close()

    expected_counts = {
        "physical_dependencies": 24,
        "capacity_finance": 19,
        "loss_episodes": 5,
        "substitution_constraints": 15,
        "regulatory_controls": 3,
        "measurement_reconciliations": 8,
        "filing_currency_audit": 11,
        "serial_control_map": 8,
        "writing_gate_rows": 9,
        "chapter_claims": 31,
        "chapter_unknowns": 17,
    }
    actual_counts = {
        "physical_dependencies": len(dependencies),
        "capacity_finance": len(finance),
        "loss_episodes": len(losses),
        "substitution_constraints": len(substitution),
        "regulatory_controls": len(regulatory),
        "measurement_reconciliations": len(reconciliations),
        "filing_currency_audit": len(FILING_CURRENCY_AUDIT),
        "serial_control_map": len(SERIAL_CONTROL_MAP),
        "writing_gate_rows": len(WRITING_GATE_AUDIT),
        "chapter_claims": len(claims),
        "chapter_unknowns": len(unknowns),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected Chapter 8 row counts: {actual_counts}")
    if source_grade_mismatches:
        raise ValueError(f"Ledger evidence grades do not match claim grades: {source_grade_mismatches}")

    realized_losses = [row for row in losses if row["episode_status"] == "realized"]
    if len(realized_losses) != 1:
        raise ValueError(f"Expected one realized loss episode, found {len(realized_losses)}")
    if float(realized_losses[0]["recognized_charge_amount"]) != 4_500_000_000:
        raise ValueError("Unexpected realized H20 charge")

    vision_rows = [
        row
        for row in finance
        if "vision" in row["status_ladder"] or "guideline" in row["status_ladder"]
    ]
    if any(int(row["additive_eligible"]) for row in vision_rows):
        raise ValueError("Announced semiconductor visions cannot be additive-eligible")
    if any(int(row["additive_eligible"]) for row in reconciliations):
        raise ValueError("Measurement-reconciliation rows cannot be additive-eligible")

    claim_ids = {row["claim_id"] for row in claims}
    invalid_reconciliation_claims: list[tuple[str, str]] = []
    for row in reconciliations:
        corroborating = str(row["corroborating_claim_ids"] or "")
        for claim_id in [value for value in corroborating.split(";") if value]:
            if claim_id not in claim_ids:
                invalid_reconciliation_claims.append((row["reconciliation_id"], claim_id))
    if invalid_reconciliation_claims:
        raise ValueError(
            f"Reconciliation rows reference claims outside Chapter 8: {invalid_reconciliation_claims}"
        )
    if any(row["status"] != "pass" for row in WRITING_GATE_AUDIT):
        raise ValueError("Chapter 8 writing gate has not passed")

    outputs = {
        "semiconductor_physical_dependencies.csv": dependencies,
        "semiconductor_capacity_finance.csv": finance,
        "semiconductor_loss_episodes.csv": losses,
        "semiconductor_substitution_constraints.csv": substitution,
        "semiconductor_regulatory_controls.csv": regulatory,
        "semiconductor_measurement_reconciliations.csv": reconciliations,
        "serial_physical_control_map.csv": SERIAL_CONTROL_MAP,
        "filing_currency_audit.csv": FILING_CURRENCY_AUDIT,
        "writing_gate_audit.csv": WRITING_GATE_AUDIT,
        "chapter_08_claims.csv": claims,
        "chapter_08_unknowns.csv": unknowns,
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
        **actual_counts,
        "realized_loss_episodes": len(realized_losses),
        "non_realized_or_unquantified_loss_pathways": len(losses) - len(realized_losses),
        "realized_h20_charge_usd": float(realized_losses[0]["recognized_charge_amount"]),
        "finance_aggregate_total": None,
        "finance_aggregate_total_status": "intentionally_not_calculated",
        "source_grade_mismatch_rows": len(source_grade_mismatches),
        "writing_gate_status": "passed",
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "limitations": [
            "Named suppliers are not supplier shares and do not establish interchangeable qualified capacity.",
            "Wafer starts, completed dies, HBM bits, HBM stacks, advanced packages, boards, racks and powered deployments are different physical states.",
            "Manufacturing commitments, cash capex, ending PP&E, supplier payables, customer deposits, RPO, public-equity proceeds and investment visions are different financial states.",
            "Currency, date, accounting perimeter, overlap and status differ across capacity-finance rows; no cross-row finance total is calculated.",
            "Only the H20 export-control charge is a quantified realized loss episode; other rows are conditional pathways or an unquantified historically observed mechanism.",
            "Arista's historically observed supplier-liability mechanism is unquantified and is not counted as a discrete realized AI loss episode.",
            "Current public disclosure does not support architecture-specific supplier shares, package-specific spare capacity, deposit refundability or time-to-substitute estimates.",
            "The networking and optics layer now has primary-source counterparties and finance channels but still lacks a product-level bill of materials and equivalent-volume substitution schedule.",
            "The substrate and process-specific lithography control layers remain incomplete and must be researched before the writing gate can pass.",
            "Regulatory rows preserve product and policy dates; they are not evidence that an individual license was issued or a shipment occurred.",
            "Reconciliation rows explain measurement or timing differences and are deliberately nonadditive; they do not resolve the retained contract-level unknowns.",
        ],
    }
    (OUTPUT / "chapter_08_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
