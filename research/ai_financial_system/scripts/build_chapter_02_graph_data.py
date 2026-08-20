#!/usr/bin/env python3
"""Freeze reconciled Chapter 2 capital-flow and dependency-graph datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "ai_financial_system"
DATABASE = ROOT / "reports" / "ai_financial_system" / "research.db"
OUTPUT = ROOT / "reports" / "ai_financial_system" / "chapters" / "chapter_02" / "data"


def rows(connection: sqlite3.Connection, query: str) -> list[dict]:
    return [dict(row) for row in connection.execute(query)]


def write_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def load_policy() -> dict[str, tuple[str, set[str]]]:
    policy: dict[str, tuple[str, set[str]]] = {}
    path = RESEARCH / "data" / "graph_selection_policy.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            policy[row["overlap_group"]] = (
                row["selection_mode"],
                {value for value in row["selected_event_ids"].split(";") if value},
            )
    return policy


def selected_funding(connection: sqlite3.Connection) -> list[dict]:
    policy = load_policy()
    events = rows(
        connection,
        """
        SELECT f.*, c.evidence_grade, c.confidence
        FROM funding_timeline f
        JOIN claims c ON c.claim_id = f.source_claim_id
        WHERE f.investor_entity_id IS NOT NULL
        ORDER BY f.event_date, f.funding_event_id
        """,
    )
    selected: list[dict] = []
    for event in events:
        group = event["overlap_group"]
        if group in policy:
            mode, event_ids = policy[group]
            if mode == "exclude" or event["funding_event_id"] not in event_ids:
                continue
        elif not event["additive_eligible"]:
            continue
        selected.append(event)

    expected = {
        "OAI-ROUND-2025": ("USD", 41_000_000_000),
        "OAI-ROUND-2026": ("USD", 122_000_000_000),
        "ANTH-SERIES-H-2026": ("USD", 65_000_000_000),
        "MISTRAL-SERIES-C-2025": ("EUR", 1_700_000_000),
        "XAI-SERIES-E-2026": ("USD", 20_000_000_000),
        "COHERE-2025-ROUND": ("USD", 600_000_000),
    }
    for group, (currency, expected_total) in expected.items():
        group_events = [event for event in selected if event["overlap_group"] == group]
        currencies = {event["currency"] for event in group_events}
        total = sum(float(event["committed_amount"] or 0) for event in group_events)
        if currencies != {currency} or total != expected_total:
            raise ValueError(
                f"{group} reconciliation failed: currencies={currencies}, total={total}"
            )
    return selected


def capital_edges(connection: sqlite3.Connection) -> list[dict]:
    names = {
        row["entity_id"]: row["canonical_name"]
        for row in rows(connection, "SELECT entity_id, canonical_name FROM entities")
    }
    output: list[dict] = []
    for event in selected_funding(connection):
        for basis, field in (("committed", "committed_amount"), ("funded", "funded_amount")):
            amount = event[field]
            if amount is None or float(amount) <= 0:
                continue
            output.append(
                {
                    "edge_id": f'{event["funding_event_id"]}-{basis}',
                    "basis": basis,
                    "from_entity_id": event["investor_entity_id"],
                    "from_entity": names[event["investor_entity_id"]],
                    "to_entity_id": event["recipient_entity_id"],
                    "to_entity": names[event["recipient_entity_id"]],
                    "relationship": event["instrument_type"],
                    "amount": amount,
                    "currency": event["currency"],
                    "amount_billions": float(amount) / 1_000_000_000,
                    "status": event["funding_status"],
                    "measurement_class": "capital_flow_or_commitment",
                    "cash_eligible": 1 if basis == "funded" else 0,
                    "source_table": "funding_timeline",
                    "source_record_id": event["funding_event_id"],
                    "source_claim_id": event["source_claim_id"],
                    "evidence_grade": event["evidence_grade"],
                    "confidence": event["confidence"],
                    "notes": event["notes"],
                }
            )

    explicit = rows(
        connection,
        """
        SELECT f.*, c.evidence_grade, c.confidence
        FROM financial_flows f JOIN claims c ON c.claim_id = f.source_claim_id
        ORDER BY f.flow_id
        """,
    )
    for flow in explicit:
        if flow["amount"] is None or float(flow["amount"]) <= 0:
            continue
        funded = "funded" in flow["amount_status"] or "drawn" in flow["amount_status"]
        noncash = "noncash" in flow["amount_status"]
        output.append(
            {
                "edge_id": flow["flow_id"],
                "basis": "funded_or_recognized" if funded else "committed_or_announced",
                "from_entity_id": flow["from_entity_id"],
                "from_entity": names[flow["from_entity_id"]],
                "to_entity_id": flow["to_entity_id"],
                "to_entity": names[flow["to_entity_id"]],
                "relationship": flow["instrument"],
                "amount": flow["amount"],
                "currency": flow["currency"],
                "amount_billions": float(flow["amount"]) / 1_000_000_000,
                "status": flow["amount_status"],
                "measurement_class": "noncash_asset_transfer" if noncash else "project_or_transaction_flow",
                "cash_eligible": 1 if funded and not noncash else 0,
                "source_table": "financial_flows",
                "source_record_id": flow["flow_id"],
                "source_claim_id": flow["source_claim_id"],
                "evidence_grade": flow["evidence_grade"],
                "confidence": flow["confidence"],
                "notes": flow["notes"],
            }
        )
    return output


def dependency_edges(connection: sqlite3.Connection) -> list[dict]:
    output: list[dict] = []
    specifications = [
        (
            "dependencies",
            "dependency_id",
            "dependent_entity_id",
            "provider_entity_id",
            "dependency_type",
            "concentration_measure",
            "substitution_difficulty",
            "source_claim_id",
        ),
        (
            "guarantees",
            "guarantee_id",
            "guarantor_entity_id",
            "beneficiary_entity_id",
            "guarantee_type",
            "maximum_exposure",
            "contingent_status",
            "source_claim_id",
        ),
        (
            "contracts",
            "contract_id",
            "party_a_entity_id",
            "party_b_entity_id",
            "contract_type",
            "stated_value",
            "commitment_status",
            "source_claim_id",
        ),
        (
            "compute_supply",
            "supply_id",
            "supplier_entity_id",
            "customer_entity_id",
            "product_or_service",
            "quantity_text",
            "contract_status",
            "source_claim_id",
        ),
        (
            "ownership",
            "ownership_id",
            "owner_entity_id",
            "owned_entity_id",
            "instrument_type",
            "economic_percentage",
            "control_rights",
            "source_claim_id",
        ),
        (
            "power_agreements",
            "power_id",
            "seller_entity_id",
            "buyer_entity_id",
            "agreement_type",
            "capacity_mw",
            "commitment_status",
            "source_claim_id",
        ),
    ]
    for table, id_col, from_col, to_col, relation_col, amount_col, status_col, claim_col in specifications:
        query = f"""
            SELECT t.{id_col} AS record_id, t.{from_col} AS from_id,
                   t.{to_col} AS to_id, t.{relation_col} AS relationship,
                   t.{amount_col} AS exposure, t.{status_col} AS status,
                   t.{claim_col} AS claim_id, c.evidence_grade, c.confidence
            FROM {table} t JOIN claims c ON c.claim_id = t.{claim_col}
            WHERE t.{from_col} IS NOT NULL AND t.{to_col} IS NOT NULL
            ORDER BY t.{id_col}
        """
        for record in rows(connection, query):
            output.append(
                {
                    "edge_id": record["record_id"],
                    "from_entity_id": record["from_id"],
                    "to_entity_id": record["to_id"],
                    "relationship": record["relationship"],
                    "exposure_or_capacity": record["exposure"],
                    "status_or_severity": record["status"],
                    "source_table": table,
                    "source_claim_id": record["claim_id"],
                    "evidence_grade": record["evidence_grade"],
                    "confidence": record["confidence"],
                }
            )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        capital = capital_edges(connection)
        dependencies = dependency_edges(connection)
    finally:
        connection.close()

    capital_columns = list(capital[0])
    dependency_columns = list(dependencies[0])
    write_csv(OUTPUT / "capital_edges_all_bases.csv", capital, capital_columns)
    write_csv(
        OUTPUT / "capital_sankey_committed.csv",
        [row for row in capital if row["basis"] == "committed" and row["currency"] == "USD"],
        capital_columns,
    )
    write_csv(
        OUTPUT / "capital_sankey_funded_cash.csv",
        [
            row
            for row in capital
            if row["currency"] == "USD"
            and row["cash_eligible"] == 1
            and row["basis"] in {"funded", "funded_or_recognized"}
        ],
        capital_columns,
    )
    write_csv(OUTPUT / "dependency_graph_edges.csv", dependencies, dependency_columns)
    summary: dict[tuple[str, str], dict] = {}
    for row in capital:
        if row["source_table"] != "funding_timeline":
            continue
        key = (row["to_entity_id"], row["currency"])
        current = summary.setdefault(
            key,
            {
                "recipient_entity_id": row["to_entity_id"],
                "recipient": row["to_entity"],
                "currency": row["currency"],
                "selected_committed": 0.0,
                "verified_funded_floor": 0.0,
                "unverified_or_future_gap": 0.0,
            },
        )
        if row["basis"] == "committed":
            current["selected_committed"] += float(row["amount"])
        elif row["basis"] == "funded":
            current["verified_funded_floor"] += float(row["amount"])
    for current in summary.values():
        current["unverified_or_future_gap"] = (
            current["selected_committed"] - current["verified_funded_floor"]
        )
    summary_rows = sorted(
        summary.values(),
        key=lambda row: (row["currency"], -row["selected_committed"], row["recipient"]),
    )
    summary_columns = list(summary_rows[0])
    write_csv(OUTPUT / "capital_summary_by_recipient.csv", summary_rows, summary_columns)

    output_paths = sorted(OUTPUT.glob("*.csv"))
    manifest = {
        "database_sha256": hashlib.sha256(DATABASE.read_bytes()).hexdigest(),
        "selection_policy_sha256": hashlib.sha256(
            (RESEARCH / "data" / "graph_selection_policy.csv").read_bytes()
        ).hexdigest(),
        "row_counts": {
            "capital_edges_all_bases": len(capital),
            "capital_sankey_committed": sum(
                row["basis"] == "committed" and row["currency"] == "USD"
                for row in capital
            ),
            "capital_sankey_funded_cash": sum(
                row["currency"] == "USD"
                and row["cash_eligible"] == 1
                and row["basis"] in {"funded", "funded_or_recognized"}
                for row in capital
            ),
            "dependency_graph_edges": len(dependencies),
            "capital_summary_by_recipient": len(summary_rows),
        },
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_paths
        },
        "reconciliation_rule": (
            "Round aggregates and investor components are never both selected. "
            "Committed and verified funded capital remain separate."
        ),
    }
    (OUTPUT / "graph_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
