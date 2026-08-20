#!/usr/bin/env python3
"""Build a reproducible, non-additive relationship-overlap audit.

The audit counts ledger classes and counterparties, not dollars. A relationship
record can describe a legal, operating, ownership, funding, or support role.
The outputs therefore measure documented role overlap in the retained research
population; they are not exposure, centrality, prevalence, or loss estimates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "reports/ai_financial_system/research.db"
OUTPUT = ROOT / "reports/ai_financial_system/relationship_overlap_audit"

LEDGERS = (
    ("dependency", "dependencies", "dependent_entity_id", "provider_entity_id"),
    ("guarantee", "guarantees", "guarantor_entity_id", "beneficiary_entity_id"),
    ("contract", "contracts", "party_a_entity_id", "party_b_entity_id"),
    ("compute", "compute_supply", "supplier_entity_id", "customer_entity_id"),
    ("ownership", "ownership", "owner_entity_id", "owned_entity_id"),
    ("power", "power_agreements", "seller_entity_id", "buyer_entity_id"),
    ("funding", "funding_timeline", "investor_entity_id", "recipient_entity_id"),
    ("flow", "financial_flows", "from_entity_id", "to_entity_id"),
    ("debt", "debt_instruments", "issuer_entity_id", "lender_or_holder_entity_id"),
    ("lease", "leases", "lessee_entity_id", "lessor_entity_id"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    names = {
        row["entity_id"]: row["canonical_name"]
        for row in connection.execute("SELECT entity_id, canonical_name FROM entities")
    }

    edges: list[tuple[str, str, str]] = []
    for ledger, table, left, right in LEDGERS:
        query = (
            f"SELECT {left} AS left_id, {right} AS right_id FROM {table} "
            f"WHERE {left} IS NOT NULL AND {right} IS NOT NULL"
        )
        for row in connection.execute(query):
            if row["left_id"] != row["right_id"]:
                edges.append((ledger, row["left_id"], row["right_id"]))
    connection.close()

    pair_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_records: dict[tuple[str, str], int] = defaultdict(int)
    entity_types: dict[str, set[str]] = defaultdict(set)
    entity_neighbors: dict[str, set[str]] = defaultdict(set)
    entity_records: dict[str, int] = defaultdict(int)

    for ledger, left, right in edges:
        pair = tuple(sorted((left, right)))
        pair_types[pair].add(ledger)
        pair_records[pair] += 1
        for entity, counterparty in ((left, right), (right, left)):
            entity_types[entity].add(ledger)
            entity_neighbors[entity].add(counterparty)
            entity_records[entity] += 1

    pair_rows = [
        {
            "entity_a_id": pair[0],
            "entity_a": names[pair[0]],
            "entity_b_id": pair[1],
            "entity_b": names[pair[1]],
            "relationship_class_count": len(pair_types[pair]),
            "relationship_record_count": pair_records[pair],
            "relationship_classes": ";".join(sorted(pair_types[pair])),
        }
        for pair in pair_types
        if len(pair_types[pair]) >= 2
    ]
    pair_rows.sort(
        key=lambda row: (
            -int(row["relationship_class_count"]),
            -int(row["relationship_record_count"]),
            str(row["entity_a"]),
            str(row["entity_b"]),
        )
    )

    entity_rows = [
        {
            "entity_id": entity,
            "entity": names[entity],
            "relationship_class_count": len(entity_types[entity]),
            "counterparty_count": len(entity_neighbors[entity]),
            "relationship_record_count": entity_records[entity],
            "relationship_classes": ";".join(sorted(entity_types[entity])),
        }
        for entity in entity_types
        if len(entity_types[entity]) >= 4
    ]
    entity_rows.sort(
        key=lambda row: (
            -int(row["relationship_class_count"]),
            -int(row["counterparty_count"]),
            -int(row["relationship_record_count"]),
            str(row["entity"]),
        )
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs_path = OUTPUT / "multi_ledger_entity_pairs.csv"
    entities_path = OUTPUT / "multi_role_entities.csv"
    write_csv(pairs_path, list(pair_rows[0]), pair_rows)
    write_csv(entities_path, list(entity_rows[0]), entity_rows)

    summary = {
        "status": "PASS_NONADDITIVE_RELATIONSHIP_OVERLAP_AUDIT",
        "database_sha256": sha256(DATABASE),
        "method": {
            "ledger_classes": [ledger for ledger, *_ in LEDGERS],
            "unit": "retained bilateral relationship record",
            "pair_rule": "unordered canonical entity pair",
            "exclusions": "null endpoints and self-relations",
            "interpretation_boundary": (
                "Counts document role overlap in the retained case-study population. "
                "They are not dollar exposure, prevalence, causal centrality, or loss estimates."
            ),
        },
        "population": {
            "relationship_records": len(edges),
            "entities": len(entity_types),
            "unique_entity_pairs": len(pair_types),
            "pairs_with_at_least_2_classes": sum(
                len(classes) >= 2 for classes in pair_types.values()
            ),
            "pairs_with_at_least_3_classes": sum(
                len(classes) >= 3 for classes in pair_types.values()
            ),
            "pairs_with_at_least_4_classes": sum(
                len(classes) >= 4 for classes in pair_types.values()
            ),
            "entities_with_at_least_5_classes": sum(
                len(classes) >= 5 for classes in entity_types.values()
            ),
        },
        "outputs": {
            pairs_path.name: sha256(pairs_path),
            entities_path.name: sha256(entities_path),
        },
    }
    summary_path = OUTPUT / "relationship_overlap_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
