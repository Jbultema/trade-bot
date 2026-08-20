#!/usr/bin/env python3
"""Dedicated acceptance validator for the Chapter 10 rebuilt-v3 packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AI = ROOT / "research/ai_financial_system"
DB = ROOT / "reports/ai_financial_system/research.db"
V1 = AI / "staging/lane_1_chapter_10_rebuilt_v1"
REVIEW = AI / "staging/lane_2_verify_chapter_10_rebuilt_v1"
V2_AUTHOR = AI / "staging/lane_3_chapter_10_rebuilt_v2"
V2_REVIEW = AI / "staging/lane_4_verify_chapter_10_rebuilt_v2"
BASELINE_SHA = "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
V1_MANIFEST_SHA = "0b387ceaee5ed81706b5400f3ec391950e6af99c9b36e6208f78b6dbcee347f6"
REVIEW_MANIFEST_SHA = "04e96776fee3576bc9de2144e53fe095946bc2ee2b2b0b1dea7da97cb6874f49"
V2_AUTHOR_MANIFEST_SHA = "ab0108513e9ceb9d9fa6e532133405cb4f8b2fcad96b212b44209c28347207f0"
V2_REVIEW_MANIFEST_SHA = "d5d9bc7ae7cf28da974699adff6ed7297b627b868b8440f16af4cae02951c890"

EXACT_V1_FILES = {
    "proposed_atomic_claims.csv": 26,
    "proposed_evidence_links.csv": 43,
    "proposed_sources.csv": 27,
    "proposed_unknowns.csv": 15,
    "proposed_handoffs.csv": 3,
    "proposed_confidence_history.csv": 26,
    "proposed_economic_importance.csv": 26,
    "public_capital_awards.csv": 10,
    "public_tax_incentives.csv": 5,
    "public_equity_positions.csv": 4,
    "public_infrastructure_cost_allocation.csv": 4,
    "sovereign_capital_cases.csv": 6,
    "public_capital_measurement_reconciliations.csv": 10,
    "risk_pathways.csv": 5,
    "arithmetic_nonadditivity_controls.csv": 7,
    "contradiction_log.csv": 10,
    "dependency_map.csv": 29,
}

REQUIRED_TIMELINE_CLASSES = {
    "award_legal_state",
    "sovereign_measurement",
    "tax_legal_transition",
    "tax_effective_transition",
    "equity_legal_transition",
    "equity_disclosure_transition",
    "equity_closing_transition",
    "cash_measurement",
    "equity_measurement",
    "accounting_measurement",
    "infrastructure_filing",
    "infrastructure_order",
    "sovereign_policy_transition",
}

ALLOWED_FIELD_DISPOSITIONS = {
    "canonical_destination",
    "typed_junction_destination",
    "governed_hold",
    "duplicate_reference",
    "rejected_not_canonical",
}
ALLOWED_SEMANTIC_VERDICTS = {"supported", "narrowed", "held", "rejected"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(target: Path, name: str) -> list[dict[str, str]]:
    path = target / name
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError(f"invalid header: {name}")
        result = list(reader)
    if any(None in row for row in result):
        raise ValueError(f"malformed row: {name}")
    return result


def csv_identity(rows_: list[dict[str, str]], key: str) -> bool:
    identities = [row[key] for row in rows_]
    return all(identities) and len(identities) == len(set(identities))


def check_path_identity(
    target: Path,
    row: dict[str, str],
    path_field: str,
    hash_field: str,
    bytes_field: str | None = None,
) -> bool:
    relative = row[path_field]
    if not relative:
        return False
    path = target / relative
    if not path.is_file() or sha(path) != row[hash_field]:
        return False
    return bytes_field is None or str(path.stat().st_size) == row[bytes_field]


def exact_db_rows(
    connection: sqlite3.Connection,
    table: str,
    key: str,
    packet_rows: list[dict[str, str]],
) -> bool:
    connection.row_factory = sqlite3.Row
    for packet in packet_rows:
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {key}=?",  # noqa: S608
            (packet[key],),
        ).fetchone()
        if row is None:
            return False
        for column in row.keys():  # noqa: SIM118
            expected = "" if row[column] is None else str(row[column])
            if packet.get(column) != expected:
                return False
    return True


def exact_db_evidence_links(
    connection: sqlite3.Connection,
    packet_rows: list[dict[str, str]],
) -> bool:
    connection.row_factory = sqlite3.Row
    for packet in packet_rows:
        row = connection.execute(
            "SELECT * FROM evidence_links "
            "WHERE claim_id=? AND source_id=? AND locator=?",
            (packet["claim_id"], packet["source_id"], packet["locator"]),
        ).fetchone()
        if row is None:
            return False
        for column in row.keys():  # noqa: SIM118
            expected = "" if row[column] is None else str(row[column])
            if packet.get(column) != expected:
                return False
    return True


def record(
    checks: dict[str, dict[str, Any]],
    check_id: str,
    passed: bool,
    detail: str,
) -> None:
    checks[check_id] = {"passed": bool(passed), "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=HERE)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    target = args.target.resolve()
    checks: dict[str, dict[str, Any]] = {}

    record(
        checks,
        "protected.canonical_database_hash",
        sha(DB) == BASELINE_SHA,
        f"expected={BASELINE_SHA}",
    )
    record(
        checks,
        "protected.v1_manifest_hash",
        sha(V1 / "lane_manifest.json") == V1_MANIFEST_SHA,
        f"expected={V1_MANIFEST_SHA}",
    )
    record(
        checks,
        "protected.review_manifest_hash",
        sha(REVIEW / "lane_manifest.json") == REVIEW_MANIFEST_SHA,
        f"expected={REVIEW_MANIFEST_SHA}",
    )
    record(
        checks,
        "protected.v2_author_manifest_hash",
        sha(V2_AUTHOR / "lane_manifest.json") == V2_AUTHOR_MANIFEST_SHA,
        f"expected={V2_AUTHOR_MANIFEST_SHA}",
    )
    record(
        checks,
        "protected.v2_review_manifest_hash",
        sha(V2_REVIEW / "lane_manifest.json") == V2_REVIEW_MANIFEST_SHA,
        f"expected={V2_REVIEW_MANIFEST_SHA}",
    )
    connection = sqlite3.connect(DB)
    record(
        checks,
        "protected.database_integrity",
        connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
        "PRAGMA integrity_check",
    )
    record(
        checks,
        "protected.database_foreign_keys",
        not list(connection.execute("PRAGMA foreign_key_check")),
        "PRAGMA foreign_key_check",
    )

    exact_counts = {
        name: len(rows(target, name))
        for name in EXACT_V1_FILES
    }
    exact_bytes = {
        name: sha(target / name) == sha(V1 / name)
        for name in EXACT_V1_FILES
    }
    record(
        checks,
        "preservation.exact_v1_projections",
        exact_counts == EXACT_V1_FILES and all(exact_bytes.values()),
        f"counts={exact_counts}; exact_byte_files={sum(exact_bytes.values())}",
    )

    claims = rows(target, "proposed_atomic_claims.csv")
    links = rows(target, "proposed_evidence_links.csv")
    sources = rows(target, "proposed_sources.csv")
    claim_ids = {row["claim_id"] for row in claims}
    source_ids = {row["source_id"] for row in sources}
    record(
        checks,
        "preservation.exact_canonical_database_rows",
        exact_db_rows(connection, "claims", "claim_id", claims)
        and exact_db_evidence_links(connection, links)
        and exact_db_rows(connection, "sources", "source_id", sources),
        "26 parent claims, 43 evidence links, and 27 sources match canonical rows",
    )
    connection.close()
    record(
        checks,
        "preservation.canonical_fk_and_identity",
        csv_identity(claims, "claim_id")
        and csv_identity(links, "evidence_link_id")
        and csv_identity(sources, "source_id")
        and all(
            link["claim_id"] in claim_ids and link["source_id"] in source_ids
            for link in links
        )
        and claim_ids == {link["claim_id"] for link in links},
        "canonical identities unique and all 26 parents support-linked",
    )

    typed_rows = rows(target, "typed_row_dispositions.csv")
    typed_fields = rows(target, "typed_field_dispositions.csv")
    original_typed_rows = rows(V1, "typed_row_dispositions.csv")
    original_typed_fields = rows(V1, "typed_field_dispositions.csv")
    original_source_files = sorted(
        {row["source_file"] for row in original_typed_rows}
    )
    inventory_bytes = all(
        sha(target / "source_inventory" / name) == sha(V1 / name)
        for name in original_source_files
    )
    transformed_row_projection = [
        {
            **row,
            "source_file": row["source_file"].removeprefix("source_inventory/"),
            "row_disposition": "preserved_rebuilt_v1_source_inventory_row",
            "publication_status": "held_author_v2_inventory_not_publication",
        }
        for row in typed_rows
    ]
    record(
        checks,
        "P003.lossless_typed_inventory",
        len(typed_rows) == len(original_typed_rows) == 411
        and len(typed_fields) == len(original_typed_fields) == 4188
        and inventory_bytes
        and all(
            transformed_row_projection[index]["typed_row_audit_id"]
            == original_typed_rows[index]["typed_row_audit_id"]
            for index in range(411)
        ),
        (
            f"rows={len(typed_rows)}; scalars={len(typed_fields)}; "
            f"source_files={len(original_source_files)}"
        ),
    )
    original_field_projection = [
        (
            row["field_audit_id"],
            row["typed_row_audit_id"],
            row["source_file"],
            row["source_row_ordinal"],
            row["record_id"],
            row["field"],
            row["source_value"],
            row["value_state"],
        )
        for row in original_typed_fields
    ]
    v2_field_projection = [
        (
            row["field_audit_id"],
            row["typed_row_audit_id"],
            row["source_file"].removeprefix("source_inventory/"),
            row["source_row_ordinal"],
            row["record_id"],
            row["field"],
            row["source_value"],
            row["value_state"],
        )
        for row in typed_fields
    ]
    row_ids = {row["typed_row_audit_id"] for row in typed_rows}
    blank_fields = [row for row in typed_fields if row["source_value"] == ""]
    critical_blank_fields = [
        row
        for row in blank_fields
        if row["field"]
        in {
            "source_version_id",
            "external_projection_reference",
            "resolved_locator",
            "semantic_verdict",
        }
    ]
    record(
        checks,
        "P003.field_identity_and_values",
        original_field_projection == v2_field_projection
        and csv_identity(typed_fields, "field_audit_id")
        and all(row["typed_row_audit_id"] in row_ids for row in typed_fields),
        "every original scalar identity, value, and blank state is retained",
    )
    record(
        checks,
        "P003.field_specific_dispositions",
        all(
            row["field_disposition"] in ALLOWED_FIELD_DISPOSITIONS
            and row["narrowing_or_hold"]
            and row["blank_decision"]
            for row in typed_fields
        )
        and len(blank_fields) == 585
        and len(critical_blank_fields) == 140
        and all(
            row["field_disposition"] == "governed_hold"
            and not row["destination"]
            and "not_zero_false" in (
                row["narrowing_or_hold"] + row["blank_decision"]
            )
            for row in critical_blank_fields
        )
        and all(
            row["field_disposition"] != "governed_hold" or not row["destination"]
            for row in typed_fields
        ),
        (
            f"blank_scalars={len(blank_fields)}; "
            f"critical_governed_holds={len(critical_blank_fields)}; "
            f"dispositions={dict(Counter(row['field_disposition'] for row in typed_fields))}"
        ),
    )

    versions = rows(target, "normalized_source_versions.csv")
    references = rows(target, "source_version_references.csv")
    exceptions = rows(target, "source_exceptions.csv")
    derivatives = rows(target, "source_derivatives.csv")
    version_by_id = {row["source_version_id"]: row for row in versions}
    reference_by_source = {row["source_id"]: row for row in references}
    selected = [
        row for row in versions if row["selected_governing_version"] == "1"
    ]
    preserved = [
        row for row in selected if row["preservation_status"] == "preserved_complete"
    ]
    held = [
        row
        for row in selected
        if row["preservation_status"] == "retrieval_or_validity_hold"
    ]
    original_urls = {row["source_id"]: row["url"] for row in sources}
    derivative_by_version = {
        row["source_version_id"]: row for row in derivatives
    }
    preserved_valid = all(
        check_path_identity(
            target,
            version,
            "local_archive_path",
            "content_sha256",
            "byte_count",
        )
        and version["visible_text_path"]
        and (target / version["visible_text_path"]).is_file()
        and version["rights_access_boundary"]
        and version["response_headers_path"]
        and version["transfer_metadata_path"]
        and version["source_version_id"] in derivative_by_version
        and check_path_identity(
            target,
            derivative_by_version[version["source_version_id"]],
            "derivative_path",
            "derivative_sha256",
            "derivative_bytes",
        )
        for version in preserved
    )
    record(
        checks,
        "P001.normalized_source_versions",
        len(versions) == 37
        and csv_identity(versions, "source_version_id")
        and len(references) == len(reference_by_source) == 27
        and set(reference_by_source) == source_ids
        and len(selected) == 27
        and len(preserved) == 24
        and len(held) == 3
        and preserved_valid
        and all(
            reference_by_source[row["source_id"]]["source_version_id"]
            == row["source_version_id"]
            and reference_by_source[row["source_id"]][
                "external_projection_reference"
            ]
            == f"normalized_source_versions.csv#{row['source_version_id']}"
            and reference_by_source[row["source_id"]]["rights_class"]
            and reference_by_source[row["source_id"]]["preservation_state"]
            == row["preservation_status"]
            for row in selected
        ),
        (
            f"attempts={len(versions)}; selected=27; "
            f"preserved={len(preserved)}; held={len(held)}"
        ),
    )
    failed_versions = {
        row["source_version_id"]
        for row in versions
        if row["preservation_status"] == "retrieval_or_validity_hold"
    }
    record(
        checks,
        "P001.append_only_failures_and_cited_urls",
        len(exceptions) == 13
        and {row["source_version_id"] for row in exceptions} == failed_versions
        and all(row["append_only_status"] == "retained_failure_not_overwritten" for row in exceptions)
        and all(
            any(
                version["source_id"] == source_id
                and version["version_role"] == "cited_original"
                and version["canonical_cited_url"] == url
                for version in versions
            )
            for source_id, url in original_urls.items()
        ),
        f"retained_failure_attempts={len(exceptions)}",
    )

    locator_rows = rows(target, "locator_semantic_dispositions.csv")
    locator_by_link = {row["evidence_link_id"]: row for row in locator_rows}
    verdict_counts = Counter(row["semantic_verdict"] for row in locator_rows)
    held_source_ids = {row["source_id"] for row in held}
    locator_join_valid = all(
        row["evidence_link_id"] in {link["evidence_link_id"] for link in links}
        and row["claim_id"] in claim_ids
        and row["source_id"] in source_ids
        and row["source_version_id"] in version_by_id
        and row["source_version_id"]
        == reference_by_source[row["source_id"]]["source_version_id"]
        and row["semantic_verdict"] in ALLOWED_SEMANTIC_VERDICTS
        and row["resolved_locator"]
        and row["required_narrowing"]
        and row["publication_eligibility"]
        == "held_pending_fresh_nonauthor_review"
        for row in locator_rows
    )
    record(
        checks,
        "P002.all_relationships_freshly_reviewed",
        len(locator_rows) == len(locator_by_link) == 43
        and set(locator_by_link) == {link["evidence_link_id"] for link in links}
        and locator_join_valid
        and verdict_counts
        == Counter({"supported": 36, "narrowed": 4, "held": 3})
        and all(
            row["semantic_verdict"] == "held"
            for row in locator_rows
            if row["source_id"] in held_source_ids
        )
        and all(
            row["review_excerpt_aid"]
            for row in locator_rows
            if row["semantic_verdict"] in {"supported", "narrowed"}
        ),
        f"verdicts={dict(verdict_counts)}",
    )

    investai = locator_by_link.get("EL-CH10-0039", {})
    investai_lines = (
        target
        / "source_derivatives"
        / "SRC-0155_v0001_visible_text.txt"
    ).read_text(encoding="utf-8").splitlines()
    expected_investai_locator = (
        "SRC-0155_v0001_visible_text.txt visible-text lines 1-5; "
        "visible-text lines 73-76"
    )
    expected_investai_excerpt = " ".join(
        investai_lines[0:5] + investai_lines[72:76]
    )
    child_edges_100024 = [
        row
        for row in rows(target, "atomic_child_evidence_links.csv")
        if row["parent_claim_id"] == "CLM-100024"
    ]
    exact_investai_replay = (
        investai.get("claim_id") == "CLM-100024"
        and investai.get("source_id") == "SRC-0155"
        and investai.get("resolved_locator") == expected_investai_locator
        and investai.get("review_excerpt_aid") == expected_investai_excerpt
        and "€200 billion" in expected_investai_excerpt
        and "€20 billion for AI gigafactories" in expected_investai_excerpt
        and investai.get("semantic_verdict") == "narrowed"
        and len(child_edges_100024) == 3
        and all(
            row["locator_disposition_id"]
            == investai.get("locator_disposition_id")
            and row["semantic_verdict"] == "narrowed"
            for row in child_edges_100024
        )
    )
    record(
        checks,
        "CH10-V2-P001.exact_statement_to_locator_replay",
        exact_investai_replay,
        (
            "composite lines 1-5 and 73-76; exact frozen excerpt contains "
            "200-billion title and included 20-billion facility statement"
        ),
    )

    external_claims = rows(target, "external_claim_dependencies.csv")
    external_links = rows(target, "external_claim_evidence_links.csv")
    record_junctions = rows(target, "record_claim_junctions.csv")
    external = external_claims[0] if len(external_claims) == 1 else {}
    external_link = external_links[0] if len(external_links) == 1 else {}
    canonical_external = sqlite3.connect(DB)
    canonical_external.row_factory = sqlite3.Row
    canonical_claim = canonical_external.execute(
        "SELECT * FROM claims WHERE claim_id='CLM-050004'"
    ).fetchone()
    canonical_evidence = canonical_external.execute(
        "SELECT * FROM evidence_links WHERE claim_id='CLM-050004' "
        "AND source_id='SRC-0087'"
    ).fetchone()
    canonical_external.close()
    exact_external = bool(canonical_claim) and all(
        external.get(key) == ("" if canonical_claim[key] is None else str(canonical_claim[key]))
        for key in canonical_claim.keys()  # noqa: SIM118
    )
    exact_external_link = bool(canonical_evidence) and (
        external_link.get("claim_id") == canonical_evidence["claim_id"]
        and external_link.get("source_id") == canonical_evidence["source_id"]
        and external_link.get("canonical_locator") == canonical_evidence["locator"]
        and external_link.get("support_type") == canonical_evidence["support_type"]
        and external_link.get("excerpt_note") == canonical_evidence["excerpt_note"]
    )
    record(
        checks,
        "P004.packet_local_external_claim_closure",
        exact_external
        and external_link.get("claim_id") == "CLM-050004"
        and external_link.get("source_id") == "SRC-0087"
        and external_link.get("source_version_id")
        == reference_by_source["SRC-0087"]["source_version_id"]
        and external_link.get("semantic_verdict") in {"supported", "narrowed"}
        and len(record_junctions) == 2
        and {row["record_id"] for row in record_junctions}
        == {"PIC-100004", "DEP-CH10-0023"}
        and all(
            row["claim_id"] == "CLM-050004"
            and row["closure_status"] == "PASS_PACKET_LOCAL_SOURCE_CHAIN_INCLUDED"
            for row in record_junctions
        ),
        (
            "CLM-050004 and SRC-0087 are included with typed junctions for "
            "PIC-100004 and DEP-CH10-0023"
        ),
    )
    record(
        checks,
        "P004.external_canonical_link_identity",
        exact_external_link,
        "external claim source, locator, support type, and excerpt match canonical link",
    )

    entities = rows(target, "entity_resolution.csv")
    aliases = rows(target, "entity_aliases.csv")
    entity_ids = {row["entity_id"] for row in entities}
    unresolved_aliases = [
        row for row in aliases if row["resolution_status"] == "governed_unresolved_hold"
    ]
    required_aliases = {
        "CoreWeave",
        "G42",
        "Mubadala",
        "Microsoft",
        "NVIDIA",
        "BlackRock GIP",
        "EIB",
        "EIF",
        "U.S. Department of Commerce",
        "European Commission",
        "Government of Canada",
    }
    record(
        checks,
        "P005.recursive_entity_fk_closure",
        csv_identity(entities, "entity_resolution_id")
        and csv_identity(aliases, "entity_alias_id")
        and {"ENT-ENTERGY", "ENT-META"}.issubset(entity_ids)
        and all(
            not row["parent_entity_id"] or row["parent_entity_id"] in entity_ids
            for row in entities
        )
        and all(
            row["canonical_entity_id"] in entity_ids
            for row in aliases
            if row["canonical_entity_id"]
        )
        and required_aliases.issubset({row["alias"] for row in aliases})
        and all(
            not row["canonical_entity_id"]
            and row["alias_provenance"]
            and row["resolution_note"]
            for row in unresolved_aliases
        ),
        (
            f"entities={len(entities)}; aliases={len(aliases)}; "
            f"unresolved_holds={len(unresolved_aliases)}"
        ),
    )

    children = rows(target, "atomic_child_claims.csv")
    child_links = rows(target, "atomic_child_evidence_links.csv")
    coverage = rows(target, "parent_child_coverage.csv")
    child_ids = {row["child_claim_id"] for row in children}
    linked_child_ids = {row["child_claim_id"] for row in child_links}
    record(
        checks,
        "P006.atomic_child_coverage",
        len(children) == 79
        and csv_identity(children, "child_claim_id")
        and len(coverage) == 26
        and {row["parent_claim_id"] for row in children} == claim_ids
        and {row["parent_claim_id"] for row in coverage} == claim_ids
        and child_ids == linked_child_ids
        and all(
            row["parent_claim_id"] in claim_ids
            and row["evidence_link_id"] in locator_by_link
            and row["source_version_id"]
            == locator_by_link[row["evidence_link_id"]]["source_version_id"]
            and row["locator_disposition_id"]
            == locator_by_link[row["evidence_link_id"]]["locator_disposition_id"]
            and row["semantic_verdict"]
            == locator_by_link[row["evidence_link_id"]]["semantic_verdict"]
            for row in child_links
        )
        and all(
            row["canonical_parent_preservation"]
            == "parent_text_and_id_retained_exactly"
            for row in children
        ),
        f"parents=26; atomic_children={len(children)}; child_evidence_edges={len(child_links)}",
    )

    falsifiers = rows(target, "falsifiers.csv")
    child_falsifiers = rows(target, "atomic_child_falsifiers.csv")
    child_by_id = {row["child_claim_id"]: row for row in children}
    parent_observations = {
        row["contradictory_document_or_observation"] for row in falsifiers
    }
    child_observations = {
        row["contradictory_document_or_observation"] for row in child_falsifiers
    }
    generic_template = "new transaction evidence"
    record(
        checks,
        "P007.claim_and_assertion_specific_falsifiers",
        len(falsifiers) == len(parent_observations) == 26
        and {row["claim_id"] for row in falsifiers} == claim_ids
        and all(
            row["frozen_trigger_id"]
            and row["deterministic_effect_if_observed"]
            and generic_template not in row["contradictory_document_or_observation"].casefold()
            for row in falsifiers
        )
        and len(child_falsifiers) == len(children)
        and {row["child_claim_id"] for row in child_falsifiers} == child_ids
        and len(child_observations) == len(child_falsifiers)
        and all(
            row["child_claim_id"] in child_by_id
            and child_by_id[row["child_claim_id"]]["atomic_assertion"]
            in row["contradictory_document_or_observation"]
            and "[reviewed source perimeter " in row["contradictory_document_or_observation"]
            and row["deterministic_effect_if_observed"]
            and "narrow or supersede only this child"
            not in row["deterministic_effect_if_observed"]
            for row in child_falsifiers
        ),
        (
            f"parent_falsifiers={len(falsifiers)}; "
            f"child_falsifiers={len(child_falsifiers)}"
        ),
    )

    timeline = rows(target, "timeline.csv")
    timeline_classes = {row["timeline_class"] for row in timeline}
    record(
        checks,
        "P008.complete_nonadditive_timeline",
        len(timeline) == 36
        and csv_identity(timeline, "timeline_id")
        and REQUIRED_TIMELINE_CLASSES.issubset(timeline_classes)
        and all(
            row["claim_id"] in claim_ids
            and row["source_id"] in source_ids
            and row["evidence_link_id"] in locator_by_link
            and row["event_date"]
            and row["nonadditivity_boundary"]
            for row in timeline
        ),
        f"events={len(timeline)}; classes={sorted(timeline_classes)}",
    )

    statuses = rows(target, "claim_review_status.csv")
    history = rows(target, "fresh_semantic_review_history.csv")
    inherited = {row["claim_id"]: row for row in claims}
    html_text = html.unescape(
        (target / "proposed_chapter.html").read_text(encoding="utf-8")
    )
    record(
        checks,
        "P009.separate_inherited_and_fresh_status",
        len(statuses) == len(history) == 26
        and {row["claim_id"] for row in statuses} == claim_ids
        and {row["claim_id"] for row in history} == claim_ids
        and all(
            row["inherited_canonical_evidence_grade"]
            == inherited[row["claim_id"]]["evidence_grade"]
            and row["inherited_canonical_confidence"]
            == inherited[row["claim_id"]]["confidence"]
            and row["inherited_canonical_review_status"]
            == inherited[row["claim_id"]]["review_status"]
            and row["fresh_packet_semantic_status"].startswith("author_review_")
            and row["publication_eligibility"]
            == "NOT_ELIGIBLE_AUTHOR_PACKET_ONLY"
            for row in statuses
        )
        and all(
            row["publication_authorized"] == "0"
            and row["grade_change_authorized"] == "0"
            and row["confidence_change_authorized"] == "0"
            for row in history
        )
        and "Inherited canonical status" in html_text
        and "Fresh packet semantic review" in html_text
        and "NOT ELIGIBLE FOR PUBLICATION" in html_text
        and all(html_text.count(row["claim_text"]) == 1 for row in claims),
        "canonical status is labeled inherited; author semantic status and publication hold are separate",
    )

    arithmetic = rows(target, "arithmetic_nonadditivity_controls.csv")
    record(
        checks,
        "measurement.arithmetic_and_nonadditivity",
        len(arithmetic) == 7 and all(row["status"] == "PASS" for row in arithmetic)
        and all(
            row["additive_eligible"] == "0"
            for row in rows(
                target,
                "public_capital_measurement_reconciliations.csv",
            )
        ),
        "seven frozen arithmetic controls pass; all reconciliations remain nonadditive",
    )

    governing = rows(target, "governing_input_inventory.csv")
    current_governing = all(
        (ROOT / row["path"]).is_file()
        and sha(ROOT / row["path"]) == row["sha256"]
        and str((ROOT / row["path"]).stat().st_size) == row["bytes"]
        for row in governing
    )
    record(
        checks,
        "P010.governing_input_provenance",
        len(governing) == 89
        and current_governing
        and {row["sha256"] for row in governing}.issuperset(
            {
                V1_MANIFEST_SHA,
                REVIEW_MANIFEST_SHA,
                V2_AUTHOR_MANIFEST_SHA,
                V2_REVIEW_MANIFEST_SHA,
            }
        ),
        f"governing_inputs={len(governing)}; all hashes and byte counts current",
    )

    mechanically_dependent_or_author_files = {
        "author_report.md",
        "build_counts.json",
        "build_packet.py",
        "governing_input_inventory.csv",
        "locator_semantic_dispositions.csv",
        "seal_manifest.py",
        "validate_packet.py",
        "verify_clean_builds.py",
    }
    runtime_files = {
        "deterministic_build_results.json",
        "generic_validation_results.json",
        "lane_manifest.json",
        "manifest_verification.json",
        "no_write_audit.csv",
        "ruff_results.json",
        "two_clean_build_comparison.csv",
        "validation_results.json",
    }
    v2_manifest = json.loads(
        (V2_AUTHOR / "lane_manifest.json").read_text(encoding="utf-8")
    )
    frozen_entries = [
        entry
        for entry in v2_manifest["files"]
        if entry["path"] not in mechanically_dependent_or_author_files
        and entry["path"] not in runtime_files
    ]
    frozen_mismatches = [
        entry["path"]
        for entry in frozen_entries
        if not (target / entry["path"]).is_file()
        or sha(target / entry["path"]) != entry["sha256"]
    ]
    v2_diff = rows(target, "v2_artifact_diff.csv")
    diff_by_path = {row["path"]: row for row in v2_diff}
    diff_inventory_valid = (
        len(v2_diff) == len(v2_manifest["files"])
        and len(diff_by_path) == len(v2_diff)
        and all(
            row["classification"] != "UNAUTHORIZED_DRIFT"
            and row["v2_sha256"]
            == next(
                entry["sha256"]
                for entry in v2_manifest["files"]
                if entry["path"] == row["path"]
            )
            for row in v2_diff
        )
    )
    record(
        checks,
        "CH10-V2-P001.unrelated_v2_artifacts_byte_frozen",
        not frozen_mismatches and diff_inventory_valid,
        (
            f"frozen_files={len(frozen_entries)} "
            f"mismatches={frozen_mismatches[:10]} "
            f"diff_rows={len(v2_diff)}"
        ),
    )

    patch_checks = {
        "CH10-V2-P001": [
            "CH10-V2-P001.exact_statement_to_locator_replay",
            "CH10-V2-P001.unrelated_v2_artifacts_byte_frozen",
        ],
        "CH10-P001": [
            "P001.normalized_source_versions",
            "P001.append_only_failures_and_cited_urls",
        ],
        "CH10-P002": ["P002.all_relationships_freshly_reviewed"],
        "CH10-P003": [
            "P003.lossless_typed_inventory",
            "P003.field_identity_and_values",
            "P003.field_specific_dispositions",
        ],
        "CH10-P004": [
            "P004.packet_local_external_claim_closure",
            "P004.external_canonical_link_identity",
        ],
        "CH10-P005": ["P005.recursive_entity_fk_closure"],
        "CH10-P006": ["P006.atomic_child_coverage"],
        "CH10-P007": ["P007.claim_and_assertion_specific_falsifiers"],
        "CH10-P008": ["P008.complete_nonadditive_timeline"],
        "CH10-P009": ["P009.separate_inherited_and_fresh_status"],
        "CH10-P010": ["P010.governing_input_provenance"],
    }
    patch_status = {
        patch_id: (
            "PASS"
            if all(checks[check_id]["passed"] for check_id in check_ids)
            else "FAIL"
        )
        for patch_id, check_ids in patch_checks.items()
    }
    failures = [
        check_id
        for check_id, payload in checks.items()
        if not payload["passed"]
    ]
    payload = {
        "validator": "chapter_10_rebuilt_v3_dedicated_acceptance_validator",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "patch_status": patch_status,
        "failures": failures,
        "boundary": (
            "author validation only; fresh uninvolved acceptance and publication "
            "remain unauthorized"
        ),
    }
    if not args.no_write:
        (target / "validation_results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
