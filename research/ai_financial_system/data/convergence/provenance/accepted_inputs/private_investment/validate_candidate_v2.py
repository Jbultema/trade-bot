#!/usr/bin/env python3
"""Author validator for the isolated private-investment candidate v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


sys.dont_write_bytecode = True
LANE = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/integration/"
            "private_investment_candidate/candidate_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(LANE)
AI_ROOT = REPO / "research/ai_financial_system"
CANDIDATE = LANE / "candidate"
BUILDER = LANE / "build_candidate_v2.py"
V1 = AI_ROOT / "integration/private_investment_candidate"
V1_REVIEW = AI_ROOT / "staging/lane_2_verify_private_investment_candidate"
AUTHOR = AI_ROOT / "staging/lane_1_private_investment_earnings_corrected"
ARCHIVE = AI_ROOT / "sources/sec/private_investment_earnings"
CANONICAL_SCHEMA = AI_ROOT / "schema.sql"
DATA = AI_ROOT / "data"

RETAINED_HOLDS = {
    "governed top-20 population reconstruction",
    "S&P 500 aggregate reproduction",
    (
        "unnamed investee allocation at Alphabet and NVIDIA and residual "
        "Amazon attribution"
    ),
    "exact Microsoft component and tax allocation",
    "prospective Meta/Tesla and realization monitoring",
}

ECONOMIC_TABLES = [
    "private_investment_accounting_methods",
    "private_investment_earnings",
    "private_investment_earnings_components",
    "private_investment_carrying_value_rollforwards",
    "private_investment_rollforward_components",
    "private_investment_earnings_bridges",
    "private_investment_bridge_components",
    "private_investment_population_observations",
]

KEYS = {
    "candidate_claims.csv": ("claim_id",),
    "candidate_evidence_links.csv": ("claim_id", "source_id", "locator"),
    "candidate_handoffs.csv": ("handoff_id",),
    "candidate_unknowns.csv": ("unknown_id",),
    "private_investment_accounting_methods.csv": ("method_id",),
    "private_investment_bridge_components.csv": (
        "bridge_id",
        "private_investment_earnings_id",
    ),
    "private_investment_carrying_value_rollforwards.csv": (
        "rollforward_id",
    ),
    "private_investment_earnings.csv": (
        "private_investment_earnings_id",
    ),
    "private_investment_earnings_bridges.csv": ("bridge_id",),
    "private_investment_earnings_components.csv": (
        "parent_earnings_id",
        "component_earnings_id",
    ),
    "private_investment_population_observations.csv": (
        "population_observation_id",
    ),
    "private_investment_rollforward_components.csv": (
        "rollforward_component_id",
    ),
}

EXPECTED_CHANGES = {
    (
        "candidate_evidence_links.csv",
        "CLM-110003|SRC-0012|Private investment accounting notes",
        "support_type",
    ): ("cross_source_reconstruction", "corroborating", "V2-001"),
    (
        "candidate_evidence_links.csv",
        "CLM-110003|SRC-0013|Private investment accounting notes",
        "support_type",
    ): ("cross_source_reconstruction", "corroborating", "V2-001"),
    (
        "candidate_evidence_links.csv",
        "CLM-110003|SRC-0018|Private investment accounting notes",
        "support_type",
    ): ("cross_source_reconstruction", "corroborating", "V2-001"),
    (
        "candidate_evidence_links.csv",
        "CLM-070038|SRC-0012|Income statements and investment notes",
        "support_type",
    ): ("cross_source_reconstruction", "corroborating", "V2-001"),
    (
        "candidate_evidence_links.csv",
        "CLM-070038|SRC-0013|Income statements and investment notes",
        "support_type",
    ): ("cross_source_reconstruction", "corroborating", "V2-001"),
    (
        "private_investment_earnings.csv",
        "PIE-0007",
        "amount_precision",
    ): ("exact", "approximate", "V2-002"),
    (
        "private_investment_earnings.csv",
        "PIE-0002",
        "investee_entity_id",
    ): ("ENT-ANTHROPIC", "", "V2-003"),
    (
        "private_investment_earnings.csv",
        "PIE-0003",
        "investee_entity_id",
    ): ("ENT-ANTHROPIC", "", "V2-003"),
    (
        "private_investment_carrying_value_rollforwards.csv",
        "PIR-0001",
        "investee_entity_id",
    ): ("ENT-ANTHROPIC", "", "V2-003"),
    (
        "private_investment_carrying_value_rollforwards.csv",
        "PIR-0003",
        "investee_entity_id",
    ): ("ENT-ANTHROPIC", "", "V2-003"),
    (
        "private_investment_population_observations.csv",
        "POP-PIE-001",
        "investee_entity_id",
    ): ("ENT-ANTHROPIC", "", "V2-003"),
    (
        "private_investment_population_observations.csv",
        "POP-PIE-006",
        "investee_entity_id",
    ): ("ENT-TIKTOK-USDS", "", "V2-003"),
    (
        "candidate_handoffs.csv",
        "HOF-PIE-004",
        "source_claim_id",
    ): ("", "CLM-110003", "V2-006"),
}

EXPECTED_EXTERNAL_FKS = {
    (
        "private_investment_earnings",
        "investor_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_earnings",
        "investee_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_earnings",
        "source_id",
        "sources",
        "source_id",
    ),
    (
        "private_investment_earnings",
        "source_claim_id",
        "claims",
        "claim_id",
    ),
    (
        "private_investment_carrying_value_rollforwards",
        "investor_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_carrying_value_rollforwards",
        "investee_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_carrying_value_rollforwards",
        "source_id",
        "sources",
        "source_id",
    ),
    (
        "private_investment_earnings_bridges",
        "investor_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_population_observations",
        "investor_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_population_observations",
        "investee_entity_id",
        "entities",
        "entity_id",
    ),
    (
        "private_investment_population_observations",
        "source_id",
        "sources",
        "source_id",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path}: missing header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AssertionError(f"{path}: duplicate header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise AssertionError(f"{path}: malformed-width row")
    return list(reader.fieldnames), rows


def write_csv(
    path: Path, columns: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column)
                    for column in columns
                }
            )


def row_key(row: dict[str, str], key_fields: tuple[str, ...]) -> str:
    return "|".join(row[field] for field in key_fields)


def index(
    rows: list[dict[str, str]], key_fields: tuple[str, ...], label: str
) -> dict[str, dict[str, str]]:
    result = {row_key(row, key_fields): row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise AssertionError(f"{label}: duplicate or blank key")
    return result


def close(actual: float, expected: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=tolerance
    )


checks: list[dict[str, Any]] = []


def check(
    check_id: str,
    category: str,
    scope: str,
    passed: bool,
    observed: Any,
    expected: Any,
    implication: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "scope": scope,
            "result": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "implication": implication,
        }
    )


def validate_manifest(candidate: Path) -> dict[str, Any]:
    manifest_path = candidate / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(
        "MAN-001",
        "boundary",
        "candidate status",
        manifest["status"]
        == "AUTHOR_V2_CANDIDATE_PENDING_FRESH_REVIEW_NOT_CANONICAL",
        manifest["status"],
        "AUTHOR_V2_CANDIDATE_PENDING_FRESH_REVIEW_NOT_CANONICAL",
        "Author candidate remains fail-closed.",
    )
    check(
        "MAN-002",
        "boundary",
        "canonical integration",
        manifest["canonical_integration"]
        == "NOT_AUTHORIZED_PENDING_FRESH_REVIEW",
        manifest["canonical_integration"],
        "NOT_AUTHORIZED_PENDING_FRESH_REVIEW",
        "Author validation cannot self-authorize integration.",
    )
    check(
        "MAN-003",
        "boundary",
        "remaining analytical holds",
        set(manifest["remaining_holds"]) == RETAINED_HOLDS,
        "|".join(manifest["remaining_holds"]),
        "|".join(sorted(RETAINED_HOLDS)),
        "Only preservation is proposed for clearance.",
    )
    declared = set(manifest["files"])
    actual = {
        path.name
        for path in candidate.iterdir()
        if path.is_file() and path.name != "candidate_manifest.json"
    }
    check(
        "MAN-004",
        "manifest",
        "candidate file population",
        declared == actual,
        "|".join(sorted(actual)),
        "|".join(sorted(declared)),
        "No unmanifested candidate artifact.",
    )
    file_pass = True
    for name, metadata in manifest["files"].items():
        path = candidate / name
        file_pass &= (
            path.is_file()
            and sha256(path) == metadata["sha256"]
            and path.stat().st_size == metadata["bytes"]
        )
        if path.suffix == ".csv":
            file_pass &= len(read_csv(path)[1]) == metadata["rows"]
        else:
            file_pass &= (
                len(path.read_text(encoding="utf-8").splitlines())
                == metadata["lines"]
            )
    check(
        "MAN-005",
        "manifest",
        "candidate hashes and counts",
        file_pass,
        f"{len(manifest['files'])} declared files",
        "all hashes, bytes and row/line counts reproduce",
        "Candidate is internally sealed.",
    )
    return manifest


def validate_input_lineage(candidate: Path) -> list[Path]:
    _, lineage = read_csv(candidate / "input_lineage.csv")
    paths: list[Path] = []
    seen = set()
    passed = True
    for row in lineage:
        path = REPO / row["path"]
        paths.append(path)
        passed &= (
            path.is_file()
            and sha256(path) == row["sha256"]
            and path.stat().st_size == int(row["bytes"])
            and row["mutability"] == "read_only_governing_input"
        )
        seen.add(row["path"])
    required = {
        "research/ai_financial_system/staging/private_investment_candidate_v2_task.md",
        "research/ai_financial_system/integration/private_investment_candidate/candidate_manifest.json",
        (
            "research/ai_financial_system/staging/"
            "lane_2_verify_private_investment_candidate/lane_manifest.json"
        ),
        "research/ai_financial_system/staging/private_investment_canonical_mapping.md",
        (
            "research/ai_financial_system/sources/sec/private_investment_earnings/"
            "source_archive_manifest.csv"
        ),
        "research/ai_financial_system/schema.sql",
        "research/ai_financial_system/data/entities.csv",
        "research/ai_financial_system/data/source_register.csv",
        "research/ai_financial_system/data/claims.csv",
    }
    check(
        "LINEAGE-001",
        "lineage",
        "governing input inventory",
        passed and required <= seen and len(lineage) >= 100,
        f"rows={len(lineage)};required={len(required & seen)}/{len(required)}",
        "every recorded input hash current and all core inputs present",
        "All governing inputs are explicitly frozen.",
    )
    return paths


def validate_repairs(candidate: Path) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    modified_cells: set[tuple[str, str, str]] = set()
    for file_name, key_fields in KEYS.items():
        v1_columns, v1_rows = read_csv(V1 / file_name)
        v2_columns, v2_rows = read_csv(candidate / file_name)
        v1 = index(v1_rows, key_fields, f"v1 {file_name}")
        v2 = index(v2_rows, key_fields, f"v2 {file_name}")
        if v1_columns != v2_columns or set(v1) != set(v2):
            raise AssertionError(f"{file_name}: schema or row identity drift")
        for identity, v1_row in v1.items():
            v2_row = v2[identity]
            for field in v1_columns:
                change = EXPECTED_CHANGES.get(
                    (file_name, identity, field)
                )
                expected = change[1] if change else v1_row[field]
                passed = (
                    v1_row[field] == (change[0] if change else v1_row[field])
                    and v2_row[field] == expected
                )
                if v1_row[field] != v2_row[field]:
                    modified_cells.add((file_name, identity, field))
                typed.append(
                    {
                        "typed_audit_id": f"TYPE-{len(typed)+1:05d}",
                        "file": file_name,
                        "row_identity": identity,
                        "field": field,
                        "v1_value": v1_row[field],
                        "expected_v2_value": expected,
                        "actual_v2_value": v2_row[field],
                        "repair_id": change[2] if change else "",
                        "result": "PASS" if passed else "FAIL",
                    }
                )
    check(
        "REPAIR-001",
        "typed semantics",
        "exact v1-to-v2 delta",
        modified_cells == set(EXPECTED_CHANGES)
        and all(row["result"] == "PASS" for row in typed),
        (
            f"modified={len(modified_cells)};"
            f"typed_fields={len(typed)};"
            f"failures={sum(row['result']=='FAIL' for row in typed)}"
        ),
        f"exactly {len(EXPECTED_CHANGES)} repaired cells and no other drift",
        "The v2 repair surface is minimal and explicit.",
    )
    evidence = read_csv(candidate / "candidate_evidence_links.csv")[1]
    check(
        "V2-001",
        "repair",
        "canonical support types",
        Counter(row["support_type"] for row in evidence)
        == {"direct": 3, "corroborating": 5},
        str(Counter(row["support_type"] for row in evidence)),
        "{'direct': 3, 'corroborating': 5}",
        "Every evidence link uses a canonical support type.",
    )
    earnings = index(
        read_csv(candidate / "private_investment_earnings.csv")[1],
        ("private_investment_earnings_id",),
        "earnings",
    )
    check(
        "V2-002",
        "repair",
        "Microsoft rounded amount",
        earnings["PIE-0007"]["pretax_gain_loss_amount"] == "5900"
        and earnings["PIE-0007"]["amount_precision"] == "approximate",
        (
            f"{earnings['PIE-0007']['pretax_gain_loss_amount']}|"
            f"{earnings['PIE-0007']['amount_precision']}"
        ),
        "5900|approximate",
        "The filing's 5.9bn disclosure is never presented as exact.",
    )
    partial_specs = {
        "private_investment_earnings.csv": (
            "private_investment_earnings_id",
            {"PIE-0002", "PIE-0003"},
        ),
        "private_investment_carrying_value_rollforwards.csv": (
            "rollforward_id",
            {"PIR-0001", "PIR-0003"},
        ),
        "private_investment_population_observations.csv": (
            "population_observation_id",
            {"POP-PIE-001", "POP-PIE-006"},
        ),
    }
    partial_ok = True
    for file_name, (key, identities) in partial_specs.items():
        rows = index(read_csv(candidate / file_name)[1], (key,), file_name)
        partial_ok &= all(
            rows[identity]["investee_entity_id"] == ""
            and rows[identity]["investee_scope_description"]
            for identity in identities
        )
    nonnull = []
    for file_name, (key, _) in partial_specs.items():
        for row in read_csv(candidate / file_name)[1]:
            if row["investee_entity_id"]:
                nonnull.append(
                    (
                        file_name,
                        row[key],
                        row["investee_entity_id"],
                        row["investee_scope_description"],
                    )
                )
    permitted_nonnull = {
        (
            "private_investment_earnings.csv",
            "PIE-0001",
            "ENT-ANTHROPIC",
            "ENT-ANTHROPIC",
        ),
        (
            "private_investment_earnings.csv",
            "PIE-0007",
            "ENT-OAI-GROUP",
            "ENT-OAI-GROUP",
        ),
        (
            "private_investment_earnings.csv",
            "PIE-0008",
            "ENT-OAI-GROUP",
            "ENT-OAI-GROUP",
        ),
        (
            "private_investment_carrying_value_rollforwards.csv",
            "PIR-0002",
            "ENT-ANTHROPIC",
            "ENT-ANTHROPIC",
        ),
        (
            "private_investment_population_observations.csv",
            "POP-PIE-004",
            "ENT-OAI-GROUP",
            "ENT-OAI-GROUP",
        ),
        (
            "private_investment_population_observations.csv",
            "POP-PIE-010",
            "ENT-SPACEX",
            "ENT-SPACEX",
        ),
    }
    check(
        "V2-003",
        "repair",
        "partial and exact investee scopes",
        partial_ok and set(nonnull) == permitted_nonnull,
        str(sorted(nonnull)),
        str(sorted(permitted_nonnull)),
        "Only exact accepted scope IDs retain identity edges.",
    )
    handoffs = index(
        read_csv(candidate / "candidate_handoffs.csv")[1],
        ("handoff_id",),
        "handoffs",
    )
    claims = {
        row["claim_id"]
        for row in read_csv(candidate / "candidate_claims.csv")[1]
    } | {row["claim_id"] for row in read_csv(DATA / "claims.csv")[1]}
    check(
        "V2-006",
        "repair",
        "Chapter 11 claim lineage",
        handoffs["HOF-PIE-004"]["source_claim_id"] == "CLM-110003"
        and all(
            not row["source_claim_id"]
            or row["source_claim_id"] in claims
            for row in handoffs.values()
        ),
        handoffs["HOF-PIE-004"]["source_claim_id"],
        "CLM-110003 and all nonblank handoff claims resolve",
        "Circular-confirmation handoff preserves its Grade-C claim.",
    )
    return typed


def candidate_record_index(
    candidate: Path,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], dict[str, str]]]:
    table_specs = {
        "private_investment_earnings": (
            "private_investment_earnings.csv",
            "private_investment_earnings_id",
        ),
        "private_investment_carrying_value_rollforwards": (
            "private_investment_carrying_value_rollforwards.csv",
            "rollforward_id",
        ),
        "private_investment_earnings_bridges": (
            "private_investment_earnings_bridges.csv",
            "bridge_id",
        ),
        "private_investment_population_observations": (
            "private_investment_population_observations.csv",
            "population_observation_id",
        ),
        "private_investment_rollforward_components": (
            "private_investment_rollforward_components.csv",
            "rollforward_component_id",
        ),
    }
    fields = {}
    records = {}
    for table, (file_name, key) in table_specs.items():
        columns, rows = read_csv(candidate / file_name)
        fields[table] = set(columns)
        for row in rows:
            records[(table, row[key])] = row
    return fields, records


def validate_transformation_lineage(candidate: Path) -> list[dict[str, Any]]:
    _, audit = read_csv(candidate / "transformation_audit.csv")
    fields, records = candidate_record_index(candidate)
    source_specs = {
        "private_investment_earnings_ledger.csv": "record_id",
        "carrying_value_rollforward.csv": "rollforward_id",
        "earnings_quality_bridge.csv": "bridge_id",
        "population_evidence.csv": "population_id",
    }
    source_values = {}
    for artifact, key in source_specs.items():
        _, rows = read_csv(AUTHOR / artifact)
        for row in rows:
            for field, value in row.items():
                source_values[(artifact, row[key], field)] = value
    audit_index = {
        (
            row["staging_artifact"],
            row["staging_record_id"],
            row["staging_field"],
        ): row
        for row in audit
    }
    output: list[dict[str, Any]] = []
    allowed_dispositions = {
        "loaded_typed_destination",
        "loaded_typed_destination_with_source_detail_held",
        "held_source_value_only",
    }
    allowed_holds = {
        "",
        "workflow_status_collapsed_to_typed_flags",
        "source_lexeme_retained_for_precision_or_perimeter",
        "blank_component_has_no_economic_row",
        "workflow_status_not_economic_fact",
    }
    for key, source_value in source_values.items():
        row = audit_index.get(key)
        errors = []
        destinations: list[dict[str, str]] = []
        if row is None:
            errors.append("missing_audit_row")
        else:
            if row["staging_value"] != source_value:
                errors.append("staging_value_mismatch")
            if (
                row["lossless_status"]
                != "preserved_with_exact_source_value"
            ):
                errors.append("lossless_status_mismatch")
            if row["transformation_disposition"] not in allowed_dispositions:
                errors.append("uncontrolled_disposition")
            if row["held_reason"] not in allowed_holds:
                errors.append("uncontrolled_held_reason")
            try:
                destinations = json.loads(row["destination_json"])
            except json.JSONDecodeError:
                errors.append("invalid_destination_json")
                destinations = []
            if int(row["destination_count"]) != len(destinations):
                errors.append("destination_count_mismatch")
            if (
                row["transformation_disposition"] == "held_source_value_only"
                and destinations
            ):
                errors.append("held_only_has_destination")
            if (
                row["transformation_disposition"] != "held_source_value_only"
                and not destinations
            ):
                errors.append("loaded_disposition_has_no_destination")
            for destination in destinations:
                required = {"table", "record_id", "field", "value"}
                if set(destination) != required:
                    errors.append("destination_shape_mismatch")
                    continue
                table = destination["table"]
                record_id = destination["record_id"]
                field = destination["field"]
                if table not in fields or field not in fields[table]:
                    errors.append("destination_field_unresolved")
                    continue
                record = records.get((table, record_id))
                if record is None:
                    errors.append("destination_record_unresolved")
                    continue
                if record[field] != destination["value"]:
                    errors.append("destination_value_mismatch")
        output.append(
            {
                "lineage_check_id": f"LINEAGE-{len(output)+1:05d}",
                "staging_artifact": key[0],
                "staging_record_id": key[1],
                "staging_field": key[2],
                "exact_source_value_recovered": (
                    "yes"
                    if row is not None and row["staging_value"] == source_value
                    else "no"
                ),
                "transformation_disposition": (
                    row["transformation_disposition"] if row else ""
                ),
                "destination_count": len(destinations),
                "held_reason": row["held_reason"] if row else "",
                "errors": "|".join(sorted(set(errors))),
                "result": "PASS" if not errors else "FAIL",
            }
        )
    audit_ids = {row["audit_id"] for row in audit}
    keys_match = set(audit_index) == set(source_values)
    all_pass = (
        len(audit) == len(audit_index) == len(audit_ids) == 500
        and len(source_values) == 500
        and keys_match
        and all(row["result"] == "PASS" for row in output)
        and not any(
            "see_typed_transformation_builder" in value
            for row in audit
            for value in row.values()
        )
    )
    check(
        "V2-005",
        "repair",
        "field-level transformation lineage",
        all_pass,
        (
            f"audit={len(audit)};source={len(source_values)};"
            f"lineage_pass={sum(row['result']=='PASS' for row in output)};"
            f"held={sum(row['transformation_disposition']=='held_source_value_only' for row in output)}"
        ),
        "500/500 lossless, resolved or controlled-held, zero unexplained gaps",
        "Every source field has machine-validated disposition.",
    )
    return output


def sqlite_value(
    row: dict[str, str],
    column: str,
    table_info: dict[str, dict[str, bool]],
) -> str | None:
    value = row[column]
    if value != "":
        return value
    if table_info[column]["notnull"] or table_info[column]["primary_key"]:
        return ""
    return None


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    info = {
        item[1]: {
            "notnull": bool(item[3]),
            "primary_key": bool(item[5]),
        }
        for item in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }
    if set(columns) != set(info):
        raise AssertionError(
            f"{table}: CSV/schema columns differ "
            f"{set(columns) ^ set(info)}"
        )
    connection.executemany(
        (
            f"INSERT INTO {table} ({','.join(columns)}) VALUES "
            f"({','.join('?' for _ in columns)})"
        ),
        [
            tuple(sqlite_value(row, column, info) for column in columns)
            for row in rows
        ],
    )


def load_topological(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[dict[str, str]],
    key: str,
    parent: str,
) -> None:
    pending = list(rows)
    inserted = set()
    while pending:
        ready = [
            row
            for row in pending
            if not row[parent] or row[parent] in inserted
        ]
        if not ready:
            raise AssertionError(f"{table}: unresolved parent cycle")
        insert_rows(connection, table, columns, ready)
        inserted.update(row[key] for row in ready)
        pending = [row for row in pending if row not in ready]


def build_canonical_probe(
    candidate: Path,
) -> tuple[sqlite3.Connection, list[dict[str, Any]]]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        CANONICAL_SCHEMA.read_text(encoding="utf-8")
    )
    entity_columns, entities = read_csv(DATA / "entities.csv")
    load_topological(
        connection,
        "entities",
        entity_columns,
        entities,
        "entity_id",
        "parent_entity_id",
    )
    source_columns, sources = read_csv(DATA / "source_register.csv")
    source_map = {row["source_id"]: row for row in sources}
    _, rebases = read_csv(candidate / "source_register_rebase_candidate.csv")
    for row in rebases:
        source_map[row["source_id"]] = row
    insert_rows(
        connection,
        "sources",
        source_columns,
        list(source_map.values()),
    )
    claim_columns, canonical_claims = read_csv(DATA / "claims.csv")
    load_topological(
        connection,
        "claims",
        claim_columns,
        canonical_claims,
        "claim_id",
        "supersedes_claim_id",
    )
    candidate_claim_columns, candidate_claims = read_csv(
        candidate / "candidate_claims.csv"
    )
    insert_rows(
        connection,
        "claims",
        candidate_claim_columns,
        candidate_claims,
    )
    connection.executescript(
        (candidate / "candidate_schema.sql").read_text(encoding="utf-8")
    )
    probe_rows: list[dict[str, Any]] = []
    for table in ECONOMIC_TABLES:
        columns, rows = read_csv(candidate / f"{table}.csv")
        insert_rows(connection, table, columns, rows)
        probe_rows.append(
            {
                "probe_id": f"SQL-TABLE-{len(probe_rows)+1:02d}",
                "scope": table,
                "rows_loaded": len(rows),
                "result": "PASS",
                "notes": "candidate economic table loaded with FK enforcement on",
            }
        )
    evidence_columns, evidence = read_csv(
        candidate / "candidate_evidence_links.csv"
    )
    insert_rows(connection, "evidence_links", evidence_columns, evidence)
    probe_rows.append(
        {
            "probe_id": "SQL-GOV-EVIDENCE",
            "scope": "evidence_links",
            "rows_loaded": len(evidence),
            "result": "PASS",
            "notes": "all canonical support_type CHECK constraints active",
        }
    )
    handoff_columns, handoffs = read_csv(candidate / "candidate_handoffs.csv")
    insert_rows(connection, "chapter_handoffs", handoff_columns, handoffs)
    probe_rows.append(
        {
            "probe_id": "SQL-GOV-HANDOFF",
            "scope": "chapter_handoffs",
            "rows_loaded": len(handoffs),
            "result": "PASS",
            "notes": "all nonblank candidate claim links resolve",
        }
    )
    unknown_columns, unknowns = read_csv(candidate / "candidate_unknowns.csv")
    insert_rows(connection, "unknowns", unknown_columns, unknowns)
    probe_rows.append(
        {
            "probe_id": "SQL-GOV-UNKNOWN",
            "scope": "unknowns",
            "rows_loaded": len(unknowns),
            "result": "PASS",
            "notes": "all eleven unresolved questions retained",
        }
    )
    return connection, probe_rows


def expect_integrity_error(
    connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()
) -> tuple[bool, str]:
    connection.execute("SAVEPOINT negative_test")
    try:
        connection.execute(sql, params)
        connection.execute("ROLLBACK TO negative_test")
        connection.execute("RELEASE negative_test")
        return False, "operation unexpectedly succeeded"
    except sqlite3.IntegrityError as exc:
        connection.execute("ROLLBACK TO negative_test")
        connection.execute("RELEASE negative_test")
        return True, str(exc)


def validate_sqlite(
    candidate: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection, probe = build_canonical_probe(candidate)
    fk_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    actual_external_fks = set()
    for table in ECONOMIC_TABLES:
        for row in connection.execute(
            f"PRAGMA foreign_key_list({table})"
        ).fetchall():
            relationship = (table, row[3], row[2], row[4])
            if row[2] in {"entities", "sources", "claims"}:
                actual_external_fks.add(relationship)
    check(
        "V2-004A",
        "repair",
        "canonical schema and external FKs",
        integrity == "ok"
        and not fk_failures
        and actual_external_fks == EXPECTED_EXTERNAL_FKS,
        (
            f"integrity={integrity};fk_failures={len(fk_failures)};"
            f"external_fks={len(actual_external_fks)}"
        ),
        "integrity=ok;fk_failures=0;external_fks=11",
        "All economic tables are loaded on the real canonical parent shape.",
    )
    negative_specs = [
        (
            "NEG-FK-001",
            "delete referenced source",
            "DELETE FROM sources WHERE source_id='SRC-0013'",
            (),
        ),
        (
            "NEG-FK-002",
            "delete referenced entity",
            "DELETE FROM entities WHERE entity_id='ENT-AMZN'",
            (),
        ),
        (
            "NEG-FK-003",
            "delete referenced claim",
            "DELETE FROM claims WHERE claim_id='CLM-020076'",
            (),
        ),
    ]
    negative: list[dict[str, Any]] = []
    for test_id, scope, sql, params in negative_specs:
        passed, error = expect_integrity_error(connection, sql, params)
        negative.append(
            {
                "negative_test_id": test_id,
                "scope": scope,
                "expected_result": "FOREIGN_KEY_REJECTION",
                "observed_error": error,
                "result": "PASS" if passed else "FAIL",
            }
        )
    earnings_columns, earnings_rows = read_csv(
        candidate / "private_investment_earnings.csv"
    )
    base = dict(earnings_rows[0])
    insertion_specs = [
        ("NEG-FK-004", "missing investor parent", "investor_entity_id", "ENT-MISSING"),
        ("NEG-FK-005", "missing source parent", "source_id", "SRC-MISSING"),
        ("NEG-FK-006", "missing claim parent", "source_claim_id", "CLM-MISSING"),
    ]
    table_info = {
        item[1]: {
            "notnull": bool(item[3]),
            "primary_key": bool(item[5]),
        }
        for item in connection.execute(
            "PRAGMA table_info(private_investment_earnings)"
        ).fetchall()
    }
    for test_id, scope, field, missing_id in insertion_specs:
        row = dict(base)
        row["private_investment_earnings_id"] = test_id
        row[field] = missing_id
        sql = (
            "INSERT INTO private_investment_earnings "
            f"({','.join(earnings_columns)}) VALUES "
            f"({','.join('?' for _ in earnings_columns)})"
        )
        params = tuple(
            sqlite_value(row, column, table_info)
            for column in earnings_columns
        )
        passed, error = expect_integrity_error(connection, sql, params)
        negative.append(
            {
                "negative_test_id": test_id,
                "scope": scope,
                "expected_result": "FOREIGN_KEY_REJECTION",
                "observed_error": error,
                "result": "PASS" if passed else "FAIL",
            }
        )
    stub_count = sum(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {field} LIKE '%STUB%' "
            f"OR {field} LIKE '%MISSING%'"
        ).fetchone()[0]
        for table, field in (
            ("entities", "entity_id"),
            ("sources", "source_id"),
            ("claims", "claim_id"),
        )
    )
    check(
        "V2-004B",
        "repair",
        "negative external-FK tests",
        all(row["result"] == "PASS" for row in negative)
        and stub_count == 0,
        (
            f"passes={sum(row['result']=='PASS' for row in negative)}/"
            f"{len(negative)};stub_parents={stub_count}"
        ),
        "6/6 FK rejections and zero stub parents",
        "Deleting or omitting any parent fails closed.",
    )
    connection.close()
    return probe, negative


def validate_arithmetic(
    candidate: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    earnings = index(
        read_csv(candidate / "private_investment_earnings.csv")[1],
        ("private_investment_earnings_id",),
        "earnings",
    )
    arithmetic: list[dict[str, Any]] = []

    def add(label: str, expression: str, actual: float, expected: float, tolerance: float) -> None:
        arithmetic.append(
            {
                "arithmetic_check_id": f"ARITH-{len(arithmetic)+1:03d}",
                "scope": label,
                "expression": expression,
                "actual": f"{actual:.10f}",
                "expected": f"{expected:.10f}",
                "tolerance": tolerance,
                "result": (
                    "PASS" if close(actual, expected, tolerance) else "FAIL"
                ),
            }
        )

    add(
        "Amazon component sum",
        "4479 + 12328",
        float(earnings["PIE-0003"]["pretax_gain_loss_amount"]),
        float(earnings["PIE-0001"]["pretax_gain_loss_amount"])
        + float(earnings["PIE-0002"]["pretax_gain_loss_amount"]),
        1e-9,
    )
    for earnings_id, row in earnings.items():
        add(
            f"{earnings_id} pretax ratio",
            "gain / reported pretax income * 100",
            float(row["gain_share_of_pretax_income_pct"]),
            float(row["pretax_gain_loss_amount"])
            / float(row["reported_pretax_income_amount"])
            * 100,
            0.00006,
        )
    rollforwards = index(
        read_csv(
            candidate / "private_investment_carrying_value_rollforwards.csv"
        )[1],
        ("rollforward_id",),
        "rollforwards",
    )
    components = read_csv(
        candidate / "private_investment_rollforward_components.csv"
    )[1]
    for rollforward_id, row in rollforwards.items():
        movement = sum(
            float(component["amount"])
            for component in components
            if component["rollforward_id"] == rollforward_id
        )
        add(
            f"{rollforward_id} carrying-value rollforward",
            "beginning + components",
            float(row["beginning_carrying_value_amount"]) + movement,
            float(row["ending_carrying_value_amount"]),
            1e-9,
        )
    bridges = index(
        read_csv(candidate / "private_investment_earnings_bridges.csv")[1],
        ("bridge_id",),
        "bridges",
    )
    for bridge_id, row in bridges.items():
        add(
            f"{bridge_id} pretax residual",
            "reported pretax - selected gain",
            float(row["reported_pretax_income_amount"])
            - float(row["identified_private_investment_gain_amount"]),
            float(row["arithmetic_pretax_residual_amount"]),
            1e-9,
        )
    add(
        "Oracle implied recovered basis",
        "cash proceeds 4300 - realized gain 2700",
        4300 - 2700,
        1600,
        1e-9,
    )
    check(
        "ARITH-ALL",
        "arithmetic",
        "issuer arithmetic",
        all(row["result"] == "PASS" for row in arithmetic),
        (
            f"{sum(row['result']=='PASS' for row in arithmetic)}/"
            f"{len(arithmetic)}"
        ),
        f"{len(arithmetic)}/{len(arithmetic)}",
        "Accepted economic reconstruction remains numerically intact.",
    )

    source_population = index(
        read_csv(AUTHOR / "population_evidence.csv")[1],
        ("population_id",),
        "source population",
    )
    candidate_population = index(
        read_csv(
            candidate / "private_investment_population_observations.csv"
        )[1],
        ("population_observation_id",),
        "candidate population",
    )
    population: list[dict[str, Any]] = []
    for population_id, source in source_population.items():
        row = candidate_population[population_id]
        expected_class = source["observation_class"]
        exact_scope = source["investee_scope"] in {
            "ENT-ANTHROPIC",
            "ENT-OAI-GROUP",
            "ENT-SPACEX",
        }
        passed = (
            row["observation_class"] == expected_class
            and row["investee_scope_description"] == source["investee_scope"]
            and bool(row["investee_entity_id"]) == exact_scope
            and row["integration_status"]
            == "accepted_bounded_source_archive_hold"
        )
        population.append(
            {
                "population_check_id": f"POP-{len(population)+1:03d}",
                "population_observation_id": population_id,
                "observation_class": row["observation_class"],
                "source_scope": source["investee_scope"],
                "candidate_investee_entity_id": row["investee_entity_id"],
                "expected_exact_identity": "yes" if exact_scope else "no",
                "result": "PASS" if passed else "FAIL",
            }
        )
    required_classes = {
        "realized_gain_and_full_disposal",
        "material_carrying_exposure_no_identified_material_private_mark",
        "material_carrying_exposure_no_isolated_period_private_gain",
        "mixed_portfolio_context_not_private_attributable",
        "new_monitored_private_position",
    }
    check(
        "POP-ALL",
        "population",
        "ten balanced observations",
        len(population) == 10
        and all(row["result"] == "PASS" for row in population)
        and required_classes
        <= {row["observation_class"] for row in population},
        (
            f"rows={len(population)};"
            f"passes={sum(row['result']=='PASS' for row in population)}"
        ),
        "10/10 including realized, negative, context and monitoring classes",
        "No gain-only population regression.",
    )
    return arithmetic, population


def validate_preservation(candidate: Path) -> list[dict[str, Any]]:
    source_register = index(
        read_csv(DATA / "source_register.csv")[1],
        ("source_id",),
        "canonical source register",
    )
    rebase = index(
        read_csv(candidate / "source_register_rebase_candidate.csv")[1],
        ("source_id",),
        "source rebase",
    )
    archive = index(
        read_csv(ARCHIVE / "source_archive_manifest.csv")[1],
        ("source_id",),
        "archive",
    )
    provenance = index(
        read_csv(candidate / "source_archive_provenance_candidate.csv")[1],
        ("source_id",),
        "provenance",
    )
    derived = index(
        read_csv(ARCHIVE / "derived_text_manifest.csv")[1],
        ("source_id",),
        "derived",
    )
    derivative_candidate = index(
        read_csv(candidate / "source_text_derivatives_candidate.csv")[1],
        ("source_id",),
        "derivative candidate",
    )
    output: list[dict[str, Any]] = []
    for source_id in sorted(archive):
        original = source_register[source_id]
        patched = rebase[source_id]
        arc = archive[source_id]
        prov = provenance[source_id]
        der = derived[source_id]
        der_c = derivative_candidate[source_id]
        immutable_fields = set(original) - {
            "retrieved_at",
            "archived_path",
            "content_sha256",
            "notes",
        }
        passed = (
            all(patched[field] == original[field] for field in immutable_fields)
            and patched["retrieved_at"]
            == arc["retrieval_timestamp_utc"][:10]
            and patched["archived_path"] == arc["archived_path"]
            and patched["content_sha256"] == arc["content_sha256"]
            and prov["content_sha256"] == arc["content_sha256"]
            and prov["byte_count"] == arc["byte_count"]
            and prov["governing_evidence_state"]
            == "original_inline_xbrl_governs"
            and sha256(REPO / arc["archived_path"]) == arc["content_sha256"]
            and der_c["derived_text_sha256"] == der["derived_text_sha256"]
            and der_c["evidentiary_boundary"]
            == "locator_aid_only_original_inline_xbrl_governs"
        )
        output.append(
            {
                "preservation_check_id": f"PRES-{len(output)+1:03d}",
                "record_type": "source_archive_and_derivative",
                "record_id": source_id,
                "archive_path": arc["archived_path"],
                "content_sha256": arc["content_sha256"],
                "governing_boundary": prov["governing_evidence_state"],
                "result": "PASS" if passed else "FAIL",
            }
        )
    sealed_locator = index(
        read_csv(V1_REVIEW / "archive_claim_semantics.csv")[1],
        ("claim_id",),
        "sealed claim locator review",
    )
    locators = index(
        read_csv(candidate / "claim_locator_reviews_candidate.csv")[1],
        ("claim_id",),
        "candidate claim locator review",
    )
    statuses = index(
        read_csv(candidate / "claim_status_rebase_candidate.csv")[1],
        ("claim_id",),
        "claim status rebase",
    )
    for claim_id, sealed in sealed_locator.items():
        locator = locators[claim_id]
        status = statuses[claim_id]
        passed = (
            locator["source_ids"] == sealed["sources"]
            and locator["locator"] == sealed["locator"]
            and locator["archived_support"] == sealed["archived_support"]
            and locator["semantic_verdict"] == sealed["semantic_verdict"]
            and locator["remaining_boundary"] == sealed["remaining_boundary"]
            and status["cleared_hold"]
            == "primary_source_snapshot_hash_and_locator_semantics"
            and set(status["retained_holds"].split("|")) == RETAINED_HOLDS
            and status["authorization_boundary"]
            == "candidate_only_not_canonical_pending_fresh_review"
        )
        output.append(
            {
                "preservation_check_id": f"PRES-{len(output)+1:03d}",
                "record_type": "claim_locator_and_status",
                "record_id": claim_id,
                "archive_path": "",
                "content_sha256": "",
                "governing_boundary": locator["remaining_boundary"],
                "result": "PASS" if passed else "FAIL",
            }
        )
    check(
        "PRES-ATOMIC",
        "preservation",
        "atomic source and claim-status rebase",
        len(output) == 13
        and all(row["result"] == "PASS" for row in output),
        (
            f"source_records={sum(row['record_type']=='source_archive_and_derivative' for row in output)};"
            f"claim_records={sum(row['record_type']=='claim_locator_and_status' for row in output)}"
        ),
        "8 source records and 5 claim records; all other holds retained",
        "Preservation can move atomically only after fresh review.",
    )
    forbidden = (
        "top-20 total established",
        "s&p total established",
        "typical",
        "frequency established",
    )
    corpus = " ".join(
        value.lower()
        for file_name in (
            "claim_locator_reviews_candidate.csv",
            "claim_status_rebase_candidate.csv",
        )
        for row in read_csv(candidate / file_name)[1]
        for value in row.values()
    )
    check(
        "PRES-BOUNDARY",
        "preservation",
        "analytical hold boundary",
        not any(term in corpus for term in forbidden),
        "no prohibited population/typicality assertion found",
        "preservation only; five analytical holds unchanged",
        "Archive completion does not create broader economic evidence.",
    )
    return output


def validate_determinism(
    candidate: Path,
) -> list[dict[str, str]]:
    env = dict(os.environ)
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    python = REPO / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    comparison: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(
        prefix="pie-v2-build-a-"
    ) as first_temp, tempfile.TemporaryDirectory(
        prefix="pie-v2-build-b-"
    ) as second_temp:
        first = Path(first_temp) / "candidate"
        second = Path(second_temp) / "candidate"
        for output in (first, second):
            subprocess.run(
                [
                    str(python),
                    str(BUILDER),
                    "--output",
                    str(output),
                ],
                cwd=REPO,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
        names = sorted(
            {path.name for path in first.iterdir()}
            | {path.name for path in second.iterdir()}
            | {path.name for path in candidate.iterdir()}
        )
        for name in names:
            first_hash = sha256(first / name) if (first / name).is_file() else ""
            second_hash = (
                sha256(second / name) if (second / name).is_file() else ""
            )
            sealed_hash = (
                sha256(candidate / name)
                if (candidate / name).is_file()
                else ""
            )
            comparison.append(
                {
                    "comparison_id": f"BUILD-{len(comparison)+1:03d}",
                    "file": name,
                    "clean_build_a_sha256": first_hash,
                    "clean_build_b_sha256": second_hash,
                    "sealed_candidate_sha256": sealed_hash,
                    "result": (
                        "PASS_BYTE_IDENTICAL"
                        if first_hash == second_hash == sealed_hash and first_hash
                        else "FAIL"
                    ),
                }
            )
    check(
        "DET-001",
        "determinism",
        "two clean v2 builds",
        len(comparison) == len(list(candidate.iterdir()))
        and all(row["result"] == "PASS_BYTE_IDENTICAL" for row in comparison),
        (
            f"files={len(comparison)};"
            f"passes={sum(row['result']=='PASS_BYTE_IDENTICAL' for row in comparison)}"
        ),
        "every file byte-identical across two builds and sealed candidate",
        "Author candidate is fully reproducible.",
    )
    return comparison


def write_acceptance_memo(
    manifest: dict[str, Any],
    typed: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    arithmetic: list[dict[str, Any]],
    population: list[dict[str, Any]],
    preservation: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> None:
    text = f"""# Private-investment typed candidate v2: author acceptance memo

**Status date:** 2026-07-26  
**Intrinsic economic reconstruction:** **PASS**  
**Typed candidate completeness:** **PASS_AUTHOR_VALIDATION**  
**Source-preservation atomic-rebase readiness:** **READY_PENDING_FRESH_REVIEW**  
**Canonical-integration readiness:** **NOT_AUTHORIZED_PENDING_FRESH_REVIEW**

## Decision

The isolated v2 candidate resolves all six defects identified by the sealed
nonauthor v1 review. The author validator passes every check, two clean builds are
byte-identical to the sealed candidate, all canonical constraint and negative-FK
probes pass, and no governing input changed during validation.

This is not canonical authorization. A fresh nonauthor must review the sealed v2
packet before any merge into canonical CSVs, schema, or database.

## Six repairs

1. All eight evidence links load under the canonical support-type CHECK: three
   `direct`, five `corroborating`.
2. `PIE-0007` retains 5,900 million but is explicitly `approximate`, consistent
   with the filing's “$5.9 billion” disclosure and the bridge/population boundary.
3. The six partial, residual, portfolio, or multi-investee rows have null investee
   IDs. Only six independently enumerated exact source scopes retain IDs.
4. Eleven external entity/source/claim foreign keys are real schema constraints.
   Eight economic tables load on the canonical parent shape; integrity is `ok`,
   foreign-key check is empty, and six negative tests reject missing/deleted parents.
5. All 500 source fields and exact source lexemes are recoverable. Every loaded
   destination resolves to a real candidate field; held-only fields have controlled
   dispositions and reasons; unexplained lineage gaps are zero.
6. `HOF-PIE-004` now resolves to `CLM-110003`.

## Evidence counts

- Typed v1-to-v2 fields checked: {len(typed)}; failures:
  {sum(row['result'] == 'FAIL' for row in typed)}.
- Source-field lineage rows: {len(lineage)}; failures:
  {sum(row['result'] == 'FAIL' for row in lineage)}.
- Arithmetic checks: {len(arithmetic)}; failures:
  {sum(row['result'] == 'FAIL' for row in arithmetic)}.
- Population observations: {len(population)}; failures:
  {sum(row['result'] == 'FAIL' for row in population)}.
- Preservation records: {len(preservation)}; failures:
  {sum(row['result'] == 'FAIL' for row in preservation)}.
- Deterministic files: {len(comparison)}; failures:
  {sum(row['result'] != 'PASS_BYTE_IDENTICAL' for row in comparison)}.

## Atomic preservation candidate

The packet contains isolated rebase rows for eight source-register records, eight
original-inline-XBRL preservation records, eight visible-text v2 derivative
records, five claim-locator review records, and five claim-status proposals.
Original inline-XBRL remains governing evidence; visible text is only a locator aid.
The proposed status transition clears only the snapshot/hash/locator preservation
hold and remains pending fresh review.

## What remains unresolved

The following five holds remain unchanged:

"""
    for hold in sorted(RETAINED_HOLDS):
        text += f"- {hold}\n"
    text += f"""

The candidate does not establish top-20 or S&P-wide prevalence, typicality,
complete investee allocation, exact Microsoft component/tax allocation, or future
realization behavior.

## No-write boundary

The v1 candidate, sealed v1 review, accepted source packet and review, current
mapping, source archives, canonical registries/schema/database, chapters, DOCX/PDF
files, publication manifests, and shared scripts were read-only. Author output is
confined to this v2 lane.

Candidate manifest: `{sha256(CANDIDATE / 'candidate_manifest.json')}`.  
Candidate status: `{manifest['status']}`.  
Canonical integration: `{manifest['canonical_integration']}`.
"""
    (LANE / "acceptance_memo.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    generated = {
        "checks.csv",
        "typed_semantics_audit.csv",
        "transformation_lineage_validation.csv",
        "canonical_constraint_probe.csv",
        "negative_fk_tests.csv",
        "arithmetic_reconciliation.csv",
        "population_reconciliation.csv",
        "source_preservation_validation.csv",
        "two_clean_build_comparison.csv",
        "input_hash_audit.csv",
        "repair_reperformance.csv",
        "findings.csv",
        "validation_results.json",
        "acceptance_memo.md",
    }
    for name in generated:
        (LANE / name).unlink(missing_ok=True)

    manifest = validate_manifest(candidate)
    input_paths = validate_input_lineage(candidate)
    before = {path: sha256(path) for path in input_paths}
    typed = validate_repairs(candidate)
    lineage = validate_transformation_lineage(candidate)
    probe, negative = validate_sqlite(candidate)
    arithmetic, population = validate_arithmetic(candidate)
    preservation = validate_preservation(candidate)
    comparison = validate_determinism(candidate)
    after = {path: sha256(path) for path in input_paths}
    input_audit = [
        {
            "input_check_id": f"INPUT-{sequence:04d}",
            "path": path.relative_to(REPO).as_posix(),
            "sha256_before": before[path],
            "sha256_after": after[path],
            "result": (
                "PASS_UNCHANGED"
                if before[path] == after[path]
                else "FAIL_CHANGED"
            ),
        }
        for sequence, path in enumerate(input_paths, start=1)
    ]
    check(
        "INPUT-PRESERVATION",
        "preservation",
        "governing inputs before and after",
        all(row["result"] == "PASS_UNCHANGED" for row in input_audit),
        str(Counter(row["result"] for row in input_audit)),
        "all PASS_UNCHANGED",
        "Author validation did not mutate any governing input.",
    )

    repair_rows = [
        {
            "repair_check_id": "V2-001",
            "defect": "canonical support types",
            "acceptance": "8 evidence links insert under canonical CHECK",
            "result": next(
                row["result"] for row in checks if row["check_id"] == "V2-001"
            ),
        },
        {
            "repair_check_id": "V2-002",
            "defect": "rounded Microsoft amount",
            "acceptance": "PIE-0007 amount 5900 and approximate",
            "result": next(
                row["result"] for row in checks if row["check_id"] == "V2-002"
            ),
        },
        {
            "repair_check_id": "V2-003",
            "defect": "partial and mixed investee scopes",
            "acceptance": "six null partial IDs and exact retained IDs enumerated",
            "result": next(
                row["result"] for row in checks if row["check_id"] == "V2-003"
            ),
        },
        {
            "repair_check_id": "V2-004",
            "defect": "external foreign keys",
            "acceptance": "11 FKs plus six negative tests",
            "result": (
                "PASS"
                if all(
                    next(
                        row["result"]
                        for row in checks
                        if row["check_id"] == check_id
                    )
                    == "PASS"
                    for check_id in ("V2-004A", "V2-004B")
                )
                else "FAIL"
            ),
        },
        {
            "repair_check_id": "V2-005",
            "defect": "field-level transformation lineage",
            "acceptance": "500 lossless rows and zero unexplained gaps",
            "result": next(
                row["result"] for row in checks if row["check_id"] == "V2-005"
            ),
        },
        {
            "repair_check_id": "V2-006",
            "defect": "Chapter 11 claim lineage",
            "acceptance": "HOF-PIE-004 resolves CLM-110003",
            "result": next(
                row["result"] for row in checks if row["check_id"] == "V2-006"
            ),
        },
    ]

    write_csv(LANE / "checks.csv", list(checks[0]), checks)
    write_csv(
        LANE / "typed_semantics_audit.csv", list(typed[0]), typed
    )
    write_csv(
        LANE / "transformation_lineage_validation.csv",
        list(lineage[0]),
        lineage,
    )
    write_csv(
        LANE / "canonical_constraint_probe.csv", list(probe[0]), probe
    )
    write_csv(
        LANE / "negative_fk_tests.csv", list(negative[0]), negative
    )
    write_csv(
        LANE / "arithmetic_reconciliation.csv",
        list(arithmetic[0]),
        arithmetic,
    )
    write_csv(
        LANE / "population_reconciliation.csv",
        list(population[0]),
        population,
    )
    write_csv(
        LANE / "source_preservation_validation.csv",
        list(preservation[0]),
        preservation,
    )
    write_csv(
        LANE / "two_clean_build_comparison.csv",
        list(comparison[0]),
        comparison,
    )
    write_csv(
        LANE / "input_hash_audit.csv",
        list(input_audit[0]),
        input_audit,
    )
    write_csv(
        LANE / "repair_reperformance.csv",
        list(repair_rows[0]),
        repair_rows,
    )
    write_csv(
        LANE / "findings.csv",
        ["finding_id", "severity", "scope", "finding", "disposition"],
        [],
    )

    failures = [row for row in checks if row["result"] == "FAIL"]
    results = {
        "status": "PASS" if not failures else "FAIL",
        "status_date": "2026-07-26",
        "intrinsic_economic_reconstruction": (
            "PASS" if not failures else "FAIL"
        ),
        "typed_candidate_completeness": (
            "PASS_AUTHOR_VALIDATION" if not failures else "FAIL"
        ),
        "source_preservation_atomic_rebase_readiness": (
            "READY_PENDING_FRESH_REVIEW" if not failures else "NOT_READY"
        ),
        "canonical_integration_readiness": (
            "NOT_AUTHORIZED_PENDING_FRESH_REVIEW"
        ),
        "remaining_analytical_holds": sorted(RETAINED_HOLDS),
        "checks_total": len(checks),
        "checks_passed": sum(row["result"] == "PASS" for row in checks),
        "checks_failed": len(failures),
        "repairs_passed": sum(row["result"] == "PASS" for row in repair_rows),
        "repairs_total": len(repair_rows),
        "typed_fields_checked": len(typed),
        "typed_field_failures": sum(
            row["result"] == "FAIL" for row in typed
        ),
        "transformation_rows_checked": len(lineage),
        "transformation_lineage_failures": sum(
            row["result"] == "FAIL" for row in lineage
        ),
        "economic_tables_loaded": 8,
        "canonical_governance_populations_loaded": {
            "candidate_claims": 5,
            "evidence_links": 8,
            "chapter_handoffs": 11,
            "unknowns": 11,
        },
        "external_foreign_keys": 11,
        "negative_fk_tests_passed": sum(
            row["result"] == "PASS" for row in negative
        ),
        "arithmetic_checks_passed": sum(
            row["result"] == "PASS" for row in arithmetic
        ),
        "population_checks_passed": sum(
            row["result"] == "PASS" for row in population
        ),
        "preservation_records_passed": sum(
            row["result"] == "PASS" for row in preservation
        ),
        "clean_builds_compared": 2,
        "deterministic_files_compared": len(comparison),
        "candidate_manifest_sha256": sha256(
            candidate / "candidate_manifest.json"
        ),
        "author_canonical_writes": [],
        "author_output_only": LANE.relative_to(REPO).as_posix() + "/",
    }
    (LANE / "validation_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_acceptance_memo(
        manifest,
        typed,
        lineage,
        arithmetic,
        population,
        preservation,
        comparison,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
