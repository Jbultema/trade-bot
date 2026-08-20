#!/usr/bin/env python3
"""Build the isolated private-investment typed candidate v2.

This author builder reads the sealed accepted packets, immutable v1 candidate,
sealed v1 review, current governance mapping, post-candidate source archive, and
read-only canonical registries. It writes only to the requested output directory.
It never authorizes canonical integration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
LANE = SCRIPT.parent


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
AUTHOR = AI_ROOT / "staging/lane_1_private_investment_earnings_corrected"
AUTHOR_REVIEW = (
    AI_ROOT / "staging/lane_3_verify_private_investment_earnings_corrected"
)
V1 = AI_ROOT / "integration/private_investment_candidate"
V1_REVIEW = AI_ROOT / "staging/lane_2_verify_private_investment_candidate"
MAPPING = AI_ROOT / "staging/private_investment_canonical_mapping.md"
ARCHIVE = AI_ROOT / "sources/sec/private_investment_earnings"
TASK = AI_ROOT / "staging/private_investment_candidate_v2_task.md"
SCHEMA = AI_ROOT / "schema.sql"
DATA = AI_ROOT / "data"
ARCHIVE_BUILDER = REPO / "scripts/build_private_investment_source_archive.py"
TEXT_EXTRACTOR = REPO / "scripts/extract_private_investment_sec_text.py"
DEFAULT_OUTPUT = LANE / "candidate"

EXPECTED = {
    TASK: "dea89ee4c47469902b022c89bae095cdbca1f803ed28d9644dc0d0798fba51fc",
    V1 / "candidate_manifest.json": (
        "389319e0f2e472ece58a92185a1f9f300f517c488ec1746aea70b67dd9d080f7"
    ),
    V1_REVIEW / "lane_manifest.json": (
        "28660d3f260fe27adb128dd918ec3fc4dd5515b922415446da391368c701a14a"
    ),
    AUTHOR / "lane_manifest.json": (
        "609108aca41f6d717cddce064d8130bebce592e7cfd559ceb157d415e76d258b"
    ),
    AUTHOR_REVIEW / "lane_manifest.json": (
        "8e48b9470b2ed1ea7e9315027b5d17fa3cf38e83724a16ca40970d801a98ed48"
    ),
    MAPPING: (
        "1c83fa228c5f1957b0e0c017386a3337f10c58402737aa509903215716415d44"
    ),
    ARCHIVE / "source_archive_manifest.csv": (
        "4fc2b46b3525617fcc68d93fcf1dc05c770a9fd9230829a79cda32ef5fec6acd"
    ),
    ARCHIVE / "archive_validation.json": (
        "62cc2fead33995332c7af44ed8d1de6cc9180f1983db415d588e38729d71a6e1"
    ),
    ARCHIVE / "derived_text_manifest.csv": (
        "285918598c6b5e4aa47c5aea42f7d8afb2a5d9d9eafea21446577ee6cb8f36d0"
    ),
    ARCHIVE / "derived_text_validation.json": (
        "0ab2f41d58e65506e1f2e481271103c614aa724fb39a081cb339f87aa7063fbd"
    ),
    ARCHIVE_BUILDER: (
        "b059390e7b93dabbddd73bc9ec087128e102603c73403bed5bd61bc123afbcd2"
    ),
    TEXT_EXTRACTOR: (
        "4170234e66f9ccfdd81b733b0f4c1d91fbd2d0deb5f422fa7e5a5b2cf3e9ee70"
    ),
    SCHEMA: "349b35ab87108993858a1a6ed0f3cd8f36497d1143270a44c338994af9072b3f",
    DATA / "entities.csv": (
        "261bbc3a5676845309a7ae343ba931d747a2d2e10d76b2afba2d11d396eabeeb"
    ),
    DATA / "source_register.csv": (
        "c1aa1ac7aa5574894a759cf08a77a3be8d50e68dcbf01846f5497774c4b4206c"
    ),
    DATA / "claims.csv": (
        "8c7fab446fd44159c2847a5c0bacc87a6d1f9d69cfe4e321aa1352ab231d27ea"
    ),
}

RETAINED_HOLDS = [
    "governed top-20 population reconstruction",
    "S&P 500 aggregate reproduction",
    (
        "unnamed investee allocation at Alphabet and NVIDIA and residual "
        "Amazon attribution"
    ),
    "exact Microsoft component and tax allocation",
    "prospective Meta/Tesla and realization monitoring",
]

BASE_CSV_FILES = [
    "candidate_claims.csv",
    "candidate_evidence_links.csv",
    "candidate_handoffs.csv",
    "candidate_unknowns.csv",
    "private_investment_accounting_methods.csv",
    "private_investment_bridge_components.csv",
    "private_investment_carrying_value_rollforwards.csv",
    "private_investment_earnings.csv",
    "private_investment_earnings_bridges.csv",
    "private_investment_earnings_components.csv",
    "private_investment_population_observations.csv",
    "private_investment_rollforward_components.csv",
]

SOURCE_ARTIFACTS = {
    "private_investment_earnings_ledger.csv": (
        "record_id",
        "private_investment_earnings",
        "private_investment_earnings_id",
    ),
    "carrying_value_rollforward.csv": (
        "rollforward_id",
        "private_investment_carrying_value_rollforwards",
        "rollforward_id",
    ),
    "earnings_quality_bridge.csv": (
        "bridge_id",
        "private_investment_earnings_bridges",
        "bridge_id",
    ),
    "population_evidence.csv": (
        "population_id",
        "private_investment_population_observations",
        "population_observation_id",
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
            raise ValueError(f"{path}: missing header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: malformed-width row")
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


def index(
    rows: list[dict[str, str]], key: str, label: str
) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise ValueError(f"{label}: duplicate or blank {key}")
    return result


def require_inputs() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise ValueError(f"governing input missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"governing input drift: {path.relative_to(REPO)} "
                f"{actual} != {expected}"
            )
    review = json.loads(
        (V1_REVIEW / "validation_results.json").read_text(encoding="utf-8")
    )
    if (
        review["bounded_candidate_verdict"]
        != "REJECT_PENDING_6_EXACT_REPAIRS"
        or review["exact_findings"] != 6
        or review["exact_patches"] != 6
    ):
        raise ValueError("sealed v1 review no longer expresses the six-repair task")
    archive_validation = json.loads(
        (ARCHIVE / "archive_validation.json").read_text(encoding="utf-8")
    )
    derived_validation = json.loads(
        (ARCHIVE / "derived_text_validation.json").read_text(encoding="utf-8")
    )
    if (
        archive_validation.get("status") != "PASS"
        or archive_validation.get("passed_source_count") != 8
        or derived_validation.get("status") != "PASS"
    ):
        raise ValueError("post-candidate archive is not in the accepted 8/8 state")


def governing_input_paths() -> list[tuple[Path, str]]:
    paths: dict[Path, str] = {}
    directory_roles = {
        AUTHOR: "accepted_corrected_source_packet",
        AUTHOR_REVIEW: "accepted_source_packet_review",
        V1: "immutable_v1_candidate",
        V1_REVIEW: "sealed_v1_candidate_review",
        ARCHIVE: "post_candidate_primary_source_archive",
    }
    for directory, role in directory_roles.items():
        for path in directory.rglob("*"):
            if path.is_file():
                paths[path] = role
    for path, role in {
        TASK: "author_task",
        MAPPING: "post_candidate_governance_mapping",
        ARCHIVE_BUILDER: "archive_builder",
        TEXT_EXTRACTOR: "visible_text_v2_extractor",
        SCHEMA: "canonical_schema_read_only",
        DATA / "entities.csv": "canonical_registry_read_only",
        DATA / "source_register.csv": "canonical_registry_read_only",
        DATA / "claims.csv": "canonical_registry_read_only",
        DATA / "evidence_links.csv": "canonical_registry_read_only",
        DATA / "chapter_handoffs.csv": "canonical_registry_read_only",
        DATA / "unknowns.csv": "canonical_registry_read_only",
    }.items():
        paths[path] = role
    return sorted(paths.items(), key=lambda item: item[0].as_posix())


def copy_and_repair_base(output: Path) -> None:
    for file_name in BASE_CSV_FILES:
        columns, rows = read_csv(V1 / file_name)
        if file_name == "candidate_evidence_links.csv":
            for row in rows:
                if row["support_type"] == "cross_source_reconstruction":
                    row["support_type"] = "corroborating"
        elif file_name == "private_investment_earnings.csv":
            for row in rows:
                if row["private_investment_earnings_id"] in {
                    "PIE-0002",
                    "PIE-0003",
                }:
                    row["investee_entity_id"] = ""
                if row["private_investment_earnings_id"] == "PIE-0007":
                    row["amount_precision"] = "approximate"
        elif file_name == "private_investment_carrying_value_rollforwards.csv":
            for row in rows:
                if row["rollforward_id"] in {"PIR-0001", "PIR-0003"}:
                    row["investee_entity_id"] = ""
        elif file_name == "private_investment_population_observations.csv":
            for row in rows:
                if row["population_observation_id"] in {
                    "POP-PIE-001",
                    "POP-PIE-006",
                }:
                    row["investee_entity_id"] = ""
        elif file_name == "candidate_handoffs.csv":
            for row in rows:
                if row["handoff_id"] == "HOF-PIE-004":
                    row["source_claim_id"] = "CLM-110003"
        write_csv(output / file_name, columns, rows)


def build_schema(output: Path) -> None:
    schema = (V1 / "candidate_schema.sql").read_text(encoding="utf-8")
    replacements = [
        (
            "    investor_entity_id TEXT NOT NULL,\n",
            (
                "    investor_entity_id TEXT NOT NULL "
                "REFERENCES entities(entity_id),\n"
            ),
            4,
        ),
        (
            "    investee_entity_id TEXT,\n",
            "    investee_entity_id TEXT REFERENCES entities(entity_id),\n",
            3,
        ),
        (
            "    source_id TEXT NOT NULL,\n",
            "    source_id TEXT NOT NULL REFERENCES sources(source_id),\n",
            3,
        ),
        (
            "    source_claim_id TEXT,\n",
            "    source_claim_id TEXT REFERENCES claims(claim_id),\n",
            1,
        ),
    ]
    for old, new, expected_count in replacements:
        if schema.count(old) != expected_count:
            raise ValueError(
                f"v1 schema replacement drift: {old.strip()} "
                f"count={schema.count(old)} expected={expected_count}"
            )
        schema = schema.replace(old, new)
    (output / "candidate_schema.sql").write_text(schema, encoding="utf-8")


def candidate_tables(output: Path) -> dict[str, dict[str, dict[str, str]]]:
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
    result = {}
    for table, (file_name, key) in table_specs.items():
        _, rows = read_csv(output / file_name)
        result[table] = index(rows, key, table)
    return result


def destination(
    tables: dict[str, dict[str, dict[str, str]]],
    table: str,
    record_id: str,
    field: str,
) -> dict[str, str]:
    try:
        value = tables[table][record_id][field]
    except KeyError as exc:
        raise ValueError(
            f"unresolvable transformation target "
            f"{table}:{record_id}:{field}"
        ) from exc
    return {
        "table": table,
        "record_id": record_id,
        "field": field,
        "value": value,
    }


def destination_set(
    tables: dict[str, dict[str, dict[str, str]]],
    table: str,
    record_id: str,
    fields: Iterable[str],
) -> list[dict[str, str]]:
    return [destination(tables, table, record_id, field) for field in fields]


def component_id_for_field(
    source: dict[str, str], source_field: str
) -> str | None:
    fields = [
        "additions_usd_m",
        "unrealized_gain_usd_m",
        "reclassification_usd_m",
        "impairment_or_loss_usd_m",
        "other_or_unreconciled_usd_m",
    ]
    sequence = 0
    for field in fields:
        if not source[field]:
            continue
        sequence += 1
        if field == source_field:
            return f"{source['rollforward_id']}-C{sequence:02d}"
    return None


def earnings_destinations(
    tables: dict[str, dict[str, dict[str, str]]],
    source: dict[str, str],
    field: str,
) -> tuple[list[dict[str, str]], str, str]:
    table = "private_investment_earnings"
    record_id = source["record_id"]
    mapping = {
        "record_id": ["private_investment_earnings_id"],
        "investor_entity_id": ["investor_entity_id"],
        "period": [
            "reporting_period_label",
            "reporting_period_start",
            "reporting_period_end",
            "reporting_period_date_status",
        ],
        "investee_scope": [
            "investee_entity_id",
            "investee_scope_description",
        ],
        "instrument_scope": ["instrument_scope"],
        "accounting_method": [
            "accounting_method_id",
            "accounting_method_scope",
        ],
        "recognized_location": [
            "recognition_location",
            "reported_line_item",
        ],
        "pretax_gain_loss_usd_m": [
            "pretax_gain_loss_amount",
            "currency",
            "unit_scale",
            "amount_precision",
            "amount_unparsed_suffix",
        ],
        "realized_status": ["recognition_state"],
        "cash_received_at_recognition": [
            "cash_received_amount",
            "cash_received_state",
            "cash_amount_precision",
        ],
        "reported_pretax_income_usd_m": [
            "reported_pretax_income_amount",
            "reported_pretax_income_precision",
        ],
        "gain_share_of_pretax_income_pct": [
            "gain_share_of_pretax_income_pct",
            "ratio_precision",
        ],
        "investee_attribution": ["investee_attribution_state"],
        "evidence_grade": ["evidence_grade"],
        "source_id": ["source_id"],
        "source_locator": ["source_locator"],
        "canonical_claim_id": ["source_claim_id"],
        "status": ["analyst_derived", "integration_status"],
        "notes": ["notes"],
    }
    destinations = destination_set(tables, table, record_id, mapping[field])
    if field == "status":
        return (
            destinations,
            "loaded_typed_destination_with_source_detail_held",
            "workflow_status_collapsed_to_typed_flags",
        )
    if field in {
        "pretax_gain_loss_usd_m",
        "cash_received_at_recognition",
        "reported_pretax_income_usd_m",
        "gain_share_of_pretax_income_pct",
    }:
        return (
            destinations,
            "loaded_typed_destination_with_source_detail_held",
            "source_lexeme_retained_for_precision_or_perimeter",
        )
    return destinations, "loaded_typed_destination", ""


def rollforward_destinations(
    tables: dict[str, dict[str, dict[str, str]]],
    source: dict[str, str],
    field: str,
) -> tuple[list[dict[str, str]], str, str]:
    table = "private_investment_carrying_value_rollforwards"
    record_id = source["rollforward_id"]
    direct = {
        "rollforward_id": ["rollforward_id"],
        "investor_entity_id": ["investor_entity_id"],
        "period": [
            "reporting_period_label",
            "reporting_period_start",
            "reporting_period_end",
            "reporting_period_date_status",
        ],
        "investee_scope": [
            "investee_entity_id",
            "investee_scope_description",
        ],
        "instrument_scope": ["instrument_scope"],
        "beginning_carrying_value_usd_m": [
            "beginning_carrying_value_amount",
            "currency",
            "unit_scale",
            "beginning_amount_precision",
        ],
        "ending_carrying_value_usd_m": [
            "ending_carrying_value_amount",
            "ending_amount_precision",
        ],
        "reconciliation_status": ["reconciliation_state"],
        "source_id": ["source_id"],
        "source_locator": ["source_locator"],
        "notes": ["notes"],
    }
    if field in direct:
        destinations = destination_set(tables, table, record_id, direct[field])
        if field in {
            "beginning_carrying_value_usd_m",
            "ending_carrying_value_usd_m",
        }:
            return (
                destinations,
                "loaded_typed_destination_with_source_detail_held",
                "source_lexeme_retained_for_precision_or_perimeter",
            )
        if field == "notes":
            for component in tables[
                "private_investment_rollforward_components"
            ].values():
                if component["rollforward_id"] == record_id:
                    destinations.append(
                        destination(
                            tables,
                            "private_investment_rollforward_components",
                            component["rollforward_component_id"],
                            "notes",
                        )
                    )
        return destinations, "loaded_typed_destination", ""
    component_id = component_id_for_field(source, field)
    if component_id is None:
        return (
            [],
            "held_source_value_only",
            "blank_component_has_no_economic_row",
        )
    fields = [
        "rollforward_component_id",
        "rollforward_id",
        "component_type",
        "amount",
        "currency",
        "unit_scale",
        "amount_precision",
        "amount_unparsed_suffix",
        "cash_state",
        "income_aoci_balance_sheet_class",
        "additive_eligible",
        "overlap_group",
    ]
    return (
        destination_set(
            tables,
            "private_investment_rollforward_components",
            component_id,
            fields,
        ),
        "loaded_typed_destination_with_source_detail_held",
        "source_lexeme_retained_for_precision_or_perimeter",
    )


def bridge_destinations(
    tables: dict[str, dict[str, dict[str, str]]],
    source: dict[str, str],
    field: str,
) -> tuple[list[dict[str, str]], str, str]:
    table = "private_investment_earnings_bridges"
    record_id = source["bridge_id"]
    mapping = {
        "bridge_id": ["bridge_id"],
        "investor_entity_id": ["investor_entity_id"],
        "period": [
            "reporting_period_label",
            "reporting_period_start",
            "reporting_period_end",
            "reporting_period_date_status",
        ],
        "reported_operating_income_usd_m": [
            "reported_operating_income_amount",
            "currency",
            "unit_scale",
        ],
        "reported_pretax_income_usd_m": [
            "reported_pretax_income_amount",
        ],
        "identified_private_investment_gain_usd_m": [
            "identified_private_investment_gain_amount",
            "selected_gain_precision",
        ],
        "arithmetic_pretax_residual_after_selected_gain_usd_m": [
            "arithmetic_pretax_residual_amount",
            "residual_precision",
        ],
        "private_gain_share_pretax_pct": [
            "private_gain_share_pretax_pct",
            "ratio_precision",
        ],
        "cash_flow_reversal_usd_m": [
            "cash_flow_reversal_amount",
            "cash_flow_reversal_precision",
            "cash_flow_reversal_perimeter",
        ],
        "tax_effect_status": ["tax_effect_state"],
        "model_status": ["analyst_derived_state"],
        "notes": ["notes"],
    }
    destinations = destination_set(tables, table, record_id, mapping[field])
    if field.endswith("_usd_m") or field == "private_gain_share_pretax_pct":
        return (
            destinations,
            "loaded_typed_destination_with_source_detail_held",
            "source_lexeme_retained_for_precision_or_perimeter",
        )
    return destinations, "loaded_typed_destination", ""


def population_destinations(
    tables: dict[str, dict[str, dict[str, str]]],
    source: dict[str, str],
    field: str,
) -> tuple[list[dict[str, str]], str, str]:
    table = "private_investment_population_observations"
    record_id = source["population_id"]
    mapping = {
        "population_id": ["population_observation_id"],
        "investor_entity_id": ["investor_entity_id"],
        "issuer": ["issuer_label"],
        "period": [
            "reporting_period_label",
            "reporting_period_start",
            "reporting_period_end",
            "reporting_period_date_status",
        ],
        "observation_class": ["observation_class"],
        "investee_scope": [
            "investee_entity_id",
            "investee_scope_description",
        ],
        "accounting_method": [
            "accounting_method_id",
            "accounting_method_scope",
        ],
        "carrying_value_usd_m": [
            "carrying_value_amount",
            "currency",
            "unit_scale",
        ],
        "current_period_private_gain_loss_usd_m": [
            "current_period_private_gain_loss_amount",
        ],
        "cumulative_upward_adjustments_usd_m": [
            "cumulative_upward_adjustments_amount",
        ],
        "named_position_cost_usd_m": ["named_position_cost_amount"],
        "cash_proceeds_usd_m": ["cash_proceeds_amount"],
        "reported_pretax_income_usd_m": [
            "reported_pretax_income_amount"
        ],
        "gain_share_of_pretax_pct": ["gain_share_of_pretax_pct"],
        "material_current_private_gain_identified": [
            "material_private_gain_state"
        ],
        "attribution_state": ["investee_attribution_state"],
        "amount_precision": ["amount_precision_state"],
        "operating_nonoperating_state": ["operating_nonoperating_state"],
        "source_id": ["source_id"],
        "source_url": ["source_url"],
        "notes": ["notes"],
    }
    if field == "status":
        return (
            [],
            "held_source_value_only",
            "workflow_status_not_economic_fact",
        )
    destinations = destination_set(tables, table, record_id, mapping[field])
    if field.endswith("_usd_m") or field == "gain_share_of_pretax_pct":
        return (
            destinations,
            "loaded_typed_destination_with_source_detail_held",
            "source_lexeme_retained_for_precision_or_perimeter",
        )
    return destinations, "loaded_typed_destination", ""


def build_transformation_audit(output: Path) -> None:
    tables = candidate_tables(output)
    builders = {
        "private_investment_earnings_ledger.csv": earnings_destinations,
        "carrying_value_rollforward.csv": rollforward_destinations,
        "earnings_quality_bridge.csv": bridge_destinations,
        "population_evidence.csv": population_destinations,
    }
    rows: list[dict[str, Any]] = []
    sequence = 0
    for artifact, (id_field, canonical_table, _) in SOURCE_ARTIFACTS.items():
        _, source_rows = read_csv(AUTHOR / artifact)
        for source in source_rows:
            for field, value in source.items():
                sequence += 1
                destinations, disposition, held_reason = builders[artifact](
                    tables, source, field
                )
                rows.append(
                    {
                        "audit_id": f"PIE-V2-XFM-{sequence:05d}",
                        "packet_id": "private_investment_earnings_corrected",
                        "staging_artifact": artifact,
                        "staging_record_id": source[id_field],
                        "staging_field": field,
                        "staging_value": value,
                        "declared_primary_table": canonical_table,
                        "transformation_disposition": disposition,
                        "destination_count": len(destinations),
                        "destination_json": json.dumps(
                            destinations,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "transformation_rule": (
                            "explicit_machine_validated_destination_set"
                            if destinations
                            else "controlled_held_source_value"
                        ),
                        "lossless_status": "preserved_with_exact_source_value",
                        "review_status": "author_v2_pending_fresh_review",
                        "held_reason": held_reason,
                        "notes": (
                            "Exact staging lexeme retained; destination_json "
                            "contains only real candidate table fields."
                        ),
                    }
                )
    if len(rows) != 500:
        raise ValueError(f"unexpected transformation population: {len(rows)}")
    write_csv(output / "transformation_audit.csv", list(rows[0]), rows)


def build_atomic_preservation_candidate(output: Path) -> None:
    source_columns, source_rows = read_csv(DATA / "source_register.csv")
    source_index = index(source_rows, "source_id", "source register")
    _, archive_rows = read_csv(ARCHIVE / "source_archive_manifest.csv")
    archive_index = index(archive_rows, "source_id", "archive manifest")
    _, derived_rows = read_csv(ARCHIVE / "derived_text_manifest.csv")
    derived_index = index(derived_rows, "source_id", "derived manifest")
    source_ids = sorted(archive_index)
    if source_ids != sorted(derived_index) or len(source_ids) != 8:
        raise ValueError("archive/derivative source population mismatch")

    rebase_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    derivative_rows: list[dict[str, str]] = []
    for source_id in source_ids:
        source = dict(source_index[source_id])
        archive = archive_index[source_id]
        derived = derived_index[source_id]
        source["retrieved_at"] = archive["retrieval_timestamp_utc"][:10]
        source["archived_path"] = archive["archived_path"]
        source["content_sha256"] = archive["content_sha256"]
        preservation_note = (
            "Atomic v2 preservation candidate: original inline-XBRL body "
            "governs; response headers were not separately frozen; locator "
            "semantics remain bounded by claim-level review."
        )
        source["notes"] = (
            f"{source['notes']} {preservation_note}".strip()
        )
        rebase_rows.append(source)
        provenance_rows.append(
            {
                "archive_record_id": f"ARCHIVE-{source_id}",
                "source_id": source_id,
                "source_url": archive["source_url"],
                "resolved_url": archive["resolved_url"],
                "retrieval_timestamp_utc": archive[
                    "retrieval_timestamp_utc"
                ],
                "archived_path": archive["archived_path"],
                "content_sha256": archive["content_sha256"],
                "byte_count": archive["byte_count"],
                "mime_type": archive["mime_type"],
                "preservation_class": archive["preservation_class"],
                "preservation_method": archive["preservation_method"],
                "http_status": archive["http_status"],
                "response_metadata_state": archive[
                    "response_metadata_state"
                ],
                "governing_evidence_state": (
                    "original_inline_xbrl_governs"
                ),
                "locator_review_state": (
                    "carried_forward_from_sealed_nonauthor_claim_review"
                ),
                "atomic_rebase_status": (
                    "author_v2_candidate_pending_fresh_review"
                ),
            }
        )
        derivative_rows.append(
            {
                "derivative_record_id": f"DERIVED-{source_id}",
                "source_id": source_id,
                "original_archived_path": derived[
                    "original_archived_path"
                ],
                "original_content_sha256": derived[
                    "original_content_sha256"
                ],
                "derived_text_path": derived["derived_text_path"],
                "derived_text_sha256": derived["derived_text_sha256"],
                "derived_byte_count": derived["derived_byte_count"],
                "extraction_method": derived["extraction_method"],
                "normalization_rule": derived["normalization_rule"],
                "preservation_status": derived["preservation_status"],
                "evidentiary_boundary": (
                    "locator_aid_only_original_inline_xbrl_governs"
                ),
                "atomic_rebase_status": (
                    "author_v2_candidate_pending_fresh_review"
                ),
            }
        )
    write_csv(
        output / "source_register_rebase_candidate.csv",
        source_columns,
        rebase_rows,
    )
    write_csv(
        output / "source_archive_provenance_candidate.csv",
        list(provenance_rows[0]),
        provenance_rows,
    )
    write_csv(
        output / "source_text_derivatives_candidate.csv",
        list(derivative_rows[0]),
        derivative_rows,
    )

    _, sealed_claim_review = read_csv(
        V1_REVIEW / "archive_claim_semantics.csv"
    )
    _, candidate_claims = read_csv(output / "candidate_claims.csv")
    claim_index = index(candidate_claims, "claim_id", "candidate claims")
    locator_rows: list[dict[str, str]] = []
    status_rows: list[dict[str, str]] = []
    for row in sealed_claim_review:
        claim_id = row["claim_id"]
        locator_rows.append(
            {
                "claim_locator_review_id": f"LOCATOR-{claim_id}",
                "claim_id": claim_id,
                "evidence_grade": claim_index[claim_id]["evidence_grade"],
                "source_ids": row["sources"],
                "locator": row["locator"],
                "archived_support": row["archived_support"],
                "semantic_verdict": row["semantic_verdict"],
                "remaining_boundary": row["remaining_boundary"],
                "review_provenance": (
                    "carried_forward_exactly_from_sealed_v1_nonauthor_review"
                ),
                "candidate_status": (
                    "author_v2_candidate_pending_fresh_review"
                ),
            }
        )
        status_rows.append(
            {
                "claim_status_rebase_id": f"CLAIM-STATUS-{claim_id}",
                "claim_id": claim_id,
                "prior_review_status": (
                    "accepted_bounded_source_archive_hold"
                ),
                "proposed_review_status": (
                    "accepted_bounded_preservation_cleared_other_holds_retained"
                ),
                "cleared_hold": (
                    "primary_source_snapshot_hash_and_locator_semantics"
                ),
                "retained_holds": "|".join(RETAINED_HOLDS),
                "authorization_boundary": (
                    "candidate_only_not_canonical_pending_fresh_review"
                ),
                "notes": (
                    "No population, frequency, investee allocation, exact tax, "
                    "or prospective realization conclusion is cleared."
                ),
            }
        )
    write_csv(
        output / "claim_locator_reviews_candidate.csv",
        list(locator_rows[0]),
        locator_rows,
    )
    write_csv(
        output / "claim_status_rebase_candidate.csv",
        list(status_rows[0]),
        status_rows,
    )


def build_input_lineage(output: Path) -> None:
    rows = []
    for sequence, (path, role) in enumerate(
        governing_input_paths(), start=1
    ):
        rows.append(
            {
                "input_record_id": f"V2-INPUT-{sequence:04d}",
                "path": path.relative_to(REPO).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "role": role,
                "mutability": "read_only_governing_input",
            }
        )
    write_csv(output / "input_lineage.csv", list(rows[0]), rows)


def build_manifest(output: Path) -> None:
    files = {}
    for path in sorted(
        item
        for item in output.iterdir()
        if item.is_file() and item.name != "candidate_manifest.json"
    ):
        metadata: dict[str, Any] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            _, rows = read_csv(path)
            metadata["rows"] = len(rows)
        else:
            metadata["lines"] = len(
                path.read_text(encoding="utf-8").splitlines()
            )
        files[path.name] = metadata
    manifest = {
        "status": "AUTHOR_V2_CANDIDATE_PENDING_FRESH_REVIEW_NOT_CANONICAL",
        "status_date": "2026-07-26",
        "intrinsic_economic_reconstruction": "AUTHOR_VALIDATED",
        "typed_candidate_completeness": "AUTHOR_VALIDATED_PENDING_FRESH_REVIEW",
        "source_preservation_atomic_rebase": (
            "AUTHOR_VALIDATED_PENDING_FRESH_REVIEW"
        ),
        "canonical_integration": "NOT_AUTHORIZED_PENDING_FRESH_REVIEW",
        "governing_input_hashes": {
            "task": EXPECTED[TASK],
            "immutable_v1_candidate_manifest": EXPECTED[
                V1 / "candidate_manifest.json"
            ],
            "sealed_v1_review_manifest": EXPECTED[
                V1_REVIEW / "lane_manifest.json"
            ],
            "accepted_source_manifest": EXPECTED[
                AUTHOR / "lane_manifest.json"
            ],
            "accepted_source_review_manifest": EXPECTED[
                AUTHOR_REVIEW / "lane_manifest.json"
            ],
            "immutable_v1_mapping_at_build": (
                "4baf9a89f7fe70a042a35a368ebb9bb9c62aa79a925eaa70e10176f3024ffb0c"
            ),
            "current_post_candidate_mapping": EXPECTED[MAPPING],
            "source_archive_manifest": EXPECTED[
                ARCHIVE / "source_archive_manifest.csv"
            ],
            "visible_text_v2_manifest": EXPECTED[
                ARCHIVE / "derived_text_manifest.csv"
            ],
            "canonical_schema": EXPECTED[SCHEMA],
        },
        "resolved_repairs": [
            "V2-001 canonical support types",
            "V2-002 rounded Microsoft amount",
            "V2-003 partial and mixed investee scopes",
            "V2-004 real external foreign keys",
            "V2-005 field-level transformation lineage",
            "V2-006 Chapter 11 claim lineage",
        ],
        "remaining_holds": RETAINED_HOLDS,
        "limits": [
            "This is an isolated author candidate, not canonical data.",
            "Fresh independent review is required before any canonical merge.",
            (
                "Only source preservation is proposed for atomic clearance; "
                "all five analytical holds remain."
            ),
            (
                "Original inline-XBRL governs; visible-text v2 records are "
                "search and locator aids only."
            ),
        ],
        "files": files,
        "self_hash_policy": "candidate_manifest.json excludes its own hash",
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(output: Path) -> None:
    require_inputs()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copy_and_repair_base(output)
    build_schema(output)
    build_transformation_audit(output)
    build_atomic_preservation_candidate(output)
    build_input_lineage(output)
    build_manifest(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(sha256(args.output.resolve() / "candidate_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
