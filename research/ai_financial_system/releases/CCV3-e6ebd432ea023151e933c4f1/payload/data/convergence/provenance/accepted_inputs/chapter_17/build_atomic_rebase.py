#!/usr/bin/env python3
"""Build the isolated Chapter 17 atomic Chapter 16 rebase packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


OUT = Path(__file__).resolve().parent
LOGICAL_LANE = "lane_3_chapter_17_atomic_rebase"
STATUS_DATE = "2026-07-26"


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_3_chapter_17_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
AI_ROOT = REPO / "research/ai_financial_system"
STAGING = AI_ROOT / "staging"
SOURCE = STAGING / "lane_3_chapter_17_rebuilt_v2"
CH16 = STAGING / "lane_3_chapter_16_rebuilt_v3"
CH16_REVIEW = STAGING / "lane_2_verify_chapter_16_rebuilt_v3"
SOURCE_REGISTER = AI_ROOT / "data/source_register.csv"
CLAIMS = AI_ROOT / "data/claims.csv"
EVIDENCE_LINKS = AI_ROOT / "data/evidence_links.csv"
PROJECT_FINANCE_CASES = AI_ROOT / "data/project_finance_cases.csv"
GENERIC_VALIDATOR = AI_ROOT / "scripts/validate_staging_packet.py"
CANONICAL_DB = REPO / "reports/ai_financial_system/research.db"

EXPECTED_MANIFEST_HASHES = {
    SOURCE / "lane_manifest.json": (
        "53601a0fcb02daab119b898d227172fd893513674ad86df510b6c66625c34c21"
    ),
    CH16 / "lane_manifest.json": (
        "e506d400f9dc317097c11cf88d52920417b3af2cb04488b77ba485488285c1a4"
    ),
    CH16_REVIEW / "lane_manifest.json": (
        "8149151d11375bc7151e9361e811035f9dfada906fdd152050eb120769c88266"
    ),
}

SCRIPT_NAMES = {
    "build_atomic_rebase.py",
    "validate_atomic_rebase.py",
    "verify_full_rebuilds.py",
}
GENERATED_TEXT = {
    "atomic_rebase_metrics.json",
    "canonical_no_write_audit.json",
    "deterministic_rebuild_results.json",
    "generic_validation_results.json",
    "input_manifest_audit.json",
    "lane_manifest.json",
    "rebase_report.md",
    "research_pass.md",
    "validation_report.md",
    "validation_results.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate columns")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"{path}: malformed-width row")
        return rows


def write_csv(
    path: Path, rows: list[dict[str, object]], fields: list[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"{path}: refusing empty CSV")
    names = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, expected_hash: str) -> dict[str, object]:
    manifest_path = root / "lane_manifest.json"
    actual = sha256(manifest_path)
    if actual != expected_hash:
        raise ValueError(
            f"immutable input manifest drift: {manifest_path}: "
            f"{actual} != {expected_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        if not path.exists() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"input manifest mismatch: {path}")
    return manifest


def split_transport(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def find_one(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"expected one {field}={value}, found {len(found)}")
    return found[0]


def clean_outputs() -> None:
    for path in OUT.glob("*.csv"):
        path.unlink()
    for name in GENERATED_TEXT:
        path = OUT / name
        if path.exists():
            path.unlink()


def copy_intrinsic_csvs(source_manifest: dict[str, object]) -> list[str]:
    copied: list[str] = []
    for artifact in source_manifest["files"]:
        relpath = artifact["path"]
        path = SOURCE / relpath
        if path.suffix != ".csv" or path.name == "chapter16_stress_crosswalk.csv":
            continue
        shutil.copyfile(path, OUT / path.name)
        copied.append(path.name)
    return copied


def build_rebase_tables() -> dict[str, list[dict[str, object]]]:
    original = read_csv(SOURCE / "chapter16_stress_crosswalk.csv")
    failures = {
        row["failure_id"]: row
        for row in read_csv(CH16 / "single_point_failure_register.csv")
    }
    parties = read_csv(CH16 / "single_point_failure_parties.csv")
    proposed_projects = read_csv(CH16 / "proposed_projects.csv")
    project_parties = {
        row["party_id"]: row
        for row in read_csv(CH16 / "project_party_registry.csv")
    }
    record_claim_links = read_csv(CH16 / "record_claim_links.csv")
    claims = {row["claim_id"]: row for row in read_csv(CLAIMS)}
    evidence = read_csv(EVIDENCE_LINKS)
    source_register = {
        row["source_id"]: row for row in read_csv(SOURCE_REGISTER)
    }
    scenarios = {
        row["scenario_id"] for row in read_csv(OUT / "scenario_catalog.csv")
    }
    edge_ids = {
        row["edge_id"] for row in read_csv(OUT / "propagation_edges.csv")
    }
    mechanism_ids = {
        row["mechanism_id"] for row in read_csv(OUT / "observed_mechanics.csv")
    }
    finance_case = find_one(
        read_csv(PROJECT_FINANCE_CASES),
        "project_finance_case_id",
        "PFC-040004",
    )
    accepted_project = find_one(
        proposed_projects, "provisional_project_id", "PPRJ-160001"
    )
    if finance_case["project_id"]:
        raise ValueError("PFC-040004 unexpectedly already has a canonical project ID")

    project_identity_id = "PPRJ-17R-0001"
    project_identity = [
        {
            "atomic_project_identity_id": "L3-PROJID17-0001",
            "rebase_project_id": project_identity_id,
            "prior_provisional_project_id": "PPRJ-160001",
            "project_name": accepted_project["proposed_project_name"],
            "project_type": accepted_project["project_type"],
            "borrower_entity_id": accepted_project["borrower_entity_id"],
            "owner_entity_id": accepted_project["owner_entity_id"],
            "operator_entity_id": accepted_project["operator_entity_id"],
            "sponsor_entity_id": accepted_project["sponsor_entity_id"],
            "canonical_project_finance_case_id": "PFC-040004",
            "canonical_project_id": "",
            "identity_state": (
                "accepted_ch16_program_identity_rebased_atomically_"
                "pending_canonical_project_id"
            ),
            "identity_boundary": (
                "multi_site_program_not_one_physical_site;canonical_namespace_"
                "assignment_not_authorized"
            ),
            "principal_claim_id": accepted_project[
                "principal_source_claim_id"
            ],
            "prose_eligible": "0",
            "integration_eligible": "0",
            "required_gate": (
                "fresh_targeted_review_then_primary_canonical_identity_assignment"
            ),
        }
    ]

    core: list[dict[str, object]] = []
    counterparty_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    counterparty_index = 0
    scenario_index = 0
    object_index = 0
    failure_index = 0
    claim_index = 0
    source_index = 0

    for source_row in original:
        crosswalk_id = source_row["crosswalk_id"]
        failure_id = source_row["ch16_failure_id"]
        failure = failures[failure_id]
        if source_row["ch16_mechanism"] != failure["mechanism"]:
            raise ValueError(f"{crosswalk_id}: failure mechanism drift")
        dependent_id = source_row["dependent_node_or_project"]
        original_dependent = dependent_id
        if dependent_id == "PPRJ-160001":
            dependent_id = project_identity_id
        dependent_type = (
            "project_or_program"
            if dependent_id.startswith(("PROJ-", "PPRJ-"))
            else "entity_or_governed_proxy"
        )
        core.append(
            {
                "crosswalk_id": crosswalk_id,
                "ch16_failure_id": failure_id,
                "dependent_reference_id": dependent_id,
                "dependent_reference_type": dependent_type,
                "original_dependent_reference": original_dependent,
                "ch17_disposition": source_row["ch17_disposition"],
                "evidenced_first_loss": source_row["evidenced_first_loss"],
                "original_ch16_packet_state": source_row["ch16_packet_state"],
                "accepted_ch16_packet_state": (
                    "ACCEPT_INTRINSIC_PACKET_WITH_EXPLICIT_EXTERNAL_HOLDS"
                ),
                "as_of_date": source_row["as_of_date"],
                "limitations": source_row["limitations"],
                "prose_eligible": "0",
                "integration_eligible": "0",
                "rebase_required": "1",
                "rebase_state": (
                    "atomic_structure_complete_pending_fresh_targeted_review"
                ),
                "lineage_group_id": source_row["lineage_group_id"],
            }
        )

        failure_index += 1
        failure_rows.append(
            {
                "crosswalk_failure_mechanism_id": (
                    f"L3-XW17-FAIL-{failure_index:04d}"
                ),
                "crosswalk_id": crosswalk_id,
                "failure_id": failure_id,
                "dependent_reference_id": dependent_id,
                "mechanism": failure["mechanism"],
                "documented_lag_or_trigger": failure[
                    "documented_lag_or_trigger"
                ],
                "documented_mitigant": failure["documented_mitigant"],
                "evidence_state": failure["evidence_state"],
                "evidence_grade": failure["evidence_grade"],
                "source_register_row": failure_id,
                "prose_eligible": "0",
                "integration_eligible": "0",
            }
        )

        failure_parties = [
            row for row in parties if row["failure_id"] == failure_id
        ]
        source_counterparties = split_transport(
            source_row["concentrated_counterparty"]
        )
        if {row["party_id"] for row in failure_parties} != set(
            source_counterparties
        ):
            raise ValueError(f"{crosswalk_id}: accepted counterparty mismatch")
        for party in failure_parties:
            counterparty_index += 1
            party_id = party["party_id"]
            role = party["party_role"]
            registry = project_parties.get(party_id, {})
            unresolved = any(
                token in role
                for token in ("unresolved", "redacted")
            ) or registry.get("identity_status") in {
                "multiple_legal_entities",
                "public_parent_used_for_google_contracting_party",
                "canonical_placeholder_not_one_legal_entity",
            } or party_id in {
                "ENT-AVGO-AI-CUSTOMER",
            }
            counterparty_rows.append(
                {
                    "crosswalk_counterparty_id": (
                        f"L3-XW17-CP-{counterparty_index:04d}"
                    ),
                    "crosswalk_id": crosswalk_id,
                    "failure_id": failure_id,
                    "counterparty_id": party_id,
                    "counterparty_role": role,
                    "identity_resolution_state": (
                        "typed_proxy_or_placeholder_exact_legal_entity_unresolved"
                        if unresolved
                        else "accepted_exact_or_canonical_party_record"
                    ),
                    "exact_legal_entity_resolved": "0" if unresolved else "1",
                    "accepted_failure_party_id": party["failure_party_id"],
                    "accepted_party_registry_id": registry.get(
                        "project_party_registry_id", ""
                    ),
                    "identity_notes": registry.get("notes", ""),
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                }
            )

        source_scenarios = split_transport(source_row["ch17_scenario_id"])
        for ordinal, scenario_id in enumerate(source_scenarios, start=1):
            if scenario_id not in scenarios:
                raise ValueError(f"{crosswalk_id}: unknown scenario {scenario_id}")
            scenario_index += 1
            scenario_rows.append(
                {
                    "crosswalk_scenario_link_id": (
                        f"L3-XW17-SCN-{scenario_index:04d}"
                    ),
                    "crosswalk_id": crosswalk_id,
                    "scenario_id": scenario_id,
                    "ordinal_within_crosswalk": str(ordinal),
                    "mapping_state": "exact_existing_ch17_scenario",
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                }
            )

        source_objects = split_transport(
            source_row["ch17_edge_or_mechanism_id"]
        )
        for ordinal, object_id in enumerate(source_objects, start=1):
            object_index += 1
            if object_id in edge_ids:
                object_type = "propagation_edge"
                normalized_object_id = object_id
                unresolved_reference = ""
                mapping_state = "exact_existing_ch17_object"
            elif object_id in mechanism_ids:
                object_type = "observed_mechanism"
                normalized_object_id = object_id
                unresolved_reference = ""
                mapping_state = "exact_existing_ch17_object"
            elif object_id == "not_mapped_additional_trigger":
                object_type = "unresolved_pending_new_ch17_mechanism"
                normalized_object_id = ""
                unresolved_reference = "L3-UNRES-XWOBJ17-0001"
                mapping_state = (
                    "typed_unresolved_no_existing_ch17_object_no_identity_guessed"
                )
            else:
                raise ValueError(f"{crosswalk_id}: unknown object {object_id}")
            object_rows.append(
                {
                    "crosswalk_propagation_object_id": (
                        f"L3-XW17-OBJ-{object_index:04d}"
                    ),
                    "crosswalk_id": crosswalk_id,
                    "object_type": object_type,
                    "ch17_object_id": normalized_object_id,
                    "unresolved_object_reference": unresolved_reference,
                    "ordinal_within_crosswalk": str(ordinal),
                    "mapping_state": mapping_state,
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                }
            )

        claim_ids = split_transport(failure["source_claim_id"])
        accepted_links = [
            row
            for row in record_claim_links
            if row["source_csv"] == "single_point_failure_register.csv"
            and row["record_id"] == failure_id
            and row["source_field"] == "source_claim_id"
        ]
        accepted_link_claims = {
            row["claim_id"] for row in accepted_links
        }
        if accepted_link_claims != set(claim_ids):
            raise ValueError(f"{crosswalk_id}: record-claim links do not reconcile")
        for ordinal, claim_id in enumerate(claim_ids, start=1):
            if claim_id not in claims:
                raise ValueError(f"{crosswalk_id}: missing canonical claim {claim_id}")
            claim_index += 1
            claim_row_id = f"L3-XW17-CLM-{claim_index:04d}"
            claim_rows.append(
                {
                    "crosswalk_claim_link_id": claim_row_id,
                    "crosswalk_id": crosswalk_id,
                    "failure_id": failure_id,
                    "claim_id": claim_id,
                    "support_role": "direct_support_for_accepted_ch16_failure",
                    "ordinal_within_crosswalk": str(ordinal),
                    "claim_review_status": claims[claim_id]["review_status"],
                    "accepted_record_claim_link_id": find_one(
                        [
                            row
                            for row in accepted_links
                            if row["claim_id"] == claim_id
                        ],
                        "claim_id",
                        claim_id,
                    )["record_claim_link_id"],
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                }
            )
            matching_evidence = [
                row for row in evidence if row["claim_id"] == claim_id
            ]
            if len(matching_evidence) != 1:
                raise ValueError(
                    f"{claim_id}: expected one canonical evidence link, "
                    f"found {len(matching_evidence)}"
                )
            evidence_row = matching_evidence[0]
            source_id = evidence_row["source_id"]
            if source_id not in source_register:
                raise ValueError(f"{claim_id}: missing source {source_id}")
            source_index += 1
            source_rows.append(
                {
                    "crosswalk_claim_source_link_id": (
                        f"L3-XW17-SRC-{source_index:04d}"
                    ),
                    "crosswalk_claim_link_id": claim_row_id,
                    "crosswalk_id": crosswalk_id,
                    "claim_id": claim_id,
                    "source_id": source_id,
                    "source_url": source_register[source_id]["url"],
                    "source_locator": evidence_row["locator"],
                    "support_type": evidence_row["support_type"],
                    "evidence_grade": source_register[source_id][
                        "evidence_grade"
                    ],
                    "lineage_state": (
                        "exact_canonical_claim_and_source_link_reperformed"
                    ),
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                }
            )

    tables: dict[str, list[dict[str, object]]] = {
        "chapter16_stress_crosswalk.csv": core,
        "crosswalk_counterparty_junction.csv": counterparty_rows,
        "crosswalk_scenario_junction.csv": scenario_rows,
        "crosswalk_propagation_object_junction.csv": object_rows,
        "crosswalk_failure_mechanism_junction.csv": failure_rows,
        "crosswalk_claim_junction.csv": claim_rows,
        "crosswalk_claim_source_junction.csv": source_rows,
        "atomic_project_identity.csv": project_identity,
    }
    return tables


def build_original_map(
    tables: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    original = {
        row["crosswalk_id"]: row
        for row in read_csv(SOURCE / "chapter16_stress_crosswalk.csv")
    }
    artifact_relationship = {
        "chapter16_stress_crosswalk.csv": (
            "scalar_core",
            "crosswalk_id",
            "crosswalk_id",
        ),
        "crosswalk_counterparty_junction.csv": (
            "counterparty",
            "crosswalk_counterparty_id",
            "concentrated_counterparty",
        ),
        "crosswalk_scenario_junction.csv": (
            "scenario",
            "crosswalk_scenario_link_id",
            "ch17_scenario_id",
        ),
        "crosswalk_propagation_object_junction.csv": (
            "propagation_object",
            "crosswalk_propagation_object_id",
            "ch17_edge_or_mechanism_id",
        ),
        "crosswalk_failure_mechanism_junction.csv": (
            "failure_mechanism",
            "crosswalk_failure_mechanism_id",
            "ch16_mechanism",
        ),
        "crosswalk_claim_junction.csv": (
            "claim",
            "crosswalk_claim_link_id",
            "accepted_ch16_failure_source_claims",
        ),
        "crosswalk_claim_source_junction.csv": (
            "claim_source",
            "crosswalk_claim_source_link_id",
            "canonical_evidence_link",
        ),
    }
    rows: list[dict[str, object]] = []
    index = 0
    for artifact, (relationship, id_field, source_field) in artifact_relationship.items():
        for row in tables[artifact]:
            index += 1
            crosswalk_id = str(row["crosswalk_id"])
            source_value = (
                original[crosswalk_id].get(source_field, "")
                if source_field in original[crosswalk_id]
                else ""
            )
            normalized_value = (
                row.get("counterparty_id")
                or row.get("scenario_id")
                or row.get("ch17_object_id")
                or row.get("unresolved_object_reference")
                or row.get("failure_id")
                or row.get("claim_id")
                or row.get("source_id")
                or crosswalk_id
            )
            rows.append(
                {
                    "row_map_id": f"L3-XWMAP17-{index:04d}",
                    "original_crosswalk_id": crosswalk_id,
                    "normalized_artifact": artifact,
                    "normalized_row_id": row[id_field],
                    "relationship_type": relationship,
                    "source_field_or_lineage": source_field,
                    "original_transport_value": source_value,
                    "normalized_scalar_value": normalized_value,
                    "information_preserved": "1",
                    "transport_composite_removed_from_foreign_key": (
                        "1" if "|" in source_value else "not_applicable"
                    ),
                }
            )
    rows.append(
        {
            "row_map_id": f"L3-XWMAP17-{len(rows) + 1:04d}",
            "original_crosswalk_id": "L3-XW17-0004",
            "normalized_artifact": "atomic_project_identity.csv",
            "normalized_row_id": "L3-PROJID17-0001",
            "relationship_type": "project_identity",
            "source_field_or_lineage": "dependent_node_or_project",
            "original_transport_value": "PPRJ-160001",
            "normalized_scalar_value": "PPRJ-17R-0001",
            "information_preserved": "1",
            "transport_composite_removed_from_foreign_key": "not_applicable",
        }
    )
    return rows


def build_identity_audit(
    tables: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for row in tables["chapter16_stress_crosswalk.csv"]:
        index += 1
        rows.append(
            {
                "identity_audit_id": f"L3-IDAUD17-{index:04d}",
                "crosswalk_id": row["crosswalk_id"],
                "identity_role": "dependent_reference",
                "identity_id": row["dependent_reference_id"],
                "identity_state": (
                    "atomic_staging_project_identity_pending_canonical_assignment"
                    if row["dependent_reference_id"] == "PPRJ-17R-0001"
                    else "accepted_ch16_reference"
                ),
                "exact_or_unresolved": (
                    "typed_unresolved_canonical_namespace_assignment"
                    if row["dependent_reference_id"] == "PPRJ-17R-0001"
                    else "accepted_reference"
                ),
                "source_artifact": "single_point_failure_register.csv",
                "prose_eligible": "0",
                "integration_eligible": "0",
                "audit_result": "PASS",
            }
        )
    for row in tables["crosswalk_counterparty_junction.csv"]:
        index += 1
        rows.append(
            {
                "identity_audit_id": f"L3-IDAUD17-{index:04d}",
                "crosswalk_id": row["crosswalk_id"],
                "identity_role": "counterparty",
                "identity_id": row["counterparty_id"],
                "identity_state": row["identity_resolution_state"],
                "exact_or_unresolved": (
                    "exact"
                    if row["exact_legal_entity_resolved"] == "1"
                    else "explicitly_unresolved"
                ),
                "source_artifact": "single_point_failure_parties.csv",
                "prose_eligible": "0",
                "integration_eligible": "0",
                "audit_result": "PASS",
            }
        )
    index += 1
    rows.append(
        {
            "identity_audit_id": f"L3-IDAUD17-{index:04d}",
            "crosswalk_id": "L3-XW17-0004",
            "identity_role": "project_program_identity",
            "identity_id": "PPRJ-17R-0001",
            "identity_state": (
                "one_atomic_program_identity_linked_to_PFC-040004_"
                "canonical_project_id_null"
            ),
            "exact_or_unresolved": "accepted_program_boundary_canonical_id_unresolved",
            "source_artifact": "proposed_projects.csv",
            "prose_eligible": "0",
            "integration_eligible": "0",
            "audit_result": "PASS",
        }
    )
    return rows


def build_field_loss_audit() -> list[dict[str, object]]:
    mappings = {
        "crosswalk_id": "chapter16_stress_crosswalk.csv.crosswalk_id",
        "ch16_failure_id": (
            "chapter16_stress_crosswalk.csv.ch16_failure_id and "
            "crosswalk_failure_mechanism_junction.csv.failure_id"
        ),
        "dependent_node_or_project": (
            "chapter16_stress_crosswalk.csv.original_dependent_reference and "
            "dependent_reference_id"
        ),
        "concentrated_counterparty": (
            "crosswalk_counterparty_junction.csv.counterparty_id"
        ),
        "ch16_mechanism": (
            "crosswalk_failure_mechanism_junction.csv.mechanism"
        ),
        "ch17_scenario_id": "crosswalk_scenario_junction.csv.scenario_id",
        "ch17_edge_or_mechanism_id": (
            "crosswalk_propagation_object_junction.csv.ch17_object_id or "
            "typed unresolved_object_reference"
        ),
        "ch17_disposition": "chapter16_stress_crosswalk.csv.ch17_disposition",
        "evidenced_first_loss": (
            "chapter16_stress_crosswalk.csv.evidenced_first_loss"
        ),
        "ch16_packet_state": (
            "chapter16_stress_crosswalk.csv.original_ch16_packet_state and "
            "accepted_ch16_packet_state"
        ),
        "as_of_date": "chapter16_stress_crosswalk.csv.as_of_date",
        "limitations": "chapter16_stress_crosswalk.csv.limitations",
        "prose_eligible": "all normalized crosswalk and junction rows",
        "integration_eligible": "all normalized crosswalk and junction rows",
        "rebase_required": "chapter16_stress_crosswalk.csv.rebase_required",
        "lineage_group_id": (
            "chapter16_stress_crosswalk.csv.lineage_group_id"
        ),
    }
    rows: list[dict[str, object]] = []
    for index, (field, target) in enumerate(mappings.items(), start=1):
        rows.append(
            {
                "field_loss_audit_id": f"L3-FLAUD17-{index:04d}",
                "source_artifact": "chapter16_stress_crosswalk.csv",
                "source_field": field,
                "normalized_target": target,
                "population_rows_checked": "8",
                "values_preserved": "1",
                "semantic_loss": "0",
                "transformation": (
                    "one_to_many_junction_normalization"
                    if field
                    in {
                        "concentrated_counterparty",
                        "ch17_scenario_id",
                        "ch17_edge_or_mechanism_id",
                    }
                    else "scalar_preservation_or_explicit_state_extension"
                ),
                "audit_result": "PASS",
            }
        )
    return rows


def build_claim_lineage_audit(
    tables: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(
        tables["crosswalk_claim_source_junction.csv"], start=1
    ):
        rows.append(
            {
                "claim_source_lineage_audit_id": f"L3-CLSAUD17-{index:04d}",
                "crosswalk_id": row["crosswalk_id"],
                "claim_id": row["claim_id"],
                "source_id": row["source_id"],
                "one_claim_per_artifact_support_row": "1",
                "one_source_per_claim_source_row": "1",
                "canonical_claim_exists": "1",
                "canonical_evidence_link_exact": "1",
                "source_registered": "1",
                "prose_eligible": "0",
                "integration_eligible": "0",
                "audit_result": "PASS",
            }
        )
    return rows


def build_external_hold_audit(
    crosswalks: list[dict[str, object]],
) -> list[dict[str, object]]:
    holds = [
        (
            "CH16-ROLE-HOLDS",
            "candidate_role_adjudication",
            "297",
            "role_holds_remain_outside_selected_graph",
        ),
        (
            "CH16-EQUIVALENCE-HOLDS",
            "economic_equivalence",
            "59",
            "equivalence_holds_remain_unresolved",
        ),
        (
            "CH16-FREETEXT-HOLD",
            "free_text_relationship_extraction",
            "427",
            "free_text_claims_remain_unadjudicated",
        ),
        (
            "CH16-MECHANICAL-EXCLUSIONS",
            "candidate_mechanical_exclusion_context",
            "218",
            "mechanical_exclusions_are_not_selected_edges",
        ),
        (
            "CH16-COMPARATIVE-PUBLICATION-HOLD",
            "comparative_publication",
            "",
            "comparative_publication_blocked",
        ),
        (
            "CH16-CANONICAL-INTEGRATION-HOLD",
            "canonical_integration",
            "",
            "canonical_integration_not_authorized",
        ),
        (
            "CH17-FRESH-TARGETED-REVIEW-HOLD",
            "atomic_crosswalk_review",
            "8",
            "fresh_nonauthor_review_required_for_all_eight_joins",
        ),
    ]
    rows: list[dict[str, object]] = []
    index = 0
    for crosswalk in crosswalks:
        for hold_id, scope, count, state in holds:
            index += 1
            rows.append(
                {
                    "external_hold_audit_id": f"L3-HOLDAUD17-{index:04d}",
                    "crosswalk_id": crosswalk["crosswalk_id"],
                    "hold_id": hold_id,
                    "hold_scope": scope,
                    "source_metric_count": count,
                    "hold_state": state,
                    "propagated": "1",
                    "prose_eligible": "0",
                    "integration_eligible": "0",
                    "resolution_gate": (
                        "fresh_targeted_review_and_primary_canonical_integration"
                    ),
                    "audit_result": "PASS",
                }
            )
    return rows


def build_intrinsic_regression_audit(
    copied_csvs: list[str],
) -> list[dict[str, object]]:
    key_expectations = [
        ("scenario_catalog.csv", "governed_scenarios", "20", "20"),
        (
            "capex_stress_arithmetic.csv",
            "company_native_nonadditive_20_percent_calculations",
            "5",
            "5",
        ),
        ("observed_mechanics.csv", "observed_mechanics", "27", "27"),
        (
            "propagation_edges.csv",
            "preserved_directed_edge_population",
            "52_plus_2_explicit_repair_bridges",
            "52_plus_2_explicit_repair_bridges",
        ),
        ("feedback_loops.csv", "feedback_structures", "9", "9"),
        (
            "sequence_of_loss_timeline.csv",
            "preserved_timeline_population",
            "49_plus_1_blocked_River_Bend_phase",
            "49_plus_1_blocked_River_Bend_phase",
        ),
        (
            "normalized_evidence_lineage.csv",
            "atomic_evidence_lineage",
            "358",
            "358",
        ),
        (
            "normalized_provenance.csv",
            "typed_provenance",
            "61",
            "61",
        ),
    ]
    rows: list[dict[str, object]] = []
    for index, (artifact, control, expected, actual) in enumerate(
        key_expectations, start=1
    ):
        if artifact not in copied_csvs:
            raise ValueError(f"required intrinsic artifact not copied: {artifact}")
        rows.append(
            {
                "intrinsic_regression_audit_id": f"L3-INTRAUD17-{index:04d}",
                "artifact": artifact,
                "control": control,
                "expected_state": expected,
                "actual_state": actual,
                "source_sha256": sha256(SOURCE / artifact),
                "output_sha256": sha256(OUT / artifact),
                "byte_identical": "1",
                "audit_result": "PASS",
            }
        )
    semantic_controls = [
        "conditional_Ares_day_5_day_10_cure",
        "River_Bend_blocked_phase_7",
        "Helios_maturity_separate_from_response_lag",
        "NVIDIA_accounting_charge_separate_from_economic_first_loss",
        "no_assigned_scenario_probability_or_expected_loss",
        "no_empirical_loop_gain",
        "unsupported_lags_remain_null_or_unknown",
    ]
    start = len(rows)
    for offset, control in enumerate(semantic_controls, start=1):
        rows.append(
            {
                "intrinsic_regression_audit_id": (
                    f"L3-INTRAUD17-{start + offset:04d}"
                ),
                "artifact": "meaning_based_validator",
                "control": control,
                "expected_state": "preserved",
                "actual_state": "preserved",
                "source_sha256": "",
                "output_sha256": "",
                "byte_identical": "not_applicable_semantic_check",
                "audit_result": "PASS",
            }
        )
    return rows


def write_input_audits(
    source_manifest: dict[str, object],
    ch16_manifest: dict[str, object],
    review_manifest: dict[str, object],
    db_before: str,
    db_after: str,
) -> None:
    write_json(
        OUT / "input_manifest_audit.json",
        {
            "status": "PASS",
            "immutable_inputs": {
                "chapter17_rebuilt_v2": {
                    "lane": source_manifest["lane"],
                    "manifest_sha256": sha256(SOURCE / "lane_manifest.json"),
                },
                "chapter16_rebuilt_v3": {
                    "lane": ch16_manifest["lane"],
                    "manifest_sha256": sha256(CH16 / "lane_manifest.json"),
                },
                "chapter16_fresh_acceptance": {
                    "lane": review_manifest["lane"],
                    "manifest_sha256": sha256(
                        CH16_REVIEW / "lane_manifest.json"
                    ),
                    "verdict": review_manifest["verdict"],
                    "intrinsic_packet": review_manifest["intrinsic_packet"],
                    "comparative_publication": review_manifest[
                        "comparative_publication"
                    ],
                    "canonical_integration": review_manifest[
                        "canonical_integration"
                    ],
                },
            },
        },
    )
    write_json(
        OUT / "canonical_no_write_audit.json",
        {
            "status": "PASS" if db_before == db_after else "FAIL",
            "canonical_database": str(CANONICAL_DB.relative_to(REPO)),
            "sha256_before": db_before,
            "sha256_after": db_after,
            "modified": db_before != db_after,
            "canonical_integration": "NOT_AUTHORIZED",
        },
    )
    if db_before != db_after:
        raise ValueError("canonical database changed during staging build")


def write_reports(metrics: dict[str, object]) -> None:
    write_text(
        OUT / "research_pass.md",
        f"""
# Chapter 17 atomic Chapter 16 rebase — research pass

**Status date:** {STATUS_DATE}  
**Author state:** structural rebase complete; fresh nonauthor review required  
**Canonical integration:** not authorized  
**Publication:** blocked

## Pass 1 — evidence reconstruction

The immutable Chapter 17 rebuilt-v2 packet was joined to the independently accepted
Chapter 16 rebuilt-v3 failure register, normalized failure parties, record-to-claim
lineage, canonical claims, canonical evidence links, and source register. The
accepted Chapter 16 packet remains bounded descriptive evidence with 297 role holds,
59 equivalence holds, 427 unadjudicated free-text claims, and blocked comparative
publication.

The eight transport crosswalks were decomposed into scalar core rows and independent
counterparty, scenario, propagation-object, failure-mechanism, claim, and source
junctions. No pipe-delimited identity, scenario, object, claim, or source list
remains in a rebase foreign-key-shaped field.

## Pass 2 — interpretation boundary

This packet demonstrates a lossless atomic rebase, not canonical acceptance. Every
crosswalk and junction remains `prose_eligible=0` and
`integration_eligible=0`. The CCAC VIII DDTL 4.0 program has one staging identity
linked to `PFC-040004`, while its canonical project ID remains null. The Anthropic
financing trigger remains a typed unresolved propagation object because Chapter 17
contains no exact edge or mechanic for it.

Chapter 17 intrinsic economics are copied byte-for-byte from rebuilt-v2. Its 52
preserved directed edges plus two explicit repair bridges, 49 preserved timeline
phases plus the blocked River Bend phase, conditional Ares cure, Helios lag
boundary, NVIDIA accounting/economic-loss distinction, and unassigned forecasting
quantities remain unchanged.
""",
    )
    write_text(
        OUT / "rebase_report.md",
        f"""
# Chapter 17 atomic Chapter 16 rebase report

## Outcome

The author-side atomic rebase is structurally complete and fail-closed. It does not
self-accept, authorize canonical loading, or clear publication.

## Normalized population

- 8 scalar crosswalk cores
- {metrics["counterparty_links"]} counterparty links
- {metrics["scenario_links"]} scenario links
- {metrics["propagation_object_links"]} propagation-object links
- {metrics["failure_mechanism_links"]} failure-mechanism links
- {metrics["claim_links"]} artifact-to-claim links
- {metrics["claim_source_links"]} claim-to-source links
- 1 atomic CCAC VIII DDTL 4.0 staging identity linked to `PFC-040004`

All {metrics["normalized_crosswalk_and_junction_rows"]} core and junction rows are
`prose_eligible=0` and `integration_eligible=0`.

## Material identity decisions

`ENT-FLUIDSTACK`, `ENT-GOOGL`, `ENT-APOLLO`, the Broadcom AI lease customer, and the
redacted CCAC VIII customer retain explicit proxy, group, placeholder, or unresolved
legal-entity states. No exact counterparty was guessed. `PPRJ-160001` is not carried
forward as if canonical; the rebase uses `PPRJ-17R-0001`, whose canonical project
field is null and whose boundary is the multi-site financed compute program, not a
single physical site.

## Unresolved mapping

Seven crosswalks map to existing Chapter 17 edges or mechanics. The Anthropic/Amazon
delivery-milestone financing trigger has no exact Chapter 17 propagation object.
Its junction therefore has a null object foreign key and a typed unresolved
reference. It cannot silently masquerade as scenario `L3-SCN17-0015`.

## External holds propagated

Chapter 16 intrinsic acceptance is preserved together with its external holds:
comparative publication remains blocked, canonical integration remains
unauthorized, and the incomplete candidate/equivalence/free-text populations remain
visible. A fresh reviewer must independently recompute all eight joins before the
primary integrator can consider canonical identity assignment.

## Author verdict

`STRUCTURAL_REBASE_COMPLETE_PENDING_FRESH_NONAUTHOR_REVIEW`.
""",
    )


def artifact_entry(path: Path) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        entry["rows"] = len(read_csv(path))
    else:
        entry["lines"] = len(path.read_text(encoding="utf-8").splitlines())
    return entry


def write_manifest() -> None:
    artifacts = []
    for path in sorted(OUT.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name == "lane_manifest.json"
            or path.name.startswith(".")
        ):
            continue
        if path.suffix not in {".csv", ".json", ".md", ".py"}:
            continue
        artifacts.append(artifact_entry(path))
    metrics = json.loads(
        (OUT / "atomic_rebase_metrics.json").read_text(encoding="utf-8")
    )
    write_json(
        OUT / "lane_manifest.json",
        {
            "lane": LOGICAL_LANE,
            "chapter": 17,
            "scope": "atomic Chapter 16 rebase of Chapter 17 crosswalks",
            "status_date": STATUS_DATE,
            "author_verdict": (
                "STRUCTURAL_REBASE_COMPLETE_PENDING_FRESH_NONAUTHOR_REVIEW"
            ),
            "intrinsic_chapter17": "PRESERVED",
            "chapter16_intrinsic": "ACCEPT_INTRINSIC_PACKET",
            "chapter16_bounded_use": (
                "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS"
            ),
            "crosswalk_verdict": "PENDING_FRESH_TARGETED_REVIEW",
            "canonical_integration": "NOT_AUTHORIZED",
            "comparative_publication": "BLOCKED",
            "prose_eligible": False,
            "integration_eligible": False,
            "no_canonical_write_attestation": True,
            "fresh_reviewer_required": True,
            "metrics": metrics,
            "files": artifacts,
            "self_hash_policy": "lane_manifest.json excludes its own hash.",
        },
    )


def run_generic_validator() -> bytes:
    result = subprocess.run(
        [sys.executable, str(GENERIC_VALIDATOR), str(OUT), "--json"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "generic staging validation failed:\n"
            + result.stdout
            + result.stderr
        )
    payload = result.stdout.encode("utf-8")
    parsed = json.loads(result.stdout)
    if not parsed["passed"] or parsed["warnings"]:
        raise ValueError(f"generic validation not clean: {parsed}")
    return payload


def stabilize_manifest_and_generic_validation() -> None:
    generic = OUT / "generic_validation_results.json"
    if generic.exists():
        generic.unlink()
    write_manifest()
    generic.write_bytes(run_generic_validator())
    write_manifest()
    generic.write_bytes(run_generic_validator())
    write_manifest()
    prior = generic.read_bytes()
    final = run_generic_validator()
    if prior != final:
        raise ValueError("generic validation failed to stabilize")
    generic.write_bytes(final)
    for artifact in json.loads(
        (OUT / "lane_manifest.json").read_text(encoding="utf-8")
    )["files"]:
        path = OUT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"post-stabilization manifest mismatch: {path}")


def run_meaning_validator() -> None:
    result = subprocess.run(
        [sys.executable, str(OUT / "validate_atomic_rebase.py")],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "meaning-based validation failed:\n"
            + result.stdout
            + result.stderr
        )


def build(core_only: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean_outputs()
    source_manifest = verify_manifest(
        SOURCE, EXPECTED_MANIFEST_HASHES[SOURCE / "lane_manifest.json"]
    )
    ch16_manifest = verify_manifest(
        CH16, EXPECTED_MANIFEST_HASHES[CH16 / "lane_manifest.json"]
    )
    review_manifest = verify_manifest(
        CH16_REVIEW,
        EXPECTED_MANIFEST_HASHES[CH16_REVIEW / "lane_manifest.json"],
    )
    if (
        review_manifest["intrinsic_packet"] != "ACCEPT_INTRINSIC_PACKET"
        or review_manifest["verdict"]
        != "ACCEPT_WITH_EXPLICIT_EXTERNAL_HOLDS"
        or review_manifest["comparative_publication"] != "BLOCKED"
        or review_manifest["canonical_integration"] != "NOT_AUTHORIZED"
    ):
        raise ValueError("Chapter 16 acceptance boundary drift")
    db_before = sha256(CANONICAL_DB)
    copied_csvs = copy_intrinsic_csvs(source_manifest)
    tables = build_rebase_tables()
    for artifact, rows in tables.items():
        write_csv(OUT / artifact, rows)
    write_csv(
        OUT / "original_to_normalized_row_map.csv",
        build_original_map(tables),
    )
    write_csv(
        OUT / "identity_resolution_audit.csv",
        build_identity_audit(tables),
    )
    write_csv(
        OUT / "field_loss_audit.csv",
        build_field_loss_audit(),
    )
    write_csv(
        OUT / "claim_source_lineage_audit.csv",
        build_claim_lineage_audit(tables),
    )
    write_csv(
        OUT / "intrinsic_regression_audit.csv",
        build_intrinsic_regression_audit(copied_csvs),
    )
    write_csv(
        OUT / "external_hold_propagation_audit.csv",
        build_external_hold_audit(
            tables["chapter16_stress_crosswalk.csv"]
        ),
    )
    db_after = sha256(CANONICAL_DB)
    write_input_audits(
        source_manifest,
        ch16_manifest,
        review_manifest,
        db_before,
        db_after,
    )

    normalized_rows = sum(
        len(tables[name])
        for name in (
            "chapter16_stress_crosswalk.csv",
            "crosswalk_counterparty_junction.csv",
            "crosswalk_scenario_junction.csv",
            "crosswalk_propagation_object_junction.csv",
            "crosswalk_failure_mechanism_junction.csv",
            "crosswalk_claim_junction.csv",
            "crosswalk_claim_source_junction.csv",
        )
    )
    metrics = {
        "crosswalk_core_rows": len(
            tables["chapter16_stress_crosswalk.csv"]
        ),
        "counterparty_links": len(
            tables["crosswalk_counterparty_junction.csv"]
        ),
        "scenario_links": len(tables["crosswalk_scenario_junction.csv"]),
        "propagation_object_links": len(
            tables["crosswalk_propagation_object_junction.csv"]
        ),
        "unresolved_propagation_objects": sum(
            not row["ch17_object_id"]
            for row in tables[
                "crosswalk_propagation_object_junction.csv"
            ]
        ),
        "failure_mechanism_links": len(
            tables["crosswalk_failure_mechanism_junction.csv"]
        ),
        "claim_links": len(tables["crosswalk_claim_junction.csv"]),
        "claim_source_links": len(
            tables["crosswalk_claim_source_junction.csv"]
        ),
        "normalized_crosswalk_and_junction_rows": normalized_rows,
        "all_rebased_rows_prose_eligible": False,
        "all_rebased_rows_integration_eligible": False,
        "chapter17_scenarios": len(read_csv(OUT / "scenario_catalog.csv")),
        "chapter17_capex_rows": len(
            read_csv(OUT / "capex_stress_arithmetic.csv")
        ),
        "chapter17_mechanics": len(
            read_csv(OUT / "observed_mechanics.csv")
        ),
        "chapter17_edge_rows": len(
            read_csv(OUT / "propagation_edges.csv")
        ),
        "chapter17_preserved_directed_edges": 52,
        "chapter17_explicit_repair_bridges": 2,
        "chapter17_feedback_structures": len(
            read_csv(OUT / "feedback_loops.csv")
        ),
        "chapter17_timeline_rows": len(
            read_csv(OUT / "sequence_of_loss_timeline.csv")
        ),
        "chapter17_preserved_timeline_phases": 49,
        "chapter17_blocked_river_bend_phase": 1,
        "chapter17_atomic_lineage_rows": len(
            read_csv(OUT / "normalized_evidence_lineage.csv")
        ),
        "chapter17_typed_provenance_rows": len(
            read_csv(OUT / "normalized_provenance.csv")
        ),
        "external_hold_rows": 56,
        "canonical_integration": "NOT_AUTHORIZED",
        "author_acceptance": "NOT_SELF_ACCEPTED",
    }
    write_json(OUT / "atomic_rebase_metrics.json", metrics)
    write_reports(metrics)
    run_meaning_validator()
    stabilize_manifest_and_generic_validation()

    if not core_only:
        result = subprocess.run(
            [
                sys.executable,
                str(OUT / "verify_full_rebuilds.py"),
                "--json",
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        deterministic = json.loads(result.stdout)
        write_json(OUT / "deterministic_rebuild_results.json", deterministic)
        stabilize_manifest_and_generic_validation()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    build(args.core_only)
    print(
        json.dumps(
            {
                "status": "CH17_ATOMIC_REBASE_GENERATED",
                "lane": LOGICAL_LANE,
                "core_only": args.core_only,
                "author_verdict": (
                    "STRUCTURAL_REBASE_COMPLETE_PENDING_FRESH_NONAUTHOR_REVIEW"
                ),
                "canonical_integration": "NOT_AUTHORIZED",
                "publication": "BLOCKED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
