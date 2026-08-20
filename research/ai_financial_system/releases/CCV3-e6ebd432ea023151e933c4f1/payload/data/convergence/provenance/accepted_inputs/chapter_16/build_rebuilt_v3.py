#!/usr/bin/env python3
"""Build the isolated Chapter 16 rebuilt-v3 correction packet.

The builder reads the sealed v2 packet and its independent review, writes only
to its own output directory, applies the twelve governed repairs, and retains
all comparative-publication and canonical-write blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


OUT = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_2_chapter_16_rebuilt_v2/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(OUT)
STAGING = REPO / "research/ai_financial_system/staging"
SOURCE = STAGING / "lane_2_chapter_16_rebuilt_v2"
REVIEW = STAGING / "lane_1_verify_chapter_16_rebuilt_v2"
LOGICAL_LANE = "lane_3_chapter_16_rebuilt_v3"
STATUS_DATE = "2026-07-26"
CHANGED_SOURCE_CSVS = {
    "applied_exact_patch_audit.csv",
    "counterparty_role_matrix.csv",
    "input_lineage.csv",
    "project_centrality_snapshot.csv",
    "project_party_registry.csv",
    "project_role_edges.csv",
    "single_point_failure_register.csv",
    "writing_gate_audit.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header")
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
        raise ValueError(f"{path}: refusing headerless empty CSV")
    names = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "lane_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        if not path.exists() or digest(path) != artifact["sha256"]:
            raise ValueError(f"manifest mismatch: {path}")
    return manifest


def find(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    found = [row for row in rows if row.get(field) == value]
    if len(found) != 1:
        raise ValueError(f"expected one {field}={value}, found {len(found)}")
    return found[0]


def assert_and_set(
    row: dict[str, str], field: str, expected: str, replacement: str
) -> None:
    if row[field] != expected:
        raise ValueError(
            f"{field}: expected {expected!r}, observed {row[field]!r}"
        )
    row[field] = replacement


def seed_source_csvs(source_manifest: dict[str, object]) -> None:
    for artifact in source_manifest["files"]:
        source_path = SOURCE / artifact["path"]
        if source_path.suffix in {".csv", ".sql"}:
            shutil.copyfile(source_path, OUT / source_path.name)


def apply_identity_and_clock_repairs() -> None:
    parties = read_csv(OUT / "project_party_registry.csv")
    blackstone = find(parties, "project_party_registry_id", "PPARTY-C16-003")
    assert blackstone["party_id"] == "ENT-BLACKSTONE"
    assert_and_set(
        blackstone,
        "party_type",
        "canonical_party_record",
        "canonical_proxy_or_group",
    )
    assert_and_set(
        blackstone,
        "multiplicity",
        "one_record_identity_semantics_may_be_proxy",
        "identity_not_one_to_one",
    )
    assert_and_set(
        blackstone,
        "identity_status",
        "canonical_record",
        "manager_and_affiliated_fund_group",
    )
    blackstone["notes"] = (
        "Blackstone Inc. and affiliated funds is a manager/affiliate-group "
        "reference; the exact sponsor fund and project vehicle remain unresolved."
    )

    saline = find(parties, "project_party_registry_id", "PPARTY-C16-022")
    assert saline["party_id"] == "ENT-SALINE-DC"
    assert_and_set(
        saline,
        "party_type",
        "canonical_party_record",
        "canonical_research_project_construct",
    )
    assert_and_set(
        saline,
        "multiplicity",
        "one_record_identity_semantics_may_be_proxy",
        "identity_not_one_to_one",
    )
    assert_and_set(
        saline,
        "identity_status",
        "canonical_record",
        "project_research_construct_exact_legal_entity_unresolved",
    )
    saline["notes"] = (
        "Research project construct only; the exact legal borrower, property "
        "owner, and project vehicle remain unresolved."
    )
    write_csv(OUT / "project_party_registry.csv", parties)

    roles = read_csv(OUT / "project_role_edges.csv")
    saline_role = find(roles, "project_edge_id", "PROLE-160006")
    assert_and_set(
        saline_role,
        "role",
        "project_entity",
        "project_research_construct_exact_legal_entity_unresolved",
    )
    blackstone_role = find(roles, "project_edge_id", "PROLE-160008")
    assert_and_set(
        blackstone_role,
        "role",
        "equity_sponsor",
        "equity_sponsor_affiliate_group_exact_fund_and_vehicle_unresolved",
    )
    write_csv(OUT / "project_role_edges.csv", roles)

    snapshots = read_csv(OUT / "project_centrality_snapshot.csv")
    snapshot = find(snapshots, "snapshot_id", "PCS-160001")
    snapshot["role_types"] = (
        "developer_and_equity_sponsor_end_user;"
        "equity_sponsor_affiliate_group_exact_fund_and_vehicle_unresolved;"
        "facility_tenant_and_utility_customer;"
        "parent_guarantor_and_end_user;"
        "parent_private_debt_anchor;"
        "project_research_construct_exact_legal_entity_unresolved;"
        "regulated_utility_and_storage_provider"
    )
    snapshot["identity_state"] = (
        "bounded_project_construct_with_Blackstone_sponsor_affiliate_Saline_"
        "exact_legal_entity_lender_and_owner_unresolved"
    )
    write_csv(OUT / "project_centrality_snapshot.csv", snapshots)

    matrix = read_csv(OUT / "counterparty_role_matrix.csv")
    google = find(matrix, "role_matrix_id", "CRM-160005")
    assert google["entity_id"] == "ENT-GOOGL"
    google["legal_obligor_role"] = (
        "Alphabet_parent_proxy_for_guarantee_and_contract_party_exact_Google_"
        "legal_entity_unresolved"
    )
    google["guarantor_role"] = (
        "River_Bend_and_external_VIE_support_via_Alphabet_parent_proxy_exact_"
        "Google_support_entity_unresolved"
    )
    google["role_boundary"] = (
        "Alphabet_parent_proxy; exact_Google_contracting_and_support_legal_"
        "entity_unresolved; guarantee_exposure_does_not_establish_VIE_"
        "operating_control"
    )

    dte = find(matrix, "role_matrix_id", "CRM-160009")
    assert dte["entity_id"] == "ENT-DTE"
    dte["legal_obligor_role"] = (
        "DTE_parent_proxy_for_regulated_utility_and_residual_cost_holder_exact_"
        "operating_utility_unresolved"
    )
    dte["customer_role"] = (
        "utility_supplier_role_via_DTE_parent_proxy_exact_operating_utility_"
        "unresolved"
    )
    dte["supplier_role"] = (
        "storage_provider_role_via_DTE_parent_proxy_exact_operating_utility_"
        "unresolved"
    )
    dte["manager_role"] = (
        "utility_operator_role_via_DTE_parent_proxy_exact_operating_utility_"
        "unresolved"
    )
    dte["role_boundary"] = (
        "DTE_parent_proxy; exact_operating_utility_unresolved; residual_"
        "allocation_is_not_first_loss_priority"
    )
    write_csv(OUT / "counterparty_role_matrix.csv", matrix)

    failures = read_csv(OUT / "single_point_failure_register.csv")
    ares = find(failures, "failure_id", "SPF-160006")
    assert_and_set(
        ares,
        "documented_lag_or_trigger",
        "5_business_days_or_up_to_10_with_satisfactory_action",
        (
            "default_at_day_5_unless_satisfactory_cure_evidence_delivered_"
            "within_first_5_days_conditionally_extends_to_day_10"
        ),
    )
    write_csv(OUT / "single_point_failure_register.csv", failures)


def add_identity_gate() -> None:
    gates = read_csv(OUT / "writing_gate_audit.csv")
    if any(row["gate_id"] == "WGA-160015" for row in gates):
        raise ValueError("WGA-160015 already exists in immutable source")
    gates.append(
        {
            "gate_id": "WGA-160015",
            "publication_statement": (
                "Identity-labeled project and counterparty-role prose is "
                "publication-ready"
            ),
            "required_evidence": (
                "PATCH-V16V2-001_through_008_meaning_checks_all_pass_and_fresh_"
                "nonauthor_acceptance_review"
            ),
            "packet_status": (
                "semantic_repairs_applied_pending_fresh_nonauthor_acceptance"
            ),
            "evidence_grade": "D",
            "source_claim_id": "",
            "decision": (
                "block_pending_v3_semantic_validation_and_fresh_nonauthor_"
                "acceptance"
            ),
        }
    )
    write_csv(OUT / "writing_gate_audit.csv", gates)


def repair_prior_patch_accounting() -> None:
    rows = read_csv(OUT / "applied_exact_patch_audit.csv")
    patch = find(rows, "review_patch_id", "PATCH-V16R-0044")
    patch["implementation_status"] = "APPLIED"
    patch["implementation_artifact"] = "research_pass.md"
    patch["notes"] = (
        "Rebuilt-v3 explicitly distinguishes 574 structured candidates from "
        "44 separately reviewed adjudication records and the 618-row combined ledger."
    )
    write_csv(OUT / "applied_exact_patch_audit.csv", rows)


def write_lineage(
    source_manifest: dict[str, object], review_manifest: dict[str, object]
) -> None:
    rows = [
        {
            "input_id": "INP-V16V3-001",
            "input_lane": SOURCE.name,
            "manifest_sha256": digest(SOURCE / "lane_manifest.json"),
            "manifest_file_count": len(source_manifest["files"]),
            "role": "immutable_rebuilt_v2_source_packet",
            "write_policy": "read_only",
        },
        {
            "input_id": "INP-V16V3-002",
            "input_lane": REVIEW.name,
            "manifest_sha256": digest(REVIEW / "lane_manifest.json"),
            "manifest_file_count": len(review_manifest["files"]),
            "role": "authoritative_twelve_patch_independent_review",
            "write_policy": "read_only",
        },
    ]
    write_csv(OUT / "input_lineage.csv", rows)


def write_patch_accounting(review_manifest: dict[str, object]) -> None:
    patches = read_csv(REVIEW / "exact_row_patches.csv")
    if [row["patch_id"] for row in patches] != [
        f"PATCH-V16V2-{number:03d}" for number in range(1, 13)
    ]:
        raise ValueError("review patch register is not the governed 12-patch set")
    rows: list[dict[str, object]] = []
    for ordinal, patch in enumerate(patches, start=1):
        if patch["patch_id"] == "PATCH-V16V2-009":
            status = "APPLIED_BLOCK_RETAINED"
        elif patch["patch_id"] == "PATCH-V16V2-010":
            status = "APPLIED_REVIEW_REQUIRED_PENDING_NONAUTHOR_ACCEPTANCE"
        else:
            status = "APPLIED_PENDING_INDEPENDENT_ACCEPTANCE"
        rows.append(
            {
                "application_id": f"APP-V16V3-{ordinal:03d}",
                "patch_id": patch["patch_id"],
                "source_review_manifest_sha256": digest(
                    REVIEW / "lane_manifest.json"
                ),
                "severity": patch["severity"],
                "target": patch["target"],
                "row_id": patch["row_id"],
                "field": patch["field"],
                "before_state": patch["current"],
                "after_state": patch["replacement"],
                "implementation_status": status,
                "verification_rule": f"V3-PATCH-{ordinal:03d}",
                "release_effect": patch["release_effect"],
            }
        )
    write_csv(OUT / "v3_patch_accounting.csv", rows)


def write_preservation_audit(source_manifest: dict[str, object]) -> None:
    source_hashes = {
        artifact["path"]: artifact["sha256"] for artifact in source_manifest["files"]
    }
    rows: list[dict[str, object]] = []
    for name in sorted(
        artifact["path"]
        for artifact in source_manifest["files"]
        if artifact["path"].endswith(".csv")
    ):
        observed = digest(OUT / name)
        intended = name in CHANGED_SOURCE_CSVS
        same = observed == source_hashes[name]
        rows.append(
            {
                "artifact": name,
                "source_sha256": source_hashes[name],
                "v3_sha256": observed,
                "intentionally_changed": str(intended).lower(),
                "status": (
                    "PASS_INTENTIONAL_CHANGE"
                    if intended and not same
                    else "PASS_BYTE_PRESERVED"
                    if not intended and same
                    else "FAIL"
                ),
            }
        )
    write_csv(OUT / "preserved_artifact_hash_audit.csv", rows)


def write_reports() -> None:
    write_text(
        OUT / "research_pass.md",
        """# Chapter 16 rebuilt-v3 research pass

## Governed correction

This isolated packet applies the twelve exact repairs in
`lane_1_verify_chapter_16_rebuilt_v2`. It does not modify the v2 source lane,
the independent review, canonical ledgers, any database, chapters, or DOCX
artifacts.

The candidate universe contains **574 mechanically enumerated structured
candidates**. Of those, 44 are selected through separately reviewed atomic
role-adjudication records, 356 remain unadjudicated holds (297 role holds and
59 relationship-equivalence holds), and 218 are mechanically excluded. The 44
adjudication records are separate ledger records; adding them to the 574
structured candidates produces the 618-row combined candidate/adjudication
ledger. It would be incorrect to describe all 618 rows as independently
extracted candidates.

## Identity and contract-clock repairs

Blackstone is represented as a manager/affiliated-fund group with the exact
sponsor fund and vehicle unresolved. Saline is a research project construct
with the exact borrower, owner, and project vehicle unresolved. Those meanings
are propagated through project roles and the project snapshot. Alphabet and
DTE remain parent proxies where exact contracting, support, or operating
utility entities are unresolved.

The Ares warehouse trigger now preserves the conditional branch: default
occurs at day five unless satisfactory cure evidence is delivered within the
first five days, conditionally extending the deadline to day ten.

## Persistent boundary

This is a bounded documented subgraph, not a complete AI-system relationship
census. Comparative network ordering and canonical writes remain blocked. The
schema and loader remain design evidence only. Fresh nonauthor acceptance is
required before even the repaired bounded identity prose can be treated as
accepted.
""",
    )
    write_json(
        OUT / "semantic_validation_results.json",
        {
            "lane": LOGICAL_LANE,
            "chapter": "CH16",
            "status_date": STATUS_DATE,
            "builder_semantic_status": "REPAIRS_APPLIED",
            "packet_decision": "REVIEW_REQUIRED_FRESH_NONAUTHOR_ACCEPTANCE",
            "identity_semantics": "REPAIRED_V3_PENDING_NONAUTHOR_ACCEPTANCE",
            "contract_clock": "REPAIRED_V3_PENDING_NONAUTHOR_ACCEPTANCE",
            "structured_candidates": 574,
            "selected_role_edges": 44,
            "pending_role_holds": 297,
            "pending_relationship_equivalence_holds": 59,
            "mechanical_exclusions": 218,
            "combined_candidate_and_adjudication_ledger_rows": 618,
            "record_claim_links": 1726,
            "lossless_schema_loader_mappings": 19,
            "patches_applied": 12,
            "comparative_publication": "BLOCKED",
            "canonical_database_write": "NOT_PERFORMED_NOT_AUTHORIZED",
            "free_text_relationship_adjudication": "OPEN_BLOCKER_427_CLAIMS",
            "fixed_logical_packet_identifier": True,
        },
    )
    write_text(
        OUT / "validation_report.md",
        """# Chapter 16 rebuilt-v3 semantic regression report

**Builder decision:** **REVIEW REQUIRED pending fresh nonauthor acceptance.**

The twelve exact v2-review repairs are applied in this isolated lane. Meaning-
based validation is delegated to `validate_rebuilt_v3.py`, which independently
regenerates the preserved candidate, graph, lineage, concentration, and schema
controls and tests the repaired identity and contract-clock semantics.

Two clean builds are required to reproduce every manifested artifact and the
manifest byte-for-byte through `verify_full_rebuilds.py`. This deterministic
claim is accepted only when that executable comparison returns PASS.

Comparative publication remains blocked by 297 role holds, 59 relationship-
equivalence holds, and the unadjudicated 427-claim free-text population.
Canonical write is not authorized. This packet does not claim system-wide
network rank, comparative concentration, or canonical integration readiness.
""",
    )


def run_validator() -> None:
    subprocess.run(
        [sys.executable, str(OUT / "validate_rebuilt_v3.py")],
        cwd=REPO,
        check=True,
    )


def internal_clean_build_comparison() -> dict[str, object]:
    scripts = [
        "build_rebuilt_v3.py",
        "validate_rebuilt_v3.py",
        "verify_full_rebuilds.py",
    ]

    def build(target: Path) -> tuple[dict[str, object], str]:
        for name in scripts:
            shutil.copyfile(OUT / name, target / name)
        subprocess.run(
            [sys.executable, str(target / "build_rebuilt_v3.py"), "--core-only"],
            cwd=REPO,
            check=True,
        )
        manifest_path = target / "lane_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest, digest(manifest_path)

    with tempfile.TemporaryDirectory(
        prefix=".ch16v3_core_a_", dir=str(STAGING)
    ) as left_name:
        with tempfile.TemporaryDirectory(
            prefix=".ch16v3_core_b_", dir=str(STAGING)
        ) as right_name:
            left_root = Path(left_name) / LOGICAL_LANE
            right_root = Path(right_name) / LOGICAL_LANE
            left_root.mkdir()
            right_root.mkdir()
            left, left_manifest_hash = build(left_root)
            right, right_manifest_hash = build(right_root)
            left_hashes = {
                row["path"]: row["sha256"] for row in left["files"]
            }
            right_hashes = {
                row["path"]: row["sha256"] for row in right["files"]
            }
            status = (
                "PASS"
                if left_hashes == right_hashes
                and left_manifest_hash == right_manifest_hash
                else "FAIL"
            )
            if status != "PASS":
                raise ValueError("clean core rebuilds are not byte-identical")
            return {
                "status": status,
                "logical_lane": LOGICAL_LANE,
                "clean_builds_compared": 2,
                "manifested_artifacts_compared": len(left_hashes),
                "manifest_included_in_comparison": True,
                "shared_core_manifest_sha256": left_manifest_hash,
                "physical_output_path_in_manifested_semantics": False,
                "external_full_packet_verifier": "verify_full_rebuilds.py",
                "external_full_packet_comparison_required": True,
            }


def build(core_only: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = verify_manifest(SOURCE)
    review_manifest = verify_manifest(REVIEW)
    seed_source_csvs(source_manifest)
    write_lineage(source_manifest, review_manifest)
    apply_identity_and_clock_repairs()
    add_identity_gate()
    repair_prior_patch_accounting()
    write_patch_accounting(review_manifest)
    write_preservation_audit(source_manifest)
    write_reports()
    write_json(
        OUT / "patch_application_results.json",
        {
            "lane": LOGICAL_LANE,
            "status": "PASS_12_OF_12_APPLIED_PENDING_NONAUTHOR_ACCEPTANCE",
            "patches_expected": 12,
            "patches_applied": 12,
            "source_manifest_sha256": digest(SOURCE / "lane_manifest.json"),
            "review_manifest_sha256": digest(REVIEW / "lane_manifest.json"),
            "comparative_publication": "BLOCKED",
            "canonical_write": "NOT_AUTHORIZED",
        },
    )

    # The validator seals a deterministic core packet before the builder
    # performs the independent two-build comparison.
    run_validator()
    if core_only:
        return

    write_json(
        OUT / "deterministic_rebuild_results.json",
        internal_clean_build_comparison(),
    )
    run_validator()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    build(args.core_only)
    print(
        json.dumps(
            {
                "status": "CH16_REBUILT_V3_GENERATED",
                "logical_lane": LOGICAL_LANE,
                "core_only": args.core_only,
                "comparative_publication": "BLOCKED",
                "canonical_write": "NOT_AUTHORIZED",
                "acceptance": "FRESH_NONAUTHOR_REVIEW_REQUIRED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
