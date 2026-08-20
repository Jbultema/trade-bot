#!/usr/bin/env python3
"""Meaning-based author validator for Chapter 18 rebuilt-v5.

This validator explicitly re-performs every FAR18-XP repair and the inherited
research/governance controls.  It is author-side evidence only and cannot
substitute for the fresh nonauthor review required by the task.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent


def discover_repo(start: Path) -> Path:
    for ancestor in (start, *start.parents):
        if (
            ancestor
            / "research/ai_financial_system/staging/"
            "lane_2_chapter_18_rebuilt_v3/lane_manifest.json"
        ).exists():
            return ancestor
    raise RuntimeError("cannot resolve repository root")


REPO = discover_repo(HERE)
STAGING = REPO / "research/ai_financial_system/staging"
SOURCE = STAGING / "lane_2_chapter_18_rebuilt_v3"
REVIEW = STAGING / "lane_1_verify_chapter_18_rebuilt_v3"
SOURCE_MANIFEST_SHA256 = (
    "9be24f7ce6187b359fb4500befb91ad0a54368802155d1647f38cb8f422937e5"
)
REVIEW_MANIFEST_SHA256 = (
    "077a5856a3834b43644f1d8dfe5e9cd3dabe7fa224ded4f392654555fd723373"
)
VERDICT = "READY_FOR_FRESH_NONAUTHOR_REVIEW_WITH_EXTERNAL_HOLDS"

ARTIFACT_KEYS = {
    "analogue_episode_ledger.csv": "episode_id",
    "capacity_demand_price_capital_timeline.csv": "timeline_id",
    "impairment_default_restructuring_chronology.csv": "chronology_id",
    "claimant_loss_boundary.csv": "claimant_loss_id",
    "survivor_recovery_mechanisms.csv": "mechanism_id",
    "ai_similarity_non_similarity_matrix.csv": "matrix_id",
    "contradictions.csv": "contradiction_id",
}
EXPECTED_RECORD_COUNTS = {
    "analogue_episode_ledger.csv": 10,
    "capacity_demand_price_capital_timeline.csv": 35,
    "impairment_default_restructuring_chronology.csv": 20,
    "claimant_loss_boundary.csv": 13,
    "survivor_recovery_mechanisms.csv": 16,
    "ai_similarity_non_similarity_matrix.csv": 20,
    "contradictions.csv": 14,
}
REQUIRED_FREEZE_ELEMENTS = {
    "data_version_and_cutoff",
    "population",
    "numerator",
    "denominator",
    "threshold",
    "tolerance",
    "horizon",
    "censoring",
    "cost_and_support_treatment",
    "missing_data_rule",
    "prospective_decision_rule",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def find(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, str]:
    matched = [row for row in rows if row.get(field) == value]
    if len(matched) != 1:
        raise ValueError(
            f"expected one {field}={value!r}, observed {len(matched)}"
        )
    return matched[0]


def split_refs(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def verify_manifest(root: Path, expected: str) -> dict[str, object]:
    manifest_path = root / "lane_manifest.json"
    if digest(manifest_path) != expected:
        raise AssertionError(f"immutable manifest changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        assert path.exists(), path
        assert digest(path) == artifact["sha256"], path
    return manifest


def rows_by_id(path: Path, field: str) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    values = [row[field] for row in rows]
    assert len(values) == len(set(values))
    return {row[field]: row for row in rows}


def linked_claims() -> dict[tuple[str, str], list[dict[str, str]]]:
    _, rows = read_csv(HERE / "record_claim_junction.csv")
    result: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(
            (row["artifact_name"], row["record_id"]), []
        ).append(row)
    return result


def claim_set(
    links: dict[tuple[str, str], list[dict[str, str]]],
    artifact: str,
    record_id: str,
) -> set[str]:
    return {
        row["atomic_claim_id"] for row in links[(artifact, record_id)]
    }


class CheckRunner:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def check(self, check_id: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            self.rows.append(
                {"check_id": check_id, "status": "PASS", "detail": detail}
            )
        except Exception as exc:
            self.rows.append(
                {
                    "check_id": check_id,
                    "status": "FAIL",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )


def validate_immutable_inputs() -> str:
    source_manifest = verify_manifest(SOURCE, SOURCE_MANIFEST_SHA256)
    review_manifest = verify_manifest(REVIEW, REVIEW_MANIFEST_SHA256)
    _, audit = read_csv(HERE / "immutable_input_hash_audit.csv")
    expected_rows = 2 + len(source_manifest["files"]) + len(
        review_manifest["files"]
    )
    assert expected_rows == 59
    assert len(audit) == expected_rows
    assert all(
        row["immutability_result"] == "PASS_INPUT_IMMUTABLE"
        and row["manifest_expected_sha256"]
        == row["observed_before_sha256"]
        == row["observed_after_sha256"]
        for row in audit
    )
    assert {SOURCE_MANIFEST_SHA256, REVIEW_MANIFEST_SHA256} <= {
        row["manifest_expected_sha256"]
        for row in audit
        if row["input_path"] == "lane_manifest.json"
    }
    return "59 source/review manifest entries are byte-verified before and after build"


def validate_preserved_controls() -> str:
    byte_identical = [
        "capacity_demand_price_capital_timeline.csv",
        "preserved_arithmetic_and_nonadditivity_audit.csv",
        "coefficient_transfer_prohibitions.csv",
        "evidence_extract_register.csv",
    ]
    for name in byte_identical:
        assert digest(HERE / name) == digest(SOURCE / name), name

    _, timeline = read_csv(HERE / byte_identical[0])
    assert len(timeline) == 35
    _, arithmetic = read_csv(HERE / byte_identical[1])
    assert len(arithmetic) == 20
    assert all(row["status"].startswith("PASS") for row in arithmetic)
    _, prohibitions = read_csv(HERE / byte_identical[2])
    assert len(prohibitions) == 18
    assert all(
        row["prohibited_coefficient_or_mapping"]
        and row["minimum_matching_evidence_before_reconsideration"]
        for row in prohibitions
    )
    _, extracts = read_csv(HERE / byte_identical[3])
    assert len(extracts) == 58
    assert all(
        row["archive_state"] == "no_local_source_snapshot"
        and row["reproducibility_status"]
        == "HOLD_external_content_identity_requires_live_or_archived_reverification"
        for row in extracts
    )

    _, hypotheses = read_csv(HERE / "hypothesis_tests.csv")
    assert len(hypotheses) == 16
    assert digest(HERE / "hypothesis_tests.csv") == digest(
        SOURCE / "hypothesis_tests.csv"
    )
    for row in hypotheses:
        assert (
            row["execution_status"]
            == "conceptual_only_thresholds_and_matching_not_frozen"
        )
        assert row["protocol_state"] == "not_frozen"
        assert row["predictive_use"] == "BLOCK"
        assert set(split_refs(row["required_frozen_elements"])) == (
            REQUIRED_FREEZE_ELEMENTS
        )
        assert len(split_refs(row["required_frozen_elements"])) == 11
    return (
        "17 timeline repairs, 16 conceptual hypotheses, 20 arithmetic controls, "
        "18 coefficient prohibitions, and 58 external holds are preserved"
    )


def validate_claim_extract_population() -> str:
    _, claims = read_csv(HERE / "proposed_claims.csv")
    _, evidence_links = read_csv(HERE / "proposed_evidence_links.csv")
    _, extracts = read_csv(HERE / "evidence_extract_register.csv")
    assert len(claims) == len(evidence_links) == len(extracts) == 58
    claim_ids = {row["claim_id"] for row in claims}
    assert claim_ids == {
        row["source_claim_id"] for row in evidence_links
    } == {row["source_claim_id"] for row in extracts}
    claims_by_id = {row["claim_id"]: row for row in claims}
    links_by_id = {
        row["source_claim_id"]: row for row in evidence_links
    }
    for extract in extracts:
        claim = claims_by_id[extract["source_claim_id"]]
        link = links_by_id[extract["source_claim_id"]]
        assert (
            claim["source_id"]
            == link["source_id"]
            == extract["source_id"]
        )
        assert (
            claim["source_url"]
            == link["source_url"]
            == extract["source_url"]
        )
        assert (
            claim["locator"] == link["locator"] == extract["locator"]
        )
        assert (
            link["quoted_or_paraphrased_basis"]
            == extract["bounded_local_basis"]
        )
        assert extract["basis_sha256"] == hashlib.sha256(
            extract["bounded_local_basis"].encode("utf-8")
        ).hexdigest()
    claim_36 = claims_by_id["L1-CLM18-0036"]
    assert "Fab 1 cessation" in claim_36["claim_text"]
    assert "do not establish capacity retirement or repurposing" in (
        claim_36["claim_text"]
    )
    assert (
        links_by_id["L1-CLM18-0036"]["support_status"]
        == "supports_corrected_cessation_scope"
    )
    return "58 claim/evidence/extract identities remain one-to-one; claim 0036 is cessation-bounded"


def validate_complete_junction() -> str:
    retained: set[tuple[str, str]] = set()
    principal: dict[tuple[str, str], str] = {}
    for artifact, key_field in ARTIFACT_KEYS.items():
        _, rows = read_csv(HERE / artifact)
        assert len(rows) == EXPECTED_RECORD_COUNTS[artifact]
        for row in rows:
            key = (artifact, row[key_field])
            retained.add(key)
            principal[key] = row["source_claim_id"]
    assert len(retained) == 128

    _, claims = read_csv(HERE / "proposed_claims.csv")
    claim_ids = {row["claim_id"] for row in claims}
    _, junction = read_csv(HERE / "record_claim_junction.csv")
    assert len(junction) == 214
    assert len({row["junction_id"] for row in junction}) == 214
    assert {
        (row["artifact_name"], row["record_id"]) for row in junction
    } == retained
    assert all(row["atomic_claim_id"] in claim_ids for row in junction)
    for key, principal_claim in principal.items():
        links = [
            row
            for row in junction
            if (row["artifact_name"], row["record_id"]) == key
        ]
        assert any(
            row["atomic_claim_id"] == principal_claim
            and row["support_role"] == "principal_atomic_claim"
            for row in links
        ), key

    _, resolution = read_csv(HERE / "record_scope_resolution.csv")
    resolution_pairs = {
        (row["artifact_name"], row["record_id"]) for row in resolution
    }
    assert retained <= resolution_pairs
    assert all(
        row["independent_acceptance_state"]
        == "PENDING_FRESH_NONAUTHOR_REVIEW"
        for row in resolution
    )
    _, support = read_csv(HERE / "row_claim_support.csv")
    assert len(support) == 86
    return "128 retained records resolve through 214 valid links and 86 compatibility links"


def validate_semantic_audits() -> str:
    expected = [
        ("chronology_semantic_audit.csv", 20, "v4_author_status"),
        ("claimant_semantic_audit.csv", 13, "v4_author_status"),
        ("mechanism_semantic_audit.csv", 16, "v4_author_status"),
        ("junction_semantic_audit.csv", 128, "v4_author_semantic_status"),
    ]
    for name, count, field in expected:
        _, rows = read_csv(HERE / name)
        assert len(rows) == count
        assert all(row[field].startswith("PASS_") for row in rows), name
    _, junction = read_csv(HERE / "junction_semantic_audit.csv")
    assert all(
        row["v4_author_structural_status"] == "PASS"
        and row["fresh_nonauthor_acceptance"] == "PENDING"
        for row in junction
    )
    assert sum(int(row["link_count"]) for row in junction) == 214
    return "chronology, claimant, mechanism, and 128-record junction audits all author-pass"


def validate_v5_two_repair_propagation() -> str:
    _, matrix_rows = read_csv(HERE / "ai_similarity_non_similarity_matrix.csv")
    matrix = find(matrix_rows, "matrix_id", "L1-MTX18-0009")
    assert matrix["episode_id"] == "L1-AN18-0005"
    assert matrix["source_claim_id"] == "L1-CLM18-0022"
    assert matrix["evidence_grade"] == "C"

    _, claims = read_csv(HERE / "proposed_claims.csv")
    claims_by_id = {row["claim_id"]: row for row in claims}
    assert claims_by_id["L1-CLM18-0022"]["episode_id"] == "L1-AN18-0005"
    assert claims_by_id["L1-CLM18-0026"]["episode_id"] == "L1-AN18-0006"
    assert claims_by_id["L1-CLM18-0026"]["evidence_grade"] == "C"

    _, junction = read_csv(HERE / "record_claim_junction.csv")
    links = [
        row
        for row in junction
        if row["artifact_name"] == "ai_similarity_non_similarity_matrix.csv"
        and row["record_id"] == "L1-MTX18-0009"
    ]
    assert len(links) == 4
    principal = [
        row for row in links if row["support_role"] == "principal_atomic_claim"
    ]
    assert len(principal) == 1
    assert principal[0]["atomic_claim_id"] == "L1-CLM18-0022"
    cross_episode = [
        row
        for row in links
        if row["support_role"] == "cross_episode_grade_c_support"
    ]
    assert len(cross_episode) == 1
    assert cross_episode[0]["atomic_claim_id"] == "L1-CLM18-0026"
    assert "L1-AN18-0006" in cross_episode[0]["supported_scope"]
    assert all(row["fresh_review_patch_id"] == "V4R-XP-0002" for row in links)

    _, compatibility = read_csv(HERE / "row_claim_support.csv")
    support = [
        row
        for row in compatibility
        if row["artifact_name"] == "ai_similarity_non_similarity_matrix.csv"
        and row["row_id"] == "L1-MTX18-0009"
    ]
    assert len(support) == 3
    assert {row["principal_claim_id"] for row in support} == {
        "L1-CLM18-0022"
    }
    cross_projection = [
        row
        for row in support
        if row["support_role"] == "cross_episode_grade_c_support"
    ]
    assert len(cross_projection) == 1
    assert cross_projection[0]["supporting_claim_id"] == "L1-CLM18-0026"
    assert "L1-AN18-0006" in cross_projection[0]["scope_note"]

    _, resolutions = read_csv(HERE / "record_scope_resolution.csv")
    resolution = find(resolutions, "record_id", "L1-MTX18-0009")
    assert resolution["atomic_claim_ids"] == (
        "L1-CLM18-0022|L1-CLM18-0026|L1-CLM18-0020|L1-CLM18-0023"
    )
    assert "same-episode principal" in resolution["scope_action"]
    assert "L1-AN18-0006" in resolution["scope_action"]
    assert resolution["fresh_review_patch_id"] == "V4R-XP-0002"

    _, semantic_rows = read_csv(HERE / "junction_semantic_audit.csv")
    semantic = find(semantic_rows, "record_id", "L1-MTX18-0009")
    assert semantic["principal_claim_id"] == "L1-CLM18-0022"
    assert semantic["linked_claim_ids"] == (
        "L1-CLM18-0022|L1-CLM18-0026|L1-CLM18-0020|L1-CLM18-0023"
    )
    assert semantic["fresh_review_patch_id"] == "V4R-XP-0002"
    assert "cross-episode Grade-C support" in semantic["semantic_basis"]

    _, dispositions = read_csv(HERE / "v5_exact_patch_disposition.csv")
    assert len(dispositions) == 2
    assert {row["exact_patch_id"] for row in dispositions} == {
        "V4R-XP-0001",
        "V4R-XP-0002",
    }
    assert all(
        row["author_disposition"]
        == "APPLIED_PENDING_FRESH_NONAUTHOR_REVIEW"
        for row in dispositions
    )
    return (
        "Micron 2002 identity and same-episode solar principal with explicit "
        "Grade-C cross-episode support propagate through every required surface"
    )


def repair_0001() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0003"]
    links = linked_claims()
    assert row["claimant_or_holder"] == (
        "Global Crossing creditors, old equity, and new investor"
    )
    assert "not reconstructed" in row["cash_loss_state"]
    assert "no cash distribution is asserted" in row["restructuring_effect"]
    assert row["physical_asset_state"] == "Network continued"
    assert claim_set(
        links,
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0048"}
    return "Global Crossing cash distribution is unasserted and the row is claim-0048 bounded"


def repair_0002() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0004"]
    links = linked_claims()
    assert row["claimant_or_holder"] == "WorldCom debt and equity holders"
    assert row["cash_loss_state"] == (
        "Old equity eliminated; creditor recovery not reconstructed"
    )
    assert "plan distributions and later emergence are not asserted" in (
        row["restructuring_effect"]
    )
    assert row["physical_asset_state"] == "Network continued"
    assert claim_set(
        links,
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0049"}
    return "WorldCom claimant and recovery scope is limited to claim 0049"


def repair_0003() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0005"]
    assert row["physical_asset_state"] == (
        "not_established_by_linked_atomic_claim"
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0050"}
    return "chronology 0005 makes no unsupported physical-continuity claim"


def repair_0004() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0007"]
    assert row["physical_asset_state"] == (
        "not_established_by_plan_disclosure_claim"
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0052"}
    return "chronology 0007 labels physical state unestablished"


def repair_0005() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0008"]
    assert "21,499 shares were exercised" in row["cash_loss_state"]
    assert "48.5 million" in row["restructuring_effect"]
    assert "Generation fleet largely continued" in row["physical_asset_state"]
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {
        "L1-CLM18-0053",
        "L1-CLM18-0018",
        "L1-CLM18-0051",
    }
    return "Calpine warrant-exercise and fleet-continuity propositions have bounded support"


def repair_0006() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0011"]
    assert row["physical_asset_state"] == (
        "not_established_by_linked_atomic_claim"
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0023"}
    return "chronology 0011 makes no unsupported physical-asset assertion"


def repair_0007() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0013"]
    assert row["claimant_or_holder"] == "not_identified_in_source"
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0054", "L1-CLM18-0025"}
    return "solar chronology leaves claimant identity unresolved"


def repair_0008() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0014"]
    assert row["claimant_or_holder"] == "not_identified_in_source"
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0055", "L1-CLM18-0028"}
    return "Hanjin chronology leaves claimant identity unresolved"


def repair_0009() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0016"]
    assert "prior input-cash timing is not reconstructed" in (
        row["cash_loss_state"]
    )
    assert row["restructuring_effect"] == (
        "not_established_by_linked_atomic_claim"
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0056"}
    return "Micron chronology does not invent cash timing or restructuring"


def repair_0010() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0017"]
    assert row["restructuring_effect"] == (
        "not_established_by_linked_atomic_claim"
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0057"}
    return "Micron 2002 chronology does not invent restructuring"


def repair_0011() -> str:
    rows = rows_by_id(
        HERE / "impairment_default_restructuring_chronology.csv",
        "chronology_id",
    )
    row = rows["L1-CHR18-0018"]
    assert "other claimants not_identified_in_source" in (
        row["claimant_or_holder"]
    )
    assert row["restructuring_effect"] == (
        "Fab 1 ceased operation; retirement or repurposing beyond cessation "
        "is not inferred"
    )
    assert "subsequent disposition are not reconstructed" in (
        row["physical_asset_state"]
    )
    assert claim_set(
        linked_claims(),
        "impairment_default_restructuring_chronology.csv",
        row["chronology_id"],
    ) == {"L1-CLM18-0058"}
    return "Chartered chronology stops at Fab 1 cessation"


def repair_0012() -> str:
    rows = rows_by_id(HERE / "claimant_loss_boundary.csv", "claimant_loss_id")
    row = rows["L1-CLS18-0001"]
    assert row["observed_claimant"] == (
        "Federal government as authorized funding party; other claimant "
        "classes not_reconstructed"
    )
    assert claim_set(
        linked_claims(), "claimant_loss_boundary.csv", row["claimant_loss_id"]
    ) == {"L1-CLM18-0002", "L1-CLM18-0004"}
    return "Conrail claimant field names only the supported funding party"


def repair_0013() -> str:
    rows = rows_by_id(HERE / "claimant_loss_boundary.csv", "claimant_loss_id")
    row = rows["L1-CLS18-0003"]
    assert row["observed_claimant"] == "WorldCom old equity and creditors"
    assert "lease" not in row["observed_claimant"].lower()
    assert claim_set(
        linked_claims(), "claimant_loss_boundary.csv", row["claimant_loss_id"]
    ) == {"L1-CLM18-0009", "L1-CLM18-0049"}
    return "WorldCom lease counterparties are removed from observed claimants"


def repair_0014() -> str:
    rows = rows_by_id(HERE / "claimant_loss_boundary.csv", "claimant_loss_id")
    issuer = rows["L1-CLS18-0009"]
    sector = rows["L1-CLS18-0013"]
    links = linked_claims()
    assert issuer["observed_claimant"] == "not_identified_in_source"
    assert issuer["source_claim_id"] == "L1-CLM18-0055"
    assert claim_set(
        links, "claimant_loss_boundary.csv", issuer["claimant_loss_id"]
    ) == {"L1-CLM18-0055", "L1-CLM18-0028"}
    assert sector["source_claim_id"] == "L1-CLM18-0027"
    assert claim_set(
        links, "claimant_loss_boundary.csv", sector["claimant_loss_id"]
    ) == {"L1-CLM18-0027"}
    _, matrix = read_csv(HERE / "ai_similarity_non_similarity_matrix.csv")
    assert find(matrix, "matrix_id", "L1-MTX18-0013")[
        "source_claim_id"
    ] == "L1-CLM18-0027"
    assert find(matrix, "matrix_id", "L1-MTX18-0014")[
        "source_claim_id"
    ] == "L1-CLM18-0028"
    return "Hanjin issuer and sector populations remain strictly separated"


def repair_0015() -> str:
    rows = rows_by_id(
        HERE / "survivor_recovery_mechanisms.csv", "mechanism_id"
    )
    row = rows["L1-MEC18-0012"]
    assert row["capital_action"] == (
        "funding_state_and_avoided_outlay_unresolved"
    )
    assert row["primary_beneficiary"] == "not_identified_in_source"
    assert row["primary_loser_or_cost_bearer"] == (
        "not_identified_in_source"
    )
    assert "project cancellations" in row["observed_outcome"]
    assert "funding, avoided outlay, beneficiaries, and loss bearers" in (
        row["transfer_warning"]
    )
    assert claim_set(
        linked_claims(), "survivor_recovery_mechanisms.csv", row["mechanism_id"]
    ) == {"L1-CLM18-0025"}
    return "solar mechanism retains cancellation only and leaves attribution unresolved"


def repair_0016() -> str:
    rows = rows_by_id(
        HERE / "survivor_recovery_mechanisms.csv", "mechanism_id"
    )
    row = rows["L1-MEC18-0016"]
    assert row["operational_action"] == "Fab 1 ceased operation in March 2004"
    assert row["capital_action"] == (
        "fixed_cost_and_capital_effect_not_reconstructed"
    )
    assert row["primary_beneficiary"] == "not_identified_in_source"
    assert row["primary_loser_or_cost_bearer"] == (
        "not_identified_in_source"
    )
    assert "retirement or repurposing beyond cessation is not inferred" in (
        row["observed_outcome"]
    )
    assert claim_set(
        linked_claims(), "survivor_recovery_mechanisms.csv", row["mechanism_id"]
    ) == {"L1-CLM18-0036", "L1-CLM18-0058"}
    return "Chartered mechanism is cessation-only with strict claim-0058 support"


def repair_0017() -> str:
    rows = rows_by_id(
        HERE / "survivor_recovery_mechanisms.csv", "mechanism_id"
    )
    row = rows["L1-MEC18-0018"]
    assert row["primary_beneficiary"] == "NVIDIA reporting entity only"
    assert row["primary_loser_or_cost_bearer"] == (
        "not_identified_in_source"
    )
    assert "NVIDIA operating cash flow benefited" in row["capital_action"]
    assert (
        "channel beneficiaries, unit disposition, and upstream loss "
        "allocation are unresolved"
    ) in row["observed_outcome"]
    assert claim_set(
        linked_claims(), "survivor_recovery_mechanisms.csv", row["mechanism_id"]
    ) == {"L1-CLM18-0039"}
    return "NVIDIA mechanism stops at reporting-entity working-capital benefit"


def repair_0018() -> str:
    _, fresh = read_csv(HERE / "fresh_exact_repair_disposition.csv")
    assert len(fresh) == 18
    assert {row["exact_patch_id"] for row in fresh} == {
        f"FAR18-XP-{number:04d}" for number in range(1, 19)
    }
    assert all(
        row["author_disposition"]
        == "APPLIED_PENDING_FRESH_NONAUTHOR_REVIEW"
        and row["author_validation_state"] == "PASS"
        and row["descriptive_use"] == "BLOCKED_PENDING_FRESH_REVIEW"
        and row["canonical_integration"] == "NOT_AUTHORIZED"
        and row["coefficient_transfer"] == "BLOCK"
        and row["predictive_use"] == "BLOCK"
        for row in fresh
    )
    _, disposition = read_csv(HERE / "exact_patch_disposition.csv")
    assert len(disposition) == 124
    assert len({row["exact_patch_id"] for row in disposition}) == 124
    assert all(
        row["independent_acceptance_state"]
        == "PENDING_FRESH_NONAUTHOR_REVIEW"
        for row in disposition
    )
    _, gates = read_csv(HERE / "writing_gate_audit.csv")
    assert len(gates) == 20
    assert all(
        "fresh" in row["status"].lower()
        or row["status"] == "PASS_STAGING_ONLY_NO_CHAPTER_PROSE"
        for row in gates
    )
    _, v5 = read_csv(HERE / "v5_exact_patch_disposition.csv")
    assert len(v5) == 2
    assert all(
        row["canonical_integration"] == "NOT_AUTHORIZED"
        and row["coefficient_predictive_use"] == "BLOCK"
        and row["comparative_publication"] == "BLOCKED"
        for row in v5
    )
    return (
        "all 18 prior repairs and both v5 repairs are accounted while canonical, "
        "coefficient, predictive, and publication boundaries remain sealed"
    )


def validate_no_quantitative_transfer() -> str:
    _, prohibitions = read_csv(HERE / "coefficient_transfer_prohibitions.csv")
    assert len(prohibitions) == 18
    assert all(
        row["prohibited_coefficient_or_mapping"]
        and row["minimum_matching_evidence_before_reconsideration"]
        for row in prohibitions
    )
    _, hypotheses = read_csv(HERE / "hypothesis_tests.csv")
    assert len(hypotheses) == 16
    assert all(
        row["predictive_use"] == "BLOCK"
        and row["protocol_state"] == "not_frozen"
        for row in hypotheses
    )
    return "averages, rates, scores, coefficients, probabilities, and monitoring transfer remain blocked"


def validate_canonical_no_write() -> str:
    audit = json.loads(
        (HERE / "canonical_no_write_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "PASS"
    assert audit["modified"] is False
    assert audit["sha256_before"] == audit["sha256_after"]
    assert audit["canonical_integration"] == "NOT_AUTHORIZED"
    return "canonical database hash is unchanged"


def report_text(checks: list[dict[str, str]], passed: bool) -> str:
    lines = [
        "# Chapter 18 rebuilt-v5 author validation",
        "",
        f"**Author verdict:** `{VERDICT}`",
        "",
        f"**Meaning validation:** `{'PASS' if passed else 'FAIL'}`",
        "",
        "**Author validation is acceptance:** `NO`",
        "",
        "**Fresh nonauthor review:** `REQUIRED`",
        "",
        "**Canonical integration:** `NOT_AUTHORIZED`",
        "",
        "**Bounded descriptive use:** `PENDING_FRESH_NONAUTHOR_REVIEW`",
        "",
        "**Comparative publication:** `BLOCKED_PENDING_FRESH_REVIEW_AND_EXTERNAL_HOLDS`",
        "",
        "**Coefficient transfer and predictive use:** `BLOCKED`",
        "",
        "**External source-content snapshots:** `58 HOLDS`",
        "",
        "This is a deterministic author-side semantic re-performance. It does "
        "not constitute independent acceptance and does not authorize "
        "quantitative transfer from historical analogues into current AI "
        "probabilities or timing rules.",
        "",
        "## Results",
        "",
    ]
    for row in checks:
        lines.append(
            f"- `{row['check_id']}` — **{row['status']}**: {row['detail']}"
        )
    lines.extend(
        [
            "",
            "## Release boundary",
            "",
            "The packet may proceed only to fresh nonauthor descriptive review. "
            "Canonical integration, comparative publication, coefficient "
            "transfer, predictive use, calibrated monitoring, and historical-"
            "analogue probability/timing transfer remain blocked.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    runner = CheckRunner()
    runner.check("V5-IMMUTABLE-INPUTS", validate_immutable_inputs)
    runner.check("V5-PRESERVED-CONTROLS", validate_preserved_controls)
    runner.check("V5-CLAIM-EXTRACT-POPULATION", validate_claim_extract_population)
    runner.check("V5-COMPLETE-JUNCTION", validate_complete_junction)
    runner.check("V5-SEMANTIC-AUDITS", validate_semantic_audits)
    repairs: list[tuple[str, Callable[[], str]]] = [
        ("FAR18-XP-0001", repair_0001),
        ("FAR18-XP-0002", repair_0002),
        ("FAR18-XP-0003", repair_0003),
        ("FAR18-XP-0004", repair_0004),
        ("FAR18-XP-0005", repair_0005),
        ("FAR18-XP-0006", repair_0006),
        ("FAR18-XP-0007", repair_0007),
        ("FAR18-XP-0008", repair_0008),
        ("FAR18-XP-0009", repair_0009),
        ("FAR18-XP-0010", repair_0010),
        ("FAR18-XP-0011", repair_0011),
        ("FAR18-XP-0012", repair_0012),
        ("FAR18-XP-0013", repair_0013),
        ("FAR18-XP-0014", repair_0014),
        ("FAR18-XP-0015", repair_0015),
        ("FAR18-XP-0016", repair_0016),
        ("FAR18-XP-0017", repair_0017),
        ("FAR18-XP-0018", repair_0018),
    ]
    for check_id, fn in repairs:
        runner.check(check_id, fn)
    runner.check(
        "V5-TWO-REPAIR-PROPAGATION", validate_v5_two_repair_propagation
    )
    runner.check("V5-NO-QUANTITATIVE-TRANSFER", validate_no_quantitative_transfer)
    runner.check("V5-CANONICAL-NO-WRITE", validate_canonical_no_write)

    passed = all(row["status"] == "PASS" for row in runner.rows)
    payload = {
        "passed": passed,
        "status": VERDICT if passed else "AUTHOR_VALIDATION_FAILED",
        "author_validation_is_acceptance": False,
        "fresh_nonauthor_review": "REQUIRED",
        "canonical_integration": "NOT_AUTHORIZED",
        "intrinsic_verdict": "READY_FOR_FRESH_NONAUTHOR_REVIEW",
        "descriptive_verdict": "PENDING_FRESH_NONAUTHOR_REVIEW",
        "comparative_publication": "BLOCKED_PENDING_FRESH_REVIEW_AND_EXTERNAL_HOLDS",
        "coefficient_transfer": "BLOCK",
        "predictive_use": "BLOCK",
        "checks_passed": sum(row["status"] == "PASS" for row in runner.rows),
        "checks_failed": sum(row["status"] == "FAIL" for row in runner.rows),
        "checks": runner.rows,
    }
    write_json(HERE / "validation_results.json", payload)
    write_text(HERE / "validation_report.md", report_text(runner.rows, passed))
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
