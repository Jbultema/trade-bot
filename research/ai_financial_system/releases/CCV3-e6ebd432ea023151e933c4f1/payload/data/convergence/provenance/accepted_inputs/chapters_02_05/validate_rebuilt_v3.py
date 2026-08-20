#!/usr/bin/env python3
"""Dedicated validation for the Chapters 2-5 metadata-only v3 rebase."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AI = REPO / "research/ai_financial_system"
STAGING = AI / "staging"
V2 = STAGING / "lane_3_chapters_02_05_rebuilt_v2"
UNION = STAGING / "lane_3_source_preservation_union_v2"
DOWNSTREAM = {
    "CH16": STAGING / "lane_2_verify_chapter_16_rebuilt_v3",
    "CH17": STAGING / "lane_1_verify_chapter_17_atomic_rebase",
    "CH18": STAGING / "lane_2_verify_chapter_18_rebuilt_v5",
    "CH19": STAGING / "lane_2_verify_chapter_19_rebuilt_v3",
}
SNAPSHOT = "SPU-V2-20260726-38b990b2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"bad CSV header: {path}")
        result = list(reader)
        if any(None in row for row in result):
            raise ValueError(f"malformed CSV row: {path}")
        return result


def row_signature(row: dict[str, str], excluded: set[str]) -> str:
    retained = {key: value for key, value in row.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(retained, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    checks = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    input_rows = rows(HERE / "sealed_input_inventory.csv")
    input_ok = all(
        (REPO / row["path"]).is_file()
        and digest(REPO / row["path"]) == row["sha256"]
        and (REPO / row["path"]).stat().st_size == int(row["byte_count"])
        for row in input_rows
    )
    check("governing_inputs_hash_bound", input_ok, len(input_rows), "all current")

    inventory = rows(HERE / "intrinsic_artifact_inventory.csv")
    intrinsic_ok = len(inventory) == 50 and all(
        row["v2_sha256"] == row["v3_sha256"]
        and digest(HERE / row["relative_path"]) == row["v3_sha256"]
        and row["treatment"] == "byte_identical_accepted_intrinsic"
        for row in inventory
    )
    check("accepted_intrinsic_artifacts_byte_identical", intrinsic_ok, len(inventory), 50)
    html_ok = all(
        digest(HERE / f"accepted_intrinsic_v2/chapter_{chapter:02d}/proposed_chapter.html")
        == digest(V2 / f"chapter_{chapter:02d}/proposed_chapter.html")
        for chapter in range(2, 6)
    )
    check("chapter_html_factual_semantics_byte_identical", html_ok, html_ok, True)

    claims = rows(HERE / "intrinsic_complete_claim_source_audit.csv")
    links = rows(HERE / "intrinsic_claim_source_support_junction.csv")
    locators = rows(HERE / "intrinsic_claim_source_locator_audit.csv")
    claim_projection = rows(
        HERE / "accepted_intrinsic_v2/chapter_html_projection_audit.csv"
    )
    arithmetic = rows(
        HERE / "accepted_intrinsic_v2/arithmetic_nonadditivity_audit.csv"
    )
    population_ok = (
        len(claims) == 278
        and len(links) == 396
        and len(locators) == 278
        and len(claim_projection) == 341
        and len({row["claim_id"] for row in claim_projection}) == 294
        and len(rows(HERE / "accepted_intrinsic_v2/exact_patch_disposition.csv")) == 9
        and sum(row["audit_type"] == "arithmetic_reperformance" for row in arithmetic)
        == 18
        and sum(row["audit_type"] != "arithmetic_reperformance" for row in arithmetic)
        == 4
        and len(rows(HERE / "accepted_intrinsic_v2/unknown_population_audit.csv"))
        == 126
        and len(rows(HERE / "accepted_intrinsic_v2/falsifier_audit.csv")) == 20
        and len(rows(HERE / "accepted_intrinsic_v2/handoff_blocked_field_audit.csv"))
        == 18
    )
    check(
        "frozen_intrinsic_populations",
        population_ok,
        "278|396|278|341|294|9|18|4|126|20|18",
        "exact task populations",
    )

    v2_links = rows(V2 / "claim_source_support_junction.csv")
    link_by = {row["support_junction_id"]: row for row in links}
    link_exact = len(link_by) == len(v2_links) and all(
        row_signature(
            row,
            {"source_preservation_state", "author_validation_state"},
        )
        == row_signature(
            link_by.get(row["support_junction_id"], {}),
            {"intrinsic_review_state", "canonical_integration_status"},
        )
        for row in v2_links
    )
    check("396_intrinsic_links_lossless", link_exact, len(link_by), 396)

    v2_claims = rows(V2 / "complete_claim_source_audit.csv")
    claim_by = {row["complete_audit_id"]: row for row in claims}
    claim_exact = len(claim_by) == len(v2_claims) and all(
        row_signature(
            row,
            {"external_source_preservation", "author_semantic_state"},
        )
        == row_signature(
            claim_by.get(row["complete_audit_id"], {}),
            {"intrinsic_review_state", "canonical_integration_status"},
        )
        for row in v2_claims
    )
    check("278_claim_rows_lossless_except_metadata", claim_exact, len(claim_by), 278)

    expected_projection = [
        row
        for row in rows(UNION / "chapter_02_09_support_state_projection.csv")
        if row["chapter_id"] in {"CH02", "CH03", "CH04", "CH05"}
    ]
    projection = rows(HERE / "normalized_preservation_projection_references.csv")
    projection_ok = projection == expected_projection
    counts = Counter(row["source_state_disposition"] for row in projection)
    chapter_counts = Counter(row["chapter_id"] for row in projection)
    check(
        "361_exact_union_projection_references",
        projection_ok
        and len(projection) == 361
        and dict(chapter_counts) == {"CH02": 243, "CH03": 40, "CH04": 47, "CH05": 31}
        and dict(counts)
        == {
            "preserved_bytes_available_locator_review_pending": 172,
            "no_matching_preservation_record": 186,
            "internal_P4_reconstruction": 3,
        }
        and all(row["snapshot_id"] == SNAPSHOT for row in projection),
        {
            "rows": len(projection),
            "chapters": dict(chapter_counts),
            "states": dict(counts),
        },
        "361 exact rows and exact counts",
    )
    required_fields = {
        "snapshot_id",
        "projection_row_id",
        "claim_id",
        "source_id",
        "source_version_id",
        "source_state_disposition",
        "semantic_review_authority",
        "candidate_integration_status",
    }
    check(
        "projection_reference_fields_complete",
        required_fields <= set(projection[0])
        and all(
            row["semantic_review_authority"]
            == "not_semantically_cleared_by_union_candidate"
            for row in projection
        ),
        sorted(required_fields),
        "all required fields; no semantic clearance",
    )

    stale_headers = {
        "source_preservation_state",
        "external_source_preservation",
        "external_source_persistence",
        "immutable_local_snapshot",
        "all_source_urls_canonical_or_archived",
    }
    current_csvs = [
        path
        for path in HERE.glob("*.csv")
        if path.name
        not in {
            "superseded_metadata_audit.csv",
            "sealed_input_inventory.csv",
            "intrinsic_artifact_inventory.csv",
            "no_write_audit.csv",
        }
    ]
    stale_current = []
    for path in current_csvs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            fields = set(next(csv.reader(handle)))
        if fields & stale_headers:
            stale_current.append(f"{path.name}:{sorted(fields & stale_headers)}")
    check("zero_copied_mutable_preservation_fields", not stale_current, stale_current, [])
    superseded = rows(HERE / "superseded_metadata_audit.csv")
    check(
        "historical_metadata_explicitly_superseded",
        len(superseded) == 5
        and all(row["historical_rows_copied_as_current"] == "no" for row in superseded),
        len(superseded),
        5,
    )

    src0094 = rows(HERE / "src_0094_internal_classification_audit.csv")
    check(
        "src_0094_is_internal_p4",
        len(src0094) == 3
        and all(
            row["result"] == "PASS_INTERNAL_P4"
            and row["external_source_hold"] == "no"
            and row["source_state_disposition"] == "internal_P4_reconstruction"
            for row in src0094
        ),
        len(src0094),
        "3 internal P4 rows",
    )

    dependencies = rows(HERE / "downstream_dependency_audit.csv")
    dependency_by = {row["downstream_chapter_id"]: row for row in dependencies}
    dependency_ok = (
        set(dependency_by) == set(DOWNSTREAM)
        and dependency_by["CH16"]["intrinsic_state"] == "ACCEPT_INTRINSIC_PACKET"
        and dependency_by["CH17"]["intrinsic_state"] == "ACCEPT_INTRINSIC_PACKET"
        and dependency_by["CH18"]["intrinsic_state"]
        == "ACCEPT_INTRINSIC_WITH_EXPLICIT_EXTERNAL_HOLDS"
        and "58 external source-content holds"
        in dependency_by["CH18"]["holds"]
        and dependency_by["CH19"]["intrinsic_state"]
        == "SAFE_ONLY_AS_UNRESOLVED_ARCHITECTURE"
        and all(
            row["canonical_integration"] == "NOT_AUTHORIZED"
            and row["publication"] == "BLOCKED"
            and row["handoff_readiness_change"] == "none"
            and row["review_manifest_sha256"]
            == digest(DOWNSTREAM[row["downstream_chapter_id"]] / "lane_manifest.json")
            for row in dependencies
        )
    )
    check("downstream_acceptance_states_bounded", dependency_ok, len(dependencies), 4)

    handoffs = rows(HERE / "accepted_intrinsic_v2/handoff_blocked_field_audit.csv")
    check(
        "all_18_handoffs_remain_blocked",
        all(
            row["readiness_status"] == "blocked"
            and row["downstream_execution_authorized"] == "no"
            for row in handoffs
        ),
        len(handoffs),
        "18 blocked and unauthorized",
    )
    authority = rows(HERE / "authority_separation_audit.csv")
    holds = rows(HERE / "current_governance_holds.csv")
    check(
        "source_semantic_canonical_publication_separated",
        len(authority) == 4
        and any(
            row["layer"] == "canonical_integration"
            and row["current_state"] == "NOT_AUTHORIZED"
            for row in authority
        )
        and any(
            row["layer"] == "publication" and row["current_state"] == "BLOCKED"
            for row in authority
        )
        and len(holds) == 5
        and all(row["status"] == "OPEN" for row in holds),
        f"authority={len(authority)} holds={len(holds)}",
        "4 layers; 5 open holds",
    )

    no_write = rows(HERE / "no_write_audit.csv")
    check(
        "protected_inputs_unchanged",
        all(row["result"] == "PASS_UNCHANGED" for row in no_write),
        f"{sum(row['result']=='PASS_UNCHANGED' for row in no_write)}/{len(no_write)}",
        "all unchanged",
    )
    failures = [row for row in checks if not row["passed"]]
    result = {
        "validator": "chapters_02_05_rebuilt_v3_dedicated_validator",
        "passed": not failures,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failures": failures,
        "author_verdict": (
            "READY_FOR_FRESH_NONAUTHOR_METADATA_REBASE_REVIEW"
            if not failures
            else "REVISE_METADATA_REBASE"
        ),
        "canonical_integration": "NOT_AUTHORIZED",
        "docx_replacement": "NOT_AUTHORIZED",
        "publication": "BLOCKED",
    }
    (HERE / "dedicated_validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (HERE / "dedicated_validation_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check", "passed", "observed", "expected"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in checks:
            writer.writerow(
                {
                    **row,
                    "observed": json.dumps(row["observed"], sort_keys=True),
                    "expected": json.dumps(row["expected"], sort_keys=True),
                }
            )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
