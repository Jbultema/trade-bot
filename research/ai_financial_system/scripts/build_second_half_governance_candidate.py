from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING = PROJECT_ROOT / "staging"
DEFAULT_OUTPUT = PROJECT_ROOT / "integration" / "second_half_governance_candidate"

UNKNOWN_COLUMNS = [
    "unknown_id",
    "chapter_id",
    "question",
    "materiality",
    "unknown_class",
    "downstream_blocking_chapters",
    "resolution_trigger",
    "suspected_sources",
    "resolution_plan",
    "status",
    "last_reviewed",
]

HANDOFF_COLUMNS = [
    "handoff_id",
    "upstream_chapter_id",
    "downstream_chapter_id",
    "dataset_or_ledger",
    "analytical_input",
    "required_fields",
    "readiness_status",
    "missing_evidence",
    "source_claim_id",
    "as_of_date",
    "notes",
]

CH15_UNKNOWN_GOVERNANCE = {
    "UNK-150001": (
        "critical",
        "contractual_opacity",
        "CH06;CH15;CH16",
    ),
    "UNK-150002": (
        "critical",
        "contractual_opacity",
        "CH06;CH15;CH16;CH17",
    ),
    "UNK-150003": (
        "critical",
        "disclosure_gap",
        "CH06;CH15;CH16",
    ),
    "UNK-150004": (
        "high",
        "contractual_opacity",
        "CH04;CH13;CH15;CH16",
    ),
    "UNK-150005": (
        "critical",
        "contractual_opacity",
        "CH09;CH15;CH17",
    ),
    "UNK-150006": (
        "critical",
        "contractual_opacity",
        "CH04;CH13;CH15;CH17",
    ),
    "UNK-150007": (
        "high",
        "contractual_opacity",
        "CH02;CH08;CH11;CH15;CH16;CH17",
    ),
    "UNK-150008": (
        "high",
        "contractual_opacity",
        "CH06;CH15;CH16;CH17",
    ),
    "UNK-150009": (
        "high",
        "entity_or_transaction_identity_ambiguity",
        "CH07;CH15;CH16",
    ),
    "UNK-150010": (
        "high",
        "contractual_opacity",
        "CH04;CH06;CH15;CH17",
    ),
}

UNKNOWN_DUPLICATES = {
    "PUNK-12-004",
    "PUNK-12-005",
    "PUNK-1301",
    "PUNK-1307",
    "PUNK-1308",
    "PUNK-14001",
    "PUNK-14005",
    "PUNK-14007",
    "PUNK-14008",
}

UNKNOWN_CLASS_MAP = {
    "AI-attribution_gap": "AI_attribution_gap",
    "AI_attribution_gap": "AI_attribution_gap",
    "capex_transmission_gap": "model_uncertainty",
    "commissioning_gap": "measurement_gap",
    "contractual_opacity": "contractual_opacity",
    "disclosure_gap": "disclosure_gap",
    "economic_model_gap": "model_uncertainty",
    "entity_or_transaction_identity_ambiguity": (
        "entity_or_transaction_identity_ambiguity"
    ),
    "life_measurement_gap": "measurement_gap",
    "measurement_gap": "measurement_gap",
    "physical_chain_gap": "measurement_gap",
    "power_measurement_gap": "measurement_gap",
    "price_measurement_gap": "measurement_gap",
    "reporting_perimeter": "disclosure_gap",
    "residual_value_gap": "measurement_gap",
    "state_reconciliation_gap": "overlap_or_double_counting_risk",
}

CH15_HANDOFFS = {
    "HND-150001": (
        "HND-1501",
        "CH06",
        "control_boundary_matrix",
        "subject; right holder; governance right; accounting control; "
        "dependency assessment; evidence state; effective date",
        "ready",
        "Complete Anthropic voting and board-rights structure remains unknown",
    ),
    "HND-150002": (
        "HND-1502",
        "CH04",
        "collateral_and_step_in_rights",
        "case; obligor; collateral; trigger; cure; step-in right; accounting "
        "control boundary",
        "ready",
        "Recovery amount and claimant waterfall remain unknown",
    ),
    "HND-150003": (
        "HND-1503",
        "CH07",
        "control_boundary_matrix",
        "VIE identity; support provider; support amount; primary-beneficiary "
        "status; lease-assumption right",
        "ready",
        "External VIE identity and complete lease-assumption mechanics remain "
        "unknown",
    ),
    "HND-150004": (
        "HND-1504",
        "CH09",
        "collateral_and_step_in_rights",
        "warehouse; asset; trigger; cure; account control; servicer removal; "
        "disposition right",
        "ready",
        "Asset-level advance rates marks and recovery amounts remain unknown",
    ),
    "HND-150005": (
        "HND-1505",
        "CH11",
        "control_and_dependency_overlap_map",
        "entity; observed role; inferred dependency; control dimension; "
        "overlap group; evidence state; date scope",
        "ready",
        "Private governance rights and quantified switching costs remain "
        "unknown",
    ),
    "HND-150006": (
        "HND-1506",
        "CH16",
        "control_taxonomy",
        "ownership; contractual control; accounting control; operational "
        "dependence; trigger; evidence state; effective date",
        "ready",
        "Several private contract and governance schedules remain unknown",
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames is None or any(None in row for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return rows


def write_rows(
    path: Path, columns: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_standard_unknowns(
    lane: str, chapter: int
) -> list[dict[str, str]]:
    rows = read_rows(STAGING / lane / "proposed_unknowns.csv")
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        raw_id = row["provisional_unknown_id"]
        if raw_id in UNKNOWN_DUPLICATES:
            continue
        output.append(
            {
                "unknown_id": f"UNK-{chapter:02d}{index:02d}",
                "chapter_id": row["chapter_id"],
                "question": row["question"],
                "materiality": row["materiality"],
                "unknown_class": UNKNOWN_CLASS_MAP[row["principal_class"]],
                "downstream_blocking_chapters": row[
                    "downstream_blocking_chapters"
                ],
                "resolution_trigger": row["resolution_trigger"],
                "suspected_sources": row["suspected_primary_sources"],
                "resolution_plan": (
                    "Obtain the named primary evidence and preserve the "
                    "unresolved state until the stated trigger is met. "
                    + (
                        row.get("anti_double_counting_notes")
                        or (
                            "Keep the stated perimeter and overlap identity "
                            f"{row.get('overlap_identity', 'unresolved')} "
                            "nonadditive."
                        )
                    )
                ),
                "status": row["status"],
                "last_reviewed": "2026-07-25",
            }
        )
    return output


def build_ch15_unknowns() -> list[dict[str, str]]:
    rows = read_rows(
        STAGING / "lane_2_chapter_15" / "proposed_unknowns.csv"
    )
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        raw_id = row["unknown_id"]
        materiality, unknown_class, blockers = CH15_UNKNOWN_GOVERNANCE[raw_id]
        output.append(
            {
                "unknown_id": f"UNK-15{index:02d}",
                "chapter_id": "CH15",
                "question": row["unknown_description"],
                "materiality": materiality,
                "unknown_class": unknown_class,
                "downstream_blocking_chapters": blockers,
                "resolution_trigger": (
                    f"{row['required_primary_evidence']} become public and "
                    "resolve the named right, perimeter, or remedy."
                ),
                "suspected_sources": row["required_primary_evidence"],
                "resolution_plan": (
                    f"Route to {row['owner_handoff']}; retain unresolved and "
                    "do not infer control, switching cost, recovery, or first "
                    "loss from adjacent roles."
                ),
                "status": "open",
                "last_reviewed": "2026-07-25",
            }
        )
    return output


def build_ch12_handoffs() -> list[dict[str, str]]:
    rows = read_rows(
        STAGING / "lane_1_chapter_12" / "proposed_handoffs.csv"
    )
    readiness = {
        "blocked": "blocked",
        "mechanism_only": "ready",
        "diagnostic_only": "ready",
    }
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        output.append(
            {
                "handoff_id": f"HND-12{index:02d}",
                "upstream_chapter_id": row["upstream_chapter_id"],
                "downstream_chapter_id": row["downstream_chapter_id"],
                "dataset_or_ledger": row["dataset_or_ledger"],
                "analytical_input": row["analytical_input"],
                "required_fields": row["required_fields"],
                "readiness_status": readiness[row["readiness_status"]],
                "missing_evidence": row["blocked_fields"],
                "source_claim_id": principal_claim(row["source_claim_ids"]),
                "as_of_date": row["as_of_date"],
                "notes": (
                    f"Staged status was {row['readiness_status']}. "
                    f"{row['anti_double_counting_notes']} Evidence state "
                    f"{row['evidence_state']}; grade {row['evidence_grade']}; "
                    f"perimeter {row['perimeter']}."
                    + supporting_claim_note(row["source_claim_ids"])
                ),
            }
        )
    return output


def build_standard_handoffs(
    lane: str, chapter: int
) -> list[dict[str, str]]:
    rows = read_rows(STAGING / lane / "proposed_handoffs.csv")
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        output.append(
            {
                "handoff_id": f"HND-{chapter:02d}{index:02d}",
                "upstream_chapter_id": row["upstream_chapter_id"],
                "downstream_chapter_id": row["downstream_chapter_id"],
                "dataset_or_ledger": row["dataset_or_ledger"],
                "analytical_input": row["analytical_input"],
                "required_fields": row["required_fields"],
                "readiness_status": row["readiness_status"],
                "missing_evidence": row["missing_evidence"],
                "source_claim_id": principal_claim(row["source_claim_id"]),
                "as_of_date": row["as_of_date"],
                "notes": (
                    row["notes"]
                    + f" Evidence state {row['evidence_state']}; grade "
                    f"{row['evidence_grade']}; overlap "
                    f"{row.get('overlap_identity', 'unresolved')}."
                    + supporting_claim_note(row["source_claim_id"])
                ),
            }
        )
    return output


def build_ch15_handoffs() -> list[dict[str, str]]:
    rows = read_rows(
        STAGING / "lane_2_chapter_15" / "proposed_handoffs.csv"
    )
    output: list[dict[str, str]] = []
    for row in rows:
        (
            handoff_id,
            downstream,
            dataset,
            required_fields,
            readiness,
            missing,
        ) = CH15_HANDOFFS[row["handoff_id"]]
        output.append(
            {
                "handoff_id": handoff_id,
                "upstream_chapter_id": "CH15",
                "downstream_chapter_id": downstream,
                "dataset_or_ledger": dataset,
                "analytical_input": row["deliverable"],
                "required_fields": required_fields,
                "readiness_status": readiness,
                "missing_evidence": missing,
                "source_claim_id": principal_claim(row["source_claim_id"]),
                "as_of_date": "2026-07-25",
                "notes": (
                    f"Staged dependency status was "
                    f"{row['dependency_status']}. {row['notes']} Ready means "
                    "the evidence-limited qualitative artifact is usable; it "
                    "does not "
                    "authorize an inferred control, probability, or loss."
                    + supporting_claim_note(row["source_claim_id"])
                ),
            }
        )
    return output


def ensure_unique(rows: list[dict[str, str]], key: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {key} in second-half governance candidate")


def claim_set(raw: str) -> list[str]:
    return [token for token in re.split(r"[;|]", raw) if token]


def principal_claim(raw: str) -> str:
    claims = claim_set(raw)
    return claims[0] if claims else ""


def supporting_claim_note(raw: str) -> str:
    claims = claim_set(raw)
    return f" Supporting claim set: {';'.join(claims)}." if len(claims) > 1 else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    unknowns = (
        build_standard_unknowns("lane_1_chapter_12", 12)
        + build_standard_unknowns("lane_2_chapter_13", 13)
        + build_standard_unknowns("lane_3_chapter_14", 14)
        + build_ch15_unknowns()
    )
    handoffs = (
        build_ch12_handoffs()
        + build_standard_handoffs("lane_2_chapter_13", 13)
        + build_standard_handoffs("lane_3_chapter_14", 14)
        + build_ch15_handoffs()
    )
    ensure_unique(unknowns, "unknown_id")
    ensure_unique(handoffs, "handoff_id")
    write_rows(output_dir / "unknowns.csv", UNKNOWN_COLUMNS, unknowns)
    write_rows(output_dir / "chapter_handoffs.csv", HANDOFF_COLUMNS, handoffs)
    print(f"unknowns={len(unknowns)}")
    print(f"chapter_handoffs={len(handoffs)}")


if __name__ == "__main__":
    main()
