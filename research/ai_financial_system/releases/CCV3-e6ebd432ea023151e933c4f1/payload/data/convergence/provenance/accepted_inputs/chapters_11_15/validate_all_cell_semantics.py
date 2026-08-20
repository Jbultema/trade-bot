#!/usr/bin/env python3
"""Independently validate all-cell semantic-reference disposition coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAIM_TOKEN = re.compile(
    r"(?<![A-Z0-9-])(?:L[123]-)?CLM-\d+(?![A-Z0-9-])"
)
SOURCE_TOKEN = re.compile(
    r"(?<![A-Z0-9-])(?:L[123]-)?SRC-\d+(?![A-Z0-9-])"
)

EXPECTED_HELD_REFERENCES = {
    ("CH11", "L3-GATE-0002", "notes", "claim", "L3-CLM-0030"),
    ("CH11", "L3-GATE-0014", "notes", "claim", "L3-CLM-0028"),
    ("CH12", "EQ-12-005", "currently_available_inputs", "claim", "CLM-040003"),
    ("CH12", "EQ-12-006", "currently_available_inputs", "claim", "CLM-040001"),
    ("CH12", "EQ-12-007", "currently_available_inputs", "claim", "CLM-050017"),
    ("CH12", "EQ-12-007", "currently_available_inputs", "claim", "CLM-050028"),
    ("CH13", "FLT-007", "notes", "claim", "CLM-040010"),
    ("CH13", "FLT-007", "notes", "claim", "CLM-040013"),
}


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path} contains an over-width row")
    return rows


def discover_from_every_scalar(
    fields: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Discover claims and sources from every value, without column routing."""

    expected: dict[str, dict[str, str]] = {}
    for field in fields:
        value = field["original_value"]
        token_groups = (
            ("claim", CLAIM_TOKEN.findall(value)),
            ("source", SOURCE_TOKEN.findall(value)),
        )
        for token_type, tokens in token_groups:
            for token in dict.fromkeys(tokens):
                occurrence_id = stable_id(
                    "SEM",
                    field["source_row_id"],
                    field["source_column"],
                    token_type,
                    token,
                )
                expected[occurrence_id] = {
                    "chapter_id": field["chapter_id"],
                    "source_row_id": field["source_row_id"],
                    "source_primary_key": field["source_primary_key"],
                    "source_column": field["source_column"],
                    "token_type": token_type,
                    "token": token,
                }
        if "locator" in field["source_column"].lower() and value:
            occurrence_id = stable_id(
                "SEM",
                field["source_row_id"],
                field["source_column"],
                "locator",
                value,
            )
            expected[occurrence_id] = {
                "chapter_id": field["chapter_id"],
                "source_row_id": field["source_row_id"],
                "source_primary_key": field["source_primary_key"],
                "source_column": field["source_column"],
                "token_type": "locator",
                "token": value,
            }
    return expected


def validate(target: Path) -> dict[str, object]:
    errors: list[str] = []
    fields = read_csv(target / "field_dispositions.csv")
    dispositions = read_csv(target / "semantic_reference_dispositions.csv")
    held = read_csv(target / "unprojected_semantic_references.csv")
    relationships = read_csv(target / "claim_source_locator_projection.csv")

    expected = discover_from_every_scalar(fields)
    disposition_by_id = {
        row["semantic_reference_occurrence_id"]: row for row in dispositions
    }
    if len(disposition_by_id) != len(dispositions):
        errors.append("duplicate semantic-reference occurrence identities")
    if set(disposition_by_id) != set(expected):
        missing = sorted(set(expected) - set(disposition_by_id))
        extra = sorted(set(disposition_by_id) - set(expected))
        errors.append(
            f"all-cell occurrence mismatch missing={missing[:10]} extra={extra[:10]}"
        )

    relationships_by_id = {row["relationship_id"]: row for row in relationships}
    held_by_id = {row["governed_hold_id"]: row for row in held}
    if len(held_by_id) != len(held):
        errors.append("duplicate governed semantic-reference hold identities")

    for occurrence_id, expected_row in expected.items():
        row = disposition_by_id.get(occurrence_id)
        if row is None:
            continue
        for key, value in expected_row.items():
            if row[key] != value:
                errors.append(f"{occurrence_id} differs on {key}")
        relationship_ids = [
            value for value in row["projected_relationship_ids"].split("|") if value
        ]
        hold_id = row["governed_hold_id"]
        if row["disposition"] == "projected_relationship_reference":
            if not relationship_ids or hold_id:
                errors.append(f"{occurrence_id} has an invalid projected disposition")
            for relationship_id in relationship_ids:
                relationship = relationships_by_id.get(relationship_id)
                if relationship is None:
                    errors.append(
                        f"{occurrence_id} references missing relationship {relationship_id}"
                    )
                    continue
                if relationship["source_row_id"] != row["source_row_id"]:
                    errors.append(f"{occurrence_id} crosses source rows")
                if row["token_type"] == "claim" and (
                    relationship["accepted_claim_id"] != row["token"]
                ):
                    errors.append(f"{occurrence_id} claim token is not projected")
                if row["token_type"] == "source" and (
                    relationship["accepted_source_id"] != row["token"]
                ):
                    errors.append(f"{occurrence_id} source token is not projected")
        elif row["disposition"] == "governed_non_support_reference_hold":
            if relationship_ids or not hold_id or hold_id not in held_by_id:
                errors.append(f"{occurrence_id} has an invalid held disposition")
        elif row["disposition"] == "canonical_source_identity_mapping_reference":
            if (
                row["token_type"] != "source"
                or row["source_column"] != "existing_source_id"
                or relationship_ids
                or hold_id
            ):
                errors.append(
                    f"{occurrence_id} has an invalid source-mapping disposition"
                )
        else:
            errors.append(f"{occurrence_id} has unknown disposition {row['disposition']}")
        if (
            row["evidence_state"] != "not_strengthened"
            or row["model_use_state"] != "held_to_acceptance_record"
            or row["publication_eligibility"] != "held"
        ):
            errors.append(f"{occurrence_id} weakens a retained authority boundary")

    held_identities = {
        (
            row["chapter_id"],
            row["source_primary_key"],
            row["source_column"],
            row["token_type"],
            row["token"],
        )
        for row in held
    }
    if held_identities != EXPECTED_HELD_REFERENCES:
        errors.append(
            "explicit held-reference population differs from the eight reviewed omissions"
        )
    for row in held:
        if (
            row["disposition"] != "governed_non_support_reference_hold"
            or row["evidence_state"] != "not_strengthened"
            or row["publication_eligibility"] != "held"
            or row["reference_role"] != "contextual_or_methodological_reference"
        ):
            errors.append(f"{row['finding_id']} is not conservatively held")
        if disposition_by_id.get(
            row["semantic_reference_occurrence_id"], {}
        ).get("governed_hold_id") != row["governed_hold_id"]:
            errors.append(f"{row['finding_id']} is not bound to its disposition")

    disposition_counts = Counter(row["disposition"] for row in dispositions)
    result: dict[str, object] = {
        "status": "PASS" if not errors else "FAIL",
        "validator_independence": (
            "discovers claim/source tokens from every original scalar value; "
            "does not use builder column-name routing or source_projection_coverage"
        ),
        "accepted_scalar_fields_scanned": len(fields),
        "semantic_reference_occurrences": len(expected),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "explicit_governed_reference_holds": len(held),
        "reviewed_omissions_exactly_accounted_for": (
            held_identities == EXPECTED_HELD_REFERENCES
        ),
        "errors": errors,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=HERE)
    args = parser.parse_args()
    target = args.target.resolve()
    result = validate(target)
    (target / "all_cell_semantic_validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{result['status']}: occurrences="
        f"{result['semantic_reference_occurrences']} "
        f"holds={result['explicit_governed_reference_holds']}"
    )
    for error in result["errors"]:
        print(f"ERROR: {error}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
