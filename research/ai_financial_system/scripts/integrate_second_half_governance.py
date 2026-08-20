from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = PROJECT_ROOT / "integration"
CANDIDATE = INTEGRATION / "second_half_governance_candidate"

TABLES = {
    "unknowns.csv": (
        INTEGRATION / "accepted_new_unknowns.csv",
        "unknown_id",
        30,
    ),
    "chapter_handoffs.csv": (
        INTEGRATION / "accepted_chapter_handoffs.csv",
        "handoff_id",
        21,
    ),
}


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames is None or any(None in row for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return list(reader.fieldnames), rows


def merge_table(
    candidate_path: Path,
    accepted_path: Path,
    key: str,
    expected_candidate_rows: int,
) -> None:
    candidate_fields, candidate_rows = read_table(candidate_path)
    accepted_fields, accepted_rows = read_table(accepted_path)
    if candidate_fields != accepted_fields:
        raise ValueError(f"Schema mismatch: {candidate_path} vs {accepted_path}")
    if len(candidate_rows) != expected_candidate_rows:
        raise ValueError(
            f"Unexpected candidate population in {candidate_path}: "
            f"{len(candidate_rows)}"
        )

    merged = {row[key]: row for row in accepted_rows}
    for row in candidate_rows:
        identifier = row[key]
        prior = merged.get(identifier)
        if prior is not None and prior != row:
            # This script owns the bounded second-half candidate IDs. Re-running
            # it must replace a previously integrated serialization after a
            # downstream database gate identifies a defect.
            print(f"{accepted_path.name}: replacing managed {key}={identifier}")
        merged[identifier] = row

    ordered = sorted(merged.values(), key=lambda row: row[key])
    with accepted_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=accepted_fields)
        writer.writeheader()
        writer.writerows(ordered)
    print(
        f"{accepted_path.name}: prior={len(accepted_rows)} "
        f"candidate={len(candidate_rows)} merged={len(ordered)}"
    )


def main() -> None:
    for candidate_name, (
        accepted_path,
        key,
        expected_rows,
    ) in TABLES.items():
        merge_table(
            CANDIDATE / candidate_name,
            accepted_path,
            key,
            expected_rows,
        )


if __name__ == "__main__":
    main()
