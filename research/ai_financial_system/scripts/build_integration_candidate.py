from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
CANONICAL_DATA = PROJECT_ROOT / "data"
INTEGRATION = PROJECT_ROOT / "integration"
DEFAULT_OUTPUT = INTEGRATION / "candidate_data"

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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV width mismatch: {path}")
    return list(reader.fieldnames), rows


def write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def merge_identical_or_add(
    base_path: Path,
    additions_path: Path,
    output_path: Path,
    key_columns: tuple[str, ...],
) -> dict[str, int]:
    base_columns, base_rows = read_rows(base_path)
    addition_columns, addition_rows = read_rows(additions_path)
    if base_columns != addition_columns:
        raise ValueError(
            f"Header mismatch for {base_path.name}: "
            f"{base_columns!r} != {addition_columns!r}"
        )
    base_by_key = {
        tuple(row[column] for column in key_columns): row for row in base_rows
    }
    if len(base_by_key) != len(base_rows):
        raise ValueError(f"Duplicate canonical key in {base_path}: {key_columns}")
    addition_seen: set[tuple[str, ...]] = set()
    added = 0
    already_present = 0
    for row in addition_rows:
        key = tuple(row[column] for column in key_columns)
        if key in addition_seen:
            raise ValueError(f"Duplicate integration key in {additions_path}: {key}")
        addition_seen.add(key)
        existing = base_by_key.get(key)
        if existing is not None:
            if existing != row:
                raise ValueError(
                    f"Conflicting integration row for existing key {key} "
                    f"in {additions_path}"
                )
            already_present += 1
            continue
        base_by_key[key] = row
        base_rows.append(row)
        added += 1
    write_rows(output_path, base_columns, base_rows)
    return {"added": added, "already_present": already_present}


def upsert_rows(
    base_path: Path,
    replacements_path: Path,
    output_path: Path,
    key_columns: tuple[str, ...],
) -> dict[str, int]:
    base_columns, base_rows = read_rows(base_path)
    replacement_columns, replacement_rows = read_rows(replacements_path)
    if base_columns != replacement_columns:
        raise ValueError(
            f"Header mismatch for {base_path.name}: "
            f"{base_columns!r} != {replacement_columns!r}"
        )
    replacement_map: dict[tuple[str, ...], dict[str, str]] = {}
    for row in replacement_rows:
        key = tuple(row[column] for column in key_columns)
        if key in replacement_map:
            raise ValueError(f"Duplicate replacement key in {replacements_path}: {key}")
        replacement_map[key] = row

    updated = 0
    output_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in base_rows:
        key = tuple(row[column] for column in key_columns)
        if key in replacement_map:
            output_rows.append(replacement_map[key])
            seen.add(key)
            updated += 1
        else:
            output_rows.append(row)
    for key, row in replacement_map.items():
        if key not in seen:
            output_rows.append(row)
    write_rows(output_path, base_columns, output_rows)
    return {"updated": updated, "added": len(replacement_map) - updated}


def build_unknowns(output_path: Path) -> dict[str, int]:
    _, base_rows = read_rows(CANONICAL_DATA / "unknowns.csv")
    addition_columns, addition_rows = read_rows(
        INTEGRATION / "accepted_new_unknowns.csv"
    )
    if addition_columns != UNKNOWN_COLUMNS:
        raise ValueError(
            "accepted_new_unknowns.csv does not use the governed unknown schema"
        )
    normalized = [
        {column: row.get(column, "") for column in UNKNOWN_COLUMNS}
        for row in base_rows
    ]
    base_by_id = {row["unknown_id"]: row for row in normalized}
    if len(base_by_id) != len(normalized):
        raise ValueError("Duplicate canonical unknown_id")
    addition_seen: set[str] = set()
    added = 0
    already_present = 0
    for row in addition_rows:
        unknown_id = row["unknown_id"]
        if unknown_id in addition_seen:
            raise ValueError(f"Duplicate unknown_id: {row['unknown_id']}")
        addition_seen.add(unknown_id)
        existing = base_by_id.get(unknown_id)
        if existing is not None:
            if existing != row:
                raise ValueError(
                    f"Conflicting integration row for existing unknown_id: {unknown_id}"
                )
            already_present += 1
            continue
        base_by_id[unknown_id] = row
        normalized.append(row)
        added += 1
    write_rows(output_path, UNKNOWN_COLUMNS, normalized)
    return {"added": added, "already_present": already_present}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_files = sorted(CANONICAL_DATA.glob("*.csv"))
    if not canonical_files:
        raise FileNotFoundError(f"No canonical CSVs found in {CANONICAL_DATA}")

    specialized = {
        "claims.csv",
        "evidence_links.csv",
        "unknowns.csv",
        "economic_exposures.csv",
        "exposure_components.csv",
        "chapter_handoffs.csv",
    }
    for source in canonical_files:
        if source.name not in specialized:
            shutil.copy2(source, output_dir / source.name)

    additions = {
        "claims": merge_identical_or_add(
            CANONICAL_DATA / "claims.csv",
            INTEGRATION / "accepted_new_claims.csv",
            output_dir / "claims.csv",
            ("claim_id",),
        ),
        "evidence_links": merge_identical_or_add(
            CANONICAL_DATA / "evidence_links.csv",
            INTEGRATION / "accepted_new_evidence_links.csv",
            output_dir / "evidence_links.csv",
            ("claim_id", "source_id", "locator"),
        ),
        "unknowns": build_unknowns(output_dir / "unknowns.csv"),
        "economic_exposures": merge_identical_or_add(
            CANONICAL_DATA / "economic_exposures.csv",
            INTEGRATION / "accepted_new_exposures.csv",
            output_dir / "economic_exposures.csv",
            ("exposure_id",),
        ),
        "exposure_components": merge_identical_or_add(
            CANONICAL_DATA / "exposure_components.csv",
            INTEGRATION / "accepted_new_exposure_components.csv",
            output_dir / "exposure_components.csv",
            ("exposure_id", "component_type", "component_id"),
        ),
        "chapter_handoffs": upsert_rows(
            CANONICAL_DATA / "chapter_handoffs.csv",
            INTEGRATION / "accepted_chapter_handoffs.csv",
            output_dir / "chapter_handoffs.csv",
            ("handoff_id",),
        ),
    }

    output_files = sorted(output_dir.glob("*.csv"))
    if {path.name for path in output_files} != {
        path.name for path in canonical_files
    }:
        raise ValueError("Candidate data tree does not match canonical file population")

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "canonical_data_root": str(CANONICAL_DATA.relative_to(REPO_ROOT)),
        "candidate_data_root": str(output_dir.relative_to(REPO_ROOT)),
        "accepted_additions": additions,
        "files": {
            path.name: {
                "rows": len(read_rows(path)[1]),
                "sha256": sha256(path),
            }
            for path in output_files
        },
        "limits": [
            "This candidate integrates adjudicated new claims, evidence links, "
            "unknowns, and primary-accepted economic exposures.",
            "Accepted chapter-handoff replacements and additions are candidate-only "
            "until the complete integration audit passes.",
            "No canonical data file is modified by this builder.",
        ],
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_candidate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
