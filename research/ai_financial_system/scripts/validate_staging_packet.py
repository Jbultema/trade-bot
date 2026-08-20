from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGISTER = PROJECT_ROOT / "data" / "source_register.csv"


@dataclass
class Validation:
    lane: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    files_checked: int = 0
    csv_rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "passed": self.passed,
            "files_checked": self.files_checked,
            "csv_rows_checked": self.csv_rows_checked,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV has duplicate header names")
        rows = list(reader)
        for index, row in enumerate(rows, start=2):
            if None in row:
                raise ValueError(f"row {index} has more values than header columns")
        return rows


def iter_manifest_files(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if isinstance(manifest.get("files"), list):
        yield from manifest["files"]
    elif isinstance(manifest.get("files"), dict):
        for relpath, metadata in manifest["files"].items():
            yield {"path": relpath, **metadata}
    for package in manifest.get("packages", {}).values():
        if isinstance(package, list):
            yield from package
        elif isinstance(package, dict):
            yield from package.get("files", [])
    yield from manifest.get("review_artifacts", [])


def load_source_urls() -> dict[str, str]:
    return {
        row["source_id"]: row["url"]
        for row in csv_rows(SOURCE_REGISTER)
        if row.get("source_id")
    }


def find_column(fieldnames: Iterable[str], suffix: str) -> str | None:
    return next((name for name in fieldnames if name.endswith(suffix)), None)


def primary_key_columns(path: Path, fieldnames: Iterable[str]) -> tuple[str, ...]:
    """Infer a legacy row identity when a manifest does not declare one."""
    names = tuple(fieldnames)
    if path.name == "proposed_exposure_components.csv":
        if "provisional_component_id" in names:
            return ("provisional_component_id",)
        if "component_id" in names:
            return ("component_id",)
        if "provisional_exposure_id" in names and "component_id" in names:
            return ("provisional_exposure_id", "component_id")
    provisional = next(
        (
            name
            for name in names
            if name.startswith("provisional_") and name.endswith("_id")
        ),
        None,
    )
    if provisional:
        return (provisional,)
    first = names[0] if names else None
    return (first,) if first and first.endswith("_id") else ()


def manifest_primary_key_rules(
    manifest: dict[str, Any],
    result: Validation,
) -> dict[str, tuple[str, ...]]:
    """Return explicit per-file primary keys declared by the packet manifest.

    An empty list is an explicit declaration that the generic validator should
    not impose row-identity uniqueness on that CSV. This is useful for files
    whose integrity is governed by a dedicated schema validator. Omitting the
    property retains the legacy inference behavior for older packets.
    """
    rules: dict[str, tuple[str, ...]] = {}
    for entry in iter_manifest_files(manifest):
        if "primary_key_columns" not in entry:
            continue
        relpath = entry.get("path")
        if not isinstance(relpath, str) or not relpath:
            continue
        declared = entry["primary_key_columns"]
        if not isinstance(declared, list):
            result.errors.append(
                f"manifest primary_key_columns for {relpath} must be a list"
            )
            rules[relpath] = ()
            continue
        if any(not isinstance(column, str) or not column.strip() for column in declared):
            result.errors.append(
                f"manifest primary_key_columns for {relpath} "
                "must contain nonblank strings"
            )
            rules[relpath] = ()
            continue
        columns = tuple(column.strip() for column in declared)
        if len(columns) != len(set(columns)):
            result.errors.append(
                f"manifest primary_key_columns for {relpath} contains duplicates"
            )
            rules[relpath] = ()
            continue
        if Path(relpath).suffix.lower() != ".csv":
            result.errors.append(
                f"manifest primary_key_columns is only valid for CSV files: {relpath}"
            )
            rules[relpath] = ()
            continue
        rules[relpath] = columns
    return rules


def validate_manifest(lane_dir: Path, result: Validation) -> dict[str, Any]:
    manifest_path = lane_dir / "lane_manifest.json"
    if not manifest_path.exists():
        result.errors.append("lane_manifest.json is missing")
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        result.errors.append(f"lane_manifest.json is unreadable: {exc}")
        return {}

    declared_lane = manifest.get("lane", manifest.get("lane_name"))
    if declared_lane != lane_dir.name:
        result.errors.append(
            f"manifest lane {declared_lane!r} does not match directory {lane_dir.name!r}"
        )

    entries = list(iter_manifest_files(manifest))
    if not entries:
        result.errors.append("manifest contains no file entries")
        return manifest

    declared_paths: set[str] = set()
    for entry in entries:
        relpath = entry.get("path")
        if not relpath:
            result.errors.append("manifest file entry lacks path")
            continue
        if relpath in declared_paths:
            result.errors.append(f"duplicate manifest path: {relpath}")
            continue
        declared_paths.add(relpath)
        path = lane_dir / relpath
        if not path.exists():
            result.errors.append(f"manifest file is missing: {relpath}")
            continue
        result.files_checked += 1
        expected_hash = entry.get("sha256")
        actual_hash = sha256(path)
        if expected_hash != actual_hash:
            result.errors.append(
                f"SHA-256 mismatch for {relpath}: expected {expected_hash}, got {actual_hash}"
            )
        if path.suffix.lower() == ".csv":
            try:
                rows = csv_rows(path)
            except ValueError as exc:
                result.errors.append(f"invalid CSV {relpath}: {exc}")
                continue
            result.csv_rows_checked += len(rows)
            expected_rows = entry.get("rows", entry.get("record_count"))
            if expected_rows is not None and expected_rows != len(rows):
                result.errors.append(
                    f"row-count mismatch for {relpath}: "
                    f"expected {expected_rows}, got {len(rows)}"
                )
        elif any(
            entry.get(key) is not None
            for key in ("rows_or_lines", "lines", "line_count")
        ):
            expected_lines = entry.get(
                "rows_or_lines", entry.get("lines", entry.get("line_count"))
            )
            actual_lines = len(path.read_text(encoding="utf-8").splitlines())
            if expected_lines != actual_lines:
                result.errors.append(
                    f"line-count mismatch for {relpath}: "
                    f"expected {expected_lines}, got {actual_lines}"
                )

    cross_reviews = {
        path.relative_to(lane_dir).as_posix()
        for path in lane_dir.rglob("cross_review_*.md")
    }
    undeclared_csvs = {
        path.relative_to(lane_dir).as_posix()
        for path in lane_dir.rglob("*.csv")
        if path.relative_to(lane_dir).as_posix() not in declared_paths
    }
    if undeclared_csvs:
        result.errors.append(
            "CSV files absent from manifest: " + ", ".join(sorted(undeclared_csvs))
        )
    undeclared_reviews = cross_reviews - declared_paths
    if undeclared_reviews:
        result.warnings.append(
            "Cross-review files are intentionally outside the evidentiary manifest: "
            + ", ".join(sorted(undeclared_reviews))
        )
    result.checks.append("manifest paths, hashes, row counts, and line counts")
    return manifest


def validate_csv_links(
    lane_dir: Path,
    result: Validation,
    manifest: dict[str, Any],
) -> None:
    source_urls = load_source_urls()
    declared_key_rules = manifest_primary_key_rules(manifest, result)
    ids: dict[str, str] = {}
    claim_ids: set[str] = set()
    source_claim_refs: list[tuple[str, int, str]] = []

    for path in sorted(lane_dir.rglob("*.csv")):
        relpath = path.relative_to(lane_dir).as_posix()
        try:
            rows = csv_rows(path)
        except ValueError:
            continue
        if not rows:
            continue
        fieldnames = rows[0].keys()
        if relpath in declared_key_rules:
            key_columns = declared_key_rules[relpath]
            missing_columns = [
                column for column in key_columns if column not in fieldnames
            ]
            if missing_columns:
                result.errors.append(
                    f"{relpath} manifest primary_key_columns are absent from CSV: "
                    + ", ".join(missing_columns)
                )
                key_columns = ()
        else:
            key_columns = primary_key_columns(path, fieldnames)
        if key_columns:
            for line, row in enumerate(rows, start=2):
                values = tuple(row.get(column, "").strip() for column in key_columns)
                if any(not value for value in values):
                    result.errors.append(
                        f"{relpath}:{line} has blank primary-key component "
                        f"{key_columns}"
                    )
                    continue
                provisional_id = "|".join(values)
                identity = f"{relpath}:{provisional_id}"
                prior = ids.get(identity)
                if prior:
                    result.errors.append(
                        f"duplicate provisional ID {provisional_id}: {prior}, "
                        f"{relpath}:{line}"
                    )
                else:
                    ids[identity] = f"{relpath}:{line}"
                if key_columns in (("provisional_claim_id",), ("claim_id",)) and (
                    path.name == "proposed_claims.csv"
                ):
                    claim_ids.add(provisional_id)

        if "source_id" in fieldnames and "source_url" in fieldnames:
            for line, row in enumerate(rows, start=2):
                source_ids = [
                    item.strip()
                    for item in row.get("source_id", "").split("|")
                    if item.strip()
                ]
                source_urls_in_row = [
                    item.strip() for item in row.get("source_url", "").split("|")
                ]
                if not source_ids:
                    result.errors.append(f"{relpath}:{line} has blank source_id")
                    continue
                if len(source_ids) != len(source_urls_in_row):
                    result.errors.append(
                        f"{relpath}:{line} has {len(source_ids)} source IDs but "
                        f"{len(source_urls_in_row)} URLs"
                    )
                    continue
                for source_id, source_url in zip(
                    source_ids,
                    source_urls_in_row,
                    strict=True,
                ):
                    canonical_url = source_urls.get(source_id)
                    if canonical_url is None:
                        if not source_id.startswith(("L1-", "L2-", "L3-")):
                            result.warnings.append(
                                f"{relpath}:{line} references unregistered source "
                                f"{source_id}"
                            )
                    elif source_url != canonical_url:
                        result.errors.append(
                            f"{relpath}:{line} URL mismatch for {source_id}: "
                            f"{source_url!r} != {canonical_url!r}"
                        )

        source_claim_column = find_column(fieldnames, "source_claim_id")
        if source_claim_column:
            for line, row in enumerate(rows, start=2):
                source_claim_id = row.get(source_claim_column, "").strip()
                if source_claim_id:
                    source_claim_refs.append((relpath, line, source_claim_id))

    for relpath, line, source_claim_id in source_claim_refs:
        if source_claim_id.startswith(
            ("L1-", "L2-", "L3-", "PCLM-")
        ) and source_claim_id not in claim_ids:
            result.errors.append(
                f"{relpath}:{line} references absent staged claim {source_claim_id}"
            )

    result.checks.append(
        "CSV shape and manifest-declared or legacy provisional-ID uniqueness"
    )
    result.checks.append("canonical source-ID and URL consistency")
    result.checks.append("staged source-claim referential integrity")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an isolated AI-financial-system staging packet."
    )
    parser.add_argument("lane_dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    lane_dir = args.lane_dir.resolve()
    result = Validation(lane=lane_dir.name)
    if not lane_dir.is_dir():
        result.errors.append(f"lane directory does not exist: {lane_dir}")
    else:
        manifest = validate_manifest(lane_dir, result)
        validate_csv_links(lane_dir, result, manifest)

    if args.json_output:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status}: {result.lane} | files={result.files_checked} "
            f"csv_rows={result.csv_rows_checked}"
        )
        for check in result.checks:
            print(f"  check: {check}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        for error in result.errors:
            print(f"  error: {error}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
