from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "staging/chapters_06_09_source_preservation_inventory.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "sources/chapters_06_09/source_archive_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "sources/chapters_06_09/source_archive_validation.json"
)
ELIGIBLE_CLASSES = {"P1", "P2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["source_id"]: row for row in csv.DictReader(handle)}


def validate(
    inventory_path: Path, manifest_path: Path
) -> dict[str, object]:
    inventory = load_inventory(inventory_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    checks = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    eligible = {
        source_id
        for source_id, row in inventory.items()
        if row["preservation_class"] in ELIGIBLE_CLASSES
        and row["url"].startswith(("https://", "http://"))
    }
    record_ids = [str(row.get("source_id", "")) for row in records]
    check(
        "one_record_per_eligible_source",
        len(record_ids) == len(set(record_ids)) and set(record_ids) == eligible,
        {
            "eligible_count": len(eligible),
            "record_count": len(record_ids),
            "duplicates": sorted(
                source_id
                for source_id in set(record_ids)
                if record_ids.count(source_id) > 1
            ),
            "missing": sorted(eligible - set(record_ids)),
            "unexpected": sorted(set(record_ids) - eligible),
        },
    )

    row_audit = []
    for record in records:
        source_id = str(record.get("source_id", ""))
        inventory_row = inventory.get(source_id, {})
        status = str(record.get("status", ""))
        preserved = status.startswith("preserved")
        archive_value = str(record.get("local_archive_path", ""))
        metadata_value = str(record.get("response_metadata_path", ""))
        archive_path = REPO_ROOT / archive_value if archive_value else None
        metadata_path = REPO_ROOT / metadata_value if metadata_value else None
        archive_exists = bool(archive_path and archive_path.is_file())
        metadata_exists = bool(metadata_path and metadata_path.is_file())
        actual_hash = sha256(archive_path) if archive_exists and archive_path else ""
        actual_size = archive_path.stat().st_size if archive_exists and archive_path else 0
        expected_hash = str(record.get("content_sha256", ""))
        expected_size = int(record.get("byte_length", 0) or 0)
        url_match = record.get("original_url") == inventory_row.get("url")
        if preserved:
            row_pass = all(
                (
                    archive_exists,
                    metadata_exists,
                    bool(expected_hash),
                    actual_hash == expected_hash,
                    actual_size == expected_size,
                    url_match,
                    bool(record.get("final_resolved_url")),
                    bool(record.get("retrieval_completed_utc")),
                    bool(record.get("mime_type")),
                )
            )
        else:
            row_pass = all(
                (
                    status == "retrieval_failed",
                    not archive_value,
                    not expected_hash,
                    expected_size == 0,
                    bool(record.get("exception_reason")),
                    url_match,
                )
            )
        row_audit.append(
            {
                "source_id": source_id,
                "status": status,
                "passed": row_pass,
                "archive_exists": archive_exists,
                "metadata_exists": metadata_exists,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "url_match": url_match,
                "exception_reason": record.get("exception_reason", ""),
            }
        )
    failed_rows = [row for row in row_audit if not row["passed"]]
    check(
        "record_integrity",
        not failed_rows,
        {"row_count": len(row_audit), "failed_rows": failed_rows},
    )

    preserved_count = sum(
        str(row.get("status", "")).startswith("preserved") for row in records
    )
    failed_count = sum(row.get("status") == "retrieval_failed" for row in records)
    check(
        "manifest_counts",
        (
            manifest.get("record_count") == len(records)
            and manifest.get("preserved_count") == preserved_count
            and manifest.get("failed_count") == failed_count
        ),
        {
            "reported": {
                "record_count": manifest.get("record_count"),
                "preserved_count": manifest.get("preserved_count"),
                "failed_count": manifest.get("failed_count"),
            },
            "recomputed": {
                "record_count": len(records),
                "preserved_count": preserved_count,
                "failed_count": failed_count,
            },
        },
    )

    duplicate_hashes: dict[str, list[str]] = {}
    for row in row_audit:
        digest = str(row["actual_sha256"])
        if digest:
            duplicate_hashes.setdefault(digest, []).append(str(row["source_id"]))
    cross_source_duplicate_hashes = {
        digest: source_ids
        for digest, source_ids in duplicate_hashes.items()
        if len(set(source_ids)) > 1
    }
    check(
        "cross_source_duplicate_hashes_disclosed",
        True,
        cross_source_duplicate_hashes,
    )

    failed_checks = [row["check"] for row in checks if not row["passed"]]
    return {
        "status": "PASS" if not failed_checks else "FAIL",
        "inventory": str(inventory_path.relative_to(REPO_ROOT)),
        "inventory_sha256": sha256(inventory_path),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "checks": checks,
        "failed_checks": failed_checks,
        "row_audit": row_audit,
        "boundary": (
            "Byte and manifest validation only. Semantic locator support and "
            "claim scope require independent review."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.inventory = args.inventory.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    result = validate(args.inventory, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
