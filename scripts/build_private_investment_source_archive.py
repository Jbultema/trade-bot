#!/usr/bin/env python3
"""Validate and inventory preserved SEC filings for private-investment research."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "research/ai_financial_system/staging/"
    "lane_1_private_investment_earnings_corrected/"
    "primary_source_preservation_audit.csv"
)
ARCHIVE = (
    ROOT
    / "research/ai_financial_system/sources/sec/private_investment_earnings"
)
SOURCE_REGISTER = ROOT / "research/ai_financial_system/data/source_register.csv"
MANIFEST = ARCHIVE / "source_archive_manifest.csv"
VALIDATION = ARCHIVE / "archive_validation.json"

ARCHIVE_FILES = {
    "SRC-0010": "SRC-0010_msft_20260331.htm",
    "SRC-0011": "SRC-0011_meta_20260331.htm",
    "SRC-0012": "SRC-0012_goog_20260331.htm",
    "SRC-0013": "SRC-0013_amzn_20260331.htm",
    "SRC-0014": "SRC-0014_orcl_20260228.htm",
    "SRC-0018": "SRC-0018_nvda_20260426.htm",
    "SRC-0064": "SRC-0064_tsla_20260331.htm",
    "SRC-0078": "SRC-0078_meta_20250930.htm",
}

EXPECTED = {
    "SRC-0010": ("MICROSOFT", "10-Q", "March 31, 2026"),
    "SRC-0011": ("Meta Platforms", "10-Q", "March 31, 2026"),
    "SRC-0012": ("Alphabet", "10-Q", "March 31, 2026"),
    "SRC-0013": ("AMAZON", "10-Q", "March 31, 2026"),
    "SRC-0014": ("ORACLE", "10-Q", "February 28, 2026"),
    "SRC-0018": ("NVIDIA", "10-Q", "April 26, 2026"),
    "SRC-0064": ("Tesla", "10-Q", "March 31, 2026"),
    "SRC-0078": ("Meta Platforms", "10-Q", "September 30, 2025"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inline_value(text: str, concept: str) -> str:
    pattern = re.compile(
        rf"name\s*=\s*['\"]dei:{re.escape(concept)}['\"][^>]*>(.*?)</ix:",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    value = re.sub(r"<[^>]+>", "", match.group(1))
    value = html.unescape(value).replace("\xa0", " ")
    return " ".join(value.split())


def main() -> None:
    with AUDIT.open(newline="", encoding="utf-8") as stream:
        audit_rows = list(csv.DictReader(stream))
    with SOURCE_REGISTER.open(newline="", encoding="utf-8") as stream:
        source_register = {
            row["source_id"]: row for row in csv.DictReader(stream)
        }

    checks: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for audit in sorted(audit_rows, key=lambda row: row["source_id"]):
        source_id = audit["source_id"]
        filename = ARCHIVE_FILES[source_id]
        path = ARCHIVE / filename
        expected_entity, expected_type, expected_period = EXPECTED[source_id]
        source = source_register[source_id]
        exists = path.is_file()
        raw = path.read_bytes() if exists else b""
        text = raw.decode("utf-8", errors="replace")
        registrant = inline_value(text, "EntityRegistrantName")
        document_type = inline_value(text, "DocumentType")
        period_end = inline_value(text, "DocumentPeriodEndDate")
        looks_like_filing = (
            exists
            and len(raw) > 100_000
            and "<html" in text.lower()
            and "inline xbrl" in text.lower()
            and "your request originates from an undeclared automated tool" not in text.lower()
            and "request rate threshold exceeded" not in text.lower()
        )
        identity_match = (
            expected_entity.lower() in registrant.lower()
            and document_type == expected_type
            and period_end == expected_period
        )
        status = (
            "ARCHIVED_CONTENT_HASHED_AND_IDENTITY_VERIFIED"
            if looks_like_filing and identity_match
            else "FAILED_ARCHIVE_VALIDATION"
        )
        relative_path = path.relative_to(ROOT).as_posix()
        retrieval_timestamp = (
            dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
            if exists
            else ""
        )
        manifest_rows.append(
            {
                "source_id": source_id,
                "source_url": audit["source_url"],
                "resolved_url": audit["source_url"],
                "retrieval_timestamp_utc": retrieval_timestamp,
                "publisher": source["publisher"],
                "document_title": source["title"],
                "publication_or_filing_date": source["publication_date"],
                "mime_type": "text/html; inline-xbrl",
                "archived_path": relative_path,
                "content_sha256": sha256(path) if exists else "",
                "byte_count": len(raw),
                "preservation_class": "P1_public_primary_document",
                "preservation_method": (
                    "curl_-L_--fail_--retry_2_with_declared_research_user_agent"
                ),
                "http_status": (
                    "200_verified_by_curl_fail_closed_success"
                    if exists
                    else "retrieval_failed"
                ),
                "response_metadata_state": (
                    "body_preserved_headers_not_separately_frozen"
                    if exists
                    else "unavailable"
                ),
                "access_or_licensing_restriction": "public_SEC_EDGAR",
                "extraction_or_ocr_method": "none_original_inline_xbrl_bytes",
                "cited_locator_relationship": "pending_claim_level_locator_review",
                "registrant": registrant,
                "document_type": document_type,
                "document_period_end": period_end,
                "preservation_status": (
                    "preserved_complete"
                    if status.startswith("ARCHIVED_")
                    else "retrieval_failed"
                ),
                "exception_reason": "",
                "publication_use": (
                    "eligible_subject_to_claim_level_semantic_review"
                    if status.startswith("ARCHIVED_")
                    else "blocked"
                ),
            }
        )
        checks.append(
            {
                "source_id": source_id,
                "exists": exists,
                "looks_like_sec_inline_xbrl": looks_like_filing,
                "identity_match": identity_match,
                "status": "PASS" if status.startswith("ARCHIVED_") else "FAIL",
            }
        )

    fieldnames = [
        "source_id",
        "source_url",
        "resolved_url",
        "retrieval_timestamp_utc",
        "publisher",
        "document_title",
        "publication_or_filing_date",
        "mime_type",
        "archived_path",
        "content_sha256",
        "byte_count",
        "preservation_class",
        "preservation_method",
        "http_status",
        "response_metadata_state",
        "access_or_licensing_restriction",
        "extraction_or_ocr_method",
        "cited_locator_relationship",
        "registrant",
        "document_type",
        "document_period_end",
        "preservation_status",
        "exception_reason",
        "publication_use",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "source_count": len(checks),
        "passed_source_count": sum(row["status"] == "PASS" for row in checks),
        "failed_source_count": sum(row["status"] != "PASS" for row in checks),
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "checks": checks,
    }
    VALIDATION.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        raise SystemExit("private-investment source archive validation failed")


if __name__ == "__main__":
    main()
