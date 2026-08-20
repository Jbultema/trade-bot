from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "staging/chapters_06_09_source_preservation_inventory.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "sources/chapters_06_09"
USER_AGENT = (
    "TradeBot-AI-Financial-System-Research/1.0 "
    "contact: local-research-project"
)
PRESERVABLE_CLASSES = {"P1", "P2"}
SELECTED_HEADERS = {
    "content-type",
    "content-length",
    "content-disposition",
    "etag",
    "last-modified",
    "cache-control",
    "date",
    "server",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "source"


def extension_for(response: requests.Response, url: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(
        r"filename\*?=(?:UTF-8'')?[\"']?([^;\"']+)",
        disposition,
    )
    if match:
        suffix = Path(match.group(1)).suffix
        if suffix:
            return suffix.lower()
    url_suffix = Path(urlparse(response.url or url).path).suffix
    if url_suffix and len(url_suffix) <= 8:
        return url_suffix.lower()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    return mimetypes.guess_extension(content_type) or ".bin"


def version_path(output_dir: Path, source_id: str, extension: str, digest: str) -> Path:
    return output_dir / f"{safe_component(source_id)}_{digest[:16]}{extension}"


def existing_hashes(output_dir: Path, source_id: str) -> dict[str, Path]:
    results = {}
    for path in output_dir.glob(f"{safe_component(source_id)}_*"):
        if path.suffix == ".json" or not path.is_file():
            continue
        results[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    return results


def latest_existing_record(
    output_dir: Path,
    source_id: str,
    original_url: str,
) -> dict[str, object] | None:
    candidates = sorted(
        output_dir.glob(f"{safe_component(source_id)}_*.response.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("source_id") == source_id
            and record.get("original_url") == original_url
            and str(record.get("status", "")).startswith("preserved")
            and record.get("content_sha256")
            and record.get("local_archive_path")
        ):
            record["resume_state"] = "reused_preserved_record"
            return record
    return None


def selected_headers(response: requests.Response) -> dict[str, str]:
    return {
        key.casefold(): value
        for key, value in response.headers.items()
        if key.casefold() in SELECTED_HEADERS
    }


def load_rows(path: Path, source_ids: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The canonical source register records evidence grades, while bounded
    # preservation inventories record preservation classes.  Accept either
    # schema so a reviewed set of canonical source IDs can be archived without
    # creating a hand-maintained duplicate register.
    grade_to_class = {"A": "P1", "B": "P2"}
    for row in rows:
        if not row.get("preservation_class"):
            row["preservation_class"] = grade_to_class.get(
                row.get("evidence_grade", ""),
                "P4",
            )
    if source_ids:
        rows = [row for row in rows if row["source_id"] in source_ids]
    return rows


def fetch_one(
    session: requests.Session,
    row: dict[str, str],
    output_dir: Path,
    timeout: float,
    insecure_hosts: set[str],
) -> dict[str, object]:
    source_id = row["source_id"]
    started = dt.datetime.now(dt.UTC)
    requested_host = (urlparse(row["url"]).hostname or "").casefold()
    verify_tls = requested_host not in insecure_hosts
    base = {
        "source_id": source_id,
        "original_url": row["url"],
        "publisher": row["publisher"],
        "title": row["title"],
        "publication_date": row["publication_date"],
        "preservation_class": row["preservation_class"],
        "retrieval_started_utc": started.isoformat(),
        "preservation_method": "sequential_http_get",
        "request_user_agent": session.headers.get("User-Agent", ""),
        "tls_verification": (
            "verified"
            if verify_tls
            else f"disabled_for_explicit_host:{requested_host}"
        ),
        "access_or_licensing_restriction": "",
        "extraction_method": "none_original_bytes",
        "locator_relationship": "pending_semantic_locator_review",
    }
    try:
        response = session.get(
            row["url"],
            timeout=timeout,
            allow_redirects=True,
            verify=verify_tls,
        )
        response.raise_for_status()
        payload = response.content
        if not payload:
            raise ValueError("empty response body")
        digest = sha256_bytes(payload)
        extension = extension_for(response, row["url"])
        known = existing_hashes(output_dir, source_id)
        if digest in known:
            archive_path = known[digest]
            write_state = "existing_identical_bytes"
        else:
            archive_path = version_path(output_dir, source_id, extension, digest)
            archive_path.write_bytes(payload)
            write_state = "new_version_written"
        metadata_path = output_dir / f"{archive_path.stem}.response.json"
        record = {
            **base,
            "status": (
                "preserved_complete"
                if row["preservation_class"] == "P1"
                else "preserved_response_plus_metadata"
            ),
            "http_status": response.status_code,
            "final_resolved_url": response.url,
            "redirect_history": [
                {"status": item.status_code, "url": item.url}
                for item in response.history
            ],
            "retrieval_completed_utc": dt.datetime.now(dt.UTC).isoformat(),
            "mime_type": response.headers.get("content-type", ""),
            "byte_length": len(payload),
            "content_sha256": digest,
            "local_archive_path": str(archive_path.relative_to(REPO_ROOT)),
            "response_metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
            "selected_response_headers": selected_headers(response),
            "write_state": write_state,
            "exception_reason": "",
        }
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record
    except Exception as exc:  # network failures must remain explicit evidence
        return {
            **base,
            "status": "retrieval_failed",
            "http_status": "",
            "final_resolved_url": "",
            "redirect_history": [],
            "retrieval_completed_utc": dt.datetime.now(dt.UTC).isoformat(),
            "mime_type": "",
            "byte_length": 0,
            "content_sha256": "",
            "local_archive_path": "",
            "response_metadata_path": "",
            "selected_response_headers": {},
            "write_state": "no_bytes_written",
            "exception_reason": f"{type(exc).__name__}: {exc}",
        }


def write_manifest(records: list[dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "source_archive_manifest.json"
    payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "record_count": len(records),
        "preserved_count": sum(
            str(row["status"]).startswith("preserved") for row in records
        ),
        "failed_count": sum(row["status"] == "retrieval_failed" for row in records),
        "boundary": (
            "Byte preservation does not establish truth, scope, evidence grade, "
            "or semantic locator support."
        ),
        "records": records,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--delay-seconds", type=float, default=0.75)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--user-agent", default=USER_AGENT)
    parser.add_argument(
        "--allow-insecure-host",
        action="append",
        default=[],
        help=(
            "Disable TLS certificate verification only for an explicitly named "
            "host. The exception is recorded in response metadata."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid preserved response records and fetch only missing sources.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Perform sequential network retrievals; otherwise print a plan only.",
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()

    rows = load_rows(args.input, set(args.source_id))
    eligible = [
        row
        for row in rows
        if row["preservation_class"] in PRESERVABLE_CLASSES
        and row["url"].startswith(("https://", "http://"))
    ]
    plan = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "selected_source_count": len(rows),
        "eligible_source_count": len(eligible),
        "skipped_rights_or_url_count": len(rows) - len(eligible),
        "source_ids": [row["source_id"] for row in eligible],
        "mode": "fetch" if args.fetch else "plan_only",
    }
    if not args.fetch:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    records = []
    insecure_hosts = {host.casefold() for host in args.allow_insecure_host}
    for row in eligible:
        existing = (
            latest_existing_record(
                args.output_dir,
                row["source_id"],
                row["url"],
            )
            if args.resume
            else None
        )
        if existing is not None:
            records.append(existing)
            continue
        if records:
            time.sleep(max(args.delay_seconds, 0))
        records.append(
            fetch_one(
                session,
                row,
                args.output_dir,
                args.timeout_seconds,
                insecure_hosts,
            )
        )
    manifest = write_manifest(records, args.output_dir)
    result = {
        **plan,
        "manifest": str(manifest),
        "preserved_count": sum(
            str(row["status"]).startswith("preserved") for row in records
        ),
        "failed_count": sum(row["status"] == "retrieval_failed" for row in records),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
