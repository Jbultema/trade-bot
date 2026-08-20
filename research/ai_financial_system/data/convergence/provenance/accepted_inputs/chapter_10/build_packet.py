#!/usr/bin/env python3
"""Build the narrow, lossless Chapter 10 rebuilt-v3 author packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AI = ROOT / "research/ai_financial_system"
DB = ROOT / "reports/ai_financial_system/research.db"
V1 = AI / "staging/lane_1_chapter_10_rebuilt_v1"
REVIEW = AI / "staging/lane_2_verify_chapter_10_rebuilt_v1"
V2_AUTHOR = AI / "staging/lane_3_chapter_10_rebuilt_v2"
V2_REVIEW = AI / "staging/lane_4_verify_chapter_10_rebuilt_v2"
BASELINE_SHA = "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
V1_MANIFEST_SHA = "0b387ceaee5ed81706b5400f3ec391950e6af99c9b36e6208f78b6dbcee347f6"
REVIEW_MANIFEST_SHA = "04e96776fee3576bc9de2144e53fe095946bc2ee2b2b0b1dea7da97cb6874f49"
V2_AUTHOR_MANIFEST_SHA = "ab0108513e9ceb9d9fa6e532133405cb4f8b2fcad96b212b44209c28347207f0"
V2_REVIEW_MANIFEST_SHA = "d5d9bc7ae7cf28da974699adff6ed7297b627b868b8440f16af4cae02951c890"

STATIC_FILES = {
    "build_packet.py",
    "collect_validation_provenance.py",
    "extract_source_derivatives.py",
    "retrieve_sources.sh",
    "seal_manifest.py",
    "validate_packet.py",
    "verify_clean_builds.py",
}
STATIC_DIRECTORIES = {"response_metadata", "source_archive", "source_derivatives"}

ENRICHED_FILES = {
    "entity_resolution.csv",
    "falsifiers.csv",
    "locator_semantic_dispositions.csv",
    "source_version_references.csv",
    "timeline.csv",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV: {path}")
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    if not rows and not fields:
        raise ValueError(f"cannot infer empty CSV schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_db_row(row: sqlite3.Row) -> dict[str, str]:
    return {
        key: "" if row[key] is None else str(row[key])
        for key in row.keys()  # noqa: SIM118
    }


def prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if output != HERE:
        for name in sorted(STATIC_FILES):
            shutil.copy2(HERE / name, output / name)
        for name in sorted(STATIC_DIRECTORIES):
            shutil.copytree(HERE / name, output / name)
    inventory = output / "source_inventory"
    if inventory.exists():
        shutil.rmtree(inventory)
    inventory.mkdir()


def verify_protected_inputs() -> None:
    if sha(DB) != BASELINE_SHA:
        raise ValueError("canonical database drift")
    if sha(V1 / "lane_manifest.json") != V1_MANIFEST_SHA:
        raise ValueError("rebuilt-v1 author manifest drift")
    if sha(REVIEW / "lane_manifest.json") != REVIEW_MANIFEST_SHA:
        raise ValueError("rebuilt-v1 review manifest drift")
    if sha(V2_AUTHOR / "lane_manifest.json") != V2_AUTHOR_MANIFEST_SHA:
        raise ValueError("rebuilt-v2 author manifest drift")
    if sha(V2_REVIEW / "lane_manifest.json") != V2_REVIEW_MANIFEST_SHA:
        raise ValueError("rebuilt-v2 review manifest drift")
    connection = sqlite3.connect(DB)
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("canonical database integrity failure")
    if list(connection.execute("PRAGMA foreign_key_check")):
        raise ValueError("canonical database foreign-key failure")
    connection.close()


def copy_lossless_source_inventory(output: Path) -> list[str]:
    typed_rows = read_csv(V1 / "typed_row_dispositions.csv")
    source_files = sorted({row["source_file"] for row in typed_rows})
    for name in source_files:
        shutil.copy2(V1 / name, output / "source_inventory" / name)
        if name not in ENRICHED_FILES:
            shutil.copy2(V1 / name, output / name)
    return source_files


MATERIAL_ALIASES = [
    ("CoreWeave", "ENT-CRWV", "proposed_atomic_claims.csv:CLM-100019"),
    ("CoreWeave Inc.", "ENT-CRWV", "proposed_sources.csv:SRC-0076"),
    ("G42", "ENT-G42", "proposed_atomic_claims.csv:CLM-100021"),
    ("Group 42", "ENT-G42", "proposed_sources.csv:SRC-0152"),
    ("Mubadala", "ENT-MUBADALA", "proposed_atomic_claims.csv:CLM-100021"),
    ("Mubadala Investment Company", "ENT-MUBADALA", "proposed_sources.csv:SRC-0152"),
    ("Microsoft", "ENT-MSFT", "proposed_sources.csv:SRC-0154"),
    ("NVIDIA", "ENT-NVDA", "proposed_sources.csv:SRC-0154"),
    ("EIB", "ENT-EIB", "proposed_atomic_claims.csv:CLM-100025"),
    ("European Investment Bank", "ENT-EIB", "proposed_sources.csv:SRC-0156"),
    ("EIF", "ENT-EIF", "proposed_atomic_claims.csv:CLM-100025"),
    ("European Investment Fund", "ENT-EIF", "proposed_sources.csv:SRC-0156"),
    ("U.S. Department of Commerce", "ENT-US-DOC", "proposed_sources.csv:SRC-0136"),
    ("Commerce", "ENT-US-DOC", "proposed_evidence_links.csv:EL-CH10-0006"),
    ("European Commission", "ENT-EU-COMMISSION", "proposed_sources.csv:SRC-0155"),
    ("Government of Canada", "ENT-CANADA", "proposed_sources.csv:SRC-0075"),
    ("AIATC", "ENT-AIATC", "proposed_atomic_claims.csv:CLM-100021"),
    ("Public Investment Fund", "ENT-PIF", "proposed_sources.csv:SRC-0151"),
    ("PIF", "ENT-PIF", "proposed_atomic_claims.csv:CLM-100020"),
    ("Aligned Data Centers", "ENT-ALIGNED", "proposed_atomic_claims.csv:CLM-100023"),
    ("Cohere", "ENT-COHERE", "proposed_atomic_claims.csv:CLM-100019"),
    ("Green Chile", "ENT-GREEN-CHILE", "proposed_atomic_claims.csv:CLM-100018"),
    ("Oracle", "ENT-ORCL", "proposed_atomic_claims.csv:CLM-100018"),
    ("BlackRock GIP", "", "proposed_atomic_claims.csv:CLM-100023"),
    ("BlackRock Global Infrastructure Partners", "", "proposed_sources.csv:SRC-0154"),
    ("Internal Revenue Service", "", "proposed_sources.csv:SRC-0149"),
    ("United States Congress", "", "proposed_sources.csv:SRC-0148"),
    ("Louisiana Public Service Commission", "", "proposed_sources.csv:SRC-0087"),
    ("Michigan Public Service Commission", "", "proposed_sources.csv:SRC-0098"),
]


def build_entities(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial: set[str] = set()
    for path in sorted((output / "source_inventory").glob("*.csv")):
        for row in read_csv(path):
            for value in row.values():
                initial.update(re.findall(r"\bENT-[A-Z0-9-]+\b", value))
    initial.update(entity_id for _, entity_id, _ in MATERIAL_ALIASES if entity_id)

    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    pending = sorted(initial)
    entities: dict[str, dict[str, str]] = {}
    while pending:
        entity_id = pending.pop(0)
        if entity_id in entities:
            continue
        row = connection.execute(
            "SELECT * FROM entities WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unresolved canonical entity: {entity_id}")
        normalized = normalize_db_row(row)
        entities[entity_id] = normalized
        parent = normalized["parent_entity_id"]
        if parent and parent not in entities:
            pending.append(parent)
    connection.close()

    entity_rows = [
        {
            "entity_resolution_id": f"ER-CH10-V2-{index:04d}",
            **entities[entity_id],
            "resolution_status": "exact_canonical_identity_recursive_fk_closed",
        }
        for index, entity_id in enumerate(sorted(entities), start=1)
    ]
    alias_rows = []
    for index, (alias, entity_id, provenance) in enumerate(
        MATERIAL_ALIASES,
        start=1,
    ):
        resolved = bool(entity_id)
        alias_rows.append(
            {
                "entity_alias_id": f"EA-CH10-{index:04d}",
                "alias": alias,
                "canonical_entity_id": entity_id,
                "resolution_status": (
                    "resolved_exact_or_documented_short_name"
                    if resolved
                    else "governed_unresolved_hold"
                ),
                "alias_provenance": provenance,
                "resolution_note": (
                    "canonical row included in recursive entity closure"
                    if resolved
                    else (
                        "no separate canonical identity exists; do not conflate the "
                        "named party with a partnership, regulator, or parent"
                    )
                ),
                "publication_eligibility": "held_pending_fresh_nonauthor_review",
            }
        )
    write_csv(output / "entity_resolution.csv", entity_rows)
    write_csv(output / "entity_aliases.csv", alias_rows)
    return entity_rows, alias_rows


def attempt_name(path: Path) -> str:
    return path.name.removesuffix(".curl.json")


def build_source_versions(
    output: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    sources = read_csv(output / "proposed_sources.csv")
    source_by_id = {row["source_id"]: row for row in sources}
    attempts_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_versions: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []

    for metadata_path in sorted((output / "response_metadata").glob("*.curl.json")):
        attempt = attempt_name(metadata_path)
        source_id, version = attempt.rsplit("_", 1)
        if source_id not in source_by_id:
            raise ValueError(f"unexpected source attempt: {attempt}")
        transfer = json.loads(metadata_path.read_text(encoding="utf-8"))
        failure_path = output / "response_metadata" / f"{attempt}.failure.txt"
        archives = sorted((output / "source_archive").glob(f"{attempt}.*"))
        success = bool(archives) and not failure_path.exists()
        archive = archives[0] if archives else None
        derivative = output / "source_derivatives" / f"{attempt}_visible_text.txt"
        source = source_by_id[source_id]
        version_id = f"CH10-SV-{source_id}-{version}"
        failure_text = (
            failure_path.read_text(encoding="utf-8").strip()
            if failure_path.exists()
            else ""
        )
        row = {
            "source_version_id": version_id,
            "source_id": source_id,
            "version_ordinal": version,
            "version_role": (
                "cited_original" if version == "v0001" else "lawful_official_alternate"
            ),
            "original_url": transfer.get("url", source["url"]),
            "canonical_cited_url": source["url"],
            "final_resolved_url": transfer.get("url_effective", ""),
            "acquisition_date_utc": "2026-07-26",
            "publication_date": source["publication_date"],
            "preservation_status": (
                "preserved_complete" if success else "retrieval_or_validity_hold"
            ),
            "http_status": str(transfer.get("http_code", "")),
            "mime_type": transfer.get("content_type") or "",
            "byte_count": archive.stat().st_size if success and archive else "",
            "content_sha256": sha(archive) if success and archive else "",
            "local_archive_path": (
                archive.relative_to(output).as_posix()
                if success and archive
                else ""
            ),
            "response_headers_path": (
                f"response_metadata/{attempt}.headers"
                if (output / "response_metadata" / f"{attempt}.headers").is_file()
                else ""
            ),
            "transfer_metadata_path": metadata_path.relative_to(output).as_posix(),
            "visible_text_path": (
                derivative.relative_to(output).as_posix()
                if success and derivative.is_file()
                else ""
            ),
            "rights_access_boundary": (
                "public_primary_lawful_packet_archive"
                if success
                else "access_control_respected_no_circumvention"
            ),
            "exception_state": (
                "" if success else failure_text.replace("\n", "; ")
            ),
            "semantic_clearance_boundary": (
                "byte_identity_only_not_claim_support"
                if success
                else "no_bytes_or_invalid_response_no_semantic_support"
            ),
            "selected_governing_version": "0",
        }
        attempts_by_source[source_id].append(row)
        all_versions.append(row)
        if not success:
            exceptions.append(
                {
                    "source_exception_id": f"SE-CH10-{len(exceptions)+1:04d}",
                    "source_version_id": version_id,
                    "source_id": source_id,
                    "attempt_url": row["original_url"],
                    "http_status": row["http_status"],
                    "exception_state": row["exception_state"],
                    "append_only_status": "retained_failure_not_overwritten",
                    "semantic_effect": "relationship_requires_held_verdict_unless_a_lawful_successful_alternate_is_selected",
                }
            )

    selected: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        attempts = attempts_by_source[source["source_id"]]
        if not attempts:
            raise ValueError(f"no normalized attempt: {source['source_id']}")
        successful = [
            row for row in attempts if row["preservation_status"] == "preserved_complete"
        ]
        chosen = sorted(successful or attempts, key=lambda row: row["version_ordinal"])[-1]
        chosen["selected_governing_version"] = "1"
        selected[source["source_id"]] = chosen
        references.append(
            {
                "source_version_reference_id": f"SVR-CH10-{index:04d}",
                "source_id": source["source_id"],
                "source_version_id": chosen["source_version_id"],
                "venue": source["publisher"],
                "publication_date": source["publication_date"],
                "rights_class": chosen["rights_access_boundary"],
                "preservation_state": chosen["preservation_status"],
                "external_projection_reference": (
                    "normalized_source_versions.csv#"
                    + chosen["source_version_id"]
                ),
                "boundary": chosen["semantic_clearance_boundary"],
            }
        )

    derivative_rows = []
    for version in all_versions:
        if version["preservation_status"] != "preserved_complete":
            continue
        path = output / str(version["visible_text_path"])
        derivative_rows.append(
            {
                "source_derivative_id": f"SD-CH10-{len(derivative_rows)+1:04d}",
                "source_version_id": version["source_version_id"],
                "source_id": version["source_id"],
                "derivative_path": version["visible_text_path"],
                "derivative_sha256": sha(path),
                "derivative_bytes": path.stat().st_size,
                "derivative_lines": len(path.read_text(encoding="utf-8").splitlines()),
                "derivation_method": "beautifulsoup_visible_text_or_pypdf_page_marked_text",
                "parent_bytes_sha256": version["content_sha256"],
                "semantic_boundary": "locator_aid_not_independent_source",
            }
        )

    write_csv(output / "normalized_source_versions.csv", all_versions)
    write_csv(output / "source_version_references.csv", references)
    write_csv(output / "source_exceptions.csv", exceptions)
    write_csv(output / "source_derivatives.csv", derivative_rows)
    return all_versions, references, exceptions, selected


LOCATOR_SPECS: dict[str, tuple[str, str, str]] = {
    "EL-CH10-0001": ("completion of project milestones", "supported", "Supports milestone-conditioned TSMC disbursement; no cash receipt is inferred."),
    "EL-CH10-0002": ("completion of project milestones", "supported", "Supports milestone-conditioned Intel award status before later modification."),
    "EL-CH10-0003": ("completion of project milestones", "supported", "Supports milestone-conditioned Micron award status; ceiling is not cash."),
    "EL-CH10-0004": ("completion of project milestones", "supported", "Supports SK hynix award and loan ceilings, not a loan draw."),
    "EL-CH10-0005": ("completion of project milestones", "supported", "Supports Samsung milestone disbursement and stated project perimeter."),
    "EL-CH10-0006": ("up to $6.6 billion in direct funding", "supported", "Supports separate direct-funding and loan ceilings and three-fab perimeter."),
    "EL-CH10-0007": ("irrevocable, absolute and unconditional guarantee", "supported", "Supports TSMC parent guarantee and Arizona loan entitlement; no draw is inferred."),
    "EL-CH10-0008": ("up to $7.865 billion in direct funding", "supported", "Supports the November 2024 ceiling, four-state perimeter, and milestone status."),
    "EL-CH10-0009": ("As of August 27, 2025", "supported", "Supports later modification and legacy accounting boundary through issuer year end."),
    "EL-CH10-0010": ("up to $6.165 billion in direct funding", "supported", "Supports original Idaho/New York final award and milestone status."),
    "EL-CH10-0011": ("total CHIPS Act grants of up to $6.4 billion", "supported", "Supports current issuer-rounded total, unchanged 6.1 billion allocation, and separate Manassas agreement."),
    "EL-CH10-0012": ("Proceeds from government incentives", "supported", "Cash-flow table reports 2,989 million aggregate proceeds; it does not allocate them solely to federal CHIPS or one fab."),
    "EL-CH10-0013": ("up to $4.745 billion in direct funding", "supported", "Supports Samsung ceiling, project value, Texas scope, and milestones."),
    "EL-CH10-0014": ("up to $458 million in direct funding", "supported", "Supports SK hynix grant, loan, project value, and HBM scope."),
    "EL-CH10-0015": ("agreed to make a total of $8,869,800,000", "supported", "Supports exact total and both disbursement components as agreement terms."),
    "EL-CH10-0016": ("$8,869,800,000.00 of disbursements", "supported", "Executed contract supports exchange of disbursements for securities."),
    "EL-CH10-0017": ("up to 433,323,000 shares", "supported", "Supports issued, escrowed, warrant populations, and foundry trigger."),
    "EL-CH10-0018": ("Warrants to purchase 240,516,150 shares", "supported", "Executed contract supports security counts and control terms."),
    "EL-CH10-0019": ("we received the full amount of the accelerated disbursements", "supported", "Supports year-end cash receipt, escrow release, and legacy accounting states."),
    "EL-CH10-0020": ("", "held", "Cited agency report and official alternate both returned access failures; no locator clearance is invented."),
    "EL-CH10-0021": ("433,323,000 8.4%", "supported", "Supports proxy beneficial-ownership population and stated assumptions."),
    "EL-CH10-0022": ("shall apply to property placed in service after December 31, 2025", "supported", "Supports enacted 35-percent amendment and effective placed-in-service boundary."),
    "EL-CH10-0023": ("advanced manufacturing investment credit is equal to 35%", "narrowed", "Supports current form implementation; the instructions also retain legacy 25-percent text and do not verify any taxpayer return."),
    "EL-CH10-0024": ("solely by reason of such", "supported", "Supports coexistence eligibility principle subject to all other section 48D requirements."),
    "EL-CH10-0025": ("tax credits attributable to the U.S. Advanced Manufacturing Investment Credit", "supported", "Supports 2025 and 2024 recognition plus possible cash refund, not cash collected."),
    "EL-CH10-0026": ("In addition to the direct funding of up to $6.6 billion", "supported", "Separates direct award, proposed loan, and stated private investment."),
    "EL-CH10-0027": ("capital-related incentives recognized in 2025", "supported", "Issuer note separates recognized grants, tax credits, equity attribution, and cash states."),
    "EL-CH10-0028": ("inserting ‘‘35 percent’’", "supported", "Statutory rate applies to qualified basis; it is not a cash award headline."),
    "EL-CH10-0029": ("transmission system upgrades have been completed (at xAI expense)", "supported", "Supports customer-funded dedicated upgrades, prevailing tariff, and curtailment."),
    "EL-CH10-0030": ("other customers will only be asked to bear the final 16%", "narrowed", "Supports Entergy's current plan and conditional NPV residual; it is testimony, not a final commission adjudication."),
    "EL-CH10-0031": ("Customer will pay the full cost", "supported", "Supports fifteen-year storage revenue requirement and collateral mechanics."),
    "EL-CH10-0032": ("unrecovered costs shall not be borne by ratepayers", "supported", "Commission order assigns collateral insufficiency tail to DTE rather than other customers."),
    "EL-CH10-0033": ("finalized its investment of up to $240 million", "supported", "Supports public ceiling and total project amount; no disbursement schedule is inferred."),
    "EL-CH10-0034": ("built and operated by CoreWeave", "supported", "Supports CoreWeave operator and Cohere anchor-tenant roles."),
    "EL-CH10-0035": ("", "held", "Both cited PIF pages were access-blocked; ownership and mandate remain version/locator held."),
    "EL-CH10-0036": ("", "held", "Both cited Mubadala page attempts were access-blocked; MGX founding semantics remain held."),
    "EL-CH10-0037": ("initially seek to unlock $30 billion", "supported", "Supports 30-billion equity target and debt-inclusive 100-billion mobilization potential, not funded capital."),
    "EL-CH10-0038": ("enterprise value of approximately $40 billion", "supported", "Official transaction-party alternate supports closing, enterprise value, and separate growth commitment."),
    "EL-CH10-0039": ("initiative to mobilise €200 billion", "narrowed", "Supports mobilization headline and included 20-billion facility; absence of appropriation is a bounded source-scope inference."),
    "EL-CH10-0040": ("advisory and possible financing for AI gigafactories", "narrowed", "Official EIB alternate supports advisory and possible financing for up to five projects; it is not a project loan or cash ledger."),
    "EL-CH10-0041": ("commitment from Evest to fund the full cost", "supported", "Supports customer contracts and separate CIAC versus system-line recovery channels."),
    "EL-CH10-0042": ("broadly-syndicated offerings", "supported", "Executed contract supports transfer limits, voting terms, escrow, and contingent controls."),
    "EL-CH10-0043": ("rights to audit our compliance", "supported", "Issuer filing supports audits, milestone restrictions, reduction, termination, and clawback mechanisms."),
}


def locate_anchor(path: Path, anchor: str) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    target = anchor.casefold()
    found = next(
        (index for index, line in enumerate(lines) if target in line.casefold()),
        None,
    )
    if found is None:
        return "", ""
    start = max(0, found - 4)
    end = min(len(lines), found + 5)
    page = ""
    for prior in reversed(lines[: found + 1]):
        match = re.fullmatch(r"\[\[PDF_PAGE_(\d{4})\]\]", prior)
        if match:
            page = f" PDF p.{int(match.group(1))};"
            break
    locator = (
        f"{path.name}{page} visible-text lines {start + 1}-{end}"
    )
    excerpt = " ".join(lines[start:end])
    return locator, excerpt[:1200]


def build_locator_reviews(
    output: Path,
    selected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    links = read_csv(output / "proposed_evidence_links.csv")
    if set(LOCATOR_SPECS) != {row["evidence_link_id"] for row in links}:
        raise ValueError("locator design does not cover exact evidence-link population")
    locator_by_id: dict[str, dict[str, Any]] = {}
    rows = []
    for index, link in enumerate(links, start=1):
        evidence_id = link["evidence_link_id"]
        anchor, designed_verdict, narrowing = LOCATOR_SPECS[evidence_id]
        version = selected[link["source_id"]]
        derivative_path = (
            output / str(version["visible_text_path"])
            if version["visible_text_path"]
            else None
        )
        resolved = excerpt = ""
        verdict = designed_verdict
        if derivative_path and anchor:
            if evidence_id == "EL-CH10-0039":
                lines = derivative_path.read_text(encoding="utf-8").splitlines()
                resolved = (
                    f"{derivative_path.name} visible-text lines 1-5; "
                    "visible-text lines 73-76"
                )
                excerpt = " ".join(lines[0:5] + lines[72:76])
            else:
                resolved, excerpt = locate_anchor(derivative_path, anchor)
        if (
            version["preservation_status"] != "preserved_complete"
            or not derivative_path
            or (anchor and not resolved)
        ):
            verdict = "held"
            resolved = (
                f"VERSION_OR_LOCATOR_HOLD:{version['source_version_id']};"
                f"cited={link['locator']}"
            )
            excerpt = ""
            if designed_verdict != "held":
                narrowing = (
                    "Preserved version did not yield the designed anchor; "
                    "relationship remains held rather than inferred."
                )
        row = {
            "locator_disposition_id": f"LSD-CH10-{index:04d}",
            "evidence_link_id": evidence_id,
            "claim_id": link["claim_id"],
            "source_id": link["source_id"],
            "source_version_id": version["source_version_id"],
            "cited_locator": link["locator"],
            "resolved_locator": resolved,
            "support_type": link["support_type"],
            "inherited_evidence_grade": next(
                claim["evidence_grade"]
                for claim in read_csv(output / "proposed_atomic_claims.csv")
                if claim["claim_id"] == link["claim_id"]
            ),
            "fresh_packet_semantic_status": (
                f"author_review_{verdict}_pending_uninvolved_acceptance"
            ),
            "semantic_verdict": verdict,
            "required_narrowing": narrowing,
            "review_excerpt_aid": excerpt,
            "preservation_boundary": "bytes_and_derivatives_do_not_self_clear_support",
            "publication_eligibility": "held_pending_fresh_nonauthor_review",
        }
        rows.append(row)
        locator_by_id[evidence_id] = row
    write_csv(output / "locator_semantic_dispositions.csv", rows)
    return rows, locator_by_id


ATOMIC_CHILD_SPECS: list[tuple[str, str, str, str]] = [
    ("CLM-100001", "Final CHIPS award announcements state recipient award ceilings.", "EL-CH10-0001|EL-CH10-0002|EL-CH10-0003|EL-CH10-0004|EL-CH10-0005", "legal_state"),
    ("CLM-100001", "Direct CHIPS funding is released after specified construction, technology, production, or commercial milestones rather than at announcement.", "EL-CH10-0001|EL-CH10-0002|EL-CH10-0003|EL-CH10-0004|EL-CH10-0005", "cash_timing"),
    ("CLM-100002", "TSMC Arizona's final direct-funding ceiling is 6.6 billion dollars.", "EL-CH10-0006", "amount"),
    ("CLM-100002", "TSMC Arizona has access to up to 5 billion dollars of government loans, which is a ceiling rather than a demonstrated draw.", "EL-CH10-0006|EL-CH10-0007", "amount_state"),
    ("CLM-100002", "The TSMC Arizona award perimeter covers three Arizona fabs.", "EL-CH10-0006", "project_perimeter"),
    ("CLM-100002", "TSMC provided an unconditional parent guarantee of its Arizona subsidiary's obligations to Commerce.", "EL-CH10-0007", "control"),
    ("CLM-100003", "Intel's November 2024 commercial CHIPS direct-funding ceiling was 7.865 billion dollars.", "EL-CH10-0008", "amount"),
    ("CLM-100003", "The original Intel award covered milestone-based projects in Arizona, New Mexico, Ohio, and Oregon.", "EL-CH10-0008", "project_perimeter"),
    ("CLM-100003", "Intel's original commercial award was materially modified in August 2025.", "EL-CH10-0009", "supersession"),
    ("CLM-100004", "Micron's current issuer-stated federal direct-funding total is up to 6.4 billion dollars.", "EL-CH10-0011", "amount"),
    ("CLM-100004", "Micron's unchanged 6.1-billion-dollar Idaho and New York program was reallocated to add a second Boise fab.", "EL-CH10-0010|EL-CH10-0011", "project_perimeter"),
    ("CLM-100004", "Micron entered a separate up-to-275-million-dollar Manassas direct-funding agreement.", "EL-CH10-0011", "amount"),
    ("CLM-100005", "Micron's cash-flow table reports 2.989 billion dollars of government-incentive proceeds for the first nine months of fiscal 2026.", "EL-CH10-0012", "cash"),
    ("CLM-100005", "The reported Micron proceeds are aggregate government incentives and are not allocated solely to federal CHIPS or one fab.", "EL-CH10-0012", "attribution_boundary"),
    ("CLM-100006", "Samsung's final direct-funding ceiling is 4.745 billion dollars.", "EL-CH10-0013", "amount"),
    ("CLM-100006", "Samsung stated a more-than-37-billion-dollar Central Texas semiconductor program.", "EL-CH10-0013", "project_perimeter"),
    ("CLM-100006", "Samsung award disbursement is conditioned on project milestones.", "EL-CH10-0013", "cash_timing"),
    ("CLM-100007", "SK hynix's final direct-funding ceiling is 458 million dollars.", "EL-CH10-0014", "amount"),
    ("CLM-100007", "SK hynix has up to 500 million dollars of public loan authority, not a demonstrated draw.", "EL-CH10-0014", "amount_state"),
    ("CLM-100007", "The SK hynix project has a stated approximately 3.87-billion-dollar cost.", "EL-CH10-0014", "project_perimeter"),
    ("CLM-100007", "The SK hynix Indiana perimeter is an HBM packaging and research project.", "EL-CH10-0014", "project_perimeter"),
    ("CLM-100008", "The August 2025 Intel government agreement stated total disbursements of 8.8698 billion dollars.", "EL-CH10-0015|EL-CH10-0016", "amount"),
    ("CLM-100008", "The Intel agreement total includes 5.695 billion dollars of accelerated commercial CHIPS funds.", "EL-CH10-0015|EL-CH10-0016", "amount_component"),
    ("CLM-100008", "The Intel agreement total includes 3.1748 billion dollars of Secure Enclave funds.", "EL-CH10-0015|EL-CH10-0016", "amount_component"),
    ("CLM-100008", "The stated Intel disbursements were exchanged for equity and contingent-value securities.", "EL-CH10-0015|EL-CH10-0016", "legal_form"),
    ("CLM-100009", "Intel agreed to issue up to 433.323 million common shares to Commerce.", "EL-CH10-0017|EL-CH10-0018", "security_population"),
    ("CLM-100009", "The Intel common-share population includes 274.583 million issued shares.", "EL-CH10-0017", "security_population"),
    ("CLM-100009", "The Intel common-share population includes 158.740 million escrowed shares.", "EL-CH10-0017", "security_population"),
    ("CLM-100009", "Intel agreed to warrants for up to 240.51615 million shares.", "EL-CH10-0017|EL-CH10-0018", "security_population"),
    ("CLM-100009", "The Intel warrants become exercisable only if Intel ceases to own at least 51 percent of its foundry business.", "EL-CH10-0017|EL-CH10-0018", "control"),
    ("CLM-100010", "By Intel's 2025 year end it had received the full 5.7 billion dollars of accelerated commercial disbursements.", "EL-CH10-0019|EL-CH10-0020", "cash"),
    ("CLM-100010", "Intel had released approximately 3 million escrow shares after Secure Enclave cash receipts by 2025 year end.", "EL-CH10-0019", "security_state"),
    ("CLM-100010", "Intel retained grant accounting for 2.3 billion dollars received before the August 2025 modification.", "EL-CH10-0019", "accounting"),
    ("CLM-100011", "Intel's proxy reports 433.323 million shares and 8.4 percent beneficial ownership for the United States government.", "EL-CH10-0021", "security_population"),
    ("CLM-100011", "The proxy ownership calculation assumes full release of specified escrow shares.", "EL-CH10-0021", "measurement_assumption"),
    ("CLM-100011", "The proxy ownership calculation excludes foundry-contingent warrants.", "EL-CH10-0021", "measurement_assumption"),
    ("CLM-100012", "Public Law 119-21 changed section 48D's advanced manufacturing investment credit rate from 25 percent to 35 percent.", "EL-CH10-0022|EL-CH10-0023", "tax_rate"),
    ("CLM-100012", "The 35-percent section 48D rate applies to qualified property placed in service after December 31, 2025.", "EL-CH10-0022|EL-CH10-0023", "effective_date"),
    ("CLM-100013", "Specified Commerce CHIPS agreements alone do not disqualify otherwise qualifying expenditures from section 48D qualified investment.", "EL-CH10-0024", "tax_eligibility"),
    ("CLM-100014", "Intel recognized 5.4 billion dollars of advanced manufacturing investment tax credits in 2025.", "EL-CH10-0025", "accounting_amount"),
    ("CLM-100014", "Intel recognized 2.6 billion dollars of advanced manufacturing investment tax credits in 2024.", "EL-CH10-0025", "accounting_amount"),
    ("CLM-100014", "Intel states that advanced manufacturing investment credits may be refunded in cash to the extent they exceed tax liabilities.", "EL-CH10-0025", "cash_option"),
    ("CLM-100015", "Direct-award ceilings, public-loan principal, recognized tax credits, and stated private investment are distinct measures.", "EL-CH10-0026|EL-CH10-0027|EL-CH10-0028", "measurement_rule"),
    ("CLM-100015", "The distinct public-capital measures are not additive without compatible amount states and overlap controls.", "EL-CH10-0026|EL-CH10-0027|EL-CH10-0028", "nonadditivity"),
    ("CLM-100016", "xAI funded the identified Lowry substation and additional grid upgrades at xAI expense rather than MLGW capital cost.", "EL-CH10-0029", "cost_allocation"),
    ("CLM-100016", "xAI pays the prevailing industrial electricity rate.", "EL-CH10-0029", "contract_term"),
    ("CLM-100016", "xAI accepted contractual electricity curtailment.", "EL-CH10-0029", "control"),
    ("CLM-100017", "Entergy expects Evest agreement revenue to recover 100 percent of the system-line revenue requirement during the original term.", "EL-CH10-0030", "cost_recovery"),
    ("CLM-100017", "If no later customer contribution occurs, Entergy's current plan assigns other customers the final 16 percent of estimated project cost on an NPV basis after the original term.", "EL-CH10-0030", "residual_risk"),
    ("CLM-100018", "Green Chile must pay each approved storage project's full fifteen-year revenue requirement.", "EL-CH10-0031|EL-CH10-0032", "cost_recovery"),
    ("CLM-100018", "DTE bears service costs not recovered from Green Chile, Oracle support, or collateral rather than assigning them to other ratepayers.", "EL-CH10-0031|EL-CH10-0032", "residual_risk"),
    ("CLM-100019", "Canada finalized a contribution ceiling of 240 million Canadian dollars for the Cohere compute project.", "EL-CH10-0033", "amount"),
    ("CLM-100019", "The Cohere compute project has a stated total value of 725 million Canadian dollars.", "EL-CH10-0033|EL-CH10-0034", "project_perimeter"),
    ("CLM-100019", "CoreWeave is identified as builder and operator and Cohere as anchor tenant.", "EL-CH10-0034", "party_role"),
    ("CLM-100019", "The public sources do not allocate the remaining 485 million Canadian dollars.", "EL-CH10-0033|EL-CH10-0034", "unknown_boundary"),
    ("CLM-100019", "The public sources do not disclose the public contribution's disbursement schedule.", "EL-CH10-0033|EL-CH10-0034", "unknown_boundary"),
    ("CLM-100020", "PIF announced HUMAIN as a PIF-owned company chaired by the Saudi Crown Prince.", "EL-CH10-0035", "ownership"),
    ("CLM-100020", "HUMAIN's announced mandate spans data centers, cloud, hardware procurement, models, and applications.", "EL-CH10-0035", "operating_scope"),
    ("CLM-100020", "The cited announcement did not disclose initial capitalization or a funded project amount.", "EL-CH10-0035", "unknown_boundary"),
    ("CLM-100021", "AIATC announced the creation of MGX.", "EL-CH10-0036", "legal_state"),
    ("CLM-100021", "Mubadala and G42 were named as MGX founding partners.", "EL-CH10-0036", "party_role"),
    ("CLM-100021", "MGX's announced mandate covers AI infrastructure, semiconductors, and core technologies.", "EL-CH10-0036", "operating_scope"),
    ("CLM-100021", "The cited announcement did not disclose MGX ownership percentages or initial funded capital.", "EL-CH10-0036", "unknown_boundary"),
    ("CLM-100022", "AIP's launch target was 30 billion dollars of equity capital.", "EL-CH10-0037", "target_amount"),
    ("CLM-100022", "AIP stated up to 100 billion dollars of total investment potential including debt.", "EL-CH10-0037", "target_amount"),
    ("CLM-100022", "The 100-billion-dollar figure is a mobilization target rather than a funded pool.", "EL-CH10-0037", "amount_state"),
    ("CLM-100023", "The AIP, MGX, and GIP consortium closed the acquisition of all Aligned Data Centers equity in July 2026.", "EL-CH10-0038", "transaction_state"),
    ("CLM-100023", "The transaction announcement states an approximately 40-billion-dollar enterprise value.", "EL-CH10-0038", "valuation_measure"),
    ("CLM-100023", "The consortium separately committed an additional 5 billion dollars of growth capital.", "EL-CH10-0038", "commitment"),
    ("CLM-100024", "The European Commission announced a 200-billion-euro InvestAI mobilization objective.", "EL-CH10-0039", "target_amount"),
    ("CLM-100024", "The InvestAI headline includes a proposed 20-billion-euro AI Gigafactory facility.", "EL-CH10-0039", "amount_component"),
    ("CLM-100024", "The announcement is not evidence that 200 billion euros was appropriated or disbursed as public cash.", "EL-CH10-0039", "amount_state"),
    ("CLM-100025", "The December 2025 Commission-EIB Group memorandum provides advisory support and a potential financing framework.", "EL-CH10-0040", "framework"),
    ("CLM-100025", "The memorandum addresses potential support for up to five AI Gigafactories.", "EL-CH10-0040", "project_perimeter"),
    ("CLM-100025", "The memorandum is not itself a project loan commitment or cash disbursement.", "EL-CH10-0040", "amount_state"),
    ("CLM-100026", "Public-capital control can operate through milestones, audits, reductions, termination, and clawbacks.", "EL-CH10-0043", "control"),
    ("CLM-100026", "Public-capital control can operate through equity voting, transfer, escrow, and contingent warrant terms.", "EL-CH10-0042", "control"),
    ("CLM-100026", "Public-capital control can operate through customer contracts and utility cost allocation.", "EL-CH10-0041", "control"),
    ("CLM-100026", "The packet's control mechanisms can transmit risk before a conventional credit loss occurs.", "EL-CH10-0041|EL-CH10-0042|EL-CH10-0043", "mechanism_inference"),
]


def build_atomic_children(
    output: Path,
    locator_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = read_csv(output / "proposed_atomic_claims.csv")
    parent_ids = {row["claim_id"] for row in parents}
    if {item[0] for item in ATOMIC_CHILD_SPECS} != parent_ids:
        raise ValueError("atomic child design does not cover exact 26 parents")
    counters: Counter[str] = Counter()
    children = []
    junctions = []
    for parent_id, assertion, evidence_text, assertion_type in ATOMIC_CHILD_SPECS:
        counters[parent_id] += 1
        child_id = f"{parent_id}-A{counters[parent_id]:02d}"
        evidence_ids = evidence_text.split("|")
        verdicts = [locator_by_id[item]["semantic_verdict"] for item in evidence_ids]
        if all(item == "supported" for item in verdicts):
            status = "supported"
        elif all(item == "held" for item in verdicts):
            status = "held"
        else:
            status = "narrowed"
        children.append(
            {
                "child_claim_id": child_id,
                "parent_claim_id": parent_id,
                "child_ordinal": counters[parent_id],
                "atomic_assertion": assertion,
                "assertion_type": assertion_type,
                "fresh_packet_semantic_status": (
                    f"author_review_{status}_pending_uninvolved_acceptance"
                ),
                "canonical_parent_preservation": "parent_text_and_id_retained_exactly",
                "publication_eligibility": "held_pending_fresh_nonauthor_review",
            }
        )
        for evidence_id in evidence_ids:
            locator = locator_by_id[evidence_id]
            junctions.append(
                {
                    "atomic_child_evidence_id": f"ACE-CH10-{len(junctions)+1:04d}",
                    "child_claim_id": child_id,
                    "parent_claim_id": parent_id,
                    "evidence_link_id": evidence_id,
                    "source_id": locator["source_id"],
                    "source_version_id": locator["source_version_id"],
                    "locator_disposition_id": locator["locator_disposition_id"],
                    "semantic_verdict": locator["semantic_verdict"],
                    "mapping_role": "exact_reviewed_locator_for_atomic_assertion",
                }
            )
    coverage = []
    parent_by_id = {row["claim_id"]: row for row in parents}
    for parent_id in sorted(parent_ids):
        parent_children = [
            row for row in children if row["parent_claim_id"] == parent_id
        ]
        coverage.append(
            {
                "parent_child_coverage_id": f"PCC-CH10-{len(coverage)+1:04d}",
                "parent_claim_id": parent_id,
                "parent_claim_text_sha256": hashlib.sha256(
                    parent_by_id[parent_id]["claim_text"].encode("utf-8")
                ).hexdigest(),
                "atomic_child_count": len(parent_children),
                "coverage_status": "lossless_parent_preserved_and_atomic_children_complete",
                "review_boundary": "author_decomposition_pending_uninvolved_acceptance",
            }
        )
    write_csv(output / "atomic_child_claims.csv", children)
    write_csv(output / "atomic_child_evidence_links.csv", junctions)
    write_csv(output / "parent_child_coverage.csv", coverage)
    return children, junctions


PARENT_FALSIFIERS = {
    "CLM-100001": "Recipient-specific award agreements or transaction ledgers show that a stated final ceiling was unconditional cash at announcement rather than milestone-dependent.",
    "CLM-100002": "TSMC Arizona loan statements show a draw above or below the stated 5-billion-dollar ceiling, or executed guarantee documents remove TSMC parent recourse.",
    "CLM-100003": "An executed Commerce amendment shows the November 2024 Intel ceiling remained legally unchanged after August 2025.",
    "CLM-100004": "A later Micron filing or signed amendment changes the 6.1-billion-dollar allocation, 275-million-dollar Manassas agreement, or issuer-rounded 6.4-billion-dollar total.",
    "CLM-100005": "Micron's transaction ledger allocates the 2.989-billion-dollar proceeds entirely to one federal CHIPS award or establishes a different nine-month amount.",
    "CLM-100006": "Samsung's executed award or recipient cash ledger changes the 4.745-billion-dollar ceiling, project perimeter, or milestone condition.",
    "CLM-100007": "SK hynix loan records establish a funded draw or credit-subsidy state different from the separate 458-million grant and 500-million loan ceilings.",
    "CLM-100008": "An executed Intel-Commerce amendment changes the 8.8698-billion-dollar total, its 5.695/3.1748 components, or the securities exchange.",
    "CLM-100009": "Intel's transfer-agent or warrant records change the 274.583-million issued, 158.740-million escrowed, 240.51615-million warrant, or 51-percent trigger terms.",
    "CLM-100010": "Intel's audited cash, escrow, or accounting records change the full 5.7-billion receipt, approximately 3-million release, or 2.3-billion legacy-grant treatment.",
    "CLM-100011": "A later Schedule 13D/G, proxy, or transfer-agent record changes the 433.323-million, 8.4-percent, escrow, or warrant assumptions.",
    "CLM-100012": "A statutory correction or binding tax authority changes the 35-percent rate or post-2025 placed-in-service boundary.",
    "CLM-100013": "Binding IRS guidance or a court decision holds that a specified Commerce agreement alone disqualifies otherwise qualifying section 48D investment.",
    "CLM-100014": "Intel's amended audited filing or tax records change the 5.4-billion/2.6-billion recognition amounts or establish that no excess credit can be refunded.",
    "CLM-100015": "A governing accounting or legal rule establishes a compatible, nonoverlapping basis on which award ceilings, loan principal, recognized credits, and private investment are additive.",
    "CLM-100016": "MLGW invoices, capital records, or the executed service agreement show ordinary MLGW customers funded the dedicated xAI upgrades, or remove prevailing-rate or curtailment terms.",
    "CLM-100017": "A final LPSC order or unredacted Evest agreement assigns the post-term residual back to Evest or establishes a percentage different from the testified final 16-percent NPV share.",
    "CLM-100018": "A final MPSC order, amended contract, or default waterfall assigns unrecovered Green Chile storage costs to other ratepayers rather than DTE.",
    "CLM-100019": "A signed Canada-Cohere-CoreWeave sources-and-uses schedule shows a different public ceiling, total project cost, operator role, disbursement state, or government funding of the residual 485 million.",
    "CLM-100020": "HUMAIN audited standalone statements disclose opening capital or funded projects, or governing records show PIF ownership or the stated mandate is different.",
    "CLM-100021": "MGX constitutional documents or audited statements disclose ownership and initial funded capital, or show different founding partners or investment scope.",
    "CLM-100022": "AIP subscription and funding records establish that 100 billion dollars was committed or funded rather than a debt-inclusive mobilization target.",
    "CLM-100023": "Closing statements or audited consortium records change the 100-percent equity close, approximately 40-billion enterprise value, or separate 5-billion growth commitment.",
    "CLM-100024": "EU budget authority, fund-close documents, or cash ledgers establish 200 billion euros of appropriated or disbursed public money rather than mobilization intent.",
    "CLM-100025": "An EIB credit approval, signed project loan, or disbursement ledger converts the memorandum's possible financing into a binding project commitment or cash payment.",
    "CLM-100026": "Executed agreements and audited loss records show the named milestone, clawback, equity, contract, and utility controls do not operate before conventional credit loss.",
}


def build_falsifiers(
    output: Path,
    children: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = read_csv(output / "proposed_atomic_claims.csv")
    frozen_trigger = {
        "CLM-100001": "FROZEN-CH10-01",
        "CLM-100002": "FROZEN-CH10-01|FROZEN-CH10-02",
        "CLM-100003": "FROZEN-CH10-03",
        "CLM-100004": "FROZEN-CH10-01",
        "CLM-100005": "FROZEN-CH10-01",
        "CLM-100006": "FROZEN-CH10-01",
        "CLM-100007": "FROZEN-CH10-01|FROZEN-CH10-02",
        "CLM-100008": "FROZEN-CH10-03",
        "CLM-100009": "FROZEN-CH10-03",
        "CLM-100010": "FROZEN-CH10-03",
        "CLM-100011": "FROZEN-CH10-03",
        "CLM-100012": "FROZEN-CH10-04",
        "CLM-100013": "FROZEN-CH10-04",
        "CLM-100014": "FROZEN-CH10-04",
        "CLM-100015": "FROZEN-CH10-MEASUREMENT",
        "CLM-100016": "FROZEN-CH10-06",
        "CLM-100017": "FROZEN-CH10-06",
        "CLM-100018": "FROZEN-CH10-06",
        "CLM-100019": "FROZEN-CH10-07",
        "CLM-100020": "FROZEN-CH10-05",
        "CLM-100021": "FROZEN-CH10-05",
        "CLM-100022": "FROZEN-CH10-05",
        "CLM-100023": "FROZEN-CH10-05",
        "CLM-100024": "FROZEN-CH10-05",
        "CLM-100025": "FROZEN-CH10-05",
        "CLM-100026": "FROZEN-CH10-CONTROL",
    }
    parent_rows = [
        {
            "falsifier_id": f"FAL-CH10-{index:03d}",
            "claim_id": claim["claim_id"],
            "frozen_trigger_id": frozen_trigger[claim["claim_id"]],
            "contradictory_document_or_observation": PARENT_FALSIFIERS[
                claim["claim_id"]
            ],
            "deterministic_effect_if_observed": (
                "mark the contradicted child scope superseded, recompute the "
                "parent aggregate and dependent ledgers, and append a dated "
                "confidence-history row without overwriting the canonical parent"
            ),
            "status": "specific_test_not_executed",
        }
        for index, claim in enumerate(parents, start=1)
    ]
    junctions_by_child: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in junctions:
        junctions_by_child[row["child_claim_id"]].append(row)
    effects = {
        "amount": "supersede the child amount and recompute linked nonadditivity controls",
        "amount_component": "change the component only and recompute the parent reconciliation",
        "amount_state": "change legal or cash state without adding incompatible measures",
        "cash": "change the cash state and append confidence history",
        "cash_timing": "change timing status from ceiling to the evidenced cash state",
        "accounting": "change accounting classification while preserving legal history",
        "accounting_amount": "supersede the recognized amount for the specified period",
        "attribution_boundary": "replace the attribution perimeter and recompute only affected program cash",
        "cash_option": "remove or narrow the refund mechanism without changing recognized tax expense",
        "commitment": "supersede the commitment state and preserve enterprise value as a separate measure",
        "contract_term": "replace the contradicted term and recompute the cost-allocation pathway",
        "control": "narrow or remove the specified control mechanism",
        "cost_allocation": "reassign the identified cost payer and recompute the infrastructure pathway",
        "cost_recovery": "replace the recovery term and recompute residual exposure",
        "effective_date": "supersede the effective-date boundary and reclassify affected placed-in-service periods",
        "framework": "replace framework status without classifying it as funded capital",
        "legal_form": "replace the instrument classification and recompute overlapping measure states",
        "legal_state": "supersede the stated legal state while retaining the dated announcement history",
        "measurement_assumption": "replace the measurement assumption and recompute the stated population or percentage",
        "measurement_rule": "replace the rule and rerun all seven nonadditivity controls",
        "mechanism_inference": "remove the unsupported mechanism inference while preserving the observed source facts",
        "nonadditivity": "rerun all dependent reconciliations and permit addition only if state and overlap tests pass",
        "operating_scope": "narrow the named operating perimeter only",
        "ownership": "replace the ownership state and recursively recheck entity and alias closure",
        "party_role": "replace the named party role and recheck the affected entity junction",
        "project_perimeter": "replace the stated project perimeter without changing unrelated amount states",
        "residual_risk": "reassign the residual holder and recompute the cost-allocation pathway",
        "security_population": "supersede only the contradicted security population and recompute ownership",
        "security_state": "replace the release or escrow state and recompute beneficial ownership",
        "supersession": "restore or replace the prior legal state and append a dated supersession record",
        "target_amount": "replace the target only; do not reclassify it as funded without proof",
        "tax_eligibility": "remove the eligibility statement for the contradicted expenditure perimeter",
        "tax_rate": "replace the statutory rate and recompute only qualifying-basis tax measures",
        "transaction_state": "replace closing status and preserve announced terms as historical only",
        "unknown_boundary": "populate only the newly disclosed field and retain all other unknown holds",
        "valuation_measure": "replace enterprise value only; do not infer purchase-price cash or partner funding",
    }
    contradictory_tests = {
        "amount": "An executed amendment or audited amount schedule with the same currency, date, and perimeter reports a different numeric value than",
        "amount_component": "An executed agreement allocation schedule reports a different component or total decomposition than",
        "amount_state": "A dated obligation, draw, repayment, or cash ledger assigns a different funded-versus-ceiling state than",
        "attribution_boundary": "A program-level cash rollforward allocates the aggregate proceeds contrary to",
        "cash": "An audited cash-flow note or agency disbursement ledger reports no receipt, a different amount, or a different period than",
        "cash_option": "Binding tax guidance or an audited tax note eliminates the stated cash-refund mechanism in",
        "cash_timing": "An executed disbursement schedule or dated cash ledger shows payment at announcement or on conditions different from",
        "commitment": "A signed financing commitment or closing statement reports no commitment or a different committed amount than",
        "contract_term": "A filed executed service agreement states a different tariff, term, or curtailment obligation than",
        "control": "An executed agreement amendment removes or changes the guarantee, trigger, audit, escrow, clawback, or voting control stated in",
        "cost_allocation": "A utility invoice, contribution agreement, or commission order assigns the dedicated capital cost to a different payer than",
        "cost_recovery": "A final commission order or executed service agreement states a different recovery term, percentage, or payer than",
        "effective_date": "A later statute, correction, or controlling tax guidance sets a different effective or placed-in-service boundary than",
        "framework": "A final financing contract either creates funded project capital or terminates the advisory framework contrary to",
        "legal_form": "An executed instrument or issuer accounting note classifies the exchange as a different legal form than",
        "legal_state": "An official rescission, executed agreement, or final order establishes that the stated award or vehicle never reached the legal state in",
        "measurement_assumption": "A proxy amendment or cap-table reconciliation uses assumptions different from",
        "measurement_rule": "A controlling accounting, legal, or measurement rule permits aggregation of the exact states separated by",
        "mechanism_inference": "A contract or technical filing directly attributes a mechanism that contradicts",
        "nonadditivity": "A reconciled transaction ledger proves matched perimeter, timing, instrument, and overlap states that permit addition contrary to",
        "operating_scope": "A charter, audited segment note, or official mandate excludes a named operating function in",
        "ownership": "A charter, cap table, or audited ownership note reports a different controller or ownership state than",
        "party_role": "An executed partnership, lease, or acquisition agreement assigns the named party a different role than",
        "project_perimeter": "A final award schedule, executed amendment, or audited project note identifies a different site, facility count, or project perimeter than",
        "residual_risk": "A final commission order or executed contract assigns the post-term or unrecovered residual to a different party than",
        "security_population": "A transfer-agent ledger, executed securities amendment, or proxy correction reports a different share or warrant population than",
        "security_state": "A transfer-agent or escrow-agent ledger reports a different released-versus-escrowed state than",
        "supersession": "A later executed amendment or issuer filing establishes that the prior award remained unmodified contrary to",
        "target_amount": "A governing investment mandate or audited capital statement reports a different target or proves funded capital contrary to",
        "tax_eligibility": "Controlling IRS guidance or a final tax determination disqualifies the specified Commerce-agreement expenditures contrary to",
        "tax_rate": "A later statute or controlling correction sets a rate other than the one in",
        "transaction_state": "A termination notice, court filing, or transaction-party statement reports that the acquisition did not close as stated in",
        "unknown_boundary": "A cap table, funding schedule, or disbursement ledger supplies the exact field described as undisclosed in",
        "valuation_measure": "A signed closing statement or transaction-party correction reports a different enterprise-value measure than",
    }
    child_rows = []
    for child in children:
        edges = junctions_by_child[child["child_claim_id"]]
        effect = effects.get(
            child["assertion_type"],
            "narrow or supersede only this child and append a dated review record",
        )
        contradictory_test = contradictory_tests.get(
            child["assertion_type"],
            "A controlling primary document establishes the contrary of",
        )
        source_ids = "|".join(
            sorted({edge["source_id"] for edge in edges})
        )
        child_rows.append(
            {
                "atomic_falsifier_id": f"AF-CH10-{len(child_rows)+1:04d}",
                "child_claim_id": child["child_claim_id"],
                "parent_claim_id": child["parent_claim_id"],
                "contradictory_document_or_observation": (
                    f"{contradictory_test}: {child['atomic_assertion']} "
                    f"[reviewed source perimeter {source_ids}]"
                ),
                "deterministic_effect_if_observed": effect,
                "status": "assertion_specific_test_not_executed",
            }
        )
    write_csv(output / "falsifiers.csv", parent_rows)
    write_csv(output / "atomic_child_falsifiers.csv", child_rows)
    return parent_rows, child_rows


def build_external_claim_closure(
    output: Path,
    locator_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    claim = connection.execute(
        "SELECT * FROM claims WHERE claim_id='CLM-050004'"
    ).fetchone()
    link = connection.execute(
        "SELECT * FROM evidence_links WHERE claim_id='CLM-050004'"
    ).fetchone()
    connection.close()
    if claim is None or link is None:
        raise ValueError("external Chapter 5 dependency missing from canonical database")
    normalized_claim = normalize_db_row(claim)
    selected_locator = locator_by_id["EL-CH10-0041"]
    external_claims = [
        {
            "external_claim_dependency_id": "ECD-CH10-0001",
            **normalized_claim,
            "dependency_role": "governed_external_claim_for_PIC-100004_and_DEP-CH10-0023",
            "packet_local_status": "included_exact_canonical_external_claim",
            "publication_eligibility": "held_pending_fresh_nonauthor_review",
        }
    ]
    external_links = [
        {
            "external_claim_evidence_id": "ECE-CH10-0001",
            "external_claim_dependency_id": "ECD-CH10-0001",
            "claim_id": "CLM-050004",
            "source_id": str(link["source_id"]),
            "source_version_id": selected_locator["source_version_id"],
            "canonical_locator": str(link["locator"]),
            "resolved_locator": selected_locator["resolved_locator"],
            "support_type": str(link["support_type"]),
            "excerpt_note": str(link["excerpt_note"]),
            "semantic_verdict": selected_locator["semantic_verdict"],
            "semantic_boundary": (
                "Chapter-10 author review reuses the preserved SRC-0087 version; "
                "fresh uninvolved acceptance remains required"
            ),
        }
    ]
    junctions = [
        {
            "record_claim_junction_id": "RCJ-CH10-0001",
            "record_id": "PIC-100004",
            "dataset": "public_infrastructure_cost_allocation",
            "claim_id": "CLM-050004",
            "claim_scope": "external_claim_dependency",
            "external_claim_dependency_id": "ECD-CH10-0001",
            "closure_status": "PASS_PACKET_LOCAL_SOURCE_CHAIN_INCLUDED",
        },
        {
            "record_claim_junction_id": "RCJ-CH10-0002",
            "record_id": "DEP-CH10-0023",
            "dataset": "dependency_map",
            "claim_id": "CLM-050004",
            "claim_scope": "external_claim_dependency",
            "external_claim_dependency_id": "ECD-CH10-0001",
            "closure_status": "PASS_PACKET_LOCAL_SOURCE_CHAIN_INCLUDED",
        },
    ]
    write_csv(output / "external_claim_dependencies.csv", external_claims)
    write_csv(output / "external_claim_evidence_links.csv", external_links)
    write_csv(output / "record_claim_junctions.csv", junctions)
    return external_claims, external_links


def build_timeline(output: Path) -> list[dict[str, Any]]:
    legacy = read_csv(V1 / "timeline.csv")
    evidence = read_csv(output / "proposed_evidence_links.csv")
    primary_by_claim: dict[str, dict[str, str]] = {}
    for row in evidence:
        primary_by_claim.setdefault(row["claim_id"], row)
        if row["support_type"] == "direct":
            primary_by_claim[row["claim_id"]] = row
    rows = []
    for row in legacy:
        link = primary_by_claim[row["claim_id"]]
        timeline_class = (
            "sovereign_measurement"
            if row["record_id"].startswith("SOV-")
            else "award_legal_state"
        )
        rows.append(
            {
                **row,
                "timeline_class": timeline_class,
                "legal_state": row["event_state"],
                "cash_state": "not_established_by_event",
                "accounting_state": "not_applicable_or_not_established",
                "source_id": link["source_id"],
                "evidence_link_id": link["evidence_link_id"],
                "date_precision": (
                    "month" if len(row["event_date"]) == 7 else "day"
                ),
                "nonadditivity_boundary": row["amount_state"],
            }
        )
    additions = [
        ("2024-10-22", "TAX-48D-GUIDANCE", "tax_guidance_issued", "tax_legal_transition", "CLM-100013", "SRC-0150", "EL-CH10-0024", "guidance_effective", "no_cash_established", "eligibility_principle_only"),
        ("2025-07-04", "TAX-48D-RATE", "statute_enacted", "tax_legal_transition", "CLM-100012", "SRC-0148", "EL-CH10-0022", "enacted_rate_change", "no_cash_established", "statutory_rate_not_recognition"),
        ("2026-01-01", "TAX-48D-RATE", "placed_in_service_effective_boundary", "tax_effective_transition", "CLM-100012", "SRC-0148", "EL-CH10-0022", "35_percent_for_qualifying_property_after_2025", "no_cash_established", "qualified_basis_and_taxpayer_election_still_required"),
        ("2025-08-22", "PEQ-100001", "intel_securities_agreement_executed", "equity_legal_transition", "CLM-100008", "SRC-0142", "EL-CH10-0016", "executed_agreement", "disbursements_contractual_not_yet_all_cash", "equity_exchange_terms"),
        ("2025-08-25", "PEQ-100001", "intel_form_8k_filed", "equity_disclosure_transition", "CLM-100009", "SRC-0141", "EL-CH10-0017", "issuer_filing", "cash_timing_conditional", "initial_terms_not_year_end_accounting"),
        ("2025-08-26", "PEQ-100001", "contractual_closing_date_scheduled", "equity_closing_transition", "CLM-100008", "SRC-0141", "EL-CH10-0015", "scheduled_or_later_by_mutual_agreement", "cash_only_as_received", "not_independent_proof_of_actual_close_date"),
        ("2025-12-27", "PCA-100003", "intel_full_accelerated_cash_measurement", "cash_measurement", "CLM-100010", "SRC-0143", "EL-CH10-0019", "issuer_year_end_measurement", "5.7_billion_received", "post_modification_equity_attribution"),
        ("2025-12-27", "PEQ-100002", "intel_escrow_release_measurement", "equity_measurement", "CLM-100010", "SRC-0143", "EL-CH10-0019", "issuer_year_end_measurement", "secure_enclave_cash_received_amount_not_fully_stated_here", "approximately_3_million_shares_released"),
        ("2025-12-27", "PCR-100003", "intel_legacy_grant_accounting_measurement", "accounting_measurement", "CLM-100010", "SRC-0143", "EL-CH10-0019", "issuer_year_end_measurement", "2.3_billion_pre_modification_receipts", "legacy_grant_accounting_retained"),
        ("2026-03-20", "PEQ-100004", "intel_proxy_ownership_measurement", "equity_measurement", "CLM-100011", "SRC-0144", "EL-CH10-0021", "proxy_measurement", "not_cash", "escrow_assumed_warrants_excluded"),
        ("2025-10-31", "PIC-100003", "dte_special_contract_application_and_testimony", "infrastructure_filing", "CLM-100018", "SRC-0097", "EL-CH10-0031", "application_and_testimony", "customer_payment_formula_not_cash_ledger", "proposed_contract_state"),
        ("2025-12-18", "PIC-100003", "mpsc_conditional_approval_order", "infrastructure_order", "CLM-100018", "SRC-0098", "EL-CH10-0032", "conditional_commission_order", "no_specific_cash_balance", "DTE_residual_after_collateral"),
        ("2026-07-02", "PIC-100002", "entergy_public_redacted_testimony", "infrastructure_filing", "CLM-100017", "SRC-0087", "EL-CH10-0030", "utility_testimony_current_plan", "no_project_cash_balance", "not_final_commission_adjudication"),
        ("2025-02-11", "SOV-100005", "investai_mobilization_launch", "sovereign_policy_transition", "CLM-100024", "SRC-0155", "EL-CH10-0039", "mobilization_objective", "not_appropriated_or_disbursed_cash", "20_billion_facility_inside_200_billion_headline"),
    ]
    for event in additions:
        (
            date,
            record_id,
            state,
            timeline_class,
            claim_id,
            source_id,
            evidence_id,
            legal_state,
            cash_state,
            accounting_state,
        ) = event
        rows.append(
            {
                "timeline_id": f"TL-CH10-{len(rows)+1:04d}",
                "event_date": date,
                "record_id": record_id,
                "event_state": state,
                "amount_state": "legal_cash_accounting_states_not_additive",
                "claim_id": claim_id,
                "timeline_class": timeline_class,
                "legal_state": legal_state,
                "cash_state": cash_state,
                "accounting_state": accounting_state,
                "source_id": source_id,
                "evidence_link_id": evidence_id,
                "date_precision": "day",
                "nonadditivity_boundary": "event_records_one_transition_not_an_additive_amount",
            }
        )
    write_csv(output / "timeline.csv", rows)
    return rows


def build_review_status(
    output: Path,
    locators: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = read_csv(output / "proposed_atomic_claims.csv")
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in locators:
        by_claim[row["claim_id"]].append(row)
    statuses = []
    history = []
    for index, claim in enumerate(claims, start=1):
        verdicts = [row["semantic_verdict"] for row in by_claim[claim["claim_id"]]]
        if all(item == "supported" for item in verdicts):
            aggregate = "supported"
        elif all(item == "held" for item in verdicts):
            aggregate = "held"
        else:
            aggregate = "narrowed_or_mixed"
        statuses.append(
            {
                "claim_review_status_id": f"CRS-CH10-{index:04d}",
                "claim_id": claim["claim_id"],
                "inherited_canonical_evidence_grade": claim["evidence_grade"],
                "inherited_canonical_confidence": claim["confidence"],
                "inherited_canonical_review_status": claim["review_status"],
                "source_preservation_status": (
                    "all_selected_versions_preserved"
                    if all(
                        not row["resolved_locator"].startswith(
                            "VERSION_OR_LOCATOR_HOLD"
                        )
                        for row in by_claim[claim["claim_id"]]
                    )
                    else "one_or_more_selected_versions_held"
                ),
                "fresh_packet_semantic_status": (
                    f"author_review_{aggregate}_pending_uninvolved_acceptance"
                ),
                "fresh_relationship_count": len(verdicts),
                "fresh_supported_count": verdicts.count("supported"),
                "fresh_narrowed_count": verdicts.count("narrowed"),
                "fresh_held_count": verdicts.count("held"),
                "publication_eligibility": "NOT_ELIGIBLE_AUTHOR_PACKET_ONLY",
                "status_boundary": "inherited_grade_and_confidence_are_not_fresh_locator_clearance",
            }
        )
        history.append(
            {
                "fresh_review_history_id": f"FRH-CH10-{index:04d}",
                "claim_id": claim["claim_id"],
                "review_date": "2026-07-26",
                "review_actor": "rebuilt_v2_author",
                "inherited_history_reference": next(
                    row["history_id"]
                    for row in read_csv(
                        output / "proposed_confidence_history.csv"
                    )
                    if row["claim_id"] == claim["claim_id"]
                ),
                "fresh_packet_semantic_status": (
                    f"author_review_{aggregate}_pending_uninvolved_acceptance"
                ),
                "grade_change_authorized": "0",
                "confidence_change_authorized": "0",
                "publication_authorized": "0",
                "note": "separate author semantic pass; canonical history remains exact and unchanged",
            }
        )
    write_csv(output / "claim_review_status.csv", statuses)
    write_csv(output / "fresh_semantic_review_history.csv", history)
    return statuses, history


GOVERNED_HOLD_FIELDS = {
    "archived_path",
    "content_sha256",
    "source_version_id",
    "external_projection_reference",
    "resolved_locator",
    "semantic_verdict",
    "committed_amount",
    "disbursed_amount",
    "recognized_amount",
    "qualified_basis",
    "cash_collected_or_offset",
    "cash_consideration",
    "beneficial_ownership_percentage",
    "customer_contribution",
    "public_or_ratepayer_residual",
    "disclosed_cost",
    "funded_amount",
    "effective_date",
    "measurement_date",
    "preliminary_amount",
    "final_award_ceiling",
    "headline_or_target_amount",
}


def build_typed_inventory(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    legacy_rows = read_csv(V1 / "typed_row_dispositions.csv")
    legacy_fields = read_csv(V1 / "typed_field_dispositions.csv")
    typed_rows = [
        {
            **row,
            "source_file": f"source_inventory/{row['source_file']}",
            "row_disposition": "preserved_rebuilt_v1_source_inventory_row",
            "publication_status": "held_author_v2_inventory_not_publication",
        }
        for row in legacy_rows
    ]
    typed_fields = []
    for row in legacy_fields:
        current = dict(row)
        current["source_file"] = f"source_inventory/{row['source_file']}"
        if row["source_value"]:
            current["blank_decision"] = "not_blank"
            current["narrowing_or_hold"] = "populated_value_preserved_exactly"
        else:
            original_name = row["source_file"]
            critical = (
                (original_name == "source_version_references.csv" and row["field"] in {"source_version_id", "external_projection_reference"})
                or (original_name == "locator_semantic_dispositions.csv" and row["field"] in {"resolved_locator", "semantic_verdict"})
            )
            contextual_hold = critical or row["field"] in GOVERNED_HOLD_FIELDS
            if contextual_hold:
                current["field_disposition"] = "governed_hold"
                current["destination"] = ""
                current["blank_decision"] = (
                    "governed_hold_missing_unreviewed_undisclosed_or_not_yet_measured"
                )
                current["narrowing_or_hold"] = (
                    f"governed_hold:{row['record_id']}:{row['field']}:"
                    "blank_preserved_not_zero_false_verified_or_inapplicable"
                )
            else:
                current["blank_decision"] = (
                    "explicit_structural_blank_or_not_applicable_for_this_row"
                )
                current["narrowing_or_hold"] = (
                    f"explicit_row_decision:{row['record_id']}:{row['field']}:"
                    "blank_preserved_not_zero_false_or_verified"
                )
        typed_fields.append(current)
    write_csv(output / "typed_row_dispositions.csv", typed_rows)
    write_csv(output / "typed_field_dispositions.csv", typed_fields)
    return typed_rows, typed_fields


def build_html(
    output: Path,
    children: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> None:
    claims = read_csv(output / "proposed_atomic_claims.csv")
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for child in children:
        children_by_parent[child["parent_claim_id"]].append(child)
    status_by_claim = {row["claim_id"]: row for row in statuses}
    items = []
    for claim in claims:
        status = status_by_claim[claim["claim_id"]]
        child_html = "".join(
            "<li><code>"
            + html.escape(child["child_claim_id"])
            + "</code> "
            + html.escape(child["atomic_assertion"])
            + " <em>"
            + html.escape(child["fresh_packet_semantic_status"])
            + "</em></li>"
            for child in children_by_parent[claim["claim_id"]]
        )
        items.append(
            f"<section id='{html.escape(claim['claim_id'])}'>"
            f"<h3>{html.escape(claim['claim_id'])}</h3>"
            f"<p>{html.escape(claim['claim_text'])}</p>"
            "<p class='status'><strong>Inherited canonical status only:</strong> "
            f"grade {html.escape(claim['evidence_grade'])}; confidence "
            f"{html.escape(claim['confidence'])}; review status "
            f"{html.escape(claim['review_status'])}. "
            "<strong>Fresh packet semantic review:</strong> "
            f"{html.escape(status['fresh_packet_semantic_status'])}; "
            "<strong>publication:</strong> NOT ELIGIBLE FOR PUBLICATION.</p>"
            f"<ol>{child_html}</ol></section>"
        )
    tables = []
    for filename in [
        "public_capital_awards.csv",
        "public_tax_incentives.csv",
        "public_equity_positions.csv",
        "public_infrastructure_cost_allocation.csv",
        "sovereign_capital_cases.csv",
        "public_capital_measurement_reconciliations.csv",
    ]:
        rows = read_csv(output / filename)
        headers = list(rows[0])
        tables.append(
            f"<h2>{html.escape(filename.removesuffix('.csv').replace('_', ' ').title())}</h2>"
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(name)}</th>" for name in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(row[name])}</td>" for name in headers
                )
                + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chapter 10 rebuilt-v2 proposed packet</title>
<style>body{{font-family:Georgia,serif;max-width:1180px;margin:36px auto;line-height:1.45;color:#172033}}h1,h2,h3{{font-family:Arial,sans-serif}}table{{border-collapse:collapse;width:100%;font:11px Arial,sans-serif}}th,td{{border:1px solid #cbd5e1;padding:5px;vertical-align:top}}.hold{{border-left:5px solid #b45309;background:#fff7ed;padding:12px}}.status{{background:#f1f5f9;padding:8px}}code{{font-size:11px}}</style></head><body>
<h1>Chapter 10 — Public Capital, Tax Expenditure, and Sovereign Finance</h1>
<p class="hold"><strong>Author boundary:</strong> corrected staging packet only. It preserves exactly 26 canonical parent claims and separately labels inherited grade/confidence versus fresh author semantic review. Preserved bytes do not self-clear support; three source identities remain access-held; every relationship still requires fresh uninvolved acceptance. This packet creates no canonical, DOCX, publication, or substantive-acceptance authority. Announced, committed, funded, drawn, spent, recognized, guaranteed, contingent, and tax-offset amounts remain distinct and nonadditive.</p>
<h2>Canonical parent claims and atomic children</h2>
{''.join(items)}
<h2>Packet-local external support closure</h2>
<p>PIC-100004 and DEP-CH10-0023 resolve through governed external dependency CLM-050004, its canonical SRC-0087 relationship, the selected immutable source version, and a reviewed locator disposition. This inclusion does not convert the Chapter 5 claim into a Chapter 10 canonical parent.</p>
{''.join(tables)}
<h2>Interpretation boundary</h2>
<p>The packet does not infer that an announcement is committed, a commitment is funded, a facility ceiling is drawn, a tax incentive is cash received, public support is first-loss capital, or national policy is enforceable project finance.</p>
</body></html>"""
    (output / "proposed_chapter.html").write_text(document, encoding="utf-8")


def build_governing_inventory(output: Path) -> list[dict[str, Any]]:
    paths = [
        V2_AUTHOR / "lane_manifest.json",
        V2_REVIEW / "lane_manifest.json",
        V2_REVIEW / "review_report.md",
        V2_REVIEW / "patch_list.csv",
        V1 / "lane_manifest.json",
        REVIEW / "lane_manifest.json",
        REVIEW / "review_report.md",
        REVIEW / "patch_list.csv",
        REVIEW / "semantic_findings.csv",
        DB,
    ]
    for row in read_csv(V1 / "governing_input_inventory.csv"):
        path = ROOT / row["path"]
        if path.is_file():
            paths.append(path)
    unique = list(dict.fromkeys(paths))
    rows = [
        {
            "input_inventory_id": f"GIN-CH10-V3-{index:04d}",
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "authority": (
                "protected_author_or_review_input"
                if (
                    path.is_relative_to(V1)
                    or path.is_relative_to(REVIEW)
                    or path.is_relative_to(V2_AUTHOR)
                    or path.is_relative_to(V2_REVIEW)
                )
                else "governing_read_only_input"
            ),
        }
        for index, path in enumerate(unique, start=1)
    ]
    write_csv(output / "governing_input_inventory.csv", rows)
    return rows


def build(output: Path) -> dict[str, Any]:
    verify_protected_inputs()
    prepare_output(output)
    source_files = copy_lossless_source_inventory(output)
    shutil.copy2(V1 / "arithmetic_nonadditivity_controls.csv", output / "arithmetic_nonadditivity_controls.csv")
    shutil.copy2(V1 / "contradiction_log.csv", output / "contradiction_log.csv")
    shutil.copy2(V1 / "dependency_map.csv", output / "dependency_map.csv")

    entity_rows, aliases = build_entities(output)
    versions, references, exceptions, selected = build_source_versions(output)
    locators, locator_by_id = build_locator_reviews(output, selected)
    external_claims, external_links = build_external_claim_closure(
        output,
        locator_by_id,
    )
    children, child_links = build_atomic_children(output, locator_by_id)
    parent_falsifiers, child_falsifiers = build_falsifiers(
        output,
        children,
        child_links,
    )
    timeline = build_timeline(output)
    statuses, fresh_history = build_review_status(output, locators)
    typed_rows, typed_fields = build_typed_inventory(output)
    governing = build_governing_inventory(output)
    build_html(output, children, statuses)

    semantic_counts = Counter(row["semantic_verdict"] for row in locators)
    field_counts = Counter(row["field_disposition"] for row in typed_fields)
    counts: dict[str, Any] = {
        "canonical_parent_claims": len(read_csv(output / "proposed_atomic_claims.csv")),
        "canonical_evidence_links": len(read_csv(output / "proposed_evidence_links.csv")),
        "canonical_source_identities": len(read_csv(output / "proposed_sources.csv")),
        "legacy_source_inventory_files": len(source_files),
        "legacy_source_inventory_rows": len(typed_rows),
        "legacy_source_inventory_scalars": len(typed_fields),
        "legacy_blank_scalars": sum(not row["source_value"] for row in typed_fields),
        "legacy_critical_governed_holds": sum(
            row["field_disposition"] == "governed_hold"
            and (
                (
                    row["source_file"].endswith("source_version_references.csv")
                    and row["field"]
                    in {"source_version_id", "external_projection_reference"}
                )
                or (
                    row["source_file"].endswith(
                        "locator_semantic_dispositions.csv"
                    )
                    and row["field"]
                    in {"resolved_locator", "semantic_verdict"}
                )
            )
            for row in typed_fields
        ),
        "field_dispositions": dict(sorted(field_counts.items())),
        "normalized_version_attempts": len(versions),
        "selected_source_versions": len(references),
        "selected_preserved_versions": sum(
            row["preservation_state"] == "preserved_complete"
            for row in references
        ),
        "selected_version_holds": sum(
            row["preservation_state"] != "preserved_complete"
            for row in references
        ),
        "append_only_source_exceptions": len(exceptions),
        "semantic_relationships": len(locators),
        "semantic_verdicts": dict(sorted(semantic_counts.items())),
        "recursive_entities": len(entity_rows),
        "entity_aliases": len(aliases),
        "unresolved_alias_holds": sum(
            row["resolution_status"] == "governed_unresolved_hold"
            for row in aliases
        ),
        "external_claim_dependencies": len(external_claims),
        "external_claim_evidence_links": len(external_links),
        "atomic_children": len(children),
        "atomic_child_evidence_links": len(child_links),
        "parent_falsifiers": len(parent_falsifiers),
        "atomic_child_falsifiers": len(child_falsifiers),
        "timeline_events": len(timeline),
        "claim_review_status_rows": len(statuses),
        "fresh_semantic_history_rows": len(fresh_history),
        "governing_inputs": len(governing),
        "legacy_source_inventory_tree_sha256": tree_digest(
            output / "source_inventory"
        ),
    }
    write_json(output / "build_counts.json", counts)
    report = f"""# Chapter 10 rebuilt-v3 narrow locator correction

## Outcome

This isolated candidate applies only CH10-V2-P001 to the sealed rebuilt-v2 packet
without editing canonical/shared data, the canonical chapter, DOCX, or publication artifacts. It
preserves the exact 26 canonical parent claims and the rebuilt-v1 411-row /
4,188-scalar source inventory, while keeping all author conclusions subject to
fresh uninvolved review.

`EL-CH10-0039` now has one composite exact locator: visible-text lines 1-5
preserve the 200-billion-euro title, and lines 73-76 preserve the substantive
statement that the initiative includes a 20-billion-euro AI-gigafactory fund.
No claim text, source identity, grade, amount state, semantic verdict, or
authority was changed.

## Exact populations

- canonical parents: {counts['canonical_parent_claims']}
- canonical Chapter 10 evidence links: {counts['canonical_evidence_links']}
- canonical Chapter 10 source identities: {counts['canonical_source_identities']}
- lossless legacy source rows/scalars: {counts['legacy_source_inventory_rows']} / {counts['legacy_source_inventory_scalars']}
- legacy blank scalars individually dispositioned: {counts['legacy_blank_scalars']}
- critical legacy version/projection/locator/verdict holds: {counts['legacy_critical_governed_holds']}
- normalized source attempts: {counts['normalized_version_attempts']} ({counts['selected_preserved_versions']} selected preserved; {counts['selected_version_holds']} selected access-held)
- semantic relationships: {counts['semantic_relationships']} with verdicts {json.dumps(counts['semantic_verdicts'], sort_keys=True)}
- recursive canonical entities / aliases / unresolved alias holds: {counts['recursive_entities']} / {counts['entity_aliases']} / {counts['unresolved_alias_holds']}
- governed external claims: {counts['external_claim_dependencies']}
- atomic children / child-evidence edges: {counts['atomic_children']} / {counts['atomic_child_evidence_links']}
- expanded timeline events: {counts['timeline_events']}

## Authority boundary

Preservation is not semantic clearance. Author semantic review is not uninvolved
acceptance. Inherited canonical A/verified/reviewed values are retained exactly but
are separately labeled and are not publication eligibility. Source access failures
and unresolved aliases remain append-only governed holds. This packet creates no
canonical, DOCX, chapter-replacement, substantive-acceptance, or publication authority.
"""
    (output / "author_report.md").write_text(report, encoding="utf-8")

    mechanically_dependent = {
        "author_report.md": "v3 correction report",
        "build_counts.json": "governing input population changed",
        "build_packet.py": "locator-only v3 builder",
        "governing_input_inventory.csv": "v2 author and review manifests added",
        "locator_semantic_dispositions.csv": "EL-CH10-0039 locator and excerpt corrected",
        "seal_manifest.py": "v3 manifest metadata",
        "validate_packet.py": "exact replay and v2 freeze validation",
        "verify_clean_builds.py": "v2 author and review protected targets added",
    }
    runtime = {
        "deterministic_build_results.json",
        "generic_validation_results.json",
        "lane_manifest.json",
        "manifest_verification.json",
        "no_write_audit.csv",
        "ruff_results.json",
        "two_clean_build_comparison.csv",
        "validation_results.json",
    }
    v2_manifest = json.loads(
        (V2_AUTHOR / "lane_manifest.json").read_text(encoding="utf-8")
    )
    diff_rows = []
    for entry in v2_manifest["files"]:
        relative_path = entry["path"]
        current_path = output / relative_path
        if relative_path in runtime:
            current_sha = ""
            classification = "runtime_recomputed_after_clean_build"
            reason = "excluded_from_intrinsic_v3_freeze_comparison"
        else:
            current_sha = sha(current_path) if current_path.is_file() else ""
        if relative_path in mechanically_dependent:
            classification = "authorized_mechanical_change"
            reason = mechanically_dependent[relative_path]
        elif relative_path not in runtime:
            classification = (
                "byte_identical_frozen"
                if current_sha == entry["sha256"]
                else "UNAUTHORIZED_DRIFT"
            )
            reason = "unrelated_v2_artifact_must_remain_byte_identical"
        diff_rows.append(
            {
                "path": relative_path,
                "v2_sha256": entry["sha256"],
                "v3_sha256": current_sha,
                "classification": classification,
                "reason": reason,
            }
        )
    write_csv(output / "v2_artifact_diff.csv", diff_rows)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
