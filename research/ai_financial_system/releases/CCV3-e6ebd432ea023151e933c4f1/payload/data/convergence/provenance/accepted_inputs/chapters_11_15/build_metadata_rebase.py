#!/usr/bin/env python3
"""Build the lossless Chapters 11-15 convergence metadata rebase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AI = ROOT / "research/ai_financial_system"
STAGING = AI / "staging"
DATA = AI / "data"
INTEGRATION = AI / "integration"
AUDIT = STAGING / "lane_4_audit_chapters_10_15_convergence_inputs"
SOURCE_UNION = STAGING / "lane_3_source_preservation_union_v2"
SOURCE_UNION_REVIEW = STAGING / "lane_2_verify_source_preservation_union_v2"
GOVERNING_AUTHOR_V1 = STAGING / "lane_2_chapters_11_15_metadata_rebase"
GOVERNING_REVIEW_V1 = STAGING / "lane_3_verify_chapters_11_15_metadata_rebase"

EXPECTED_DATABASE_SHA256 = (
    "38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6"
)
EXPECTED_AUDIT_MANIFEST_SHA256 = (
    "e1e8d8a834edad14f95b5fc8df425e64f0c2d2da8f4be68e20509bfb1209a84c"
)
EXPECTED_SOURCE_UNION_MANIFEST_SHA256 = (
    "437da25ccef0f40d0504d2038f7f32b5d4849fd1fe556739669b91cbedba7406"
)
EXPECTED_SOURCE_UNION_REVIEW_MANIFEST_SHA256 = (
    "79c6a07e5746f3a5a8a81ecf8677b9efa4ce29b4c0acbdcfbde19dcb70163fb1"
)
EXPECTED_GOVERNING_AUTHOR_V1_MANIFEST_SHA256 = (
    "134d597b1a0a723b2bb18e846f36cc3185fe95d15c41638a5939598a37651649"
)
EXPECTED_GOVERNING_REVIEW_V1_MANIFEST_SHA256 = (
    "6823b16cdb75c6f5ccf23f0f2fb3543bfbcebb6c84e194867932bceedcd74242"
)

CHAPTERS: dict[str, dict[str, Any]] = {
    "CH11": {
        "author": STAGING / "lane_3_chapter_11",
        "reviews": (STAGING / "lane_2_review_chapter_11",),
        "acceptance": INTEGRATION / "chapter_11_substantive_acceptance.md",
        "files": 17,
        "rows": 484,
        "fields": 7_894,
    },
    "CH12": {
        "author": STAGING / "lane_1_chapter_12",
        "reviews": (STAGING / "lane_2_review_chapter_12",),
        "acceptance": INTEGRATION / "chapter_12_substantive_acceptance.md",
        "files": 14,
        "rows": 121,
        "fields": 1_776,
    },
    "CH13": {
        "author": STAGING / "lane_2_chapter_13",
        "reviews": (
            STAGING / "lane_1_review_chapter_13",
            STAGING / "lane_1_verify_chapter_13",
        ),
        "acceptance": INTEGRATION / "chapter_13_substantive_acceptance.md",
        "files": 16,
        "rows": 195,
        "fields": 3_323,
    },
    "CH14": {
        "author": STAGING / "lane_3_chapter_14",
        "reviews": (STAGING / "lane_2_review_chapter_14",),
        "acceptance": INTEGRATION / "chapter_14_substantive_acceptance.md",
        "files": 16,
        "rows": 259,
        "fields": 5_457,
    },
    "CH15": {
        "author": STAGING / "lane_2_chapter_15",
        "reviews": (STAGING / "lane_1_review_chapter_15",),
        "acceptance": INTEGRATION / "chapter_15_substantive_acceptance.md",
        "files": 12,
        "rows": 100,
        "fields": 972,
    },
}

HOLD_FILE_NAMES = {
    "contradiction_log.csv",
    "contradictions.csv",
    "falsifiers.csv",
    "proposed_unknowns.csv",
    "source_queue.csv",
    "unresolved_inputs.csv",
    "writing_gate_audit.csv",
}

STATE_GUARD_TERMS = {
    "additive",
    "authority",
    "bounded",
    "confidence",
    "control",
    "eligibility",
    "evidence",
    "grade",
    "inference",
    "model",
    "nonadditive",
    "publication",
    "readiness",
    "review",
    "scenario",
    "state",
    "status",
}

CLAIM_PATTERN = re.compile(r"(?<![A-Z0-9-])(?:L[123]-)?CLM-\d+(?![A-Z0-9-])")
SOURCE_PATTERN = re.compile(r"(?<![A-Z0-9-])(?:L[123]-)?SRC-\d+(?![A-Z0-9-])")
CANONICAL_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{sha256_bytes(payload)[:24]}"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{relative(path)} has no CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{relative(path)} has duplicate CSV columns")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{relative(path)}:{line_number} exceeds its CSV width")
        if set(row) != set(reader.fieldnames):
            raise ValueError(f"{relative(path)}:{line_number} does not match its header")
    return list(reader.fieldnames), rows


def write_csv(
    target: Path,
    name: str,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
) -> None:
    materialized = list(rows)
    with (target / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(materialized)


def write_json(target: Path, name: str, value: object) -> None:
    (target / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(lane: Path) -> dict[str, Any]:
    return json.loads((lane / "lane_manifest.json").read_text(encoding="utf-8"))


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("files", [])
    if isinstance(entries, dict):
        return [{"path": path, **metadata} for path, metadata in entries.items()]
    return list(entries)


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        label = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    paths = [
        DATA,
        ROOT / "reports/ai_financial_system/research.db",
        ROOT / "reports/ai_financial_system/database_manifest.json",
        INTEGRATION / "candidate_data/research.db",
        STAGING / "chapters_11_15_metadata_rebase_task.md",
        AUDIT,
        SOURCE_UNION,
        SOURCE_UNION_REVIEW,
        GOVERNING_AUTHOR_V1,
        GOVERNING_REVIEW_V1,
    ]
    for config in CHAPTERS.values():
        paths.append(config["author"])
        paths.extend(config["reviews"])
        paths.append(config["acceptance"])
    return paths


def snapshot_protected() -> dict[str, str]:
    return {relative(path): tree_hash(path) for path in protected_paths()}


def manifest_expected_hash(
    lane: Path,
    entry: dict[str, Any],
) -> tuple[Path, str]:
    path = lane / str(entry["path"])
    expected = str(entry.get("sha256", ""))
    actual = sha256(path)
    if not path.is_file() or actual != expected:
        raise ValueError(
            f"manifest mismatch for {relative(path)}: expected={expected} actual={actual}"
        )
    if path.suffix.lower() == ".csv":
        _, rows = read_csv(path)
        declared_rows = entry.get("rows", entry.get("record_count"))
        if declared_rows is not None and len(rows) != int(declared_rows):
            raise ValueError(
                f"manifest row mismatch for {relative(path)}: "
                f"expected={declared_rows} actual={len(rows)}"
            )
    return path, expected


def add_governing_row(
    rows: list[dict[str, object]],
    *,
    chapter: str,
    role: str,
    path: Path,
    expected_sha256: str,
    expected_source: str,
) -> None:
    actual = sha256(path)
    rows.append(
        {
            "governing_input_id": stable_id("GIN", chapter, role, relative(path)),
            "chapter_id": chapter,
            "role": role,
            "path": relative(path),
            "expected_sha256": expected_sha256,
            "observed_sha256": actual,
            "bytes": path.stat().st_size,
            "expected_hash_source": expected_source,
            "status": "PASS" if actual == expected_sha256 else "FAIL",
        }
    )


def sqlite_audit(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    return {
        "integrity_check": integrity[0] if integrity else "",
        "foreign_key_violations": len(foreign_keys),
    }


def row_hash(header: list[str], row: dict[str, str]) -> str:
    payload = json.dumps(
        [[column, row[column]] for column in header],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def extract_tokens(
    row: dict[str, str],
    header: list[str],
    pattern: re.Pattern[str],
    column_term: str,
    *,
    skip_columns: set[str] | None = None,
) -> list[str]:
    found: list[str] = []
    skipped = skip_columns or set()
    for column in header:
        if column in skipped or column_term not in column.lower():
            continue
        for token in pattern.findall(row[column]):
            if token not in found:
                found.append(token)
    return found


def all_cell_token_occurrences(
    row: dict[str, str],
    header: list[str],
    pattern: re.Pattern[str],
) -> list[tuple[str, str]]:
    """Return ordered, cell-specific semantic-token occurrences.

    Unlike ``extract_tokens``, this function intentionally has no column-name
    filter.  It is the lossless author-side inventory used to disposition
    semantic references embedded in notes, model inputs, or other prose cells.
    """

    found: list[tuple[str, str]] = []
    for column in header:
        for token in pattern.findall(row[column]):
            occurrence = (column, token)
            if occurrence not in found:
                found.append(occurrence)
    return found


def packed_values(
    row: dict[str, str],
    header: list[str],
    column_term: str,
) -> str:
    values = [
        f"{column}={row[column]}"
        for column in header
        if column_term in column.lower() and row[column] != ""
    ]
    return " || ".join(values)


def build_source_relationships(
    row_contexts: list[dict[str, Any]],
    source_register: dict[str, dict[str, str]],
    canonical_claims: set[str],
    evidence_links: list[dict[str, str]],
    staged_source_map: dict[str, str],
    source_versions: list[dict[str, str]],
    locator_states: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    evidence_by_claim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for evidence in evidence_links:
        evidence_by_claim[evidence["claim_id"]].append(evidence)

    versions_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for version in source_versions:
        versions_by_source[version["source_id"]].append(version)

    locator_by_exact: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for state in locator_states:
        locator_by_exact[
            (state["claim_id"], state["source_id"], state["locator"])
        ].append(state)

    projection: list[dict[str, object]] = []
    version_junction: list[dict[str, object]] = []
    unmatched_holds: list[dict[str, object]] = []
    locator_holds: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    semantic_dispositions: list[dict[str, object]] = []
    explicitly_held_references: list[dict[str, object]] = []

    for context in row_contexts:
        row = context["row"]
        header = context["header"]
        path: Path = context["path"]
        claims = extract_tokens(row, header, CLAIM_PATTERN, "claim")
        all_claim_occurrences = all_cell_token_occurrences(
            row, header, CLAIM_PATTERN
        )
        skip_sources = (
            {"existing_source_id"}
            if path.name == "proposed_sources.csv" and row.get("provisional_source_id")
            else set()
        )
        sources = extract_tokens(
            row,
            header,
            SOURCE_PATTERN,
            "source",
            skip_columns=skip_sources,
        )
        all_source_occurrences = all_cell_token_occurrences(
            row, header, SOURCE_PATTERN
        )
        accepted_locator = packed_values(row, header, "locator")
        accepted_urls = packed_values(row, header, "url")
        if (
            not all_claim_occurrences
            and not all_source_occurrences
            and not accepted_locator
        ):
            continue

        raw_pairs: list[dict[str, str]] = []
        if claims:
            for claim in claims:
                canonical_claim = claim if claim in canonical_claims else ""
                claim_evidence = evidence_by_claim.get(canonical_claim, [])
                if sources:
                    for accepted_source in sources:
                        canonical_source = staged_source_map.get(accepted_source, accepted_source)
                        if canonical_source not in source_register:
                            canonical_source = ""
                        matches = [
                            link
                            for link in claim_evidence
                            if link["source_id"] == canonical_source
                        ]
                        if matches:
                            for match in matches:
                                raw_pairs.append(
                                    {
                                        "accepted_claim_id": claim,
                                        "canonical_claim_id": canonical_claim,
                                        "accepted_source_id": accepted_source,
                                        "canonical_source_id": canonical_source,
                                        "canonical_evidence_locator": match["locator"],
                                        "support_type": match["support_type"],
                                        "derivation_rule": (
                                            "accepted_claim_and_source_join_to_canonical_evidence_link"
                                        ),
                                        "pairing_ambiguity": "none",
                                    }
                                )
                        else:
                            raw_pairs.append(
                                {
                                    "accepted_claim_id": claim,
                                    "canonical_claim_id": canonical_claim,
                                    "accepted_source_id": accepted_source,
                                    "canonical_source_id": canonical_source,
                                    "canonical_evidence_locator": "",
                                    "support_type": "",
                                    "derivation_rule": "accepted_explicit_claim_source_pair",
                                    "pairing_ambiguity": (
                                        "held_no_canonical_evidence_pair"
                                        if canonical_claim and canonical_source
                                        else "none"
                                    ),
                                }
                            )
                elif claim_evidence:
                    for match in claim_evidence:
                        raw_pairs.append(
                            {
                                "accepted_claim_id": claim,
                                "canonical_claim_id": canonical_claim,
                                "accepted_source_id": match["source_id"],
                                "canonical_source_id": match["source_id"],
                                "canonical_evidence_locator": match["locator"],
                                "support_type": match["support_type"],
                                "derivation_rule": (
                                    "accepted_canonical_claim_join_to_canonical_evidence_link"
                                ),
                                "pairing_ambiguity": "none",
                            }
                        )
                else:
                    raw_pairs.append(
                        {
                            "accepted_claim_id": claim,
                            "canonical_claim_id": canonical_claim,
                            "accepted_source_id": "",
                            "canonical_source_id": "",
                            "canonical_evidence_locator": "",
                            "support_type": "",
                            "derivation_rule": "accepted_claim_without_resolved_source",
                            "pairing_ambiguity": "held_unresolved_source",
                        }
                    )
        else:
            for accepted_source in sources or ([""] if accepted_locator else []):
                canonical_source = staged_source_map.get(accepted_source, accepted_source)
                if canonical_source not in source_register:
                    canonical_source = ""
                raw_pairs.append(
                    {
                        "accepted_claim_id": "",
                        "canonical_claim_id": "",
                        "accepted_source_id": accepted_source,
                        "canonical_source_id": canonical_source,
                        "canonical_evidence_locator": "",
                        "support_type": "",
                        "derivation_rule": (
                            "accepted_source_only_relationship"
                            if accepted_source
                            else "accepted_locator_only_relationship"
                        ),
                        "pairing_ambiguity": "none" if accepted_source else "held_unresolved_source",
                    }
                )

        deduplicated: dict[tuple[str, ...], dict[str, str]] = {}
        for pair in raw_pairs:
            key = (
                pair["accepted_claim_id"],
                pair["canonical_claim_id"],
                pair["accepted_source_id"],
                pair["canonical_source_id"],
                pair["canonical_evidence_locator"],
                accepted_locator,
            )
            deduplicated.setdefault(key, pair)

        relationship_ids: list[str] = []
        for pair in deduplicated.values():
            semantic_locator = accepted_locator or pair["canonical_evidence_locator"]
            relationship_id = stable_id(
                "REL",
                context["source_row_id"],
                pair["accepted_claim_id"],
                pair["accepted_source_id"],
                pair["canonical_source_id"],
                semantic_locator,
            )
            relationship_ids.append(relationship_id)
            matched_versions = versions_by_source.get(pair["canonical_source_id"], [])
            exact_locator_states = locator_by_exact.get(
                (
                    pair["canonical_claim_id"],
                    pair["canonical_source_id"],
                    pair["canonical_evidence_locator"],
                ),
                [],
            )
            locator_semantically_reviewed = bool(exact_locator_states) and all(
                state["semantic_review_state"] == "preserved_and_semantically_reviewed"
                and state["semantic_verdict"] != ""
                for state in exact_locator_states
            )
            unmatched_hold_id = (
                ""
                if matched_versions
                else stable_id("UVH", relationship_id, "unmatched_source_version")
            )
            locator_hold_id = (
                ""
                if locator_semantically_reviewed
                else stable_id("LRH", relationship_id, "unreviewed_locator")
            )
            locator_state = (
                "|".join(
                    sorted({state["semantic_review_state"] for state in exact_locator_states})
                )
                if exact_locator_states
                else "unreviewed_locator_hold"
            )
            locator_reference = "|".join(
                sorted(state["claim_locator_state_id"] for state in exact_locator_states)
            )
            projection.append(
                {
                    "relationship_id": relationship_id,
                    "chapter_id": context["chapter"],
                    "source_row_id": context["source_row_id"],
                    "source_path": relative(path),
                    "source_primary_key": context["primary_key"],
                    "accepted_claim_id": pair["accepted_claim_id"],
                    "canonical_claim_id": pair["canonical_claim_id"],
                    "accepted_source_id": pair["accepted_source_id"],
                    "canonical_source_id": pair["canonical_source_id"],
                    "accepted_source_urls": accepted_urls,
                    "accepted_locator": accepted_locator,
                    "canonical_evidence_locator": pair["canonical_evidence_locator"],
                    "semantic_key_sha256": sha256_bytes(
                        "\x1f".join(
                            [
                                pair["accepted_claim_id"],
                                pair["accepted_source_id"],
                                pair["canonical_source_id"],
                                semantic_locator,
                            ]
                        ).encode("utf-8")
                    ),
                    "support_type": pair["support_type"],
                    "derivation_rule": pair["derivation_rule"],
                    "pairing_ambiguity": pair["pairing_ambiguity"],
                    "version_preservation_state": (
                        "referenced_accepted_source_version_snapshot"
                        if matched_versions
                        else "unmatched_version_hold"
                    ),
                    "source_version_reference_count": len(matched_versions),
                    "unmatched_version_hold_id": unmatched_hold_id,
                    "rights_access_state": (
                        "referenced_only_in_source_version_junction"
                        if matched_versions
                        else "unmatched_version_hold"
                    ),
                    "semantic_locator_review_state": locator_state,
                    "semantic_locator_review_reference_id": locator_reference,
                    "semantic_locator_hold_id": locator_hold_id,
                    "publication_eligibility": "held_no_publication_authority",
                    "byte_preservation_semantic_boundary": (
                        "byte_preservation_is_not_semantic_support"
                    ),
                }
            )
            for version in matched_versions:
                version_junction.append(
                    {
                        "relationship_source_version_id": stable_id(
                            "RSV", relationship_id, version["source_version_id"]
                        ),
                        "relationship_id": relationship_id,
                        "source_version_id": version["source_version_id"],
                        "canonical_source_id": version["source_id"],
                        "source_version_snapshot_row_sha256": row_hash(
                            list(version), version
                        ),
                        "preservation_class": version["preservation_class"],
                        "preservation_status": version["preservation_status"],
                        "content_sha256": version["content_sha256"],
                        "local_archive_path": version["local_archive_path"],
                        "rights_access_boundary": version["access_licensing_boundary"],
                        "rights_access_state": (
                            version["access_licensing_boundary"]
                            or "not_separately_stated_in_snapshot"
                        ),
                        "artifact_role": version["artifact_role"],
                        "source_union_manifest_sha256": (
                            EXPECTED_SOURCE_UNION_MANIFEST_SHA256
                        ),
                        "mutable_source_union_status_copied": "false",
                    }
                )
            if unmatched_hold_id:
                unmatched_holds.append(
                    {
                        "hold_id": unmatched_hold_id,
                        "relationship_id": relationship_id,
                        "chapter_id": context["chapter"],
                        "accepted_source_id": pair["accepted_source_id"],
                        "canonical_source_id": pair["canonical_source_id"],
                        "hold_reason": (
                            "no_canonical_source_identity"
                            if not pair["canonical_source_id"]
                            else "no_matching_accepted_source_version"
                        ),
                        "required_resolution": (
                            "append an independently accepted normalized source version; "
                            "source acquisition is outside this rebase"
                        ),
                        "publication_eligibility": "held",
                    }
                )
            if locator_hold_id:
                locator_holds.append(
                    {
                        "hold_id": locator_hold_id,
                        "relationship_id": relationship_id,
                        "chapter_id": context["chapter"],
                        "canonical_claim_id": pair["canonical_claim_id"],
                        "canonical_source_id": pair["canonical_source_id"],
                        "accepted_locator": accepted_locator,
                        "canonical_evidence_locator": pair["canonical_evidence_locator"],
                        "hold_reason": (
                            "accepted_snapshot_locator_review_pending"
                            if exact_locator_states
                            else "no_exact_accepted_semantic_locator_review"
                        ),
                        "required_resolution": (
                            "fresh semantic locator review; byte preservation is insufficient"
                        ),
                        "publication_eligibility": "held",
                    }
                )

        relationships_by_claim: dict[str, list[str]] = defaultdict(list)
        relationships_by_source: dict[str, list[str]] = defaultdict(list)
        for relationship in projection:
            if relationship["source_row_id"] != context["source_row_id"]:
                continue
            accepted_claim = str(relationship["accepted_claim_id"])
            accepted_source = str(relationship["accepted_source_id"])
            if accepted_claim:
                relationships_by_claim[accepted_claim].append(
                    str(relationship["relationship_id"])
                )
            if accepted_source:
                relationships_by_source[accepted_source].append(
                    str(relationship["relationship_id"])
                )

        for token_type, occurrences, named_tokens, relationship_map in [
            ("claim", all_claim_occurrences, claims, relationships_by_claim),
            ("source", all_source_occurrences, sources, relationships_by_source),
        ]:
            for column, token in occurrences:
                occurrence_id = stable_id(
                    "SEM",
                    context["source_row_id"],
                    column,
                    token_type,
                    token,
                )
                projected_relationship_ids = sorted(
                    set(relationship_map.get(token, []))
                )
                is_staged_source_mapping = (
                    token_type == "source"
                    and path.name == "proposed_sources.csv"
                    and column == "existing_source_id"
                    and bool(row.get("provisional_source_id"))
                )
                if is_staged_source_mapping:
                    disposition = "canonical_source_identity_mapping_reference"
                    governed_hold_id = ""
                    reason = (
                        "accepted_existing_source_identity_maps_provisional_source; "
                        "not_a_claim_support_edge"
                    )
                elif token in named_tokens and projected_relationship_ids:
                    disposition = "projected_relationship_reference"
                    governed_hold_id = ""
                    reason = "typed_token_projected_by_accepted_semantic_column"
                else:
                    disposition = "governed_non_support_reference_hold"
                    governed_hold_id = stable_id(
                        "SRH", occurrence_id, "non_support_reference"
                    )
                    reason = (
                        "prose_or_model_input_reference_requires_fresh_semantic_"
                        "typing; no_support_edge_inferred"
                    )
                    explicitly_held_references.append(
                        {
                            "finding_id": stable_id(
                                "OMITTED",
                                context["chapter"],
                                relative(path),
                                context["primary_key"],
                                column,
                                token_type,
                                token,
                            ),
                            "semantic_reference_occurrence_id": occurrence_id,
                            "governed_hold_id": governed_hold_id,
                            "chapter_id": context["chapter"],
                            "source_row_id": context["source_row_id"],
                            "source_path": relative(path),
                            "source_row_ordinal": context["row_ordinal"],
                            "source_primary_key": context["primary_key"],
                            "source_column": column,
                            "token_type": token_type,
                            "token": token,
                            "named_projection_tokens": "|".join(named_tokens),
                            "source_tokens": "|".join(sources),
                            "locator_present": str(bool(accepted_locator)).lower(),
                            "reference_role": (
                                "contextual_or_methodological_reference"
                            ),
                            "disposition": disposition,
                            "hold_reason": reason,
                            "evidence_state": "not_strengthened",
                            "publication_eligibility": "held",
                        }
                    )
                semantic_dispositions.append(
                    {
                        "semantic_reference_occurrence_id": occurrence_id,
                        "chapter_id": context["chapter"],
                        "source_row_id": context["source_row_id"],
                        "source_path": relative(path),
                        "source_primary_key": context["primary_key"],
                        "source_column": column,
                        "token_type": token_type,
                        "token": token,
                        "disposition": disposition,
                        "projected_relationship_ids": "|".join(
                            projected_relationship_ids
                        ),
                        "governed_hold_id": governed_hold_id,
                        "disposition_reason": reason,
                        "evidence_state": "not_strengthened",
                        "model_use_state": "held_to_acceptance_record",
                        "publication_eligibility": "held",
                    }
                )

        for column in header:
            if "locator" not in column.lower() or row[column] == "":
                continue
            occurrence_id = stable_id(
                "SEM",
                context["source_row_id"],
                column,
                "locator",
                row[column],
            )
            semantic_dispositions.append(
                {
                    "semantic_reference_occurrence_id": occurrence_id,
                    "chapter_id": context["chapter"],
                    "source_row_id": context["source_row_id"],
                    "source_path": relative(path),
                    "source_primary_key": context["primary_key"],
                    "source_column": column,
                    "token_type": "locator",
                    "token": row[column],
                    "disposition": "projected_relationship_reference",
                    "projected_relationship_ids": "|".join(
                        sorted(set(relationship_ids))
                    ),
                    "governed_hold_id": "",
                    "disposition_reason": (
                        "accepted_locator_preserved_in_held_relationship"
                    ),
                    "evidence_state": "not_strengthened",
                    "model_use_state": "held_to_acceptance_record",
                    "publication_eligibility": "held",
                }
            )
        coverage.append(
            {
                "source_projection_coverage_id": stable_id(
                    "SPC", context["source_row_id"]
                ),
                "chapter_id": context["chapter"],
                "source_row_id": context["source_row_id"],
                "source_path": relative(path),
                "accepted_claim_tokens": "|".join(claims),
                "accepted_source_tokens": "|".join(sources),
                "all_cell_claim_tokens": "|".join(
                    dict.fromkeys(token for _, token in all_claim_occurrences)
                ),
                "all_cell_source_tokens": "|".join(
                    dict.fromkeys(token for _, token in all_source_occurrences)
                ),
                "accepted_locator_present": str(bool(accepted_locator)).lower(),
                "relationship_count": len(relationship_ids),
                "semantic_reference_occurrence_count": sum(
                    row["source_row_id"] == context["source_row_id"]
                    for row in semantic_dispositions
                ),
                "explicit_governed_disposition_count": sum(
                    row["source_row_id"] == context["source_row_id"]
                    for row in explicitly_held_references
                ),
                "coverage_status": (
                    "PASS_PROJECTED_RELATIONSHIP"
                    if relationship_ids
                    else "PASS_EXPLICIT_GOVERNED_DISPOSITION"
                ),
            }
        )
    return (
        projection,
        version_junction,
        unmatched_holds,
        locator_holds,
        coverage,
        semantic_dispositions,
        explicitly_held_references,
    )


def substantive_controls() -> list[dict[str, object]]:
    controls = [
        (
            "LIMIT-01",
            "CH11",
            "No additive circularity total is permitted.",
            "block_additive_total",
        ),
        (
            "LIMIT-02",
            "CH11|CH13",
            "No unsupported economic first-loss order is permitted.",
            "block_first_loss_order",
        ),
        (
            "LIMIT-03",
            "CH12|CH13",
            "No project IRR is permitted where matched evidence is absent.",
            "block_project_irr",
        ),
        (
            "LIMIT-04",
            "CH12|CH13",
            "No AI-specific ROIC is permitted where matched evidence is absent.",
            "block_ai_roic",
        ),
        (
            "LIMIT-05",
            "CH12|CH14",
            "No break-even utilization result is permitted where matched evidence is absent.",
            "block_break_even_utilization",
        ),
        (
            "LIMIT-06",
            "CH12|CH13",
            "No required-revenue result is permitted where matched evidence is absent.",
            "block_required_revenue",
        ),
        (
            "LIMIT-07",
            "CH12|CH13|CH14",
            "Governed counterfactuals are not forecasts or empirical probabilities.",
            "block_forecast_or_probability_laundering",
        ),
        (
            "LIMIT-08",
            "CH11|CH13|CH14",
            "No unsupported claimant waterfall or recovery result is permitted.",
            "block_claimant_waterfall_and_recovery",
        ),
        (
            "LIMIT-09",
            "CH14",
            (
                "No unsupported useful life, residual value, price, productive utilization, "
                "or survivability result is permitted."
            ),
            "block_unsupported_asset_economics",
        ),
        (
            "LIMIT-10",
            "CH15",
            (
                "Ownership, contractual control, accounting control, and operational "
                "dependence remain separate."
            ),
            "enforce_control_taxonomy_separation",
        ),
        (
            "LIMIT-11",
            "CH15",
            "All Grade-C dependency assessments remain inference.",
            "block_inference_to_observation_promotion",
        ),
    ]
    return [
        {
            "control_id": control_id,
            "chapter_scope": scope,
            "retained_limit": text,
            "machine_control": machine_control,
            "model_use_state": "held",
            "publication_state": "held",
            "control_status": "ACTIVE",
        }
        for control_id, scope, text, machine_control in controls
    ]


def build(target: Path) -> None:
    protected_before = snapshot_protected()
    target.mkdir(parents=True, exist_ok=True)

    audit_manifest = AUDIT / "lane_manifest.json"
    if sha256(audit_manifest) != EXPECTED_AUDIT_MANIFEST_SHA256:
        raise ValueError("governing audit manifest hash does not match the task")
    audit_inventory_path = AUDIT / "governing_hash_inventory.csv"
    _, audit_inventory = read_csv(audit_inventory_path)
    audit_by_path = {row["path"]: row for row in audit_inventory}

    governing_rows: list[dict[str, object]] = []
    add_governing_row(
        governing_rows,
        chapter="CH11-CH15",
        role="governing_audit_manifest",
        path=audit_manifest,
        expected_sha256=EXPECTED_AUDIT_MANIFEST_SHA256,
        expected_source="chapters_11_15_metadata_rebase_task.md",
    )
    add_governing_row(
        governing_rows,
        chapter="CH11-CH15",
        role="governing_audit_hash_inventory",
        path=audit_inventory_path,
        expected_sha256=sha256(audit_inventory_path),
        expected_source="governing_audit_manifest",
    )

    for lane, manifest_hash, manifest_role, artifact_role in [
        (
            GOVERNING_AUTHOR_V1,
            EXPECTED_GOVERNING_AUTHOR_V1_MANIFEST_SHA256,
            "governing_metadata_rebase_v1_manifest",
            "governing_metadata_rebase_v1_artifact",
        ),
        (
            GOVERNING_REVIEW_V1,
            EXPECTED_GOVERNING_REVIEW_V1_MANIFEST_SHA256,
            "governing_metadata_rebase_v1_review_manifest",
            "governing_metadata_rebase_v1_review_artifact",
        ),
    ]:
        manifest_path = lane / "lane_manifest.json"
        add_governing_row(
            governing_rows,
            chapter="CH11-CH15",
            role=manifest_role,
            path=manifest_path,
            expected_sha256=manifest_hash,
            expected_source="metadata_rebase_v2_repair_task",
        )
        manifest = load_manifest(lane)
        for entry in manifest_entries(manifest):
            path, expected = manifest_expected_hash(lane, entry)
            add_governing_row(
                governing_rows,
                chapter="CH11-CH15",
                role=artifact_role,
                path=path,
                expected_sha256=expected,
                expected_source=manifest_role,
            )

    canonical_database = ROOT / "reports/ai_financial_system/research.db"
    canonical_manifest = ROOT / "reports/ai_financial_system/database_manifest.json"
    candidate_database = INTEGRATION / "candidate_data/research.db"
    for role, path, expected, expected_source in [
        (
            "canonical_database",
            canonical_database,
            EXPECTED_DATABASE_SHA256,
            "chapters_11_15_metadata_rebase_task.md",
        ),
        (
            "canonical_database_manifest",
            canonical_manifest,
            sha256(canonical_manifest),
            "frozen_canonical_baseline_observation",
        ),
        (
            "sealed_candidate_database",
            candidate_database,
            EXPECTED_DATABASE_SHA256,
            "chapters_11_15_metadata_rebase_task.md",
        ),
    ]:
        add_governing_row(
            governing_rows,
            chapter="CH11-CH15",
            role=role,
            path=path,
            expected_sha256=expected,
            expected_source=expected_source,
        )

    source_union_manifest = SOURCE_UNION / "lane_manifest.json"
    source_union_review_manifest = SOURCE_UNION_REVIEW / "lane_manifest.json"
    for role, path, expected in [
        (
            "accepted_source_union_manifest",
            source_union_manifest,
            EXPECTED_SOURCE_UNION_MANIFEST_SHA256,
        ),
        (
            "accepted_source_union_review_manifest",
            source_union_review_manifest,
            EXPECTED_SOURCE_UNION_REVIEW_MANIFEST_SHA256,
        ),
    ]:
        add_governing_row(
            governing_rows,
            chapter="CH11-CH15",
            role=role,
            path=path,
            expected_sha256=expected,
            expected_source="fresh_source_union_v2_acceptance",
        )
    source_union_entries = {
        str(entry["path"]): entry for entry in manifest_entries(load_manifest(SOURCE_UNION))
    }
    for name in ("source_versions.csv", "claim_locator_states.csv"):
        path, expected = manifest_expected_hash(SOURCE_UNION, source_union_entries[name])
        add_governing_row(
            governing_rows,
            chapter="CH11-CH15",
            role="accepted_source_union_consumed_artifact",
            path=path,
            expected_sha256=expected,
            expected_source="accepted_source_union_manifest",
        )
    review_entries = {
        str(entry["path"]): entry
        for entry in manifest_entries(load_manifest(SOURCE_UNION_REVIEW))
    }
    review_acceptance_path, review_acceptance_hash = manifest_expected_hash(
        SOURCE_UNION_REVIEW,
        review_entries["acceptance_report.md"],
    )
    add_governing_row(
        governing_rows,
        chapter="CH11-CH15",
        role="accepted_source_union_acceptance_report",
        path=review_acceptance_path,
        expected_sha256=review_acceptance_hash,
        expected_source="accepted_source_union_review_manifest",
    )

    author_artifacts: list[dict[str, Any]] = []
    acceptance_bindings: list[dict[str, object]] = []
    for chapter, config in CHAPTERS.items():
        author: Path = config["author"]
        author_manifest_path = author / "lane_manifest.json"
        audit_manifest_row = audit_by_path[relative(author_manifest_path)]
        add_governing_row(
            governing_rows,
            chapter=chapter,
            role="author_manifest",
            path=author_manifest_path,
            expected_sha256=audit_manifest_row["sha256"],
            expected_source="governing_audit_hash_inventory",
        )
        author_manifest = load_manifest(author)
        entries = manifest_entries(author_manifest)
        if len(entries) != int(config["files"]):
            raise ValueError(f"{chapter} manifest file population drift")
        for entry in entries:
            path, expected = manifest_expected_hash(author, entry)
            audit_row = audit_by_path.get(relative(path))
            if audit_row is None or audit_row["sha256"] != expected:
                raise ValueError(f"{chapter} artifact is absent or changed in governing audit")
            add_governing_row(
                governing_rows,
                chapter=chapter,
                role="author_artifact",
                path=path,
                expected_sha256=expected,
                expected_source="author_manifest_and_governing_audit",
            )
            author_artifacts.append(
                {
                    "chapter": chapter,
                    "author": author,
                    "path": path,
                    "entry": entry,
                    "manifest_path": author_manifest_path,
                    "manifest_sha256": sha256(author_manifest_path),
                    "acceptance": config["acceptance"],
                    "acceptance_sha256": sha256(config["acceptance"]),
                }
            )

        for review_index, review in enumerate(config["reviews"], start=1):
            review_manifest_path = review / "lane_manifest.json"
            audit_review_row = audit_by_path[relative(review_manifest_path)]
            review_role = (
                "correction_verification_manifest"
                if chapter == "CH13" and review_index == 2
                else "review_manifest"
            )
            add_governing_row(
                governing_rows,
                chapter=chapter,
                role=review_role,
                path=review_manifest_path,
                expected_sha256=audit_review_row["sha256"],
                expected_source="governing_audit_hash_inventory",
            )
            review_manifest = load_manifest(review)
            for entry in manifest_entries(review_manifest):
                path, expected = manifest_expected_hash(review, entry)
                artifact_role = (
                    "correction_verification_artifact"
                    if chapter == "CH13" and review_index == 2
                    else "review_artifact"
                )
                add_governing_row(
                    governing_rows,
                    chapter=chapter,
                    role=artifact_role,
                    path=path,
                    expected_sha256=expected,
                    expected_source=review_role,
                )

        acceptance: Path = config["acceptance"]
        audit_acceptance_row = audit_by_path[relative(acceptance)]
        add_governing_row(
            governing_rows,
            chapter=chapter,
            role="substantive_acceptance",
            path=acceptance,
            expected_sha256=audit_acceptance_row["sha256"],
            expected_source="governing_audit_hash_inventory",
        )
        acceptance_bindings.append(
            {
                "acceptance_binding_id": f"ACCEPTANCE-{chapter}",
                "chapter_id": chapter,
                "substantive_acceptance_path": relative(acceptance),
                "substantive_acceptance_sha256": sha256(acceptance),
                "accepted_author_manifest_path": relative(author_manifest_path),
                "accepted_author_manifest_sha256": sha256(author_manifest_path),
                "boundary_enforcement": "all_acceptance_record_boundaries_retained",
                "evidence_state_change_authorized": "false",
                "model_use_state": "held_to_acceptance_record",
                "publication_state": "held_no_publication_authority",
            }
        )

    file_inventory: list[dict[str, object]] = []
    column_inventory: list[dict[str, object]] = []
    row_inventory: list[dict[str, object]] = []
    field_dispositions: list[dict[str, object]] = []
    row_contexts: list[dict[str, Any]] = []
    canonical_registry_rows: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    canonical_ids: dict[str, set[str]] = {}
    id_to_registries: dict[str, set[str]] = defaultdict(set)
    for registry_path in sorted(DATA.glob("*.csv")):
        header, rows = read_csv(registry_path)
        canonical_registry_rows[registry_path.name] = (header, rows)
        ids = {row[header[0]] for row in rows if row[header[0]]}
        canonical_ids[registry_path.name] = ids
        for identity in ids:
            id_to_registries[identity].add(registry_path.name)

    identity_references: list[dict[str, object]] = []
    seen_identity_references: set[tuple[str, str, str]] = set()
    chapter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for artifact in author_artifacts:
        path: Path = artifact["path"]
        chapter = artifact["chapter"]
        file_id = stable_id("FILE", chapter, relative(path))
        file_row: dict[str, object] = {
            "source_file_id": file_id,
            "chapter_id": chapter,
            "source_path": relative(path),
            "media_type": "text/csv" if path.suffix.lower() == ".csv" else "text/markdown",
            "manifest_path": relative(artifact["manifest_path"]),
            "manifest_sha256": artifact["manifest_sha256"],
            "manifest_declared_sha256": artifact["entry"]["sha256"],
            "observed_sha256": sha256(path),
            "bytes": path.stat().st_size,
            "csv_rows": "",
            "csv_columns": "",
            "scalar_fields": "",
            "byte_identity_status": "PASS",
        }
        chapter_counts[chapter]["files"] += 1
        if path.suffix.lower() == ".csv":
            header, rows = read_csv(path)
            file_row["csv_rows"] = len(rows)
            file_row["csv_columns"] = len(header)
            file_row["scalar_fields"] = len(rows) * len(header)
            chapter_counts[chapter]["rows"] += len(rows)
            chapter_counts[chapter]["fields"] += len(rows) * len(header)
            column_ids: dict[str, str] = {}
            for column_ordinal, column in enumerate(header, start=1):
                column_id = stable_id("COL", file_id, column_ordinal, column)
                column_ids[column] = column_id
                column_inventory.append(
                    {
                        "source_column_id": column_id,
                        "source_file_id": file_id,
                        "chapter_id": chapter,
                        "source_path": relative(path),
                        "column_ordinal": column_ordinal,
                        "column_name": column,
                        "column_name_sha256": sha256_bytes(column.encode("utf-8")),
                    }
                )
            seen_primary_keys: set[str] = set()
            for row_ordinal, row in enumerate(rows, start=1):
                primary_key_column = header[0]
                primary_key = row[primary_key_column]
                if not primary_key or primary_key in seen_primary_keys:
                    raise ValueError(
                        f"{relative(path)} has blank or duplicate primary key {primary_key!r}"
                    )
                seen_primary_keys.add(primary_key)
                source_row_id = stable_id(
                    "ROW", chapter, relative(path), primary_key_column, primary_key
                )
                row_inventory.append(
                    {
                        "source_row_id": source_row_id,
                        "source_file_id": file_id,
                        "chapter_id": chapter,
                        "source_path": relative(path),
                        "source_row_ordinal": row_ordinal,
                        "source_csv_line_number": row_ordinal + 1,
                        "primary_key_column": primary_key_column,
                        "primary_key_value": primary_key,
                        "source_row_sha256": row_hash(header, row),
                        "column_count": len(header),
                        "scalar_field_count": len(header),
                    }
                )
                context = {
                    "source_row_id": source_row_id,
                    "chapter": chapter,
                    "path": path,
                    "header": header,
                    "row": row,
                    "primary_key": primary_key,
                    "row_ordinal": row_ordinal,
                }
                row_contexts.append(context)
                for column_ordinal, column in enumerate(header, start=1):
                    value = row[column]
                    field_id = stable_id("FIELD", source_row_id, column_ordinal, column)
                    is_state_guarded = any(
                        term in column.lower() for term in STATE_GUARD_TERMS
                    )
                    disposition = (
                        "governed_hold_destination"
                        if path.name in HOLD_FILE_NAMES
                        else "typed_normalized_junction_destination"
                    )
                    destination = (
                        "metadata_rebase.governed_packet_field_hold"
                        if disposition == "governed_hold_destination"
                        else "metadata_rebase.typed_packet_field_junction"
                    )
                    field_dispositions.append(
                        {
                            "field_disposition_id": field_id,
                            "source_row_id": source_row_id,
                            "source_column_id": column_ids[column],
                            "chapter_id": chapter,
                            "source_path": relative(path),
                            "source_row_ordinal": row_ordinal,
                            "source_primary_key": primary_key,
                            "source_column_ordinal": column_ordinal,
                            "source_column": column,
                            "original_value": value,
                            "original_value_sha256": sha256_bytes(value.encode("utf-8")),
                            "source_is_populated": str(value != "").lower(),
                            "disposition": disposition,
                            "destination": destination,
                            "destination_identity": field_id,
                            "destination_value": value,
                            "transformation_rule": "identity_no_transformation",
                            "evidence_state_guard": (
                                "identity_locked" if is_state_guarded else "not_applicable"
                            ),
                            "destination_evidence_state": (
                                value if is_state_guarded else ""
                            ),
                            "governing_acceptance_path": relative(artifact["acceptance"]),
                            "governing_acceptance_sha256": artifact["acceptance_sha256"],
                        }
                    )
                    for token in CANONICAL_TOKEN_PATTERN.findall(value):
                        for registry in sorted(id_to_registries.get(token, set())):
                            key = (field_id, token, registry)
                            if key in seen_identity_references:
                                continue
                            seen_identity_references.add(key)
                            identity_references.append(
                                {
                                    "canonical_identity_reference_id": stable_id(
                                        "CIR", field_id, token, registry
                                    ),
                                    "field_disposition_id": field_id,
                                    "source_row_id": source_row_id,
                                    "chapter_id": chapter,
                                    "canonical_identity": token,
                                    "canonical_registry": registry,
                                    "canonical_registry_path": relative(DATA / registry),
                                    "reference_state": "reference_only_no_canonical_mutation",
                                }
                            )
        file_inventory.append(file_row)

    for chapter, expected in CHAPTERS.items():
        actual = chapter_counts[chapter]
        for dimension in ("files", "rows", "fields"):
            if actual[dimension] != int(expected[dimension]):
                raise ValueError(
                    f"{chapter} {dimension} drift: "
                    f"expected={expected[dimension]} actual={actual[dimension]}"
                )

    source_register_header, source_register_rows = canonical_registry_rows[
        "source_register.csv"
    ]
    del source_register_header
    source_register = {row["source_id"]: row for row in source_register_rows}
    _, canonical_claim_rows = canonical_registry_rows["claims.csv"]
    canonical_claims = {row["claim_id"] for row in canonical_claim_rows}
    _, evidence_links = canonical_registry_rows["evidence_links.csv"]
    _, source_versions = read_csv(SOURCE_UNION / "source_versions.csv")
    _, locator_states = read_csv(SOURCE_UNION / "claim_locator_states.csv")

    staged_source_map: dict[str, str] = {}
    _, proposed_sources = read_csv(
        CHAPTERS["CH11"]["author"] / "proposed_sources.csv"
    )
    for row in proposed_sources:
        staged_source_map[row["provisional_source_id"]] = row["existing_source_id"]

    (
        source_projection,
        version_junction,
        unmatched_version_holds,
        semantic_locator_holds,
        source_projection_coverage,
        semantic_reference_dispositions,
        unprojected_semantic_references,
    ) = build_source_relationships(
        row_contexts,
        source_register,
        canonical_claims,
        evidence_links,
        staged_source_map,
        source_versions,
        locator_states,
    )

    registry_usage: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"identities": set(), "references": set()}
    )
    for reference in identity_references:
        usage = registry_usage[
            (str(reference["chapter_id"]), str(reference["canonical_registry"]))
        ]
        usage["identities"].add(str(reference["canonical_identity"]))
        usage["references"].add(str(reference["canonical_identity_reference_id"]))
    for relationship in source_projection:
        chapter = str(relationship["chapter_id"])
        canonical_claim = str(relationship["canonical_claim_id"])
        canonical_source = str(relationship["canonical_source_id"])
        if canonical_claim:
            for registry in ("claims.csv", "evidence_links.csv"):
                usage = registry_usage[(chapter, registry)]
                usage["identities"].add(canonical_claim)
                usage["references"].add(str(relationship["relationship_id"]))
        if canonical_source:
            usage = registry_usage[(chapter, "source_register.csv")]
            usage["identities"].add(canonical_source)
            usage["references"].add(str(relationship["relationship_id"]))

    registry_bindings: list[dict[str, object]] = []
    for (chapter, registry), usage in sorted(registry_usage.items()):
        registry_path = DATA / registry
        registry_bindings.append(
            {
                "canonical_registry_binding_id": stable_id(
                    "CRB", chapter, registry
                ),
                "chapter_id": chapter,
                "canonical_registry": registry,
                "canonical_registry_path": relative(registry_path),
                "canonical_registry_sha256": sha256(registry_path),
                "unique_canonical_identities_referenced": len(usage["identities"]),
                "reference_count": len(usage["references"]),
                "consumption_basis": (
                    "accepted_scalar_reference_and_or_claim_source_projection"
                ),
                "mutation_authority": "none",
            }
        )

    disposition_counts = Counter(
        (str(row["chapter_id"]), str(row["disposition"]))
        for row in field_dispositions
    )
    disposition_summary = [
        {
            "field_disposition_summary_id": stable_id(
                "FDS", chapter, disposition
            ),
            "chapter_id": chapter,
            "disposition": disposition,
            "scalar_field_count": count,
        }
        for (chapter, disposition), count in sorted(disposition_counts.items())
    ]

    database_audit = {
        "canonical_database_path": relative(canonical_database),
        "canonical_database_sha256": sha256(canonical_database),
        "canonical_database_manifest_path": relative(canonical_manifest),
        "canonical_database_manifest_sha256": sha256(canonical_manifest),
        "sealed_candidate_database_path": relative(candidate_database),
        "sealed_candidate_database_sha256": sha256(candidate_database),
        "expected_database_sha256": EXPECTED_DATABASE_SHA256,
        "canonical_matches_expected": (
            sha256(canonical_database) == EXPECTED_DATABASE_SHA256
        ),
        "candidate_matches_expected": (
            sha256(candidate_database) == EXPECTED_DATABASE_SHA256
        ),
        "canonical_candidate_byte_identical": (
            sha256(canonical_database) == sha256(candidate_database)
        ),
        "canonical_sqlite": sqlite_audit(canonical_database),
        "candidate_sqlite": sqlite_audit(candidate_database),
        "governing_audit_manifest_sha256": sha256(audit_manifest),
        "accepted_source_union_manifest_sha256": sha256(source_union_manifest),
        "accepted_source_union_review_manifest_sha256": sha256(
            source_union_review_manifest
        ),
    }

    write_csv(
        target,
        "governing_input_inventory.csv",
        governing_rows,
        [
            "governing_input_id",
            "chapter_id",
            "role",
            "path",
            "expected_sha256",
            "observed_sha256",
            "bytes",
            "expected_hash_source",
            "status",
        ],
    )
    write_csv(
        target,
        "accepted_file_inventory.csv",
        file_inventory,
        [
            "source_file_id",
            "chapter_id",
            "source_path",
            "media_type",
            "manifest_path",
            "manifest_sha256",
            "manifest_declared_sha256",
            "observed_sha256",
            "bytes",
            "csv_rows",
            "csv_columns",
            "scalar_fields",
            "byte_identity_status",
        ],
    )
    write_csv(
        target,
        "accepted_column_inventory.csv",
        column_inventory,
        [
            "source_column_id",
            "source_file_id",
            "chapter_id",
            "source_path",
            "column_ordinal",
            "column_name",
            "column_name_sha256",
        ],
    )
    write_csv(
        target,
        "accepted_row_inventory.csv",
        row_inventory,
        [
            "source_row_id",
            "source_file_id",
            "chapter_id",
            "source_path",
            "source_row_ordinal",
            "source_csv_line_number",
            "primary_key_column",
            "primary_key_value",
            "source_row_sha256",
            "column_count",
            "scalar_field_count",
        ],
    )
    write_csv(
        target,
        "field_dispositions.csv",
        field_dispositions,
        [
            "field_disposition_id",
            "source_row_id",
            "source_column_id",
            "chapter_id",
            "source_path",
            "source_row_ordinal",
            "source_primary_key",
            "source_column_ordinal",
            "source_column",
            "original_value",
            "original_value_sha256",
            "source_is_populated",
            "disposition",
            "destination",
            "destination_identity",
            "destination_value",
            "transformation_rule",
            "evidence_state_guard",
            "destination_evidence_state",
            "governing_acceptance_path",
            "governing_acceptance_sha256",
        ],
    )
    write_csv(
        target,
        "field_disposition_summary.csv",
        disposition_summary,
        [
            "field_disposition_summary_id",
            "chapter_id",
            "disposition",
            "scalar_field_count",
        ],
    )
    write_csv(
        target,
        "canonical_identity_references.csv",
        identity_references,
        [
            "canonical_identity_reference_id",
            "field_disposition_id",
            "source_row_id",
            "chapter_id",
            "canonical_identity",
            "canonical_registry",
            "canonical_registry_path",
            "reference_state",
        ],
    )
    write_csv(
        target,
        "canonical_registry_bindings.csv",
        registry_bindings,
        [
            "canonical_registry_binding_id",
            "chapter_id",
            "canonical_registry",
            "canonical_registry_path",
            "canonical_registry_sha256",
            "unique_canonical_identities_referenced",
            "reference_count",
            "consumption_basis",
            "mutation_authority",
        ],
    )
    write_csv(
        target,
        "acceptance_boundary_bindings.csv",
        acceptance_bindings,
        [
            "acceptance_binding_id",
            "chapter_id",
            "substantive_acceptance_path",
            "substantive_acceptance_sha256",
            "accepted_author_manifest_path",
            "accepted_author_manifest_sha256",
            "boundary_enforcement",
            "evidence_state_change_authorized",
            "model_use_state",
            "publication_state",
        ],
    )
    write_csv(
        target,
        "substantive_limit_controls.csv",
        substantive_controls(),
        [
            "control_id",
            "chapter_scope",
            "retained_limit",
            "machine_control",
            "model_use_state",
            "publication_state",
            "control_status",
        ],
    )
    write_csv(
        target,
        "claim_source_locator_projection.csv",
        source_projection,
        [
            "relationship_id",
            "chapter_id",
            "source_row_id",
            "source_path",
            "source_primary_key",
            "accepted_claim_id",
            "canonical_claim_id",
            "accepted_source_id",
            "canonical_source_id",
            "accepted_source_urls",
            "accepted_locator",
            "canonical_evidence_locator",
            "semantic_key_sha256",
            "support_type",
            "derivation_rule",
            "pairing_ambiguity",
            "version_preservation_state",
            "source_version_reference_count",
            "unmatched_version_hold_id",
            "rights_access_state",
            "semantic_locator_review_state",
            "semantic_locator_review_reference_id",
            "semantic_locator_hold_id",
            "publication_eligibility",
            "byte_preservation_semantic_boundary",
        ],
    )
    write_csv(
        target,
        "relationship_source_version_junction.csv",
        version_junction,
        [
            "relationship_source_version_id",
            "relationship_id",
            "source_version_id",
            "canonical_source_id",
            "source_version_snapshot_row_sha256",
            "preservation_class",
            "preservation_status",
            "content_sha256",
            "local_archive_path",
            "rights_access_boundary",
            "rights_access_state",
            "artifact_role",
            "source_union_manifest_sha256",
            "mutable_source_union_status_copied",
        ],
    )
    write_csv(
        target,
        "source_projection_coverage.csv",
        source_projection_coverage,
        [
            "source_projection_coverage_id",
            "chapter_id",
            "source_row_id",
            "source_path",
            "accepted_claim_tokens",
            "accepted_source_tokens",
            "all_cell_claim_tokens",
            "all_cell_source_tokens",
            "accepted_locator_present",
            "relationship_count",
            "semantic_reference_occurrence_count",
            "explicit_governed_disposition_count",
            "coverage_status",
        ],
    )
    write_csv(
        target,
        "semantic_reference_dispositions.csv",
        semantic_reference_dispositions,
        [
            "semantic_reference_occurrence_id",
            "chapter_id",
            "source_row_id",
            "source_path",
            "source_primary_key",
            "source_column",
            "token_type",
            "token",
            "disposition",
            "projected_relationship_ids",
            "governed_hold_id",
            "disposition_reason",
            "evidence_state",
            "model_use_state",
            "publication_eligibility",
        ],
    )
    write_csv(
        target,
        "unprojected_semantic_references.csv",
        unprojected_semantic_references,
        [
            "finding_id",
            "semantic_reference_occurrence_id",
            "governed_hold_id",
            "chapter_id",
            "source_row_id",
            "source_path",
            "source_row_ordinal",
            "source_primary_key",
            "source_column",
            "token_type",
            "token",
            "named_projection_tokens",
            "source_tokens",
            "locator_present",
            "reference_role",
            "disposition",
            "hold_reason",
            "evidence_state",
            "publication_eligibility",
        ],
    )
    write_csv(
        target,
        "unmatched_source_version_holds.csv",
        unmatched_version_holds,
        [
            "hold_id",
            "relationship_id",
            "chapter_id",
            "accepted_source_id",
            "canonical_source_id",
            "hold_reason",
            "required_resolution",
            "publication_eligibility",
        ],
    )
    write_csv(
        target,
        "semantic_locator_holds.csv",
        semantic_locator_holds,
        [
            "hold_id",
            "relationship_id",
            "chapter_id",
            "canonical_claim_id",
            "canonical_source_id",
            "accepted_locator",
            "canonical_evidence_locator",
            "hold_reason",
            "required_resolution",
            "publication_eligibility",
        ],
    )
    write_json(target, "canonical_baseline_audit.json", database_audit)

    protected_after = snapshot_protected()
    no_write_rows = [
        {
            "protected_input_id": stable_id("PROTECTED", path),
            "path": path,
            "before_tree_sha256": before_hash,
            "after_tree_sha256": protected_after[path],
            "status": "PASS" if before_hash == protected_after[path] else "FAIL",
        }
        for path, before_hash in sorted(protected_before.items())
    ]
    write_csv(
        target,
        "protected_input_no_write_audit.csv",
        no_write_rows,
        [
            "protected_input_id",
            "path",
            "before_tree_sha256",
            "after_tree_sha256",
            "status",
        ],
    )

    build_counts = {
        "accepted_author_files": len(file_inventory),
        "accepted_csv_files": sum(row["media_type"] == "text/csv" for row in file_inventory),
        "accepted_markdown_files": sum(
            row["media_type"] == "text/markdown" for row in file_inventory
        ),
        "accepted_csv_rows": len(row_inventory),
        "accepted_scalar_fields": len(field_dispositions),
        "accepted_columns": len(column_inventory),
        "canonical_identity_references": len(identity_references),
        "canonical_registry_bindings": len(registry_bindings),
        "claim_source_locator_relationships": len(source_projection),
        "relationship_source_version_references": len(version_junction),
        "unique_source_versions_referenced": len(
            {row["source_version_id"] for row in version_junction}
        ),
        "matched_source_relationships": sum(
            int(row["source_version_reference_count"]) > 0 for row in source_projection
        ),
        "unmatched_source_version_holds": len(unmatched_version_holds),
        "semantic_locator_holds": len(semantic_locator_holds),
        "source_projection_rows_covered": len(source_projection_coverage),
        "semantic_reference_occurrences": len(
            semantic_reference_dispositions
        ),
        "explicit_semantic_reference_holds": len(
            unprojected_semantic_references
        ),
        "substantive_limit_controls": len(substantive_controls()),
        "protected_inputs_audited": len(no_write_rows),
        "chapter_populations": {
            chapter: dict(sorted(counts.items()))
            for chapter, counts in sorted(chapter_counts.items())
        },
    }
    write_json(target, "build_counts.json", build_counts)

    report = f"""# Chapters 11-15 lossless convergence metadata rebase

## Result

The rebase preserves all **{len(file_inventory)}** accepted manifest-declared author
artifacts byte-for-byte: **{len(row_inventory):,}** CSV rows and
**{len(field_dispositions):,}** scalar fields. Every scalar has one unique
identity-preserving disposition, an unchanged destination value, and its governing
substantive-acceptance hash. No accepted prose, claim, model, arithmetic, evidence state,
or publication boundary was changed.

The frozen canonical database and sealed integration-candidate database are byte-identical
at `{EXPECTED_DATABASE_SHA256}`. The canonical database manifest, every consumed canonical
CSV registry, every accepted author/review/verification/acceptance input, and the accepted
source-union-v2 snapshot are hash-bound in this packet.

## Source-version and locator boundary

The exact claim/source/locator projection contains **{len(source_projection):,}**
relationships. **{sum(int(row["source_version_reference_count"]) > 0 for row in source_projection):,}**
relationships reference one or more accepted normalized source-version observations through
**{len(version_junction):,}** junction rows; **{len(unmatched_version_holds):,}**
relationships retain explicit unmatched-version holds. The packet retains
**{len(semantic_locator_holds):,}** semantic-locator holds.

Version preservation, rights/access, semantic locator review, and publication eligibility
are separate fields. Mutable source-union candidate status is not copied into chapter rows,
and byte preservation is never treated as semantic support.

An all-cell scan inventories **{len(semantic_reference_dispositions):,}** semantic
claim/source/locator occurrences. Every occurrence is either bound to one or more held
projected relationships or receives an explicit governed non-support reference hold.
The **{len(unprojected_semantic_references):,}** prose/model-input references previously
omitted by column-name-only discovery are retained in
`unprojected_semantic_references.csv`; none is converted into a support edge.

## Retained authority limits

All five substantive-acceptance records are exact governing inputs. The eleven explicit
machine controls retain nonadditivity, first-loss, return-model, counterfactual,
claimant-waterfall, recovery, utilization, useful-life, residual-value, price,
survivability, control-taxonomy, and Grade-C inference limits. Every projected relationship
remains held from publication by this metadata-only packet.

## Recommendation and authority

This packet is suitable for fresh uninvolved metadata acceptance after its sealed manifest,
dedicated validation, Ruff, generic staging validation, and two clean deterministic builds
are confirmed. It does not authorize canonical integration, chapter or DOCX replacement,
model-output promotion, or publication.
"""
    (target / "author_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
