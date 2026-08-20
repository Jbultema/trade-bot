from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from trade_bot.research.capital_cycle_feasibility import (
    DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    REQUIRED_DIRECT_FAMILIES,
)
from trade_bot.research.capital_cycle_sources import (
    SEC_FACT_TAGS,
    _merge_observations,
    import_sec_companyfacts_panel,
)

DEFAULT_AI_FINANCIAL_SYSTEM_DIR = Path("research/ai_financial_system")
DEFAULT_AI_EPISODE: Final[str] = "ai_buildout_2022_present"
DEFAULT_SEC_USER_AGENT: Final[str] = (
    "trade-bot capital-cycle research contact github.com/Jbultema"
)

# A family is not called usable merely because one old row exists.  The policy is
# deliberately transparent and can be frozen with each prospective vintage.
AI_FAMILY_POLICY: Final[dict[str, dict[str, int]]] = {
    "demand": {"maximum_age_days": 190, "minimum_rows": 2, "minimum_entities": 2},
    "capacity": {"maximum_age_days": 190, "minimum_rows": 2, "minimum_entities": 2},
    "pricing_utilization": {
        "maximum_age_days": 100,
        "minimum_rows": 2,
        "minimum_entities": 2,
    },
    "financing": {"maximum_age_days": 190, "minimum_rows": 2, "minimum_entities": 2},
    "monetization": {"maximum_age_days": 190, "minimum_rows": 2, "minimum_entities": 2},
    "commitment_overhang": {
        "maximum_age_days": 370,
        "minimum_rows": 2,
        "minimum_entities": 2,
    },
}


@dataclass(frozen=True)
class CurrentAiEvidenceBuild:
    observations: pd.DataFrame
    coverage: pd.DataFrame
    sec_inventory: pd.DataFrame
    ledger_inventory: pd.DataFrame
    observation_path: Path
    coverage_path: Path
    manifest_path: Path


def fetch_sec_companyfacts_archive(
    *,
    entity_manifest: pd.DataFrame,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    user_agent: str | None = None,
    force: bool = False,
    timeout_seconds: float = 30.0,
    minimum_interval_seconds: float = 0.12,
) -> pd.DataFrame:
    """Fetch official SEC Company Facts sequentially into the governed raw archive."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "sec_companyfacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent
            or os.getenv("TRADE_BOT_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
    )
    rows: list[dict[str, object]] = []
    last_request = 0.0
    for _, entity in entity_manifest.iterrows():
        cik = int(entity["cik"])
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        raw_path = raw_dir / f"CIK{cik:010d}.json"
        status = "cached"
        error = ""
        http_status: int | str = ""
        if force or not raw_path.exists():
            wait = minimum_interval_seconds - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                response = session.get(url, timeout=timeout_seconds)
                last_request = time.monotonic()
                http_status = response.status_code
                response.raise_for_status()
                payload = response.content
                json.loads(payload)
                raw_path.write_bytes(payload)
                status = "downloaded"
            except (requests.RequestException, json.JSONDecodeError, OSError) as exc:
                status = "download_failed"
                error = str(exc)
        payload = raw_path.read_bytes() if raw_path.exists() else b""
        rows.append(
            {
                "episode": entity.get("episode", DEFAULT_AI_EPISODE),
                "entity": entity["entity"],
                "cik": cik,
                "source_url": url,
                "raw_path": str(raw_path),
                "status": status,
                "http_status": http_status,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(destination / "sec_companyfacts_download_inventory.csv", index=False)
    return inventory


def import_ai_financial_system_ledgers(
    *,
    project_dir: str | Path = DEFAULT_AI_FINANCIAL_SYSTEM_DIR,
    episode: str = DEFAULT_AI_EPISODE,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert audited monograph ledgers into non-additive point-in-time observations.

    This reuses primary-source provenance already reconstructed for the monograph.
    It does not infer undisclosed minimum spend, utilization, or profitability.
    """

    root = Path(project_dir) / "data"
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.max
    sources = _read_csv(root / "source_register.csv")
    links = _read_csv(root / "evidence_links.csv")
    entities = _read_csv(root / "entities.csv")
    source_lookup = _source_lookup(sources, links)
    entity_lookup = dict(
        zip(entities.get("entity_id", []), entities.get("canonical_name", []), strict=False)
    )
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    contracts = _read_csv(root / "contracts.csv")
    for _, record in contracts.iterrows():
        value = _number(record.get("minimum_commitment"))
        value_basis = "minimum_commitment"
        if value is None:
            value = _number(record.get("stated_value"))
            value_basis = "stated_value_not_proven_minimum"
        available = _first_date(record, "execution_date", "announcement_date", "effective_date")
        if value is None or pd.isna(available) or available > cutoff:
            continue
        entity = _entity_pair(record, entity_lookup, "party_a_entity_id", "party_b_entity_id")
        rows.append(
            _ledger_observation(
                episode=episode,
                family="demand",
                metric=f"contract_{value_basis}",
                entity=entity,
                period_end=available,
                available_date=available,
                value=value,
                unit=str(record.get("currency", "USD")),
                claim_id=str(record.get("source_claim_id", "")),
                source_lookup=source_lookup,
                record_id=str(record.get("contract_id", "")),
                notes=(
                    "Contractual demand evidence. A stated or upper value is not treated as "
                    "a minimum payment unless the ledger explicitly identifies one."
                ),
            )
        )

    commitments = _read_csv(root / "compute_commitments.csv")
    for _, record in commitments.iterrows():
        available = _first_date(record, "execution_date", "announcement_date")
        if pd.isna(available) or available > cutoff:
            continue
        entity = _entity_pair(
            record, entity_lookup, "supplier_entity_id", "customer_entity_id"
        )
        value = _number(record.get("minimum_spend"))
        basis = "minimum_spend"
        if value is None:
            value = _number(record.get("contract_value"))
            basis = "stated_contract_value_not_proven_minimum"
        if value is not None:
            rows.append(
                _ledger_observation(
                    episode=episode,
                    family="commitment_overhang",
                    metric=f"compute_{basis}",
                    entity=entity,
                    period_end=available,
                    available_date=available,
                    value=value,
                    unit=str(record.get("currency", "USD")),
                    claim_id=str(record.get("source_claim_id", "")),
                    source_lookup=source_lookup,
                    record_id=str(record.get("compute_commitment_id", "")),
                    notes="Non-additive commitment observation; overlap_group remains controlling.",
                )
            )
        capacity = _number(record.get("capacity_value"))
        if capacity is not None:
            rows.append(
                _ledger_observation(
                    episode=episode,
                    family="capacity",
                    metric="contracted_compute_capacity",
                    entity=entity,
                    period_end=available,
                    available_date=available,
                    value=capacity,
                    unit=str(record.get("capacity_unit", "")),
                    claim_id=str(record.get("source_claim_id", "")),
                    source_lookup=source_lookup,
                    record_id=str(record.get("compute_commitment_id", "")),
                    notes="Contracted capacity; not evidence of delivered or utilized capacity.",
                )
            )

    leases = _read_csv(root / "leases.csv")
    for _, record in leases.iterrows():
        value = _number(record.get("disclosed_commitment"))
        available = _claim_effective_date(
            str(record.get("source_claim_id", "")), root / "claims.csv"
        )
        if value is None or pd.isna(available) or available > cutoff:
            continue
        rows.append(
            _ledger_observation(
                episode=episode,
                family="commitment_overhang",
                metric="undiscounted_lease_commitment",
                entity=entity_lookup.get(
                    str(record.get("lessee_entity_id", "")),
                    str(record.get("lessee_entity_id", "")),
                ),
                period_end=available,
                available_date=available,
                value=value,
                unit=str(record.get("currency", "USD")),
                claim_id=str(record.get("source_claim_id", "")),
                source_lookup=source_lookup,
                record_id=str(record.get("lease_id", "")),
                notes="Undiscounted lease commitment; not additive to recognized lease liability.",
            )
        )

    exposures = _read_csv(root / "economic_exposures.csv")
    financing_types = {
        "debt",
        "project_debt",
        "lease_liability",
        "supplier_financing",
        "private_credit",
    }
    for _, record in exposures.iterrows():
        if str(record.get("exposure_type", "")) not in financing_types:
            continue
        value = _number(record.get("amount_high")) or _number(record.get("amount_low"))
        available = pd.to_datetime(record.get("as_of_date"), errors="coerce")
        if value is None or pd.isna(available) or available > cutoff:
            continue
        rows.append(
            _ledger_observation(
                episode=episode,
                family="financing",
                metric=f"exposure_{record.get('exposure_type', 'financing')}",
                entity=entity_lookup.get(
                    str(record.get("legal_obligor_entity_id", "")),
                    str(record.get("legal_obligor_entity_id", "")),
                ),
                period_end=available,
                available_date=available,
                value=value,
                unit=str(record.get("currency", "USD")),
                claim_id=str(record.get("source_claim_id", "")),
                source_lookup=source_lookup,
                record_id=str(record.get("exposure_id", "")),
                notes="Exposure stock; overlap and recognition status remain attached in source ledger.",
            )
        )

    observations = pd.DataFrame(rows)
    if not observations.empty:
        observations = observations[
            observations["source_url"].fillna("").astype(str).str.startswith("http")
        ].copy()
        observations = observations.drop_duplicates(
            ["family", "metric", "entity", "available_date", "accession_or_release_id"],
            keep="last",
        )
    for dataset, frame in (
        ("contracts", contracts),
        ("compute_commitments", commitments),
        ("leases", leases),
        ("economic_exposures", exposures),
    ):
        inventory.append(
            {
                "dataset": dataset,
                "input_rows": len(frame),
                "extracted_rows": (
                    int(observations["accession_or_release_id"].astype(str).str.startswith(
                        {"contracts": "CON-", "compute_commitments": "CMP-", "leases": "LEASE-", "economic_exposures": "EXP-"}[dataset]
                    ).sum())
                    if not observations.empty
                    else 0
                ),
                "status": "imported_existing_governed_ledger",
                "allocation_authority": 0.0,
            }
        )
    return observations, pd.DataFrame(inventory)


def import_archived_inline_xbrl(
    *,
    entity_manifest: pd.DataFrame,
    project_dir: str | Path = DEFAULT_AI_FINANCIAL_SYSTEM_DIR,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract standardized facts from hash-preserved official inline-XBRL filings."""

    project = Path(project_dir)
    repo_root = project.parents[1]
    sources = _read_csv(project / "data" / "source_register.csv")
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.max
    cik_lookup = {
        int(row["cik"]): row
        for _, row in entity_manifest.iterrows()
        if pd.notna(pd.to_numeric(row.get("cik"), errors="coerce"))
    }
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    if sources.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates = sources[
        sources["source_type"].astype(str).isin({"sec_10q", "sec_10k", "sec_20f"})
        & sources["archived_path"].fillna("").astype(str).ne("")
    ]
    for _, source in candidates.iterrows():
        cik_match = _cik_from_sec_url(str(source.get("url", "")))
        if cik_match is None or cik_match not in cik_lookup:
            continue
        entity = cik_lookup[cik_match]
        filed = pd.to_datetime(source.get("publication_date"), errors="coerce")
        if pd.isna(filed) or filed > cutoff:
            continue
        archive_path = repo_root / str(source["archived_path"])
        if not archive_path.exists():
            inventory.append(
                {
                    "source_id": source["source_id"],
                    "entity": entity["entity"],
                    "status": "archived_path_missing",
                    "raw_path": str(archive_path),
                    "extracted_rows": 0,
                    "allocation_authority": 0.0,
                }
            )
            continue
        content = archive_path.read_bytes()
        try:
            extracted = _extract_inline_xbrl_facts(
                content,
                entity=entity,
                source=source,
                filed=pd.Timestamp(filed),
                source_hash=hashlib.sha256(content).hexdigest(),
            )
            status = "imported_archived_inline_xbrl"
            error = ""
        except (OSError, ValueError, TypeError) as exc:
            extracted = pd.DataFrame()
            status = "parse_failed"
            error = str(exc)
        rows.extend(extracted.to_dict("records"))
        inventory.append(
            {
                "source_id": source["source_id"],
                "entity": entity["entity"],
                "status": status,
                "raw_path": str(archive_path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "extracted_rows": len(extracted),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    observations = pd.DataFrame(rows)
    if not observations.empty:
        observations = observations.drop_duplicates(
            ["entity", "metric", "period_end", "available_date", "accession_or_release_id"],
            keep="last",
        )
    return observations, pd.DataFrame(inventory)


def audit_ai_family_coverage(
    observations: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    policy: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Separate observed families from fresh, diverse, model-usable families."""

    cutoff = pd.Timestamp(as_of).normalize()
    rules = policy or AI_FAMILY_POLICY
    clean = observations.copy()
    if not clean.empty:
        clean["available_date"] = pd.to_datetime(clean["available_date"], errors="coerce")
        clean = clean[clean["available_date"].le(cutoff)].copy()
        if "eligible_for_model" in clean:
            clean = clean[clean["eligible_for_model"].fillna(False).astype(bool)].copy()
    rows = []
    for family in REQUIRED_DIRECT_FAMILIES:
        group = clean[clean.get("family", pd.Series(dtype=str)).astype(str).eq(family)]
        latest = group["available_date"].max() if not group.empty else pd.NaT
        age_days = (cutoff - latest).days if pd.notna(latest) else pd.NA
        family_rule = rules[family]
        entities = group.get("entity", pd.Series(dtype=str)).replace("", pd.NA).nunique()
        sources = group.get("source_id", pd.Series(dtype=str)).replace("", pd.NA).nunique()
        blockers = []
        if group.empty:
            blockers.append("no_governed_pit_observation")
        if len(group) < family_rule["minimum_rows"]:
            blockers.append("insufficient_observation_rows")
        if entities < family_rule["minimum_entities"]:
            blockers.append("insufficient_entity_breadth")
        if pd.isna(age_days) or int(age_days) > family_rule["maximum_age_days"]:
            blockers.append("stale_or_undated")
        rows.append(
            {
                "vintage_as_of": cutoff,
                "family": family,
                "observation_rows": len(group),
                "distinct_entities": entities,
                "distinct_sources": sources,
                "latest_available_date": latest,
                "age_days": age_days,
                "maximum_age_days": family_rule["maximum_age_days"],
                "minimum_rows": family_rule["minimum_rows"],
                "minimum_entities": family_rule["minimum_entities"],
                "observed": not group.empty,
                "available": not blockers,
                "missing_reason": ";".join(blockers),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_current_ai_evidence(
    *,
    entity_manifest: pd.DataFrame,
    as_of: str | pd.Timestamp,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    ai_financial_system_dir: str | Path = DEFAULT_AI_FINANCIAL_SYSTEM_DIR,
    fetch_sec: bool = True,
    force_sec: bool = False,
    sec_user_agent: str | None = None,
) -> CurrentAiEvidenceBuild:
    """Build the current-AI direct panel from official SEC facts and governed ledgers."""

    destination = Path(research_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if fetch_sec:
        sec_download = fetch_sec_companyfacts_archive(
            entity_manifest=entity_manifest,
            research_dir=destination,
            user_agent=sec_user_agent,
            force=force_sec,
        )
    else:
        sec_download = pd.DataFrame()
    sec_result = import_sec_companyfacts_panel(
        entity_manifest=entity_manifest,
        research_dir=destination,
        write_outputs=False,
    )
    ledgers, ledger_inventory = import_ai_financial_system_ledgers(
        project_dir=ai_financial_system_dir,
        as_of=as_of,
    )
    inline_facts, inline_inventory = import_archived_inline_xbrl(
        entity_manifest=entity_manifest,
        project_dir=ai_financial_system_dir,
        as_of=as_of,
    )
    combined = _merge_observations(sec_result.observations, inline_facts)
    combined = _merge_observations(combined, ledgers)
    if not inline_inventory.empty:
        inline_inventory = inline_inventory.assign(dataset="archived_inline_xbrl")
        ledger_inventory = pd.concat(
            [ledger_inventory, inline_inventory], ignore_index=True, sort=False
        )
    cutoff = pd.Timestamp(as_of).normalize()
    if not combined.empty:
        combined["available_date"] = pd.to_datetime(combined["available_date"], errors="coerce")
        combined = combined[combined["available_date"].le(cutoff)].copy()
    coverage = audit_ai_family_coverage(combined, as_of=cutoff)
    observation_path = destination / "current_ai_observations.csv"
    coverage_path = destination / "current_ai_coverage.csv"
    manifest_path = destination / "current_ai_build_manifest.json"
    combined.to_csv(observation_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    manifest = {
        "as_of": cutoff.date().isoformat(),
        "observation_rows": len(combined),
        "observed_families": int(coverage["observed"].sum()),
        "usable_families": int(coverage["available"].sum()),
        "required_families": len(REQUIRED_DIRECT_FAMILIES),
        "sec_fetch_attempted": fetch_sec,
        "sec_downloaded": int(sec_download.get("status", pd.Series(dtype=str)).eq("downloaded").sum()),
        "sec_download_failures": int(sec_download.get("status", pd.Series(dtype=str)).eq("download_failed").sum()),
        "sec_local_imported": int(
            sec_result.archive_inventory.get("status", pd.Series(dtype=str))
            .eq("imported")
            .sum()
        ),
        "allocation_authority": 0.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return CurrentAiEvidenceBuild(
        observations=combined,
        coverage=coverage,
        sec_inventory=sec_result.archive_inventory,
        ledger_inventory=ledger_inventory,
        observation_path=observation_path,
        coverage_path=coverage_path,
        manifest_path=manifest_path,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _cik_from_sec_url(url: str) -> int | None:
    marker = "/data/"
    if marker not in url:
        return None
    candidate = url.split(marker, 1)[1].split("/", 1)[0]
    return int(candidate) if candidate.isdigit() else None


def _extract_inline_xbrl_facts(
    content: bytes,
    *,
    entity: pd.Series,
    source: pd.Series,
    filed: pd.Timestamp,
    source_hash: str,
) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    contexts: dict[str, tuple[pd.Timestamp, pd.Timestamp, bool]] = {}
    for context in soup.find_all(lambda tag: isinstance(tag, Tag) and tag.name.endswith("context")):
        context_id = str(context.get("id", ""))
        start_tag = context.find(lambda tag: isinstance(tag, Tag) and tag.name.endswith("startdate"))
        end_tag = context.find(lambda tag: isinstance(tag, Tag) and tag.name.endswith("enddate"))
        instant_tag = context.find(lambda tag: isinstance(tag, Tag) and tag.name.endswith("instant"))
        if instant_tag is not None:
            start = end = pd.to_datetime(instant_tag.get_text(strip=True), errors="coerce")
        else:
            start = pd.to_datetime(
                start_tag.get_text(strip=True) if start_tag is not None else None,
                errors="coerce",
            )
            end = pd.to_datetime(
                end_tag.get_text(strip=True) if end_tag is not None else None,
                errors="coerce",
            )
        has_dimensions = context.find(
            lambda tag: isinstance(tag, Tag)
            and (tag.name.endswith("explicitmember") or tag.name.endswith("typedmember"))
        ) is not None
        if context_id and pd.notna(end):
            contexts[context_id] = (pd.Timestamp(start), pd.Timestamp(end), has_dimensions)
    rows: list[dict[str, object]] = []
    for fact in soup.find_all(
        lambda tag: isinstance(tag, Tag)
        and tag.name.endswith("nonfraction")
        and str(tag.get("name", "")).startswith("us-gaap:")
    ):
        tag_name = str(fact.get("name", "")).split(":", 1)[-1]
        if tag_name not in SEC_FACT_TAGS:
            continue
        context_id = str(fact.get("contextref", fact.get("contextRef", "")))
        if context_id not in contexts:
            continue
        start, period_end, has_dimensions = contexts[context_id]
        if has_dimensions:
            continue
        raw = fact.get_text("", strip=True).replace(",", "").replace("$", "")
        value = pd.to_numeric(raw.replace("(", "-").replace(")", ""), errors="coerce")
        if pd.isna(value):
            continue
        scale = pd.to_numeric(fact.get("scale", 0), errors="coerce")
        sign = -1.0 if str(fact.get("sign", "")) == "-" else 1.0
        multiplier = 10.0 ** float(scale) if pd.notna(scale) else 1.0
        family, metric_name = SEC_FACT_TAGS[tag_name]
        accession = str(source.get("url", "")).rstrip("/").split("/")[-2]
        rows.append(
            {
                "episode": entity.get("episode", DEFAULT_AI_EPISODE),
                "family": family,
                "metric": f"sec_{metric_name}",
                "entity": entity["entity"],
                "period_end": period_end,
                "available_date": filed,
                "value": float(value) * multiplier * sign,
                "unit": "USD",
                "source_id": source["source_id"],
                "source_url": source["url"],
                "source_tier": source.get("evidence_grade", "A"),
                "vintage_status": "filing_as_reported",
                "eligible_for_model": True,
                "revision_status": "value_as_contained_in_archived_inline_xbrl",
                "accession_or_release_id": accession,
                "source_archive_sha256": source_hash,
                "form": source.get("source_type", ""),
                "duration_days": (period_end - start).days if pd.notna(start) else pd.NA,
                "notes": (
                    "Consolidated no-dimension standardized fact from a hash-preserved "
                    "official filing. Duration facts require fiscal-period normalization."
                ),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _first_date(record: pd.Series, *columns: str) -> pd.Timestamp:
    for column in columns:
        parsed = pd.to_datetime(record.get(column), errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)
    return pd.NaT


def _claim_effective_date(claim_id: str, claim_path: Path) -> pd.Timestamp:
    claims = _read_csv(claim_path)
    if claims.empty:
        return pd.NaT
    match = claims[claims["claim_id"].astype(str).eq(claim_id)]
    if match.empty:
        return pd.NaT
    return pd.to_datetime(match.iloc[0].get("effective_date"), errors="coerce")


def _source_lookup(sources: pd.DataFrame, links: pd.DataFrame) -> dict[str, dict[str, str]]:
    if sources.empty or links.empty:
        return {}
    merged = links.merge(sources, on="source_id", how="left")
    merged = merged.sort_values(["claim_id", "evidence_grade", "source_id"])
    lookup: dict[str, dict[str, str]] = {}
    for claim_id, group in merged.groupby("claim_id"):
        row = group.iloc[0]
        lookup[str(claim_id)] = {
            "source_id": str(row.get("source_id", "")),
            "source_url": str(row.get("url", "")),
            "source_tier": str(row.get("evidence_grade", "")),
        }
    return lookup


def _entity_pair(
    record: pd.Series,
    lookup: dict[object, object],
    left: str,
    right: str,
) -> str:
    identifiers = [str(record.get(left, "")), str(record.get(right, ""))]
    names = [str(lookup.get(identifier, identifier)) for identifier in identifiers if identifier]
    return " / ".join(names)


def _ledger_observation(
    *,
    episode: str,
    family: str,
    metric: str,
    entity: str,
    period_end: pd.Timestamp,
    available_date: pd.Timestamp,
    value: float,
    unit: str,
    claim_id: str,
    source_lookup: dict[str, dict[str, str]],
    record_id: str,
    notes: str,
) -> dict[str, object]:
    source = source_lookup.get(claim_id, {})
    return {
        "episode": episode,
        "family": family,
        "metric": metric,
        "entity": entity,
        "period_end": period_end,
        "available_date": available_date,
        "value": value,
        "unit": unit,
        "source_id": source.get("source_id", ""),
        "source_url": source.get("source_url", ""),
        "source_tier": source.get("source_tier", ""),
        "vintage_status": "immutable_transaction_record",
        "eligible_for_model": bool(source.get("source_url", "")),
        "revision_status": "governed_monograph_ledger_as_published",
        "accession_or_release_id": record_id,
        "source_claim_id": claim_id,
        "notes": notes,
        "allocation_authority": 0.0,
    }
