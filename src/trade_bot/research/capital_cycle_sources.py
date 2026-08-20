from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from trade_bot.research.capital_cycle_feasibility import DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR

CENSUS_NRC_URL_TEMPLATE: Final[str] = (
    "https://www.census.gov/construction/nrc/pdf/newresconst_{period}.pdf"
)
CENSUS_NRC_LEGACY_TEXT_URLS: Final[dict[str, str]] = {
    "200103": "https://www.census.gov/construction/nrc/txt/c20_0103.txt",
}
CENSUS_HOUSING_EPISODES: Final[tuple[str, ...]] = (
    "housing_mortgage_2001_2011",
    "housing_recovery_2012_2019",
)
MONTH_NAME_PATTERN: Final[str] = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


@dataclass(frozen=True)
class CensusArchiveResult:
    observations: pd.DataFrame
    archive_inventory: pd.DataFrame
    observation_path: Path
    inventory_path: Path


@dataclass(frozen=True)
class SecCompanyFactsImportResult:
    observations: pd.DataFrame
    archive_inventory: pd.DataFrame
    observation_path: Path
    inventory_path: Path


@dataclass(frozen=True)
class SecFilingArchiveResult:
    filing_inventory: pd.DataFrame
    inventory_path: Path


@dataclass(frozen=True)
class RegisteredSourceArchiveResult:
    archive_inventory: pd.DataFrame
    inventory_path: Path


@dataclass(frozen=True)
class DirectObservationImportResult:
    observations: pd.DataFrame
    observation_path: Path


SEC_FACT_TAGS: Final[dict[str, tuple[str, str]]] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("monetization", "revenue"),
    "Revenues": ("monetization", "revenue"),
    "SalesRevenueNet": ("monetization", "revenue"),
    "OperatingIncomeLoss": ("monetization", "operating_income"),
    "NetCashProvidedByUsedInOperatingActivities": ("monetization", "operating_cash_flow"),
    "PaymentsToAcquirePropertyPlantAndEquipment": ("capacity", "capital_expenditure"),
    "LongTermDebtCurrent": ("financing", "long_term_debt_current"),
    "LongTermDebtNoncurrent": ("financing", "long_term_debt_noncurrent"),
    "LongTermDebtAndFinanceLeaseObligationsCurrent": (
        "financing",
        "debt_and_finance_lease_current",
    ),
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent": (
        "financing",
        "debt_and_finance_lease_noncurrent",
    ),
}
DEFAULT_SEC_USER_AGENT: Final[str] = (
    "trade-bot capital-cycle research contact github.com/Jbultema"
)
ARCHIVABLE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pdf", ".csv", ".xls", ".xlsx", ".json", ".txt", ".zip"}
)


def fetch_registered_source_documents(
    *,
    source_registry: pd.DataFrame,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    episode: str | None = None,
    maximum_documents_per_source: int = 12,
    timeout_seconds: float = 30.0,
    force: bool = False,
) -> RegisteredSourceArchiveResult:
    """Discover and archive linked official documents from governed source contracts."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "registered_sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    registry = source_registry.copy()
    if episode is not None:
        registry = registry[registry["episode"].astype(str).eq(episode)]
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_SEC_USER_AGENT})
    rows: list[dict[str, object]] = []
    last_request = 0.0
    for _, source in registry.iterrows():
        landing_url = str(source.get("source_url", ""))
        if not landing_url.startswith("http"):
            continue
        source_id = str(source["source_id"])
        landing_suffix = Path(urlparse(landing_url).path).suffix.lower()
        landing_path = raw_dir / (
            f"{_safe_component(source_id)}_landing"
            f"{landing_suffix if landing_suffix in ARCHIVABLE_SUFFIXES else '.html'}"
        )
        landing, status, error, last_request = _fetch_cached_bytes(
            session,
            landing_url,
            landing_path,
            force=force,
            timeout_seconds=timeout_seconds,
            minimum_interval_seconds=0.12,
            last_request=last_request,
        )
        if not landing:
            rows.append(
                _registered_source_inventory_row(
                    source,
                    url=landing_url,
                    path=landing_path,
                    payload=b"",
                    status=f"landing_{status}",
                    error=error,
                )
            )
            continue
        if landing_suffix in ARCHIVABLE_SUFFIXES:
            document_path = raw_dir / (
                f"{_safe_component(source_id)}_000_"
                f"{hashlib.sha256(landing_url.encode()).hexdigest()[:12]}{landing_suffix}"
            )
            if not document_path.exists() or force:
                document_path.write_bytes(landing)
            rows.append(
                _registered_source_inventory_row(
                    source,
                    url=landing_url,
                    path=document_path,
                    payload=landing,
                    status=status,
                    error=error,
                )
            )
            continue
        else:
            candidates = _document_links(
                landing,
                landing_url=landing_url,
                keywords=_source_keywords(source),
            )
        if not candidates:
            rows.append(
                _registered_source_inventory_row(
                    source,
                    url=landing_url,
                    path=landing_path,
                    payload=landing,
                    status="landing_archived_no_document_links",
                    error="",
                )
            )
            continue
        for index, document_url in enumerate(candidates[:maximum_documents_per_source]):
            suffix = Path(urlparse(document_url).path).suffix.lower() or ".bin"
            document_path = raw_dir / (
                f"{_safe_component(source_id)}_{index:03d}_"
                f"{hashlib.sha256(document_url.encode()).hexdigest()[:12]}{suffix}"
            )
            payload, document_status, document_error, last_request = _fetch_cached_bytes(
                session,
                document_url,
                document_path,
                force=force,
                timeout_seconds=timeout_seconds,
                minimum_interval_seconds=0.12,
                last_request=last_request,
            )
            rows.append(
                _registered_source_inventory_row(
                    source,
                    url=document_url,
                    path=document_path,
                    payload=payload,
                    status=document_status,
                    error=document_error,
                )
            )
    inventory = pd.DataFrame(rows)
    inventory_path = destination / "registered_source_archive_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    return RegisteredSourceArchiveResult(
        archive_inventory=inventory,
        inventory_path=inventory_path,
    )


def fetch_sec_filing_archive(
    *,
    entity_manifest: pd.DataFrame,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    forms: tuple[str, ...] = ("10-K", "10-Q", "20-F", "40-F"),
    maximum_filings_per_entity: int = 48,
    user_agent: str | None = None,
    force: bool = False,
    timeout_seconds: float = 30.0,
    minimum_interval_seconds: float = 0.12,
    filing_seed_manifest: pd.DataFrame | None = None,
) -> SecFilingArchiveResult:
    """Discover and preserve official filing HTML, including pre-XBRL episodes."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "sec_filings"
    submissions_dir = destination / "raw" / "sec_submissions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent
            or os.getenv("TRADE_BOT_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    inventory: list[dict[str, object]] = []
    last_request = 0.0
    for _, entity in entity_manifest.iterrows():
        cik = int(entity["cik"])
        entity_seeds = pd.DataFrame()
        if filing_seed_manifest is not None and not filing_seed_manifest.empty:
            seed_ciks = pd.to_numeric(
                filing_seed_manifest.get("cik", pd.Series(dtype=float)), errors="coerce"
            )
            entity_seeds = filing_seed_manifest[seed_ciks.eq(cik)].copy()
        seeded_accessions: set[str] = set()
        for _, seed in entity_seeds.iterrows():
            seed_row, last_request = _archive_seeded_sec_filing(
                session,
                entity=entity,
                seed=seed,
                raw_dir=raw_dir,
                force=force,
                timeout_seconds=timeout_seconds,
                minimum_interval_seconds=minimum_interval_seconds,
                last_request=last_request,
            )
            inventory.append(seed_row)
            seeded_accessions.add(str(seed.get("accession_number", "")))
        submission_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        submission_path = submissions_dir / f"CIK{cik:010d}.json"
        payload, status, error, last_request = _fetch_cached_bytes(
            session,
            submission_url,
            submission_path,
            force=force,
            timeout_seconds=timeout_seconds,
            minimum_interval_seconds=minimum_interval_seconds,
            last_request=last_request,
        )
        if not payload:
            inventory.append(
                _sec_filing_inventory_error(
                    entity,
                    source_url=submission_url,
                    raw_path=submission_path,
                    status=f"submissions_{status}",
                    error=error,
                )
            )
            continue
        try:
            submission = json.loads(payload)
        except json.JSONDecodeError as exc:
            inventory.append(
                _sec_filing_inventory_error(
                    entity,
                    source_url=submission_url,
                    raw_path=submission_path,
                    status="submissions_parse_failed",
                    error=str(exc),
                )
            )
            continue
        filing_rows = _submission_filing_rows(submission)
        filing_container = submission.get("filings", {})
        historical_files = (
            filing_container.get("files", [])
            if isinstance(filing_container, dict)
            else []
        )
        historical_frames = []
        for historical in historical_files if isinstance(historical_files, list) else []:
            if not isinstance(historical, dict) or not historical.get("name"):
                continue
            name = str(historical["name"])
            historical_url = f"https://data.sec.gov/submissions/{name}"
            historical_path = submissions_dir / name
            historical_payload, _, _, last_request = _fetch_cached_bytes(
                session,
                historical_url,
                historical_path,
                force=force,
                timeout_seconds=timeout_seconds,
                minimum_interval_seconds=minimum_interval_seconds,
                last_request=last_request,
            )
            if not historical_payload:
                continue
            try:
                historical_frames.append(
                    _submission_filing_rows(json.loads(historical_payload))
                )
            except json.JSONDecodeError:
                continue
        if historical_frames:
            available_frames = [
                frame for frame in [filing_rows, *historical_frames] if not frame.empty
            ]
            filing_rows = pd.concat(
                available_frames, ignore_index=True
            ).drop_duplicates("accession_number", keep="last")
        filing_start = pd.to_datetime(entity.get("filing_start"), errors="coerce")
        filing_end = pd.to_datetime(entity.get("filing_end"), errors="coerce")
        eligible = filing_rows[
            filing_rows["form"].astype(str).isin(forms)
            & filing_rows["filing_date"].between(filing_start, filing_end)
        ].sort_values("filing_date", ascending=False)
        eligible = eligible.head(maximum_filings_per_entity)
        for _, filing in eligible.iterrows():
            accession = str(filing["accession_number"])
            if accession in seeded_accessions:
                continue
            accession_compact = accession.replace("-", "")
            primary_document = str(filing["primary_document"])
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{accession_compact}/{primary_document}"
            )
            local_dir = raw_dir / f"CIK{cik:010d}"
            local_dir.mkdir(parents=True, exist_ok=True)
            filing_path = local_dir / f"{accession_compact}_{Path(primary_document).name}"
            content, filing_status, filing_error, last_request = _fetch_cached_bytes(
                session,
                filing_url,
                filing_path,
                force=force,
                timeout_seconds=timeout_seconds,
                minimum_interval_seconds=minimum_interval_seconds,
                last_request=last_request,
            )
            inventory.append(
                {
                    "episode": entity["episode"],
                    "entity": entity["entity"],
                    "sec_reported_name": submission.get("name", ""),
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "accession_number": accession,
                    "primary_document": primary_document,
                    "source_url": filing_url,
                    "raw_path": str(filing_path),
                    "status": filing_status,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest() if content else "",
                    "error": filing_error,
                    "allocation_authority": 0.0,
                }
            )
    output = pd.DataFrame(inventory)
    inventory_path = destination / "sec_filing_archive_inventory.csv"
    output.to_csv(inventory_path, index=False)
    return SecFilingArchiveResult(filing_inventory=output, inventory_path=inventory_path)


def import_fcc_telecom_capacity_release(
    *,
    pdf_path: str | Path,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    source_url: str = "https://docs.fcc.gov/public/attachments/DOC-215526A1.pdf",
) -> DirectObservationImportResult:
    """Import the FCC's August 2001 local-transmission capacity history honestly."""

    path = Path(pdf_path)
    content = path.read_bytes()
    observations = parse_fcc_telecom_capacity_release(
        content,
        source_url=source_url,
        source_hash=hashlib.sha256(content).hexdigest(),
    )
    combined = _merge_observations(existing_observations, observations)
    observation_path = Path(research_dir) / "direct_observations.csv"
    combined.to_csv(observation_path, index=False)
    return DirectObservationImportResult(
        observations=combined,
        observation_path=observation_path,
    )


def parse_fcc_telecom_capacity_release(
    content: bytes,
    *,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract Table 18.3 while retaining the report's actual information date."""

    reader = PdfReader(BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    table_start = text.rfind("Table 18.3")
    table_end = text.find("Table 18.4", table_start)
    if table_start < 0 or table_end < 0:
        raise ValueError("FCC Table 18.3 was not found in the archived report.")
    table = text[table_start:table_end]
    digital_section, separator, channel_section = table.partition(
        "Working Telecommunications Channels"
    )
    if not separator:
        raise ValueError("FCC working-channel section was not found in Table 18.3.")
    release_match = re.search(r"\b(August\s+\d{4})\b", text[:50_000], re.IGNORECASE)
    if release_match is None:
        raise ValueError("FCC report month was not found.")
    available_date = pd.to_datetime(release_match.group(1)) + pd.offsets.MonthEnd(0)
    rows: list[dict[str, object]] = []
    for metric, unit, section in (
        ("fcc_digital_transmission_links_total", "links", digital_section),
        ("fcc_working_telecom_channels_total", "thousands_of_channels", channel_section),
    ):
        for year, total in _fcc_year_total_rows(section):
            rows.append(
                {
                    "episode": "dotcom_telecom_capacity_1994_2004",
                    "family": "capacity",
                    "metric": metric,
                    "entity": "Bell Operating Companies",
                    "period_end": pd.Timestamp(year=year, month=12, day=31),
                    "available_date": available_date,
                    "value": total,
                    "unit": unit,
                    "source_id": "fcc_trends_telephone_service_200108_table_18_3",
                    "source_url": source_url,
                    "source_tier": "A",
                    "vintage_status": "archived_release",
                    "eligible_for_model": True,
                    "revision_status": "history_as_published_in_august_2001_release",
                    "accession_or_release_id": source_hash,
                    "source_archive_sha256": source_hash,
                    "notes": (
                        "All historical values first enter this panel at the August 2001 "
                        "report date; they are not backdated to the measurement year. "
                        "Table footnotes identify merger-related level shifts in 1999-2000."
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 22:
        raise ValueError(f"Expected 22 FCC capacity observations, extracted {len(result)}.")
    return result


def fetch_census_housing_release_panel(
    *,
    episode_manifest: pd.DataFrame,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    quarterly_only: bool = True,
    timeout_seconds: float = 30.0,
) -> CensusArchiveResult:
    """Archive and extract contemporaneous Census housing-release headline values."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "census_nrc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    observation_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    periods = _housing_release_periods(episode_manifest, quarterly_only=quarterly_only)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "trade-bot-capital-cycle-research github.com/Jbultema",
            "Accept": "application/pdf",
        }
    )
    for episode, period in periods:
        period_label = period.strftime("%Y%m")
        url = CENSUS_NRC_LEGACY_TEXT_URLS.get(
            period_label,
            CENSUS_NRC_URL_TEMPLATE.format(period=period_label),
        )
        suffix = ".txt" if period_label in CENSUS_NRC_LEGACY_TEXT_URLS else ".pdf"
        raw_path = raw_dir / f"newresconst_{period_label}{suffix}"
        status = "cached"
        error = ""
        if raw_path.exists():
            content = raw_path.read_bytes()
        else:
            try:
                response = session.get(url, timeout=timeout_seconds)
                response.raise_for_status()
                content = response.content
                raw_path.write_bytes(content)
                status = "downloaded"
            except requests.RequestException as exc:
                content = b""
                status = "download_failed"
                error = str(exc)
        parsed = pd.DataFrame()
        if content:
            try:
                parser = (
                    parse_census_housing_text_release
                    if suffix == ".txt"
                    else parse_census_housing_release
                )
                parsed = parser(
                    content,
                    episode=episode,
                    period=period,
                    source_url=url,
                    source_hash=hashlib.sha256(content).hexdigest(),
                )
            except (OSError, ValueError, KeyError, re.error) as exc:
                status = "parse_failed"
                error = str(exc)
        observation_rows.extend(parsed.to_dict("records"))
        inventory_rows.append(
            {
                "episode": episode,
                "period": period,
                "source_url": url,
                "raw_path": str(raw_path),
                "sha256": hashlib.sha256(content).hexdigest() if content else "",
                "bytes": len(content),
                "status": status,
                "extracted_rows": len(parsed),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    new_observations = pd.DataFrame(observation_rows)
    combined = _merge_observations(existing_observations, new_observations)
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "census_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return CensusArchiveResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def import_sec_companyfacts_panel(
    *,
    entity_manifest: pd.DataFrame,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    write_outputs: bool = True,
) -> SecCompanyFactsImportResult:
    """Import locally archived SEC Company Facts without weakening vintage controls."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "sec_companyfacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    observation_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for _, entity in entity_manifest.iterrows():
        cik = int(entity["cik"])
        raw_path = raw_dir / f"CIK{cik:010d}.json"
        if not raw_path.exists():
            inventory_rows.append(_missing_sec_inventory_row(entity, raw_path))
            continue
        content = raw_path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        try:
            payload = json.loads(content)
            extracted = _extract_sec_entity_facts(payload, entity, source_hash=source_hash)
            status = "imported"
            error = ""
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            extracted = pd.DataFrame()
            status = "parse_failed"
            error = str(exc)
        observation_rows.extend(extracted.to_dict("records"))
        inventory_rows.append(
            {
                "episode": entity["episode"],
                "entity": entity["entity"],
                "cik": cik,
                "raw_path": str(raw_path),
                "sha256": source_hash,
                "bytes": len(content),
                "status": status,
                "extracted_rows": len(extracted),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    combined = _merge_observations(existing_observations, pd.DataFrame(observation_rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "sec_companyfacts_inventory.csv"
    if write_outputs:
        combined.to_csv(observation_path, index=False)
        inventory.to_csv(inventory_path, index=False)
    return SecCompanyFactsImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def _extract_sec_entity_facts(
    payload: dict[str, object],
    entity: pd.Series,
    *,
    source_hash: str,
) -> pd.DataFrame:
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        raise ValueError("Company Facts payload has no facts object.")
    us_gaap = facts.get("us-gaap", {})
    if not isinstance(us_gaap, dict):
        raise ValueError("Company Facts payload has no us-gaap facts.")
    filing_start = pd.to_datetime(entity["filing_start"], errors="raise")
    filing_end = pd.to_datetime(entity["filing_end"], errors="raise")
    rows: list[dict[str, object]] = []
    for tag, (family, metric_name) in SEC_FACT_TAGS.items():
        fact = us_gaap.get(tag)
        if not isinstance(fact, dict):
            continue
        units = fact.get("units", {})
        if not isinstance(units, dict):
            continue
        entries = units.get("USD", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            filed = pd.to_datetime(entry.get("filed"), errors="coerce")
            period_end = pd.to_datetime(entry.get("end"), errors="coerce")
            if pd.isna(filed) or pd.isna(period_end):
                continue
            if filed < filing_start or filed > filing_end + pd.DateOffset(years=1):
                continue
            if period_end < filing_start - pd.DateOffset(years=1) or period_end > filing_end:
                continue
            form = str(entry.get("form", ""))
            if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                continue
            start = pd.to_datetime(entry.get("start"), errors="coerce")
            duration_days = (period_end - start).days if pd.notna(start) else pd.NA
            accession = str(entry.get("accn", ""))
            rows.append(
                {
                    "episode": entity["episode"],
                    "family": family,
                    "metric": f"sec_{metric_name}",
                    "entity": entity["entity"],
                    "period_end": period_end,
                    "available_date": filed,
                    "value": pd.to_numeric(entry.get("val"), errors="coerce"),
                    "unit": "USD",
                    "source_id": f"sec_companyfacts_cik_{int(entity['cik']):010d}",
                    "source_url": (
                        "https://www.sec.gov/Archives/edgar/data/"
                        f"{int(entity['cik'])}/{accession.replace('-', '')}"
                    ),
                    "source_tier": "A",
                    "vintage_status": "filing_as_reported",
                    "eligible_for_model": True,
                    "revision_status": "value_as_contained_in_filed_accession",
                    "accession_or_release_id": accession,
                    "source_archive_sha256": source_hash,
                    "form": form,
                    "fiscal_year": entry.get("fy"),
                    "fiscal_period": entry.get("fp"),
                    "duration_days": duration_days,
                    "notes": (
                        "Locally archived official SEC Company Facts value. "
                        "Duration facts require fiscal-period normalization before modeling."
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keys = ["episode", "entity", "metric", "period_end", "available_date", "accession_or_release_id"]
    return frame.drop_duplicates(keys, keep="last")


def _missing_sec_inventory_row(entity: pd.Series, raw_path: Path) -> dict[str, object]:
    return {
        "episode": entity["episode"],
        "entity": entity["entity"],
        "cik": int(entity["cik"]),
        "raw_path": str(raw_path),
        "sha256": "",
        "bytes": 0,
        "status": "raw_file_missing",
        "extracted_rows": 0,
        "error": "Official Company Facts JSON must be acquired outside the blocked environment.",
        "allocation_authority": 0.0,
    }


def parse_census_housing_release(
    content: bytes,
    *,
    episode: str,
    period: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    reader = PdfReader(BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _parse_census_release_text(
        text,
        episode=episode,
        period=period,
        source_url=source_url,
        source_hash=source_hash,
    )


def parse_census_housing_text_release(
    content: bytes,
    *,
    episode: str,
    period: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    return _parse_census_release_text(
        content.decode("latin-1"),
        episode=episode,
        period=period,
        source_url=source_url,
        source_hash=source_hash,
    )


def _parse_census_release_text(
    text: str,
    *,
    episode: str,
    period: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    normalized = " ".join(text.replace("\xa0", " ").split())
    release_date = _release_date_from_text(normalized, period)
    patterns = {
        "housing_permits_saar_thousands": (
            "demand",
            rf"(?:authorized by building permits in (?:{MONTH_NAME_PATTERN}) (?:\d{{4}} )?were at .*?rate of|rate of housing units authorized by building permits in (?:{MONTH_NAME_PATTERN}) (?:\d{{4}} )?was) ([\d,]+)",
        ),
        "housing_starts_saar_thousands": (
            "demand",
            rf"housing starts in (?:{MONTH_NAME_PATTERN}) (?:\d{{4}} )?were at .*?rate of ([\d,]+)",
        ),
        "housing_completions_saar_thousands": (
            "capacity",
            rf"housing completions in (?:{MONTH_NAME_PATTERN}) (?:\d{{4}} )?were at .*?rate of ([\d,]+)",
        ),
    }
    rows = []
    for metric, (family, pattern) in patterns.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is None:
            continue
        raw_value = float(match.group(1).replace(",", ""))
        value = raw_value / 1_000.0 if raw_value >= 10_000 else raw_value
        rows.append(
            {
                "episode": episode,
                "family": family,
                "metric": metric,
                "entity": "United States",
                "period_end": pd.Timestamp(period) + pd.offsets.MonthEnd(0),
                "available_date": release_date,
                "value": value,
                "unit": "thousands_saar",
                "source_id": f"census_nrc_{pd.Timestamp(period):%Y%m}",
                "source_url": source_url,
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
                "revision_status": "as_published_in_named_release",
                "accession_or_release_id": source_hash,
                "notes": "Headline preliminary or contemporaneously revised value from archived release.",
            }
        )
    if not rows:
        raise ValueError(f"No Census housing headline values parsed for {period:%Y-%m}.")
    return pd.DataFrame(rows)


def _release_date_from_text(text: str, period: pd.Timestamp) -> pd.Timestamp:
    patterns = (
        rf"(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY),? ({MONTH_NAME_PATTERN}) (\d{{1,2}}), (\d{{4}})",
        rf"FOR RELEASE AT .*? ({MONTH_NAME_PATTERN}) (\d{{1,2}}), (\d{{4}})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return pd.to_datetime(" ".join(match.groups()), errors="raise")
    raise ValueError(f"Release date was not found in the archived {period:%Y-%m} PDF.")


def _housing_release_periods(
    episode_manifest: pd.DataFrame,
    *,
    quarterly_only: bool,
) -> list[tuple[str, pd.Timestamp]]:
    rows: list[tuple[str, pd.Timestamp]] = []
    for _, episode in episode_manifest.iterrows():
        episode_name = str(episode.get("episode", ""))
        if episode_name not in CENSUS_HOUSING_EPISODES:
            continue
        start = pd.to_datetime(episode.get("start_date"), errors="coerce")
        end = pd.to_datetime(episode.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        frequency = "QE-DEC" if quarterly_only else "ME"
        for period in pd.date_range(start, end, freq=frequency):
            rows.append((episode_name, pd.Timestamp(period)))
    return rows


def _merge_observations(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
) -> pd.DataFrame:
    frames = [frame for frame in (existing, new) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for column in ("period_end", "available_date"):
        if column in combined:
            combined[column] = pd.to_datetime(
                combined[column], errors="coerce", format="mixed"
            )
    keys = [
        column
        for column in ("episode", "metric", "period_end", "available_date", "source_id")
        if column in combined
    ]
    return combined.drop_duplicates(keys, keep="last").sort_values(
        ["episode", "available_date", "family", "metric"]
    )


def _fetch_cached_bytes(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    force: bool,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    last_request: float,
) -> tuple[bytes, str, str, float]:
    if path.exists() and not force:
        return path.read_bytes(), "cached", "", last_request
    wait = minimum_interval_seconds - (time.monotonic() - last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        response = session.get(url, timeout=timeout_seconds)
        last_request = time.monotonic()
        response.raise_for_status()
        payload = response.content
        if not payload:
            raise ValueError("empty response body")
        path.write_bytes(payload)
        return payload, "downloaded", "", last_request
    except (requests.RequestException, OSError, ValueError) as exc:
        return b"", "download_failed", str(exc), last_request


def _submission_filing_rows(payload: dict[str, object]) -> pd.DataFrame:
    filings = payload.get("filings", {})
    recent = filings.get("recent", {}) if isinstance(filings, dict) else {}
    if not recent and all(
        key in payload
        for key in ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")
    ):
        recent = payload
    if not isinstance(recent, dict):
        return pd.DataFrame()
    column_map = {
        "accessionNumber": "accession_number",
        "filingDate": "filing_date",
        "reportDate": "report_date",
        "form": "form",
        "primaryDocument": "primary_document",
    }
    lengths = [len(recent.get(column, [])) for column in column_map]
    if not lengths or min(lengths) == 0:
        return pd.DataFrame(columns=column_map.values())
    row_count = min(lengths)
    frame = pd.DataFrame(
        {
            target: list(recent[source])[:row_count]
            for source, target in column_map.items()
        }
    )
    for column in ("filing_date", "report_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.dropna(subset=["filing_date", "accession_number", "primary_document"])


def _archive_seeded_sec_filing(
    session: requests.Session,
    *,
    entity: pd.Series,
    seed: pd.Series,
    raw_dir: Path,
    force: bool,
    timeout_seconds: float,
    minimum_interval_seconds: float,
    last_request: float,
) -> tuple[dict[str, object], float]:
    cik = int(entity["cik"])
    accession = str(seed.get("accession_number", ""))
    primary_document = str(seed.get("primary_document", ""))
    filing_url = str(seed.get("source_url", ""))
    if not filing_url.startswith("https://www.sec.gov/Archives/"):
        row = _sec_filing_inventory_error(
            entity,
            source_url=filing_url,
            raw_path=raw_dir / "invalid_seed",
            status="seed_invalid",
            error="Seed URL must point to the official SEC Archives host.",
        )
        return row, last_request
    accession_compact = accession.replace("-", "") or hashlib.sha256(
        filing_url.encode()
    ).hexdigest()[:18]
    local_dir = raw_dir / f"CIK{cik:010d}"
    local_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(filing_url).path).suffix or ".html"
    document_name = Path(primary_document).name if primary_document else f"filing{suffix}"
    filing_path = local_dir / f"{accession_compact}_{document_name}"
    content, status, error, last_request = _fetch_cached_bytes(
        session,
        filing_url,
        filing_path,
        force=force,
        timeout_seconds=timeout_seconds,
        minimum_interval_seconds=minimum_interval_seconds,
        last_request=last_request,
    )
    return (
        {
            "episode": entity["episode"],
            "entity": entity["entity"],
            "sec_reported_name": "",
            "cik": cik,
            "form": seed.get("form", ""),
            "filing_date": pd.to_datetime(seed.get("filing_date"), errors="coerce"),
            "report_date": pd.to_datetime(seed.get("report_date"), errors="coerce"),
            "accession_number": accession,
            "primary_document": primary_document,
            "source_url": filing_url,
            "raw_path": str(filing_path),
            "status": f"seed_{status}",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest() if content else "",
            "error": error,
            "allocation_authority": 0.0,
        },
        last_request,
    )


def _fcc_year_total_rows(section: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    pattern = re.compile(r"(?m)^(19\d{2}|2000)\s+(?:\d/\s+)?([\d,]+)")
    for match in pattern.finditer(section):
        rows.append((int(match.group(1)), float(match.group(2).replace(",", ""))))
    deduplicated = dict(rows)
    if set(deduplicated) != set(range(1990, 2001)):
        raise ValueError("FCC Table 18.3 did not contain a complete 1990-2000 history.")
    return sorted(deduplicated.items())


def _sec_filing_inventory_error(
    entity: pd.Series,
    *,
    source_url: str,
    raw_path: Path,
    status: str,
    error: str,
) -> dict[str, object]:
    return {
        "episode": entity["episode"],
        "entity": entity["entity"],
        "sec_reported_name": "",
        "cik": int(entity["cik"]),
        "form": "",
        "filing_date": pd.NaT,
        "report_date": pd.NaT,
        "accession_number": "",
        "primary_document": "",
        "source_url": source_url,
        "raw_path": str(raw_path),
        "status": status,
        "bytes": 0,
        "sha256": "",
        "error": error,
        "allocation_authority": 0.0,
    }


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "source"


def _source_keywords(source: pd.Series) -> set[str]:
    text = " ".join(
        str(source.get(column, ""))
        for column in ("metric", "provider", "episode", "notes")
    ).lower()
    tokens = set(re.findall(r"[a-z0-9]{4,}", text))
    stop = {
        "required",
        "identified",
        "historical",
        "official",
        "source",
        "filing",
        "release",
        "episode",
    }
    return tokens - stop


def _document_links(content: bytes, *, landing_url: str, keywords: set[str]) -> list[str]:
    soup = BeautifulSoup(content, "html.parser")
    scored: list[tuple[int, str]] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(landing_url, str(anchor["href"]))
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in ARCHIVABLE_SUFFIXES:
            continue
        context = f"{anchor.get_text(' ', strip=True)} {url}".lower()
        score = sum(token in context for token in keywords)
        scored.append((score, url))
    ranked = sorted(set(scored), key=lambda item: (-item[0], item[1]))
    return [url for _score, url in ranked]


def _registered_source_inventory_row(
    source: pd.Series,
    *,
    url: str,
    path: Path,
    payload: bytes,
    status: str,
    error: str,
) -> dict[str, object]:
    return {
        "source_id": source["source_id"],
        "episode": source.get("episode", ""),
        "family": source.get("family", ""),
        "provider": source.get("provider", ""),
        "source_url": url,
        "raw_path": str(path),
        "status": status,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
        "error": error,
        "allocation_authority": 0.0,
    }
