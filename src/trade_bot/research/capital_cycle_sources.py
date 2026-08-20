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
from pypdf.errors import PdfReadError

from trade_bot.research.capital_cycle_feasibility import DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR

CENSUS_NRC_URL_TEMPLATE: Final[str] = (
    "https://www.census.gov/construction/nrc/pdf/newresconst_{period}.pdf"
)
CENSUS_NRS_URL_TEMPLATE: Final[str] = (
    "https://www.census.gov/construction/nrs/pdf/newressales_{period}.pdf"
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
FED_SLOOS_INDEX_URL: Final[str] = "https://www.federalreserve.gov/data/sloos.htm"
FED_G17_INDEX_URL: Final[str] = "https://www.federalreserve.gov/Releases/g17/default.htm"
FED_G17_TECH_EPISODE_FAMILIES: Final[frozenset[str]] = frozenset(
    {"technology_hardware", "semiconductor_cycle"}
)
EIA_STEO_ARCHIVE_URL: Final[str] = "https://www.eia.gov/outlooks/steo/outlook.php"
EIA_STEO_OIL_EPISODE_FAMILIES: Final[frozenset[str]] = frozenset({"oil_shock"})
MONTH_ABBREVIATIONS: Final[dict[str, int]] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


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


@dataclass(frozen=True)
class HistoricalArchiveImportResult:
    observations: pd.DataFrame
    archive_inventory: pd.DataFrame
    observation_path: Path
    inventory_path: Path


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
DEFAULT_SEC_USER_AGENT: Final[str] = "trade-bot capital-cycle research contact github.com/Jbultema"
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
            filing_container.get("files", []) if isinstance(filing_container, dict) else []
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
                historical_frames.append(_submission_filing_rows(json.loads(historical_payload)))
            except json.JSONDecodeError:
                continue
        if historical_frames:
            available_frames = [
                frame for frame in [filing_rows, *historical_frames] if not frame.empty
            ]
            filing_rows = pd.concat(available_frames, ignore_index=True).drop_duplicates(
                "accession_number", keep="last"
            )
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
    economics = parse_fcc_telecom_economics_release(
        content,
        source_url=source_url,
        source_hash=hashlib.sha256(content).hexdigest(),
    )
    observations = pd.concat([observations, economics], ignore_index=True)
    combined = _merge_observations(existing_observations, observations)
    observation_path = Path(research_dir) / "direct_observations.csv"
    combined.to_csv(observation_path, index=False)
    return DirectObservationImportResult(
        observations=combined,
        observation_path=observation_path,
    )


def import_eia_dpr_archive(
    *,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
) -> HistoricalArchiveImportResult:
    """Import locally archived, contemporaneous EIA DPR release snapshots."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "eia_dpr"
    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("????-??.html")):
        content = path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        source_url = (
            f"https://www.eia.gov/petroleum/drilling/archive/{path.stem.replace('-', '/')}/"
        )
        try:
            parsed = parse_eia_dpr_release(
                content,
                source_url=source_url,
                source_hash=source_hash,
            )
            status = "imported" if not parsed.empty else "no_supported_tables"
            error = ""
        except (ValueError, OSError, KeyError) as exc:
            parsed = pd.DataFrame()
            status = "parse_failed"
            error = str(exc)
        rows.extend(parsed.to_dict("records"))
        inventory_rows.append(
            {
                "source_url": source_url,
                "raw_path": str(path),
                "sha256": source_hash,
                "bytes": len(content),
                "status": status,
                "extracted_rows": len(parsed),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
        if status == "parse_failed":
            continue
        release_date = pd.Timestamp(parsed["available_date"].iloc[0])
        for kind, parser in (
            ("dpr-data", parse_eia_dpr_workbook),
            ("duc-data", parse_eia_duc_workbook),
        ):
            workbook_path = raw_dir / f"{path.stem}-{kind}.xlsx"
            if not workbook_path.exists():
                continue
            workbook_content = workbook_path.read_bytes()
            workbook_hash = hashlib.sha256(workbook_content).hexdigest()
            workbook_url = f"{source_url}xls/{kind}.xlsx"
            try:
                workbook_rows = parser(
                    workbook_content,
                    release_date=release_date,
                    source_url=workbook_url,
                    source_hash=workbook_hash,
                )
                workbook_status = "imported"
                workbook_error = ""
            except (ValueError, OSError, KeyError, IndexError) as exc:
                workbook_rows = pd.DataFrame()
                workbook_status = "parse_failed"
                workbook_error = str(exc)
            rows.extend(workbook_rows.to_dict("records"))
            inventory_rows.append(
                {
                    "source_url": workbook_url,
                    "raw_path": str(workbook_path),
                    "sha256": workbook_hash,
                    "bytes": len(workbook_content),
                    "status": workbook_status,
                    "extracted_rows": len(workbook_rows),
                    "error": workbook_error,
                    "allocation_authority": 0.0,
                }
            )
    combined = _merge_observations(existing_observations, pd.DataFrame(rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "eia_dpr_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return HistoricalArchiveImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def fetch_fed_sloos_housing_panel(
    *,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    start_year: int = 2001,
    end_year: int = 2019,
    timeout_seconds: float = 30.0,
) -> HistoricalArchiveImportResult:
    """Archive SLOOS vintages and extract mortgage-credit standards as published."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "fed_sloos"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_SEC_USER_AGENT})
    index_response = session.get(FED_SLOOS_INDEX_URL, timeout=timeout_seconds)
    index_response.raise_for_status()
    release_urls = sorted(
        {
            urljoin(FED_SLOOS_INDEX_URL, str(link.get("href")))
            for link in BeautifulSoup(index_response.content, "html.parser").find_all(
                "a", href=True
            )
            if (
                (match := re.search(r"/(\d{6})/default\.htm$", str(link.get("href"))))
                and start_year <= int(match.group(1)[:4]) <= end_year
            )
        }
    )
    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for release_url in release_urls:
        release_id = re.search(r"/(\d{6})/default\.htm$", release_url).group(1)  # type: ignore[union-attr]
        landing_path = raw_dir / f"{release_id}-default.html"
        landing, landing_status, landing_error = _download_or_read(
            session, release_url, landing_path, timeout_seconds=timeout_seconds
        )
        chart_url = ""
        chart_path = raw_dir / f"{release_id}-chartdata.html"
        chart_content = b""
        status = landing_status
        error = landing_error
        parsed = pd.DataFrame()
        if landing:
            landing_soup = BeautifulSoup(landing, "html.parser")
            release_date = _published_date(landing_soup.get_text(" ", strip=True))
            chart_link = next(
                (
                    link
                    for link in landing_soup.find_all("a", href=True)
                    if "chartdata" in str(link.get("href")).casefold()
                ),
                None,
            )
            if release_date is None or chart_link is None:
                status = "metadata_missing"
                error = "Release date or chart-data link was not found."
            else:
                chart_url = urljoin(release_url, str(chart_link.get("href")))
                chart_suffix = Path(urlparse(chart_url).path).suffix or ".html"
                chart_path = raw_dir / f"{release_id}-chartdata{chart_suffix}"
                chart_content, status, error = _download_or_read(
                    session,
                    chart_url,
                    chart_path,
                    timeout_seconds=timeout_seconds,
                )
                if chart_content:
                    try:
                        parsed = parse_fed_sloos_chartdata(
                            chart_content,
                            release_date=release_date,
                            source_url=chart_url,
                            source_hash=hashlib.sha256(chart_content).hexdigest(),
                        )
                        status = "imported"
                        error = ""
                    except (ValueError, OSError, KeyError, IndexError) as exc:
                        status = "parse_failed"
                        error = str(exc)
        rows.extend(parsed.to_dict("records"))
        inventory_rows.append(
            {
                "release_id": release_id,
                "source_url": chart_url or release_url,
                "raw_path": str(chart_path if chart_content else landing_path),
                "sha256": (hashlib.sha256(chart_content).hexdigest() if chart_content else ""),
                "bytes": len(chart_content),
                "status": status,
                "extracted_rows": len(parsed),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    combined = _merge_observations(existing_observations, pd.DataFrame(rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "fed_sloos_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return HistoricalArchiveImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def fetch_fed_g17_technology_panel(
    *,
    episode_manifest: pd.DataFrame,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    quarterly_only: bool = True,
    timeout_seconds: float = 30.0,
) -> HistoricalArchiveImportResult:
    """Archive G.17 semiconductor production, utilization, and capacity vintages."""

    episodes = episode_manifest[
        episode_manifest["episode_family"].astype(str).isin(FED_G17_TECH_EPISODE_FAMILIES)
    ].copy()
    if episodes.empty:
        raise ValueError("No registered technology-hardware episodes were found.")
    episodes["start_date"] = pd.to_datetime(episodes["start_date"], errors="raise")
    episodes["end_date"] = pd.to_datetime(episodes["end_date"], errors="raise")
    destination = Path(research_dir)
    raw_dir = destination / "raw" / "fed_g17"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_SEC_USER_AGENT})
    index_path = raw_dir / "release-index.html"
    index_content, index_status, index_error = _download_or_read(
        session,
        FED_G17_INDEX_URL,
        index_path,
        timeout_seconds=timeout_seconds,
    )
    if not index_content:
        raise OSError(f"G.17 release index unavailable: {index_status}: {index_error}")
    release_dates = sorted(
        {
            pd.Timestamp(match.group(1))
            for link in BeautifulSoup(index_content, "html.parser").find_all("a", href=True)
            if (
                match := re.search(
                    r"(?:^|/)(\d{8})/(?:default_sup\.htm)?$",
                    str(link.get("href")),
                )
            )
        }
    )
    minimum_date = episodes["start_date"].min()
    maximum_date = episodes["end_date"].max() + pd.offsets.MonthEnd(1)
    release_dates = [date for date in release_dates if minimum_date <= date <= maximum_date]
    if quarterly_only:
        release_dates = [date for date in release_dates if date.month in {1, 4, 7, 10}]

    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for release_date in release_dates:
        if release_date < pd.Timestamp("2001-10-01"):
            source_url = (
                "https://fraser.stlouisfed.org/files/docs/releases/g17/"
                f"g17_{release_date:%Y%m%d}.pdf"
            )
            raw_path = raw_dir / f"{release_date:%Y%m%d}-fraser-g17.pdf"
            content, status, error = _download_or_read(
                session,
                source_url,
                raw_path,
                timeout_seconds=timeout_seconds,
            )
            parsed = pd.DataFrame()
            source_hash = hashlib.sha256(content).hexdigest() if content else ""
            if content:
                try:
                    parsed = parse_fed_g17_technology_pdf(
                        content,
                        release_date=release_date,
                        episode_manifest=episodes,
                        source_url=source_url,
                        source_hash=source_hash,
                    )
                    status = "imported"
                    error = ""
                except (ValueError, OSError, KeyError, IndexError) as exc:
                    status = "parse_failed"
                    error = str(exc)
            rows.extend(parsed.to_dict("records"))
            inventory_rows.append(
                {
                    "release_date": release_date,
                    "table_name": "fraser_g17_pdf",
                    "source_url": source_url,
                    "raw_path": str(raw_path),
                    "sha256": source_hash,
                    "bytes": len(content),
                    "status": status,
                    "extracted_rows": len(parsed),
                    "error": error,
                    "allocation_authority": 0.0,
                }
            )
            continue
        base_url = f"https://www.federalreserve.gov/Releases/g17/{release_date:%Y%m%d}/"
        for table_name in ("table1d_sup", "table2_sup", "table3_sup"):
            source_url = f"{base_url}{table_name}.htm"
            raw_path = raw_dir / f"{release_date:%Y%m%d}-{table_name}.html"
            content, status, error = _download_or_read(
                session,
                source_url,
                raw_path,
                timeout_seconds=timeout_seconds,
            )
            parsed_table_name = table_name
            if not content and table_name == "table1d_sup":
                parsed_table_name = "table1_sup"
                source_url = f"{base_url}{parsed_table_name}.htm"
                raw_path = raw_dir / f"{release_date:%Y%m%d}-{parsed_table_name}.html"
                content, status, error = _download_or_read(
                    session,
                    source_url,
                    raw_path,
                    timeout_seconds=timeout_seconds,
                )
            parsed = pd.DataFrame()
            source_hash = hashlib.sha256(content).hexdigest() if content else ""
            if content:
                try:
                    parsed = parse_fed_g17_technology_release(
                        content,
                        table_name=parsed_table_name,
                        episode_manifest=episodes,
                        source_url=source_url,
                        source_hash=source_hash,
                    )
                    status = "imported"
                    error = ""
                except (ValueError, OSError, KeyError, IndexError) as exc:
                    status = "parse_failed"
                    error = str(exc)
            rows.extend(parsed.to_dict("records"))
            inventory_rows.append(
                {
                    "release_date": release_date,
                    "table_name": parsed_table_name,
                    "source_url": source_url,
                    "raw_path": str(raw_path),
                    "sha256": source_hash,
                    "bytes": len(content),
                    "status": status,
                    "extracted_rows": len(parsed),
                    "error": error,
                    "allocation_authority": 0.0,
                }
            )
    combined = _merge_observations(existing_observations, pd.DataFrame(rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "fed_g17_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return HistoricalArchiveImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def parse_fed_g17_technology_release(
    content: bytes,
    *,
    table_name: str,
    episode_manifest: pd.DataFrame,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract the latest semiconductor value visible in one archived G.17 table."""

    table_specs = {
        "table1_sup": (
            "demand",
            "fed_g17_semiconductor_industrial_production",
            "index",
        ),
        "table1d_sup": (
            "demand",
            "fed_g17_semiconductor_industrial_production",
            "index",
        ),
        "table2_sup": (
            "pricing_utilization",
            "fed_g17_semiconductor_capacity_utilization",
            "percent",
        ),
        "table3_sup": (
            "capacity",
            "fed_g17_semiconductor_industrial_capacity",
            "index",
        ),
    }
    if table_name not in table_specs:
        raise ValueError(f"Unsupported G.17 table: {table_name}")
    release_date = _published_date(BeautifulSoup(content, "html.parser").get_text(" ", strip=True))
    if release_date is None:
        raise ValueError("G.17 release date was not found.")
    tables = pd.read_html(BytesIO(content), flavor="lxml")
    if not tables:
        raise ValueError("G.17 release contained no parseable table.")
    table = next(
        (
            candidate
            for candidate in tables
            if candidate.iloc[:, 0]
            .astype(str)
            .str.contains("Semiconductor", case=False, na=False)
            .any()
        ),
        None,
    )
    if table is None:
        raise ValueError("G.17 semiconductor row was not found.")
    labels = table.iloc[:, 0].astype(str).str.replace("\u00a0", " ", regex=False)
    candidates = table[labels.str.contains("Semiconductor", case=False, na=False)]
    if candidates.empty:
        raise ValueError("G.17 semiconductor row was not found.")
    value, reference_period = _latest_month_value(candidates.iloc[0])
    family, metric, unit = table_specs[table_name]
    episode_dates = episode_manifest[
        episode_manifest["episode_family"].astype(str).isin(FED_G17_TECH_EPISODE_FAMILIES)
    ].copy()
    episode_dates["start_date"] = pd.to_datetime(episode_dates["start_date"], errors="coerce")
    episode_dates["end_date"] = pd.to_datetime(episode_dates["end_date"], errors="coerce")
    eligible_episodes = episode_dates[
        episode_dates["start_date"].le(release_date)
        & episode_dates["end_date"].ge(release_date)
    ]
    rows = []
    for episode in eligible_episodes["episode"].astype(str):
        rows.append(
            {
                "episode": episode,
                "family": family,
                "metric": metric,
                "entity": "U.S. semiconductor manufacturing",
                "period_end": release_date,
                "available_date": release_date,
                "value": value,
                "unit": unit,
                "source_id": f"fed_g17_{release_date:%Y%m%d}_{table_name}",
                "source_url": source_url,
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
                "revision_status": "release_snapshot_as_published",
                "accession_or_release_id": source_hash,
                "source_archive_sha256": source_hash,
                "reference_period": reference_period,
                "notes": (
                    "Latest semiconductor observation visible in this G.17 release; "
                    "earlier table history is not backdated."
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_fed_g17_technology_pdf(
    content: bytes,
    *,
    release_date: pd.Timestamp,
    episode_manifest: pd.DataFrame,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract semiconductor observations from a pre-HTML FRASER G.17 scan."""

    pages = [page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages]
    page_specs = (
        (
            (
                "INDUSTRIAL PRODUCTION: INDUSTRY GROUPS",
                "INDUSTRIAL PRODUCTION INDEXES: SPECIAL AGGREGATES",
            ),
            "demand",
            "fed_g17_semiconductor_industrial_production",
            "index",
        ),
        (
            ("CAPACITY UTILIZATION: MANUFACTURING, MINING, AND UTILITIES", "CAPACITY UTILIZATION"),
            "pricing_utilization",
            "fed_g17_semiconductor_capacity_utilization",
            "percent",
        ),
        (
            ("INDUSTRIAL CAPACITY: MANUFACTURING, MINING, AND UTILITIES",),
            "capacity",
            "fed_g17_semiconductor_industrial_capacity",
            "index",
        ),
    )
    episode_dates = episode_manifest[
        episode_manifest["episode_family"].astype(str).isin(FED_G17_TECH_EPISODE_FAMILIES)
    ].copy()
    episode_dates["start_date"] = pd.to_datetime(episode_dates["start_date"], errors="coerce")
    episode_dates["end_date"] = pd.to_datetime(episode_dates["end_date"], errors="coerce")
    eligible_episodes = episode_dates[
        episode_dates["start_date"].le(release_date)
        & episode_dates["end_date"].ge(release_date)
    ]
    reference_period = release_date - pd.offsets.MonthEnd(1)
    rows: list[dict[str, object]] = []
    row_pattern = (
        r"Semiconductors(?: and related\s+electronic components)?\s+"
        r"\d{3,4}[\u2013-]\d\s+([^\n]+)"
    )
    for headings, family, metric, unit in page_specs:
        page = next(
            (
                text
                for text in pages
                if any(heading in text for heading in headings)
                and re.search(row_pattern, text, flags=re.IGNORECASE)
            ),
            "",
        )
        if not page and family == "capacity":
            continue
        match = re.search(row_pattern, page, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"G.17 PDF semiconductor row was not found for {family}.")
        values = [
            float(value.replace("\u2013", "-"))
            for value in re.findall(r"[\u2013-]?(?:\d+(?:\.\d*)?|\.\d+)", match.group(1))
        ]
        if family == "demand" and "INDUSTRIAL PRODUCTION: INDUSTRY GROUPS" in page:
            if len(values) < 7:
                raise ValueError("G.17 PDF production row has too few monthly values.")
            value = values[6]
        else:
            if not values:
                raise ValueError(f"G.17 PDF {family} row has no numeric values.")
            value = values[-1]
        for episode in eligible_episodes["episode"].astype(str):
            rows.append(
                {
                    "episode": episode,
                    "family": family,
                    "metric": metric,
                    "entity": "U.S. semiconductor manufacturing",
                    "period_end": release_date,
                    "available_date": release_date,
                    "value": value,
                    "unit": unit,
                    "source_id": f"fed_g17_{release_date:%Y%m%d}_fraser_pdf_{family}",
                    "source_url": source_url,
                    "source_tier": "A",
                    "vintage_status": "archived_release",
                    "eligible_for_model": True,
                    "revision_status": "release_snapshot_as_published",
                    "accession_or_release_id": source_hash,
                    "source_archive_sha256": source_hash,
                    "reference_period": reference_period,
                    "notes": (
                        "Latest semiconductor observation visible in the archived FRASER "
                        "G.17 scan; earlier table history is not backdated."
                    ),
                }
            )
    return pd.DataFrame(rows)


def fetch_eia_steo_oil_panel(
    *,
    episode_manifest: pd.DataFrame,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    quarterly_only: bool = True,
    timeout_seconds: float = 30.0,
) -> HistoricalArchiveImportResult:
    """Archive point-in-time STEO oil-market workbooks and extract release snapshots."""

    episodes = episode_manifest[
        episode_manifest["episode_family"].astype(str).isin(EIA_STEO_OIL_EPISODE_FAMILIES)
    ].copy()
    if episodes.empty:
        raise ValueError("No registered oil-shock episodes were found.")
    episodes["start_date"] = pd.to_datetime(episodes["start_date"], errors="raise")
    episodes["end_date"] = pd.to_datetime(episodes["end_date"], errors="raise")
    destination = Path(research_dir)
    raw_dir = destination / "raw" / "eia_steo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_SEC_USER_AGENT})
    index_path = raw_dir / "archive-index.html"
    index_content, index_status, index_error = _download_or_read(
        session,
        EIA_STEO_ARCHIVE_URL,
        index_path,
        timeout_seconds=timeout_seconds,
    )
    if not index_content:
        raise OSError(f"EIA STEO archive index unavailable: {index_status}: {index_error}")
    candidates: list[tuple[pd.Timestamp, str]] = []
    for link in BeautifulSoup(index_content, "html.parser").find_all("a", href=True):
        href = str(link.get("href"))
        match = re.search(r"archives/([a-z]{3})(\d{2})_base\.(?:xlsx|xls)$", href, re.IGNORECASE)
        if not match or match.group(1).casefold() not in MONTH_ABBREVIATIONS:
            continue
        year = 2000 + int(match.group(2))
        month = MONTH_ABBREVIATIONS[match.group(1).casefold()]
        release_date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
        candidates.append((release_date, urljoin(EIA_STEO_ARCHIVE_URL, href)))
    minimum_date = episodes["start_date"].min()
    maximum_date = episodes["end_date"].max() + pd.offsets.MonthEnd(1)
    candidates = sorted(
        {
            (release_date, source_url)
            for release_date, source_url in candidates
            if minimum_date <= release_date <= maximum_date
            and (not quarterly_only or release_date.month in {1, 4, 7, 10})
        }
    )

    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for release_date, source_url in candidates:
        filename = Path(urlparse(source_url).path).name
        raw_path = raw_dir / filename
        content, status, error = _download_or_read(
            session,
            source_url,
            raw_path,
            timeout_seconds=timeout_seconds,
        )
        parsed = pd.DataFrame()
        source_hash = hashlib.sha256(content).hexdigest() if content else ""
        if content:
            try:
                parsed = parse_eia_steo_oil_workbook(
                    content,
                    release_date=release_date,
                    episode_manifest=episodes,
                    source_url=source_url,
                    source_hash=source_hash,
                )
                status = "imported"
                error = ""
            except (ValueError, OSError, KeyError, IndexError) as exc:
                status = "parse_failed"
                error = str(exc)
        rows.extend(parsed.to_dict("records"))
        inventory_rows.append(
            {
                "release_date": release_date,
                "source_url": source_url,
                "raw_path": str(raw_path),
                "sha256": source_hash,
                "bytes": len(content),
                "status": status,
                "extracted_rows": len(parsed),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    combined = _merge_observations(existing_observations, pd.DataFrame(rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "eia_steo_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return HistoricalArchiveImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def parse_eia_steo_oil_workbook(
    content: bytes,
    *,
    release_date: pd.Timestamp,
    episode_manifest: pd.DataFrame,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract the latest historical oil observations published in one STEO workbook."""

    workbook = pd.ExcelFile(BytesIO(content))
    dates = (
        pd.read_excel(workbook, sheet_name="Dates", header=None)
        if "Dates" in workbook.sheet_names
        else pd.DataFrame()
    )
    reference_period = _steo_last_historical_month(dates, release_date)
    series_specs = (
        (
            "2tab",
            "WTIPUUS",
            "pricing_utilization",
            "eia_steo_wti_spot_price",
            "West Texas Intermediate crude oil",
            "dollars_per_barrel",
        ),
        (
            "3atab",
            "patc_world",
            "demand",
            "eia_steo_world_liquids_consumption",
            "World petroleum and other liquids",
            "million_barrels_per_day",
        ),
        (
            "3atab",
            "papr_world",
            "capacity",
            "eia_steo_world_liquids_production",
            "World petroleum and other liquids",
            "million_barrels_per_day",
        ),
        (
            "3atab",
            "pasc_oecd_t3",
            "capacity",
            "eia_steo_oecd_commercial_inventory",
            "OECD commercial petroleum inventories",
            "million_barrels",
        ),
        (
            "3ctab",
            "cops_opec",
            "capacity",
            "eia_steo_opec_surplus_capacity",
            "OPEC surplus crude oil production capacity",
            "million_barrels_per_day",
        ),
    )
    sheet_cache: dict[str, pd.DataFrame] = {}
    extracted: list[tuple[str, str, str, str, float]] = []
    for sheet_name, series_id, family, metric, entity, unit in series_specs:
        if sheet_name not in workbook.sheet_names:
            if family in {"demand", "pricing_utilization"} or series_id == "papr_world":
                raise ValueError(f"Required STEO sheet is missing: {sheet_name}")
            continue
        if sheet_name not in sheet_cache:
            sheet_cache[sheet_name] = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
        table = sheet_cache[sheet_name]
        value = _steo_series_value(table, series_id, reference_period)
        if value is None:
            if family in {"demand", "pricing_utilization"} or series_id == "papr_world":
                raise ValueError(
                    f"Required STEO series {series_id} is unavailable for {reference_period:%Y-%m}."
                )
            continue
        extracted.append((family, metric, entity, unit, value))

    episode_dates = episode_manifest[
        episode_manifest["episode_family"].astype(str).isin(EIA_STEO_OIL_EPISODE_FAMILIES)
    ].copy()
    episode_dates["start_date"] = pd.to_datetime(episode_dates["start_date"], errors="coerce")
    episode_dates["end_date"] = pd.to_datetime(episode_dates["end_date"], errors="coerce")
    eligible = episode_dates[
        episode_dates["start_date"].le(release_date)
        & episode_dates["end_date"].ge(release_date)
    ]
    rows: list[dict[str, object]] = []
    for episode in eligible["episode"].astype(str):
        for family, metric, entity, unit, value in extracted:
            rows.append(
                {
                    "episode": episode,
                    "family": family,
                    "metric": metric,
                    "entity": entity,
                    "period_end": release_date,
                    "available_date": release_date,
                    "value": value,
                    "unit": unit,
                    "source_id": f"eia_steo_{release_date:%Y%m}_{metric}",
                    "source_url": source_url,
                    "source_tier": "A",
                    "vintage_status": "archived_release",
                    "eligible_for_model": True,
                    "revision_status": "release_snapshot_as_published",
                    "accession_or_release_id": source_hash,
                    "source_archive_sha256": source_hash,
                    "reference_period": reference_period,
                    "notes": (
                        "Latest historical month identified by the STEO workbook; "
                        "availability is conservatively assigned to publication-month end."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _steo_last_historical_month(
    dates: pd.DataFrame,
    release_date: pd.Timestamp,
) -> pd.Timestamp:
    if dates.empty:
        return release_date - pd.offsets.MonthEnd(1)
    labels = dates.iloc[:, 0].astype(str)
    matching = dates[labels.str.contains("Last Historical Month", case=False, na=False)]
    if not matching.empty:
        values = pd.to_numeric(matching.iloc[0, 1:], errors="coerce").dropna()
        if not values.empty:
            encoded = str(int(values.iloc[0]))
            if re.fullmatch(r"\d{6}", encoded):
                return pd.Timestamp(int(encoded[:4]), int(encoded[4:]), 1) + pd.offsets.MonthEnd(0)
    return release_date - pd.offsets.MonthEnd(1)


def _steo_series_value(
    table: pd.DataFrame,
    series_id: str,
    reference_period: pd.Timestamp,
) -> float | None:
    ids = table.iloc[:, 0].astype(str).str.casefold()
    matching = table[ids.eq(series_id.casefold())]
    if matching.empty:
        return None
    years = pd.to_numeric(table.iloc[2], errors="coerce").ffill()
    months = table.iloc[3].astype(str).str[:3].str.casefold().map(MONTH_ABBREVIATIONS)
    for column in table.columns[2:]:
        year = years.get(column)
        month = months.get(column)
        if pd.isna(year) or pd.isna(month):
            continue
        period = pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthEnd(0)
        if period == reference_period:
            value = pd.to_numeric(matching.iloc[0, column], errors="coerce")
            return float(value) if pd.notna(value) else None
    return None


def _latest_month_value(row: pd.Series) -> tuple[float, pd.Timestamp]:
    month_numbers = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    candidates: list[tuple[pd.Timestamp, float]] = []
    current_year: int | None = None
    for column, raw_value in row.items():
        parts = column if isinstance(column, tuple) else (column,)
        label = " ".join(str(part) for part in parts)
        month_match = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", label, re.I)
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", label)
        if years:
            current_year = int(years[-1])
        numeric = pd.to_numeric(raw_value, errors="coerce")
        if month_match and current_year is not None and pd.notna(numeric):
            month = month_numbers[month_match.group(1)[:3].casefold()]
            candidates.append(
                (
                    pd.Timestamp(current_year, month, 1) + pd.offsets.MonthEnd(0),
                    float(numeric),
                )
            )
    if not candidates:
        raise ValueError("G.17 current-month value was not found.")
    latest = max(candidates, key=lambda item: item[0])
    return latest[1], latest[0]


def parse_fed_sloos_chartdata(
    content: bytes,
    *,
    release_date: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract the latest mortgage-tightening observation visible in a SLOOS vintage."""

    text = content.decode("utf-8", errors="replace")
    values: dict[str, float] = {}
    reference_period: pd.Timestamp | None = None
    if "<table" in text.casefold():
        soup = BeautifulSoup(content, "html.parser")
        candidates = [
            table
            for table in soup.find_all("table")
            if "mortgage" in table.get_text(" ", strip=True).casefold()
            and "tightening" in table.get_text(" ", strip=True).casefold()
        ]
        for table in candidates:
            for row in reversed(table.find_all("tr")):
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                quarter = _sloos_quarter(cells[0].get_text(" ", strip=True))
                if quarter is None:
                    continue
                for index, cell in enumerate(cells[1:], start=1):
                    headers = " ".join(cell.get("headers", [])).casefold()
                    if "p2" in headers or (headers and "p1" not in headers):
                        continue
                    numeric = pd.to_numeric(
                        cell.get_text(" ", strip=True).replace("n.a.", ""),
                        errors="coerce",
                    )
                    if pd.isna(numeric):
                        continue
                    category_match = re.search(r"p1([a-z]+)", headers)
                    category = category_match.group(1) if category_match else f"series_{index}"
                    values[category] = float(numeric)
                if values:
                    reference_period = quarter
                    break
            if values:
                break
    else:
        match = re.search(
            r"Tightening\s+Standards\s+for\s+Mortgage.*?(?=\n\s*(?:Figure|Panel)\s+\d|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            observations = re.findall(
                r"(?m)^\s*(\d{2,4}:[1-4])\s+(-?\d+(?:\.\d+)?)\s*$",
                match.group(0),
            )
            if observations:
                quarter_label, numeric = observations[-1]
                reference_period = _sloos_quarter(quarter_label)
                values["all"] = float(numeric)
    if not values or reference_period is None:
        raise ValueError("Mortgage-credit standards were not found in SLOOS chart data.")

    episode = (
        "housing_mortgage_2001_2011"
        if reference_period.year <= 2011
        else "housing_recovery_2012_2019"
    )
    rows = [
        {
            "episode": episode,
            "family": "financing",
            "metric": f"sloos_net_percent_tightening_mortgage_{category}",
            "entity": "Domestic bank respondents",
            "period_end": release_date,
            "available_date": release_date,
            "value": value,
            "unit": "net_percent",
            "source_id": f"fed_sloos_{release_date:%Y%m%d}_{category}",
            "source_url": source_url,
            "source_tier": "A",
            "vintage_status": "archived_release",
            "eligible_for_model": True,
            "revision_status": "survey_snapshot_as_published",
            "accession_or_release_id": source_hash,
            "source_archive_sha256": source_hash,
            "reference_period": reference_period,
            "notes": (
                "Latest mortgage-credit-standards result visible in this archived "
                "survey release; earlier chart history is not backdated."
            ),
        }
        for category, value in values.items()
    ]
    representative = rows[0].copy()
    representative["metric"] = "sloos_mortgage_credit_tightening"
    representative["value"] = float(pd.Series(values.values()).median())
    representative["source_id"] = f"fed_sloos_{release_date:%Y%m%d}_representative"
    representative["notes"] = (
        "Median net tightening percentage across mortgage categories reported in this "
        "release; category-level observations are retained alongside it. Earlier chart "
        "history is not backdated."
    )
    rows.append(representative)
    return pd.DataFrame(rows)


def _download_or_read(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    timeout_seconds: float,
) -> tuple[bytes, str, str]:
    if path.exists():
        return path.read_bytes(), "cached", ""
    try:
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        path.write_bytes(response.content)
        return response.content, "downloaded", ""
    except requests.RequestException as exc:
        return b"", "download_failed", str(exc)


def _published_date(text: str) -> pd.Timestamp | None:
    match = re.search(
        rf"(?:Last\s+update|For\s+release[^:]*at[^A-Z]*)(?:\s*:)?\s*((?:{MONTH_NAME_PATTERN})\s+\d{{1,2}},\s+\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    return pd.to_datetime(match.group(1)) if match else None


def _sloos_quarter(value: str) -> pd.Timestamp | None:
    match = re.fullmatch(r"\s*(\d{2}|\d{4}):Q?([1-4])\s*", value)
    if not match:
        return None
    year = int(match.group(1))
    if year < 100:
        year += 1900 if year >= 60 else 2000
    return pd.Period(year=year, quarter=int(match.group(2)), freq="Q").end_time.normalize()


def parse_eia_dpr_release(
    content: bytes,
    *,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract aggregate production and productivity from an archived DPR page."""

    text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    release_match = re.search(
        rf"Release Date:\s*({MONTH_NAME_PATTERN})\s+(\d{{1,2}}),\s+(\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    if release_match is None:
        raise ValueError("EIA DPR release date was not found.")
    release_date = pd.to_datetime(" ".join(release_match.groups()), errors="raise")
    try:
        tables = pd.read_html(BytesIO(content), flavor="lxml")
    except ValueError as exc:
        raise ValueError("EIA DPR observation tables were not found.") from exc

    table_contracts = (
        (
            "New-well oil production per rig",
            "Rig-weighted average",
            "pricing_utilization",
            "eia_new_well_oil_production_per_rig_forecast",
            "barrels_per_day_per_rig",
        ),
        (
            "New-well gas production per rig",
            "Rig-weighted average",
            "pricing_utilization",
            "eia_new_well_gas_production_per_rig_forecast",
            "thousand_cubic_feet_per_day_per_rig",
        ),
        (
            "Oil production",
            "Total",
            "demand",
            "eia_total_shale_oil_production_forecast",
            "thousand_barrels_per_day",
        ),
        (
            "Gas production",
            "Total",
            "demand",
            "eia_total_shale_gas_production_forecast",
            "million_cubic_feet_per_day",
        ),
    )
    rows: list[dict[str, object]] = []
    for header, aggregate_label, family, metric, unit in table_contracts:
        selected = _eia_table_value(tables, header=header, aggregate_label=aggregate_label)
        if selected is None:
            continue
        value, forecast_period = selected
        rows.append(
            {
                "episode": "shale_capex_2010_2017",
                "family": family,
                "metric": metric,
                "entity": "EIA major shale regions aggregate",
                "period_end": release_date,
                "available_date": release_date,
                "value": value,
                "unit": unit,
                "source_id": f"eia_dpr_{release_date:%Y%m%d}",
                "source_url": source_url,
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
                "revision_status": "forecast_snapshot_as_published",
                "accession_or_release_id": source_hash,
                "source_archive_sha256": source_hash,
                "forecast_period": forecast_period,
                "notes": (
                    "Official DPR next-month forecast snapshot. period_end records the "
                    "information-set date; forecast_period preserves the economic horizon."
                ),
            }
        )
    if not rows:
        raise ValueError("EIA DPR supported aggregate rows were not found.")
    return pd.DataFrame(rows)


def _eia_table_value(
    tables: list[pd.DataFrame],
    *,
    header: str,
    aggregate_label: str,
) -> tuple[float, str] | None:
    for table in tables:
        if not isinstance(table.columns, pd.MultiIndex):
            continue
        matching = [
            column
            for column in table.columns
            if header.lower() in str(column[0]).lower() and str(column[1]).lower() != "change"
        ]
        if not matching:
            continue
        region_column = table.columns[0]
        aggregate = table[
            table[region_column].astype(str).str.strip().str.casefold()
            == aggregate_label.casefold()
        ]
        if aggregate.empty:
            continue
        forecast_column = matching[-1]
        value = pd.to_numeric(aggregate.iloc[0][forecast_column], errors="coerce")
        if pd.isna(value):
            continue
        return float(value), str(forecast_column[1])
    return None


def parse_eia_dpr_workbook(
    content: bytes,
    *,
    release_date: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract the latest known regional rig-count aggregate from a DPR vintage."""

    workbook = pd.ExcelFile(BytesIO(content))
    latest_rows: list[tuple[pd.Timestamp, float]] = []
    for sheet_name in workbook.sheet_names:
        if sheet_name.casefold() == "regioncounties":
            continue
        sheet = pd.read_excel(workbook, sheet_name=sheet_name, header=1)
        month_column = next(
            (column for column in sheet if str(column).strip().casefold() == "month"),
            None,
        )
        rig_column = next(
            (column for column in sheet if str(column).strip().casefold() == "rig count"),
            None,
        )
        if month_column is None or rig_column is None:
            continue
        dated = sheet.assign(
            _month=pd.to_datetime(sheet[month_column], errors="coerce"),
            _rig_count=pd.to_numeric(sheet[rig_column], errors="coerce"),
        ).dropna(subset=["_month", "_rig_count"])
        known = dated[dated["_month"].le(release_date)].sort_values("_month")
        if known.empty:
            continue
        row = known.iloc[-1]
        latest_rows.append((pd.Timestamp(row["_month"]), float(row["_rig_count"])))
    if not latest_rows:
        raise ValueError("EIA DPR workbook contains no dated regional rig-count rows.")
    reference_period = max(month for month, _value in latest_rows)
    same_period = [value for month, value in latest_rows if month == reference_period]
    if not same_period:
        raise ValueError("EIA DPR workbook has no common latest rig-count period.")
    return pd.DataFrame(
        [
            _eia_workbook_observation(
                release_date=release_date,
                reference_period=reference_period,
                family="capacity",
                metric="eia_major_shale_region_rig_count_snapshot",
                value=float(sum(same_period)),
                unit="rigs",
                source_url=source_url,
                source_hash=source_hash,
                notes=(
                    "Aggregate of the latest common regional rig-count month in the named "
                    "DPR workbook vintage; historical rows are not backdated."
                ),
            )
        ]
    )


def parse_eia_duc_workbook(
    content: bytes,
    *,
    release_date: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract the total DUC inventory from a dated official workbook snapshot."""

    workbook = pd.ExcelFile(BytesIO(content))
    value: float | None = None
    reference_period: pd.Timestamp | None = None
    if "Data" in workbook.sheet_names:
        data = pd.read_excel(workbook, sheet_name="Data", header=None)
        if len(data.index) >= 5:
            duc_columns = [
                column
                for column in data.columns[1:]
                if str(data.iloc[3, column]).strip().casefold() == "duc"
            ]
            dated = pd.DataFrame(
                {
                    "date": pd.to_datetime(data.iloc[4:, 0], errors="coerce"),
                    "value": data.iloc[4:, duc_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .sum(axis=1, min_count=1),
                }
            ).dropna()
            known = dated[dated["date"].le(release_date)].sort_values("date")
            if not known.empty:
                latest = known.iloc[-1]
                reference_period = pd.Timestamp(latest["date"])
                value = float(latest["value"])

    if value is None and "webSummary" in workbook.sheet_names:
        summary = pd.read_excel(workbook, sheet_name="webSummary", header=1)
        if not summary.empty:
            entity_column = summary.columns[0]
            total = summary[
                summary[entity_column].astype(str).str.strip().str.casefold().eq("total")
            ]
            dated_columns = [
                column
                for column in summary.columns[1:]
                if pd.notna(pd.to_datetime(column, errors="coerce"))
            ]
            if not total.empty and dated_columns:
                reference_column = dated_columns[-1]
                parsed_value = pd.to_numeric(total.iloc[0][reference_column], errors="coerce")
                if pd.notna(parsed_value):
                    reference_period = pd.Timestamp(reference_column)
                    value = float(parsed_value)

    if value is None or reference_period is None:
        raise ValueError("EIA DUC workbook total history was not found.")
    return pd.DataFrame(
        [
            _eia_workbook_observation(
                release_date=release_date,
                reference_period=reference_period,
                family="commitment_overhang",
                metric="eia_total_drilled_uncompleted_wells_snapshot",
                value=value,
                unit="wells",
                source_url=source_url,
                source_hash=source_hash,
                notes=(
                    "Total drilled-but-uncompleted physical project inventory from the named "
                    "official workbook vintage; historical rows are not backdated."
                ),
            )
        ]
    )


def _eia_workbook_observation(
    *,
    release_date: pd.Timestamp,
    reference_period: pd.Timestamp,
    family: str,
    metric: str,
    value: float,
    unit: str,
    source_url: str,
    source_hash: str,
    notes: str,
) -> dict[str, object]:
    return {
        "episode": "shale_capex_2010_2017",
        "family": family,
        "metric": metric,
        "entity": "EIA major shale regions aggregate",
        "period_end": release_date,
        "available_date": release_date,
        "value": value,
        "unit": unit,
        "source_id": f"eia_dpr_workbook_{release_date:%Y%m%d}",
        "source_url": source_url,
        "source_tier": "A",
        "vintage_status": "archived_release",
        "eligible_for_model": True,
        "revision_status": "snapshot_as_published",
        "accession_or_release_id": source_hash,
        "source_archive_sha256": source_hash,
        "reference_period": reference_period,
        "notes": notes,
    }


def import_usgs_copper_mcs_archive(
    *,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
) -> HistoricalArchiveImportResult:
    """Import publication-year USGS copper MCS vintages for the mining episode."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "usgs_mcs_copper"
    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("mcs-????-copper.pdf")):
        content = path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        release_year = int(path.name.split("-")[1])
        source_url = (
            "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/"
            f"mineral-pubs/copper/mcs-{release_year}-coppe.pdf"
        )
        try:
            text = "\n".join(
                page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages
            )
            parsed = _parse_usgs_copper_mcs_text(
                text,
                release_year=release_year,
                source_url=source_url,
                source_hash=source_hash,
            )
            status = "imported"
            error = ""
        except (ValueError, OSError, KeyError) as exc:
            parsed = pd.DataFrame()
            status = "parse_failed"
            error = str(exc)
        rows.extend(parsed.to_dict("records"))
        inventory_rows.append(
            {
                "release_year": release_year,
                "source_url": source_url,
                "raw_path": str(path),
                "sha256": source_hash,
                "bytes": len(content),
                "status": status,
                "extracted_rows": len(parsed),
                "error": error,
                "allocation_authority": 0.0,
            }
        )
    combined = _merge_observations(existing_observations, pd.DataFrame(rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "usgs_copper_mcs_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return HistoricalArchiveImportResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def _parse_usgs_copper_mcs_text(
    text: str,
    *,
    release_year: int,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    normalized = " ".join(text.replace("\xa0", " ").split())
    salient_start = normalized.find("Salient Statistics")
    salient_end = normalized.find("Recycling:", salient_start)
    if salient_start < 0 or salient_end < 0:
        raise ValueError("USGS copper salient-statistics table was not found.")
    salient = normalized[salient_start:salient_end]
    contracts = (
        (
            r"\bMine(?:,\s*recoverable)?\s+(.*?)\s+Refinery:",
            "capacity",
            "usgs_copper_mine_production",
            "thousand_metric_tons_copper_content",
        ),
        (
            r"Reported, refined\s+(.*?)\s+Apparent, unmanufactured",
            "demand",
            "usgs_copper_reported_refined_consumption",
            "thousand_metric_tons_copper_content",
        ),
        (
            r"Domestic producer, cathode\s+(.*?)\s+London Metal Exchange",
            "pricing_utilization",
            "usgs_copper_domestic_producer_cathode_price",
            "cents_per_pound",
        ),
    )
    available_date = pd.Timestamp(year=release_year, month=1, day=31)
    period_end = pd.Timestamp(year=release_year - 1, month=12, day=31)
    rows: list[dict[str, object]] = []
    for pattern, family, metric, unit in contracts:
        match = re.search(pattern, salient, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"USGS copper metric was not found: {metric}")
        values = re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", match.group(1))
        if not values:
            raise ValueError(f"USGS copper values were not found: {metric}")
        value = float(values[-1].replace(",", ""))
        rows.append(
            {
                "episode": "mining_capex_2009_2017",
                "family": family,
                "metric": metric,
                "entity": "United States copper industry",
                "period_end": period_end,
                "available_date": available_date,
                "value": value,
                "unit": unit,
                "source_id": f"usgs_copper_mcs_{release_year}",
                "source_url": source_url,
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
                "revision_status": "estimate_as_published_in_annual_summary",
                "accession_or_release_id": source_hash,
                "source_archive_sha256": source_hash,
                "notes": (
                    "Prior-year estimate from the named January USGS publication; later "
                    "editions are retained as separate vintages rather than revisions."
                ),
            }
        )
    return pd.DataFrame(rows)


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


def parse_fcc_telecom_economics_release(
    content: bytes,
    *,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract demand, pricing, and monetization histories from the August 2001 FCC report."""

    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    available_date = pd.Timestamp("2001-08-31")
    rows: list[dict[str, object]] = []

    usage_match = re.search(
        r"Table 11\.2\s+Line Usage Per Day.*?Increase Over Prior Year",
        text,
        flags=re.DOTALL,
    )
    if usage_match:
        for year, total in re.findall(
            r"(?m)^(19\d{2})\s+\d+\s+\d+\s+\d+\s+(\d+)\s*$",
            usage_match.group(0),
        ):
            rows.append(
                _fcc_economics_observation(
                    year=int(year),
                    available_date=available_date,
                    family="demand",
                    metric="fcc_total_line_usage_minutes_per_day",
                    value=float(total),
                    unit="minutes_per_local_loop_per_day",
                    source_url=source_url,
                    source_hash=source_hash,
                    table="11.2",
                )
            )

    for year, revenue_per_minute in re.findall(
        r"(?m)^(19\d{2})\s+\$?(0\.\d+)\s+\$?0\.\d+\s+\d+\.\d+",
        text,
    ):
        if 1984 <= int(year) <= 1999:
            rows.append(
                _fcc_economics_observation(
                    year=int(year),
                    available_date=available_date,
                    family="pricing_utilization",
                    metric="fcc_long_distance_average_revenue_per_minute",
                    value=float(revenue_per_minute),
                    unit="usd_per_minute",
                    source_url=source_url,
                    source_hash=source_hash,
                    table="14.5",
                )
            )

    revenue_sections = re.findall(
        r"Table 16\.2\s+Telecommunications Revenues Reported by Type of Service.*?(?=Table 16\.3)",
        text,
        flags=re.DOTALL,
    )
    if revenue_sections:
        revenue_section = max(revenue_sections, key=len)
        revenue_match = re.search(
            r"Total Telecommunications Revenues\s+3/\s+"
            r"([\d,$()\s]+?)(?=\nNon-Telecommunications Revenues)",
            revenue_section,
        )
        if revenue_match:
            values = re.findall(r"\(?\$?(\d{3},\d{3})\)?", revenue_match.group(1))
            if len(values) >= 8:
                for year, value in zip(range(1992, 2000), values[:8], strict=True):
                    rows.append(
                        _fcc_economics_observation(
                            year=year,
                            available_date=available_date,
                            family="monetization",
                            metric="fcc_total_telecommunications_revenue",
                            value=float(value.replace(",", "")),
                            unit="usd_millions",
                            source_url=source_url,
                            source_hash=source_hash,
                            table="16.2",
                        )
                    )
    result = pd.DataFrame(rows).drop_duplicates(
        subset=["family", "metric", "period_end"], keep="last"
    )
    expected = {"demand", "pricing_utilization", "monetization"}
    if result.empty or set(result["family"]) != expected:
        raise ValueError("FCC demand, pricing, and monetization tables were not all found.")
    return result.reset_index(drop=True)


def _fcc_economics_observation(
    *,
    year: int,
    available_date: pd.Timestamp,
    family: str,
    metric: str,
    value: float,
    unit: str,
    source_url: str,
    source_hash: str,
    table: str,
) -> dict[str, object]:
    return {
        "episode": "dotcom_telecom_capacity_1994_2004",
        "family": family,
        "metric": metric,
        "entity": "United States telecommunications industry",
        "period_end": pd.Timestamp(year=year, month=12, day=31),
        "available_date": available_date,
        "value": value,
        "unit": unit,
        "source_id": f"fcc_trends_telephone_service_200108_table_{table.replace('.', '_')}",
        "source_url": source_url,
        "source_tier": "A",
        "vintage_status": "archived_release",
        "eligible_for_model": True,
        "revision_status": "history_as_published_in_august_2001_release",
        "accession_or_release_id": source_hash,
        "source_archive_sha256": source_hash,
        "notes": (
            "Historical value first enters the governed panel at the August 2001 report "
            "date and is not backdated to the measurement year."
        ),
    }


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
            except (OSError, ValueError, KeyError, PdfReadError, re.error) as exc:
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


def fetch_census_new_sales_panel(
    *,
    episode_manifest: pd.DataFrame,
    existing_observations: pd.DataFrame | None = None,
    research_dir: str | Path = DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR,
    quarterly_only: bool = True,
    timeout_seconds: float = 30.0,
) -> CensusArchiveResult:
    """Archive NRS releases and extract headline supply and median price snapshots."""

    destination = Path(research_dir)
    raw_dir = destination / "raw" / "census_nrs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_SEC_USER_AGENT})
    observation_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for episode, period in _housing_release_periods(
        episode_manifest, quarterly_only=quarterly_only
    ):
        period_label = period.strftime("%Y%m")
        url = CENSUS_NRS_URL_TEMPLATE.format(period=period_label)
        raw_path = raw_dir / f"newressales_{period_label}.pdf"
        content, status, error = _download_or_read(
            session, url, raw_path, timeout_seconds=timeout_seconds
        )
        parsed = pd.DataFrame()
        if content:
            try:
                parsed = parse_census_new_sales_release(
                    content,
                    episode=episode,
                    period=period,
                    source_url=url,
                    source_hash=hashlib.sha256(content).hexdigest(),
                )
                status = "imported"
                error = ""
            except (
                ValueError,
                OSError,
                KeyError,
                IndexError,
                PdfReadError,
                re.error,
            ) as exc:
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
    combined = _merge_observations(existing_observations, pd.DataFrame(observation_rows))
    inventory = pd.DataFrame(inventory_rows)
    observation_path = destination / "direct_observations.csv"
    inventory_path = destination / "census_new_sales_archive_inventory.csv"
    combined.to_csv(observation_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    return CensusArchiveResult(
        observations=combined,
        archive_inventory=inventory,
        observation_path=observation_path,
        inventory_path=inventory_path,
    )


def parse_census_new_sales_release(
    content: bytes,
    *,
    episode: str,
    period: pd.Timestamp,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    """Extract preliminary months' supply and median price from a named NRS release."""

    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    release_match = re.search(
        rf"FOR\s+(?:IMMEDIATE\s+)?RELEASE.{{0,160}}?((?:{MONTH_NAME_PATTERN})\s+\d{{1,2}},\s+\d{{4}})",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    supply_match = re.search(
        r"(?:represents|is)\s+a\s+supply\s+of\s+([\d.]+)\s+months",
        text,
        flags=re.IGNORECASE,
    )
    price_match = re.search(
        r"median sales price of new houses sold.*?was\s+\$([\d,]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if release_match is None or supply_match is None or price_match is None:
        raise ValueError("Census NRS headline release values were not found.")
    release_date = pd.to_datetime(release_match.group(1), errors="raise")
    observations = (
        ("housing_new_home_months_supply", float(supply_match.group(1)), "months"),
        (
            "housing_new_home_median_sales_price",
            float(price_match.group(1).replace(",", "")),
            "usd",
        ),
    )
    return pd.DataFrame(
        [
            {
                "episode": episode,
                "family": "pricing_utilization",
                "metric": metric,
                "entity": "United States",
                "period_end": release_date,
                "available_date": release_date,
                "value": value,
                "unit": unit,
                "source_id": f"census_nrs_{period:%Y%m}_{metric}",
                "source_url": source_url,
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
                "revision_status": "headline_preliminary_as_published",
                "accession_or_release_id": source_hash,
                "source_archive_sha256": source_hash,
                "reference_period": period,
                "notes": (
                    "Headline preliminary value from the named official monthly release; "
                    "later revisions are not backdated."
                ),
            }
            for metric, value, unit in observations
        ]
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
    keys = [
        "episode",
        "entity",
        "metric",
        "period_end",
        "available_date",
        "accession_or_release_id",
    ]
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
    frames = [
        frame for frame in (existing, new) if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for column in ("period_end", "available_date"):
        if column in combined:
            combined[column] = pd.to_datetime(combined[column], errors="coerce", format="mixed")
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
        {target: list(recent[source])[:row_count] for source, target in column_map.items()}
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
    accession_compact = (
        accession.replace("-", "") or hashlib.sha256(filing_url.encode()).hexdigest()[:18]
    )
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
        str(source.get(column, "")) for column in ("metric", "provider", "episode", "notes")
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
