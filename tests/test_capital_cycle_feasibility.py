from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.research.capital_cycle_feasibility import (
    REQUIRED_DIRECT_FAMILIES,
    audit_point_in_time_observations,
    run_capital_cycle_feasibility,
)
from trade_bot.research.capital_cycle_sources import (
    _parse_census_release_text,
    fetch_sec_filing_archive,
    import_sec_companyfacts_panel,
    parse_fcc_telecom_capacity_release,
)


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": f"episode_{index}",
                "episode_family": f"family_{index}",
                "episode_type": "break" if index < 3 else "negative_control",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "theme_ticker": "QQQ",
                "preregistered_at": "2026-08-19",
            }
            for index in range(5)
        ]
    )


def _observations() -> pd.DataFrame:
    rows = []
    for episode in _episodes()["episode"]:
        for family_index, family in enumerate(REQUIRED_DIRECT_FAMILIES):
            for month in (1, 4, 7, 10):
                rows.append(
                    {
                        "episode": episode,
                        "family": family,
                        "metric": f"metric_{family_index}",
                        "entity": "aggregate",
                        "period_end": f"2020-{month:02d}-01",
                        "available_date": f"2020-{month:02d}-15",
                        "value": float(month),
                        "source_id": f"source_{family_index}",
                        "source_url": "https://example.gov/archive",
                        "source_tier": "A",
                        "vintage_status": "archived_release",
                        "eligible_for_model": True,
                    }
                )
    return pd.DataFrame(rows)


def test_latest_vintage_and_invalid_availability_fail_closed() -> None:
    observations = _observations().iloc[:2].copy()
    observations.loc[observations.index[0], "vintage_status"] = "latest_vintage"
    observations.loc[observations.index[1], "available_date"] = "2019-12-01"

    audited = audit_point_in_time_observations(observations, _episodes())

    assert not audited["eligible_for_model"].any()
    reasons = ";".join(audited["exclusion_reason"].astype(str))
    assert "revision_or_vintage_not_safe" in reasons
    assert "invalid_available_date" in reasons


def test_complete_five_episode_panel_unlocks_only_economic_research(tmp_path) -> None:
    result = run_capital_cycle_feasibility(
        episode_manifest=_episodes(),
        observations=_observations(),
        output_dir=tmp_path,
    )
    readiness = result.readiness.iloc[0]

    assert readiness["feasibility_passed"]
    assert readiness["economic_forecasting_unlocked"]
    assert not readiness["equity_timing_unlocked"]
    assert not readiness["strategy_utility_unlocked"]
    assert readiness["allocation_authority"] == 0.0
    assert result.episode_coverage["mean_direct_family_coverage"].ge(0.50).all()
    assert all(path.exists() for path in result.output_paths.values())


def test_empty_direct_panel_reports_source_gaps_and_stops(tmp_path) -> None:
    empty = pd.DataFrame(
        columns=[
            "episode",
            "family",
            "metric",
            "entity",
            "period_end",
            "available_date",
            "value",
            "source_id",
            "source_url",
            "source_tier",
            "vintage_status",
            "eligible_for_model",
        ]
    )
    source_registry = pd.DataFrame(
        [
            {
                "episode": "episode_0",
                "family": "demand",
                "source_id": "archive",
            }
        ]
    )

    result = run_capital_cycle_feasibility(
        episode_manifest=_episodes(),
        observations=empty,
        source_registry=source_registry,
        output_dir=tmp_path,
    )
    readiness = result.readiness.iloc[0]

    assert not readiness["feasibility_passed"]
    assert "no_point_in_time_direct_observations" in readiness["blockers"]
    assert set(result.source_gaps["acquisition_status"]) == {
        "source_identified_not_extracted",
        "source_not_identified",
    }


def test_census_archive_parser_preserves_release_date_and_headline_values() -> None:
    text = """
    FOR IMMEDIATE RELEASE WEDNESDAY, AUGUST 16, 2006 AT 8:30 A.M. EDT
    Privately-owned housing units authorized by building permits in July were at a
    seasonally adjusted annual rate of 1,747,000.
    Privately-owned housing starts in July were at a seasonally adjusted annual rate
    of 1,795,000.
    Privately-owned housing completions in July were at a seasonally adjusted annual
    rate of 1,927,000.
    """

    parsed = _parse_census_release_text(
        text,
        episode="housing_mortgage_2001_2011",
        period=pd.Timestamp("2006-07-31"),
        source_url="https://www.census.gov/archive.pdf",
        source_hash="abc123",
    )

    assert parsed["available_date"].eq(pd.Timestamp("2006-08-16")).all()
    assert dict(zip(parsed["metric"], parsed["value"], strict=True)) == {
        "housing_permits_saar_thousands": 1747.0,
        "housing_starts_saar_thousands": 1795.0,
        "housing_completions_saar_thousands": 1927.0,
    }
    assert parsed["vintage_status"].eq("archived_release").all()


def test_legacy_census_text_pattern_is_supported() -> None:
    text = """
    Tuesday, April 17, 2001
    The rate of housing units authorized by building permits in March was 1,615,000.
    Housing starts in March were at a seasonally adjusted annual rate of 1,613,000.
    """
    parsed = _parse_census_release_text(
        text,
        episode="housing_mortgage_2001_2011",
        period=pd.Timestamp("2001-03-31"),
        source_url="https://www.census.gov/archive.txt",
        source_hash="legacy",
    )
    assert parsed["available_date"].eq(pd.Timestamp("2001-04-17")).all()
    assert set(parsed["metric"]) == {
        "housing_permits_saar_thousands",
        "housing_starts_saar_thousands",
    }


def test_sec_companyfacts_local_import_uses_filed_date(tmp_path) -> None:
    research_dir = tmp_path / "capital_cycle"
    raw_dir = research_dir / "raw" / "sec_companyfacts"
    raw_dir.mkdir(parents=True)
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2014-01-01",
                                "end": "2014-03-31",
                                "val": 100.0,
                                "filed": "2014-05-01",
                                "form": "10-Q",
                                "accn": "0001-14-000001",
                                "fy": 2014,
                                "fp": "Q1",
                            }
                        ]
                    }
                }
            }
        }
    }
    (raw_dir / "CIK0000000123.json").write_text(json.dumps(payload))
    entities = pd.DataFrame(
        [
            {
                "episode": "cloud_buildout_2013_2019",
                "entity": "Example",
                "cik": 123,
                "filing_start": "2013-01-01",
                "filing_end": "2019-12-31",
            }
        ]
    )
    result = import_sec_companyfacts_panel(
        entity_manifest=entities,
        research_dir=research_dir,
    )
    row = result.observations.iloc[0]
    assert row["available_date"] == pd.Timestamp("2014-05-01")
    assert row["family"] == "monetization"
    assert row["vintage_status"] == "filing_as_reported"


def test_sec_companyfacts_read_only_import_does_not_overwrite_historical_panel(
    tmp_path,
) -> None:
    research_dir = tmp_path / "capital_cycle"
    research_dir.mkdir()
    historical_path = research_dir / "direct_observations.csv"
    historical_path.write_text("episode,family\nhistorical,demand\n")
    entities = pd.DataFrame(
        [
            {
                "episode": "ai_buildout_2022_present",
                "entity": "Missing Raw File",
                "cik": 123,
                "filing_start": "2022-01-01",
                "filing_end": "2026-12-31",
            }
        ]
    )

    import_sec_companyfacts_panel(
        entity_manifest=entities,
        research_dir=research_dir,
        write_outputs=False,
    )

    assert historical_path.read_text() == "episode,family\nhistorical,demand\n"


def test_sec_filing_archive_follows_historical_submission_files(tmp_path, monkeypatch) -> None:
    current = {
        "name": "Example Corp",
        "filings": {
            "recent": {
                "accessionNumber": [],
                "filingDate": [],
                "reportDate": [],
                "form": [],
                "primaryDocument": [],
            },
            "files": [{"name": "CIK0000000123-submissions-001.json"}],
        },
    }
    historical = {
        "accessionNumber": ["0000000123-00-000001"],
        "filingDate": ["2000-03-15"],
        "reportDate": ["1999-12-31"],
        "form": ["10-K"],
        "primaryDocument": ["example10k.htm"],
    }

    class Response:
        status_code = 200

        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    def fake_get(self, url, timeout):  # noqa: ANN001
        if url.endswith("CIK0000000123.json"):
            return Response(json.dumps(current).encode())
        if url.endswith("submissions-001.json"):
            return Response(json.dumps(historical).encode())
        return Response(b"<html>official filing</html>")

    monkeypatch.setattr("requests.Session.get", fake_get)
    entities = pd.DataFrame(
        [
            {
                "episode": "dotcom_test",
                "entity": "Example",
                "cik": 123,
                "filing_start": "1999-01-01",
                "filing_end": "2001-12-31",
            }
        ]
    )

    result = fetch_sec_filing_archive(
        entity_manifest=entities,
        research_dir=tmp_path,
    )

    row = result.filing_inventory.iloc[0]
    assert row["status"] == "downloaded"
    assert row["form"] == "10-K"
    assert row["filing_date"] == pd.Timestamp("2000-03-15")
    assert Path(row["raw_path"]).exists()


def test_sec_filing_seed_survives_blocked_submissions_api(tmp_path, monkeypatch) -> None:
    class Response:
        def __init__(self, content: bytes, status_code: int = 200) -> None:
            self.content = content
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("blocked")

    def fake_get(self, url, timeout):  # noqa: ANN001
        if "data.sec.gov/submissions" in url:
            import requests

            raise requests.HTTPError("403 blocked")
        return Response(b"<html>official seeded filing</html>")

    monkeypatch.setattr("requests.Session.get", fake_get)
    entities = pd.DataFrame(
        [
            {
                "episode": "dotcom_test",
                "entity": "Example",
                "cik": 123,
                "filing_start": "1999-01-01",
                "filing_end": "2001-12-31",
            }
        ]
    )
    seeds = pd.DataFrame(
        [
            {
                "cik": 123,
                "form": "10-K",
                "filing_date": "2000-03-15",
                "report_date": "1999-12-31",
                "accession_number": "0000000123-00-000001",
                "primary_document": "example10k.htm",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/123/"
                    "000000012300000001/example10k.htm"
                ),
            }
        ]
    )

    result = fetch_sec_filing_archive(
        entity_manifest=entities,
        filing_seed_manifest=seeds,
        research_dir=tmp_path,
    )

    statuses = set(result.filing_inventory["status"])
    assert "seed_downloaded" in statuses
    assert any(str(status).startswith("submissions_download_failed") for status in statuses)


def test_fcc_capacity_parser_does_not_backdate_revised_history(monkeypatch) -> None:
    digital_rows = "\n".join(
        f"{year} {1_000 + year:,} 1 1.0 2 2.0 3 3.0" for year in range(1990, 2001)
    )
    channel_rows = "\n".join(
        f"{year} {2_000 + year:,} 1 1.0 2 2.0 3 3.0" for year in range(1990, 2001)
    )
    text = (
        "Trends in Telephone Service August 2001\n"
        "Table 18.3\nDigital Transmission Links\n"
        f"{digital_rows}\nWorking Telecommunications Channels\n"
        f"{channel_rows}\nTable 18.4"
    )

    class Page:
        def extract_text(self):  # noqa: ANN201
            return text

    class Reader:
        pages = [Page()]

    monkeypatch.setattr(
        "trade_bot.research.capital_cycle_sources.PdfReader", lambda _: Reader()
    )

    parsed = parse_fcc_telecom_capacity_release(
        b"synthetic pdf",
        source_url="https://docs.fcc.gov/report.pdf",
        source_hash="abc",
    )

    assert len(parsed) == 22
    assert parsed["available_date"].eq(pd.Timestamp("2001-08-31")).all()
    assert parsed["period_end"].min() == pd.Timestamp("1990-12-31")
    assert parsed["eligible_for_model"].all()
