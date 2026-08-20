from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from trade_bot.research.capital_cycle_feasibility import (
    REQUIRED_DIRECT_FAMILIES,
    audit_point_in_time_observations,
    run_capital_cycle_feasibility,
)
from trade_bot.research.capital_cycle_sources import (
    _parse_census_release_text,
    _parse_usgs_copper_mcs_text,
    _published_date,
    fetch_sec_filing_archive,
    import_sec_companyfacts_panel,
    parse_census_new_sales_release,
    parse_eia_dpr_release,
    parse_eia_dpr_workbook,
    parse_eia_duc_workbook,
    parse_eia_steo_oil_workbook,
    parse_fcc_telecom_capacity_release,
    parse_fcc_telecom_economics_release,
    parse_fed_g17_technology_pdf,
    parse_fed_g17_technology_release,
    parse_fed_sloos_chartdata,
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


def test_eia_dpr_parser_preserves_release_snapshot_and_forecast_horizon() -> None:
    content = b"""
    <html><body>
    <p>Release Date: June 9, 2014 | Next Release: July 14, 2014</p>
    <table>
      <thead>
        <tr><th></th><th colspan="3">New-well oil production per rig barrels/day</th></tr>
        <tr><th>Region</th><th>June 2014</th><th>July 2014</th><th>change</th></tr>
      </thead>
      <tbody><tr><td>Rig-weighted average</td><td>271</td><td>275</td><td>4</td></tr></tbody>
    </table>
    <table>
      <thead>
        <tr><th></th><th colspan="3">Oil production thousand barrels/day</th></tr>
        <tr><th>Region</th><th>June 2014</th><th>July 2014</th><th>change</th></tr>
      </thead>
      <tbody><tr><td>Total</td><td>4436</td><td>4508</td><td>72</td></tr></tbody>
    </table>
    </body></html>
    """

    parsed = parse_eia_dpr_release(
        content,
        source_url="https://www.eia.gov/petroleum/drilling/archive/2014/06/",
        source_hash="eia123",
    )

    assert parsed["available_date"].eq(pd.Timestamp("2014-06-09")).all()
    assert parsed["period_end"].eq(pd.Timestamp("2014-06-09")).all()
    assert parsed["forecast_period"].eq("July 2014").all()
    assert set(parsed["family"]) == {"demand", "pricing_utilization"}
    assert set(parsed["value"]) == {275.0, 4508.0}


def test_usgs_copper_parser_uses_publication_year_availability() -> None:
    text = """
    Salient Statistics--United States: 2008 2009 2010 2011 2012e
    Production: Mine 1,200 1,100 1,150 1,170 1,220 Refinery:
    Reported, refined 1,650 1,700 1,750 1,760 1,800
    Apparent, unmanufactured 1,600 1,650 1,700 1,720 1,770
    Price, average, cents per pound:
    Domestic producer, cathode 241.2 300.0 350.0 367.3 340
    London Metal Exchange, high-grade 230.0 290.0 340.0 360.6 332
    Recycling:
    """

    parsed = _parse_usgs_copper_mcs_text(
        text,
        release_year=2013,
        source_url="https://www.usgs.gov/official.pdf",
        source_hash="usgs123",
    )

    assert parsed["period_end"].eq(pd.Timestamp("2012-12-31")).all()
    assert parsed["available_date"].eq(pd.Timestamp("2013-01-31")).all()
    assert dict(zip(parsed["family"], parsed["value"], strict=True)) == {
        "capacity": 1220.0,
        "demand": 1800.0,
        "pricing_utilization": 340.0,
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
                    "https://www.sec.gov/Archives/edgar/data/123/000000012300000001/example10k.htm"
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

    monkeypatch.setattr("trade_bot.research.capital_cycle_sources.PdfReader", lambda _: Reader())

    parsed = parse_fcc_telecom_capacity_release(
        b"synthetic pdf",
        source_url="https://docs.fcc.gov/report.pdf",
        source_hash="abc",
    )

    assert len(parsed) == 22
    assert parsed["available_date"].eq(pd.Timestamp("2001-08-31")).all()
    assert parsed["period_end"].min() == pd.Timestamp("1990-12-31")
    assert parsed["eligible_for_model"].all()


def test_fcc_economics_parser_adds_three_families_without_backdating(monkeypatch) -> None:
    text = """
Table 11.2
Line Usage Per Day
1998 45 6 8 60
1999 50 7 9 65
Increase Over Prior Year
Table 14.5
Indicators of Long Distance Prices
1984 $0.32 $0.41 103.9 $0.52 $0.66
1985 $0.31 $0.39 107.6 $0.48 $0.60
Table 16.2
Telecommunications Revenues Reported by Type of Service
Total Telecommunications Revenues  3/ 153,409 165,342 174,890 190,076 211,782 231,168 246,392 268,505
Non-Telecommunications Revenues
Table 16.3
"""

    class Page:
        def extract_text(self):  # noqa: ANN201
            return text

    class Reader:
        pages = [Page()]

    monkeypatch.setattr("trade_bot.research.capital_cycle_sources.PdfReader", lambda _: Reader())
    parsed = parse_fcc_telecom_economics_release(
        b"synthetic pdf",
        source_url="https://docs.fcc.gov/report.pdf",
        source_hash="fcc1",
    )

    assert set(parsed["family"]) == {
        "demand",
        "pricing_utilization",
        "monetization",
    }
    assert len(parsed) == 12
    assert parsed["available_date"].eq(pd.Timestamp("2001-08-31")).all()
    assert parsed["period_end"].max() == pd.Timestamp("1999-12-31")


def test_eia_dpr_workbook_uses_release_date_without_backdating() -> None:
    workbook = BytesIO()
    rows = [
        ["Production", "Production"],
        ["Month", "Rig count"],
        [pd.Timestamp("2016-10-01"), 10],
        [pd.Timestamp("2016-11-01"), 12],
        [pd.Timestamp("2017-01-01"), 99],
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Alpha", index=False, header=False)
        pd.DataFrame(
            [
                ["Production", "Production"],
                ["Month", "Rig count"],
                [pd.Timestamp("2016-10-01"), 20],
                [pd.Timestamp("2016-11-01"), 23],
            ]
        ).to_excel(writer, sheet_name="Beta", index=False, header=False)

    parsed = parse_eia_dpr_workbook(
        workbook.getvalue(),
        release_date=pd.Timestamp("2016-12-12"),
        source_url="https://www.eia.gov/petroleum/drilling/archive/2016/12/xls/dpr-data.xlsx",
        source_hash="abc",
    )

    assert len(parsed) == 1
    assert parsed.iloc[0]["family"] == "capacity"
    assert parsed.iloc[0]["value"] == 35.0
    assert parsed.iloc[0]["reference_period"] == pd.Timestamp("2016-11-01")
    assert parsed.iloc[0]["period_end"] == pd.Timestamp("2016-12-12")
    assert parsed.iloc[0]["available_date"] == pd.Timestamp("2016-12-12")


def test_eia_duc_workbook_uses_named_vintage_without_backdating() -> None:
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                [None, None, None, None],
                [None, pd.Timestamp("2016-09-01"), pd.Timestamp("2016-10-01"), "change"],
                ["Alpha", 100, 110, 10],
                ["Total", 400, 425, 25],
            ]
        ).to_excel(writer, sheet_name="webSummary", index=False, header=False)

    parsed = parse_eia_duc_workbook(
        workbook.getvalue(),
        release_date=pd.Timestamp("2016-12-12"),
        source_url="https://www.eia.gov/petroleum/drilling/archive/2016/12/xls/duc-data.xlsx",
        source_hash="def",
    )

    assert len(parsed) == 1
    assert parsed.iloc[0]["family"] == "commitment_overhang"
    assert parsed.iloc[0]["value"] == 425.0
    assert parsed.iloc[0]["reference_period"] == pd.Timestamp("2016-10-01")
    assert parsed.iloc[0]["period_end"] == pd.Timestamp("2016-12-12")
    assert parsed.iloc[0]["available_date"] == pd.Timestamp("2016-12-12")


def test_eia_steo_workbook_uses_last_historical_month_without_backdating() -> None:
    workbook = BytesIO()
    dates = pd.DataFrame(
        [
            ["Forecast Month -", None, None, "December 2022"],
            [None, None, None, None],
            ["Table Beginning Year---", None, None, 2022],
            [None, None, None, None],
            ["Table Beginning Month---", None, None, 202201],
            [None, None, None, None],
            ["Last Historical Month---", None, None, 202211],
        ]
    )

    def sheet(rows: list[list[object]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["Table of Contents", "Synthetic STEO", None, None, None],
                [None, "EIA STEO December 2022", None, None, None],
                [None, None, 2022, None, None],
                [None, None, "Oct", "Nov", "Dec"],
                *rows,
            ]
        )

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        dates.to_excel(writer, sheet_name="Dates", index=False, header=False)
        sheet([["WTIPUUS", "WTI", 87.0, 84.0, 80.0]]).to_excel(
            writer, sheet_name="2tab", index=False, header=False
        )
        sheet(
            [
                ["patc_world", "World consumption", 100.0, 101.0, 102.0],
                ["papr_world", "World production", 99.0, 100.5, 101.5],
                ["pasc_oecd_t3", "OECD inventory", 2700.0, 2680.0, 2660.0],
            ]
        ).to_excel(writer, sheet_name="3atab", index=False, header=False)
        sheet([["cops_opec", "OPEC surplus capacity", 2.0, 1.8, 1.7]]).to_excel(
            writer, sheet_name="3ctab", index=False, header=False
        )
    episodes = pd.DataFrame(
        [
            {
                "episode": "oil_geopolitical_shock_2021_2023",
                "episode_family": "oil_shock",
                "start_date": "2021-01-01",
                "end_date": "2023-12-31",
            }
        ]
    )

    parsed = parse_eia_steo_oil_workbook(
        workbook.getvalue(),
        release_date=pd.Timestamp("2022-12-31"),
        episode_manifest=episodes,
        source_url="https://www.eia.gov/outlooks/steo/archives/dec22_base.xlsx",
        source_hash="steo-test",
    )

    assert len(parsed) == 5
    assert parsed["period_end"].eq(pd.Timestamp("2022-12-31")).all()
    assert parsed["available_date"].eq(pd.Timestamp("2022-12-31")).all()
    assert parsed["reference_period"].eq(pd.Timestamp("2022-11-30")).all()
    values = dict(zip(parsed["metric"], parsed["value"], strict=True))
    assert values["eia_steo_wti_spot_price"] == 84.0
    assert values["eia_steo_world_liquids_consumption"] == 101.0
    assert values["eia_steo_opec_surplus_capacity"] == 1.8


def test_eia_steo_parser_supports_legacy_workbook_without_dates_sheet() -> None:
    workbook = BytesIO()

    def sheet(rows: list[list[object]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [None, None, 2022, 2022, 2022],
                [None, None, None, None, None],
                [None, None, 2022, 2022, 2022],
                [None, None, "Oct", "Nov", "Dec"],
                *rows,
            ]
        )

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        sheet([["WTIPUUS", "WTI", 87.0, 84.0, 80.0]]).to_excel(
            writer, sheet_name="2tab", index=False, header=False
        )
        sheet(
            [
                ["patc_world", "World consumption", 100.0, 101.0, 102.0],
                ["papr_world", "World production", 99.0, 100.5, 101.5],
                ["pasc_oecd_t3", "OECD inventory", 2700.0, 2680.0, 2660.0],
            ]
        ).to_excel(writer, sheet_name="3atab", index=False, header=False)
        sheet([["cops_opec", "OPEC surplus capacity", 2.0, 1.8, 1.7]]).to_excel(
            writer, sheet_name="3ctab", index=False, header=False
        )
    episodes = pd.DataFrame(
        [
            {
                "episode": "oil_geopolitical_shock_2021_2023",
                "episode_family": "oil_shock",
                "start_date": "2021-01-01",
                "end_date": "2023-12-31",
            }
        ]
    )

    parsed = parse_eia_steo_oil_workbook(
        workbook.getvalue(),
        release_date=pd.Timestamp("2022-12-31"),
        episode_manifest=episodes,
        source_url="https://www.eia.gov/outlooks/steo/archives/dec22_base.xlsx",
        source_hash="steo-test-no-dates",
    )

    assert len(parsed) == 5
    assert parsed["reference_period"].eq(pd.Timestamp("2022-11-30")).all()


def test_fed_sloos_legacy_chartdata_does_not_backdate_history() -> None:
    content = b"""
Panel 3: Net Percentage of Domestic Respondents Tightening
          Standards for Mortgages to Individuals

      Mortgages
      ---------
04:4        1.90
05:1       -7.80

Figure 3: Other material
"""

    parsed = parse_fed_sloos_chartdata(
        content,
        release_date=pd.Timestamp("2005-02-07"),
        source_url="https://www.federalreserve.gov/archive/chartdata.txt",
        source_hash="fed1",
    )

    assert len(parsed) == 2
    representative = parsed[parsed["metric"].eq("sloos_mortgage_credit_tightening")].iloc[0]
    assert representative["value"] == -7.8
    assert parsed.iloc[0]["reference_period"] == pd.Timestamp("2005-03-31")
    assert parsed.iloc[0]["period_end"] == pd.Timestamp("2005-02-07")
    assert parsed.iloc[0]["available_date"] == pd.Timestamp("2005-02-07")


def test_fed_sloos_published_date_uses_complete_month_name() -> None:
    assert _published_date("Last\nupdate: February 7, 2005") == pd.Timestamp("2005-02-07")


def test_fed_g17_parser_uses_latest_visible_month_and_matching_episode() -> None:
    content = b"""
    <html><body>
      <p>Last update: December 14, 2018</p>
      <table>
        <thead>
          <tr><th rowspan="2">Industry</th><th colspan="2">2018</th></tr>
          <tr><th>Oct.</th><th>Nov.</th></tr>
        </thead>
        <tbody>
          <tr><td>Semiconductor and other electronic component</td><td>165.1</td><td>168.3</td></tr>
          <tr><td>Other manufacturing</td><td>99.0</td><td>100.0</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    episodes = pd.DataFrame(
        [
            {
                "episode": "semiconductor_inventory_correction_2017_2020",
                "episode_family": "semiconductor_cycle",
                "start_date": "2017-01-01",
                "end_date": "2020-02-29",
            },
            {
                "episode": "housing_mortgage_2001_2011",
                "episode_family": "housing_mortgage",
                "start_date": "2001-01-01",
                "end_date": "2011-12-31",
            },
        ]
    )

    parsed = parse_fed_g17_technology_release(
        content,
        table_name="table1d_sup",
        episode_manifest=episodes,
        source_url="https://www.federalreserve.gov/Releases/g17/20181214/table1d_sup.htm",
        source_hash="fed-g17-test",
    )

    assert len(parsed) == 1
    observation = parsed.iloc[0]
    assert observation["episode"] == "semiconductor_inventory_correction_2017_2020"
    assert observation["family"] == "demand"
    assert observation["metric"] == "fed_g17_semiconductor_industrial_production"
    assert observation["value"] == 168.3
    assert observation["reference_period"] == pd.Timestamp("2018-11-30")
    assert observation["period_end"] == pd.Timestamp("2018-12-14")
    assert observation["available_date"] == pd.Timestamp("2018-12-14")
    assert observation["vintage_status"] == "archived_release"


def test_fed_g17_pdf_parser_uses_release_snapshot_without_backdating(monkeypatch) -> None:
    page_texts = [
        """INDUSTRIAL PRODUCTION: INDUSTRY GROUPS
        Semiconductors and related
        electronic components 3672-9 3.48 100 110 120 130 140 150 90 91 92 93 94 95
        """,
        """CAPACITY UTILIZATION: MANUFACTURING, MINING, AND UTILITIES
        Semiconductors and related
        electronic components 3672-9 4.17 70 71 72 73 74 75 76 77 78 79
        """,
        """INDUSTRIAL CAPACITY: MANUFACTURING, MINING, AND UTILITIES
        Semiconductors and related
        electronic components 3672-9 20 21 22 23 24 200 201 202 203 204 205
        """,
    ]

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        pages = [Page(text) for text in page_texts]

    monkeypatch.setattr("trade_bot.research.capital_cycle_sources.PdfReader", lambda _: Reader())
    episodes = pd.DataFrame(
        [
            {
                "episode": "dotcom_technology_hardware_1997_2004",
                "episode_family": "technology_hardware",
                "start_date": "1997-12-01",
                "end_date": "2004-12-31",
            }
        ]
    )

    parsed = parse_fed_g17_technology_pdf(
        b"synthetic pdf",
        release_date=pd.Timestamp("2000-10-17"),
        episode_manifest=episodes,
        source_url="https://fraser.stlouisfed.org/g17_20001017.pdf",
        source_hash="fraser-test",
    )

    assert dict(zip(parsed["family"], parsed["value"], strict=True)) == {
        "demand": 150.0,
        "pricing_utilization": 79.0,
        "capacity": 205.0,
    }
    assert parsed["reference_period"].eq(pd.Timestamp("2000-09-30")).all()
    assert parsed["period_end"].eq(pd.Timestamp("2000-10-17")).all()
    assert parsed["available_date"].eq(pd.Timestamp("2000-10-17")).all()


def test_fed_sloos_html_chartdata_preserves_mortgage_categories() -> None:
    content = b"""
<table>
  <tr><th></th><th id="F3P1">Tightening Standards for Mortgage Loans</th>
      <th id="F3P2">Stronger Demand for Mortgage Loans</th></tr>
  <tr><td>2007:3</td><td headers="F3P1 F3P1PRIME date1">14.3</td>
      <td headers="F3P2 F3P2PRIME date1">-10.0</td></tr>
  <tr><td>2007:4</td><td headers="F3P1 F3P1PRIME date2">40.8</td>
      <td headers="F3P2 F3P2PRIME date2">-51.0</td></tr>
</table>
"""

    parsed = parse_fed_sloos_chartdata(
        content,
        release_date=pd.Timestamp("2007-11-05"),
        source_url="https://www.federalreserve.gov/archive/chartdata.htm",
        source_hash="fed2",
    )

    assert len(parsed) == 2
    detail = parsed[parsed["metric"].str.endswith("prime")].iloc[0]
    assert detail["value"] == 40.8
    assert "sloos_mortgage_credit_tightening" in set(parsed["metric"])
    assert parsed.iloc[0]["reference_period"] == pd.Timestamp("2007-12-31")


def test_census_new_sales_parser_preserves_preliminary_release(monkeypatch) -> None:
    text = """
FOR IMMEDIATE RELEASE FRIDAY, JANUARY 27, 2006 AT 10:00 A.M. EST
NEW RESIDENTIAL SALES IN DECEMBER 2005
The median sales price of new houses sold in December 2005 was $221,800;
the average sales price was $272,900. This represents a supply of 4.9 months
at the current sales rate.
"""

    class Page:
        def extract_text(self):  # noqa: ANN201
            return text

    class Reader:
        pages = [Page()]

    monkeypatch.setattr("trade_bot.research.capital_cycle_sources.PdfReader", lambda _: Reader())
    parsed = parse_census_new_sales_release(
        b"synthetic pdf",
        episode="housing_mortgage_2001_2011",
        period=pd.Timestamp("2005-12-31"),
        source_url="https://www.census.gov/newressales_200512.pdf",
        source_hash="nrs1",
    )

    assert len(parsed) == 2
    assert set(parsed["value"]) == {4.9, 221800.0}
    assert parsed["period_end"].eq(pd.Timestamp("2006-01-27")).all()
    assert parsed["available_date"].eq(pd.Timestamp("2006-01-27")).all()
    assert parsed["reference_period"].eq(pd.Timestamp("2005-12-31")).all()
