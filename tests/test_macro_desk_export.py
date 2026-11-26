import pandas as pd

from trade_bot.research.macro_desk_export import external_research_context, macro_context


def test_missing_macro_cache_is_explicit_and_not_historical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = macro_context()
    assert result["status"] == "unavailable"
    assert result["historical_replay_allowed"] is False
    assert result["publication_times_verified"] is False


def test_context_uses_only_public_catalog_and_omits_future_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/cache").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/macro_fred.yaml").write_text("series:\n  - id: CPIAUCSL\n    name: CPI\n    category: inflation_realized\n")
    dates = pd.DatetimeIndex(["2020-01-01", "2099-01-01"])
    pd.DataFrame({"CPIAUCSL": [100., 200.], "unregistered": [123., 456.]}, index=dates).to_parquet("data/cache/fred_series.parquet")
    result = macro_context()
    assert len(result["series"]) == 1
    assert result["series"][0]["observations"] == [{"observation_date": "2020-01-01", "value": 100.}]
    assert len(result["source_sha256"]) == 64


def test_external_context_is_allowlisted_and_shows_match_freshness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/42macro_alignment").mkdir(parents=True)
    pd.DataFrame([{"published_date": "2026-11-25", "video_id": "12345678901", "title": "Public title",
                   "matched_market_date": "2026-11-20", "days_from_tradebot": 5,
                   "private_holdings": "must not export"}]).to_csv("reports/42macro_alignment/daily_comparison.csv", index=False)
    result = external_research_context()
    assert result["rows"][0]["days_from_tradebot"] == 5
    assert "private_holdings" not in result["rows"][0]
    assert len(result["source_sha256"]) == 64
