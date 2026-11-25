from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from trade_bot.data import market_data


def test_safe_download_end_excludes_an_in_progress_session() -> None:
    during_market = datetime(2026, 11, 24, 18, 55, tzinfo=UTC)
    after_market = datetime(2026, 11, 24, 22, 0, tzinfo=UTC)

    assert market_data._safe_yahoo_end_date(during_market).isoformat() == "2026-11-24"
    assert market_data._safe_yahoo_end_date(after_market).isoformat() == "2026-11-25"


def test_refresh_discards_cached_in_progress_session(tmp_path, monkeypatch) -> None:
    cached_index = pd.bdate_range("2024-06-03", "2024-06-14")
    cached = pd.DataFrame({"QQQ": range(len(cached_index))}, index=cached_index)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")
    fetched_index = pd.bdate_range("2024-06-03", "2024-06-13")
    fetched = pd.DataFrame({"QQQ": range(len(fetched_index))}, index=fetched_index)
    monkeypatch.setattr(market_data, "_safe_yahoo_end_date", lambda _now: date(2024, 6, 14))
    monkeypatch.setattr(market_data, "fetch_yahoo_prices", lambda *args, **kwargs: fetched)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2024-01-01",
        end=None,
        cache_dir=tmp_path,
        refresh=True,
    )

    assert prices.index.max() == pd.Timestamp("2024-06-13")
    stored = pd.read_parquet(tmp_path / "yahoo_prices.parquet")
    assert stored.index.max() == pd.Timestamp("2024-06-13")


def test_refresh_fails_when_fetch_returns_empty_columns_instead_of_using_cache(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [100.0, 101.0], "SPY": [400.0, 402.0]}, index=index)
    cache_path = tmp_path / "yahoo_prices.parquet"
    cached.to_parquet(cache_path)

    fetched = pd.DataFrame({"QQQ": [None, None], "SPY": [401.0, 403.0]}, index=index)
    monkeypatch.setattr(market_data, "fetch_yahoo_prices", lambda *args, **kwargs: fetched)

    with pytest.raises(market_data.MarketDataRefreshError, match="QQQ"):
        market_data.load_or_fetch_yahoo_prices(
            ["QQQ", "SPY"],
            start="2024-01-01",
            end="2024-01-03",
            cache_dir=tmp_path,
            refresh=True,
        )


def test_refresh_fails_closed_when_fetch_raises(tmp_path, monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [100.0, 101.0]}, index=index)
    (tmp_path / "yahoo_prices.parquet").parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated Yahoo timeout")

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", raise_timeout)

    with pytest.raises(market_data.MarketDataRefreshError, match="cached prices were not"):
        market_data.load_or_fetch_yahoo_prices(
            ["QQQ"],
            start="2024-01-01",
            end="2024-01-03",
            cache_dir=tmp_path,
            refresh=True,
        )


def test_explicit_refresh_uses_overlap_for_complete_cache(tmp_path, monkeypatch) -> None:
    cached_index = pd.bdate_range("2024-01-01", "2024-06-03")
    cached = pd.DataFrame({"QQQ": range(len(cached_index))}, index=cached_index)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")
    fresh_index = pd.bdate_range("2024-05-13", "2024-06-14")
    fetched = pd.DataFrame({"QQQ": range(95, 95 + len(fresh_index))}, index=fresh_index)
    fetched.loc[cached.index.intersection(fresh_index), "QQQ"] = cached["QQQ"]
    requested_starts: list[str] = []

    def fake_fetch(tickers, *args, **kwargs):
        requested_starts.append(kwargs["start"])
        return fetched

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fake_fetch)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2005-01-01",
        end="2024-06-14",
        cache_dir=tmp_path,
        refresh=True,
    )

    assert requested_starts == ["2024-05-13"]
    assert prices.index.min() == cached_index.min()
    metadata = json.loads((tmp_path / "yahoo_prices.metadata.json").read_text(encoding="utf-8"))
    assert metadata["refresh_mode"] == "explicit_incremental"


def test_cached_load_fetches_columns_that_are_present_but_empty(tmp_path, monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [None, None], "SPY": [400.0, 402.0]}, index=index)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")

    fetched = pd.DataFrame({"QQQ": [100.0, 101.0]}, index=index)
    requested: list[str] = []

    def fake_fetch(tickers, *args, **kwargs):
        requested.extend(tickers)
        return fetched

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fake_fetch)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ", "SPY"],
        start="2024-01-01",
        end="2024-01-03",
        cache_dir=tmp_path,
        refresh=False,
    )

    assert requested == ["QQQ"]
    assert prices.loc[index[-1], "QQQ"] == 101.0
    assert prices.loc[index[-1], "SPY"] == 402.0
    metadata = json.loads((tmp_path / "yahoo_prices.metadata.json").read_text(encoding="utf-8"))
    assert metadata["vendor"] == "Yahoo Finance"
    assert metadata["fetched_tickers"] == ["QQQ"]
    assert metadata["adjusted"] is True
    assert metadata["known_limitations"]


def test_stale_complete_cache_is_refreshed_before_use(tmp_path, monkeypatch) -> None:
    cached_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [100.0, 101.0]}, index=cached_index)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")
    fresh_index = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"])
    fetched = pd.DataFrame({"QQQ": [105.0, 106.0, 107.0]}, index=fresh_index)
    requested: list[str] = []
    requested_starts: list[str] = []

    def fake_fetch(tickers, *args, **kwargs):
        requested.extend(tickers)
        requested_starts.append(kwargs["start"])
        return fetched

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fake_fetch)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2024-01-01",
        end="2024-01-12",
        cache_dir=tmp_path,
    )

    assert requested == ["QQQ"]
    assert requested_starts == ["2024-01-01"]
    assert prices.loc[fresh_index[-1], "QQQ"] == 107.0


def test_stale_cache_refresh_uses_overlapping_incremental_window(tmp_path, monkeypatch) -> None:
    cached_index = pd.bdate_range("2024-01-01", "2024-06-03")
    cached = pd.DataFrame({"QQQ": range(len(cached_index))}, index=cached_index)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")
    fresh_index = pd.bdate_range("2024-05-13", "2024-06-14")
    fetched = pd.DataFrame({"QQQ": range(95, 95 + len(fresh_index))}, index=fresh_index)
    fetched.loc[cached.index.intersection(fresh_index), "QQQ"] = cached["QQQ"]
    requested_starts: list[str] = []

    def fake_fetch(tickers, *args, **kwargs):
        requested_starts.append(kwargs["start"])
        return fetched

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fake_fetch)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2005-01-01",
        end="2024-06-14",
        cache_dir=tmp_path,
    )

    assert requested_starts == ["2024-05-13"]
    assert prices.index.min() == cached_index.min()
    metadata = json.loads((tmp_path / "yahoo_prices.metadata.json").read_text(encoding="utf-8"))
    assert metadata["requested_start"] == "2005-01-01"
    assert metadata["fetched_start"] == "2024-05-13"


def test_stale_complete_cache_fails_when_update_fails(tmp_path, monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pd.DataFrame({"QQQ": [100.0, 101.0]}, index=index).to_parquet(tmp_path / "yahoo_prices.parquet")

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated Yahoo timeout")

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", raise_timeout)

    with pytest.raises(market_data.StaleMarketDataError, match="refusing to run"):
        market_data.load_or_fetch_yahoo_prices(
            ["QQQ"],
            start="2024-01-01",
            end="2024-01-12",
            cache_dir=tmp_path,
        )


def test_revised_overlap_refetches_full_history_without_splicing(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", "2024-06-14")
    cached = pd.DataFrame({"QQQ": 200.0}, index=dates[:-5])
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")
    starts = []

    def fetch(tickers, *, start, **kwargs):
        starts.append(start)
        return pd.DataFrame({"QQQ": 100.0}, index=dates[dates >= start])

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fetch)
    result = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"], "2024-01-02", "2024-06-14", tmp_path, refresh=True
    )
    assert len(starts) == 2
    assert starts[-1] == "2024-01-02"
    assert result["QQQ"].eq(100.0).all()


def test_failed_revision_refetch_preserves_cache(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", "2024-06-14")
    cache = tmp_path / "yahoo_prices.parquet"
    pd.DataFrame({"QQQ": 200.0}, index=dates).to_parquet(cache)
    before = cache.read_bytes()

    def fetch(tickers, *, start, **kwargs):
        if start == "2024-01-02":
            raise TimeoutError("full history unavailable")
        return pd.DataFrame({"QQQ": 100.0}, index=dates[dates >= start])

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fetch)
    with pytest.raises(market_data.MarketDataRefreshError, match="full refresh"):
        market_data.load_or_fetch_yahoo_prices(
            ["QQQ"], "2024-01-02", "2024-06-14", tmp_path, refresh=True
        )
    assert cache.read_bytes() == before


def test_fresh_benchmark_does_not_hide_stale_security():
    dates = pd.bdate_range("2024-01-02", "2024-01-12")
    prices = pd.DataFrame({"SPY": 100.0, "QQQ": float("nan")}, index=dates)
    prices.loc[dates[0], "QQQ"] = 100.0
    with pytest.raises(market_data.StaleMarketDataError):
        market_data._require_terminal_freshness(prices, end="2024-01-12")
