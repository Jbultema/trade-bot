from __future__ import annotations

import json

import pandas as pd
import pytest

from trade_bot.data import market_data


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

    def fake_fetch(tickers, *args, **kwargs):
        requested.extend(tickers)
        return fetched

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", fake_fetch)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2024-01-01",
        end="2024-01-12",
        cache_dir=tmp_path,
    )

    assert requested == ["QQQ"]
    assert prices.loc[fresh_index[-1], "QQQ"] == 107.0


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
