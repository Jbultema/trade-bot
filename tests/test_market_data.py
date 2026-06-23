from __future__ import annotations

import pandas as pd

from trade_bot.data import market_data


def test_refresh_uses_cached_prices_when_fetch_returns_empty_columns(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [100.0, 101.0], "SPY": [400.0, 402.0]}, index=index)
    cache_path = tmp_path / "yahoo_prices.parquet"
    cached.to_parquet(cache_path)

    fetched = pd.DataFrame({"QQQ": [None, None], "SPY": [401.0, 403.0]}, index=index)
    monkeypatch.setattr(market_data, "fetch_yahoo_prices", lambda *args, **kwargs: fetched)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ", "SPY"],
        start="2024-01-01",
        end=None,
        cache_dir=tmp_path,
        refresh=True,
    )

    assert prices.loc[index[-1], "QQQ"] == 101.0
    assert prices.loc[index[-1], "SPY"] == 403.0


def test_refresh_uses_cached_prices_when_fetch_raises(tmp_path, monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cached = pd.DataFrame({"QQQ": [100.0, 101.0]}, index=index)
    (tmp_path / "yahoo_prices.parquet").parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(tmp_path / "yahoo_prices.parquet")

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated Yahoo timeout")

    monkeypatch.setattr(market_data, "fetch_yahoo_prices", raise_timeout)

    prices = market_data.load_or_fetch_yahoo_prices(
        ["QQQ"],
        start="2024-01-01",
        end=None,
        cache_dir=tmp_path,
        refresh=True,
    )

    assert prices.loc[index[-1], "QQQ"] == 101.0


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
        end=None,
        cache_dir=tmp_path,
        refresh=False,
    )

    assert requested == ["QQQ"]
    assert prices.loc[index[-1], "QQQ"] == 101.0
    assert prices.loc[index[-1], "SPY"] == 402.0
