from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from trade_bot.DEFAULTS import DEFAULT_DATA_ADJUSTED


def load_or_fetch_yahoo_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    *,
    adjusted: bool = DEFAULT_DATA_ADJUSTED,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide daily close-price frame indexed by date."""
    ordered_tickers = sorted(set(tickers))
    cache_path = Path(cache_dir) / "yahoo_prices.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: pd.DataFrame | None = None

    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        if set(ordered_tickers).issubset(cached.columns):
            return _filter_prices(cached[ordered_tickers], start=start, end=end)

    if cached is not None and not refresh:
        missing = sorted(set(ordered_tickers) - set(cached.columns))
        fetched = fetch_yahoo_prices(
            missing,
            start=start,
            end=end,
            adjusted=adjusted,
        )
        prices = cached.combine_first(fetched).reindex(
            columns=sorted(set(cached.columns) | set(fetched.columns))
        )
        prices.to_parquet(cache_path)
        return _filter_prices(prices.reindex(columns=ordered_tickers), start=start, end=end)

    prices = fetch_yahoo_prices(
        ordered_tickers,
        start=start,
        end=end,
        adjusted=adjusted,
    )
    prices.to_parquet(cache_path)
    return prices


def fetch_yahoo_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    *,
    adjusted: bool = DEFAULT_DATA_ADJUSTED,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required.")

    chunks = [tickers[index : index + 60] for index in range(0, len(tickers), 60)]
    frames = [_fetch_yahoo_price_chunk(chunk, start, end, adjusted=adjusted) for chunk in chunks]
    prices = pd.concat(frames, axis=1).reindex(columns=tickers)
    return prices.dropna(how="all")


def _fetch_yahoo_price_chunk(
    tickers: list[str],
    start: str,
    end: str | None,
    *,
    adjusted: bool,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=adjusted,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no rows.")

    field = "Close" if adjusted else "Adj Close"
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise RuntimeError(f"Yahoo Finance response did not include {field!r}.")
        prices = raw[field].copy()
    else:
        prices = raw[[field]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all")
    return prices


def _filter_prices(prices: pd.DataFrame, *, start: str, end: str | None) -> pd.DataFrame:
    filtered = prices.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    return filtered.dropna(how="all")
