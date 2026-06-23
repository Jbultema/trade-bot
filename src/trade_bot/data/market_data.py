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
    cached = _read_cached_prices(cache_path)

    if cached is not None and not refresh:
        cached_filtered = _filter_prices(
            cached.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )
        missing = _missing_or_empty_tickers(cached_filtered, ordered_tickers)
        if not missing:
            return cached_filtered

        fetched = fetch_yahoo_prices(
            missing,
            start=start,
            end=end,
            adjusted=adjusted,
        )
        prices = _merge_price_frames(cached, fetched, ordered_tickers)
        prices.to_parquet(cache_path)
        return _filter_prices(
            prices.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )

    try:
        fetched = fetch_yahoo_prices(
            ordered_tickers,
            start=start,
            end=end,
            adjusted=adjusted,
        )
    except Exception:
        if cached is None:
            raise
        cached_filtered = _filter_prices(
            cached.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )
        if cached_filtered.empty:
            raise
        return cached_filtered

    prices = _merge_price_frames(cached, fetched, ordered_tickers)
    filtered = _filter_prices(
        prices.reindex(columns=ordered_tickers),
        start=start,
        end=end,
    )
    if filtered.empty and cached is not None:
        cached_filtered = _filter_prices(
            cached.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )
        if not cached_filtered.empty:
            return cached_filtered
    prices.to_parquet(cache_path)
    return filtered


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


def _read_cached_prices(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    cached = pd.read_parquet(cache_path)
    cached.index = pd.to_datetime(cached.index)
    return cached.sort_index()


def _merge_price_frames(
    cached: pd.DataFrame | None,
    fetched: pd.DataFrame,
    ordered_tickers: list[str],
) -> pd.DataFrame:
    fetched = fetched.copy()
    fetched.index = pd.to_datetime(fetched.index)
    fetched = fetched.sort_index()
    if cached is None:
        return fetched.reindex(columns=ordered_tickers)
    columns = sorted(set(cached.columns) | set(fetched.columns) | set(ordered_tickers))
    return fetched.reindex(columns=columns).combine_first(cached.reindex(columns=columns))


def _missing_or_empty_tickers(prices: pd.DataFrame, tickers: list[str]) -> list[str]:
    missing: list[str] = []
    for ticker in tickers:
        if ticker not in prices.columns or prices[ticker].dropna().empty:
            missing.append(ticker)
    return missing


def _filter_prices(prices: pd.DataFrame, *, start: str, end: str | None) -> pd.DataFrame:
    filtered = prices.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    return filtered.dropna(how="all")
