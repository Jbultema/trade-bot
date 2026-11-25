from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from trade_bot.DEFAULTS import (
    DEFAULT_DATA_ADJUSTED,
    DEFAULT_MAX_PRICE_STALENESS_SESSIONS,
)

_INCREMENTAL_REFRESH_OVERLAP_DAYS = 21


class MarketDataRefreshError(RuntimeError):
    """Raised when an explicit market-data refresh cannot be completed honestly."""


class StaleMarketDataError(RuntimeError):
    """Raised when a price frame ends too far before its requested as-of date."""


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
    fetch_end = end or _safe_yahoo_end_date(datetime.now(UTC)).isoformat()
    freshness_end = end or (
        pd.Timestamp(fetch_end) - pd.Timedelta(days=1)
    ).date().isoformat()
    if cached is not None and end is None:
        cached = cached.loc[cached.index < pd.Timestamp(fetch_end)]

    if cached is not None and not refresh:
        cached_filtered = _filter_prices(
            cached.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )
        missing = _missing_or_empty_tickers(cached_filtered, ordered_tickers)
        stale = _terminal_staleness_sessions(
            cached_filtered.drop(columns=missing), end=freshness_end
        ) if len(missing) < len(ordered_tickers) else 0
        if not missing and stale <= DEFAULT_MAX_PRICE_STALENESS_SESSIONS:
            return cached_filtered

        fetch_tickers = ordered_tickers if stale > DEFAULT_MAX_PRICE_STALENESS_SESSIONS else missing
        fetch_start = start
        if stale > DEFAULT_MAX_PRICE_STALENESS_SESSIONS and not missing:
            fetch_start = _incremental_refresh_start(cached_filtered, requested_start=start)
        try:
            fetched = fetch_yahoo_prices(
                fetch_tickers,
                start=fetch_start,
                end=fetch_end,
                adjusted=adjusted,
            )
        except Exception as exc:
            if stale > DEFAULT_MAX_PRICE_STALENESS_SESSIONS:
                raise StaleMarketDataError(
                    f"Cached market data is {stale} business sessions stale and its refresh "
                    "failed; refusing to run on the stale frame."
                ) from exc
            raise
        _require_usable_fetched_prices(fetched, fetch_tickers, explicit_refresh=False)
        fetched, revised, fetch_start = _refresh_revised_history(
            cached, fetched, fetch_start=fetch_start, end=fetch_end, adjusted=adjusted
        )
        _require_terminal_freshness(fetched, end=freshness_end)
        prices = _merge_price_frames(cached, fetched, ordered_tickers, replace_tickers=revised)
        filtered = _filter_prices(
            prices.reindex(columns=ordered_tickers),
            start=start,
            end=end,
        )
        _require_terminal_freshness(filtered, end=freshness_end)
        prices.to_parquet(cache_path)
        _write_price_cache_metadata(
            cache_path.with_suffix(".metadata.json"),
            prices=prices,
            fetched_tickers=fetch_tickers,
            requested_tickers=ordered_tickers,
            start=start,
            end=end,
            adjusted=adjusted,
            refresh_mode="incremental",
            fetched_start=fetch_start,
            fetched_end_exclusive=fetch_end,
        )
        return filtered

    fetch_start = start
    if cached is not None:
        cached_requested = cached.reindex(columns=ordered_tickers)
        if not _missing_or_empty_tickers(cached_requested, ordered_tickers):
            fetch_start = _incremental_refresh_start(cached_requested, requested_start=start)

    try:
        fetched = fetch_yahoo_prices(
            ordered_tickers,
            start=fetch_start,
            end=fetch_end,
            adjusted=adjusted,
        )
    except Exception as exc:
        raise MarketDataRefreshError(
            "Explicit Yahoo price refresh failed; cached prices were not substituted."
        ) from exc

    _require_usable_fetched_prices(fetched, ordered_tickers, explicit_refresh=True)
    fetched, revised, fetch_start = _refresh_revised_history(
        cached, fetched, fetch_start=fetch_start, end=fetch_end, adjusted=adjusted
    )
    _require_terminal_freshness(fetched, end=freshness_end)

    prices = _merge_price_frames(cached, fetched, ordered_tickers, replace_tickers=revised)
    filtered = _filter_prices(
        prices.reindex(columns=ordered_tickers),
        start=start,
        end=end,
    )
    _require_terminal_freshness(filtered, end=freshness_end)
    prices.to_parquet(cache_path)
    _write_price_cache_metadata(
        cache_path.with_suffix(".metadata.json"),
        prices=prices,
        fetched_tickers=ordered_tickers,
        requested_tickers=ordered_tickers,
        start=start,
        end=end,
        adjusted=adjusted,
        refresh_mode=(
            "explicit_incremental"
            if refresh and fetch_start != start
            else "explicit" if refresh else "initial"
        ),
        fetched_start=fetch_start,
        fetched_end_exclusive=fetch_end,
    )
    return filtered


def _write_price_cache_metadata(
    path: Path,
    *,
    prices: pd.DataFrame,
    fetched_tickers: list[str],
    requested_tickers: list[str],
    start: str,
    end: str | None,
    adjusted: bool,
    refresh_mode: str,
    fetched_start: str | None = None,
    fetched_end_exclusive: str | None = None,
) -> None:
    valid_index = pd.to_datetime(prices.dropna(how="all").index, errors="coerce")
    valid_index = valid_index[~pd.isna(valid_index)]
    payload = {
        "schema_version": 1,
        "vendor": "Yahoo Finance",
        "client": "yfinance",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "refresh_mode": refresh_mode,
        "adjusted": bool(adjusted),
        "price_field": "Close" if adjusted else "Adj Close",
        "requested_start": start,
        "requested_end": end,
        "fetched_start": fetched_start or start,
        "fetched_end_exclusive": fetched_end_exclusive or end,
        "requested_tickers": requested_tickers,
        "fetched_tickers": fetched_tickers,
        "cache_rows": int(len(prices)),
        "cache_start": valid_index.min().date().isoformat() if len(valid_index) else None,
        "cache_end": valid_index.max().date().isoformat() if len(valid_index) else None,
        "known_limitations": [
            "current-universe replay unless separate membership evidence is supplied",
            "delisting returns are not independently verified",
            "vendor-adjusted history may be revised retroactively",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    *,
    replace_tickers: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    fetched = fetched.copy()
    fetched.index = pd.to_datetime(fetched.index)
    fetched = fetched.sort_index()
    if cached is None:
        return fetched.reindex(columns=ordered_tickers)
    cached = cached.drop(columns=list(replace_tickers), errors="ignore")
    columns = sorted(set(cached.columns) | set(fetched.columns) | set(ordered_tickers))
    return fetched.reindex(columns=columns).combine_first(cached.reindex(columns=columns))


def _refresh_revised_history(
    cached: pd.DataFrame | None,
    fetched: pd.DataFrame,
    *,
    fetch_start: str,
    end: str,
    adjusted: bool,
) -> tuple[pd.DataFrame, list[str], str]:
    if cached is None or cached.empty:
        return fetched, [], fetch_start
    revised, replace = [], []
    for ticker in fetched.columns.intersection(cached.columns):
        old = cached[ticker].dropna()
        fresh = fetched[ticker].dropna()
        if old.empty or fresh.empty or old.index.min() >= fresh.index.min():
            continue
        if pd.Timestamp(fetch_start) <= old.index.min():
            replace.append(ticker)
            continue
        overlap = old.index.intersection(fresh.index)
        if not len(overlap) or not np.allclose(
            old.loc[overlap], fresh.loc[overlap], rtol=1e-6, atol=1e-8
        ):
            revised.append(ticker)
    if not revised:
        return fetched, replace, fetch_start
    # Corporate actions revise adjusted levels before the overlap too. Never
    # splice those levels or guess a rescaling factor from a short window.
    full_start = min(pd.Timestamp(fetch_start), cached.index.min()).date().isoformat()
    try:
        full = fetch_yahoo_prices(revised, start=full_start, end=end, adjusted=adjusted)
        _require_usable_fetched_prices(full, revised, explicit_refresh=True)
    except Exception as exc:
        raise MarketDataRefreshError(
            f"Revised adjustment history for {revised} requires a full refresh; cache unchanged."
        ) from exc
    combined = fetched.drop(columns=revised).reindex(fetched.index.union(full.index))
    for ticker in revised:
        combined[ticker] = full[ticker]
    return combined.sort_index(), [*replace, *revised], full_start


def _missing_or_empty_tickers(prices: pd.DataFrame, tickers: list[str]) -> list[str]:
    missing: list[str] = []
    for ticker in tickers:
        if ticker not in prices.columns or prices[ticker].dropna().empty:
            missing.append(ticker)
    return missing


def _require_usable_fetched_prices(
    fetched: pd.DataFrame,
    tickers: list[str],
    *,
    explicit_refresh: bool,
) -> None:
    unusable = _missing_or_empty_tickers(fetched, tickers)
    if not unusable:
        return
    prefix = "Explicit refresh" if explicit_refresh else "Market-data update"
    raise MarketDataRefreshError(
        f"{prefix} returned no usable prices for {unusable}; cached columns were not substituted."
    )


def _require_terminal_freshness(prices: pd.DataFrame, *, end: str | None) -> None:
    staleness = _terminal_staleness_sessions(prices, end=end)
    if staleness <= DEFAULT_MAX_PRICE_STALENESS_SESSIONS:
        return
    latest = pd.to_datetime(prices.index, errors="coerce").max()
    raise StaleMarketDataError(
        f"Market data ends at {latest.date() if pd.notna(latest) else 'no valid date'}, "
        f"{staleness} business sessions before the requested as-of date; maximum allowed is "
        f"{DEFAULT_MAX_PRICE_STALENESS_SESSIONS}."
    )


def _terminal_staleness_sessions(prices: pd.DataFrame, *, end: str | None) -> int:
    if prices.empty:
        return DEFAULT_MAX_PRICE_STALENESS_SESSIONS + 1
    valid_index = pd.to_datetime(prices.dropna(how="all").index, errors="coerce")
    valid_index = valid_index[~pd.isna(valid_index)]
    if len(valid_index) == 0:
        return DEFAULT_MAX_PRICE_STALENESS_SESSIONS + 1
    # One fresh column cannot conceal a stale security held by a strategy.
    endpoints = [column.dropna().index.max() for _, column in prices.items()]
    if not endpoints or any(pd.isna(value) for value in endpoints):
        return DEFAULT_MAX_PRICE_STALENESS_SESSIONS + 1
    latest = pd.Timestamp(min(endpoints)).tz_localize(None).normalize()
    expected = (
        pd.Timestamp(end).tz_localize(None).normalize()
        if end is not None
        else pd.Timestamp.now().tz_localize(None).normalize()
    )
    if latest >= expected:
        return 0
    return len(pd.bdate_range(latest + pd.offsets.BDay(1), expected))


def _incremental_refresh_start(prices: pd.DataFrame, *, requested_start: str) -> str:
    valid_index = pd.to_datetime(prices.dropna(how="all").index, errors="coerce")
    valid_index = valid_index[~pd.isna(valid_index)]
    if len(valid_index) == 0:
        return requested_start
    overlap_start = pd.Timestamp(valid_index.max()).normalize() - pd.Timedelta(
        days=_INCREMENTAL_REFRESH_OVERLAP_DAYS
    )
    return max(pd.Timestamp(requested_start).normalize(), overlap_start).date().isoformat()


def _safe_yahoo_end_date(captured_at: datetime) -> date:
    eastern = captured_at.astimezone(ZoneInfo("America/New_York"))
    if eastern.weekday() < 5 and eastern.time() < time(16, 15):
        return eastern.date()
    return eastern.date() + timedelta(days=1)


def _filter_prices(prices: pd.DataFrame, *, start: str, end: str | None) -> pd.DataFrame:
    filtered = prices.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    return filtered.dropna(how="all")
