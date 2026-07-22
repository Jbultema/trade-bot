from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import DEFAULT_MAX_PRICE_STALENESS_SESSIONS, TRADING_DAYS_PER_YEAR


def bounded_forward_fill(
    prices: pd.DataFrame | pd.Series,
    *,
    limit: int = DEFAULT_MAX_PRICE_STALENESS_SESSIONS,
) -> pd.DataFrame | pd.Series:
    """Fill ordinary market-calendar gaps without carrying a dead price forever."""
    return prices.ffill(limit=limit)


def unusable_required_price_columns(
    prices: pd.DataFrame,
    columns: Iterable[str],
    *,
    max_staleness_sessions: int = DEFAULT_MAX_PRICE_STALENESS_SESSIONS,
) -> list[str]:
    """Return required columns that are absent, empty, or stale at the frame's end."""

    required = list(dict.fromkeys(str(column) for column in columns))
    if not required:
        return []
    if prices.empty:
        return required

    frame = prices.sort_index()
    market_rows = frame.notna().any(axis=1).to_numpy()
    market_positions = np.flatnonzero(market_rows)
    if len(market_positions) == 0:
        return required
    latest_market_position = int(market_positions[-1])

    unusable: list[str] = []
    for column in required:
        if column not in frame.columns:
            unusable.append(column)
            continue
        valid_positions = np.flatnonzero(frame[column].notna().to_numpy())
        if len(valid_positions) == 0:
            unusable.append(column)
            continue
        if latest_market_position - int(valid_positions[-1]) > max_staleness_sessions:
            unusable.append(column)
    return unusable


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return bounded_forward_fill(prices).pct_change(fill_method=None).fillna(0.0)


def moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return bounded_forward_fill(prices).rolling(window=window, min_periods=window).mean()


def lookback_returns(prices: pd.DataFrame, lookback_days: int, skip_days: int = 0) -> pd.DataFrame:
    shifted = bounded_forward_fill(prices).shift(skip_days)
    return shifted / shifted.shift(lookback_days) - 1.0


def realized_volatility(returns: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def ulcer_index(equity: pd.Series) -> float:
    """Return the root mean square drawdown for an equity curve."""

    clean_equity = equity.dropna().astype(float)
    if clean_equity.empty:
        return float("nan")
    drawdowns = drawdown(clean_equity).clip(upper=0.0)
    return float(np.sqrt(np.square(drawdowns).mean()))


def rolling_drawdown(equity: pd.Series, window: int) -> pd.Series:
    running_max = equity.rolling(window=window, min_periods=window).max()
    return equity / running_max - 1.0
