from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import TRADING_DAYS_PER_YEAR


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.ffill().pct_change(fill_method=None).fillna(0.0)


def moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.ffill().rolling(window=window, min_periods=window).mean()


def lookback_returns(prices: pd.DataFrame, lookback_days: int, skip_days: int = 0) -> pd.DataFrame:
    shifted = prices.ffill().shift(skip_days)
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
