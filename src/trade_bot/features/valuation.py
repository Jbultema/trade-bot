from __future__ import annotations

import pandas as pd


def rolling_peak_discount(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return price discount from the rolling high as negative percentages."""
    filled = prices.ffill()
    rolling_high = filled.rolling(window=window, min_periods=max(2, window // 3)).max()
    return (filled / rolling_high - 1.0).fillna(0.0)


def trend_discount(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return price discount or premium versus a moving-average trend anchor."""
    filled = prices.ffill()
    trend = filled.rolling(window=window, min_periods=max(2, window // 3)).mean()
    return (filled / trend - 1.0).fillna(0.0)


def normalized_discount(
    discount: pd.DataFrame | pd.Series,
    *,
    trigger_drawdown: float,
    deep_drawdown: float,
) -> pd.DataFrame | pd.Series:
    """Map drawdown discounts into 0-1 opportunity scores."""
    trigger = abs(trigger_drawdown)
    deep = max(abs(deep_drawdown), trigger + 1e-6)
    return ((-discount - trigger) / (deep - trigger)).clip(lower=0.0, upper=1.0).fillna(0.0)


def relative_repair_score(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    lookback_days: int,
    *,
    tolerance: float = -0.015,
    scale: float = 0.05,
) -> pd.Series:
    """Score whether a relative relationship is repairing rather than still breaking."""
    if numerator not in prices or denominator not in prices:
        return pd.Series(1.0, index=prices.index)
    relative = prices[numerator].ffill() / prices[denominator].ffill()
    change = relative.pct_change(lookback_days, fill_method=None)
    return ((change - tolerance) / scale).clip(lower=0.0, upper=1.0).fillna(0.5)
