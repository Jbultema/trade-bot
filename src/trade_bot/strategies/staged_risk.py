"""Native staged cash budgets with market, credit and participation confirmation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def staged_budget(prices: pd.DataFrame, *, lookback: int = 126, mode: str = "confirmed") -> pd.Series:
    if lookback not in (126, 252) or mode not in {"gentle", "confirmed", "repair"}:
        raise ValueError("Unknown preregistered staged-risk specification")
    required = ["SPY", "QQQ", "HYG", "LQD", "RSP"]
    if not set(required) <= set(prices):
        raise ValueError("Staged risk requires market, credit and participation evidence")
    panel = prices[required].astype(float)
    if (not panel.index.is_monotonic_increasing or not panel.index.is_unique
            or not panel.columns.is_unique or np.isinf(panel.to_numpy()).any()
            or panel.le(0).any(axis=None)):
        raise ValueError("Invalid staged-risk observations")
    market = panel.SPY
    observed = panel.notna().all(axis=1)
    if (observed.cummax() & ~observed).any():
        raise ValueError("Missing staged-risk evidence after initialization")
    credit, breadth = panel.HYG / panel.LQD, panel.RSP / market
    dd = market.div(market.rolling(lookback, min_periods=lookback).max()).sub(1)
    confirmations = (panel.QQQ.lt(panel.QQQ.rolling(63).mean()).astype(int)
        + credit.lt(credit.rolling(63).mean()).astype(int)
        + breadth.lt(breadth.rolling(63).mean()).astype(int))
    valid = panel.notna().rolling(lookback, min_periods=lookback).sum().eq(lookback).all(axis=1) & dd.notna()
    warning = dd.le(-.06)
    hard = dd.le(-.08) & confirmations.ge(2)
    liquidation = dd.le(-.15) & confirmations.ge(2)
    watch = dd.le(-.03) | confirmations.ge(2)
    desired = pd.Series(1., index=prices.index)
    desired.loc[watch] = .85
    desired.loc[warning] = .65
    desired.loc[hard] = .50 if mode == "gentle" else .35
    desired.loc[liquidation] = .35 if mode == "gentle" else .20
    desired.loc[~valid] = 0.
    if mode != "repair":
        return desired.rename("native_risk_budget")
    repair = market.gt(market.rolling(21).mean()) & market.pct_change(5, fill_method=None).gt(0)
    result, previous = [], 0.
    for budget, repaired, available in zip(desired, repair, valid, strict=True):
        if not available:
            previous = 0.
        elif budget < previous:
            previous = budget
        elif repaired:
            previous = min(budget, previous + .15)
        result.append(previous)
    return pd.Series(result, index=prices.index, name="native_risk_budget")


def size_native_weights(prices, weights, defensive_ticker, *, lookback=126, mode="confirmed"):
    budget = staged_budget(prices, lookback=lookback, mode=mode)
    risk = [name for name in weights if name != defensive_ticker]
    result = weights.copy()
    original = result[risk].sum(axis=1)
    result[risk] = result[risk].mul(budget, axis=0)
    released = original - result[risk].sum(axis=1)
    if defensive_ticker:
        if defensive_ticker not in result:
            raise ValueError("Native defensive instrument is absent")
        result[defensive_ticker] += released
    return result
