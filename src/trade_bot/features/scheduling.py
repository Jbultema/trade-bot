"""Causal planned review dates, independent of a captured file's endpoint."""
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=32)
def planned_ends(first_year: int, last_year: int, rule: str) -> pd.DatetimeIndex:
    start, stop = f"{first_year}-01-01", f"{last_year+1}-01-10"
    calendar = xcals.get_calendar("XNYS", start=start, end=stop)
    holidays = calendar.regular_holidays.holidays(start, stop)
    planned = pd.date_range(start, stop, freq=pd.offsets.CustomBusinessDay(holidays=holidays))
    groups = planned.to_period(rule)
    return pd.DatetimeIndex(pd.Series(planned, index=planned).groupby(groups).max().to_numpy())


def rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    if index.empty or rule.lower() in {"daily", "d"}:
        return index
    planned = planned_ends(index.min().year, index.max().year, rule)
    return index[index.isin(planned)]
