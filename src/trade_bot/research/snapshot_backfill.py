from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, time

import pandas as pd


@dataclass(frozen=True)
class SnapshotBackfillPlan:
    market_dates: tuple[pd.Timestamp, ...]
    daily_market_dates: tuple[pd.Timestamp, ...]
    weekly_market_dates: tuple[pd.Timestamp, ...]
    start_date: str
    end_date: str
    daily_cutoff_date: str
    weekly_frequency: str


def build_snapshot_backfill_dates(
    available_dates: Iterable[object],
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    years: int = 2,
    daily_tail_days: int = 30,
    weekly_frequency: str = "W-WED",
) -> list[pd.Timestamp]:
    return list(
        build_snapshot_backfill_plan(
            available_dates,
            start_date=start_date,
            end_date=end_date,
            years=years,
            daily_tail_days=daily_tail_days,
            weekly_frequency=weekly_frequency,
        ).market_dates
    )


def build_snapshot_backfill_plan(
    available_dates: Iterable[object],
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    years: int = 2,
    daily_tail_days: int = 30,
    weekly_frequency: str = "W-WED",
) -> SnapshotBackfillPlan:
    if years <= 0:
        msg = "years must be positive"
        raise ValueError(msg)
    if daily_tail_days < 0:
        msg = "daily_tail_days must be non-negative"
        raise ValueError(msg)

    normalized_dates = _normalize_dates(available_dates)
    if normalized_dates.empty:
        msg = "No available market dates were supplied for snapshot backfill."
        raise ValueError(msg)

    requested_end = (
        _normalize_one_date(end_date) if end_date is not None else normalized_dates.max()
    )
    available_through_end = normalized_dates[normalized_dates <= requested_end]
    if available_through_end.empty:
        msg = f"No available market dates exist on or before {requested_end.date()}."
        raise ValueError(msg)
    actual_end = available_through_end.max()
    requested_start = (
        _normalize_one_date(start_date)
        if start_date is not None
        else actual_end - pd.DateOffset(years=years)
    )
    if requested_start > actual_end:
        msg = "start_date must be on or before end_date."
        raise ValueError(msg)

    candidate_dates = available_through_end[available_through_end >= requested_start]
    if candidate_dates.empty:
        msg = "No available market dates fall inside the requested backfill window."
        raise ValueError(msg)

    daily_cutoff = actual_end - pd.Timedelta(days=daily_tail_days)
    daily_dates = tuple(candidate_dates[candidate_dates >= daily_cutoff])
    weekly_dates = _latest_date_per_week(
        candidate_dates[candidate_dates < daily_cutoff],
        weekly_frequency=weekly_frequency,
    )
    market_dates = tuple(
        pd.DatetimeIndex([*weekly_dates, *daily_dates]).drop_duplicates().sort_values()
    )

    return SnapshotBackfillPlan(
        market_dates=market_dates,
        daily_market_dates=daily_dates,
        weekly_market_dates=weekly_dates,
        start_date=str(candidate_dates.min().date()),
        end_date=str(actual_end.date()),
        daily_cutoff_date=str(daily_cutoff.date()),
        weekly_frequency=weekly_frequency,
    )


def snapshot_created_at_for_market_date(
    market_date: str | pd.Timestamp,
    *,
    market_close_utc_hour: int = 22,
) -> str:
    if market_close_utc_hour < 0 or market_close_utc_hour > 23:
        msg = "market_close_utc_hour must be between 0 and 23."
        raise ValueError(msg)
    date_value = _normalize_one_date(market_date).date()
    timestamp = pd.Timestamp.combine(
        date_value,
        time(hour=market_close_utc_hour, tzinfo=UTC),
    )
    return timestamp.isoformat()


def _latest_date_per_week(
    dates: pd.DatetimeIndex,
    *,
    weekly_frequency: str,
) -> tuple[pd.Timestamp, ...]:
    if dates.empty:
        return ()
    frame = pd.DataFrame({"market_date": dates.sort_values()})
    frame["weekly_bucket"] = frame["market_date"].dt.to_period(weekly_frequency).astype(str)
    weekly = frame.groupby("weekly_bucket", sort=True)["market_date"].max()
    return tuple(pd.DatetimeIndex(weekly).sort_values())


def _normalize_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(list(values), errors="coerce")
    dates = pd.DatetimeIndex(dates).dropna()
    if dates.empty:
        return dates
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    return dates.normalize().drop_duplicates().sort_values()


def _normalize_one_date(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        msg = f"Invalid date value: {value!r}"
        raise ValueError(msg)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()
