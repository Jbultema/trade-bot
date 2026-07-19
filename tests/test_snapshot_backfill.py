from __future__ import annotations

import pandas as pd

from trade_bot.research.snapshot_backfill import (
    build_snapshot_backfill_dates,
    build_snapshot_backfill_plan,
    snapshot_created_at_for_market_date,
)


def test_snapshot_backfill_plan_keeps_recent_daily_and_older_weekly_dates() -> None:
    available_dates = pd.bdate_range("2024-07-01", "2026-07-17")

    plan = build_snapshot_backfill_plan(
        available_dates,
        end_date="2026-07-17",
        years=2,
        daily_tail_days=30,
        weekly_frequency="W-WED",
    )

    selected_dates = pd.DatetimeIndex(plan.market_dates)
    recent_dates = available_dates[available_dates >= pd.Timestamp("2026-06-17")]
    older_dates = selected_dates[selected_dates < pd.Timestamp("2026-06-17")]

    assert selected_dates[-1] == pd.Timestamp("2026-07-17")
    assert set(recent_dates) <= set(selected_dates)
    assert selected_dates.is_unique
    assert selected_dates.min() >= pd.Timestamp("2024-07-17")
    assert older_dates.to_period("W-WED").nunique() == len(older_dates)
    assert len(plan.daily_market_dates) == len(recent_dates)
    assert len(plan.weekly_market_dates) == older_dates.to_period("W-WED").nunique()


def test_snapshot_backfill_dates_matches_plan_market_dates() -> None:
    available_dates = pd.bdate_range("2026-01-01", "2026-02-28")

    dates = build_snapshot_backfill_dates(
        available_dates,
        end_date="2026-02-27",
        years=1,
        daily_tail_days=5,
        weekly_frequency="W-FRI",
    )
    plan = build_snapshot_backfill_plan(
        available_dates,
        end_date="2026-02-27",
        years=1,
        daily_tail_days=5,
        weekly_frequency="W-FRI",
    )

    assert dates == list(plan.market_dates)


def test_snapshot_created_at_for_market_date_uses_configured_utc_close_hour() -> None:
    assert (
        snapshot_created_at_for_market_date("2026-07-17", market_close_utc_hour=22)
        == "2026-07-17T22:00:00+00:00"
    )
