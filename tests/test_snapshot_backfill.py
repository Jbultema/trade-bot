from __future__ import annotations

import pandas as pd

from trade_bot.research.snapshot_backfill import (
    build_snapshot_backfill_dates,
    build_snapshot_backfill_plan,
    missing_market_backfill_dates,
    missing_period_backfill_dates,
    required_daily_retention_count,
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


def test_required_daily_retention_keeps_requested_tail_above_normal_floor() -> None:
    available_dates = pd.bdate_range("2026-08-01", "2026-11-10")
    plan = build_snapshot_backfill_plan(
        available_dates,
        start_date="2026-09-11",
        end_date="2026-11-10",
        daily_tail_days=90,
    )

    assert len(plan.daily_market_dates) > 30
    assert required_daily_retention_count(plan, minimum_market_days=30) == len(
        plan.daily_market_dates
    )


def test_required_daily_retention_includes_newer_concurrent_snapshot() -> None:
    available_dates = pd.bdate_range("2026-08-01", "2026-11-10")
    plan = build_snapshot_backfill_plan(
        available_dates,
        start_date="2026-09-11",
        end_date="2026-11-10",
        daily_tail_days=90,
    )

    retention_count = required_daily_retention_count(
        plan,
        minimum_market_days=30,
        retained_market_dates=[*plan.daily_market_dates, "2026-11-11"],
    )

    assert retention_count == len(plan.daily_market_dates) + 1


def test_required_daily_retention_preserves_normal_floor_for_short_tail() -> None:
    available_dates = pd.bdate_range("2026-10-01", "2026-11-10")
    plan = build_snapshot_backfill_plan(
        available_dates,
        end_date="2026-11-10",
        daily_tail_days=7,
    )

    assert required_daily_retention_count(plan, minimum_market_days=30) == 30


def test_snapshot_created_at_for_market_date_uses_configured_utc_close_hour() -> None:
    assert (
        snapshot_created_at_for_market_date("2026-07-17", market_close_utc_hour=22)
        == "2026-07-17T22:00:00+00:00"
    )


def test_missing_period_backfill_dates_adds_absent_canonical_quarter_ends() -> None:
    planned = pd.bdate_range("2025-01-01", "2025-12-31")
    existing = ["2025-02-14", "2025-08-05"]

    selected = missing_period_backfill_dates(
        planned,
        existing,
        period_frequency="Q",
    )

    assert selected == (
        pd.Timestamp("2025-03-31"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2025-09-30"),
        pd.Timestamp("2025-12-31"),
    )


def test_missing_period_backfill_accepts_existing_retention_week_anchor() -> None:
    planned = pd.bdate_range("2025-01-01", "2025-06-30")
    existing = ["2025-04-01"]

    selected = missing_period_backfill_dates(
        planned,
        existing,
        period_frequency="Q",
        retention_frequency="W-WED",
    )

    assert selected == (pd.Timestamp("2025-06-30"),)


def test_missing_market_backfill_dates_keeps_exact_daily_gaps() -> None:
    planned = pd.bdate_range("2026-10-01", "2026-10-09")
    existing = ["2026-10-01", "2026-10-02", "2026-10-06", "2026-10-08"]

    selected = missing_market_backfill_dates(planned, existing)

    assert selected == (
        pd.Timestamp("2026-10-05"),
        pd.Timestamp("2026-10-07"),
        pd.Timestamp("2026-10-09"),
    )
