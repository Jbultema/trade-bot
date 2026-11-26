"""A partial file cannot manufacture an early weekly or monthly review."""

import pandas as pd

from trade_bot.backtest.engine import _rebalance_weights, _scheduled_rebalance_dates


def test_incomplete_week_never_invents_a_tuesday_review():
    dates = pd.bdate_range("2026-11-02", "2026-11-11")
    target = pd.DataFrame({"SPY": range(len(dates))}, index=dates, dtype=float)
    prefix = target.loc[:"2026-11-10"]
    pd.testing.assert_frame_equal(_rebalance_weights(prefix, "W-WED"),
        _rebalance_weights(target, "W-WED").loc[prefix.index])
    assert pd.Timestamp("2026-11-10") not in _scheduled_rebalance_dates(prefix, "W-WED")


def test_regular_holiday_uses_known_previous_session_not_the_last_observed_row():
    dates = pd.bdate_range("2024-12-19", "2024-12-27").difference(pd.DatetimeIndex(["2024-12-25"]))
    weights = pd.DataFrame({"SPY": 1.}, index=dates)
    assert _scheduled_rebalance_dates(weights, "W-WED").tolist() == [pd.Timestamp("2024-12-24")]


def test_incomplete_month_does_not_rebalance_at_capture_cutoff():
    dates = pd.bdate_range("2026-11-02", "2026-11-10")
    target = pd.DataFrame({"SPY": 1.}, index=dates)
    assert _scheduled_rebalance_dates(target, "M").empty
    assert _rebalance_weights(target, "M").eq(0).all().all()
