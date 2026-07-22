from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.operating_history import _quant_risk_multiplier, _sample_history_dates


def test_operating_history_samples_weekly_with_daily_recent_tail() -> None:
    index = pd.bdate_range("2025-01-01", "2026-07-08")

    sampled = _sample_history_dates(
        index,
        start_date="2025-01-01",
        end_date="2026-07-08",
        frequency="W-WED",
        max_points=8,
        daily_tail_market_days=30,
        min_history_days=0,
    )

    sampled_index = pd.DatetimeIndex(sampled)
    expected_recent = index[-30:]
    tail_start = expected_recent.min()
    historical = sampled_index[sampled_index < tail_start]
    recent = sampled_index[sampled_index >= tail_start]

    assert len(historical) == 8
    assert set(recent.date) == set(expected_recent.date)
    assert historical.to_series().dt.to_period("W-WED").nunique() == len(historical)


def test_fast_history_uses_live_binding_constraint_rule() -> None:
    assert _quant_risk_multiplier("yellow", 0.76) == pytest.approx(0.76)
    assert _quant_risk_multiplier("orange", 0.80) == pytest.approx(0.65)
