from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_bot.research.defensive_correction_search import (
    MECHANISM_ROSTER,
    apply_existing_sleeve_relief,
    apply_satellite_bridge,
    build_mechanism_weight_path,
    build_point_in_time_correction_signals,
    effective_defensive_weight_path,
    ordinary_session_mask,
)


def _prices(periods: int = 900) -> pd.DataFrame:
    index = pd.bdate_range("2018-01-02", periods=periods)
    trend = np.linspace(100.0, 180.0, periods)
    return pd.DataFrame(
        {
            "SPY": trend,
            "QQQ": trend * 1.05,
            "RSP": trend * 0.98,
            "HYG": trend * 0.90,
            "LQD": trend * 0.85,
            "VIXY": np.linspace(80.0, 20.0, periods),
            "SPLV": trend * 0.95,
            "BIL": np.linspace(100.0, 105.0, periods),
        },
        index=index,
    )


def test_point_in_time_signals_do_not_use_same_day_price_jump() -> None:
    prices = _prices(300)
    prices.loc[prices.index[-80] :, ["SPY", "QQQ"]] = np.linspace(
        120.0,
        70.0,
        80,
    )[:, None]
    signals_before = build_point_in_time_correction_signals(prices)
    prices.loc[prices.index[-1], ["SPY", "QQQ"]] = 500.0
    signals_after = build_point_in_time_correction_signals(prices)

    assert not bool(signals_before.loc[prices.index[-1], "dual_trend_intact"])
    assert not bool(signals_after.loc[prices.index[-1], "dual_trend_intact"])


def test_existing_sleeve_relief_preserves_residual_cash_semantics() -> None:
    index = pd.bdate_range("2025-01-02", periods=2)
    base = pd.DataFrame({"QQQ": 0.25, "BIL": 0.02}, index=index)

    adjusted, relief = apply_existing_sleeve_relief(
        base,
        pd.Series(0.05, index=index),
    )

    assert relief.tolist() == pytest.approx([0.05, 0.05])
    assert adjusted["QQQ"].tolist() == pytest.approx([0.30, 0.30])
    assert adjusted["BIL"].tolist() == pytest.approx([0.0, 0.0])
    assert adjusted.sum(axis=1).tolist() == pytest.approx([0.30, 0.30])
    assert effective_defensive_weight_path(adjusted).tolist() == pytest.approx(
        [0.70, 0.70]
    )


def test_satellite_bridge_changes_destination_without_overallocating() -> None:
    index = pd.bdate_range("2025-01-02", periods=2)
    base = pd.DataFrame({"QQQ": 0.30, "BIL": 0.70}, index=index)

    adjusted, relief = apply_satellite_bridge(
        base,
        pd.Series(0.10, index=index),
        destination="SPLV",
        price_available=pd.Series(True, index=index),
    )

    assert relief.tolist() == pytest.approx([0.10, 0.10])
    assert adjusted["QQQ"].tolist() == pytest.approx([0.30, 0.30])
    assert adjusted["SPLV"].tolist() == pytest.approx([0.10, 0.10])
    assert adjusted["BIL"].tolist() == pytest.approx([0.60, 0.60])
    assert adjusted.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_ordinary_session_mask_excludes_named_crises_but_keeps_normal_days() -> None:
    index = pd.to_datetime(
        [
            "2008-09-15",
            "2014-05-01",
            "2020-03-16",
            "2025-04-01",
        ]
    )

    mask = ordinary_session_mask(index)

    assert mask.tolist() == [False, True, False, True]


def test_all_distinct_mechanisms_are_bounded_and_preserve_weight_budget() -> None:
    prices = _prices()
    signals = build_point_in_time_correction_signals(prices)
    base = pd.DataFrame(
        {"QQQ": 0.30, "BIL": 0.70},
        index=prices.index,
    )
    # Create a sharp defensive ramp so ramp-based candidates are exercised too.
    base.loc[base.index[-100:-95], ["QQQ", "BIL"]] = [0.70, 0.30]
    family = pd.Series(0.40, index=prices.index)

    for mechanism in MECHANISM_ROSTER:
        adjusted, relief = build_mechanism_weight_path(
            mechanism,
            base,
            prices,
            signals,
            family_median_defense=family,
        )

        assert relief.ge(0.0).all(), mechanism
        assert relief.le(0.20 + 1e-12).all(), mechanism
        assert adjusted.ge(-1e-12).all(axis=None), mechanism
        assert adjusted.sum(axis=1).le(1.0 + 1e-10).all(), mechanism
