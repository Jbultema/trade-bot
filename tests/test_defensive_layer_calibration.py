from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.defensive_layer_calibration import (
    _simulate_weekly_policy,
    frozen_weight_outcome,
    layer_flags,
)


def test_layer_flags_classify_material_defense_independently() -> None:
    flags = layer_flags(
        base_defensive=0.60,
        scenario_defensive_add=0.08,
        portfolio_defensive_add=0.03,
        base_threshold=0.55,
        scenario_add_threshold=0.05,
        portfolio_add_threshold=0.01,
    )

    assert flags == {
        "base_layer": True,
        "scenario_layer": True,
        "quantitative_sizing_layer": True,
        "portfolio_layer": True,
        "layer_count": 3,
        "layer_combination": "all_three",
    }


def test_layer_flags_do_not_count_immaterial_downstream_changes() -> None:
    flags = layer_flags(
        base_defensive=0.56,
        scenario_defensive_add=0.049,
        portfolio_defensive_add=0.009,
        base_threshold=0.55,
        scenario_add_threshold=0.05,
        portfolio_add_threshold=0.01,
    )

    assert flags["layer_combination"] == "base_only"
    assert flags["layer_count"] == 1


def test_frozen_weight_outcome_measures_layered_regret_and_cost() -> None:
    dates = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 110.0, 121.0, 133.1],
            "BIL": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    weights = pd.Series({"SPY": 0.50, "BIL": 0.50})

    gross_return, gross_drawdown = frozen_weight_outcome(
        prices, weights, 0, 3, initial_cost=0.0
    )
    net_return, net_drawdown = frozen_weight_outcome(
        prices, weights, 0, 3, initial_cost=0.01
    )

    assert gross_return == pytest.approx(0.1655)
    assert gross_drawdown == pytest.approx(0.0)
    assert net_return == pytest.approx(0.153845)
    assert net_drawdown == pytest.approx(-0.01)


def test_weekly_policy_replay_applies_weights_after_each_origin() -> None:
    dates = pd.bdate_range("2026-01-02", periods=6)
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]},
        index=dates,
    )
    origins = [dates[0], dates[3]]

    metrics = _simulate_weekly_policy(
        origins,
        prices,
        lambda _date: pd.Series({"SPY": 1.0}),
        transaction_cost_bps=0.0,
    )

    assert metrics["terminal_wealth"] == pytest.approx(1.61051)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
