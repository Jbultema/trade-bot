from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import run_backtest
from trade_bot.config import ExecutionConfig
from trade_bot.data.fred_data import FredSeries
from trade_bot.research.signal_inclusion import (
    apply_macro_pressure_overlay,
    run_signal_inclusion_tests,
)


def test_macro_pressure_overlay_moves_risk_weight_to_defensive_asset() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    target_weights = pd.DataFrame({"SPY": 1.0, "BIL": 0.0}, index=index)
    pressure = pd.Series([0.0, 0.3, 0.7, 0.8, 0.1], index=index)

    overlay = apply_macro_pressure_overlay(
        target_weights,
        pressure,
        defensive_ticker="BIL",
        pressure_threshold=0.65,
        risk_multiplier=0.5,
    )

    assert overlay.loc[index[0], "SPY"] == 1.0
    assert overlay.loc[index[2], "SPY"] == 0.5
    assert overlay.loc[index[2], "BIL"] == 0.5
    assert overlay.sum(axis=1).eq(1.0).all()


def test_signal_inclusion_tests_macro_category_against_base_strategy() -> None:
    index = pd.bdate_range("2023-01-02", periods=420)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + value * 0.1 for value in range(420)],
            "BIL": [100.0 + value * 0.01 for value in range(420)],
        },
        index=index,
    )
    target_weights = pd.DataFrame({"SPY": 1.0, "BIL": 0.0}, index=index)
    execution = ExecutionConfig(initial_capital=100.0, rebalance="D", signal_lag_days=1)
    base_result = run_backtest("base", prices, target_weights, execution)
    macro_data = pd.DataFrame({"STRESS": [float(value) for value in range(420)]}, index=index)
    catalog = (FredSeries("STRESS", "Stress", "financial_conditions", "risk_off_when_rising"),)

    inclusion = run_signal_inclusion_tests(
        prices,
        macro_data,
        catalog,
        base_result,
        execution,
        base_strategy_name="base",
        publication_lag_days=0,
        lookback_days=60,
        min_observations=40,
        pressure_threshold=0.50,
        risk_multiplier=0.50,
    )

    assert "macro:financial_conditions" in set(inclusion.summary["signal_group"])
    row = inclusion.summary.iloc[0]
    assert row["test_status"] == "tested"
    assert row["active_days"] > 0
    assert "macro_filter_financial_conditions" in inclusion.results
