from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import (
    StaleHeldPositionError,
    build_execution_causality_trace,
    run_backtest,
)
from trade_bot.config import DrawdownControlConfig, ExecutionConfig


def test_backtest_shifts_weights_by_signal_lag() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 121.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    target_weights = pd.DataFrame(
        {"SPY": [1.0, 1.0, 1.0]},
        index=prices.index,
    )

    result = run_backtest(
        "lagged",
        prices,
        target_weights,
        ExecutionConfig(
            initial_capital=100.0,
            transaction_cost_bps=0.0,
            rebalance="daily",
            signal_lag_days=1,
        ),
    )

    assert result.weights["SPY"].tolist() == [0.0, 1.0, 1.0]
    assert result.returns.tolist() == pytest.approx([0.0, 0.10, 0.10])
    assert round(result.equity.iloc[-1], 2) == 121.00


def test_execution_causality_trace_exposes_close_boundary_assumption() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"SPY": range(100, 110)}, index=dates, dtype=float)
    targets = pd.DataFrame({"SPY": 1.0}, index=dates)

    lag_one = build_execution_causality_trace(
        prices,
        targets,
        ExecutionConfig(rebalance="W-WED", signal_lag_days=1),
    )
    lag_two = build_execution_causality_trace(
        prices,
        targets,
        ExecutionConfig(rebalance="W-WED", signal_lag_days=2),
    )

    assert lag_one["boundary_fill_approximation"].all()
    assert set(lag_one["causal_status"]) == {"close_boundary_approximation"}
    assert not lag_two["boundary_fill_approximation"].any()
    assert set(lag_two["causal_status"]) == {"strictly_after_feature_close"}


def test_backtest_normalizes_overinvested_long_only_weights() -> None:
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "QQQ": [100.0, 100.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    target_weights = pd.DataFrame(
        {
            "SPY": [2.0, 2.0, 2.0],
            "QQQ": [1.0, 1.0, 1.0],
        },
        index=prices.index,
    )

    result = run_backtest(
        "normalized",
        prices,
        target_weights,
        ExecutionConfig(
            initial_capital=100.0,
            transaction_cost_bps=0.0,
            rebalance="daily",
            signal_lag_days=1,
        ),
    )

    invested_rows = result.weights.sum(axis=1).iloc[1:]
    assert invested_rows.eq(1.0).all()
    assert result.weights["SPY"].iloc[1] == 2.0 / 3.0


def test_drawdown_control_does_not_use_same_day_returns_to_avoid_trigger_day() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 100.0, 50.0, 50.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    target_weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=prices.index)

    result = run_backtest(
        "drawdown_lagged",
        prices,
        target_weights,
        ExecutionConfig(
            initial_capital=100.0,
            transaction_cost_bps=0.0,
            rebalance="daily",
            signal_lag_days=1,
        ),
        drawdown_control=DrawdownControlConfig(
            equity_lookback_days=2,
            max_drawdown=-0.10,
            risk_multiplier=0.0,
        ),
    )

    assert result.weights["SPY"].iloc[2] == 1.0
    assert result.returns.iloc[2] == -0.50
    assert result.weights["SPY"].iloc[3] == 0.0


def test_backtest_fails_closed_when_a_held_asset_exceeds_price_staleness_limit() -> None:
    dates = pd.bdate_range("2024-01-02", periods=11)
    prices = pd.DataFrame(
        {
            "DEAD": [100.0, 101.0, 102.0, *([float("nan")] * 8)],
            "BIL": [100.0 + index * 0.01 for index in range(11)],
        },
        index=dates,
    )
    target_weights = pd.DataFrame(
        {"DEAD": 1.0, "BIL": 0.0},
        index=dates,
    )

    with pytest.raises(StaleHeldPositionError, match="DEAD"):
        run_backtest(
            "stale_exit",
            prices,
            target_weights,
            ExecutionConfig(
                initial_capital=100.0,
                transaction_cost_bps=0.0,
                rebalance="daily",
                signal_lag_days=1,
            ),
        )


def test_backtest_allows_an_unavailable_asset_that_was_never_held() -> None:
    dates = pd.bdate_range("2024-01-02", periods=11)
    prices = pd.DataFrame(
        {
            "DEAD": [100.0, 101.0, 102.0, *([float("nan")] * 8)],
            "BIL": [100.0 + index * 0.01 for index in range(11)],
        },
        index=dates,
    )
    target_weights = pd.DataFrame({"DEAD": 0.0, "BIL": 1.0}, index=dates)

    result = run_backtest(
        "unused_stale_asset",
        prices,
        target_weights,
        ExecutionConfig(
            initial_capital=100.0,
            transaction_cost_bps=0.0,
            rebalance="daily",
            signal_lag_days=1,
        ),
    )

    assert result.weights["DEAD"].eq(0.0).all()
