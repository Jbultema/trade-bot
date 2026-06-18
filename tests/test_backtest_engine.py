from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import run_backtest
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
