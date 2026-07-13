from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import run_backtest
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.research.signal_state import (
    build_signal_state_report,
    confirmation_gated_weights,
    top_down_regime_signal,
)


def test_top_down_regime_signal_uses_available_market_proxies() -> None:
    dates = pd.bdate_range("2025-01-02", periods=90)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i for i in range(90)],
            "QQQ": [100 + i * 1.2 for i in range(90)],
            "RSP": [100 + i * 0.8 for i in range(90)],
            "HYG": [100 + i * 0.2 for i in range(90)],
            "LQD": [100 + i * 0.1 for i in range(90)],
            "VIXY": [120 - i * 0.4 for i in range(90)],
        },
        index=dates,
    )

    signal = top_down_regime_signal(prices)

    assert signal.iloc[-1]["top_down_signal"] == "bullish"
    assert float(signal.iloc[-1]["top_down_score"]) >= 0.6


def test_confirmation_gated_weights_routes_unconfirmed_risk_to_defensive() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    weights = pd.DataFrame({"QQQ": [0.5] * 4, "BIL": [0.5] * 4}, index=dates)
    signal_frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["QQQ"] * 4,
            "confirmation_scale": [1.0, 0.5, 0.0, 0.0],
        }
    )

    gated = confirmation_gated_weights(weights, signal_frame, defensive_ticker="BIL")

    assert gated.iloc[0]["QQQ"] == 0.5
    assert gated.iloc[1]["QQQ"] == 0.25
    assert gated.iloc[1]["BIL"] == 0.75
    assert gated.iloc[2]["QQQ"] == 0.0
    assert gated.iloc[2]["BIL"] == 1.0


def test_signal_state_report_compares_native_and_confirmation_gated_backtests() -> None:
    dates = pd.bdate_range("2025-01-02", periods=140)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.2 for i in range(140)],
            "QQQ": [100 + i * 0.4 for i in range(140)],
            "BIL": [100 + i * 0.01 for i in range(140)],
        },
        index=dates,
    )
    target_weights = pd.DataFrame({"QQQ": 0.7, "BIL": 0.3}, index=dates)
    execution = ExecutionConfig(signal_lag_days=1, rebalance="D", transaction_cost_bps=0.0)
    result = run_backtest("candidate", prices, target_weights, execution)
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ"],
        defensive_ticker="BIL",
        lookback_days=63,
        skip_days=0,
        top_n=1,
        min_return=0.0,
        trend_filter_days=21,
        volatility_lookback_days=21,
    )

    report = build_signal_state_report(
        result=result,
        prices=prices,
        strategy=strategy,
        execution=execution,
    )

    assert not report.assets.empty
    assert not report.backtest.empty
    assert set(report.backtest["variant"]) == {
        "Native strategy",
        "Confirmation-gated overlay",
    }
