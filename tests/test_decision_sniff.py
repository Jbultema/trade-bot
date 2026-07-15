from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.backtest.engine import run_backtest
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.dashboard.decision_sniff import build_operational_sniff_read


def test_build_operational_sniff_read_for_configured_strategy() -> None:
    dates = pd.bdate_range("2025-01-02", periods=150)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.2 for i in range(150)],
            "QQQ": [100 + i * 0.4 for i in range(150)],
            "BIL": [100 + i * 0.01 for i in range(150)],
        },
        index=dates,
    )
    execution = ExecutionConfig(signal_lag_days=1, rebalance="D", transaction_cost_bps=0.0)
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
    weights = pd.DataFrame({"QQQ": 0.7, "BIL": 0.3}, index=dates)
    result = run_backtest("candidate", prices, weights, execution)
    baseline_run = SimpleNamespace(prices=prices, results={"candidate": result})
    bot_config = SimpleNamespace(
        strategies={"candidate": strategy},
        execution=execution,
        primary_strategy="candidate",
    )

    read = build_operational_sniff_read(
        baseline_run=baseline_run,
        bot_config=bot_config,
        strategy_name="candidate",
    )

    assert read is not None
    assert read.strategy_name == "candidate"
    assert read.benchmark_ticker == "QQQ"
    assert not read.report.assets.empty
    assert read.false_alarm_update["historical_episode_starts"] >= 0
