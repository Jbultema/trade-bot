from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.backtest_pbo import (
    estimate_probability_of_backtest_overfitting,
    run_backtest_pbo_gauntlet,
)


def test_probability_of_backtest_overfitting_estimates_split_ranks() -> None:
    index = pd.bdate_range("2020-01-01", periods=160)
    returns = pd.DataFrame(index=index)
    returns["steady"] = 0.0008
    returns["overfit_a"] = [
        0.004 if (idx // 40) in {0, 2} else -0.003
        for idx in range(len(index))
    ]
    returns["overfit_b"] = [
        0.004 if (idx // 40) in {1, 3} else -0.003
        for idx in range(len(index))
    ]

    result = estimate_probability_of_backtest_overfitting(
        returns,
        partitions=4,
        metric="mean_return",
    )

    summary = result.summary.iloc[0]
    assert summary["strategy_count"] == 3
    assert summary["valid_splits"] == 6
    assert 0.0 <= summary["pbo_probability"] <= 1.0
    assert {"selected_strategy", "relative_rank", "overfit"}.issubset(result.splits.columns)
    assert not result.strategy_selection.empty


def test_backtest_pbo_gauntlet_writes_artifacts(tmp_path: Path) -> None:
    config = BotConfig(
        data=DataConfig(start="2020-01-01", cache_dir=str(tmp_path)),
        execution=ExecutionConfig(
            initial_capital=1000.0,
            rebalance="W-WED",
            signal_lag_days=1,
            transaction_cost_bps=1.0,
        ),
        primary_strategy="qqq_momo",
        universe={"test": ["QQQ", "SMH", "SPY", "VEA", "BIL"]},
        strategies={
            "qqq_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["QQQ", "SMH"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=-1.0,
            ),
            "global_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "VEA"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=-1.0,
            ),
            "spy_hold": StrategyConfig(
                type="buy_hold",
                tickers=["SPY"],
            ),
        },
    )

    gauntlet = run_backtest_pbo_gauntlet(
        config=config,
        prices=_prices(),
        output_dir=tmp_path / "pbo",
        experiment_root=tmp_path / "missing_experiments",
        top_n=3,
        partitions=4,
        metric="mean_return",
        min_observations=60,
    )

    assert gauntlet.artifacts["summary"].exists()
    assert gauntlet.artifacts["splits"].exists()
    assert gauntlet.artifacts["strategy_selection"].exists()
    assert gauntlet.artifacts["strategy_stats"].exists()
    assert gauntlet.result.summary.iloc[0]["strategy_count"] >= 2
    assert "PBO probability" in gauntlet.readout


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=260)
    frame = pd.DataFrame(index=index)
    for offset, ticker in enumerate(["QQQ", "SMH", "SPY", "VEA", "BIL"], start=1):
        drift = 0.00015 * offset
        cycle = pd.Series(range(len(index)), index=index).map(lambda value: (value % 23) / 12000)
        frame[ticker] = 100.0 * (1.0 + drift + cycle).cumprod()
    return frame
