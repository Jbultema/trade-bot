from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.config import (
    BotConfig,
    DataConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
)
from trade_bot.research.backtest_qc import run_backtest_qc_gauntlet


def test_backtest_qc_gauntlet_writes_core_artifacts(tmp_path: Path) -> None:
    config = BotConfig(
        data=DataConfig(start="2020-01-01", cache_dir=str(tmp_path)),
        execution=ExecutionConfig(
            initial_capital=1000.0,
            transaction_cost_bps=1.0,
            rebalance="W-WED",
            signal_lag_days=1,
        ),
        primary_strategy="candidate",
        universe={"test": ["QQQ", "NVDA", "SMH", "AMZN", "BIL", "SPY"]},
        strategies={
            "candidate": StrategyConfig(
                type="dual_momentum",
                tickers=["QQQ", "NVDA", "SMH", "AMZN"],
                lookback_days=21,
                skip_days=2,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                volatility_lookback_days=21,
                trend_filter_days=None,
                max_asset_weight=0.60,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.18,
                    lookback_days=21,
                ),
                drawdown_control=DrawdownControlConfig(
                    equity_lookback_days=42,
                    max_drawdown=-0.12,
                    risk_multiplier=0.50,
                ),
            )
        },
    )

    gauntlet = run_backtest_qc_gauntlet(
        config=config,
        prices=_prices(),
        strategy_name="candidate",
        output_dir=tmp_path / "qc",
        benchmark_tickers=("SPY", "QQQ"),
    )

    assert gauntlet.headline.iloc[0]["cagr"] == gauntlet.headline.iloc[0]["cagr"]
    assert "No direct leakage" in gauntlet.readout
    assert gauntlet.artifacts["summary"].exists()
    assert (tmp_path / "qc" / "causality.csv").exists()
    assert (tmp_path / "qc" / "universe_ablations.csv").exists()
    causality = pd.read_csv(tmp_path / "qc" / "causality.csv")
    assert bool(causality.iloc[0]["passed"])


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=320)
    frame = pd.DataFrame(index=index)
    for offset, ticker in enumerate(["QQQ", "NVDA", "SMH", "AMZN", "BIL", "SPY"], start=1):
        drift = 0.0002 * offset
        cycle = pd.Series(range(len(index)), index=index).map(lambda value: (value % 17) / 10000)
        frame[ticker] = 100.0 * (1.0 + drift + cycle).cumprod()
    return frame
