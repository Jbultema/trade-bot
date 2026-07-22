from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.baselines import (
    assemble_configured_baseline_from_results,
    build_configured_strategy_results,
    run_configured_baselines_from_frames,
    slice_backtest_results,
)


def test_sliced_causal_results_match_truncated_recomputation() -> None:
    index = pd.bdate_range("2022-01-03", periods=520)
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, len(index)))),
            "BIL": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.0004, len(index)))),
        },
        index=index,
    )
    config = BotConfig(
        data=DataConfig(start=str(index.min().date())),
        execution=ExecutionConfig(rebalance="W-WED", signal_lag_days=1),
        primary_strategy="core",
        universe={"core": ["SPY", "BIL"]},
        strategies={
            "core": StrategyConfig(
                type="absolute_momentum",
                tickers=["SPY"],
                defensive_ticker="BIL",
                moving_average_days=80,
            )
        },
    )
    through = index[-75]
    truncated_prices = prices.loc[:through]

    direct = run_configured_baselines_from_frames(
        config,
        prices=truncated_prices,
        event_config_path=None,
        news_config_path=None,
        as_of=through,
    )
    full_results = build_configured_strategy_results(config, prices)
    reused = assemble_configured_baseline_from_results(
        config,
        prices=truncated_prices,
        results=slice_backtest_results(full_results, through),
        event_config_path=None,
        news_config_path=None,
        as_of=through,
    )

    assert_frame_equal(direct.results["core"].weights, reused.results["core"].weights)
    assert_series_equal(direct.results["core"].returns, reused.results["core"].returns)
    assert_frame_equal(direct.trade_decision.position_plan, reused.trade_decision.position_plan)
    assert_frame_equal(direct.trade_decision.summary, reused.trade_decision.summary)
    assert direct.current_state.risk_score == reused.current_state.risk_score
    assert direct.current_state.risk_status == reused.current_state.risk_status
