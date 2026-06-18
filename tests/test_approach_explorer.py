from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.approach_explorer import (
    build_approach_catalog,
    build_approach_mechanics,
    build_approach_steps,
    build_latest_approach_weights,
    strategy_from_catalog_row,
)


def test_approach_catalog_includes_baselines_and_experiment_manifests(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_01"
    iteration_dir.mkdir()
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ"],
        lookback_days=63,
        skip_days=5,
        top_n=1,
        defensive_ticker="BIL",
    )
    pd.DataFrame(
        [
            {
                "strategy": "experiment_candidate",
                "phase": "broad",
                "family": "core",
                "role": "candidate_core",
                "parent": "",
                "hypothesis": "Test broad momentum.",
                "strategy_json": json.dumps(strategy.model_dump(mode="json")),
            }
        ]
    ).to_csv(iteration_dir / "candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "experiment_candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.9,
                "cagr": 0.1,
                "max_drawdown": -0.2,
                "calmar": 0.5,
            }
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    catalog = build_approach_catalog(_config(), experiment_root=tmp_path)

    assert {"baseline", "experiment"} == set(catalog["source"])
    experiment_row = catalog[catalog["source"] == "experiment"].iloc[0]
    assert experiment_row["promotion_decision"] == "promote_candidate"
    assert strategy_from_catalog_row(experiment_row).type == "dual_momentum"


def test_approach_explanation_surfaces_strategy_mechanics() -> None:
    config = _config()
    strategy = config.strategies["dual"]

    mechanics = build_approach_mechanics(strategy, config)
    steps = build_approach_steps(strategy)

    assert "Momentum lookback" in set(mechanics["component"])
    assert "Defensive asset" in set(mechanics["component"])
    assert steps["detail"].str.contains("absolute return hurdle").any()


def test_latest_approach_weights_uses_loaded_prices() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
    )
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0],
            "QQQ": [100.0, 102.0, 104.0, 106.0],
            "BIL": [100.0, 100.01, 100.02, 100.03],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    latest_weights = build_latest_approach_weights(prices, strategy)

    assert latest_weights.iloc[0]["ticker"] == "QQQ"
    assert latest_weights.iloc[0]["weight"] == 1.0


def _config() -> BotConfig:
    return BotConfig(
        data=DataConfig(start="2020-01-01"),
        execution=ExecutionConfig(
            initial_capital=100.0,
            transaction_cost_bps=5.0,
            rebalance="W-FRI",
            signal_lag_days=1,
        ),
        universe={"core": ["SPY", "QQQ", "BIL"]},
        strategies={
            "dual": StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ"],
                lookback_days=63,
                skip_days=5,
                top_n=1,
                defensive_ticker="BIL",
                min_return=0.01,
            )
        },
    )
