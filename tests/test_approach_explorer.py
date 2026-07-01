from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.approach_explorer import (
    build_approach_allocation_transition_events,
    build_approach_backtest_result,
    build_approach_catalog,
    build_approach_change_log,
    build_approach_decision_events,
    build_approach_explanation,
    build_approach_exposure_history,
    build_approach_holding_stats,
    build_approach_mechanics,
    build_approach_position_summary,
    build_approach_steps,
    build_approach_weight_history,
    build_latest_approach_weights,
    execution_for_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.future_state_ml import StrategyDrawdownModelConfig


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
    drawdown_config = StrategyDrawdownModelConfig(
        model="base_rate",
        horizon_days=5,
        train_window_days=20,
        min_train_observations=5,
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
                "strategy_drawdown_model": "base_rate_ai_5d_dd8%",
                "strategy_drawdown_model_json": json.dumps(drawdown_config.__dict__),
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
                "calmar": 0.6,
            }
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    catalog = build_approach_catalog(_config(), experiment_root=tmp_path)

    assert {"baseline", "experiment"} == set(catalog["source"])
    experiment_row = catalog[catalog["source"] == "experiment"].iloc[0]
    assert experiment_row["promotion_decision"] == "promote_candidate"
    assert experiment_row["research_status"] == "operational_candidate"
    assert strategy_from_catalog_row(experiment_row).type == "dual_momentum"
    assert strategy_drawdown_model_from_catalog_row(experiment_row) == drawdown_config


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


def test_approach_plain_english_explains_saved_experiment_candidate() -> None:
    config = _config()
    strategy = config.strategies["dual"]
    row = pd.Series(
        {
            "family": "ai_beta",
            "role": "satellite",
            "promotion_decision": "evolve_next_iteration",
            "parent": "i09_ai_beta_bubble_escape_broader",
            "phase": "active_trading",
        }
    )

    execution = execution_for_catalog_row(row, config.execution)
    drawdown_model = StrategyDrawdownModelConfig(model="base_rate", horizon_days=21)
    explanation = " ".join(
        build_approach_explanation(
            strategy,
            row,
            config,
            execution=execution,
            strategy_drawdown_model=drawdown_model,
        )
    )

    assert "dual momentum" in explanation
    assert "top 1" in explanation
    assert "BIL" in explanation
    assert "strategy-specific ML drawdown guard" in explanation
    assert "i09_ai_beta_bubble_escape_broader" in explanation
    assert execution.rebalance == "D"


def test_approach_position_history_outputs_chart_and_change_tables() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
    )
    index = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            "QQQ": [100.0, 104.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0],
            "BIL": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07],
        },
        index=index,
    )

    result, missing = build_approach_backtest_result(
        prices,
        strategy,
        ExecutionConfig(rebalance="D", signal_lag_days=1),
    )

    assert result is not None
    assert missing == []
    weight_history = build_approach_weight_history(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=8,
    )
    exposure_history = build_approach_exposure_history(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=8,
    )
    summary = build_approach_position_summary(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=8,
        material_change=0.05,
    )
    change_log = build_approach_change_log(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=8,
        material_change=0.05,
    )
    holding_stats = build_approach_holding_stats(result.weights, lookback_days=8)

    assert not weight_history.empty
    assert {"risk_assets", "defensive", "cash_or_unallocated"}.issubset(exposure_history.columns)
    assert summary["metric"].str.contains("Material change days").any()
    assert not change_log.empty
    assert "position_after" in change_log.columns
    assert not holding_stats.empty


def test_approach_backtest_reconstructs_strategy_drawdown_ml_overlay() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
    )
    index = pd.bdate_range("2024-01-01", periods=40)
    trend = pd.Series(range(40), dtype=float)
    qqq = pd.concat(
        [100.0 + trend.iloc[:20] * 0.80, 116.0 - pd.Series(range(20), dtype=float) * 1.40],
        ignore_index=True,
    )
    prices = pd.DataFrame(
        {
            "SPY": 100.0 + trend * 0.20,
            "QQQ": qqq.to_numpy(),
            "BIL": 100.0 + trend * 0.01,
        },
        index=index,
    )
    drawdown_model = StrategyDrawdownModelConfig(
        model="base_rate",
        horizon_days=5,
        train_window_days=20,
        min_train_observations=5,
        activation_probability=0.0,
        stress_multiplier=0.50,
        min_multiplier=0.50,
    )

    result, missing = build_approach_backtest_result(
        prices,
        strategy,
        ExecutionConfig(rebalance="D", signal_lag_days=1),
        strategy_drawdown_model=drawdown_model,
    )

    assert result is not None
    assert missing == []
    assert "BIL" in result.target_weights.columns
    assert result.target_weights["BIL"].max() > 0.0


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


def test_approach_position_history_supports_custom_windows_and_transition_events() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
    )
    index = pd.bdate_range("2024-01-01", periods=14)
    prices = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 90, 88, 87, 89, 94, 98, 101, 103, 104, 105],
            "QQQ": [100, 103, 106, 109, 92, 86, 84, 88, 96, 103, 110, 114, 116, 118],
            "BIL": [100 + idx * 0.01 for idx in range(14)],
        },
        index=index,
        dtype=float,
    )
    result, missing = build_approach_backtest_result(
        prices,
        strategy,
        ExecutionConfig(rebalance="D", signal_lag_days=1),
    )

    assert result is not None
    assert missing == []
    start = index[4]
    end = index[11]
    weight_history = build_approach_weight_history(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=None,
        start=start,
        end=end,
    )
    exposure_history = build_approach_exposure_history(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=None,
        start=start,
        end=end,
    )
    summary = build_approach_position_summary(
        result.weights,
        defensive_ticker="BIL",
        lookback_days=None,
        start=start,
        end=end,
    )
    events = build_approach_allocation_transition_events(
        result,
        defensive_ticker="BIL",
        start=start,
        end=end,
        material_change=0.01,
    )

    assert weight_history.index.min() >= start
    assert weight_history.index.max() <= end
    assert exposure_history.index.min() >= start
    assert {"risk_assets", "defensive", "cash_or_unallocated"}.issubset(exposure_history.columns)
    assert summary["interpretation"].str.contains("selected").any()
    assert "Worst drawdown point" in set(events["event"])
    assert "risk_weight_at_event" in events.columns


def test_approach_decision_events_keep_multiple_material_allocation_moves() -> None:
    dates = pd.bdate_range("2026-01-02", periods=8)
    equity = pd.Series([100, 101, 99, 102, 105, 104, 108, 110], index=dates, dtype=float)
    weights = pd.DataFrame(
        {
            "QQQ": [0.70, 0.70, 0.25, 0.25, 0.65, 0.65, 0.35, 0.35],
            "IWM": [0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.25, 0.25],
            "BIL": [0.20, 0.20, 0.70, 0.70, 0.25, 0.25, 0.40, 0.40],
        },
        index=dates,
    )
    result = BacktestResult(
        name="candidate",
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        gross_returns=equity.pct_change().fillna(0.0),
        weights=weights,
        target_weights=weights,
        turnover=weights.diff().abs().sum(axis=1).fillna(0.0),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    events = build_approach_decision_events(
        result,
        defensive_ticker="BIL",
        start=dates[0],
        end=dates[-1],
        material_change=0.10,
        max_events=10,
    )

    assert len(events) >= 3
    assert {"De-risking move", "Re-risking move"}.issubset(set(events["event"]))
    assert {"inferred_driver", "top_adds", "top_reductions", "forward_return_3m"}.issubset(
        events.columns
    )
    derisk = events[events["event"] == "De-risking move"].iloc[0]
    assert derisk["top_reductions"] != "none"
    assert "off-ramp" in derisk["inferred_driver"]
