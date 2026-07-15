from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.leadership_diagnostics import (
    build_factor_beta_frame,
    build_qqq_underperformance_frame,
    build_router_model_comparison,
    build_router_scenario_summary,
    build_router_selection_summary,
    build_scenario_strategy_heatmap,
    build_tech_dependence_frame,
    run_leadership_diagnostics,
    run_leadership_impairment,
    run_walk_forward_strategy_router,
)
from trade_bot.strategies.momentum import build_strategy_weights


def test_tech_dependence_and_betas_are_reported() -> None:
    results = _results()
    prices = _prices()

    tech = build_tech_dependence_frame(results)
    betas = build_factor_beta_frame(results, prices.pct_change(fill_method=None).fillna(0.0))

    tech_row = tech[tech["strategy"] == "tech_momo"].iloc[0]
    assert tech_row["avg_tech_ai_weight"] > 0
    assert tech_row["current_tech_ai_weight"] >= 0
    assert set(betas["factor"]) >= {"QQQ", "SPY"}
    assert betas["beta"].notna().any()


def test_impairment_heatmap_and_router_have_rows() -> None:
    results = _results()
    prices = _prices()

    impairment = run_leadership_impairment(results, prices)
    underperformance = build_qqq_underperformance_frame(results, prices, peers=("SPY",))
    heatmap = build_scenario_strategy_heatmap(results, prices)
    folds, router, scores = run_walk_forward_strategy_router(
        results,
        prices,
        horizons=(21,),
        min_train_days=80,
        origin_step_days=21,
    )
    selection = build_router_selection_summary(folds)
    scenarios = build_router_scenario_summary(folds)
    comparison = build_router_model_comparison(folds)

    assert {"native", "tech_returns_haircut_25pct"}.issubset(set(impairment["scenario"]))
    assert not underperformance.empty
    assert not heatmap.empty
    assert not folds.empty
    assert not router.empty
    assert not scores.empty
    assert not selection.empty
    assert not scenarios.empty
    assert not comparison.empty
    assert "state_router_shrinkage_blend" in set(comparison["model"])
    assert {"selection_rate", "mean_excess_vs_benchmark"}.issubset(selection.columns)
    assert {"scenario_bucket", "selected_mean_excess_vs_benchmark"}.issubset(scenarios.columns)
    assert {"shrinkage_blend_mean_excess_vs_benchmark", "prior_best_mean_excess_vs_benchmark"}.issubset(
        router.columns
    )
    assert {"score_source", "shrinkage_weight", "prior_best_baseline"}.issubset(scores.columns)
    assert router.iloc[0]["folds"] > 0


def test_run_leadership_diagnostics_writes_artifacts(tmp_path: Path) -> None:
    prices = _prices()
    config = BotConfig(
        data=DataConfig(start="2020-01-01", cache_dir=str(tmp_path)),
        execution=ExecutionConfig(initial_capital=1000.0, rebalance="W-WED"),
        primary_strategy="tech_momo",
        universe={"test": ["QQQ", "SMH", "SPY", "VEA", "BIL"]},
        strategies={
            "tech_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["QQQ", "SMH"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=0.0,
            ),
            "global_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "VEA"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=0.0,
            ),
        },
    )

    run = run_leadership_diagnostics(
        config=config,
        prices=prices,
        output_dir=tmp_path / "leadership",
        experiment_root=tmp_path / "missing_experiments",
        top_n=1,
        router_horizons=(21,),
        min_train_days=80,
    )

    assert run.artifacts["summary"].exists()
    assert (tmp_path / "leadership" / "strategy_tech_dependence.csv").exists()
    assert (tmp_path / "leadership" / "walk_forward_router_selection.csv").exists()
    assert (tmp_path / "leadership" / "walk_forward_router_scenarios.csv").exists()
    assert (tmp_path / "leadership" / "walk_forward_router_comparison.csv").exists()
    assert (tmp_path / "leadership" / "walk_forward_router_scores.csv").exists()
    assert set(run.selected_strategies) == {"tech_momo", "global_momo"}


def _results() -> dict[str, BacktestResult]:
    prices = _prices()
    execution = ExecutionConfig(initial_capital=1000.0, rebalance="W-WED")
    strategies = {
        "tech_momo": StrategyConfig(
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
    }
    results = {}
    for name, strategy in strategies.items():
        columns = [*strategy.tickers, str(strategy.defensive_ticker)]
        strategy_prices = prices[columns]
        weights = build_strategy_weights(strategy_prices, strategy)
        results[name] = run_backtest(name, strategy_prices, weights, execution)
    return results


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=260)
    frame = pd.DataFrame(index=index)
    base = pd.Series(range(len(index)), index=index, dtype=float)
    frame["QQQ"] = 100.0 * (1.0 + 0.0010 + (base % 13) / 20000).cumprod()
    frame["SMH"] = 100.0 * (1.0 + 0.0012 + (base % 17) / 25000).cumprod()
    frame["SPY"] = 100.0 * (1.0 + 0.0006 + ((base + 3) % 11) / 25000).cumprod()
    frame["VEA"] = 100.0 * (1.0 + 0.0005 + ((base + 5) % 19) / 30000).cumprod()
    frame["RSP"] = 100.0 * (1.0 + 0.00055 + ((base + 7) % 23) / 35000).cumprod()
    frame["IWM"] = 100.0 * (1.0 + 0.00045 + ((base + 2) % 29) / 40000).cumprod()
    frame["GLD"] = 100.0 * (1.0 + 0.0002 + ((base + 1) % 7) / 50000).cumprod()
    frame["TLT"] = 100.0 * (1.0 + 0.0001 + ((base + 6) % 5) / 50000).cumprod()
    frame["BIL"] = 100.0 * pd.Series(1.0 + 0.00005, index=index).cumprod()
    return frame
