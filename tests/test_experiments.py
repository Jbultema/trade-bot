from __future__ import annotations

import json

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.research.experiments import (
    ScenarioSizingConfig,
    apply_scenario_position_sizing,
    build_experiment_scorecard,
    generate_iteration_candidates,
)


def test_iteration_one_has_bounded_candidate_batch() -> None:
    candidates = generate_iteration_candidates(1)

    assert 3 <= len(candidates) <= 10
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_broad_iterations_cover_multiple_research_families() -> None:
    candidates = generate_iteration_candidates(2)

    assert 3 <= len(candidates) <= 10
    assert {"risk_adjusted_momentum", "off_ramp", "ai_infrastructure"}.issubset(
        {candidate.family for candidate in candidates}
    )


def test_later_iterations_evolve_previous_candidate_configs() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.0,
    )
    previous_scorecards = pd.DataFrame(
        {
            "iteration": [3],
            "strategy": ["parent_strategy"],
            "family": ["core_cross_asset"],
            "role": ["candidate_core"],
            "promotion_decision": ["evolve_next_iteration"],
            "promotion_score": [0.7],
        }
    )
    previous_candidates = pd.DataFrame(
        {
            "strategy": ["parent_strategy"],
            "strategy_json": [json.dumps(strategy.model_dump(mode="json"))],
        }
    )

    candidates = generate_iteration_candidates(
        4,
        previous_scorecards=previous_scorecards,
        previous_candidates=previous_candidates,
    )

    assert 3 <= len(candidates) <= 10
    assert {candidate.parent for candidate in candidates} == {"parent_strategy"}
    assert all(candidate.phase == "deep" for candidate in candidates)


def test_experiment_scorecard_marks_left_tail_rejects() -> None:
    candidates = generate_iteration_candidates(1)
    names = [candidate.name for candidate in candidates]
    metrics = pd.DataFrame(
        {
            "cagr": [0.10 for _ in names],
            "sharpe": [0.8 for _ in names],
            "sortino": [1.1 for _ in names],
            "max_drawdown": [-0.20 for _ in names],
            "calmar": [0.5 for _ in names],
            "average_turnover": [0.05 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    metrics.loc[names[0], "max_drawdown"] = -0.50
    window_summary = pd.DataFrame(
        {
            "worst_cagr": [-0.01 for _ in names for _ in range(3)],
            "positive_window_rate": [0.8 for _ in names for _ in range(3)],
        },
        index=pd.MultiIndex.from_product(
            [names, ["1y", "3y", "5y"]],
            names=["strategy", "window"],
        ),
    )

    benchmark_metrics = pd.DataFrame(
        {
            "cagr": [0.08, 0.12],
            "max_drawdown": [-0.30, -0.40],
            "calmar": [0.27, 0.30],
        },
        index=["benchmark_spy", "benchmark_qqq"],
    )

    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        benchmark_metrics=benchmark_metrics,
    )

    assert scorecard.loc[names[0], "promotion_decision"] == "reject_left_tail"
    assert "promotion_score" in scorecard.columns
    assert "family" in scorecard.columns
    assert round(float(scorecard.loc[names[1], "excess_cagr_vs_spy"]), 6) == 0.02


def test_scorecard_includes_walk_forward_and_regime_robustness() -> None:
    candidates = generate_iteration_candidates(1)
    names = [candidate.name for candidate in candidates]
    metrics = pd.DataFrame(
        {
            "cagr": [0.10 for _ in names],
            "sharpe": [0.8 for _ in names],
            "sortino": [1.1 for _ in names],
            "max_drawdown": [-0.20 for _ in names],
            "calmar": [0.5 for _ in names],
            "average_turnover": [0.05 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    window_summary = pd.DataFrame(
        {
            "worst_cagr": [-0.01 for _ in names for _ in range(3)],
            "positive_window_rate": [0.8 for _ in names for _ in range(3)],
        },
        index=pd.MultiIndex.from_product(
            [names, ["1y", "3y", "5y"]],
            names=["strategy", "window"],
        ),
    )
    regime_summary = pd.DataFrame(
        {
            "worst_regime_cagr": [-0.08 for _ in names],
            "worst_regime_return": [-0.05 for _ in names],
            "left_tail_regime_cagr": [-0.06 for _ in names],
            "left_tail_regime_return": [-0.04 for _ in names],
            "transition_regime_hit_rate": [0.6 for _ in names],
            "regime_positive_rate": [0.7 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    walk_forward_summary = pd.DataFrame(
        {
            "walk_forward_median_cagr": [0.09 for _ in names],
            "walk_forward_worst_cagr": [-0.04 for _ in names],
            "walk_forward_positive_rate": [0.7 for _ in names],
            "walk_forward_median_calmar": [0.45 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )

    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        regime_summary=regime_summary,
        walk_forward_summary=walk_forward_summary,
    )

    assert "robustness_score" in scorecard.columns
    assert "walk_forward_positive_rate" in scorecard.columns
    assert "left_tail_regime_return" in scorecard.columns
    assert "left_tail_regime_cagr" in scorecard.columns


def test_scenario_position_sizing_moves_stress_residual_to_defensive_ticker() -> None:
    index = pd.bdate_range("2026-01-01", periods=90)
    prices = pd.DataFrame(
        {
            "SPY": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "QQQ": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "RSP": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "HYG": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "LQD": pd.Series(range(100, 190), index=index, dtype=float),
            "VIXY": pd.Series(range(50, 140), index=index, dtype=float),
            "UUP": pd.Series(range(50, 140), index=index, dtype=float),
            "BIL": pd.Series(100.0, index=index),
        }
    )
    target_weights = pd.DataFrame(
        {"SPY": 1.0, "BIL": 0.0},
        index=index,
    )

    adjusted = apply_scenario_position_sizing(
        target_weights,
        prices,
        ScenarioSizingConfig(profile="test", stress_multiplier=0.2, min_multiplier=0.1),
        defensive_ticker="BIL",
    )

    assert adjusted["SPY"].iloc[-1] < 0.8
    assert adjusted["BIL"].iloc[-1] > 0.2
    assert round(float(adjusted.iloc[-1].sum()), 8) == 1.0
