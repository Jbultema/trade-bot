from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.research.experiment_monitor import (
    latest_experiment_iteration,
    load_experiment_candidates,
    load_experiment_regime_metrics,
    load_experiment_scorecards,
    load_experiment_walk_forward,
    summarize_experiment_families,
    summarize_experiment_history,
    summarize_experiment_operating_systems,
)


def test_load_experiment_scorecards_reads_iteration_directories(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_01"
    iteration_dir.mkdir()
    pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "role": ["candidate_core", "satellite"],
            "promotion_decision": ["promote_candidate", "reject_left_tail"],
            "promotion_score": [0.9, 0.1],
            "calmar": [0.8, 0.2],
            "cagr": [0.2, 0.1],
            "max_drawdown": [-0.2, -0.5],
        }
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    scorecards = load_experiment_scorecards(tmp_path)
    summary = summarize_experiment_history(scorecards)
    family_summary = summarize_experiment_families(scorecards)

    assert scorecards["iteration"].tolist() == [1, 1]
    assert scorecards.loc[0, "iteration_rank"] == 1
    assert set(summary["promotion_decision"]) == {"promote_candidate", "reject_left_tail"}
    assert family_summary.loc[0, "promoted"] == 1
    assert latest_experiment_iteration(scorecards) == 1


def test_load_experiment_monitor_robustness_artifacts(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_21"
    iteration_dir.mkdir()
    pd.DataFrame(
        {
            "strategy": ["candidate_a"],
            "phase": ["operating_system"],
            "family": ["core_cross_asset"],
            "role": ["operating_system"],
            "scenario_sizing": ["balanced"],
            "promotion_decision": ["promote_candidate"],
            "promotion_score": [0.9],
            "robustness_score": [0.8],
            "cagr": [0.12],
            "max_drawdown": [-0.18],
            "calmar": [0.67],
            "walk_forward_positive_rate": [0.7],
            "left_tail_regime_cagr": [-0.03],
            "hypothesis": ["test"],
        }
    ).to_csv(iteration_dir / "scorecard.csv", index=False)
    pd.DataFrame(
        {
            "name": ["candidate_a"],
            "regime": ["covid_crash"],
            "regime_type": ["left_tail"],
            "total_return": [-0.02],
            "cagr": [-0.12],
        }
    ).to_csv(iteration_dir / "regime_metrics.csv", index=False)
    pd.DataFrame(
        {
            "name": ["candidate_a"],
            "walk_forward_positive_rate": [0.7],
            "walk_forward_worst_cagr": [-0.05],
        }
    ).to_csv(iteration_dir / "walk_forward_summary.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["candidate_a"],
            "scenario_sizing": ["balanced"],
            "strategy_json": ["{}"],
        }
    ).to_csv(iteration_dir / "candidates.csv", index=False)

    scorecards = load_experiment_scorecards(tmp_path)
    regimes = load_experiment_regime_metrics(tmp_path)
    walk_forward = load_experiment_walk_forward(tmp_path)
    candidates = load_experiment_candidates(tmp_path)
    operating_systems = summarize_experiment_operating_systems(scorecards)

    assert regimes.loc[0, "strategy"] == "candidate_a"
    assert walk_forward.loc[0, "strategy"] == "candidate_a"
    assert candidates.loc[0, "scenario_sizing"] == "balanced"
    assert operating_systems.loc[0, "strategy"] == "candidate_a"
