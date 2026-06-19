from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.research.experiment_monitor import (
    build_strategy_family_map,
    latest_experiment_iteration,
    load_decision_sanity_impacts,
    load_experiment_candidates,
    load_experiment_regime_metrics,
    load_experiment_scorecards,
    load_experiment_walk_forward,
    strategy_family_takeaways,
    summarize_decision_sanity_impacts,
    summarize_experiment_families,
    summarize_experiment_history,
    summarize_experiment_operating_systems,
    summarize_family_clusters,
    summarize_risk_behavior_matrix,
    summarize_strategy_archetypes,
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


def test_load_and_summarize_decision_sanity_impacts(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_77"
    iteration_dir.mkdir()
    pd.DataFrame(
        {
            "raw_strategy": ["raw_a", "raw_b"],
            "capped_strategy": ["cap_a", "cap_b"],
            "family": ["ai", "macro"],
            "decision_sanity": ["confirmation_cap", "confirmation_cap"],
            "delta_promotion_score": [0.04, -0.02],
            "delta_cagr": [0.01, -0.003],
            "delta_max_drawdown": [0.02, -0.01],
            "delta_calmar": [0.03, -0.01],
            "delta_average_turnover": [0.001, 0.002],
            "delta_walk_forward_positive_rate": [0.1, 0.0],
            "delta_left_tail_regime_return": [0.02, -0.01],
        }
    ).to_csv(iteration_dir / "decision_sanity_impact.csv", index=False)

    impacts = load_decision_sanity_impacts(tmp_path)
    summary = summarize_decision_sanity_impacts(impacts)

    assert impacts["iteration"].tolist() == [77, 77]
    assert summary.loc[0, "pairs"] == 2
    assert summary.loc[0, "decision_sanity"] == "confirmation_cap"
    assert summary.loc[0, "adoption_read"] in {
        "promote_for_monitoring",
        "mixed_keep_testing",
        "tune_or_reject",
    }


def test_strategy_family_map_explains_archetypes_and_risk_behaviors() -> None:
    scorecards = pd.DataFrame(
        {
            "iteration": [1, 1, 2, 3],
            "strategy": [
                "i_ai_cycle_reentry",
                "i_dip_reentry",
                "i_sector_regime_low_churn",
                "buy_hold_spy",
            ],
            "phase": ["ai_risk_cycle", "dip_reentry", "sector_regime_rotation", "baseline"],
            "family": [
                "cycle_aggressive_ai_escape",
                "dip_reentry",
                "sector_regime_low_churn",
                "reference",
            ],
            "role": ["risk_cycle_candidate", "candidate", "sector_regime_candidate", "baseline"],
            "promotion_decision": [
                "promote_candidate",
                "evolve_next_iteration",
                "promote_candidate",
                "reference",
            ],
            "promotion_score": [0.91, 0.72, 0.84, 0.4],
            "cagr": [0.16, 0.13, 0.11, 0.1],
            "max_drawdown": [-0.18, -0.22, -0.16, -0.34],
            "calmar": [0.88, 0.59, 0.69, 0.29],
            "average_turnover": [0.12, 0.08, 0.05, 0.0],
            "walk_forward_positive_rate": [0.8, 0.7, 0.75, 0.6],
            "left_tail_regime_return": [-0.08, -0.11, -0.06, -0.25],
        }
    )
    candidates = pd.DataFrame(
        {
            "iteration": [1, 1, 2, 3],
            "strategy": scorecards["strategy"],
            "hypothesis": [
                "AI escape reentry adds risk after drawdowns when credit repairs.",
                "Buy the dip only after washout and recovery confirmation.",
                "Sector regime rotation with low churn across SPDR sectors.",
                "Reference buy and hold benchmark.",
            ],
            "strategy_json": [
                json.dumps(
                    {
                        "type": "ai_risk_cycle_overlay",
                        "tickers": ["SPY", "RSP", "QQQ", "SMH", "SOXX", "BIL"],
                        "satellite_tickers": ["QQQ", "SMH", "SOXX", "NVDA"],
                        "defensive_ticker": "BIL",
                    }
                ),
                json.dumps(
                    {
                        "type": "dual_momentum",
                        "tickers": ["SPY", "QQQ", "IWM", "BIL"],
                        "defensive_ticker": "BIL",
                    }
                ),
                json.dumps(
                    {
                        "type": "sector_regime_rotation",
                        "tickers": [
                            "XLK",
                            "XLF",
                            "XLE",
                            "XLU",
                            "BIL",
                            "SGOV",
                            "IEF",
                            "LQD",
                            "GLD",
                        ],
                        "defensive_ticker": "BIL",
                    }
                ),
                json.dumps({"type": "buy_hold", "tickers": ["SPY"]}),
            ],
        }
    )

    family_map = build_strategy_family_map(scorecards, candidates)
    ai_row = family_map[family_map["strategy"] == "i_ai_cycle_reentry"].iloc[0]
    dip_row = family_map[family_map["strategy"] == "i_dip_reentry"].iloc[0]
    sector_row = family_map[family_map["strategy"] == "i_sector_regime_low_churn"].iloc[0]
    baseline_row = family_map[family_map["strategy"] == "buy_hold_spy"].iloc[0]

    assert ai_row["strategy_archetype"] == "AI risk-cycle reentry"
    assert ai_row["risk_behavior"] == "Dip-reentry"
    assert dip_row["strategy_archetype"] == "Dip reentry / buy-the-dip"
    assert dip_row["defensive_expression"] == "T-bills/cash"
    assert sector_row["strategy_archetype"] == "Sector and factor rotation"
    assert sector_row["risk_behavior"] == "Sector-regime gating"
    assert sector_row["defensive_expression"] == "Multi-asset defense"
    assert baseline_row["strategy_archetype"] == "Static baseline / reference"

    archetypes = summarize_strategy_archetypes(family_map)
    risk_matrix = summarize_risk_behavior_matrix(family_map)
    clusters = summarize_family_clusters(family_map)
    takeaways = strategy_family_takeaways(family_map)

    assert "AI risk-cycle reentry" in set(archetypes["strategy_archetype"])
    assert "Sector-regime gating" in set(risk_matrix["risk_behavior"])
    assert not clusters.empty
    assert any("Highest promotion-score family" in takeaway for takeaway in takeaways)
