from __future__ import annotations

import pandas as pd

from trade_bot.research.curation import (
    PRUNED_STATUS,
    add_research_status,
    rank_strategy_candidates,
    select_curated_strategy_shelf,
)


def test_curated_shelf_preserves_score_order_when_one_family_dominates() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": f"candidate_{index:02d}",
                "family": "same_family",
                "phase": "operating_system",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 1.0 - index / 100.0,
                "robustness_score": 0.8,
                "calmar": 1.0 - index / 100.0,
            }
            for index in range(30)
        ]
    )

    curated = select_curated_strategy_shelf(rank_strategy_candidates(rows), limit=25)

    assert curated["strategy"].tolist() == [f"candidate_{index:02d}" for index in range(25)]
    assert curated["curation_rank"].tolist() == list(range(1, 26))
    assert "curation_reason" in curated.columns


def test_curated_shelf_adds_family_champions_beyond_raw_top_scores() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": f"core_{index}",
                "family": "core_cross_asset",
                "phase": "operating_system",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 1.0 - index / 100.0,
                "robustness_score": 0.8,
                "calmar": 1.0,
            }
            for index in range(10)
        ]
        + [
            {
                "strategy": "credit_gate_candidate",
                "family": "credit_gate",
                "phase": "final_deep_dive",
                "role": "final_candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.70,
                "robustness_score": 0.75,
                "calmar": 0.8,
            },
            {
                "strategy": "active_ai_candidate",
                "family": "active_ai_beta",
                "phase": "active_trading",
                "role": "active_candidate",
                "promotion_decision": "evolve_next_iteration",
                "promotion_score": 0.69,
                "robustness_score": 0.74,
                "calmar": 0.75,
            },
        ]
    )

    curated = select_curated_strategy_shelf(rank_strategy_candidates(rows), limit=7)

    assert "credit_gate_candidate" in set(curated["strategy"])
    assert "active_ai_candidate" in set(curated["strategy"])
    assert set(curated["curation_bucket"]) >= {"score_anchor", "family_champion"}


def test_research_status_prunes_low_growth_and_failed_risk_profiles() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": "low_growth_ml_probe",
                "phase": "sklearn_future_state",
                "family": "future_state_ml",
                "promotion_decision": "promote_candidate",
                "cagr": 0.035,
                "calmar": 0.35,
                "max_drawdown": -0.08,
            },
            {
                "strategy": "high_growth_candidate",
                "phase": "high_cagr_ml_guardrail",
                "family": "high_cagr_ai_escape",
                "promotion_decision": "promote_candidate",
                "cagr": 0.145,
                "calmar": 0.70,
                "max_drawdown": -0.21,
                "operability_label": "paper_operable",
            },
        ]
    )

    classified = add_research_status(rows).set_index("strategy")

    assert classified.loc["low_growth_ml_probe", "research_status"] == PRUNED_STATUS
    assert classified.loc["low_growth_ml_probe", "prune_reason"] == "low_cagr_below_5pct"
    assert classified.loc["high_growth_candidate", "research_status"] == "operational_candidate"


def test_curated_shelf_excludes_pruned_dead_ends_by_default() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": "dead_end",
                "family": "ml_probe",
                "phase": "sklearn_future_state",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 1.0,
                "cagr": 0.03,
                "calmar": 0.25,
                "max_drawdown": -0.05,
            },
            {
                "strategy": "keeper",
                "family": "ai_escape",
                "phase": "high_cagr_ml_guardrail",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.80,
                "cagr": 0.14,
                "calmar": 0.70,
                "max_drawdown": -0.21,
            },
        ]
    )

    curated = select_curated_strategy_shelf(rank_strategy_candidates(rows), limit=5)

    assert "keeper" in set(curated["strategy"])
    assert "dead_end" not in set(curated["strategy"])


def test_curated_shelf_keeps_only_core_reference_anchors_by_default() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": "keeper",
                "family": "ai_escape",
                "phase": "operating_system",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.80,
                "cagr": 0.14,
                "calmar": 0.70,
                "max_drawdown": -0.21,
            },
            {
                "strategy": "i41_ref_us_60_40",
                "family": "reference_portfolio",
                "phase": "reference",
                "role": "reference_portfolio",
                "promotion_decision": "reject_or_hold_for_reference",
                "promotion_score": 0.40,
                "cagr": 0.07,
                "calmar": 0.35,
                "max_drawdown": -0.25,
            },
            {
                "strategy": "i41_ref_all_weather",
                "family": "reference_portfolio",
                "phase": "reference",
                "role": "reference_portfolio",
                "promotion_decision": "reject_or_hold_for_reference",
                "promotion_score": 0.50,
                "cagr": 0.06,
                "calmar": 0.33,
                "max_drawdown": -0.18,
            },
        ]
    )

    curated = select_curated_strategy_shelf(rank_strategy_candidates(rows), limit=10)

    assert "i41_ref_us_60_40" in set(curated["strategy"])
    assert "i41_ref_all_weather" not in set(curated["strategy"])


def test_growth_utility_can_rank_higher_cagr_tolerable_drawdown_above_safer_lower_growth() -> None:
    rows = pd.DataFrame(
        [
            {
                "strategy": "lower_growth_lower_drawdown",
                "family": "growth_frontier",
                "phase": "growth_frontier",
                "role": "growth_frontier_candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.85,
                "robustness_score": 0.85,
                "calmar": 0.73,
                "cagr": 0.1117,
                "max_drawdown": -0.1534,
                "walk_forward_positive_rate": 0.85,
                "worst_3y_cagr": 0.02,
                "left_tail_regime_return": -0.08,
                "operability_label": "weekly_cadence",
            },
            {
                "strategy": "higher_growth_tolerable_drawdown",
                "family": "growth_frontier",
                "phase": "growth_frontier",
                "role": "growth_frontier_candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.85,
                "robustness_score": 0.85,
                "calmar": 0.74,
                "cagr": 0.1487,
                "max_drawdown": -0.2001,
                "walk_forward_positive_rate": 0.85,
                "worst_3y_cagr": 0.02,
                "left_tail_regime_return": -0.08,
                "operability_label": "weekly_cadence",
            },
        ]
    )

    ranked = rank_strategy_candidates(rows)

    assert ranked.iloc[0]["strategy"] == "higher_growth_tolerable_drawdown"
    assert ranked.iloc[0]["growth_constrained_utility_score"] > ranked.iloc[1]["growth_constrained_utility_score"]
