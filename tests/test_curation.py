from __future__ import annotations

import pandas as pd

from trade_bot.research.curation import rank_strategy_candidates, select_curated_strategy_shelf


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
