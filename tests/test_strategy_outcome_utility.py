from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.strategy_outcome_utility import (
    drawdown_hard_penalty,
    drawdown_recovery_return,
    drawdown_soft_penalty,
    enrich_strategy_outcome_utility,
    terminal_wealth_from_cagr,
)


def test_terminal_wealth_includes_end_of_year_contributions() -> None:
    wealth = terminal_wealth_from_cagr(
        0.10,
        years=2,
        starting_account_value=100.0,
        annual_contribution=10.0,
    )

    assert float(wealth.iloc[0]) == pytest.approx(142.0)


def test_drawdown_recovery_math() -> None:
    recovery = drawdown_recovery_return(pd.Series([-0.20]))

    assert float(recovery.iloc[0]) == pytest.approx(0.25)


def test_soft_and_hard_drawdown_penalties_follow_growth_policy() -> None:
    drawdowns = pd.Series([-0.15, -0.22, -0.26, -0.30])

    soft = drawdown_soft_penalty(drawdowns)
    hard = drawdown_hard_penalty(drawdowns)

    assert soft.iloc[0] == pytest.approx(0.0)
    assert soft.iloc[1] == pytest.approx(0.0)
    assert 0.0 < soft.iloc[2] < 1.0
    assert soft.iloc[3] == pytest.approx(1.0)
    assert hard.iloc[:3].sum() == pytest.approx(0.0)
    assert hard.iloc[3] == pytest.approx(1.0)


def test_growth_constrained_utility_prefers_high_cagr_inside_drawdown_band() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["safer_lower_growth", "higher_growth_tolerable_drawdown"],
            "cagr": [0.1117, 0.1487],
            "max_drawdown": [-0.1534, -0.2001],
            "walk_forward_positive_rate": [0.85, 0.85],
            "worst_3y_cagr": [0.02, 0.02],
            "left_tail_regime_return": [-0.08, -0.08],
            "operability_label": ["weekly_cadence", "weekly_cadence"],
        }
    )

    enriched = enrich_strategy_outcome_utility(frame).set_index("strategy")

    assert (
        enriched.loc["higher_growth_tolerable_drawdown", "growth_constrained_utility_score"]
        > enriched.loc["safer_lower_growth", "growth_constrained_utility_score"]
    )
    assert enriched.loc["higher_growth_tolerable_drawdown", "growth_utility_tier"] in {
        "growth_champion_candidate",
        "growth_challenger_candidate",
    }


def test_growth_utility_rejects_hard_drawdown_band() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["too_deep"],
            "cagr": [0.18],
            "max_drawdown": [-0.30],
            "walk_forward_positive_rate": [0.90],
            "worst_3y_cagr": [0.03],
            "left_tail_regime_return": [-0.05],
            "operability_label": ["weekly_cadence"],
        }
    )

    enriched = enrich_strategy_outcome_utility(frame).iloc[0]

    assert enriched["drawdown_hard_penalty"] == pytest.approx(1.0)
    assert enriched["growth_utility_tier"] == "growth_reject_hard_drawdown"
