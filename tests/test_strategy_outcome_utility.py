from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.strategy_outcome_utility import (
    OutcomeBootstrapConfig,
    bootstrap_outcome_paths,
    drawdown_hard_penalty,
    drawdown_recovery_return,
    drawdown_soft_penalty,
    enrich_strategy_outcome_utility,
    summarize_bootstrap_outcomes,
    terminal_wealth_from_cagr,
)


def test_terminal_wealth_includes_end_of_year_contributions() -> None:
    wealth = terminal_wealth_from_cagr(
        0.10,
        years=2,
        starting_account_value=100.0,
        annual_contribution=10.0,
        contribution_timing="end_of_year",
    )

    assert float(wealth.iloc[0]) == pytest.approx(142.0)


def test_terminal_wealth_splits_default_contributions_monthly() -> None:
    wealth = terminal_wealth_from_cagr(
        0.12,
        years=1,
        starting_account_value=100.0,
        annual_contribution=12.0,
    )
    monthly_rate = 1.12 ** (1.0 / 12.0) - 1.0
    expected = 100.0 * 1.12 + ((1.0 + monthly_rate) ** 12.0 - 1.0) / monthly_rate

    assert float(wealth.iloc[0]) == pytest.approx(expected)


def test_bootstrap_outcome_paths_split_default_contributions_monthly() -> None:
    monthly_rate = 1.12 ** (1.0 / 12.0) - 1.0
    paths = bootstrap_outcome_paths(
        pd.Series([monthly_rate] * 12),
        config=OutcomeBootstrapConfig(
            horizon_years=1,
            starting_account_value=100.0,
            annual_contribution=12.0,
            trading_days_per_year=12,
            paths=3,
            block_days=1,
            random_seed=7,
        ),
    )
    expected = 100.0 * 1.12 + ((1.0 + monthly_rate) ** 12.0 - 1.0) / monthly_rate

    assert paths["terminal_wealth"].tolist() == pytest.approx([expected, expected, expected])


def test_bootstrap_outcome_paths_can_use_end_of_year_contributions() -> None:
    paths = bootstrap_outcome_paths(
        pd.Series([0.0, 0.0, 0.0, 0.0]),
        config=OutcomeBootstrapConfig(
            horizon_years=2,
            starting_account_value=100.0,
            annual_contribution=10.0,
            contribution_timing="end_of_year",
            trading_days_per_year=2,
            paths=3,
            block_days=1,
            random_seed=7,
        ),
    )
    summary = summarize_bootstrap_outcomes(paths)

    assert paths["terminal_wealth"].tolist() == pytest.approx([120.0, 120.0, 120.0])
    assert paths["max_drawdown"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert paths["ulcer_index"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert summary["paths"] == 3
    assert summary["terminal_wealth_p50"] == pytest.approx(120.0)


def test_bootstrap_outcome_paths_capture_sequence_drawdown() -> None:
    paths = bootstrap_outcome_paths(
        pd.Series([-0.10, 0.02, 0.03, 0.01]),
        config=OutcomeBootstrapConfig(
            horizon_years=1,
            starting_account_value=100.0,
            annual_contribution=0.0,
            trading_days_per_year=4,
            paths=20,
            block_days=2,
            random_seed=11,
        ),
    )

    assert paths["terminal_wealth"].notna().all()
    assert paths["max_drawdown"].min() < 0.0
    assert paths["ulcer_index"].max() > 0.0


def test_bootstrap_drawdown_is_not_masked_by_external_contributions() -> None:
    returns = pd.Series([-0.10, 0.02, 0.03, 0.01])
    base = OutcomeBootstrapConfig(
        horizon_years=1,
        starting_account_value=100.0,
        annual_contribution=0.0,
        trading_days_per_year=4,
        paths=20,
        block_days=2,
        random_seed=11,
    )
    funded = OutcomeBootstrapConfig(**{**base.__dict__, "annual_contribution": 1_000.0})

    without_contributions = bootstrap_outcome_paths(returns, config=base)
    with_contributions = bootstrap_outcome_paths(returns, config=funded)

    assert with_contributions["terminal_wealth"].gt(
        without_contributions["terminal_wealth"]
    ).all()
    assert with_contributions["max_drawdown"].tolist() == pytest.approx(
        without_contributions["max_drawdown"].tolist()
    )
    assert with_contributions["ulcer_index"].tolist() == pytest.approx(
        without_contributions["ulcer_index"].tolist()
    )


def test_bootstrap_summary_reports_catastrophic_tail_and_target_attainment() -> None:
    paths = pd.DataFrame(
        {
            "terminal_wealth": [80.0, 120.0, 150.0, 200.0],
            "max_drawdown": [-0.35, -0.25, -0.15, -0.05],
            "ulcer_index": [0.20, 0.15, 0.08, 0.03],
        }
    )

    summary = summarize_bootstrap_outcomes(paths, target_terminal_wealth=140.0)

    assert summary["drawdown_over_10_probability"] == pytest.approx(0.75)
    assert summary["drawdown_over_20_probability"] == pytest.approx(0.50)
    assert summary["drawdown_over_30_probability"] == pytest.approx(0.25)
    assert summary["expected_drawdown_if_over_20"] == pytest.approx(-0.30)
    assert summary["target_wealth_success_probability"] == pytest.approx(0.50)


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
