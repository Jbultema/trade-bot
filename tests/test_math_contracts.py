from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.portfolio.risk import (
    PortfolioRiskConfig,
    _build_scenario_budget,
    _marginal_risk_contribution,
    _tail_stats,
)
from trade_bot.research.event_risk import MarketEvent
from trade_bot.research.trade_decision import _event_context, _scenario_context


def test_performance_metrics_lock_core_formulas() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=index)
    equity = 100.0 * (1.0 + returns).cumprod()
    turnover = pd.Series([0.0, 0.10, 0.20, 0.30], index=index)
    costs = pd.Series([0.0, 0.0001, 0.0002, 0.0003], index=index)

    metrics = calculate_metrics("contract", returns, equity, turnover, costs)

    years = (index[-1] - index[0]).days / 365.25
    expected_cagr = (equity.iloc[-1] / 100.0) ** (1.0 / years) - 1.0
    expected_annual_vol = returns.std() * np.sqrt(252)
    expected_sharpe = returns.mean() * 252 / expected_annual_vol
    downside_vol = returns.clip(upper=0.0).std() * np.sqrt(252)
    expected_sortino = returns.mean() * 252 / downside_vol
    expected_max_drawdown = (equity / equity.cummax() - 1.0).min()

    assert metrics.cagr == pytest.approx(expected_cagr)
    assert metrics.annualized_volatility == pytest.approx(expected_annual_vol)
    assert metrics.sharpe == pytest.approx(expected_sharpe)
    assert metrics.sortino == pytest.approx(expected_sortino)
    assert metrics.max_drawdown == pytest.approx(expected_max_drawdown)
    assert metrics.calmar == pytest.approx(expected_cagr / abs(expected_max_drawdown))
    assert metrics.average_turnover == pytest.approx(turnover.mean())
    assert metrics.total_transaction_cost == pytest.approx(costs.sum())


def test_tail_stats_return_positive_var_and_expected_shortfall_loss_magnitudes() -> None:
    portfolio_returns = pd.Series([-0.10, -0.05, 0.01, 0.02, 0.03])

    stats = _tail_stats(portfolio_returns, 0.75)

    assert stats["value_at_risk"] == pytest.approx(0.05)
    assert stats["expected_shortfall"] == pytest.approx(0.075)
    assert stats["worst_day"] == pytest.approx(-0.10)
    assert stats["observations"] == 5.0


def test_scenario_budget_uses_one_month_risk_buckets_as_sizing_inputs() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "probability": 0.40,
                "risk_bucket": "risk_off",
                "scenario": "Credit-led risk-off",
                "family": "credit",
                "preferred_exposure": "BIL",
                "avoid_exposure": "High beta",
            },
            {
                "horizon": "1m",
                "probability": 0.20,
                "risk_bucket": "transition",
                "scenario": "Choppy rotation",
                "family": "rotation",
                "preferred_exposure": "Quality",
                "avoid_exposure": "Overtrading",
            },
            {
                "horizon": "1m",
                "probability": 0.10,
                "risk_bucket": "risk_on_fragile",
                "scenario": "Narrow AI melt-up",
                "family": "ai_concentration",
                "preferred_exposure": "QQQ SMH",
                "avoid_exposure": "AI capex",
            },
            {
                "horizon": "1m",
                "probability": 0.30,
                "risk_bucket": "risk_on",
                "scenario": "Broad risk-on",
                "family": "risk_on",
                "preferred_exposure": "SPY",
                "avoid_exposure": "None",
            },
        ]
    )

    budget = _build_scenario_budget(scenario_lattice, PortfolioRiskConfig())

    expected_multiplier = 1.0 - 0.55 * 0.40 - 0.20 * 0.20 - 0.15 * 0.10
    expected_min_defensive = 0.40 * 0.40 + 0.20 * 0.20 + 0.10 * 0.10 + 0.10 * 0.10

    assert budget.risk_off_probability == pytest.approx(0.40)
    assert budget.transition_probability == pytest.approx(0.20)
    assert budget.fragile_upside_probability == pytest.approx(0.10)
    assert budget.risk_on_probability == pytest.approx(0.30)
    assert budget.ai_unwind_probability == pytest.approx(0.10)
    assert budget.scenario_risk_multiplier == pytest.approx(expected_multiplier)
    assert budget.min_defensive_weight == pytest.approx(expected_min_defensive)


def test_trade_decision_context_multipliers_are_policy_rules() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "rank": 1,
                "scenario": "Risk-off",
                "probability": 0.30,
                "risk_bucket": "risk_off",
                "expected_bot_posture": "Reduce risk.",
            },
            {
                "horizon": "1m",
                "rank": 2,
                "scenario": "Transition",
                "probability": 0.25,
                "risk_bucket": "transition",
                "expected_bot_posture": "Use smaller sizing.",
            },
            {
                "horizon": "1m",
                "rank": 3,
                "scenario": "Fragile upside",
                "probability": 0.20,
                "risk_bucket": "risk_on_fragile",
                "expected_bot_posture": "Participate with cap.",
            },
            {
                "horizon": "1m",
                "rank": 4,
                "scenario": "Risk-on",
                "probability": 0.25,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Hold risk.",
            },
        ]
    )
    events = (
        MarketEvent(
            event_id="leading",
            name="Leading stress",
            date=pd.Timestamp("2026-06-17"),
            category="credit",
            direction="escalation",
            description="test",
            current=True,
            phase="leading_warning",
        ),
        MarketEvent(
            event_id="uncertain",
            name="Uncertain stress",
            date=pd.Timestamp("2026-06-17"),
            category="policy",
            direction="uncertain",
            description="test",
            current=True,
        ),
    )

    scenario_context = _scenario_context(scenario_lattice)
    event_context = _event_context(events)

    assert scenario_context["risk_multiplier"] == pytest.approx(
        1.0 - 0.55 * 0.30 - 0.20 * 0.25 - 0.15 * 0.20
    )
    assert event_context["event_pressure"] == pytest.approx(0.07 + 0.02)
    assert event_context["risk_multiplier"] == pytest.approx(0.91)


def test_marginal_risk_contribution_is_covariance_based_and_sums_to_one() -> None:
    index = pd.bdate_range("2025-01-01", periods=80)
    base = pd.Series(np.sin(np.arange(80) / 5.0) * 0.01, index=index)
    returns = pd.DataFrame(
        {
            "SPY": base,
            "QQQ": base * 1.5 + 0.001,
            "BIL": pd.Series(0.0001, index=index),
        }
    )
    weights = pd.Series({"SPY": 0.40, "QQQ": 0.40, "BIL": 0.20})

    contribution = _marginal_risk_contribution(returns, weights, PortfolioRiskConfig())

    assert not contribution.empty
    assert contribution["risk_contribution_pct"].sum() == pytest.approx(1.0)
    assert contribution.loc[contribution["ticker"] == "QQQ", "risk_contribution_pct"].iloc[0] > (
        contribution.loc[contribution["ticker"] == "SPY", "risk_contribution_pct"].iloc[0]
    )
