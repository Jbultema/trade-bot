from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_bot.portfolio.risk import (
    PortfolioRiskConfig,
    _max_constraint_row,
    build_portfolio_risk,
)


def test_zero_scenario_authority_preserves_independent_base_limits() -> None:
    config = PortfolioRiskConfig(
        scenario_budget_authority=0.0,
        scenario_weighted_stress_authority=0.0,
    )
    target_weights = pd.Series({"BIL": 0.80, "QQQ": 0.20})

    risk = build_portfolio_risk(
        _risk_prices(),
        target_weights,
        _scenario_lattice(),
        current_weights=target_weights,
        config=config,
    )

    budget = risk.scenario_risk_budget.iloc[0]
    summary = risk.summary.iloc[0]
    assert budget["max_equity_beta"] == pytest.approx(config.base_max_equity_beta)
    assert budget["max_expected_shortfall_95"] == pytest.approx(
        config.base_max_expected_shortfall_95
    )
    assert budget["max_stress_loss"] == pytest.approx(config.base_max_stress_loss)
    assert budget["min_defensive_weight"] == pytest.approx(config.base_min_defensive_weight)
    assert summary["scenario_budget_authority"] == 0.0
    assert summary["scenario_weighted_stress_authority"] == 0.0
    assert "scenario_weighted_stress" not in summary["applied_constraints"]


def test_portfolio_risk_engine_builds_constraints_and_risk_adjusted_weights() -> None:
    prices = _risk_prices()
    target_weights = pd.Series({"QQQ": 0.80, "SPY": 0.20})

    risk = build_portfolio_risk(
        prices,
        target_weights,
        _scenario_lattice(),
        current_weights=target_weights,
    )

    summary = risk.summary.iloc[0]
    constraints = set(risk.constraint_report["constraint"])

    assert summary["portfolio_risk_level"] in {"risk_reduced", "constraint_breach"}
    assert risk.risk_adjusted_weights.get("BIL", 0.0) > 0.0
    assert risk.risk_adjusted_weights.get("QQQ", 0.0) < target_weights["QQQ"]
    assert "max_stress_loss" in constraints
    assert "equity_beta" in constraints
    assert not risk.factor_exposures.empty
    assert not risk.beta_decomposition.empty
    assert not risk.tail_risk.empty
    assert not risk.stress_tests.empty
    assert not risk.marginal_risk_contribution.empty
    assert not risk.scenario_risk_budget.empty
    assert risk.sizing_adjustments["ticker"].str.contains("BIL").any()


def test_defensive_residual_is_not_reported_as_single_risk_asset_breach() -> None:
    target_weights = pd.Series({"BIL": 0.80, "QQQ": 0.20})

    risk = build_portfolio_risk(
        _risk_prices(),
        target_weights,
        pd.DataFrame(),
        current_weights=target_weights,
    )

    max_asset = risk.constraint_report.set_index("constraint").loc["max_single_asset"]
    assert max_asset["post_value"] == pytest.approx(0.20)
    assert max_asset["status"] != "breach"


def test_constraint_comparison_ignores_machine_precision_boundary_noise() -> None:
    row = _max_constraint_row(
        "scenario_weighted_stress_loss",
        0.10,
        0.10 + 1e-17,
        0.10,
        (),
    )

    assert row["status"] == "ok"


def _risk_prices() -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=320)
    base_returns = pd.Series(
        0.0004 + np.sin(np.arange(len(index)) / 13.0) * 0.003,
        index=index,
    )
    spy = 100.0 * (1.0 + base_returns).cumprod()
    qqq = 100.0 * (1.0 + base_returns * 1.35 + 0.0002).cumprod()
    smh = 100.0 * (1.0 + base_returns * 1.65 + 0.0003).cumprod()
    rsp = 100.0 * (1.0 + base_returns * 0.85).cumprod()
    tlt = 100.0 * (1.0 - base_returns * 0.35).cumprod()
    hyg = 100.0 * (1.0 + base_returns * 0.65).cumprod()
    gld = 100.0 * (1.0 - base_returns * 0.15 + 0.0001).cumprod()
    bil = 100.0 * (1.0 + pd.Series(0.0001, index=index)).cumprod()
    return pd.DataFrame(
        {
            "SPY": spy,
            "QQQ": qqq,
            "SMH": smh,
            "RSP": rsp,
            "TLT": tlt,
            "HYG": hyg,
            "GLD": gld,
            "BIL": bil,
        },
        index=index,
    )


def _scenario_lattice() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": "1m",
                "scenario": "Credit-led risk-off",
                "scenario_id": "credit_risk_off",
                "family": "credit",
                "probability": 0.45,
                "risk_bucket": "risk_off",
                "expected_bot_posture": "Reduce risk.",
                "preferred_exposure": "BIL",
                "avoid_exposure": "High beta",
            },
            {
                "horizon": "1m",
                "scenario": "AI capex unwind",
                "scenario_id": "ai_capex_unwind",
                "family": "ai_capex",
                "probability": 0.20,
                "risk_bucket": "risk_on_fragile",
                "expected_bot_posture": "Cut AI beta if confirmation deteriorates.",
                "preferred_exposure": "Quality",
                "avoid_exposure": "AI beta",
            },
            {
                "horizon": "1m",
                "scenario": "Rates transition",
                "scenario_id": "rates_transition",
                "family": "inflation_rates",
                "probability": 0.25,
                "risk_bucket": "transition",
                "expected_bot_posture": "Keep risk smaller.",
                "preferred_exposure": "Defensive",
                "avoid_exposure": "Duration and high beta",
            },
            {
                "horizon": "1m",
                "scenario": "Broad risk-on",
                "scenario_id": "risk_on",
                "family": "market_trend",
                "probability": 0.10,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Hold risk.",
                "preferred_exposure": "SPY",
                "avoid_exposure": "None",
            },
        ]
    )
