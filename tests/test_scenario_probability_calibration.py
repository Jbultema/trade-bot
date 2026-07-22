from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.defensive_judgement import DefensiveJudgementHorizon
from trade_bot.research.scenario_probability_calibration import (
    build_calibration_metrics,
    build_scenario_outcomes,
    calibration_authority,
    effective_scenario_multiplier,
)


def test_scenario_outcomes_use_existing_defensive_judgement_contract() -> None:
    dates = pd.bdate_range("2026-01-02", periods=8)
    states = pd.DataFrame(
        {
            "origin_date": [dates[0], dates[1]],
            "risk_off_probability": [0.8, 0.2],
        }
    )
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 100.0, 97.0, 94.0, 95.0, 96.0, 97.0, 98.0],
            "BIL": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07],
        },
        index=dates,
    )
    horizon = DefensiveJudgementHorizon(
        label="3d",
        trading_days=3,
        drawdown_correct_threshold=-0.03,
        false_alarm_excess_threshold=0.01,
        false_alarm_drawdown_floor=-0.02,
    )

    outcomes = build_scenario_outcomes(
        states,
        prices,
        horizons=(horizon,),
        probability_column="risk_off_probability",
        benchmark_ticker="SPY",
        cash_ticker="BIL",
    )

    assert outcomes["realized_risk_off"].tolist() == [1, 1]
    assert outcomes["predicted_risk_off_probability"].tolist() == [0.8, 0.2]


def test_calibrated_probabilities_have_positive_skill_and_discrimination() -> None:
    outcomes = pd.DataFrame(
        {
            "origin_date": pd.bdate_range("2025-01-01", periods=8),
            "horizon": "1m",
            "forward_days": 21,
            "predicted_risk_off_probability": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
            "realized_risk_off": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    metrics = build_calibration_metrics(outcomes, bootstrap_samples=20, seed=7).iloc[0]

    assert metrics["brier_skill"] > 0
    assert metrics["auc"] == pytest.approx(1.0)


def test_scenario_authority_requires_calibration_and_discrimination() -> None:
    assert calibration_authority(
        brier_skill=-0.1,
        auc=0.8,
        observations=500,
        min_observations=100,
    ) == 0.0
    assert calibration_authority(
        brier_skill=0.2,
        auc=0.5,
        observations=500,
        min_observations=100,
    ) == 0.0
    assert calibration_authority(
        brier_skill=0.25,
        auc=0.75,
        observations=500,
        min_observations=100,
    ) == pytest.approx((0.25 * 0.5) ** 0.5)


def test_effective_multiplier_shrinks_toward_no_adjustment() -> None:
    assert effective_scenario_multiplier(0.76, 0.0) == pytest.approx(1.0)
    assert effective_scenario_multiplier(0.76, 0.5) == pytest.approx(0.88)
    assert effective_scenario_multiplier(0.76, 1.0) == pytest.approx(0.76)
