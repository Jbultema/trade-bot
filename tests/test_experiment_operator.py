from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.experiment_operator import (
    build_experiment_operator_plan,
    recommend_required_horizon,
    recommend_trial_capital,
)
from trade_bot.research.launch_readiness import LaunchReadinessRun


def test_i111_like_plan_uses_three_month_decision_horizon_and_qqq_contract() -> None:
    result = _result("i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145")
    launch_run = _launch_run(label="ready", score=0.82)

    plan = build_experiment_operator_plan(result, launch_run=launch_run)

    assert plan.required_horizon == "3m"
    assert plan.primary_benchmark == "buy_hold_qqq"
    assert plan.cash_floor == "buy_hold_bil"
    assert plan.secondary_benchmark == "buy_hold_spy"
    assert plan.recommended_capital == 10_000.0
    assert set(plan.success_contract["outcome"]) == {"validate", "continue", "fail", "context"}


def test_live_wait_trial_caps_to_smallest_preset_and_needs_six_months() -> None:
    capital, rationale = recommend_trial_capital(
        confidence_score=0.42,
        launch_label="wait",
        mode="live",
    )

    assert capital == 1_000.0
    assert "Low launch confidence" in rationale
    assert (
        recommend_required_horizon(
            launch_label="wait",
            confidence_score=0.42,
            signal_cycle_days=21,
        )
        == "6m"
    )


def test_existing_monitoring_window_validates_after_required_horizon() -> None:
    result = _result("i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145")
    launch_run = _launch_run(label="ready", score=0.82)
    windows = pd.DataFrame(
        [
            {
                "window_id": "win-1",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "mode": "paper",
                "account": "default_paper_account",
                "strategy_name": result.name,
                "start_date": "2026-01-01",
            }
        ]
    )
    valuations = pd.DataFrame(
        [
            {
                "window_id": "win-1",
                "valuation_date": "2026-04-10",
                "cumulative_return": 0.08,
                "benchmark_cumulative_return": 0.03,
                "excess_return": 0.05,
                "drawdown": -0.02,
            }
        ]
    )

    plan = build_experiment_operator_plan(
        result,
        launch_run=launch_run,
        monitoring_windows=windows,
        valuations=valuations,
    )

    assert plan.status_label == "validate"
    assert plan.current_status.iloc[0]["excess_return"] == 0.05


def _launch_run(*, label: str, score: float) -> LaunchReadinessRun:
    return LaunchReadinessRun(
        windows=pd.DataFrame(),
        summary=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
        ramp_plan=pd.DataFrame(),
        recommendation={
            "launch_label": label,
            "launch_score": score,
            "launch_action": "Launch the intended sleeve now.",
            "horizon": "3m",
        },
    )


def _result(name: str) -> BacktestResult:
    index = pd.bdate_range("2025-01-01", periods=160)
    returns = pd.Series(0.001, index=index, name=name)
    weights = pd.DataFrame(
        {
            "QQQ": [1.0] * 40 + [0.35] * 40 + [0.75] * 40 + [0.35] * 40,
            "BIL": [0.0] * 40 + [0.65] * 40 + [0.25] * 40 + [0.65] * 40,
        },
        index=index,
    )
    equity = (1.0 + returns).cumprod()
    turnover = weights.diff().abs().sum(axis=1).fillna(1.0)
    costs = pd.Series(0.0, index=index, name=name)
    return BacktestResult(
        name=name,
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=turnover,
        transaction_costs=costs,
    )
