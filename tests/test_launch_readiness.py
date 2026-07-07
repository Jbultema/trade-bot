from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.launch_readiness import (
    build_launch_ramp_plan,
    build_launch_readiness,
)


def test_launch_readiness_scores_clean_entry_history() -> None:
    index = pd.bdate_range("2020-01-01", periods=520)
    benchmark_returns = pd.Series(0.0008, index=index)
    strategy_returns = pd.Series(0.0015, index=index)

    run = build_launch_readiness(
        _result("strategy", strategy_returns),
        benchmark_result=_result("buy_hold_spy", benchmark_returns),
        horizons={"3m": 63},
        ramp_weeks=(0, 4),
        primary_horizon="3m",
        start_frequency="M",
    )

    assert not run.windows.empty
    assert not run.summary.empty
    assert run.recommendation["launch_label"] in {"ready", "set"}
    assert run.summary["beat_rate"].max() == 1.0
    assert run.recommendation["operating_boundary"]


def test_launch_readiness_counts_bad_starts() -> None:
    index = pd.bdate_range("2020-01-01", periods=520)
    returns = pd.Series(0.0005, index=index)
    returns.iloc[22:43] = -0.015

    run = build_launch_readiness(
        _result("fragile_strategy", returns),
        benchmark_result=_result("buy_hold_spy", pd.Series(0.0, index=index)),
        horizons={"3m": 63},
        ramp_weeks=(0,),
        primary_horizon="3m",
        start_frequency="M",
        bad_start_drawdown=-0.08,
    )

    assert run.windows["bad_start"].any()
    assert run.summary["bad_start_rate"].max() > 0.0


def test_launch_readiness_skips_invalid_window_values() -> None:
    index = pd.bdate_range("2020-01-01", periods=520)
    strategy_returns = pd.Series(0.0012, index=index)
    strategy_returns.iloc[10] = pd.NA
    strategy_returns.iloc[80] = float("inf")
    benchmark_returns = pd.Series(0.0006, index=index)

    run = build_launch_readiness(
        _result("strategy", strategy_returns),
        benchmark_result=_result("buy_hold_spy", benchmark_returns),
        horizons={"3m": 63},
        ramp_weeks=(0, 4, 8),
        primary_horizon="3m",
        start_frequency="M",
    )

    assert not run.windows.empty
    finite_columns = [
        "total_return",
        "cagr",
        "max_drawdown",
        "first_month_drawdown",
        "benchmark_return",
        "excess_return",
    ]
    assert not run.windows[finite_columns].isna().any().any()
    assert not run.summary[
        ["median_return", "worst_return", "median_max_drawdown", "launch_score"]
    ].isna().any().any()


def test_current_state_can_lower_launch_readiness_without_changing_backtest() -> None:
    index = pd.bdate_range("2020-01-01", periods=520)
    strategy_returns = pd.Series(0.0015, index=index)
    benchmark_returns = pd.Series(0.0008, index=index)

    calm = build_launch_readiness(
        _result("strategy", strategy_returns),
        benchmark_result=_result("buy_hold_spy", benchmark_returns),
        current_state=_state("green", 0.8, risk_off=0.05, transition=0.20),
        horizons={"3m": 63},
        ramp_weeks=(0,),
        primary_horizon="3m",
        start_frequency="M",
    )
    stressed = build_launch_readiness(
        _result("strategy", strategy_returns),
        benchmark_result=_result("buy_hold_spy", benchmark_returns),
        current_state=_state("red", 0.1, risk_off=0.80, transition=0.20),
        horizons={"3m": 63},
        ramp_weeks=(0,),
        primary_horizon="3m",
        start_frequency="M",
    )

    assert stressed.summary["current_entry_score"].iloc[0] < calm.summary[
        "current_entry_score"
    ].iloc[0]
    assert stressed.recommendation["launch_label"] != "ready"


def test_launch_ramp_plan_wait_keeps_capital_reserved() -> None:
    plan = build_launch_ramp_plan(
        capital_to_launch=1_000,
        target_fraction=0.50,
        ramp_weeks=8,
        launch_label="wait",
    )

    assert plan.iloc[0]["capital_deployed"] == 0.0
    assert plan.iloc[0]["cash_reserved"] == 500.0


def test_launch_ramp_plan_set_immediate_opens_only_starter() -> None:
    plan = build_launch_ramp_plan(
        capital_to_launch=1_000,
        target_fraction=0.50,
        ramp_weeks=0,
        launch_label="set",
    )

    assert plan.iloc[0]["target_fraction_of_strategy"] == 0.25
    assert plan.iloc[0]["account_fraction_deployed"] == 0.125
    assert plan.iloc[0]["capital_deployed"] == 125.0
    assert plan.iloc[0]["cash_reserved"] == 375.0
    assert "starter" in str(plan.iloc[0]["instruction"]).lower()


def test_launch_ramp_plan_stages_to_target() -> None:
    plan = build_launch_ramp_plan(
        capital_to_launch=1_000,
        target_fraction=0.50,
        ramp_weeks=4,
        launch_label="set",
    )

    assert plan.iloc[0]["capital_deployed"] == 125.0
    assert plan.iloc[-1]["capital_deployed"] == 500.0
    assert plan.iloc[-1]["account_fraction_deployed"] == 0.50


def _result(name: str, returns: pd.Series) -> BacktestResult:
    equity = 100.0 * (1.0 + returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=returns.rename(name),
        gross_returns=returns.rename(name),
        weights=pd.DataFrame({"SPY": 1.0}, index=returns.index),
        target_weights=pd.DataFrame({"SPY": 1.0}, index=returns.index),
        turnover=pd.Series(0.0, index=returns.index, name=name),
        transaction_costs=pd.Series(0.0, index=returns.index, name=name),
    )


def _state(
    risk_status: str,
    risk_score: float,
    *,
    risk_off: float,
    transition: float,
) -> _State:
    return _State(
        risk_status=risk_status,
        risk_score=risk_score,
        risk_off=risk_off,
        transition=transition,
    )


class _State:
    def __init__(
        self,
        risk_status: str,
        risk_score: float,
        *,
        risk_off: float,
        transition: float,
    ) -> None:
        self.market_date = "2026-01-01"
        self.risk_status = risk_status
        self.risk_score = risk_score
        self.scenario_lattice = pd.DataFrame(
            [
                {"horizon": "1m", "risk_bucket": "risk_off", "probability": risk_off},
                {"horizon": "1m", "risk_bucket": "transition", "probability": transition},
            ]
        )
        bearish_scores = [-1.0] * 7 if risk_status == "red" else []
        self.confirmation_matrix = pd.DataFrame(
            {"score": bearish_scores or [1.0, 1.0, 0.0, 0.0]}
        )
