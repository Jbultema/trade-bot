from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.launch_readiness import (
    build_aggregate_launch_readiness,
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


def test_launch_readiness_ramp_protocols_change_short_window_returns() -> None:
    index = pd.bdate_range("2020-01-01", periods=260)
    returns = pd.Series(0.0010, index=index)
    returns.iloc[1:8] = -0.025
    returns.iloc[8:25] = 0.004

    run = build_launch_readiness(
        _result("early_drawdown_strategy", returns),
        benchmark_result=_result("buy_hold_spy", pd.Series(0.0, index=index)),
        horizons={"1m": 21},
        ramp_weeks=(0, 4, 8),
        primary_horizon="1m",
        start_frequency="M",
    )

    first_start = run.windows[run.windows["start_date"].eq("2020-01-01")]
    protocol_returns = first_start.set_index("protocol")["total_return"]

    assert protocol_returns["Immediate full launch"] < protocol_returns["25% now / 4w ramp"]
    assert protocol_returns.max() - protocol_returns.min() > 0.005
    assert protocol_returns.nunique() == 3


def test_aggregate_launch_readiness_builds_transition_and_protocol_frames() -> None:
    index = pd.bdate_range("2020-01-01", periods=900)
    benchmark_returns = pd.Series(0.0004, index=index)
    strong_returns = pd.Series(0.0012, index=index)
    choppy_returns = pd.Series(0.0006, index=index)
    choppy_returns.iloc[22:43] = -0.012
    choppy_returns.iloc[300:330] = 0.004

    run = build_aggregate_launch_readiness(
        {
            "strong": _result("strong", strong_returns),
            "choppy": _result("choppy", choppy_returns),
        },
        benchmark_result=_result("buy_hold_spy", benchmark_returns),
        horizons={"1m": 21, "3m": 63, "1y": 252},
        ramp_weeks=(0, 4, 8, 12),
        start_frequency="M",
    )

    assert run.strategy_count == 2
    assert set(run.horizon_label_counts["horizon"]) == {"1m", "3m", "1y"}
    assert set(run.horizon_label_counts["launch_label"]) == {"no_go", "wait", "set", "ready"}
    assert not run.horizon_transition_matrix.empty
    pair_counts = run.horizon_transition_matrix.groupby(
        ["from_horizon", "to_horizon"]
    )["count"].sum()
    assert (pair_counts == 2).all()
    assert not run.protocol_separation.empty
    assert "material_separation_rate" in run.protocol_separation_by_horizon.columns


def test_aggregate_launch_protocol_separation_detects_material_ramp_effect() -> None:
    index = pd.bdate_range("2020-01-01", periods=320)
    returns = pd.Series(0.0010, index=index)
    returns.iloc[1:8] = -0.040
    returns.iloc[8:25] = 0.006

    run = build_aggregate_launch_readiness(
        {"early_drawdown_strategy": _result("early_drawdown_strategy", returns)},
        benchmark_result=_result("buy_hold_spy", pd.Series(0.0, index=index)),
        horizons={"1m": 21},
        ramp_weeks=(0, 4, 8),
        start_frequency="M",
        protocol_small_spread=0.0001,
        protocol_material_spread=0.001,
    )

    assert run.protocol_separation["protocol_spread"].max() > 0.001
    assert "material" in set(run.protocol_separation["separation_label"])


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
