from __future__ import annotations

import pandas as pd

from trade_bot.config import ExecutionConfig
from trade_bot.research.i111_execution_hardening import (
    _execution_profiles,
    _mechanism_summary,
    _summary_markdown,
    default_execution_hardening_specs,
)


def test_execution_profiles_cover_each_weekday_daily_and_lag_stress() -> None:
    profiles = dict(_execution_profiles(ExecutionConfig()))

    assert profiles["wednesday_lag1"].rebalance == "W-WED"
    assert profiles["monday_lag1"].rebalance == "W-MON"
    assert profiles["tuesday_lag1"].rebalance == "W-TUE"
    assert profiles["thursday_lag1"].rebalance == "W-THU"
    assert profiles["friday_lag1"].rebalance == "W-FRI"
    assert profiles["daily_lag1"].rebalance == "D"
    assert profiles["wednesday_lag5"].signal_lag_days == 5


def test_default_specs_include_risk_sleeve_and_path_controls() -> None:
    specs = {spec.name: spec for spec in default_execution_hardening_specs()}

    assert specs["risk_sleeve_ai50_extreme"].updates["risk_repair_ai_cap_basis"] == "risk_sleeve"
    assert specs["weight_buffer08"].updates["risk_repair_min_rebalance_change"] == 0.08
    assert specs["hold10_step30"].updates["risk_repair_min_hold_days"] == 10


def test_mechanism_summary_requires_edge_preservation_and_tail_improvement() -> None:
    rows = []
    for mechanism, cagr, drawdown in (
        ("native_reference", 0.22, -0.30),
        ("qualified", 0.218, -0.20),
        ("return_tradeoff", 0.19, -0.20),
    ):
        for execution in ("wednesday_lag1", "daily_lag1"):
            rows.append(
                {
                    "mechanism": mechanism,
                    "execution": execution,
                    "cagr": cagr,
                    "max_drawdown": drawdown,
                    "average_turnover": 0.08,
                    "failure": drawdown < -0.22,
                }
            )

    summary = _mechanism_summary(pd.DataFrame(rows)).set_index("mechanism")

    assert summary.loc["qualified", "research_status"] == "promotion_like"
    assert summary.loc["return_tradeoff", "research_status"] == "tradeoff_only"


def test_summary_separates_transaction_cost_drag_from_tail_fragility() -> None:
    metrics = pd.DataFrame(
        [
            {
                "mechanism": "native_reference",
                "execution": execution,
                "cagr": cagr,
                "max_drawdown": drawdown,
            }
            for execution, cagr, drawdown in (
                ("wednesday_lag1", 0.22, -0.20),
                ("daily_lag1", 0.19, -0.29),
                ("monday_lag1", 0.20, -0.27),
            )
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {
                "mechanism": "native_reference",
                "wednesday_cagr": 0.22,
                "worst_execution_drawdown": -0.29,
                "median_execution_cagr": 0.20,
                "research_status": "no_robust_improvement",
            }
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "mechanism": mechanism,
                "execution": execution,
                "cagr": cagr,
                "max_drawdown": drawdown,
            }
            for mechanism, execution, cagr, drawdown in (
                ("full", "daily_lag1", 0.19, -0.29),
                ("without_transaction_costs", "daily_lag1", 0.22, -0.28),
                ("without_drawdown_guard", "daily_lag1", 0.195, -0.30),
            )
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"execution": "wednesday_lag1", "return_2022": -0.10},
            {"execution": "daily_lag1", "return_2022": -0.20},
        ]
    )

    summary = _summary_markdown(metrics, mechanisms, decomposition, diagnostics)

    assert "Daily turnover therefore explains much of the return drag" in summary
    assert "but not the tail gap" in summary
    assert "not by transaction costs" not in summary


def test_summary_reports_promotion_like_rows_conditionally() -> None:
    metrics = pd.DataFrame(
        [
            {
                "mechanism": "native_reference",
                "execution": execution,
                "cagr": cagr,
                "max_drawdown": drawdown,
            }
            for execution, cagr, drawdown in (
                ("wednesday_lag1", 0.22, -0.20),
                ("daily_lag1", 0.19, -0.29),
                ("monday_lag1", 0.20, -0.27),
            )
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {
                "mechanism": "qualified",
                "wednesday_cagr": 0.22,
                "worst_execution_drawdown": -0.24,
                "median_execution_cagr": 0.21,
                "research_status": "promotion_like",
            }
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "mechanism": mechanism,
                "execution": "daily_lag1",
                "cagr": cagr,
                "max_drawdown": drawdown,
            }
            for mechanism, cagr, drawdown in (
                ("full", 0.19, -0.29),
                ("without_transaction_costs", 0.22, -0.28),
                ("without_drawdown_guard", 0.195, -0.30),
            )
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"execution": "wednesday_lag1", "return_2022": -0.10},
            {"execution": "daily_lag1", "return_2022": -0.20},
        ]
    )

    summary = _summary_markdown(metrics, mechanisms, decomposition, diagnostics)

    assert "1 V2.2 mechanism(s) met the retrospective promotion-like screen" in summary
    assert "No V2.2 mechanism met" not in summary
