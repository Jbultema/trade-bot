from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.risk_policy_backtest import (
    apply_budget_overlay_to_result,
    build_policy_budget_frame,
)


def test_budget_overlay_scales_risk_assets_into_defensive_ticker() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 102.0, 104.0],
            "BIL": [100.0, 100.01, 100.02],
        },
        index=dates,
    )
    weights = pd.DataFrame({"QQQ": [1.0, 1.0, 1.0], "BIL": [0.0, 0.0, 0.0]}, index=dates)
    returns = pd.Series([0.0, 0.02, 0.0196], index=dates)
    result = BacktestResult(
        name="base",
        equity=pd.Series([100.0, 102.0, 104.0], index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series([1.0, 0.0, 0.0], index=dates),
        transaction_costs=pd.Series([0.0, 0.0, 0.0], index=dates),
    )
    budget = pd.Series([0.5, 0.5, 1.0], index=dates)

    adjusted = apply_budget_overlay_to_result(
        result,
        prices,
        budget,
        defensive_ticker="BIL",
        transaction_cost_bps=0.0,
        name="adjusted",
    )

    assert adjusted.weights.loc[dates[0], "QQQ"] == 0.5
    assert adjusted.weights.loc[dates[0], "BIL"] == 0.5
    assert adjusted.weights.loc[dates[2], "QQQ"] == 1.0


def test_signal_confirm_floor_uses_known_snapshot_fields() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    snapshots = pd.DataFrame(
        {
            "market_date": ["2024-01-02"],
            "event_name": ["test_break"],
            "risk_budget_multiplier": [0.25],
            "current_risk_asset_weight": [0.8],
            "portfolio_risk_multiplier": [0.5],
            "portfolio_constraints": ["scenario_weighted_stress"],
            "hard_defensive_action_flag": [True],
            "decision_sanity_break_count": [1],
            "prebreak_stage": ["early_watch"],
            "target_staged_risk_budget_multiplier": [0.75],
            "days_to_break": [90],
        }
    )

    budgets = build_policy_budget_frame(snapshots, dates, max_forward_fill_days=3)

    assert budgets.loc[pd.Timestamp("2024-01-02"), "actual_snapshot_budget"] == 0.25
    assert budgets.loc[pd.Timestamp("2024-01-02"), "signal_confirm_floor_75"] == 0.75
    assert budgets.loc[pd.Timestamp("2024-01-08"), "signal_confirm_floor_75"] == 1.0
