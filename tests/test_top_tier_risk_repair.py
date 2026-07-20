from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.top_tier_risk_repair import (
    DefensiveReliefSpec,
    _stage_from_days_to_break,
    apply_defensive_relief_variant,
    summarize_top_tier_repair,
)


def test_defensive_relief_reallocates_excess_bil_when_trend_is_constructive() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "SPY": range(100, 360),
            "QQQ": range(100, 360),
            "SMH": range(100, 360),
            "HYG": range(100, 360),
            "LQD": range(100, 360),
            "BIL": [100.0] * 260,
        },
        index=dates,
        dtype=float,
    )
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    weights["BIL"] = 0.90
    weights["SPY"] = 0.10
    returns = pd.Series(0.0, index=dates)
    base = BacktestResult(
        name="base",
        equity=pd.Series(100.0, index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=dates),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    repaired = apply_defensive_relief_variant(
        base,
        prices,
        DefensiveReliefSpec("relief", max_defensive_weight=0.65),
        transaction_cost_bps=0.0,
    )

    active = repaired.weights["BIL"] < weights["BIL"]
    assert active.any()
    assert repaired.weights.loc[active, "BIL"].max() <= 0.65
    assert repaired.weights.loc[active, "SPY"].median() > weights.loc[active, "SPY"].median()


def test_prebreak_stage_mapping_separates_early_and_confirmed_windows() -> None:
    assert _stage_from_days_to_break(150) == "long_lead"
    assert _stage_from_days_to_break(90) == "early_watch"
    assert _stage_from_days_to_break(30) == "confirmed_prebreak"
    assert _stage_from_days_to_break(5) == "break_window"
    assert _stage_from_days_to_break(-5) is None


def test_top_tier_summary_promotes_positive_return_without_early_defense_increase() -> None:
    metrics = pd.DataFrame(
        {
            "variant_name": ["v", "v"],
            "cagr": [0.20, 0.21],
            "max_drawdown": [-0.20, -0.19],
            "calmar": [1.0, 1.1],
            "delta_cagr_vs_base": [0.01, 0.02],
            "delta_max_drawdown_vs_base": [0.01, 0.02],
            "delta_calmar_vs_base": [0.1, 0.2],
            "average_ai_growth_weight": [0.4, 0.4],
        }
    )
    behavior = pd.DataFrame(
        {
            "variant_name": ["v", "v"],
            "median_delta_hard_defensive_day_rate": [-0.01, -0.02],
            "median_delta_max_hard_defensive_run_days": [-5, -4],
        }
    )
    prebreak = pd.DataFrame(
        {
            "variant_name": ["v", "v"],
            "stage": ["early_watch", "confirmed_prebreak"],
            "delta_average_defensive_weight": [-0.10, 0.0],
            "delta_hard_defensive_day_rate": [-0.10, 0.0],
        }
    )

    summary = summarize_top_tier_repair(metrics, behavior, prebreak)

    assert summary.iloc[0]["promotion_gate"] == "0_promote_candidate"
