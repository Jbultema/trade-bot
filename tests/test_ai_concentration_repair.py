from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.ai_concentration_repair import (
    AiRepairSpec,
    _stress_signal,
    apply_ai_repair_variant,
    summarize_variant_metrics,
)


def test_ai_repair_variant_caps_ai_weight_and_releases_to_bil() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * 260,
            "SPY": [100.0] * 260,
            "SMH": [100.0] * 260,
            "HYG": [100.0] * 260,
            "LQD": [100.0] * 260,
            "BIL": [100.0] * 260,
        },
        index=dates,
    )
    prices.loc[dates[126]:, "QQQ"] = 85.0
    prices.loc[dates[126]:, "SMH"] = 80.0
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    weights["QQQ"] = 0.80
    weights["BIL"] = 0.20
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
    spec = AiRepairSpec(
        name="ai_drawdown_smh_trend_cap55_bil",
        stress_signal="ai_drawdown_smh_trend",
        ai_cap=0.55,
        destination="bil",
    )

    repaired = apply_ai_repair_variant(base, prices, spec, transaction_cost_bps=0.0)

    active = repaired.weights["BIL"] > weights["BIL"]
    assert active.any()
    assert repaired.weights.loc[active, "QQQ"].max() <= 0.55
    assert repaired.weights.loc[active, "BIL"].min() >= 0.45


def test_variant_summary_marks_promising_when_cagr_and_drawdown_improve() -> None:
    variants = pd.DataFrame(
        {
            "variant_name": ["v1", "v1"],
            "stress_signal": ["s", "s"],
            "ai_cap": [0.55, 0.55],
            "destination": ["bil", "bil"],
            "cagr": [0.11, 0.12],
            "max_drawdown": [-0.18, -0.19],
            "calmar": [0.61, 0.63],
            "delta_cagr_vs_base": [0.01, 0.005],
            "delta_max_drawdown_vs_base": [0.02, 0.015],
            "active_day_rate": [0.1, 0.1],
        }
    )
    windows = pd.DataFrame(
        {
            "window_name": ["2011_2012_ai_growth_wound", "2011_2012_ai_growth_wound"],
            "variant_name": ["v1", "v1"],
            "median_delta_window_max_drawdown_vs_base": [0.03, 0.02],
        }
    )

    summary = summarize_variant_metrics(variants, windows)

    assert summary.iloc[0]["research_read"] == "promising"


def test_owned_ai_leader_stress_signal_generalizes_single_name_drawdown() -> None:
    dates = pd.bdate_range("2024-01-01", periods=160)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * 160,
            "NVDA": [100.0] * 160,
            "MSFT": [100.0] * 160,
        },
        index=dates,
    )
    prices.loc[dates[126]:, "NVDA"] = 70.0
    weights = pd.DataFrame(0.0, index=dates, columns=["NVDA", "MSFT", "BIL"])
    weights["NVDA"] = 0.7
    weights["MSFT"] = 0.1
    weights["BIL"] = 0.2
    returns = pd.Series(0.0, index=dates)
    result = BacktestResult(
        name="base",
        equity=pd.Series(100.0, index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=dates),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    stress = _stress_signal("owned_ai_leader_drawdown", prices, result)

    assert stress.any()


def test_combined_breadth_or_relative_break_signal_returns_boolean_series() -> None:
    dates = pd.bdate_range("2024-01-01", periods=160)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * 160,
            "SMH": [100.0] * 160,
            "SOXX": [100.0] * 160,
            "NVDA": [100.0] * 160,
            "MSFT": [100.0] * 160,
        },
        index=dates,
    )
    prices.loc[dates[126]:, "QQQ"] = 80.0
    prices.loc[dates[126]:, ["SMH", "SOXX", "NVDA"]] = 70.0
    weights = pd.DataFrame(0.0, index=dates, columns=["QQQ", "BIL"])
    weights["QQQ"] = 1.0
    returns = pd.Series(0.0, index=dates)
    result = BacktestResult(
        name="base",
        equity=pd.Series(100.0, index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=dates),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    stress = _stress_signal("ai_breadth_or_relative_break", prices, result)

    assert stress.dtype == bool
    assert stress.any()


def test_dual_confirm_persistence_and_sticky_signals_are_boolean_series() -> None:
    dates = pd.bdate_range("2024-01-01", periods=170)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0] * 170,
            "SMH": [100.0] * 170,
            "SOXX": [100.0] * 170,
            "NVDA": [100.0] * 170,
            "MSFT": [100.0] * 170,
        },
        index=dates,
    )
    prices.loc[dates[126]:, "QQQ"] = 80.0
    prices.loc[dates[126]:, ["SMH", "SOXX", "NVDA"]] = 70.0
    weights = pd.DataFrame(0.0, index=dates, columns=["QQQ", "BIL"])
    weights["QQQ"] = 1.0
    returns = pd.Series(0.0, index=dates)
    result = BacktestResult(
        name="base",
        equity=pd.Series(100.0, index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=dates),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    persist = _stress_signal("ai_dual_confirm_break_persist3", prices, result)
    sticky = _stress_signal("ai_dual_confirm_break_sticky10", prices, result)

    assert persist.dtype == bool
    assert sticky.dtype == bool
    assert persist.any()
    assert sticky.any()
