from __future__ import annotations

import pandas as pd

from trade_bot.research.ai_repair_validation import (
    default_validation_specs,
    summarize_exposure_segments,
    summarize_validation,
)


def test_default_validation_specs_include_generalized_dual_confirm_first() -> None:
    specs = default_validation_specs()

    assert specs[0].name == "ai_dual_confirm_break_cap45_bil"
    assert specs[0].stress_signal == "ai_dual_confirm_break"
    assert any(spec.stress_signal == "nvda_drawdown" for spec in specs)


def test_validation_summary_promotes_robust_generalized_variant() -> None:
    variant_metrics = pd.DataFrame(
        {
            "strategy": ["s1", "s2"],
            "family": ["f", "f"],
            "variant_name": ["v", "v"],
            "stress_signal": ["ai_dual_confirm_break", "ai_dual_confirm_break"],
            "ai_cap": [0.45, 0.45],
            "destination": ["bil", "bil"],
            "cagr": [0.12, 0.13],
            "max_drawdown": [-0.20, -0.21],
            "calmar": [0.60, 0.62],
            "delta_cagr_vs_base": [0.002, 0.003],
            "delta_max_drawdown_vs_base": [0.010, 0.012],
            "active_day_rate": [0.01, 0.02],
        }
    )
    window_summary = pd.DataFrame(
        {
            "window_name": ["2011_2012_ai_growth_wound"],
            "variant_name": ["v"],
            "strategies": [2],
            "median_window_return": [-0.02],
            "median_window_max_drawdown": [-0.20],
            "median_delta_window_return_vs_base": [0.01],
            "median_delta_window_max_drawdown_vs_base": [0.01],
        }
    )
    yearly = pd.DataFrame(
        {
            "variant_name": ["v"] * 4,
            "delta_cagr": [0.003, 0.001, 0.0, -0.001],
        }
    )
    rolling = pd.DataFrame(
        {
            "variant_name": ["v"] * 6,
            "window_years": [1, 1, 3, 3, 5, 5],
            "delta_cagr": [0.001, -0.001, 0.002, 0.001, 0.001, 0.001],
        }
    )
    era = pd.DataFrame(
        {
            "variant_name": ["v", "v"],
            "window_name": ["current_ai_cycle", "rates_growth_wound"],
            "delta_total_return": [0.0, 0.01],
        }
    )

    summary = summarize_validation(
        variant_metrics=variant_metrics,
        yearly_deltas=yearly,
        rolling_deltas=rolling,
        era_deltas=era,
        window_summary=window_summary,
    )

    assert summary.iloc[0]["promotion_gate"] == "promote_candidate"


def test_validation_summary_keeps_nvda_trigger_as_diagnostic() -> None:
    variant_metrics = pd.DataFrame(
        {
            "strategy": ["s1"],
            "family": ["f"],
            "variant_name": ["nvda"],
            "stress_signal": ["nvda_drawdown"],
            "ai_cap": [0.35],
            "destination": ["bil_gld_tlt"],
            "cagr": [0.14],
            "max_drawdown": [-0.17],
            "calmar": [0.80],
            "delta_cagr_vs_base": [0.02],
            "delta_max_drawdown_vs_base": [0.03],
            "active_day_rate": [0.02],
        }
    )
    window_summary = pd.DataFrame(
        {
            "window_name": ["2011_2012_ai_growth_wound"],
            "variant_name": ["nvda"],
            "strategies": [1],
            "median_window_return": [0.01],
            "median_window_max_drawdown": [-0.17],
            "median_delta_window_return_vs_base": [0.03],
            "median_delta_window_max_drawdown_vs_base": [0.03],
        }
    )

    summary = summarize_validation(
        variant_metrics=variant_metrics,
        yearly_deltas=pd.DataFrame({"variant_name": ["nvda"], "delta_cagr": [0.01]}),
        rolling_deltas=pd.DataFrame(
            {"variant_name": ["nvda"], "window_years": [3], "delta_cagr": [0.01]}
        ),
        era_deltas=pd.DataFrame(
            {
                "variant_name": ["nvda"],
                "window_name": ["current_ai_cycle"],
                "delta_total_return": [0.0],
            }
        ),
        window_summary=window_summary,
    )

    assert summary.iloc[0]["promotion_gate"] == "diagnostic_only_single_name"


def test_exposure_segments_show_where_variant_works() -> None:
    variant_metrics = pd.DataFrame(
        {
            "variant_name": ["v", "v", "v"],
            "average_ai_growth_weight": [0.10, 0.40, 0.70],
            "delta_cagr_vs_base": [0.0, 0.001, 0.003],
            "delta_max_drawdown_vs_base": [0.0, 0.004, 0.020],
        }
    )

    summary = summarize_exposure_segments(variant_metrics)

    high = summary[summary["ai_exposure_bucket"].eq("high_ai_exposure")].iloc[0]
    assert high["strategies"] == 1
    assert high["median_delta_max_drawdown"] == 0.020
