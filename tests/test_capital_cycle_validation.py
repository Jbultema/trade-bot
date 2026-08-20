from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.research.capital_cycle_validation import (
    MODEL_FEATURES,
    build_ablation_summary,
    build_capital_cycle_origin_panel,
    build_nested_capital_cycle_predictions,
    build_validation_readiness,
    run_capital_cycle_validation,
)


def _prices(periods: int = 1_300) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=periods)
    cycle = 0.08 * np.sin(np.linspace(0, 12 * np.pi, periods))
    trend = np.linspace(0, 1.0, periods)
    qqq = 100 * np.exp(trend + cycle)
    return pd.DataFrame(
        {
            "SPY": 100 * np.exp(0.75 * trend + 0.35 * cycle),
            "QQQ": qqq,
            "RSP": 100 * np.exp(0.60 * trend + 0.20 * cycle),
            "SMH": 100 * np.exp(1.15 * trend + 1.30 * cycle),
        },
        index=dates,
    )


def _synthetic_origin_panel(rows: int = 160) -> pd.DataFrame:
    dates = pd.bdate_range("2005-01-01", periods=rows * 21, freq="B")[::21]
    rollover = np.tile(np.linspace(0.05, 0.95, 20), rows // 20)
    labels = (rollover > 0.68).astype(float)
    panel = pd.DataFrame(
        {
            "origin_date": dates,
            "episode": [f"episode_{index // 32}" for index in range(rows)],
            "episode_weight": 1.0,
            "capital_vulnerability": np.clip(rollover + 0.10, 0, 1),
            "rollover_impulse": rollover,
            "vulnerability_rollover_interaction": rollover * np.clip(rollover + 0.10, 0, 1),
            "state_ordinal": rollover,
            "real_economy_confirmations": (rollover > 0.60).astype(float),
            "independent_rollover_confirmations": (rollover > 0.50).astype(float) * 2,
            "theme_break_10_63d": labels,
            "theme_break_10_63d_maturity_date": dates + pd.offsets.BDay(63),
            "theme_break_10_63d_forward_return": np.where(labels > 0, -0.12, 0.08),
            "theme_break_10_63d_forward_drawdown": np.where(labels > 0, -0.15, -0.03),
        }
    )
    for family in (
        "capital_intensity_overbuild",
        "financing_reflexivity",
        "monetization_efficiency",
        "infrastructure_demand_rollover",
        "theme_market_break",
        "margin_leverage",
        "rates_capital_competition",
        "equity_concentration",
    ):
        panel[f"family__{family}"] = rollover
        panel[f"direct__{family}"] = rollover
    return panel


def test_origin_panel_is_prior_only_and_keeps_unmatured_targets_missing() -> None:
    prices = _prices(950)
    fundamentals = pd.DataFrame(
        [
            {
                "available_date": "2019-01-01",
                "period_end": "2018-12-31",
                "entity": "sample",
                "metric": "capex_ocf",
                "value": 0.5,
            },
            {
                "available_date": "2030-01-01",
                "period_end": "2025-12-31",
                "entity": "sample",
                "metric": "capex_ocf",
                "value": 0.99,
            },
        ]
    )
    panel = build_capital_cycle_origin_panel(
        prices=prices,
        fundamentals=fundamentals,
        min_history_days=756,
        origin_step_days=42,
    )

    assert not panel.empty
    assert panel["allocation_authority"].eq(0.0).all()
    assert panel.iloc[-1]["theme_break_20_126d"] != panel.iloc[-1]["theme_break_20_126d"]
    assert panel["direct__capital_intensity_overbuild"].notna().any()


def test_nested_predictions_compare_ablations_out_of_sample() -> None:
    panel = _synthetic_origin_panel()
    predictions, diagnostics = build_nested_capital_cycle_predictions(panel)
    summary = build_ablation_summary(diagnostics)

    assert not predictions.empty
    assert diagnostics["status"].eq("evaluated").any()
    assert set(MODEL_FEATURES).intersection(set(summary["model"]))
    assert predictions["allocation_authority"].eq(0.0).all()
    assert "incremental_improvement_vs_existing_market" in summary


def test_missing_upstream_history_fails_readiness_closed() -> None:
    panel = _synthetic_origin_panel(160)
    direct_columns = [column for column in panel if column.startswith("direct__")]
    panel[direct_columns] = np.nan
    _predictions, diagnostics = build_nested_capital_cycle_predictions(panel)
    readiness = build_validation_readiness(panel, diagnostics, pd.DataFrame()).iloc[0]

    assert not readiness["historical_research_ready"]
    assert not readiness["promotion_ready"]
    assert "no_historical_direct_upstream_origins" in readiness["blockers"]


def test_run_writes_auditable_zero_authority_artifacts(tmp_path) -> None:
    result = run_capital_cycle_validation(
        prices=_prices(1_050),
        min_history_days=756,
        origin_step_days=63,
        output_dir=tmp_path,
    )

    assert set(result.output_paths) == {
        "origin_panel",
        "oos_predictions",
        "diagnostics",
        "episode_diagnostics",
        "leave_episode_out",
        "uncertainty",
        "ablation_summary",
        "utility_summary",
        "stage_gates",
        "readiness",
    }
    assert all(path.exists() for path in result.output_paths.values())
    assert result.readiness.iloc[0]["allocation_authority"] == 0.0
    assert not result.stage_gates.iloc[0]["economic_forecasting_gate_passed"]
    assert result.utility_summary.empty
