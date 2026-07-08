from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.simulation_lab import (
    _drawdown_distribution_row,
    _reference_option_frame,
    _simulation_ablation_comparison_figure,
    _simulation_calibration_read,
    _simulation_comparison_row,
    _simulation_overlay_histogram,
    _simulation_validation_band_figure,
    _simulation_validation_metric_status,
    _simulation_validation_row,
    _simulation_validation_verdict,
    _validation_history_plain_english_read,
)


def test_reference_option_frame_uses_configured_benchmarks_and_excludes_selected() -> None:
    baseline_run = SimpleNamespace(
        results={
            "buy_hold_spy": object(),
            "buy_hold_qqq": object(),
            "unrelated_strategy": object(),
        }
    )

    frame = _reference_option_frame(baseline_run, "buy_hold_spy")

    assert frame.to_dict("records") == [{"strategy": "buy_hold_qqq", "label": "Hold QQQ"}]


def test_simulation_comparison_row_reports_selected_minus_reference_delta() -> None:
    row = _simulation_comparison_row(
        label="Hold SPY",
        strategy="buy_hold_spy",
        deterministic_wealth=1_100_000.0,
        bootstrap_summary={
            "terminal_wealth_p10": 900_000.0,
            "terminal_wealth_p50": 1_200_000.0,
            "terminal_wealth_p90": 1_500_000.0,
        },
        forward_summary={
            "terminal_wealth_p10": 850_000.0,
            "terminal_wealth_p50": 1_250_000.0,
            "terminal_wealth_p90": 1_650_000.0,
            "max_drawdown_p50": -0.18,
            "severe_drawdown_probability": 0.12,
        },
        selected_forward_median=1_750_000.0,
    )

    assert row["forward_median"] == 1_250_000.0
    assert row["selected_minus_row_forward_median"] == 500_000.0
    assert row["median_forward_drawdown"] == -0.18


def test_simulation_overlay_histogram_adds_one_trace_per_available_portfolio() -> None:
    paths = pd.DataFrame({"terminal_wealth": [950_000.0, 1_100_000.0, 1_250_000.0]})

    fig = _simulation_overlay_histogram(
        [
            {"label": "Selected", "paths": paths, "color": "#2563eb"},
            {"label": "Hold SPY", "paths": paths * 0.9, "color": "#0f766e"},
        ],
        column="terminal_wealth",
        title="Forward paths",
        xaxis_title="Terminal wealth",
    )

    assert len(fig.data) == 2
    assert fig.layout.barmode == "overlay"
    assert fig.data[0].name == "Selected"
    assert fig.data[1].name == "Hold SPY"
    assert fig.layout.legend.y < 0
    assert fig.layout.margin.b >= 70


def test_drawdown_distribution_row_reports_path_tail_risk() -> None:
    paths = pd.DataFrame(
        {
            "max_drawdown": [-0.10, -0.25, -0.35],
            "ulcer_index": [0.03, 0.08, 0.12],
        }
    )

    row = _drawdown_distribution_row(label="Selected forward", strategy="demo", paths=paths)

    assert row["paths"] == 3
    assert row["max_drawdown_p50"] == -0.25
    assert row["ulcer_index_p50"] == 0.08
    assert row["breach_soft_band"] == 2 / 3
    assert row["breach_hard_band"] == 1 / 3


def test_simulation_validation_row_compares_forward_paths_to_history() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 140.0], dtype=float)
    returns = equity.pct_change().fillna(0.0)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=pd.DataFrame(index=equity.index),
        target_weights=pd.DataFrame(index=equity.index),
        turnover=pd.Series([0.0] * len(equity)),
        transaction_costs=pd.Series([0.0] * len(equity)),
    )

    row = _simulation_validation_row(
        label="Selected strategy",
        strategy="demo",
        result=result,
        deterministic_wealth=1_000_000.0,
        bootstrap_summary={"terminal_wealth_p50": 950_000.0},
        forward_summary={
            "terminal_wealth_p50": 900_000.0,
            "max_drawdown_p50": -0.40,
            "ulcer_index_p50": 0.14,
        },
    )

    assert row["portfolio"] == "Selected strategy"
    assert row["historical_max_drawdown"] == -0.25
    assert row["forward_vs_deterministic"] == pytest.approx(-0.10)
    assert row["calibration_read"] == "simulation_more_stressed_than_history"


def test_simulation_calibration_read_labels_broadly_past_like_cases() -> None:
    assert (
        _simulation_calibration_read(
            deterministic_delta=0.01,
            drawdown_delta=-0.01,
            ulcer_delta=0.0,
        )
        == "broadly_past_like"
    )


def test_validation_history_read_explains_use_and_limits() -> None:
    ablation = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "label": "Baseline",
                "coverage_error": 0.002,
                "median_abs_error": 0.050,
            },
            {
                "variant": "duration_covariate",
                "label": "Duration + covariate",
                "coverage_error": -0.022,
                "median_abs_error": 0.048,
            },
        ]
    )

    read = _validation_history_plain_english_read(
        validity_read="calibrated_enough_for_research",
        interval_coverage=0.778,
        target_coverage=0.80,
        coverage_error=-0.022,
        median_abs_error=0.048,
        launch_accuracy=0.086,
        latest_ablation=ablation,
    )

    assert "roughly calibrated" in read
    assert "planning range, not as a launch trigger" in read
    assert "Ablation is mixed" in read


def test_simulation_validation_statuses_call_out_generous_or_weak_metrics() -> None:
    assert _simulation_validation_metric_status("coverage", -0.022) == "good"
    assert _simulation_validation_metric_status("coverage", 0.062) == "warn"
    assert _simulation_validation_metric_status("coverage", 0.12) == "bad"
    assert _simulation_validation_metric_status("median", 0.048) == "warn"
    assert _simulation_validation_metric_status("launch", 0.086) == "bad"
    assert _simulation_validation_metric_status("action_score", 0.70) == "warn"
    assert _simulation_validation_metric_status("overrisk", 0.55) == "bad"


def test_simulation_validation_verdict_prioritizes_weak_launch_signal() -> None:
    verdict = _simulation_validation_verdict(
        coverage_error=-0.022,
        median_abs_error=0.048,
        launch_accuracy=0.086,
        launch_action_score=0.63,
        launch_overrisk_rate=0.52,
    )

    assert verdict["status"] == "bad"
    assert "not decision-ready" in verdict["title"]


def test_validation_history_figures_make_band_and_ablation_visuals() -> None:
    origins = pd.DataFrame(
        {
            "origin_date": ["2026-01-31", "2026-02-28"],
            "realized_return": [0.02, -0.03],
            "simulated_p10_return": [-0.02, -0.01],
            "simulated_p50_return": [0.01, 0.02],
            "simulated_p90_return": [0.05, 0.06],
            "realized_in_interval": [True, False],
            "p50_error": [-0.01, 0.05],
        }
    )
    ablation = pd.DataFrame(
        {
            "variant": ["baseline", "duration"],
            "label": ["Baseline", "Duration"],
            "coverage_error": [0.002, -0.022],
            "median_abs_error": [0.050, 0.048],
        }
    )

    band = _simulation_validation_band_figure(origins)
    ablation_fig = _simulation_ablation_comparison_figure(ablation)

    assert len(band.data) == 4
    assert band.data[-1].name == "Realized return"
    assert len(ablation_fig.data) == 2
    assert ablation_fig.layout.barmode == "group"
