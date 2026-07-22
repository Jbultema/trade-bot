from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.simulation_lab import (
    _render_simulation_lab,
    _render_simulation_lab_direct_view,
)
from trade_bot.dashboard_v2.components.cards import (
    render_callout,
    render_card_grid,
    render_section_header,
)
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.experiment_service import scorecards
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime
from trade_bot.dashboard_v2.services.warehouse_service import simulation_validation_summary


def render_simulation_page(runtime: DashboardRuntime) -> None:
    with timed("simulation.summary"):
        runs, metrics = simulation_validation_summary(runtime.paths.run_store_path)
    latest = runs.head(1)
    latest_metrics = _latest_run_metrics(runs, metrics)
    primary_summary = _latest_primary_summary(latest_metrics)
    distribution_read = _distribution_calibration_read(latest, primary_summary)
    action_read = _action_readiness_read(latest, primary_summary)
    current_market_date = str(runtime.baseline_run.current_state.market_date)
    promoted_strategy = str(runtime.promoted_book.strategy_name)
    render_card_grid(
        [
            ("Validation Runs", len(runs)),
            ("Validated Strategy", _latest_value(latest, "strategy")),
            ("Validation Market Date", _latest_value(latest, "market_date")),
            ("Current Snapshot Date", current_market_date),
            ("Coverage", _fmt_pct(_latest_value(latest, "primary_interval_coverage"))),
            ("Median Miss", _fmt_pct(_latest_value(latest, "primary_median_abs_error"))),
        ]
    )
    render_callout(
        _validation_context_read(
            latest,
            current_market_date=current_market_date,
            promoted_strategy=promoted_strategy,
        )
    )
    render_section_header("Validation Interpretation")
    render_card_grid(
        [
            (
                "Distribution Calibration",
                _read_label(distribution_read),
            ),
            (
                "Go/No-Go Accuracy",
                _fmt_pct(_latest_metric_value(primary_summary, "launch_decision_accuracy")),
            ),
            (
                "Action Score",
                _fmt_pct(_latest_metric_value(primary_summary, "launch_action_score")),
            ),
            (
                "Over-Risk Rate",
                _fmt_pct(_latest_metric_value(primary_summary, "launch_overrisk_rate")),
            ),
            ("Action Readiness", _read_label(action_read)),
        ]
    )
    render_callout(
        "Return-band calibration evaluates distribution coverage, median bias, and drawdown "
        "probability calibration. Decision readiness is separate and checks whether the "
        "simulated wait/ramp/full action matched hindsight without taking too much risk. "
        "A research-calibrated band is not, by itself, permission to trade."
    )

    view = st.pills(
        "Simulation view",
        ["Strategy simulations", "Validation", "Full Workbench"],
        default="Strategy simulations",
        selection_mode="single",
        key="dashboard_v2_simulation_view",
    )
    selected_view = view or "Strategy simulations"
    if selected_view == "Strategy simulations":
        render_section_header("Strategy Simulations")
        _render_simulation_lab_direct_view(
            "Strategy Simulations",
            bot_config=runtime.bot_config,
            baseline_run=runtime.baseline_run,
            experiment_scorecards=scorecards(),
            warehouse_path=str(runtime.paths.run_store_path),
        )
    elif selected_view == "Validation":
        render_section_header("Validation Summary")
        if runs.empty:
            st.info("No simulation validation runs are persisted yet.")
        else:
            _render_metric_dataframe(_display_metrics(runs.head(25)))
        render_section_header("Per-Horizon Metrics")
        if latest_metrics.empty:
            st.info("No per-horizon simulation metrics are available yet.")
        else:
            summary = (
                latest_metrics[
                    latest_metrics.get("metric_scope", pd.Series(dtype=str))
                    .astype(str)
                    .eq("horizon_summary")
                ]
                if "metric_scope" in latest_metrics
                else latest_metrics
            )
            _render_metric_dataframe(_display_metrics(summary.head(100)))
    else:
        render_callout(
            "This loads the full Simulation Lab. Path engines remain gated inside that page.",
            heavy=True,
        )
        _render_simulation_lab(
            runtime.bot_config,
            runtime.baseline_run,
            scorecards(),
            warehouse_path=str(runtime.paths.run_store_path),
        )


def _latest_run_metrics(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if (
        runs.empty
        or metrics.empty
        or "validation_run_id" not in runs
        or "validation_run_id" not in metrics
    ):
        return pd.DataFrame()
    latest_id = str(runs.iloc[0]["validation_run_id"])
    return metrics[metrics["validation_run_id"].astype(str) == latest_id]


def _latest_primary_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "metric_scope" not in metrics:
        return pd.DataFrame()
    return metrics[metrics["metric_scope"].astype(str).eq("primary_summary")].head(1)


def _latest_metric_value(frame: pd.DataFrame, column: str) -> object:
    return _latest_value(frame, column)


def _validation_context_read(
    latest: pd.DataFrame,
    *,
    current_market_date: str,
    promoted_strategy: str,
) -> str:
    if latest.empty:
        return (
            f"No persisted validation run is available. The current snapshot is "
            f"{current_market_date}, and the promoted strategy is {promoted_strategy}."
        )
    strategy = _latest_value(latest, "strategy")
    validation_date = _latest_value(latest, "market_date")
    snapshot_run_id = str(_latest_value(latest, "snapshot_run_id"))
    return (
        f"Latest persisted validation: {strategy}, market date {validation_date}, source "
        f"snapshot {snapshot_run_id}. Current dashboard context: snapshot "
        f"{current_market_date}, promoted strategy {promoted_strategy}. These contexts can "
        "differ; the validation result applies only to the named strategy and run."
    )


def _decision_readiness(primary_summary: pd.DataFrame) -> str:
    launch_accuracy = _safe_float(_latest_metric_value(primary_summary, "launch_decision_accuracy"))
    action_score = _safe_float(_latest_metric_value(primary_summary, "launch_action_score"))
    over_risk_rate = _safe_float(_latest_metric_value(primary_summary, "launch_overrisk_rate"))
    if all(value is None for value in (launch_accuracy, action_score, over_risk_rate)):
        return "Not evaluated"
    if (
        launch_accuracy is None
        or action_score is None
        or over_risk_rate is None
        or launch_accuracy < 0.60
        or action_score < 0.75
        or over_risk_rate > 0.25
    ):
        return "Not decision-ready"
    return "Action checks passed"


def _distribution_calibration_read(
    latest_run: pd.DataFrame,
    primary_summary: pd.DataFrame,
) -> str:
    persisted = _first_read(
        _latest_value(latest_run, "primary_distribution_calibration_read"),
        _latest_value(primary_summary, "distribution_calibration_read"),
    )
    if persisted:
        return persisted
    legacy = _first_read(
        _latest_value(primary_summary, "validity_read"),
        _latest_value(latest_run, "primary_validity_read"),
    )
    if legacy.startswith("return_bands_calibrated__"):
        return "return_bands_calibrated_for_research"
    if legacy == "return_bands_and_action_checks_ready_for_research":
        return "return_bands_calibrated_for_research"
    return legacy


def _action_readiness_read(
    latest_run: pd.DataFrame,
    primary_summary: pd.DataFrame,
) -> str:
    persisted = _first_read(
        _latest_value(latest_run, "primary_action_readiness_read"),
        _latest_value(primary_summary, "action_readiness_read"),
    )
    if persisted:
        return persisted
    legacy = _first_read(
        _latest_value(primary_summary, "validity_read"),
        _latest_value(latest_run, "primary_validity_read"),
    )
    if legacy.startswith("return_bands_calibrated__"):
        return legacy.removeprefix("return_bands_calibrated__")
    if legacy == "return_bands_and_action_checks_ready_for_research":
        return "action_checks_ready_for_research"
    fallback = _decision_readiness(primary_summary)
    return {
        "Not evaluated": "action_checks_not_evaluated",
        "Not decision-ready": "action_checks_not_ready",
        "Action checks passed": "action_checks_ready_for_research",
    }[fallback]


def _first_read(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {"n/a", "nan", "none"}:
            return text
    return ""


def _read_label(value: object) -> str:
    text = str(value or "").strip()
    if not text or text == "n/a":
        return "Not evaluated"
    return text.replace("_", " ").title()


def _latest_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame:
        return "n/a"
    return frame.iloc[0][column]


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
