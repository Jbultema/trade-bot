from __future__ import annotations

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
    render_card_grid(
        [
            ("Validation Runs", len(runs)),
            ("Metric Rows", len(metrics)),
            ("Latest Strategy", _latest_value(latest, "strategy")),
            ("Latest Horizons", _latest_value(latest, "horizons")),
            ("Coverage", _fmt_pct(_latest_value(latest, "primary_interval_coverage"))),
            ("Median Miss", _fmt_pct(_latest_value(latest, "primary_median_absolute_error"))),
        ]
    )
    render_callout(
        "Simulation V2 starts with strategy paths and persisted validation results. Bootstrap/regime/factor engines remain explicit loads."
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
                    latest_metrics.get("metric_scope", pd.Series(dtype=str)).astype(str).eq("horizon_summary")
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
    if runs.empty or metrics.empty or "validation_run_id" not in runs or "validation_run_id" not in metrics:
        return pd.DataFrame()
    latest_id = str(runs.iloc[0]["validation_run_id"])
    return metrics[metrics["validation_run_id"].astype(str) == latest_id]


def _latest_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame:
        return "n/a"
    return frame.iloc[0][column]


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"
