from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.monitoring import _render_monitoring
from trade_bot.dashboard.trends import (
    filter_history_time_range,
    load_monitoring_trend_frame,
    long_metric_line_figure,
)
from trade_bot.dashboard_v2.components.cards import render_callout, render_card_grid
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime
from trade_bot.dashboard_v2.services.warehouse_service import (
    champion_challenger_frame,
    monitoring_windows,
)


def render_monitoring_page(runtime: DashboardRuntime) -> None:
    warehouse_path = runtime.paths.run_store_path
    with timed("monitoring.summary"):
        frame = champion_challenger_frame(warehouse_path)
        windows = monitoring_windows(warehouse_path)

    active = windows[windows.get("status", pd.Series(dtype=str)).astype(str).eq("active")] if not windows.empty and "status" in windows else windows
    valued = frame[frame.get("valuation_date", pd.Series(dtype=str)).notna()] if not frame.empty and "valuation_date" in frame else pd.DataFrame()
    ahead = frame[frame.get("forward_status", pd.Series(dtype=str)).astype(str).eq("ahead_of_benchmark")] if not frame.empty and "forward_status" in frame else pd.DataFrame()
    lagging = frame[frame.get("forward_status", pd.Series(dtype=str)).astype(str).eq("lagging_benchmark")] if not frame.empty and "forward_status" in frame else pd.DataFrame()
    render_card_grid(
        [
            ("Active Windows", len(active)),
            ("Valued Today", len(valued)),
            ("Ahead", len(ahead)),
            ("Lagging", len(lagging)),
            ("Champions", _count_role(frame, "champion")),
            ("Challengers", _count_role(frame, "challenger")),
        ]
    )

    view = st.pills(
        "Monitoring view",
        ["Readout", "Trends", "Controls / full workbench"],
        default="Readout",
        selection_mode="single",
        key="dashboard_v2_monitoring_view",
    )
    selected_view = view or "Readout"
    if selected_view == "Readout":
        st.subheader("Champion / Challenger Readout")
        if frame.empty:
            st.info("No active monitoring rows are available. Seed or start a monitoring window.")
            return
        columns = [
            column
            for column in [
                "window_role",
                "start_date",
                "strategy_name",
                "forward_status",
                "valuation_date",
                "cumulative_return",
                "benchmark_cumulative_return",
                "drawdown",
                "beta_adjusted_spy_delta",
            ]
            if column in frame
        ]
        _render_metric_dataframe(_display_metrics(frame[columns].head(80)))
    elif selected_view == "Trends":
        render_callout("Forward trend plots read valuation history from DuckDB.", heavy=True)
        trends = load_monitoring_trend_frame(str(warehouse_path))
        trends = _render_monitoring_trend_range_controls(trends)
        figure = long_metric_line_figure(
            trends,
            category_column="window_label",
            value_column="cumulative_return",
            title="Cumulative Return by Monitoring Window",
            yaxis_title="Cumulative return",
            percent=True,
            top_n=8,
            height=420,
        )
        if figure is None:
            st.info("No monitoring trend history is available yet.")
        else:
            st.plotly_chart(figure, use_container_width=True)
    else:
        render_callout(
            "This loads the full Monitoring workbench, including management controls and detail tables.",
            heavy=True,
        )
        _render_monitoring(warehouse_path)


def _count_role(frame: pd.DataFrame, role: str) -> int:
    if frame.empty or "window_role" not in frame:
        return 0
    return int(frame["window_role"].astype(str).eq(role).sum())


def _render_monitoring_trend_range_controls(trends: pd.DataFrame) -> pd.DataFrame:
    control_cols = st.columns([1, 1, 1, 2])
    range_choice = control_cols[0].selectbox(
        "Time range",
        ["1M", "3M", "6M", "YTD", "1Y", "All", "Custom"],
        index=5,
        key="dashboard_v2_monitoring_trend_range",
    )
    custom_start = None
    custom_end = None
    if range_choice == "Custom":
        custom_start = control_cols[1].date_input(
            "Start",
            value=date.today() - timedelta(days=92),
            key="dashboard_v2_monitoring_trend_start",
        )
        custom_end = control_cols[2].date_input(
            "End",
            value=date.today(),
            key="dashboard_v2_monitoring_trend_end",
        )
    filtered = filter_history_time_range(
        trends,
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    control_cols[3].caption(
        f"Showing {len(filtered):,} valuation rows"
        + (f" across {filtered['window_label'].nunique():,} windows" if "window_label" in filtered else "")
    )
    return filtered
