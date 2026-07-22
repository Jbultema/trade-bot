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
from trade_bot.dashboard_v2.components.cards import (
    render_callout,
    render_card_grid,
    render_chart,
    render_section_header,
)
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

    active = (
        windows[windows.get("status", pd.Series(dtype=str)).astype(str).eq("active")]
        if not windows.empty and "status" in windows
        else windows
    )
    market_date = str(runtime.baseline_run.current_state.market_date)
    valued = _valuation_rows_for_market_date(frame, market_date)
    ahead = _rows_with_forward_status(valued, "ahead_of_benchmark")
    lagging = _rows_with_forward_status(valued, "lagging_benchmark")
    render_card_grid(
        [
            ("Active Windows", len(active)),
            ("Valued on Snapshot Date", len(valued)),
            ("Ahead", len(ahead)),
            ("Lagging", len(lagging)),
            ("Champions", _count_role(frame, "champion")),
            ("Challengers", _count_role(frame, "challenger")),
        ]
    )
    render_callout(_monitoring_freshness_read(frame, market_date))

    view = st.pills(
        "Monitoring view",
        ["Trends", "Readout", "Controls", "Full Workbench"],
        default="Trends",
        selection_mode="single",
        key="dashboard_v2_monitoring_view",
    )
    selected_view = view or "Trends"
    if selected_view == "Trends":
        render_callout(
            "Forward trend plots read valuation history from DuckDB. The default selection "
            "prioritizes champion and reference windows, and the selectors expose every "
            "available window instead of silently dropping all but the largest movers.",
            heavy=True,
        )
        trends = load_monitoring_trend_frame(str(warehouse_path))
        trends = _render_monitoring_trend_range_controls(trends)
        trends = _render_monitoring_trend_selection_controls(trends)
        figure = long_metric_line_figure(
            trends,
            category_column="display_window_label",
            value_column="cumulative_return",
            title="Cumulative Return by Monitoring Window",
            yaxis_title="Cumulative return",
            percent=True,
            top_n=max(
                1,
                (
                    int(trends["display_window_label"].nunique())
                    if "display_window_label" in trends
                    else 1
                ),
            ),
            height=420,
        )
        if figure is None or not figure.data:
            st.info("No monitoring trend history is available yet.")
        else:
            render_chart(figure)
    elif selected_view == "Readout":
        render_section_header("Champion / Challenger Readout")
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
    elif selected_view == "Controls":
        render_callout(
            "Controls opens the operating management surface for starting, updating, and valuing monitoring windows.",
            heavy=True,
        )
        _render_monitoring(warehouse_path)
    else:
        render_callout(
            "This loads the complete Monitoring workbench, including management controls and detail tables.",
            heavy=True,
        )
        _render_monitoring(warehouse_path)


def _count_role(frame: pd.DataFrame, role: str) -> int:
    if frame.empty or "window_role" not in frame:
        return 0
    return int(frame["window_role"].astype(str).eq(role).sum())


def _valuation_rows_for_market_date(frame: pd.DataFrame, market_date: object) -> pd.DataFrame:
    if frame.empty or "valuation_date" not in frame:
        return pd.DataFrame(columns=frame.columns)
    target = pd.to_datetime(market_date, errors="coerce", utc=True)
    if pd.isna(target):
        return pd.DataFrame(columns=frame.columns)
    valuation_dates = _normalized_monitoring_dates(frame["valuation_date"])
    return frame[valuation_dates.eq(target.normalize())]


def _rows_with_forward_status(frame: pd.DataFrame, status: str) -> pd.DataFrame:
    if frame.empty or "forward_status" not in frame:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame["forward_status"].astype(str).eq(status)]


def _monitoring_freshness_read(frame: pd.DataFrame, market_date: object) -> str:
    target = pd.to_datetime(market_date, errors="coerce", utc=True)
    if frame.empty or "valuation_date" not in frame:
        return f"Snapshot market date: {market_date}. No paper valuations are available."
    valuation_dates = _normalized_monitoring_dates(frame["valuation_date"]).dropna()
    if valuation_dates.empty:
        return f"Snapshot market date: {market_date}. No valid paper valuation dates are available."
    latest = valuation_dates.max().date().isoformat()
    if not pd.isna(target) and latest == target.date().isoformat():
        return f"Snapshot market date: {target.date().isoformat()}. Paper valuations are current."
    return (
        f"Snapshot market date: {market_date}. Latest paper valuation: {latest}. "
        "A non-null valuation is not counted as current unless its date matches the snapshot."
    )


def _normalized_monitoring_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True, format="mixed").dt.normalize()


def _with_monitoring_trend_labels(trends: pd.DataFrame) -> pd.DataFrame:
    if trends.empty:
        return trends.copy()
    output = trends.copy()
    roles = output.get("window_role", pd.Series("unlabeled", index=output.index))
    labels = output.get("window_label", pd.Series("unknown window", index=output.index))
    accounts = output.get("account", pd.Series("", index=output.index)).fillna("").astype(str)
    account_prefix = accounts.where(accounts.eq(""), accounts + " | ")
    output["display_window_role"] = (
        roles.fillna("unlabeled").astype(str).str.replace("_", " ").str.title()
    )
    output["display_window_label"] = (
        output["display_window_role"]
        + " | "
        + account_prefix
        + labels.fillna("unknown window").astype(str)
    )
    return output


def _default_monitoring_trend_labels(
    trends: pd.DataFrame,
    *,
    limit: int = 8,
) -> list[str]:
    if trends.empty or "display_window_label" not in trends:
        return []
    working = trends.copy()
    working["cumulative_return"] = pd.to_numeric(working.get("cumulative_return"), errors="coerce")
    if "history_time" in working:
        working["history_time"] = pd.to_datetime(working["history_time"], errors="coerce")
        working = working.sort_values("history_time")
    latest = working.drop_duplicates("display_window_label", keep="last").copy()
    roles = latest.get("window_role", pd.Series("", index=latest.index)).astype(str)
    protected = (
        latest[roles.isin(["champion", "reference"])]["display_window_label"].astype(str).tolist()
    )
    remaining = latest[~latest["display_window_label"].astype(str).isin(protected)].copy()
    remaining["_absolute_return"] = remaining["cumulative_return"].abs()
    fill_count = max(0, int(limit) - len(protected))
    fillers = (
        remaining.sort_values("_absolute_return", ascending=False)["display_window_label"]
        .astype(str)
        .head(fill_count)
        .tolist()
    )
    return protected + fillers


def _render_monitoring_trend_selection_controls(trends: pd.DataFrame) -> pd.DataFrame:
    labeled = _with_monitoring_trend_labels(trends)
    if labeled.empty:
        return labeled
    role_options = sorted(labeled["display_window_role"].astype(str).unique().tolist())
    selected_roles = st.multiselect(
        "Monitoring roles",
        role_options,
        default=role_options,
        key="dashboard_v2_monitoring_trend_roles",
    )
    role_filtered = labeled[labeled["display_window_role"].astype(str).isin(selected_roles)]
    window_options = sorted(role_filtered["display_window_label"].astype(str).unique().tolist())
    defaults = _default_monitoring_trend_labels(role_filtered)
    selected_windows = st.multiselect(
        "Monitoring windows",
        window_options,
        default=defaults,
        key="dashboard_v2_monitoring_trend_windows",
        help="Champion and reference windows are selected first; add or remove any window here.",
    )
    filtered = role_filtered[
        role_filtered["display_window_label"].astype(str).isin(selected_windows)
    ]
    st.caption(
        f"Showing {len(selected_windows):,} of {len(window_options):,} windows after role and "
        "time-range filters."
    )
    return filtered


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
        + (
            f" across {filtered['window_label'].nunique():,} windows"
            if "window_label" in filtered
            else ""
        )
    )
    return filtered
