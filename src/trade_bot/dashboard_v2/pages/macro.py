from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.news_macro import _render_news_and_macro
from trade_bot.dashboard_v2.components.cards import render_callout, render_card_grid
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime


def render_macro_page(runtime: DashboardRuntime) -> None:
    current_state = runtime.baseline_run.current_state
    macro_data = runtime.baseline_run.macro_data
    prices = runtime.baseline_run.prices
    render_card_grid(
        [
            ("Price Series", prices.shape[1]),
            ("Macro Series", macro_data.shape[1]),
            ("Macro Signals", len(current_state.macro_signals)),
            ("Pressure Groups", _macro_pressure_count(current_state.macro_category_summary)),
            ("News Items", len(runtime.baseline_run.news_monitor.triage)),
        ]
    )

    view = st.pills(
        "Macro view",
        ["Visual Explorer", "Signal Tables", "Full Workbench"],
        default="Visual Explorer",
        selection_mode="single",
        key="dashboard_v2_macro_view",
    )
    selected_view = view or "Visual Explorer"
    if selected_view == "Visual Explorer":
        _render_visual_explorer(prices=prices, macro_data=macro_data)
    elif selected_view == "Signal Tables":
        _render_signal_tables(runtime)
    else:
        render_callout(
            "This loads the full News & Macro workbench with narrative diagnostics, event triage, and raw tables.",
            heavy=True,
        )
        _render_news_and_macro(
            runtime.baseline_run,
            run_store_path=runtime.paths.run_store_path,
            artifact_dir=runtime.paths.artifact_dir,
            job_log_dir=runtime.paths.job_log_dir,
        )


def _render_visual_explorer(*, prices: pd.DataFrame, macro_data: pd.DataFrame) -> None:
    st.subheader("Macro Visual Explorer")
    st.caption(
        "Use this to inspect raw market proxies and macro series before drilling into narrative tables."
    )
    range_cols = st.columns([1, 1, 2])
    range_choice = range_cols[0].selectbox(
        "Time range",
        ["6M", "1Y", "3Y", "5Y", "All", "Custom"],
        index=2,
        key="dashboard_v2_macro_range",
    )
    custom_start = None
    custom_end = None
    if range_choice == "Custom":
        custom_start = range_cols[1].date_input(
            "Start",
            value=date.today() - timedelta(days=365),
            key="dashboard_v2_macro_start",
        )
        custom_end = range_cols[2].date_input(
            "End",
            value=date.today(),
            key="dashboard_v2_macro_end",
        )

    price_options = list(prices.columns.astype(str)) if not prices.empty else []
    default_tickers = [ticker for ticker in ["SPY", "QQQ", "BIL", "VEA", "IWM", "GLD", "TLT"] if ticker in price_options]
    selected_tickers = st.multiselect(
        "Market tickers",
        price_options,
        default=default_tickers[:6] or price_options[:4],
        key="dashboard_v2_macro_price_tickers",
    )
    price_frame = _slice_time_range(
        prices[selected_tickers] if selected_tickers and not prices.empty else pd.DataFrame(),
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    price_figure = _indexed_price_figure(price_frame, title="Selected Market Proxies")
    if price_figure is None:
        st.info("No selected price series are available.")
    else:
        st.plotly_chart(price_figure, use_container_width=True)

    numeric_macro = macro_data.select_dtypes(include="number") if not macro_data.empty else pd.DataFrame()
    macro_options = list(numeric_macro.columns.astype(str))
    macro_defaults = macro_options[:5]
    macro_cols = st.columns([3, 1])
    selected_macro = macro_cols[0].multiselect(
        "Macro series",
        macro_options,
        default=macro_defaults,
        key="dashboard_v2_macro_series",
    )
    transform = macro_cols[1].selectbox(
        "Macro scale",
        ["Z-score", "Level"],
        index=0,
        key="dashboard_v2_macro_scale",
    )
    macro_frame = _slice_time_range(
        numeric_macro[selected_macro] if selected_macro and not numeric_macro.empty else pd.DataFrame(),
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    macro_figure = _macro_signal_figure(macro_frame, title="Selected Macro Signals", zscore=transform == "Z-score")
    if macro_figure is None:
        st.info("No selected macro series are available.")
    else:
        st.plotly_chart(macro_figure, use_container_width=True)


def _render_signal_tables(runtime: DashboardRuntime) -> None:
    current_state = runtime.baseline_run.current_state
    st.subheader("Macro Signal Tables")
    if current_state.macro_category_summary.empty:
        st.info("No macro category summary is available.")
    else:
        st.markdown("**Category summary**")
        _render_metric_dataframe(_display_metrics(current_state.macro_category_summary))
    if current_state.macro_signals.empty:
        st.info("No macro signal table is available.")
    else:
        st.markdown("**Signal detail**")
        _render_metric_dataframe(_display_metrics(current_state.macro_signals))
    triage = runtime.baseline_run.news_monitor.triage
    if not triage.empty:
        st.markdown("**Current news/event triage**")
        _render_metric_dataframe(_display_metrics(triage.head(80)))


def _slice_time_range(
    frame: pd.DataFrame,
    range_choice: str,
    *,
    custom_start: date | None,
    custom_end: date | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce")
    working = working[working.index.notna()].sort_index()
    if working.empty or range_choice == "All":
        return working
    end = pd.to_datetime(custom_end) if custom_end is not None else working.index.max()
    if range_choice == "Custom":
        start = pd.to_datetime(custom_start) if custom_start is not None else working.index.min()
    else:
        days = {"6M": 183, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}.get(range_choice, 365 * 3)
        start = end - pd.Timedelta(days=days)
    return working[(working.index >= start) & (working.index <= end)]


def _indexed_price_figure(frame: pd.DataFrame, *, title: str) -> go.Figure | None:
    if frame.empty:
        return None
    normalized = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if normalized.empty:
        return None
    normalized = normalized.ffill()
    base = normalized.apply(lambda series: series.dropna().iloc[0] if not series.dropna().empty else pd.NA)
    indexed = normalized.divide(base.replace(0, pd.NA), axis=1) - 1.0
    fig = go.Figure()
    for column in indexed.columns:
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=indexed[column],
                mode="lines",
                name=str(column),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=390,
        yaxis={"title": "Indexed return", "tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 55, "b": 50},
        legend={"orientation": "h", "y": -0.20},
    )
    return fig


def _macro_signal_figure(frame: pd.DataFrame, *, title: str, zscore: bool) -> go.Figure | None:
    if frame.empty:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all").ffill()
    if numeric.empty:
        return None
    display = numeric
    y_title = "Level"
    hover = "%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"
    if zscore:
        display = numeric.apply(_zscore)
        y_title = "Z-score"
        hover = "%{x|%Y-%m-%d}<br>%{y:.2f} z<extra></extra>"
    fig = go.Figure()
    for column in display.columns:
        fig.add_trace(
            go.Scatter(
                x=display.index,
                y=display[column],
                mode="lines",
                name=str(column),
                hovertemplate=hover,
            )
        )
    fig.update_layout(
        title=title,
        height=390,
        yaxis_title=y_title,
        margin={"l": 20, "r": 20, "t": 55, "b": 50},
        legend={"orientation": "h", "y": -0.20},
    )
    return fig


def _zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    std = float(clean.std())
    if std == 0.0 or pd.isna(std):
        return clean * 0.0
    return (clean - float(clean.mean())) / std


def _macro_pressure_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    state_columns = [
        column
        for column in ("risk_state", "near_term_state", "latest_pressure_state", "state")
        if column in frame
    ]
    if not state_columns:
        return 0
    return int(frame[state_columns[0]].astype(str).str.contains("risk_pressure").sum())
