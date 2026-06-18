from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _default_strategy_selection,
    _display_metrics,
    _result_date_bounds,
    _window_start_from_preset,
)
from trade_bot.DEFAULT import DEFAULT_PERFORMANCE_WINDOW, DEFAULT_PERFORMANCE_WINDOWS
from trade_bot.reporting.report import (
    latest_positions_frame,
    make_equity_drawdown_figure,
    window_performance_frame,
)
from trade_bot.research.baselines import BaselineRun


def _render_performance(baseline_run: BaselineRun) -> None:
    st.subheader("Performance")
    _render_metric_dataframe(_display_metrics(baseline_run.metrics))

    st.subheader("Windowed Performance")
    strategy_names = list(baseline_run.results)
    earliest_result_date, latest_result_date = _result_date_bounds(baseline_run.results)
    window_columns = st.columns([1, 2])
    window_preset = window_columns[0].selectbox(
        "Window",
        list(DEFAULT_PERFORMANCE_WINDOWS),
        index=list(DEFAULT_PERFORMANCE_WINDOWS).index(DEFAULT_PERFORMANCE_WINDOW),
    )
    selected_performance_strategies = window_columns[1].multiselect(
        "Approaches",
        strategy_names,
        default=_default_strategy_selection(strategy_names),
    )

    custom_start_date: date | None = None
    window_end = latest_result_date
    if window_preset == "Custom":
        custom_columns = st.columns(2)
        custom_start_date = cast(
            date,
            custom_columns[0].date_input(
                "Start",
                value=max(earliest_result_date, latest_result_date - pd.DateOffset(days=90)).date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
            ),
        )
        custom_end_date = cast(
            date,
            custom_columns[1].date_input(
                "End",
                value=latest_result_date.date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
            ),
        )
        window_end = min(
            latest_result_date, max(earliest_result_date, pd.Timestamp(custom_end_date))
        )

    window_start = _window_start_from_preset(
        window_preset,
        earliest=earliest_result_date,
        latest=latest_result_date,
        custom_start=custom_start_date,
    )
    if window_start > window_end:
        window_start = window_end

    if selected_performance_strategies:
        st.plotly_chart(
            make_equity_drawdown_figure(
                baseline_run.results,
                strategy_names=selected_performance_strategies,
                start=window_start,
                end=window_end,
                rebase=True,
                title=f"Growth of $1: {window_start.date()} to {window_end.date()}",
            ),
            use_container_width=True,
        )
        window_stats = window_performance_frame(
            baseline_run.results,
            strategy_names=selected_performance_strategies,
            start=window_start,
            end=window_end,
        )
        _render_metric_dataframe(_display_metrics(window_stats))
    else:
        st.warning("Select at least one approach.")

    st.subheader("Rolling Window Summary")
    _render_metric_dataframe(_display_metrics(baseline_run.window_summary))

    st.subheader("Calendar Year Returns")
    st.dataframe(
        baseline_run.calendar_returns.map(lambda value: f"{value:.2%}"),
        use_container_width=True,
    )

    st.subheader("Full-History Equity and Drawdown")
    st.plotly_chart(make_equity_drawdown_figure(baseline_run.results), use_container_width=True)

    st.subheader("Latest Positions")
    positions = latest_positions_frame(baseline_run.results)
    st.dataframe(positions.map(lambda value: f"{value:.2%}"), use_container_width=True)
