from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from trade_bot.dashboard.metric_explainers import (
    metric_categories,
    metric_detail,
    metric_guide_frame,
    metric_help,
)
from trade_bot.research.action_headline import ActionHeadline


def _helped_metric(
    container: Any,
    label: str,
    value: object,
    *,
    key: str | None = None,
) -> None:
    container.metric(label, value, help=metric_help(key or label))


def _metric_column_config(
    frame: pd.DataFrame,
    *,
    column_help: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        help_text = column_help.get(column) if column_help else None
        if not help_text:
            help_text = metric_help(column)
        if help_text:
            config[column] = st.column_config.Column(help=help_text)
    return config


def _render_metric_dataframe(
    frame: pd.DataFrame,
    *,
    use_container_width: bool = True,
    hide_index: bool | None = None,
    column_help: Mapping[str, str] | None = None,
) -> None:
    kwargs: dict[str, Any] = {"use_container_width": use_container_width}
    column_config = _metric_column_config(frame, column_help=column_help)
    if column_config:
        kwargs["column_config"] = column_config
    if hide_index is not None:
        kwargs["hide_index"] = hide_index
    st.dataframe(frame, **kwargs)


def _render_metric_guide() -> None:
    with st.expander("Metric Guide", expanded=False):
        st.caption(
            "Quick reference for the scorecard, risk, scenario, and performance terms used across the dashboard."
        )
        filter_cols = st.columns([1, 2])
        category_options = ["all", *metric_categories()]
        selected_category = filter_cols[0].selectbox(
            "Metric category",
            category_options,
            key="metric_guide_category",
        )
        search_text = filter_cols[1].text_input(
            "Search",
            "",
            key="metric_guide_search",
        )
        guide = metric_guide_frame(
            category=None if selected_category == "all" else selected_category,
            search=search_text,
        )
        if guide.empty:
            st.write("No matching metric explainers.")
            return

        selected_metric = st.selectbox(
            "Detailed metric",
            list(guide["metric"]),
            key="metric_guide_metric",
        )
        detail = metric_detail(str(selected_metric))
        if detail is not None:
            st.markdown(f"**{detail.metric}**")
            st.write(detail.plain_english)
            st.markdown(f"**Calculation:** {detail.calculation}")
            st.markdown(f"**How to read:** {detail.how_to_read}")
            st.markdown(f"**Watch out:** {detail.caution}")

        guide_columns = ["metric", "category", "plain_english", "how_to_read", "caution"]
        st.dataframe(guide[guide_columns], use_container_width=True, hide_index=True)


def _render_action_headline(headline: ActionHeadline) -> None:
    st.markdown(
        f"""
        <div class="action-banner action-{html.escape(headline.level)}">
            <p class="headline-label">Action Headline</p>
            <div class="headline-title">{html.escape(headline.headline)}</div>
            <p class="headline-copy">{html.escape(headline.explanation)}</p>
            <p class="headline-next">Next: {html.escape(headline.next_action)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row = headline.metrics.iloc[0]
    cols = st.columns(6)
    _helped_metric(cols[0], "Severity", str(headline.severity))
    _helped_metric(cols[1], "Risk", str(metric_row["risk_status"]), key="risk_status")
    _helped_metric(cols[2], "Risk Score", f"{float(metric_row['risk_score']):.2f}")
    _helped_metric(cols[3], "Max Change", f"{float(metric_row['max_position_change']):.1%}")
    _helped_metric(cols[4], "Active News", f"{int(metric_row['active_news_items'])}")
    _helped_metric(cols[5], "Open Tickets", f"{int(metric_row['open_ticket_count'])}")
    if not headline.drivers.empty:
        st.dataframe(headline.drivers, use_container_width=True)
