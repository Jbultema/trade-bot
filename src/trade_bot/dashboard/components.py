from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from trade_bot.dashboard.metric_explainers import (
    metric_categories,
    metric_detail,
    metric_guide_frame,
    metric_help,
)
from trade_bot.dashboard.ticket_explainers import (
    ticket_categories,
    ticket_detail,
    ticket_guide_frame,
)
from trade_bot.research.action_headline import ActionHeadline

_MISSING_SELECTION = object()


def _clearable_selectbox(
    label: str,
    options: Sequence[Any],
    *,
    key: str,
    default_index: int | None = 0,
    format_func: Callable[[Any], str] = str,
    help: str | None = None,
    placeholder: str | None = None,
    label_visibility: str = "visible",
) -> Any | None:
    option_list = list(options)
    if not option_list:
        return None

    select_col, clear_col = st.columns([24, 1], vertical_alignment="bottom")
    with clear_col:
        if st.button(
            "x",
            key=f"{key}__clear",
            help=f"Clear {label}",
            width="stretch",
        ):
            st.session_state[key] = None

    current_value = st.session_state.get(key, _MISSING_SELECTION)
    if current_value is _MISSING_SELECTION:
        index = (
            default_index
            if default_index is not None and 0 <= default_index < len(option_list)
            else None
        )
    elif current_value is None:
        index = None
    elif current_value in option_list:
        index = option_list.index(current_value)
    else:
        st.session_state[key] = None
        index = None

    with select_col:
        return st.selectbox(
            label,
            option_list,
            index=index,
            format_func=format_func,
            key=key,
            help=help,
            placeholder=placeholder or "Type to search...",
            label_visibility=label_visibility,
        )


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
    width: str | int = "stretch",
    hide_index: bool | None = None,
    column_help: Mapping[str, str] | None = None,
) -> None:
    kwargs: dict[str, Any] = {"width": width}
    column_config = _metric_column_config(frame, column_help=column_help)
    if column_config:
        kwargs["column_config"] = column_config
    if hide_index is not None:
        kwargs["hide_index"] = hide_index
    st.dataframe(frame, **kwargs)


def _render_runtime_notice(
    title: str,
    detail: str,
    *,
    tone: str = "warning",
) -> None:
    safe_tone = tone if tone in {"neutral", "warning", "critical", "success"} else "warning"
    st.markdown(
        f"""
        <div class="runtime-notice runtime-notice-{safe_tone}">
            <span class="runtime-notice-kicker">Runtime note</span>
            <strong>{html.escape(title)}</strong>
            <p>{html.escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

        selected_metric = _clearable_selectbox(
            "Detailed metric",
            list(guide["metric"]),
            key="metric_guide_metric",
            placeholder="Search metrics...",
        )
        if selected_metric is None:
            st.info("Choose a metric to inspect.")
            return
        detail = metric_detail(str(selected_metric))
        if detail is not None:
            st.markdown(f"**{detail.metric}**")
            st.write(detail.plain_english)
            st.markdown(f"**Calculation:** {detail.calculation}")
            st.markdown(f"**How to read:** {detail.how_to_read}")
            st.markdown(f"**Watch out:** {detail.caution}")

        guide_columns = ["metric", "category", "plain_english", "how_to_read", "caution"]
        st.dataframe(guide[guide_columns], width="stretch", hide_index=True)


def _render_metric_info_rail() -> None:
    with st.container(key="quick_reference_rail"):
        _render_metric_info_rail_content()


def _render_metric_info_rail_content() -> None:
    st.markdown(
        """
        <div class="metric-info-rail">
            <p class="metric-info-kicker">Quick Reference</p>
            <div class="metric-info-title">Term Lookup</div>
            <p class="metric-info-copy">
                Search a metric, ticket field, workflow term, or ticker to see
                what it means, how to read it, and what can go wrong.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    category_options = ["all", *_lookup_categories()]
    selected_category = st.selectbox(
        "Category",
        category_options,
        key="metric_info_rail_category",
    )
    search_text = st.text_input(
        "Search term",
        "",
        key="metric_info_rail_search",
        placeholder="CAGR, ulcer, ticket_id, SPY...",
    )
    guide = _lookup_guide_frame(
        category=None if selected_category == "all" else selected_category,
        search=search_text,
    )
    if guide.empty:
        st.info("No matching terms. Try a shorter search.")
        return

    option_keys = list(guide["lookup_key"])
    option_labels = dict(zip(guide["lookup_key"], guide["lookup_label"], strict=False))
    lookup_context = (selected_category, search_text.strip().lower())
    if st.session_state.get("metric_info_rail_context") != lookup_context:
        st.session_state["metric_info_rail_metric"] = _default_lookup_option(option_keys, guide)
        st.session_state["metric_info_rail_context"] = lookup_context
    elif (
        st.session_state.get("metric_info_rail_metric") is not None
        and st.session_state.get("metric_info_rail_metric") not in option_keys
    ):
        st.session_state["metric_info_rail_metric"] = _default_lookup_option(option_keys, guide)
    selected_key = _clearable_selectbox(
        "Term",
        option_keys,
        format_func=lambda key: option_labels.get(str(key), str(key)),
        key="metric_info_rail_metric",
        placeholder="Search matching terms...",
    )
    if selected_key is None:
        st.info("Choose a matching term to inspect.")
        return
    selected_row = guide[guide["lookup_key"] == selected_key].iloc[0]
    detail = _lookup_detail(str(selected_row["kind"]), str(selected_row["term"]))
    if detail is None:
        return

    st.markdown(
        f"""
        <div class="metric-info-card">
            <p class="metric-info-card-label">{html.escape(str(selected_row["kind"]))} / {html.escape(detail.category)}</p>
            <h3>{html.escape(str(selected_row["term"]))}</h3>
            <p>{html.escape(detail.plain_english)}</p>
            <div class="metric-info-card-section">
                <span>Calculation</span>
                <p>{html.escape(detail.calculation)}</p>
            </div>
            <div class="metric-info-card-section">
                <span>How to read</span>
                <p>{html.escape(detail.how_to_read)}</p>
            </div>
            <div class="metric-info-card-section">
                <span>Watch out</span>
                <p>{html.escape(detail.caution)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Matching terms", expanded=False):
        st.dataframe(
            guide[["term", "kind", "category", "plain_english"]].head(20),
            width="stretch",
            hide_index=True,
        )


def _default_metric_option(metric_options: list[str]) -> str:
    for preferred in ("Ulcer Index", "Max Drawdown", "CAGR"):
        if preferred in metric_options:
            return preferred
    return metric_options[0]


def _lookup_categories() -> list[str]:
    categories = [*metric_categories(), *ticket_categories()]
    return list(dict.fromkeys(categories))


def _lookup_guide_frame(
    *,
    category: str | None = None,
    search: str = "",
) -> pd.DataFrame:
    metric_frame = metric_guide_frame(
        category=category if category in metric_categories() else None,
        search=search,
    ).rename(columns={"metric": "term"})
    if not metric_frame.empty:
        metric_frame = metric_frame.assign(kind="Metric")

    ticket_frame = ticket_guide_frame(
        category=category if category in ticket_categories() else None,
        search=search,
    )
    if category and category not in metric_categories() and category in ticket_categories():
        metric_frame = metric_frame.iloc[0:0]
    if category and category in metric_categories() and category not in ticket_categories():
        ticket_frame = ticket_frame.iloc[0:0]

    frame = pd.concat([ticket_frame, metric_frame], ignore_index=True, sort=False)
    if frame.empty:
        return frame
    frame = frame.assign(
        lookup_key=frame["kind"].astype(str).str.lower() + "::" + frame["term"].astype(str),
        lookup_label=frame["term"].astype(str) + " (" + frame["kind"].astype(str) + ")",
        match_rank=_lookup_match_rank(frame, search),
    )
    return (
        frame.sort_values(["match_rank", "kind", "term"], kind="stable")
        .drop(columns=["match_rank"])
        .reset_index(drop=True)
    )


def _lookup_detail(kind: str, term: str) -> Any:
    if kind.lower() == "metric":
        return metric_detail(term)
    return ticket_detail(term)


def _lookup_match_rank(frame: pd.DataFrame, search: str) -> pd.Series:
    query = search.strip().lower()
    if not query:
        return pd.Series(3, index=frame.index)
    terms = frame["term"].astype(str).str.lower()
    aliases = frame.get("aliases", pd.Series("", index=frame.index)).astype(str).str.lower()
    normalized_query = _normalize_lookup_key(query)
    normalized_terms = terms.map(_normalize_lookup_key)
    exact_term = (terms == query) | (normalized_terms == normalized_query)
    exact_alias = aliases.str.split(", ").apply(
        lambda values: query in values
        or normalized_query in {_normalize_lookup_key(value) for value in values}
    )
    contains_term = terms.str.contains(re.escape(query), na=False) | normalized_terms.str.contains(
        re.escape(normalized_query),
        na=False,
    )
    return pd.Series(
        [
            0 if exact else 1 if alias else 2 if contains else 3
            for exact, alias, contains in zip(
                exact_term,
                exact_alias,
                contains_term,
                strict=False,
            )
        ],
        index=frame.index,
    )


def _normalize_lookup_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("$", " dollar ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _default_lookup_option(option_keys: list[str], guide: pd.DataFrame) -> str:
    label_by_key = dict(zip(guide["lookup_key"], guide["term"], strict=False))
    for key in option_keys:
        if str(label_by_key.get(key, "")).lower() in {"ulcer index", "max drawdown", "cagr"}:
            return key
    return option_keys[0]


def _render_action_headline(headline: ActionHeadline) -> None:
    st.markdown(
        f"""
        <div class="action-banner action-{html.escape(headline.level)}">
            <p class="headline-label">Action Headline</p>
            <div class="headline-title">{html.escape(headline.headline)}</div>
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
        st.dataframe(headline.drivers, width="stretch")
