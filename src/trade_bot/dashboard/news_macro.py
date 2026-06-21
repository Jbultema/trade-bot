from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.research.baselines import BaselineRun


def _dedupe_display_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return frame.copy()
    return frame[available_columns].drop_duplicates().reset_index(drop=True)


def _current_event_rollup(current_event_scenarios: pd.DataFrame) -> pd.DataFrame:
    if current_event_scenarios.empty or "event_id" not in current_event_scenarios:
        return pd.DataFrame()

    group_columns = [
        "event_id",
        "event_name",
        "event_date",
        "category",
        "direction",
        "event_phase",
        "confirmation_window",
    ]
    available_group_columns = [
        column for column in group_columns if column in current_event_scenarios.columns
    ]
    rollup = (
        current_event_scenarios.groupby("event_id", as_index=False, dropna=False)
        .agg(
            **{
                column: (column, "first")
                for column in available_group_columns
                if column != "event_id"
            },
            scenario_count=(
                ("scenario", "nunique")
                if "scenario" in current_event_scenarios
                else ("event_id", "size")
            ),
            scenarios=(
                ("scenario", _join_unique_values)
                if "scenario" in current_event_scenarios
                else ("event_id", "size")
            ),
            risk_postures=(
                ("risk_posture", _join_unique_values)
                if "risk_posture" in current_event_scenarios
                else ("event_id", "size")
            ),
        )
        .sort_values(["event_date", "event_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if "event_id" not in rollup and "event_id" in current_event_scenarios:
        rollup.insert(0, "event_id", current_event_scenarios["event_id"].drop_duplicates().values)
    return rollup


def _join_unique_values(values: pd.Series) -> str:
    unique_values = [str(value) for value in values.dropna().unique() if str(value)]
    return "; ".join(unique_values)


def _render_news_and_macro(baseline_run: BaselineRun) -> None:
    current_state = baseline_run.current_state
    regime_instability = getattr(current_state, "regime_instability", pd.DataFrame())
    regime_instability_components = getattr(
        current_state,
        "regime_instability_components",
        pd.DataFrame(),
    )

    st.subheader("Signal Coverage")
    coverage_cols = st.columns(3)
    _helped_metric(coverage_cols[0], "Market Proxies", f"{baseline_run.prices.shape[1]:,}")
    _helped_metric(coverage_cols[1], "Macro Configured", f"{len(baseline_run.macro_catalog):,}")
    _helped_metric(coverage_cols[2], "Macro Loaded", f"{baseline_run.macro_data.shape[1]:,}")
    st.dataframe(current_state.signal_coverage, use_container_width=True)

    st.subheader("Regime Pulse / Growth-Inflation Map")
    if current_state.regime_pulse_cycles.empty:
        st.write("No regime-pulse diagnostics are available.")
    else:
        lead_regime = (
            current_state.growth_inflation_map.iloc[0]
            if not current_state.growth_inflation_map.empty
            else pd.Series(dtype=object)
        )
        stocks = _asset_regime_pulse_row(current_state.regime_pulse_assets, "stocks")
        bonds = _asset_regime_pulse_row(current_state.regime_pulse_assets, "bonds")
        weather_cols = st.columns(4)
        _helped_metric(
            weather_cols[0],
            "Lead Regime",
            _lead_regime_label(lead_regime),
        )
        _helped_metric(
            weather_cols[1],
            "Stocks",
            str(stocks.get("regime_pulse_read", "missing")),
        )
        _helped_metric(
            weather_cols[2],
            "Bonds",
            str(bonds.get("regime_pulse_read", "missing")),
        )
        _helped_metric(
            weather_cols[3],
            "Cycle Count",
            f"{len(current_state.regime_pulse_cycles):,}",
        )
        cycle_tab, asset_tab, grid_tab, crowding_tab = st.tabs(
            ["Cycles", "Asset Reads", "Growth-Inflation Map", "Positioning / Crowding"]
        )
        with cycle_tab:
            _render_metric_dataframe(_display_metrics(current_state.regime_pulse_cycles))
        with asset_tab:
            _render_metric_dataframe(_display_metrics(current_state.regime_pulse_assets))
        with grid_tab:
            _render_metric_dataframe(_display_metrics(current_state.growth_inflation_map))
        with crowding_tab:
            if current_state.positioning_summary.empty:
                st.write("No positioning/crowding proxy diagnostics are available.")
            else:
                st.caption(
                    "Proxy layer: 3-month return z-score plus 14-day RSI. This is not yet true "
                    "ETF-flow, survey, or futures positioning data."
                )
                _render_metric_dataframe(_display_metrics(current_state.positioning_summary))
                with st.expander("Ticker-level crowding proxy", expanded=False):
                    _render_metric_dataframe(
                        _display_metrics(current_state.positioning_crowding.head(100))
                    )

    st.subheader("Regime Instability Index")
    if regime_instability.empty:
        st.write("No regime-instability diagnostics are available.")
    else:
        row = regime_instability.iloc[0]
        instability_cols = st.columns(4)
        _helped_metric(
            instability_cols[0],
            "Instability State",
            str(row.get("regime_instability_state", "n/a")).upper(),
        )
        _helped_metric(
            instability_cols[1],
            "Instability Score",
            f"{float(row.get('regime_instability_score', 0.0)):.2f}",
        )
        _helped_metric(
            instability_cols[2],
            "SPY +/-1% YTD",
            f"{float(row.get('spy_ytd_large_move_share', 0.0)):.1%}",
        )
        _helped_metric(instability_cols[3], "Trading Use", "Watch Only")
        st.caption(
            "Research signal only: useful for monitoring transition risk, not yet granted sizing authority."
        )
        _render_metric_dataframe(_display_metrics(regime_instability_components))

    st.subheader("Macro State")
    if current_state.macro_category_summary.empty:
        st.write("No macro diagnostics available.")
    else:
        _render_metric_dataframe(_display_metrics(current_state.macro_category_summary))
        macro_category_options = ["all", *sorted(current_state.macro_signals["category"].unique())]
        macro_category = st.selectbox("Macro category", macro_category_options)
        macro_near_term_options = [
            "all",
            *sorted(current_state.macro_signals["near_term_state"].unique()),
        ]
        macro_near_term = st.selectbox("Near-term macro state", macro_near_term_options)
        macro_signals = current_state.macro_signals
        if macro_category != "all":
            macro_signals = macro_signals[macro_signals["category"] == macro_category]
        if macro_near_term != "all":
            macro_signals = macro_signals[macro_signals["near_term_state"] == macro_near_term]
        _render_metric_dataframe(_display_metrics(macro_signals))

    st.subheader("News Intake Monitor")
    news_monitor = baseline_run.news_monitor
    source_coverage = getattr(news_monitor, "source_coverage", None)
    if source_coverage is not None and not source_coverage.empty:
        st.caption("Coverage audit for the narrative buckets we expect the monitor to watch.")
        _render_metric_dataframe(_display_metrics(source_coverage))

    if news_monitor.source_health.empty:
        st.write("No news sources are configured.")
    else:
        st.dataframe(news_monitor.source_health, use_container_width=True)

    if news_monitor.triage.empty:
        st.write("No recent news items were triaged.")
    else:
        activation_options = ["all", *sorted(news_monitor.triage["activation_status"].unique())]
        category_options = ["all", *sorted(news_monitor.triage["category"].unique())]
        news_activation = st.selectbox("News activation status", activation_options)
        news_category = st.selectbox("News category", category_options)
        news_view = news_monitor.triage.copy()
        if news_activation != "all":
            news_view = news_view[news_view["activation_status"] == news_activation]
        if news_category != "all":
            news_view = news_view[news_view["category"] == news_category]
        news_columns = [
            "title",
            "source",
            "published_at",
            "category",
            "direction",
            "phase",
            "urgency_score",
            "activation_status",
            "event_id",
            "risk_channels",
            "candidate_proxies",
            "confirmation_window",
            "url",
        ]
        news_display = _dedupe_display_rows(news_view, news_columns)
        _render_metric_dataframe(_display_metrics(news_display.head(100)))

    st.subheader("Event-Risk Monitor")
    event_risk = baseline_run.event_risk
    if event_risk.current_event_scenarios.empty:
        st.write("No current-event scenario playbook is configured.")
    else:
        st.caption(
            "Current event rollup: one row per event. Scenario rows below intentionally expand "
            "each event into multiple possible market paths."
        )
        current_event_rollup = _current_event_rollup(event_risk.current_event_scenarios)
        _render_metric_dataframe(_display_metrics(current_event_rollup), hide_index=True)
        with st.expander("Expanded current-event scenario playbook", expanded=False):
            scenario_columns = [
                "event_id",
                "event_name",
                "event_date",
                "category",
                "direction",
                "event_phase",
                "scenario",
                "confirmation",
                "risk_posture",
                "off_ramp",
                "confirmation_window",
            ]
            scenario_display = _dedupe_display_rows(
                event_risk.current_event_scenarios,
                scenario_columns,
            )
            _render_metric_dataframe(_display_metrics(scenario_display), hide_index=True)

    if event_risk.event_summary.empty:
        st.write("No historical event-window diagnostics are available.")
    else:
        st.caption(
            "Historical diagnostics repeat each event by window; use the window column to compare "
            "pre-event and post-event behavior."
        )
        event_window_filter = st.multiselect(
            "Event windows",
            sorted(event_risk.event_summary["window"].unique()),
            default=["post_5d", "post_21d"],
        )
        event_summary = event_risk.event_summary
        if event_window_filter:
            event_summary = event_summary[event_summary["window"].isin(event_window_filter)]
        _render_metric_dataframe(_display_metrics(event_summary))

    if not event_risk.strategy_event_returns.empty:
        complete_strategy_events = event_risk.strategy_event_returns[
            event_risk.strategy_event_returns["complete"]
        ]
        _render_metric_dataframe(_display_metrics(complete_strategy_events))

    st.subheader("Data Quality")
    _render_metric_dataframe(_display_metrics(current_state.data_quality))


def _asset_regime_pulse_row(assets: pd.DataFrame, asset_class: str) -> pd.Series:
    if assets.empty or "asset_class" not in assets:
        return pd.Series(dtype=object)
    rows = assets[assets["asset_class"] == asset_class]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _lead_regime_label(row: pd.Series) -> str:
    if row.empty:
        return "missing"
    regime = str(row.get("regime", "missing"))
    probability = row.get("probability")
    if pd.isna(probability):
        return regime
    return f"{regime} {float(probability):.0%}"
