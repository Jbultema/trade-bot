from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.trends import (
    latest_per_market_date,
    load_snapshot_trend_frames,
    long_metric_line_figure,
)
from trade_bot.DEFAULTS import (
    DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT,
    DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT,
    DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.driver_rotation import (
    build_driver_rotation_table,
    summarize_driver_rotation,
)
from trade_bot.research.narrative_signals import (
    build_narrative_signal_table,
    summarize_narrative_signals,
)


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


def _render_news_and_macro(
    baseline_run: BaselineRun,
    *,
    run_store_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
) -> None:
    current_state = baseline_run.current_state
    regime_instability = getattr(current_state, "regime_instability", pd.DataFrame())
    regime_instability_components = getattr(
        current_state,
        "regime_instability_components",
        pd.DataFrame(),
    )
    narrative_signals = build_narrative_signal_table(
        baseline_run.prices,
        news_triage=baseline_run.news_monitor.triage,
        events=baseline_run.event_risk.events,
    )
    narrative_summary = summarize_narrative_signals(narrative_signals)

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

    st.subheader("Driver Rotation")
    driver_rotation = build_driver_rotation_table(
        baseline_run.prices,
        current_state,
        narrative_signals=narrative_signals,
        news_triage=baseline_run.news_monitor.triage,
    )
    driver_summary = summarize_driver_rotation(driver_rotation)
    driver_cols = st.columns(5)
    _helped_metric(
        driver_cols[0],
        "Normally Important",
        f"{int(driver_rotation['normally_important'].sum()):,}" if not driver_rotation.empty else "0",
    )
    _helped_metric(
        driver_cols[1],
        "Currently Active",
        f"{int(driver_rotation['currently_active'].sum()):,}" if not driver_rotation.empty else "0",
    )
    _helped_metric(
        driver_cols[2],
        "Emerging",
        f"{int(driver_rotation['emerging_importance'].sum()):,}" if not driver_rotation.empty else "0",
    )
    _helped_metric(
        driver_cols[3],
        "Fading",
        f"{int(driver_rotation['fading_importance'].sum()):,}" if not driver_rotation.empty else "0",
    )
    _helped_metric(driver_cols[4], "Driver Read", driver_summary["answer"])
    st.caption(
        f"{driver_summary['detail']} X-axis is historical relevance from ML diagnostics "
        "or fallback research priors; Y-axis is current activation. Color is model authority."
    )
    if driver_rotation.empty:
        st.write("No driver-rotation diagnostics are available.")
    else:
        quadrant_tab, heatmap_tab, table_tab = st.tabs(["Quadrant", "Heatmap", "Driver Table"])
        with quadrant_tab:
            st.plotly_chart(
                _driver_rotation_scatter_figure(driver_rotation),
                use_container_width=True,
            )
        with heatmap_tab:
            st.plotly_chart(
                _driver_rotation_heatmap_figure(driver_rotation),
                use_container_width=True,
            )
        with table_tab:
            display_columns = [
                "driver_label",
                "model_role",
                "primary_rotation_state",
                "proven_relevance",
                "current_activation",
                "previous_30d_activation",
                "previous_90d_activation",
                "change_30d",
                "change_90d",
                "data_support",
                "evidence",
                "interpretation",
            ]
            _render_metric_dataframe(
                _display_metrics(driver_rotation[display_columns]),
                hide_index=True,
            )
        _render_macro_driver_history(
            run_store_path=str(run_store_path),
            artifact_dir=str(artifact_dir),
            job_log_dir=str(job_log_dir),
        )

    st.subheader("Cross-Source Insight Diagnostics")
    signal_cols = st.columns(4)
    operating_signals = narrative_signals[
        narrative_signals["data_support"].isin(DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT)
    ]
    research_only_signals = narrative_signals[
        narrative_signals["data_support"].isin(DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT)
    ]
    unsupported_signals = narrative_signals[
        narrative_signals["data_support"].isin(DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT)
    ]
    active_count = int(operating_signals["status"].isin(["active", "warning"]).sum())
    unsupported_count = int(
        narrative_signals["data_support"].isin(DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT).sum()
    )
    _helped_metric(signal_cols[0], "Operating Context", f"{len(operating_signals):,}")
    _helped_metric(signal_cols[1], "Active Operating", f"{active_count:,}")
    _helped_metric(signal_cols[2], "Research-Only", f"{len(research_only_signals):,}")
    _helped_metric(signal_cols[3], "Sizing Authority", "None")
    st.caption(
        f"{narrative_summary['detail']} These rows synthesize recurring themes from the "
        "external sources into diagnostics. Direct/proxy rows are operating context; thin proxies "
        "and unsupported watchlists are research-only unless ablation evidence promotes them."
    )
    narrative_tabs = st.tabs(["Operating Context", "Research-Only Thin Proxies"])
    with narrative_tabs[0]:
        _render_metric_dataframe(_display_metrics(operating_signals), hide_index=True)
    with narrative_tabs[1]:
        st.caption(
            "Thin proxies can explain a narrative but are not treated as model drivers without "
            "marginal-contribution evidence."
        )
        _render_metric_dataframe(_display_metrics(research_only_signals), hide_index=True)
    with st.expander("Unsupported watchlist and full diagnostic audit", expanded=False):
        st.caption(
            f"{unsupported_count:,} watchlist gap(s): data we might want but do not currently have "
            "in a reliable local feed. These are intentionally hidden from the default operating "
            "surface because they are context gaps, not validated trading signals."
        )
        _render_metric_dataframe(_display_metrics(unsupported_signals), hide_index=True)
        st.caption("Full cross-source diagnostic table")
        _render_metric_dataframe(_display_metrics(narrative_signals), hide_index=True)

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
    with st.expander("News source coverage and health audit", expanded=False):
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

    with st.expander("Data quality audit", expanded=False):
        _render_metric_dataframe(_display_metrics(current_state.data_quality))


def _render_macro_driver_history(
    *,
    run_store_path: str,
    artifact_dir: str,
    job_log_dir: str,
) -> None:
    _metric_history, _component_history, scenario_driver_history, driver_rotation_history = (
        load_snapshot_trend_frames(run_store_path, artifact_dir, job_log_dir)
    )
    scenario_driver_history = latest_per_market_date(
        scenario_driver_history,
        subset=["driver"],
    )
    driver_rotation_history = latest_per_market_date(
        driver_rotation_history,
        subset=["driver"],
    )
    st.caption("Macro and scenario driver momentum from saved snapshots")
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            scenario_driver_history,
            category_column="driver",
            value_column="score",
            title="Scenario Driver Scores Over Time",
            yaxis_title="Score",
            top_n=7,
            height=300,
        )
        if figure.data:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No saved scenario-driver trend is available yet.")
    with cols[1]:
        figure = long_metric_line_figure(
            driver_rotation_history,
            category_column="driver_label",
            value_column="current_activation",
            title="Driver Rotation Activation Over Time",
            yaxis_title="Activation",
            top_n=7,
            height=300,
        )
        if figure.data:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No saved driver-rotation trend is available yet.")


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


def _driver_rotation_scatter_figure(rotation: pd.DataFrame) -> go.Figure:
    role_colors = {
        "allocation_driver": "#0f766e",
        "validated_context": "#b7791f",
        "explainer_only": "#4f46e5",
        "unsupported": "#9ca3af",
    }
    fig = go.Figure()
    for role, frame in rotation.groupby("model_role", sort=False):
        color = role_colors.get(str(role), "#64748b")
        fig.add_trace(
            go.Scatter(
                x=frame["proven_relevance"],
                y=frame["current_activation"],
                mode="markers",
                name=str(role).replace("_", " ").title(),
                text=frame["driver_label"],
                marker={
                    "size": 12 + frame["current_activation"].astype(float) * 18,
                    "color": color,
                    "opacity": 0.82,
                    "line": {"color": "white", "width": 1},
                },
                customdata=frame[
                    [
                        "primary_rotation_state",
                        "change_30d",
                        "change_90d",
                        "data_support",
                    ]
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Historical relevance: %{x:.0%}<br>"
                    "Current activation: %{y:.0%}<br>"
                    "State: %{customdata[0]}<br>"
                    "Change vs 30d: %{customdata[1]:+.0%}<br>"
                    "Change vs 90d: %{customdata[2]:+.0%}<br>"
                    "Support: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    label_rows = rotation.sort_values(
        ["current_activation", "proven_relevance"],
        ascending=False,
    ).reset_index(drop=True)
    for ordinal, row in label_rows.iterrows():
        label_style = _driver_rotation_label_style(row, ordinal)
        fig.add_annotation(
            x=float(row["proven_relevance"]),
            y=float(row["current_activation"]),
            text=str(row["driver_label"]),
            xref="x",
            yref="y",
            showarrow=False,
            xshift=label_style["xshift"],
            yshift=label_style["yshift"],
            xanchor=label_style["xanchor"],
            yanchor=label_style["yanchor"],
            align="left",
            font={"size": 12, "color": "#667085"},
            opacity=0.86,
        )
    for _, row in (
        rotation.dropna(subset=["previous_30d_activation"])
        .assign(abs_change=lambda frame: frame["change_30d"].abs())
        .sort_values("abs_change", ascending=False)
        .head(10)
        .iterrows()
    ):
        previous = float(row["previous_30d_activation"])
        current = float(row["current_activation"])
        if abs(current - previous) < 0.08:
            continue
        fig.add_annotation(
            x=float(row["proven_relevance"]),
            y=current,
            ax=float(row["proven_relevance"]),
            ay=previous,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=1.2,
            arrowcolor="#111827",
            opacity=0.55,
        )
    fig.add_vline(x=0.45, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=0.45, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="Driver Rotation: Historical Relevance vs Current Activation",
        xaxis_title="Proven historical relevance",
        yaxis_title="Current activation / pressure",
        xaxis={"range": [-0.06, 1.08], "tickformat": ".0%"},
        yaxis={"range": [-0.05, 1.08], "tickformat": ".0%"},
        template="plotly_white",
        legend_title_text="Model role",
        margin={"l": 24, "r": 28, "t": 62, "b": 24},
        height=520,
    )
    return fig


def _driver_rotation_label_style(row: pd.Series, ordinal: int) -> dict[str, int | str]:
    x = float(row.get("proven_relevance", 0.0))
    y = float(row.get("current_activation", 0.0))
    xshift = 12 if x < 0.55 else -12
    yshift = 16
    xanchor = "left" if x < 0.55 else "right"
    yanchor = "bottom"

    if x <= 0.08:
        xshift = 18
        xanchor = "left"
    elif x >= 0.92:
        xshift = -20
        xanchor = "right"

    if y >= 0.92:
        yshift = -22
        yanchor = "top"
    elif y <= 0.08:
        yshift = 22
        yanchor = "bottom"

    if y >= 0.75 and 0.25 <= x <= 0.50:
        xshift = 14 if ordinal % 2 == 0 else -14
        xanchor = "left" if xshift > 0 else "right"
        yshift = -22 - 8 * (ordinal % 3)
        yanchor = "top"
    elif y >= 0.75 and 0.50 < x < 0.82:
        xshift = 14 if ordinal % 2 == 0 else -14
        xanchor = "left" if xshift > 0 else "right"
        yshift = -18 - 8 * (ordinal % 2)
        yanchor = "top"

    return {
        "xshift": xshift,
        "yshift": yshift,
        "xanchor": xanchor,
        "yanchor": yanchor,
    }


def _driver_rotation_heatmap_figure(rotation: pd.DataFrame) -> go.Figure:
    view = rotation.sort_values(
        ["current_activation", "proven_relevance"],
        ascending=False,
    ).head(18)
    metrics = [
        "previous_90d_activation",
        "previous_30d_activation",
        "current_activation",
        "proven_relevance",
    ]
    z = view[metrics].astype(float).to_numpy()
    labels = [
        "90d proxy activation",
        "30d proxy activation",
        "Current activation",
        "Historical relevance",
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=view["driver_label"],
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.45, "#fef3c7"],
                [1.0, "#b91c1c"],
            ],
            colorbar={"title": "Score", "tickformat": ".0%"},
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Driver Activation Heatmap",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=max(420, 32 * len(view) + 120),
    )
    return fig
