from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import (
    _clearable_selectbox,
    _helped_metric,
    _render_metric_dataframe,
)
from trade_bot.dashboard.formatting import _display_metrics, _display_trade_frame
from trade_bot.dashboard.trends import (
    filter_history_time_range,
    load_monitoring_trend_frame,
    long_metric_line_figure,
)
from trade_bot.DEFAULTS import (
    DEFAULT_MONITORING_COHORT_START_DATE,
    DEFAULT_MONITORING_ENVELOPE_BREACH_SHARE,
    DEFAULT_MONITORING_ENVELOPE_REVIEW_SHARE,
    DEFAULT_MONITORING_ENVELOPE_WATCH_SHARE,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_RUN_STORE_DB_PATH,
)
from trade_bot.research.factor_attribution import build_ticket_shortfall_audit
from trade_bot.storage.warehouse import TradingWarehouse


def _render_monitoring(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> None:
    st.subheader("Champion / Challenger Monitoring")
    st.caption(
        "Forward paper windows that separate the strategies we might operate from the broader research backlog."
    )
    frame, windows, counts, registry, top_candidates, reference_candidates = (
        _load_monitoring_frames(str(warehouse_path))
    )

    _render_monitoring_controls(
        str(warehouse_path),
        windows,
        registry,
        top_candidates,
        reference_candidates,
    )

    if windows.empty:
        st.info(
            "No monitoring windows have been seeded yet. Run `poetry run trade-bot "
            "migrate-warehouse`, then `poetry run trade-bot seed-monitoring-windows`, "
            "then `poetry run trade-bot run-paper-valuation`."
        )
        if not top_candidates.empty:
            st.caption(f"Top {len(top_candidates):,} ranked experiment candidates")
            _render_monitoring_candidates(top_candidates)
        if not reference_candidates.empty:
            st.caption("Reference portfolio policies")
            _render_monitoring_candidates(reference_candidates)
        if not counts.empty:
            st.caption("Warehouse table counts")
            st.dataframe(counts, width="stretch")
        return

    latest_valuation_rows = (
        int(frame["valuation_date"].notna().sum()) if "valuation_date" in frame else 0
    )
    start_cohorts = _monitoring_start_cohorts(windows)
    champions = int((windows["window_role"] == "champion").sum()) if "window_role" in windows else 0
    challengers = (
        int((windows["window_role"] == "challenger").sum()) if "window_role" in windows else 0
    )
    ahead = int((frame.get("forward_status", pd.Series(dtype=str)) == "ahead_of_benchmark").sum())

    references = (
        int((windows["window_role"] == "reference").sum()) if "window_role" in windows else 0
    )

    cols = st.columns(8)
    _helped_metric(cols[0], "Active Windows", f"{len(windows):,}")
    _helped_metric(cols[1], "Start Cohorts", f"{len(start_cohorts):,}")
    _helped_metric(cols[2], "Top Candidates", f"{len(top_candidates):,}")
    _helped_metric(cols[3], "References", f"{references:,}")
    _helped_metric(cols[4], "Champions", f"{champions:,}")
    _helped_metric(cols[5], "Challengers", f"{challengers:,}")
    _helped_metric(cols[6], "Valued Today", f"{latest_valuation_rows:,}")
    _helped_metric(cols[7], "Ahead", f"{ahead:,}")

    st.markdown("**Current operating readout**")
    selected_start_cohort = _monitoring_start_cohort_selector(start_cohorts)
    display_frame = _monitoring_display_frame(frame, start_cohort=selected_start_cohort)
    st.write(_monitoring_takeaway(display_frame, start_cohort=selected_start_cohort))
    leaderboard_columns = [
        "start_date",
        "monitoring_days",
        "window_role",
        "strategy_name",
        "mode",
        "account",
        "forward_status",
        "valuation_date",
        "equity",
        "cumulative_return",
        "benchmark_cumulative_return",
        "excess_return",
        "drawdown",
        "beta_adjusted_spy_delta",
        "stocks_percent_of_max_sleeve",
        "defensive_percent_of_max_sleeve",
        "gold_percent_of_max_sleeve",
        "crypto_percent_of_max_sleeve",
        "credit_percent_of_max_sleeve",
        "snapshot_cagr",
        "snapshot_calmar",
        "snapshot_max_drawdown",
        "promotion_decision",
        "promotion_score",
        "selection_adjusted_promotion_score",
        "overfit_risk_label",
        "validation_tier",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
    ]
    available_columns = [
        column for column in leaderboard_columns if column in display_frame.columns
    ]
    _render_metric_dataframe(_display_metrics(display_frame[available_columns]))
    _render_monitoring_forward_trends(str(warehouse_path), display_frame)

    detail_tab, shortfall_tab, top_tab, reference_tab, registry_tab, warehouse_tab = st.tabs(
        [
            "Monitoring Windows",
            "Shortfall / Drift",
            "Top Candidates",
            "Reference Portfolios",
            "Strategy Registry",
            "Warehouse Health",
        ]
    )
    with detail_tab:
        window_columns = [
            "window_role",
            "mode",
            "account",
            "strategy_name",
            "status",
            "start_date",
            "capital_base",
            "rebalance_cadence",
            "promotion_rule",
            "kill_rule",
            "notes",
        ]
        st.dataframe(
            windows[[column for column in window_columns if column in windows.columns]],
            width="stretch",
        )
    with shortfall_tab:
        _render_shortfall_and_execution_audit(str(warehouse_path), display_frame)
    with top_tab:
        st.caption(
            f"Curated top {DEFAULT_MONITORING_TOP_N} candidates from latest runtime snapshots "
            "and the experiment registry. The shelf anchors on score, then diversifies by strategy "
            "family and operating role; research-only rows are visible, but only snapshot-ready rows "
            "can receive daily paper valuations."
        )
        _render_monitoring_candidates(top_candidates)
    with reference_tab:
        st.caption(
            "Static policy portfolios with explicit sizing. These are comparison anchors, not tactical recommendations."
        )
        _render_monitoring_candidates(reference_candidates)
    with registry_tab:
        registry_columns = [
            "strategy_name",
            "role",
            "status",
            "family",
            "strategy_version",
            "source",
            "notes",
        ]
        if registry.empty:
            st.write("No strategy registry entries yet.")
        else:
            registry_view = registry[
                [column for column in registry_columns if column in registry.columns]
            ].rename(columns={"family": "category"})
            st.dataframe(registry_view, width="stretch")
    with warehouse_tab:
        st.dataframe(counts, width="stretch")


def _render_shortfall_and_execution_audit(warehouse_path: str, frame: pd.DataFrame) -> None:
    st.caption(
        "Implementation shortfall checks whether the forward paper/live process actually followed "
        "the recommendations. V1 uses recommendation-ticket and execution compliance; actual-vs-ideal "
        "account-equity attribution can be added once daily account valuation rows are logged."
    )
    warehouse = TradingWarehouse(warehouse_path)
    tickets = warehouse.read_table("journal_recommendation_tickets")
    executions = warehouse.read_table("journal_executions")
    audit = build_ticket_shortfall_audit(tickets, executions)

    cols = st.columns(5)
    _helped_metric(cols[0], "Tickets", f"{len(audit):,}")
    _helped_metric(
        cols[1],
        "Executed",
        f"{int((audit.get('execution_status', pd.Series(dtype=str)) == 'executed').sum()):,}",
    )
    _helped_metric(
        cols[2],
        "Unexecuted",
        f"{int((audit.get('execution_status', pd.Series(dtype=str)) == 'not_executed').sum()):,}",
    )
    _helped_metric(
        cols[3],
        "Price Band Breaks",
        f"{_false_count(audit, 'inside_price_band'):,}",
    )
    _helped_metric(
        cols[4],
        "Size Band Breaks",
        f"{_false_count(audit, 'inside_size_band'):,}",
    )

    if audit.empty:
        st.write("No recommendation tickets have been migrated into the warehouse yet.")
    else:
        st.caption("Ticket/execution shortfall audit")
        _render_metric_dataframe(_display_trade_frame(audit), hide_index=True)

    if not frame.empty:
        st.caption("Latest ideal monitoring valuation state")
        valuation_columns = [
            "start_date",
            "monitoring_days",
            "window_role",
            "strategy_name",
            "valuation_date",
            "equity",
            "cumulative_return",
            "benchmark_cumulative_return",
            "excess_return",
            "drawdown",
            "beta_adjusted_spy_delta",
            "stocks_percent_of_max_sleeve",
            "defensive_percent_of_max_sleeve",
            "gold_percent_of_max_sleeve",
            "crypto_percent_of_max_sleeve",
            "credit_percent_of_max_sleeve",
            "forward_status",
        ]
        available = [column for column in valuation_columns if column in frame.columns]
        if available:
            _render_metric_dataframe(_display_metrics(frame[available]), hide_index=True)
        _render_monitoring_drift_envelope(frame)


def _render_monitoring_forward_trends(warehouse_path: str, display_frame: pd.DataFrame) -> None:
    history = load_monitoring_trend_frame(warehouse_path)
    if history.empty:
        return
    if not display_frame.empty and "window_id" in display_frame and "window_id" in history:
        window_ids = set(display_frame["window_id"].dropna().astype(str))
        if window_ids:
            history = history[history["window_id"].astype(str).isin(window_ids)].copy()
    if history.empty:
        return
    st.caption("Champion/challenger forward trends")
    history = _render_monitoring_trend_range_controls(history, key_prefix="legacy_monitoring")
    if history.empty:
        st.info("No monitoring valuation rows are available for the selected time range.")
        return
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            history,
            category_column="window_label",
            value_column="excess_return",
            title="Excess Return Since Monitoring Start",
            yaxis_title="Excess return",
            percent=True,
            top_n=8,
            height=300,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    with cols[1]:
        figure = long_metric_line_figure(
            history,
            category_column="window_label",
            value_column="drawdown",
            title="Forward Drawdown",
            yaxis_title="Drawdown",
            percent=True,
            top_n=8,
            height=300,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            history,
            category_column="window_label",
            value_column="drawdown_envelope_used",
            title="Drawdown Envelope Used",
            yaxis_title="Envelope used",
            percent=True,
            top_n=8,
            height=280,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    with cols[1]:
        figure = long_metric_line_figure(
            history,
            category_column="window_label",
            value_column="beta_adjusted_spy_delta",
            title="Beta-Adjusted S&P Delta",
            yaxis_title="Delta",
            percent=True,
            top_n=8,
            height=280,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")


def _render_monitoring_trend_range_controls(
    history: pd.DataFrame,
    *,
    key_prefix: str,
) -> pd.DataFrame:
    control_cols = st.columns([1, 1, 1, 2])
    range_choice = control_cols[0].selectbox(
        "Trend time range",
        ["1M", "3M", "6M", "YTD", "1Y", "All", "Custom"],
        index=5,
        key=f"{key_prefix}_trend_range",
        help="Filter the monitoring valuation history before drawing forward trend charts.",
    )
    custom_start = None
    custom_end = None
    if range_choice == "Custom":
        custom_start = control_cols[1].date_input(
            "Trend start",
            value=date.today() - timedelta(days=92),
            key=f"{key_prefix}_trend_start",
        )
        custom_end = control_cols[2].date_input(
            "Trend end",
            value=date.today(),
            key=f"{key_prefix}_trend_end",
        )
    filtered = filter_history_time_range(
        history,
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    window_count = filtered["window_label"].nunique() if "window_label" in filtered else 0
    control_cols[3].caption(
        f"Showing {len(filtered):,} valuation rows across {int(window_count):,} windows."
    )
    return filtered


def _monitoring_start_cohorts(windows: pd.DataFrame) -> list[str]:
    if windows.empty or "start_date" not in windows:
        return []
    values = (
        pd.to_datetime(windows["start_date"], errors="coerce")
        .dropna()
        .dt.date.astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [str(value) for value in values]


def _monitoring_start_cohort_selector(start_cohorts: list[str]) -> str:
    if not start_cohorts:
        return "All starts"
    options = ["All starts", *start_cohorts]
    return (
        st.pills(
            "Monitoring start cohort",
            options,
            selection_mode="single",
            default="All starts",
            key="monitoring_start_cohort",
            help=(
                "Filter the operating readout by monitoring start date. Use this to compare "
                "YTD paper starts against newer starts after missed rally windows."
            ),
            width="stretch",
        )
        or "All starts"
    )


def _monitoring_display_frame(frame: pd.DataFrame, *, start_cohort: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    if "start_date" in output:
        start_dates = pd.to_datetime(output["start_date"], errors="coerce")
        output["start_date"] = start_dates.dt.date.astype("string").fillna("")
        if start_cohort != "All starts":
            output = output[output["start_date"].astype(str) == start_cohort].copy()
        valuation_dates = (
            pd.to_datetime(output["valuation_date"], errors="coerce")
            if "valuation_date" in output
            else pd.Series(pd.NaT, index=output.index)
        )
        fallback_end = pd.Timestamp(date.today())
        elapsed_days = (
            valuation_dates.fillna(fallback_end) - start_dates.reindex(output.index)
        ).dt.days
        output["monitoring_days"] = elapsed_days.where(elapsed_days >= 0)
    else:
        output["monitoring_days"] = pd.NA
    if output.empty:
        return output
    role_order = output.get("window_role", pd.Series("", index=output.index)).map(
        {"champion": 0, "challenger": 1, "reference": 2}
    )
    output["_role_order"] = role_order.fillna(3)
    sort_columns = [
        column
        for column in [
            "_role_order",
            "strategy_name",
            "start_date",
            "account",
        ]
        if column in output
    ]
    if sort_columns:
        output = output.sort_values(
            sort_columns,
            ascending=[True, True, False, True][: len(sort_columns)],
            na_position="last",
        )
    return output.drop(columns=["_role_order"], errors="ignore")


def _false_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    scoped = frame
    if "execution_status" in scoped:
        scoped = scoped[scoped["execution_status"].astype(str) == "executed"]
    if scoped.empty:
        return 0
    return int((~scoped[column].fillna(False).astype(bool)).sum())


def _render_monitoring_drift_envelope(frame: pd.DataFrame) -> None:
    envelope = _monitoring_drift_envelope_frame(frame)
    if envelope.empty:
        return
    st.caption("Live drift versus backtest envelope")
    status_counts = envelope["envelope_status"].value_counts().to_dict()
    cols = st.columns(4)
    _helped_metric(cols[0], "Inside", f"{int(status_counts.get('inside', 0)):,}")
    _helped_metric(cols[1], "Watch", f"{int(status_counts.get('watch', 0)):,}")
    _helped_metric(cols[2], "Review", f"{int(status_counts.get('review', 0)):,}")
    _helped_metric(cols[3], "Breach", f"{int(status_counts.get('breach', 0)):,}")
    st.plotly_chart(_monitoring_drift_envelope_figure(envelope), width="stretch")
    _render_metric_dataframe(_display_metrics(envelope), hide_index=True)


def _monitoring_drift_envelope_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = {"strategy_name", "window_role", "drawdown", "snapshot_max_drawdown"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows = []
    for _, row in frame.iterrows():
        drawdown_value = _safe_float(row.get("drawdown"))
        snapshot_drawdown = _safe_float(row.get("snapshot_max_drawdown"))
        envelope_used = _drawdown_envelope_used(drawdown_value, snapshot_drawdown)
        rows.append(
            {
                "window_role": row.get("window_role", ""),
                "start_date": row.get("start_date", ""),
                "monitoring_days": row.get("monitoring_days", None),
                "strategy_name": row.get("strategy_name", ""),
                "forward_status": row.get("forward_status", ""),
                "valuation_date": row.get("valuation_date", ""),
                "current_drawdown": drawdown_value,
                "backtest_max_drawdown": snapshot_drawdown,
                "drawdown_envelope_used": envelope_used,
                "envelope_status": _monitoring_envelope_status(envelope_used),
                "beta_adjusted_spy_delta": _safe_float(row.get("beta_adjusted_spy_delta")),
                "stocks_percent_of_max_sleeve": _safe_float(
                    row.get("stocks_percent_of_max_sleeve")
                ),
                "defensive_percent_of_max_sleeve": _safe_float(
                    row.get("defensive_percent_of_max_sleeve")
                ),
                "read": _monitoring_envelope_read(envelope_used),
            }
        )
    output = pd.DataFrame(rows)
    status_order = {"breach": 0, "review": 1, "watch": 2, "inside": 3, "no_snapshot": 4}
    output["_status_order"] = output["envelope_status"].map(status_order).fillna(5)
    return output.sort_values(
        ["_status_order", "strategy_name", "start_date", "drawdown_envelope_used"],
        ascending=[True, True, False, False],
    ).drop(
        columns=["_status_order"],
    )


def _monitoring_drift_envelope_figure(envelope: pd.DataFrame) -> go.Figure:
    status_colors = {
        "inside": "#0f766e",
        "watch": "#b7791f",
        "review": "#ea580c",
        "breach": "#b91c1c",
        "no_snapshot": "#64748b",
    }
    plot_frame = envelope.copy()
    plot_frame["drawdown_envelope_used"] = pd.to_numeric(
        plot_frame["drawdown_envelope_used"],
        errors="coerce",
    )
    plot_frame["window_label"] = (
        plot_frame["strategy_name"].astype(str) + " | " + plot_frame["start_date"].astype(str)
    )
    figure = go.Figure(
        go.Bar(
            x=plot_frame["window_label"],
            y=plot_frame["drawdown_envelope_used"],
            marker_color=plot_frame["envelope_status"].map(status_colors).fillna("#64748b"),
            customdata=plot_frame[
                [
                    "window_role",
                    "start_date",
                    "monitoring_days",
                    "envelope_status",
                    "current_drawdown",
                    "backtest_max_drawdown",
                    "forward_status",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>Role: %{customdata[0]}"
                "<br>Start: %{customdata[1]}"
                "<br>Days monitored: %{customdata[2]}"
                "<br>Status: %{customdata[3]}"
                "<br>Envelope used: %{y:.1%}"
                "<br>Current drawdown: %{customdata[4]:.1%}"
                "<br>Backtest max drawdown: %{customdata[5]:.1%}"
                "<br>Forward status: %{customdata[6]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=DEFAULT_MONITORING_ENVELOPE_WATCH_SHARE,
        line_dash="dash",
        line_color="#b7791f",
        annotation_text="watch",
        annotation_position="top left",
    )
    figure.add_hline(
        y=DEFAULT_MONITORING_ENVELOPE_REVIEW_SHARE,
        line_dash="dash",
        line_color="#ea580c",
        annotation_text="review",
        annotation_position="top left",
    )
    figure.add_hline(
        y=DEFAULT_MONITORING_ENVELOPE_BREACH_SHARE,
        line_dash="dot",
        line_color="#b91c1c",
        annotation_text="breach",
        annotation_position="top left",
    )
    figure.update_layout(
        title="Forward Drawdown Used Versus Backtest Max Drawdown",
        template="plotly_white",
        yaxis={"title": "Drawdown envelope used", "tickformat": ".0%", "range": [0, 1.15]},
        xaxis={"title": "", "tickangle": -25},
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 100},
        showlegend=False,
    )
    return figure


def _drawdown_envelope_used(
    current_drawdown: float | None,
    backtest_max_drawdown: float | None,
) -> float | None:
    if current_drawdown is None or backtest_max_drawdown is None:
        return None
    denominator = abs(float(backtest_max_drawdown))
    if denominator <= 0:
        return None
    return abs(float(current_drawdown)) / denominator


def _monitoring_envelope_status(envelope_used: float | None) -> str:
    if envelope_used is None or pd.isna(envelope_used):
        return "no_snapshot"
    if envelope_used >= DEFAULT_MONITORING_ENVELOPE_BREACH_SHARE:
        return "breach"
    if envelope_used >= DEFAULT_MONITORING_ENVELOPE_REVIEW_SHARE:
        return "review"
    if envelope_used >= DEFAULT_MONITORING_ENVELOPE_WATCH_SHARE:
        return "watch"
    return "inside"


def _monitoring_envelope_read(envelope_used: float | None) -> str:
    status = _monitoring_envelope_status(envelope_used)
    if status == "breach":
        return (
            "Forward drawdown has exceeded the historical max-drawdown envelope; review or pause."
        )
    if status == "review":
        return (
            "Forward drawdown is close to the historical envelope; inspect attribution and trades."
        )
    if status == "watch":
        return "Forward drawdown is material but still within the tested envelope."
    if status == "inside":
        return "Forward drawdown is comfortably inside the tested envelope."
    return "No historical max-drawdown snapshot is available for this row."


def _safe_float(value: object) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(output):
        return None
    return output


def _render_monitoring_controls(
    warehouse_path: str,
    windows: pd.DataFrame,
    registry: pd.DataFrame,
    top_candidates: pd.DataFrame,
    reference_candidates: pd.DataFrame,
) -> None:
    with st.expander("Monitoring Controls", expanded=False):
        st.write(
            "Use this to start paper monitoring for a candidate, promote/demote roles, "
            "pause/close windows, or run separate champion windows with different account labels."
        )
        start_tab, manage_tab = st.tabs(["Start Monitoring", "Manage Active Windows"])
        with start_tab:
            candidate_group = st.radio(
                "Candidate set",
                ["Top candidates", "Reference portfolios", "All registry"],
                horizontal=True,
            )
            candidates = _candidate_control_frame(
                candidate_group,
                registry,
                top_candidates,
                reference_candidates,
            )
            if candidates.empty:
                st.write("No candidates available. Run `poetry run trade-bot migrate-warehouse`.")
            else:
                option_labels = [_candidate_label(row) for _, row in candidates.iterrows()]
                selected_label = _clearable_selectbox(
                    "Strategy",
                    option_labels,
                    key="monitor_start_strategy",
                    placeholder="Search strategies to monitor...",
                )
                if selected_label is None:
                    st.info("Choose a strategy to start or update monitoring.")
                else:
                    selected_index = option_labels.index(selected_label)
                    selected = candidates.iloc[selected_index]
                    selected_summary = (
                        selected[
                            [
                                column
                                for column in [
                                    "strategy_name",
                                    "status",
                                    "family",
                                    "source",
                                    "monitoring_state",
                                    "snapshot_valuation_ready",
                                    "promotion_score",
                                    "snapshot_cagr",
                                    "snapshot_calmar",
                                    "snapshot_max_drawdown",
                                    "calmar",
                                    "max_drawdown",
                                    "notes",
                                ]
                                if column in selected.index
                            ]
                        ]
                        .to_frame("selected")
                        .T
                    )
                    st.dataframe(_display_metrics(selected_summary), width="stretch")

                    cols = st.columns(5)
                    role = cols[0].selectbox(
                        "Role",
                        ["challenger", "champion", "reference"],
                        index=0,
                        key="monitor_new_role",
                    )
                    mode = cols[1].selectbox("Mode", ["paper", "live"], key="monitor_new_mode")
                    account = cols[2].text_input(
                        "Account label",
                        "default_paper_account",
                        key="monitor_new_account",
                    )
                    capital_base = cols[3].number_input(
                        "Paper capital",
                        min_value=1.0,
                        value=10_000.0,
                        step=1_000.0,
                        key="monitor_new_capital",
                    )
                    start_date = cols[4].date_input(
                        "Start date",
                        date.fromisoformat(DEFAULT_MONITORING_COHORT_START_DATE),
                        key="monitor_new_start_date",
                        help=(
                            "Use a shared cohort start for fair YTD comparisons, or choose a "
                            "strategy-specific adoption date when that is the research question."
                        ),
                    )
                    demote_other = st.checkbox(
                        "Make this the only active champion for this mode/account",
                        value=False,
                        key="monitor_new_demote_other",
                        help=(
                            "Leave unchecked if you intentionally want multiple champion windows, "
                            "for example different strategy sizes or account sleeves."
                        ),
                    )
                    if st.button("Start / Update Monitoring", type="primary"):
                        warehouse = TradingWarehouse(warehouse_path)
                        try:
                            result = warehouse.monitor_strategy(
                                str(selected["strategy_name"]),
                                role=role,
                                mode=mode,
                                account=account,
                                capital_base=float(capital_base),
                                start_date=start_date.isoformat(),
                                demote_other_champions=bool(demote_other),
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            _load_monitoring_frames.clear()
                            st.success(
                                f"Monitoring {result.strategy_name} as {result.role} "
                                f"with ${float(capital_base):,.0f}."
                            )
                            st.rerun()

        with manage_tab:
            if windows.empty:
                st.write("No active windows to manage yet.")
                return
            window_labels = [_window_label(row) for _, row in windows.iterrows()]
            selected_window_label = _clearable_selectbox(
                "Active window",
                window_labels,
                key="monitor_active_window",
                placeholder="Search active windows...",
            )
            if selected_window_label is None:
                st.info("Choose an active window to manage it.")
                return
            selected_window_index = window_labels.index(selected_window_label)
            selected_window = windows.iloc[selected_window_index]
            current_role = str(selected_window.get("window_role", "challenger"))
            current_status = str(selected_window.get("status", "active"))
            current_capital = float(selected_window.get("capital_base", 10_000.0))
            current_start = pd.to_datetime(
                selected_window.get("start_date", DEFAULT_MONITORING_COHORT_START_DATE),
                errors="coerce",
            )
            if pd.isna(current_start):
                current_start_date = date.fromisoformat(DEFAULT_MONITORING_COHORT_START_DATE)
            else:
                current_start_date = current_start.date()
            cols = st.columns(5)
            role_options = ["champion", "challenger", "reference"]
            status_options = ["active", "paused", "closed", "killed", "archived"]
            next_role = cols[0].selectbox(
                "Role",
                role_options,
                index=role_options.index(current_role) if current_role in role_options else 1,
                key="monitor_manage_role",
            )
            next_status = cols[1].selectbox(
                "Status",
                status_options,
                index=(
                    status_options.index(current_status) if current_status in status_options else 0
                ),
                key="monitor_manage_status",
            )
            next_capital = cols[2].number_input(
                "Paper capital",
                min_value=1.0,
                value=current_capital,
                step=1_000.0,
                key="monitor_manage_capital",
            )
            next_start_date = cols[3].date_input(
                "Start date",
                value=current_start_date,
                key="monitor_manage_start_date",
                help="Changing this clears stale paper valuation rows for the selected window.",
            )
            demote_other = cols[4].checkbox(
                "Only champion",
                value=False,
                key="monitor_manage_demote_other",
                help="If checked while role is champion, demote other active champions for the same mode/account.",
            )
            if st.button("Apply Window Changes"):
                warehouse = TradingWarehouse(warehouse_path)
                try:
                    updated = warehouse.update_monitoring_window(
                        str(selected_window["window_id"]),
                        role=next_role,
                        status=next_status,
                        capital_base=float(next_capital),
                        start_date=next_start_date.isoformat(),
                        demote_other_champions=bool(demote_other),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if updated:
                        _load_monitoring_frames.clear()
                        st.success("Monitoring window updated.")
                        st.rerun()
                    else:
                        st.error("Monitoring window was not found.")


def _candidate_control_frame(
    candidate_group: str,
    registry: pd.DataFrame,
    top_candidates: pd.DataFrame,
    reference_candidates: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_group in {"Top candidates", "Top experiments"}:
        frame = top_candidates.copy()
    elif candidate_group == "Reference portfolios":
        frame = reference_candidates.copy()
    else:
        frame = registry.copy()
        if not frame.empty:
            frame["monitoring_state"] = "registry_only"
            frame["snapshot_valuation_ready"] = False
    if frame.empty:
        return frame
    if "family" not in frame and "category" in frame:
        frame = frame.rename(columns={"category": "family"})
    sort_columns = [column for column in ["rank", "strategy_name"] if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns, na_position="last")
    return frame.drop_duplicates("strategy_name", keep="first").reset_index(drop=True)


def _candidate_label(row: pd.Series) -> str:
    strategy = str(row.get("strategy_name", ""))
    status = str(row.get("status", ""))
    family = str(row.get("family", ""))
    state = str(row.get("monitoring_state", ""))
    parts = [strategy]
    detail = ", ".join(part for part in [status, family, state] if part and part != "nan")
    if detail:
        parts.append(f"({detail})")
    return " ".join(parts)


def _window_label(row: pd.Series) -> str:
    strategy = str(row.get("strategy_name", ""))
    role = str(row.get("window_role", ""))
    status = str(row.get("status", ""))
    account = str(row.get("account", ""))
    start_date = str(row.get("start_date", ""))
    capital = float(row.get("capital_base", 0.0))
    return f"{strategy} | start {start_date} | {role} | {status} | {account} | ${capital:,.0f}"


@st.cache_data(show_spinner=False, ttl=60)
def _load_monitoring_frames(
    warehouse_path: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    warehouse = TradingWarehouse(warehouse_path)
    return (
        warehouse.champion_challenger_frame(),
        warehouse.list_monitoring_windows(status="active"),
        warehouse.table_counts(),
        warehouse.list_strategy_registry(),
        warehouse.top_monitoring_candidates(limit=DEFAULT_MONITORING_TOP_N),
        warehouse.reference_monitoring_candidates(),
    )


def _render_monitoring_candidates(top_candidates: pd.DataFrame) -> None:
    if top_candidates.empty:
        st.write(
            "No ranked monitoring candidates are available. Run `poetry run trade-bot migrate-warehouse`."
        )
        return
    columns = [
        "rank",
        "strategy_name",
        "curation_bucket",
        "curation_reason",
        "monitoring_state",
        "window_role",
        "window_status",
        "snapshot_valuation_ready",
        "promotion_decision",
        "promotion_score",
        "selection_adjusted_promotion_score",
        "snapshot_cagr",
        "snapshot_calmar",
        "snapshot_max_drawdown",
        "snapshot_average_turnover",
        "validation_tier",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "cagr",
        "max_drawdown",
        "calmar",
        "average_turnover",
        "source",
        "family",
        "notes",
    ]
    view = top_candidates[
        [column for column in columns if column in top_candidates.columns]
    ].rename(columns={"family": "category"})
    _render_metric_dataframe(_display_metrics(view))


def _monitoring_takeaway(frame: pd.DataFrame, *, start_cohort: str = "All starts") -> str:
    cohort = "" if start_cohort == "All starts" else f" for the {start_cohort} start cohort"
    if frame.empty:
        return (
            f"Monitoring windows exist, but no champion/challenger rows are available{cohort} yet."
        )
    valued = frame[frame.get("valuation_date", pd.Series(dtype=object)).notna()]
    if valued.empty:
        return (
            f"The windows{cohort} are seeded, but they have not been valued yet. Run the daily paper "
            "valuation command after each snapshot refresh so forward returns start accumulating."
        )
    ahead = int((valued["forward_status"] == "ahead_of_benchmark").sum())
    lagging = int((valued["forward_status"] == "lagging_benchmark").sum())
    drawdown = int((valued["forward_status"] == "review_drawdown").sum())
    if drawdown:
        return f"{drawdown} monitored strategy{cohort} is in drawdown review; do not promote until the failure mode is understood."
    if ahead and not lagging:
        return f"{ahead} monitored strategy{cohort} is ahead of benchmark on the current paper window; keep collecting forward evidence before promotion."
    if lagging:
        return f"{lagging} monitored strategy{cohort} is lagging benchmark; treat the current champion/challenger set as under review."
    return f"The monitored set{cohort} is broadly in line with benchmark; no promotion or kill decision is obvious from forward valuation alone."
