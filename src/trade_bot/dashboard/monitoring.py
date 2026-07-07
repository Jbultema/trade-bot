from __future__ import annotations

from datetime import date
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
            st.dataframe(counts, use_container_width=True)
        return

    latest_valuation_rows = (
        int(frame["valuation_date"].notna().sum()) if "valuation_date" in frame else 0
    )
    champions = int((windows["window_role"] == "champion").sum()) if "window_role" in windows else 0
    challengers = (
        int((windows["window_role"] == "challenger").sum()) if "window_role" in windows else 0
    )
    ahead = int((frame.get("forward_status", pd.Series(dtype=str)) == "ahead_of_benchmark").sum())

    references = (
        int((windows["window_role"] == "reference").sum()) if "window_role" in windows else 0
    )

    cols = st.columns(7)
    _helped_metric(cols[0], "Active Windows", f"{len(windows):,}")
    _helped_metric(cols[1], "Top Candidates", f"{len(top_candidates):,}")
    _helped_metric(cols[2], "References", f"{references:,}")
    _helped_metric(cols[3], "Champions", f"{champions:,}")
    _helped_metric(cols[4], "Challengers", f"{challengers:,}")
    _helped_metric(cols[5], "Valued Today", f"{latest_valuation_rows:,}")
    _helped_metric(cols[6], "Ahead", f"{ahead:,}")

    st.markdown("**Current operating readout**")
    st.write(_monitoring_takeaway(frame))
    leaderboard_columns = [
        "window_role",
        "strategy_name",
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
    available_columns = [column for column in leaderboard_columns if column in frame.columns]
    _render_metric_dataframe(_display_metrics(frame[available_columns]))

    detail_tab, shortfall_tab, top_tab, reference_tab, registry_tab, warehouse_tab = st.tabs(
        [
            "Monitoring Windows",
            "Shortfall / Drift",
            "Top Experiments",
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
            use_container_width=True,
        )
    with shortfall_tab:
        _render_shortfall_and_execution_audit(str(warehouse_path), frame)
    with top_tab:
        st.caption(
            f"Curated top {DEFAULT_MONITORING_TOP_N} candidates from the experiment registry. "
            "The shelf anchors on score, then diversifies by strategy family and operating role; "
            "research-only rows are visible, but only snapshot-ready rows can receive daily paper valuations."
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
            st.dataframe(registry_view, use_container_width=True)
    with warehouse_tab:
        st.dataframe(counts, use_container_width=True)


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
    st.plotly_chart(_monitoring_drift_envelope_figure(envelope), use_container_width=True)
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
    return output.sort_values(["_status_order", "drawdown_envelope_used"], ascending=[True, False]).drop(
        columns=["_status_order"]
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
    figure = go.Figure(
        go.Bar(
            x=plot_frame["strategy_name"],
            y=plot_frame["drawdown_envelope_used"],
            marker_color=plot_frame["envelope_status"].map(status_colors).fillna("#64748b"),
            customdata=plot_frame[
                [
                    "window_role",
                    "envelope_status",
                    "current_drawdown",
                    "backtest_max_drawdown",
                    "forward_status",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>Role: %{customdata[0]}"
                "<br>Status: %{customdata[1]}"
                "<br>Envelope used: %{y:.1%}"
                "<br>Current drawdown: %{customdata[2]:.1%}"
                "<br>Backtest max drawdown: %{customdata[3]:.1%}"
                "<br>Forward status: %{customdata[4]}<extra></extra>"
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
        return "Forward drawdown has exceeded the historical max-drawdown envelope; review or pause."
    if status == "review":
        return "Forward drawdown is close to the historical envelope; inspect attribution and trades."
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
            "Use this to start paper monitoring for an experiment, promote/demote roles, "
            "pause/close windows, or run separate champion windows with different account labels."
        )
        start_tab, manage_tab = st.tabs(["Start Monitoring", "Manage Active Windows"])
        with start_tab:
            candidate_group = st.radio(
                "Candidate set",
                ["Top experiments", "Reference portfolios", "All registry"],
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
                    st.dataframe(_display_metrics(selected_summary), use_container_width=True)

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
    if candidate_group == "Top experiments":
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
    capital = float(row.get("capital_base", 0.0))
    return f"{strategy} | {role} | {status} | {account} | ${capital:,.0f}"


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
            "No ranked experiment candidates are available. Run `poetry run trade-bot migrate-warehouse`."
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


def _monitoring_takeaway(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Monitoring windows exist, but no champion/challenger rows are available yet."
    valued = frame[frame.get("valuation_date", pd.Series(dtype=object)).notna()]
    if valued.empty:
        return (
            "The windows are seeded, but they have not been valued yet. Run the daily paper "
            "valuation command after each snapshot refresh so forward returns start accumulating."
        )
    ahead = int((valued["forward_status"] == "ahead_of_benchmark").sum())
    lagging = int((valued["forward_status"] == "lagging_benchmark").sum())
    drawdown = int((valued["forward_status"] == "review_drawdown").sum())
    if drawdown:
        return f"{drawdown} monitored strategy is in drawdown review; do not promote until the failure mode is understood."
    if ahead and not lagging:
        return f"{ahead} monitored strategy is ahead of benchmark on the current paper window; keep collecting forward evidence before promotion."
    if lagging:
        return f"{lagging} monitored strategy is lagging benchmark; treat the current champion/challenger set as under review."
    return "The monitored set is broadly in line with benchmark; no promotion or kill decision is obvious from forward valuation alone."
