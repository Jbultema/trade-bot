from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.DEFAULTS import DEFAULT_MONITORING_TOP_N, DEFAULT_RUN_STORE_DB_PATH
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

    detail_tab, top_tab, reference_tab, registry_tab, warehouse_tab = st.tabs(
        [
            "Monitoring Windows",
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
                selected_label = st.selectbox("Strategy", option_labels)
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
                    date.today(),
                    key="monitor_new_start_date",
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
            selected_window_label = st.selectbox("Active window", window_labels)
            selected_window_index = window_labels.index(selected_window_label)
            selected_window = windows.iloc[selected_window_index]
            current_role = str(selected_window.get("window_role", "challenger"))
            current_status = str(selected_window.get("status", "active"))
            current_capital = float(selected_window.get("capital_base", 10_000.0))
            cols = st.columns(4)
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
            demote_other = cols[3].checkbox(
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
