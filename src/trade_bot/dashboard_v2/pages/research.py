from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _clearable_selectbox, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.research_lab import (
    _OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY,
    _render_approach_detail_workbench,
    _render_outcome_frontier,
    _render_research_lab,
)
from trade_bot.dashboard_v2.components.cards import render_callout, render_card_grid
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.artifact_service import leadership_frames, pbo_frames
from trade_bot.dashboard_v2.services.experiment_service import (
    dashboard_frames,
    scorecards,
    top_scorecards,
)
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime


def render_research_page(runtime: DashboardRuntime) -> None:
    with timed("research.scorecards"):
        scores = scorecards()
    top = top_scorecards(scores, limit=30)
    render_card_grid(
        [
            ("Candidates", len(scores)),
            ("Displayed", len(top)),
            ("Champion CAGR", _best_metric(top, "cagr")),
            ("Best Utility", _best_metric(top, "growth_constrained_utility_score")),
            ("Paper-Ready", _count_label(scores, "monitoring_readiness_label", "snapshot_ready")),
            ("Validation Rows", _count_nonempty(scores, "validation_tier")),
        ]
    )
    render_callout(
        "Research V2 starts with scorecard summaries. Candidate diagnostics and legacy aggregate workbench are explicit loads."
    )

    view = st.pills(
        "Research view",
        [
            "Outcome Frontier",
            "Candidate Deep Dive",
            "Leaderboard",
            "Validation artifacts",
            "Full Workbench",
        ],
        default="Outcome Frontier",
        selection_mode="single",
        key="dashboard_v2_research_view",
    )
    selected_view = view or "Outcome Frontier"
    if selected_view == "Outcome Frontier":
        _render_outcome_frontier(
            bot_config=runtime.bot_config,
            baseline_run=runtime.baseline_run,
            experiment_scorecards=scores,
            experiment_candidates=pd.DataFrame(),
            warehouse_path=str(runtime.paths.run_store_path),
        )
    elif selected_view == "Leaderboard":
        _render_leaderboard(top)
    elif selected_view == "Candidate Deep Dive":
        _render_candidate(scores, runtime=runtime)
    elif selected_view == "Validation artifacts":
        _render_validation_artifacts()
    else:
        render_callout(
            "This loads the full Research Lab and its aggregate frames. Use it only when the summary-first views are not enough.",
            heavy=True,
        )
        with timed("research.legacy_frames"):
            frames = dashboard_frames()
        _render_research_lab(
            runtime.bot_config,
            runtime.baseline_run,
            frames[0],
            frames[1],
            frames[2],
            frames[3],
            frames[4],
            warehouse_path=str(runtime.paths.run_store_path),
        )


def _render_leaderboard(frame: pd.DataFrame) -> None:
    st.subheader("Top Candidate Summary")
    if frame.empty:
        st.info("No experiment scorecards are available.")
        return
    columns = [
        column
        for column in [
            "strategy",
            "growth_constrained_utility_score",
            "cagr",
            "max_drawdown",
            "calmar",
            "sharpe",
            "promotion_decision",
            "monitoring_readiness_label",
            "overfit_risk_label",
        ]
        if column in frame
    ]
    _render_metric_dataframe(_display_metrics(frame[columns]))


def _render_candidate(scores: pd.DataFrame, *, runtime: DashboardRuntime) -> None:
    st.subheader("Candidate Deep Dive")
    if scores.empty or "strategy" not in scores:
        st.info("No candidates are available.")
        return
    ordered = top_scorecards(scores, limit=max(len(scores), 1))
    selected_from_frontier = st.session_state.get(_OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY)
    strategy_options = list(ordered["strategy"].astype(str))
    last_frontier_strategy = st.session_state.get("dashboard_v2_last_frontier_strategy")
    if selected_from_frontier in strategy_options and selected_from_frontier != last_frontier_strategy:
        st.session_state["dashboard_v2_candidate"] = selected_from_frontier
        st.session_state["dashboard_v2_last_frontier_strategy"] = selected_from_frontier
    selected = _clearable_selectbox(
        "Candidate",
        strategy_options,
        key="dashboard_v2_candidate",
        placeholder="Search candidate...",
    )
    if selected is None:
        st.info("Choose a candidate.")
        return
    row = ordered[ordered["strategy"].astype(str) == str(selected)].iloc[0]
    render_card_grid(
        [
            ("CAGR", _fmt_pct(row.get("cagr"))),
            ("Max Drawdown", _fmt_pct(row.get("max_drawdown"))),
            ("Calmar", _fmt_float(row.get("calmar"))),
            ("Utility", _fmt_float(row.get("growth_constrained_utility_score"))),
            ("Readiness", row.get("monitoring_readiness_label", "n/a")),
            ("Overfit", row.get("overfit_risk_label", "n/a")),
        ]
    )
    summary_columns = [
        column
        for column in [
            "strategy",
            "hypothesis",
            "research_status",
            "promotion_decision",
            "validation_tier",
            "growth_utility_tier",
            "monitoring_readiness_label",
        ]
        if column in ordered
    ]
    _render_metric_dataframe(pd.DataFrame([row[summary_columns].to_dict()]))
    _render_candidate_artifact_read(str(selected))
    load_full_candidate_workbench = st.toggle(
        "Load full candidate workbench",
        value=False,
        key="dashboard_v2_load_candidate_workbench",
        help=(
            "Loads the former Candidate Deep Dive from the full workbench. Keep this off "
            "when you only need the fast summary."
        ),
    )
    if load_full_candidate_workbench:
        render_callout(
            "This is the former Candidate Deep Dive from the legacy workbench, now reachable from the V2 candidate view.",
            heavy=True,
        )
        with timed("research.candidate_legacy_frames"):
            frames = dashboard_frames()
        _render_approach_detail_workbench(
            bot_config=runtime.bot_config,
            baseline_run=runtime.baseline_run,
            experiment_scorecards=frames[0],
            experiment_regimes=frames[1],
            experiment_walk_forward=frames[2],
            experiment_candidates=frames[3],
        )


def _render_candidate_artifact_read(strategy_name: str) -> None:
    pbo = pbo_frames()
    selection = pbo.get("selection", pd.DataFrame())
    stats = pbo.get("stats", pd.DataFrame())
    with st.expander("PBO / overfit artifact read", expanded=False):
        if selection.empty and stats.empty:
            st.info("No PBO artifacts found. Run `poetry run trade-bot audit-backtest-pbo`.")
            return
        matches = []
        for frame in [selection, stats]:
            if not frame.empty and "strategy" in frame:
                matches.append(frame[frame["strategy"].astype(str) == strategy_name])
        combined = pd.concat([match for match in matches if not match.empty], ignore_index=True) if matches else pd.DataFrame()
        if combined.empty:
            st.info("This candidate is not present in the latest PBO artifact set.")
        else:
            _render_metric_dataframe(_display_metrics(combined))


def _render_validation_artifacts() -> None:
    st.subheader("Validation Artifacts")
    pbo = pbo_frames()
    leadership = leadership_frames()
    if not pbo["summary"].empty:
        st.markdown("**PBO summary**")
        _render_metric_dataframe(_display_metrics(pbo["summary"]))
    if not pbo["selection"].empty:
        st.markdown("**PBO selections**")
        _render_metric_dataframe(_display_metrics(pbo["selection"].head(20)))
    if not leadership["summary"].empty:
        st.markdown("**Leadership summary**")
        _render_metric_dataframe(_display_metrics(leadership["summary"].head(20)))
    if not leadership["router"].empty:
        st.markdown("**Router comparison**")
        _render_metric_dataframe(_display_metrics(leadership["router"].head(40)))
    if all(frame.empty for frame in [*pbo.values(), *leadership.values()]):
        st.info("No validation artifacts found yet.")


def _best_metric(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return "n/a"
    value = pd.to_numeric(frame[column], errors="coerce").max()
    return _fmt_pct(value) if "cagr" in column or "drawdown" in column else _fmt_float(value)


def _count_label(frame: pd.DataFrame, column: str, label: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].astype(str).eq(label).sum())


def _count_nonempty(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].notna().sum())


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
