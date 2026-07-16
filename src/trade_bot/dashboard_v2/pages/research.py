from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _clearable_selectbox, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.research_lab import (
    _OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY,
    _render_approach_detail_workbench,
    _render_outcome_decision_cards,
    _render_outcome_frontier,
    _render_research_lab,
)
from trade_bot.dashboard.strategy_candidates import (
    outcome_candidate_scorecards,
    runtime_benchmark_metrics,
    scorecard_option_label,
)
from trade_bot.dashboard_v2.components.cards import render_callout, render_card_grid
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.artifact_service import (
    cycle_tracker_frames,
    leadership_frames,
    pbo_frames,
)
from trade_bot.dashboard_v2.services.experiment_service import (
    dashboard_frames,
    scorecards,
    top_scorecards,
)
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime


def render_research_page(runtime: DashboardRuntime) -> None:
    with timed("research.scorecards"):
        scores = scorecards()
    with timed("research.candidate_universe"):
        candidates = _candidate_universe(scores, runtime=runtime)
    top = top_scorecards(candidates, limit=30)
    render_card_grid(
        [
            ("Candidates", len(candidates)),
            ("Displayed", len(top)),
            ("Champion CAGR", _best_metric(top, "cagr")),
            ("Best Utility", _best_metric(top, "growth_constrained_utility_score")),
            (
                "Snapshot-Ready",
                _count_label(candidates, "monitoring_readiness_label", "snapshot_ready"),
            ),
            ("Validation Rows", _count_nonempty(candidates, "validation_tier")),
        ]
    )
    render_callout(
        "Research V2 starts with scorecard summaries. Candidate diagnostics and the full aggregate workbench are explicit loads."
    )

    view = st.pills(
        "Research view",
        [
            "Outcome Frontier",
            "Cycle Tracker",
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
    elif selected_view == "Cycle Tracker":
        _render_cycle_tracker()
    elif selected_view == "Leaderboard":
        _render_leaderboard(top)
    elif selected_view == "Candidate Deep Dive":
        _render_candidate(candidates, runtime=runtime)
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
    ordered = ordered.copy()
    ordered["candidate_label"] = ordered.apply(scorecard_option_label, axis=1)
    label_to_strategy = dict(
        zip(ordered["candidate_label"].astype(str), ordered["strategy"].astype(str), strict=False)
    )
    strategy_to_label = dict(
        zip(ordered["strategy"].astype(str), ordered["candidate_label"].astype(str), strict=False)
    )
    candidate_labels = ordered["candidate_label"].astype(str).tolist()
    last_frontier_strategy = st.session_state.get("dashboard_v2_last_frontier_strategy")
    if (
        selected_from_frontier in strategy_to_label
        and selected_from_frontier != last_frontier_strategy
    ):
        st.session_state["dashboard_v2_candidate"] = strategy_to_label[str(selected_from_frontier)]
        st.session_state["dashboard_v2_last_frontier_strategy"] = selected_from_frontier
    selected = _clearable_selectbox(
        "Candidate",
        candidate_labels,
        key="dashboard_v2_candidate",
        placeholder="Search candidate...",
    )
    if selected is None:
        st.info("Choose a candidate.")
        return
    selected_strategy = label_to_strategy.get(str(selected), str(selected))
    row = ordered[ordered["strategy"].astype(str) == selected_strategy].iloc[0]
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
    render_callout(
        "Candidate Deep Dive includes latest runtime snapshot strategies and migrated "
        "experiment candidates. Snapshot-only candidates may have metrics and outcome "
        "diagnostics before they have full experiment artifacts.",
    )
    _render_outcome_decision_cards(
        row,
        bot_config=runtime.bot_config,
        baseline_run=runtime.baseline_run,
        experiment_scorecards=scores,
        peer_frame=ordered,
        warehouse_path=str(runtime.paths.run_store_path),
    )
    _render_candidate_artifact_read(selected_strategy)
    st.divider()
    st.subheader("Candidate Detail Tabs")
    st.caption(
        "Full drilldown for the selected candidate: performance, allocation, decision "
        "timeline, factor attribution, mechanics, robustness, and manifest/risk notes."
    )
    with timed("research.candidate_detail_frames"):
        frames = dashboard_frames()
    _render_approach_detail_workbench(
        bot_config=runtime.bot_config,
        baseline_run=runtime.baseline_run,
        experiment_scorecards=frames[0],
        experiment_regimes=frames[1],
        experiment_walk_forward=frames[2],
        experiment_candidates=frames[3],
        selected_strategy=selected_strategy,
        key_prefix="dashboard_v2_candidate",
        show_selector=False,
    )


def _candidate_universe(scores: pd.DataFrame, *, runtime: DashboardRuntime) -> pd.DataFrame:
    """Return the fast V2 candidate set, including latest runtime snapshot metrics."""

    candidates = outcome_candidate_scorecards(
        baseline_run=runtime.baseline_run,
        bot_config=runtime.bot_config,
        experiment_scorecards=scores,
        include_defensive_judgement=False,
    )
    if candidates.empty:
        return scores
    benchmark_metrics = runtime_benchmark_metrics(runtime.baseline_run)
    try:
        from trade_bot.research.strategy_outcome_utility import (
            add_outcome_frontier_flags,
            enrich_strategy_outcome_utility,
        )

        candidates = add_outcome_frontier_flags(
            enrich_strategy_outcome_utility(candidates, benchmark_metrics=benchmark_metrics)
        )
    except (KeyError, ValueError, TypeError):
        pass
    if "research_status" in candidates:
        active = candidates[
            ~candidates["research_status"].astype(str).eq("pruned_dead_end")
        ].copy()
        if not active.empty:
            candidates = active
    return candidates.reset_index(drop=True)


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


def _render_cycle_tracker() -> None:
    st.subheader("Scenario / Phase Frontier")
    st.caption(
        "Research/watch layer for speculative-cycle phases, horizon phase probabilities, "
        "and conditional winners. This view reads persisted artifacts only."
    )
    frames = cycle_tracker_frames()
    phase = frames["phase_probabilities"]
    forecast = frames["transition_forecast"]
    evidence = frames["evidence"]
    candidates = frames["candidate_scores"]
    frontier = frames["phase_candidate_frontier"]
    validation = frames["validation_metrics"]
    if all(frame.empty for frame in frames.values()):
        st.info("No cycle tracker artifacts found. Run `poetry run trade-bot run-cycle-tracker`.")
        return

    dominant_phase = "n/a"
    dominant_probability = "n/a"
    if not phase.empty and "probability" in phase:
        top_phase = phase.sort_values("probability", ascending=False).iloc[0]
        dominant_phase = str(top_phase.get("phase", "n/a"))
        dominant_probability = _fmt_pct(top_phase.get("probability"))
    top_candidate = "n/a"
    if not candidates.empty and "ticker" in candidates:
        top_candidate = str(candidates.iloc[0]["ticker"])
    render_card_grid(
        [
            ("Dominant Phase", dominant_phase),
            ("Phase Probability", dominant_probability),
            ("Candidate Rows", len(candidates)),
            ("Frontier Rows", len(frontier)),
            ("Validation Rows", len(validation)),
            ("Top Candidate", top_candidate),
        ]
    )
    render_callout(
        "Cycle Tracker is not a crash timer or allocation override. It asks which speculative-cycle phase the current market resembles, which phases are plausible by horizon, and which assets historically performed better in similar prior states.",
    )

    if not forecast.empty:
        st.markdown("**0M nowcast + forward phase frontier**")
        st.plotly_chart(_phase_frontier_figure(forecast), use_container_width=True)
        _render_metric_dataframe(_display_metrics(forecast.head(80)))
    elif not phase.empty:
        st.markdown("**0M nowcast phase probabilities**")
        st.plotly_chart(_phase_frontier_figure(phase), use_container_width=True)

    if not candidates.empty:
        st.markdown("**Current-phase conditional candidates**")
        candidate_columns = [
            column
            for column in [
                "ticker",
                "asset_role",
                "candidate_role",
                "candidate_score",
                "current_momentum_21d",
                "current_momentum_63d",
                "phase_forward_median_return",
                "phase_median_excess_vs_qqq",
                "phase_hit_rate_vs_qqq",
                "phase_origins",
                "interpretation",
            ]
            if column in candidates
        ]
        _render_metric_dataframe(_display_metrics(candidates[candidate_columns].head(30)))

    if not frontier.empty:
        st.markdown("**Scenario / phase winner frontier**")
        _render_phase_candidate_frontier(frontier)

    if not evidence.empty:
        with st.expander("Evidence components", expanded=False):
            _render_metric_dataframe(_display_metrics(evidence))
    if not validation.empty:
        with st.expander("Prior-only validation metrics", expanded=False):
            validation_columns = [
                column
                for column in [
                    "dominant_phase",
                    "horizon",
                    "ticker",
                    "asset_role",
                    "origins",
                    "median_forward_return",
                    "median_excess_vs_spy",
                    "median_excess_vs_qqq",
                    "hit_rate_vs_qqq",
                    "median_forward_drawdown",
                    "severe_drawdown_rate",
                    "phase_rank_score",
                ]
                if column in validation
            ]
            _render_metric_dataframe(_display_metrics(validation[validation_columns].head(100)))


def _render_phase_candidate_frontier(frontier: pd.DataFrame) -> None:
    data = frontier.copy()
    data["horizon_days"] = pd.to_numeric(data["horizon_days"], errors="coerce")
    data["phase_probability"] = pd.to_numeric(data["phase_probability"], errors="coerce").fillna(0.0)
    data["rank"] = pd.to_numeric(data["rank"], errors="coerce").fillna(999).astype(int)
    horizon_order = (
        data[["horizon", "horizon_days"]]
        .drop_duplicates()
        .sort_values("horizon_days")["horizon"]
        .astype(str)
        .tolist()
    )
    selected_horizon = st.pills(
        "Frontier horizon",
        horizon_order,
        default=horizon_order[0] if horizon_order else None,
        selection_mode="single",
        key="dashboard_v2_cycle_frontier_horizon",
    )
    if not selected_horizon:
        return
    horizon_frame = data[data["horizon"].astype(str).eq(str(selected_horizon))].copy()
    phase_options = (
        horizon_frame[["phase", "phase_probability"]]
        .drop_duplicates()
        .sort_values("phase_probability", ascending=False)
    )
    phase_labels = [
        f"{row.phase} ({float(row.phase_probability):.1%})"
        for row in phase_options.itertuples(index=False)
    ]
    label_to_phase = dict(
        zip(phase_labels, phase_options["phase"].astype(str).tolist(), strict=False)
    )
    selected_label = st.selectbox(
        "Dominant phase to inspect",
        phase_labels,
        index=0,
        key="dashboard_v2_cycle_frontier_phase",
    )
    selected_phase = label_to_phase.get(str(selected_label), "")
    selected = horizon_frame[horizon_frame["phase"].astype(str).eq(selected_phase)].copy()
    selected = selected.sort_values(["rank", "frontier_score"], ascending=[True, False])
    if selected.empty:
        st.info("No candidate evidence is available for the selected phase and horizon.")
        return
    render_card_grid(
        [
            ("Selected Phase", selected_phase),
            ("Phase Odds", _fmt_pct(selected["phase_probability"].iloc[0])),
            ("Top Ticker", selected.iloc[0].get("ticker", "n/a")),
            ("Top Role", selected.iloc[0].get("frontier_role", "n/a")),
        ]
    )
    st.plotly_chart(_phase_winner_figure(selected.head(8)), use_container_width=True)
    frontier_columns = [
        column
        for column in [
            "rank",
            "ticker",
            "asset_role",
            "frontier_role",
            "frontier_score",
            "median_forward_return",
            "median_excess_vs_spy",
            "median_excess_vs_qqq",
            "hit_rate_vs_qqq",
            "median_forward_drawdown",
            "origins",
            "interpretation",
        ]
        if column in selected
    ]
    _render_metric_dataframe(_display_metrics(selected[frontier_columns]))


def _phase_winner_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["frontier_score"] = pd.to_numeric(data["frontier_score"], errors="coerce").fillna(0.0)
    data = data.sort_values("frontier_score", ascending=True)
    color_map = {
        "scale_candidate": "#16a34a",
        "starter_reentry": "#2563eb",
        "watch": "#f59e0b",
        "defend": "#06b6d4",
        "avoid": "#ef4444",
    }
    colors = [
        color_map.get(str(role), "#7f8ea3")
        for role in data.get("frontier_role", pd.Series(dtype=str)).astype(str)
    ]
    figure = go.Figure(
        go.Bar(
            x=data["frontier_score"],
            y=data["ticker"].astype(str),
            orientation="h",
            marker_color=colors,
            text=data.get("frontier_role", pd.Series([""] * len(data))).astype(str),
            hovertemplate=(
                "<b>%{y}</b><br>Frontier score: %{x:.2f}<br>"
                "Role: %{text}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis_title="Frontier score",
        yaxis_title="Ticker",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=340,
    )
    return figure


def _phase_frontier_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    if "horizon" not in data:
        data["horizon"] = "0m"
    if "probability" in data:
        data["probability"] = pd.to_numeric(data["probability"], errors="coerce").fillna(0.0)
    phase_order = [
        "normal_cycle",
        "acceleration",
        "pre_break",
        "early_unwind",
        "liquidation",
        "bottoming",
        "recovery",
        "post_unwind_compounding",
    ]
    color_map = {
        "normal_cycle": "#7f8ea3",
        "acceleration": "#16a34a",
        "pre_break": "#f59e0b",
        "early_unwind": "#ef4444",
        "liquidation": "#991b1b",
        "bottoming": "#8b5cf6",
        "recovery": "#06b6d4",
        "post_unwind_compounding": "#2563eb",
    }
    figure = go.Figure()
    if "horizon_days" in data:
        order_frame = data[["horizon", "horizon_days"]].drop_duplicates().copy()
        order_frame["horizon_days"] = pd.to_numeric(
            order_frame["horizon_days"],
            errors="coerce",
        )
        horizons = (
            order_frame.sort_values("horizon_days")["horizon"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    else:
        horizons = data["horizon"].astype(str).drop_duplicates().tolist()
    for phase in phase_order:
        phase_rows = data[data["phase"].astype(str).eq(phase)]
        if phase_rows.empty:
            continue
        y_values = []
        for horizon in horizons:
            row = phase_rows[phase_rows["horizon"].astype(str).eq(horizon)]
            y_values.append(float(row["probability"].iloc[0]) if not row.empty else 0.0)
        figure.add_trace(
            go.Bar(
                x=horizons,
                y=y_values,
                name=phase.replace("_", " ").title(),
                marker_color=color_map.get(phase),
            )
        )
    figure.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        yaxis_title="Probability",
        xaxis_title="Horizon",
        legend_title_text="Phase",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=420,
    )
    return figure


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
