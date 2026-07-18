from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from trade_bot.dashboard_v2.components.cards import (
    render_callout,
    render_card_grid,
    render_chart,
    render_section_header,
)
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
    render_section_header("Top Candidate Summary")
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
    render_section_header("Candidate Deep Dive")
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
    render_section_header("Candidate Detail Tabs")
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
    render_section_header("Validation Artifacts")
    pbo = pbo_frames()
    leadership = leadership_frames()
    if not pbo["summary"].empty:
        render_section_header("PBO Summary")
        _render_metric_dataframe(_display_metrics(pbo["summary"]))
    if not pbo["selection"].empty:
        render_section_header("PBO Selections")
        _render_metric_dataframe(_display_metrics(pbo["selection"].head(20)))
    if not leadership["summary"].empty:
        render_section_header("Leadership Summary")
        _render_metric_dataframe(_display_metrics(leadership["summary"].head(20)))
    if not leadership["router"].empty:
        render_section_header("Router Comparison")
        _render_metric_dataframe(_display_metrics(leadership["router"].head(40)))
    if all(frame.empty for frame in [*pbo.values(), *leadership.values()]):
        st.info("No validation artifacts found yet.")


def _render_cycle_tracker() -> None:
    render_section_header("Scenario / Phase Frontier")
    st.caption(
        "Research/watch layer for speculative-cycle phases, horizon phase probabilities, "
        "and conditional winners. This view reads persisted artifacts only."
    )
    frames = cycle_tracker_frames()
    phase = frames["phase_probabilities"]
    forecast = frames["transition_forecast"]
    evidence = frames["evidence"]
    path_history = frames.get("path_state_history", pd.DataFrame())
    path_forecast = frames.get("path_transition_forecast", pd.DataFrame())
    candidates = frames["candidate_scores"]
    frontier = frames["phase_candidate_frontier"]
    validation = frames["validation_metrics"]
    path_validation = frames.get("path_validation_metrics", pd.DataFrame())
    reliability = frames.get("phase_reliability", pd.DataFrame())
    path_reliability = frames.get("path_reliability", pd.DataFrame())
    crisis = frames.get("crisis_playback", pd.DataFrame())
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
            ("Path Validation Rows", len(path_validation)),
            ("Top Candidate", top_candidate),
        ]
    )
    render_callout(
        "Cycle Tracker is not a crash timer or allocation override. It asks which speculative-cycle phase the current market resembles, which phases are plausible by horizon, and which assets historically performed better in similar prior states.",
    )
    _render_current_phase_candidates_expander(candidates)

    if not path_history.empty:
        _render_path_cycle_state(
            path_history,
            path_forecast,
            path_reliability,
            frontier,
            path_validation if not path_validation.empty else validation,
        )
    else:
        st.info(
            "This cycle tracker run does not include path-aware cycle state yet. Re-run `poetry run trade-bot run-cycle-tracker`."
        )

    if not reliability.empty:
        _render_cycle_reliability(reliability, dominant_phase=dominant_phase)
    else:
        st.info(
            "This cycle tracker run does not include phase reliability yet. Re-run `poetry run trade-bot run-cycle-tracker`."
        )

    if not crisis.empty:
        _render_crisis_playback(crisis)
    else:
        st.info(
            "This cycle tracker run does not include crisis playback yet. Re-run `poetry run trade-bot run-cycle-tracker`."
        )

    if path_forecast.empty and forecast.empty and not phase.empty:
        render_section_header("0M Nowcast Phase Probabilities")
        render_chart(
            _phase_frontier_figure(phase),
            title="Scenario / Phase Frontier",
            key="dashboard_v2_cycle_nowcast_phase_frontier_chart",
        )

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
    if not path_validation.empty:
        with st.expander("Path-conditioned validation metrics", expanded=False):
            st.caption(
                "Used by the path-aware winner frontier when available: each historical origin is labeled by decoded path phase before forward ticker outcomes are measured."
            )
            path_validation_columns = [
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
                if column in path_validation
            ]
            _render_metric_dataframe(
                _display_metrics(path_validation[path_validation_columns].head(100))
            )


def _render_current_phase_candidates_expander(candidates: pd.DataFrame) -> None:
    if candidates.empty:
        return
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
    display = candidates[candidate_columns] if candidate_columns else candidates
    with st.expander(
        f"Current-phase conditional candidates ({len(candidates):,})",
        expanded=False,
    ):
        _render_metric_dataframe(_display_metrics(display.head(30)))


def _render_path_cycle_state(
    path_history: pd.DataFrame,
    path_forecast: pd.DataFrame,
    path_reliability: pd.DataFrame,
    frontier: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    render_section_header("Path-Aware Cycle Tracker")
    st.caption(
        "Sequential decoder: turns simultaneous phase evidence into one plausible path using allowed transitions, phase duration, and prior unwind/recovery memory."
    )
    data = path_history.copy()
    data["as_of_date"] = pd.to_datetime(data["as_of_date"], errors="coerce")
    latest = data.sort_values("as_of_date").iloc[-1]
    duration_days = pd.to_numeric(
        pd.Series([latest.get("phase_duration_days", 0)]),
        errors="coerce",
    ).fillna(0).iloc[0]
    render_card_grid(
        [
            ("Path Phase", latest.get("path_phase", "n/a")),
            ("Evidence Phase", latest.get("evidence_phase", "n/a")),
            ("Path Probability", _fmt_pct(latest.get("path_probability"))),
            ("Duration", f"{int(duration_days):,}d"),
            ("Duration State", latest.get("phase_duration_bucket", "n/a")),
            ("Transition Read", latest.get("transition_reason", "n/a")),
        ]
    )
    render_callout(
        "This is the cycle tracker answer to path dependence: bottoming and post-unwind states are constrained unless prior drawdown, unwind, or recovery memory exists. If raw evidence and path phase disagree, treat the path phase as the operational read and the evidence phase as a diagnostic."
    )
    if not path_forecast.empty:
        _render_path_phase_behavior_inspector(
            path_forecast,
            frontier,
            validation,
        )
    if not path_reliability.empty:
        _render_path_cycle_reliability(
            path_reliability,
            path_phase=str(latest.get("path_phase", "n/a")),
        )
    with st.expander("Path state history", expanded=False):
        columns = [
            column
            for column in [
                "as_of_date",
                "evidence_phase",
                "path_phase",
                "path_probability",
                "previous_path_phase",
                "phase_duration_days",
                "phase_duration_bucket",
                "prior_unwind_seen_504d",
                "prior_bottoming_seen_504d",
                "qqq_drawdown_252d",
                "spy_drawdown_252d",
                "transition_reason",
            ]
            if column in data
        ]
        _render_metric_dataframe(_display_metrics(data[columns].tail(80)))


def _render_path_phase_behavior_inspector(
    path_forecast: pd.DataFrame,
    frontier: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    render_section_header(
        "Path Phase Behavior Inspector",
        help_text=(
            "Select a horizon and phase from the path-aware forecast to inspect expected return, "
            "drawdown, benchmark excess, and the conditional winner set for that slice."
        ),
    )
    st.caption(
        "Use this paired view to ask: if this horizon/phase dominates, what has usually happened next and which assets ranked best in similar prior states?"
    )
    forecast = path_forecast.copy()
    if "horizon_days" in forecast:
        forecast["horizon_days"] = pd.to_numeric(forecast["horizon_days"], errors="coerce")
    else:
        forecast["horizon_days"] = range(len(forecast))
    if "probability" in forecast:
        forecast["probability"] = pd.to_numeric(forecast["probability"], errors="coerce").fillna(0.0)
    else:
        forecast["probability"] = 0.0
    horizon_order = (
        forecast[["horizon", "horizon_days"]]
        .drop_duplicates()
        .sort_values("horizon_days")["horizon"]
        .astype(str)
        .tolist()
    )
    selected_horizon = st.pills(
        "Behavior horizon",
        horizon_order,
        default="3m" if "3m" in horizon_order else (horizon_order[0] if horizon_order else None),
        selection_mode="single",
        key="dashboard_v2_cycle_behavior_horizon",
    )
    if not selected_horizon:
        return
    horizon_frame = forecast[forecast["horizon"].astype(str).eq(str(selected_horizon))].copy()
    if horizon_frame.empty:
        st.info("No path-aware phase rows are available for this horizon.")
        return
    phase_options = (
        horizon_frame[["phase", "probability"]]
        .drop_duplicates()
        .sort_values("probability", ascending=False)
    )
    phase_labels = [
        f"{row.phase} ({float(row.probability):.1%})"
        for row in phase_options.itertuples(index=False)
    ]
    label_to_phase = dict(
        zip(phase_labels, phase_options["phase"].astype(str).tolist(), strict=False)
    )
    selected_label = st.selectbox(
        "Phase slice to inspect",
        phase_labels,
        index=0,
        key="dashboard_v2_cycle_behavior_phase",
    )
    selected_phase = label_to_phase.get(str(selected_label), "")
    selected_probability = 0.0
    selected_probability_rows = horizon_frame[horizon_frame["phase"].astype(str).eq(selected_phase)]
    if not selected_probability_rows.empty:
        selected_probability = float(selected_probability_rows["probability"].iloc[0])

    render_chart(
        _phase_frontier_figure(
            forecast,
            selected_horizon=str(selected_horizon),
            selected_phase=selected_phase,
        ),
        title="Path-Aware Transition Model",
        help_text=(
            "Stacked bars show phase probabilities by horizon after sequence rules are applied. "
            "The highlighted marker is the horizon/phase slice being inspected below."
        ),
        key=f"dashboard_v2_cycle_path_transition_model_chart_{selected_horizon}_{selected_phase}",
    )

    frontier_slice = _slice_horizon_phase(
        frontier,
        horizon=str(selected_horizon),
        phase_column="phase",
        phase=selected_phase,
    )
    validation_slice = _slice_horizon_phase(
        validation,
        horizon=str(selected_horizon),
        phase_column="dominant_phase",
        phase=selected_phase,
    )
    summary = _phase_behavior_summary(validation_slice)
    top_ticker = (
        str(frontier_slice.sort_values(["rank", "frontier_score"], ascending=[True, False]).iloc[0]["ticker"])
        if not frontier_slice.empty and "ticker" in frontier_slice
        else "n/a"
    )
    render_card_grid(
        [
            (
                "Selected Slice",
                f"{selected_horizon} / {selected_phase}",
                "The path-aware horizon and phase currently being inspected.",
            ),
            (
                "Phase Odds",
                _fmt_pct(selected_probability),
                "Current path-constrained probability assigned to this phase at the selected horizon.",
            ),
            (
                "Historical Origins",
                int(summary.get("origins", 0)),
                "Prior-only historical examples behind the selected phase and horizon.",
            ),
            (
                "Median Forward Return",
                _fmt_pct(summary.get("median_forward_return")),
                "Median next-window return across assets measured in historical origins for this phase.",
            ),
            (
                "Median Drawdown",
                _fmt_pct(summary.get("median_forward_drawdown")),
                "Median peak-to-trough drawdown during the selected forward window. More negative means rougher path risk.",
            ),
            (
                "Top Frontier Asset",
                top_ticker,
                "Highest-ranked ticker in the current conditional winner frontier for this phase.",
            ),
        ]
    )
    render_callout(
        f"For {selected_horizon} `{selected_phase}`, the inspector combines the current path probability "
        "with prior-only historical outcome behavior and the current conditional winner frontier. Treat this as scenario planning, not an allocation override."
    )

    outcome_figure = _phase_outcome_profile_figure(validation_slice, phase=selected_phase)
    if outcome_figure is not None:
        left, right = st.columns([0.46, 0.54])
        with left:
            render_chart(
                outcome_figure,
                title="Selected Phase Outcome Profile",
                help_text=(
                    "Bars summarize what usually happened after historical origins with this phase and horizon: "
                    "returns, benchmark excess, and drawdown."
                ),
                key=f"dashboard_v2_cycle_phase_outcome_profile_{selected_horizon}_{selected_phase}",
            )
        with right:
            if frontier_slice.empty:
                st.info("No conditional winner frontier rows are available for this slice.")
            else:
                render_chart(
                    _phase_winner_figure(
                        frontier_slice.sort_values(
                            ["rank", "frontier_score"],
                            ascending=[True, False],
                        ).head(8)
                    ),
                    title="Conditional Winner Frontier",
                    help_text=(
                        "Ranks assets that historically did best in similar phase/horizon settings, blended with current momentum and role rules."
                    ),
                    key=f"dashboard_v2_cycle_behavior_winner_chart_{selected_horizon}_{selected_phase}",
                )
    elif not frontier_slice.empty:
        render_chart(
            _phase_winner_figure(
                frontier_slice.sort_values(["rank", "frontier_score"], ascending=[True, False]).head(8)
            ),
            title="Conditional Winner Frontier",
            key=f"dashboard_v2_cycle_behavior_winner_chart_{selected_horizon}_{selected_phase}",
        )

    with st.expander("Selected phase detail table", expanded=False):
        if validation_slice.empty and frontier_slice.empty:
            st.info("No validation or frontier rows are available for this selection.")
        if not validation_slice.empty:
            st.caption("Historical outcome behavior for the selected phase and horizon.")
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
                if column in validation_slice
            ]
            _render_metric_dataframe(_display_metrics(validation_slice[validation_columns].head(40)))
        if not frontier_slice.empty:
            st.caption("Current conditional winner frontier for the selected phase and horizon.")
            frontier_columns = [
                column
                for column in [
                    "rank",
                    "ticker",
                    "asset_role",
                    "exposure_family",
                    "phase_window_role",
                    "frontier_role",
                    "evidence_quality",
                    "evidence_flags",
                    "frontier_score",
                    "validation_score",
                    "origin_confidence",
                    "origin_penalty",
                    "phase_role_fit",
                    "momentum_score",
                    "drawdown_penalty",
                    "theme_fragility_penalty",
                    "median_forward_return",
                    "median_excess_vs_spy",
                    "median_excess_vs_qqq",
                    "hit_rate_vs_qqq",
                    "median_forward_drawdown",
                    "origins",
                    "interpretation",
                ]
                if column in frontier_slice
            ]
            _render_metric_dataframe(_display_metrics(frontier_slice[frontier_columns].head(40)))


def _render_path_cycle_reliability(
    path_reliability: pd.DataFrame,
    *,
    path_phase: str,
) -> None:
    data = path_reliability.copy()
    data["horizon_days"] = pd.to_numeric(data["horizon_days"], errors="coerce")
    data["path_fit_rate"] = pd.to_numeric(data["path_fit_rate"], errors="coerce")
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0).astype(int)
    horizon_order = (
        data[["horizon", "horizon_days"]]
        .drop_duplicates()
        .sort_values("horizon_days")["horizon"]
        .astype(str)
        .tolist()
    )
    has_nowcast = data["horizon_days"].eq(0).any()
    render_section_header(
        "Path Reliability",
        help_text=_path_reliability_intro_text(has_nowcast=bool(has_nowcast)),
    )
    render_callout(
        _path_reliability_callout_text(has_nowcast=bool(has_nowcast))
    )
    selected_horizon = st.pills(
        "Path reliability horizon",
        horizon_order,
        default="3m" if "3m" in horizon_order else (horizon_order[0] if horizon_order else None),
        selection_mode="single",
        key="dashboard_v2_cycle_path_reliability_horizon",
    )
    if not selected_horizon:
        return
    selected = data[data["horizon"].astype(str).eq(str(selected_horizon))].copy()
    if selected.empty:
        st.info("No path reliability rows are available for this horizon.")
        return
    current = selected[selected["path_phase"].astype(str).eq(str(path_phase))]
    headline = (
        current.iloc[0]
        if not current.empty
        else selected.sort_values("origins", ascending=False).iloc[0]
    )
    is_nowcast = int(pd.to_numeric(pd.Series([headline.get("horizon_days")]), errors="coerce").fillna(-1).iloc[0]) == 0
    fit_help = (
        "For 0M, this is the share of historical origins where the path-constrained phase agreed with the raw evidence phase on the same date. It is a nowcast coherence check, not a forward prediction score."
        if is_nowcast
        else "For forward horizons, this is the share of historical origins where the next market window behaved the way this path-aware phase implies."
    )
    render_card_grid(
        [
            (
                "Path Fit Rate",
                _fmt_pct(headline.get("path_fit_rate")),
                fit_help,
            ),
            (
                "Path Origins",
                int(headline.get("origins", 0)),
                "Historical prior-only origins available for this path phase and horizon. More origins make the read less fragile.",
            ),
            (
                "Path Label",
                headline.get("reliability_label", "n/a"),
                "Reliability summary based on fit rate and sample size. Supportive labels mean the operational phase has historical backing.",
            ),
        ]
    )
    if is_nowcast:
        render_callout(
            f"For 0M, path-aware `{headline.get('path_phase', path_phase)}` agreed with the raw evidence phase "
            f"{_fmt_pct(headline.get('path_fit_rate'))} of the time across {int(headline.get('origins', 0))} historical origins. "
            "This does not score a future return; it scores whether the sequence-aware operational state usually lines up with same-date evidence."
        )
    else:
        render_callout(
            f"For {selected_horizon}, path-aware `{headline.get('path_phase', path_phase)}` fit history "
            f"{_fmt_pct(headline.get('path_fit_rate'))} across {int(headline.get('origins', 0))} origins. "
            f"Expected behavior: {headline.get('expected_behavior', 'n/a')}"
        )
    render_chart(
        _path_reliability_figure(selected),
        title="Path Reliability",
        help_text=_path_reliability_chart_help_text(has_nowcast=bool(has_nowcast)),
        key=f"dashboard_v2_cycle_path_reliability_chart_{selected_horizon}",
    )
    with st.expander("Path reliability audit table", expanded=False):
        columns = [
            column
            for column in [
                "path_phase",
                "horizon",
                "origins",
                "path_fit_rate",
                "median_path_probability",
                "median_phase_duration_days",
                "median_qqq_forward_return",
                "median_qqq_forward_drawdown",
                "reliability_label",
                "expected_behavior",
            ]
            if column in selected
        ]
        _render_metric_dataframe(_display_metrics(selected[columns]))


def _path_reliability_intro_text(*, has_nowcast: bool) -> str:
    if has_nowcast:
        return (
            "Trust check for the sequential operational read. 0M is a nowcast agreement check between "
            "path-constrained phase and raw evidence phase. Forward horizons ask whether realized next-window "
            "behavior matched what that path phase implies."
        )
    return (
        "Trust check for the sequential operational read. Available forward horizons ask whether realized "
        "next-window behavior matched what the path-constrained phase implies."
    )


def _path_reliability_callout_text(*, has_nowcast: bool) -> str:
    if has_nowcast:
        return (
            "Operational trust check: 0M measures whether the sequential path decoder agrees with same-date raw phase evidence. "
            "1M and longer horizons measure whether the following market window behaved the way that path phase implies. "
            "This uses prior phase memory, allowed transitions, duration, prior unwind/bottoming evidence, and drawdown preconditions."
        )
    return (
        "Forward-horizon trust check: available horizons measure whether the following market window behaved the way "
        "that path phase implies. This uses prior phase memory, allowed transitions, duration, prior unwind/bottoming "
        "evidence, and drawdown preconditions."
    )


def _path_reliability_chart_help_text(*, has_nowcast: bool) -> str:
    if has_nowcast:
        return (
            "For 0M, bars show same-date agreement between path-constrained state and raw evidence state. "
            "For 1M and longer, bars show historical hit rates for the phase's expected forward behavior."
        )
    return (
        "Bars show historical hit rates for the selected forward horizon and each path phase's expected behavior."
    )


def _render_cycle_reliability(reliability: pd.DataFrame, *, dominant_phase: str) -> None:
    render_section_header(
        "Historical Phase Reliability",
        help_text=(
            "Trust check for the raw evidence label before path-aware sequence rules. It asks whether "
            "historical evidence-only phase labels were followed by the behavior that phase implies."
        ),
    )
    st.caption(
        "Raw evidence audit: when Cycle Tracker evidence labeled a historical origin with phase X, did the next horizon behave the way that phase implies?"
    )
    data = reliability.copy()
    data["horizon_days"] = pd.to_numeric(data["horizon_days"], errors="coerce")
    data["phase_fit_rate"] = pd.to_numeric(data["phase_fit_rate"], errors="coerce")
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0).astype(int)
    horizon_order = (
        data[["horizon", "horizon_days"]]
        .drop_duplicates()
        .sort_values("horizon_days")["horizon"]
        .astype(str)
        .tolist()
    )
    selected_horizon = st.pills(
        "Reliability horizon",
        horizon_order,
        default="3m" if "3m" in horizon_order else (horizon_order[0] if horizon_order else None),
        selection_mode="single",
        key="dashboard_v2_cycle_reliability_horizon",
    )
    if not selected_horizon:
        return
    selected = data[data["horizon"].astype(str).eq(str(selected_horizon))].copy()
    if selected.empty:
        st.info("No phase reliability rows are available for this horizon.")
        return
    current = selected[selected["dominant_phase"].astype(str).eq(str(dominant_phase))]
    headline = (
        current.iloc[0]
        if not current.empty
        else selected.sort_values("origins", ascending=False).iloc[0]
    )
    is_nowcast = int(pd.to_numeric(pd.Series([headline.get("horizon_days")]), errors="coerce").fillna(-1).iloc[0]) == 0
    fit_label = "Nowcast Confidence" if is_nowcast else "Fit Rate"
    fit_help = (
        "Median raw evidence probability assigned to this phase on historical 0M nowcast origins. No forward return is measured."
        if is_nowcast
        else "How often this raw phase evidence label was followed by the expected forward behavior at the selected horizon."
    )
    render_card_grid(
        [
            (
                "Current Phase",
                dominant_phase,
                "Current raw evidence phase before path constraints. This can differ from the operational path phase.",
            ),
            (
                fit_label,
                _fmt_pct(headline.get("phase_fit_rate")),
                fit_help,
            ),
            (
                "Historical Origins",
                int(headline.get("origins", 0)),
                "Historical prior-only examples available for this raw phase and horizon. Thin samples should stay research-only.",
            ),
            (
                "Reliability Label",
                headline.get("reliability_label", "n/a"),
                "Reliability summary for the raw evidence classifier, not the path-aware operational read.",
            ),
        ]
    )
    if is_nowcast:
        render_callout(
            f"For 0M, raw `{headline.get('dominant_phase', dominant_phase)}` evidence had median nowcast confidence "
            f"{_fmt_pct(headline.get('phase_fit_rate'))} across {int(headline.get('origins', 0))} origins. "
            "This audits same-date classifier confidence before the sequential path rules; no forward outcome is scored.",
        )
    else:
        render_callout(
            f"For {selected_horizon}, raw `{headline.get('dominant_phase', dominant_phase)}` evidence fit history "
            f"{_fmt_pct(headline.get('phase_fit_rate'))} across {int(headline.get('origins', 0))} origins. "
            f"This audits the evidence classifier before the sequential path rules. Expected behavior: {headline.get('expected_behavior', 'n/a')}",
        )
    render_chart(
        _phase_reliability_figure(selected),
        title="Historical Phase Reliability",
        key=f"dashboard_v2_cycle_phase_reliability_chart_{selected_horizon}",
    )
    columns = [
        column
        for column in [
            "dominant_phase",
            "horizon",
            "origins",
            "phase_fit_rate",
            "median_phase_probability",
            "median_qqq_forward_return",
            "median_qqq_forward_drawdown",
            "severe_qqq_drawdown_rate",
            "reliability_label",
            "expected_behavior",
        ]
        if column in selected
    ]
    with st.expander("Reliability audit table", expanded=False):
        _render_metric_dataframe(_display_metrics(selected[columns]))


def _render_crisis_playback(crisis: pd.DataFrame) -> None:
    render_section_header("Historical Crisis Playback")
    st.caption(
        "Replay Cycle Tracker phase probabilities through named historical stress windows: lead-up, unwind, and recovery."
    )
    data = crisis.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"], errors="coerce")
    data["horizon_days"] = pd.to_numeric(data["horizon_days"], errors="coerce")
    data["phase_probability"] = pd.to_numeric(data["phase_probability"], errors="coerce")
    data["dominant_phase_probability"] = pd.to_numeric(
        data["dominant_phase_probability"],
        errors="coerce",
    )
    data["stage_order"] = pd.to_numeric(data["stage_order"], errors="coerce").fillna(0).astype(int)
    data["phase_fit"] = data["phase_fit"].astype(str).str.lower().isin({"true", "1", "yes"})
    crisis_options = data["crisis"].dropna().astype(str).drop_duplicates().tolist()
    if not crisis_options:
        return
    selected_crisis = st.selectbox(
        "Historical stress window",
        crisis_options,
        index=len(crisis_options) - 1,
        key="dashboard_v2_cycle_crisis_window",
    )
    horizon_options = (
        data[["horizon", "horizon_days"]]
        .drop_duplicates()
        .sort_values("horizon_days")["horizon"]
        .astype(str)
        .tolist()
    )
    selected_horizon = st.pills(
        "Playback horizon",
        horizon_options,
        default="3m" if "3m" in horizon_options else (horizon_options[0] if horizon_options else None),
        selection_mode="single",
        key="dashboard_v2_cycle_crisis_horizon",
    )
    if not selected_horizon:
        return
    selected = data[
        data["crisis"].astype(str).eq(str(selected_crisis))
        & data["horizon"].astype(str).eq(str(selected_horizon))
    ].copy()
    if selected.empty:
        st.info("No crisis playback rows are available for this selection.")
        return
    is_nowcast = selected["horizon_days"].eq(0).all()
    dominant = (
        selected[
            [
                "origin_date",
                "stage",
                "stage_order",
                "dominant_phase",
                "dominant_phase_probability",
                "phase_fit",
            ]
        ]
        .drop_duplicates()
        .sort_values("origin_date")
    )
    if is_nowcast:
        stage_summary = (
            dominant.groupby(["stage_order", "stage", "dominant_phase"])
            .agg(
                origins=("origin_date", "nunique"),
                avg_confidence=("dominant_phase_probability", "mean"),
            )
            .reset_index()
            .sort_values(["stage_order", "origins"], ascending=[True, False])
        )
        playback_metric = ("Avg Confidence", _fmt_pct(dominant["dominant_phase_probability"].mean()))
    else:
        stage_summary = (
            dominant.groupby(["stage_order", "stage", "dominant_phase"])
            .agg(origins=("origin_date", "nunique"), fit_rate=("phase_fit", "mean"))
            .reset_index()
            .sort_values(["stage_order", "origins"], ascending=[True, False])
        )
        playback_metric = ("Playback Fit", _fmt_pct(dominant["phase_fit"].mean()))
    render_card_grid(
        [
            ("Window", str(selected_crisis).replace("_", " ").title()),
            ("Origins", int(dominant["origin_date"].nunique())),
            ("Most Common Phase", dominant["dominant_phase"].mode().iloc[0]),
            playback_metric,
        ]
    )
    render_chart(
        _crisis_playback_figure(selected),
        title="Historical Crisis Playback",
        key=f"dashboard_v2_cycle_crisis_playback_chart_{selected_crisis}_{selected_horizon}",
    )
    with st.expander("Crisis stage summary", expanded=True):
        _render_metric_dataframe(_display_metrics(stage_summary))


def _phase_winner_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["frontier_score"] = pd.to_numeric(data["frontier_score"], errors="coerce").fillna(0.0)
    if "origins" not in data:
        data["origins"] = 0
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0).astype(int)
    for column in ["median_forward_return", "median_forward_drawdown"]:
        if column not in data:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["asset_role", "exposure_family", "evidence_quality", "evidence_flags"]:
        if column not in data:
            data[column] = "unknown"
        data[column] = data[column].fillna("unknown").astype(str)
    data = data.sort_values("frontier_score", ascending=True)
    color_map = {
        "scale_candidate": "#16a34a",
        "starter_reentry": "#2563eb",
        "scale_reentry": "#2563eb",
        "reentry_watch": "#8b5cf6",
        "thin_sample_watch": "#f97316",
        "watch": "#f59e0b",
        "defend": "#06b6d4",
        "ballast": "#0ea5e9",
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
            customdata=data[
                [
                    "asset_role",
                    "exposure_family",
                    "evidence_quality",
                    "evidence_flags",
                    "origins",
                    "median_forward_return",
                    "median_forward_drawdown",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>Frontier score: %{x:.2f}<br>"
                "Role: %{text}<br>"
                "Asset role: %{customdata[0]}<br>"
                "Family: %{customdata[1]}<br>"
                "Evidence: %{customdata[2]} (%{customdata[4]} origins)<br>"
                "Flags: %{customdata[3]}<br>"
                "Median return: %{customdata[5]:.1%}<br>"
                "Median drawdown: %{customdata[6]:.1%}<extra></extra>"
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


def _slice_horizon_phase(
    frame: pd.DataFrame,
    *,
    horizon: str,
    phase_column: str,
    phase: str,
) -> pd.DataFrame:
    if frame.empty or "horizon" not in frame or phase_column not in frame:
        return pd.DataFrame()
    return frame[
        frame["horizon"].astype(str).eq(str(horizon))
        & frame[phase_column].astype(str).eq(str(phase))
    ].copy()


def _phase_behavior_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "origins": 0.0,
            "median_forward_return": float("nan"),
            "median_forward_drawdown": float("nan"),
            "median_excess_vs_spy": float("nan"),
            "median_excess_vs_qqq": float("nan"),
            "hit_rate_vs_qqq": float("nan"),
        }
    summary: dict[str, float] = {}
    origins = (
        pd.to_numeric(frame["origins"], errors="coerce").dropna()
        if "origins" in frame
        else pd.Series(dtype=float)
    )
    summary["origins"] = float(origins.max()) if not origins.empty else float(len(frame))
    for column in [
        "median_forward_return",
        "median_forward_drawdown",
        "median_excess_vs_spy",
        "median_excess_vs_qqq",
        "hit_rate_vs_qqq",
    ]:
        values = (
            pd.to_numeric(frame[column], errors="coerce").dropna()
            if column in frame
            else pd.Series(dtype=float)
        )
        summary[column] = float(values.median()) if not values.empty else float("nan")
    return summary


def _phase_outcome_profile_figure(frame: pd.DataFrame, *, phase: str) -> go.Figure | None:
    if frame.empty:
        return None
    summary = _phase_behavior_summary(frame)
    metrics = [
        ("Forward return", "median_forward_return", "#16a34a"),
        ("Excess vs SPY", "median_excess_vs_spy", "#06b6d4"),
        ("Excess vs QQQ", "median_excess_vs_qqq", "#2563eb"),
        ("Forward drawdown", "median_forward_drawdown", "#ef4444"),
    ]
    labels = []
    values = []
    colors = []
    for label, column, color in metrics:
        value = summary.get(column)
        if value is None or pd.isna(value):
            continue
        labels.append(label)
        values.append(value)
        colors.append(color)
    if not labels:
        return None
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="#7f8ea3", line_width=1)
    hit_rate = summary.get("hit_rate_vs_qqq")
    title_suffix = f" | QQQ hit rate {_fmt_pct(hit_rate)}" if hit_rate is not None and not pd.isna(hit_rate) else ""
    figure.update_layout(
        title_text=f"{phase.replace('_', ' ').title()} outcome profile{title_suffix}",
        yaxis_title="Median outcome",
        yaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        height=340,
        showlegend=False,
    )
    return figure


def _phase_reliability_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["phase_fit_rate"] = pd.to_numeric(data["phase_fit_rate"], errors="coerce").fillna(0.0)
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0.0)
    if "horizon_days" in data:
        data["horizon_days"] = pd.to_numeric(data["horizon_days"], errors="coerce")
        is_nowcast = data["horizon_days"].eq(0).all()
    else:
        is_nowcast = False
    metric_label = "Nowcast confidence" if is_nowcast else "Phase-fit rate"
    data = data.sort_values("phase_fit_rate", ascending=True)
    color_map = {
        "historically_supportive": "#16a34a",
        "mixed_but_useful": "#f59e0b",
        "weak_or_context_only": "#ef4444",
        "not_reliable": "#991b1b",
        "thin_sample": "#7f8ea3",
        "nowcast_confidence": "#0f766e",
    }
    figure = go.Figure(
        go.Bar(
            x=data["phase_fit_rate"],
            y=data["dominant_phase"].astype(str),
            orientation="h",
            marker_color=[
                color_map.get(str(label), "#7f8ea3")
                for label in data.get("reliability_label", pd.Series(dtype=str)).astype(str)
            ],
            customdata=data[["origins", "reliability_label"]].to_numpy(),
            hovertemplate=(
                f"<b>%{{y}}</b><br>{metric_label}: %{{x:.1%}}<br>"
                "Origins: %{customdata[0]:.0f}<br>Label: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis_title=metric_label,
        xaxis_tickformat=".0%",
        yaxis_title="Cycle phase",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=360,
    )
    return figure


def _path_reliability_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["path_fit_rate"] = pd.to_numeric(data["path_fit_rate"], errors="coerce").fillna(0.0)
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0.0)
    data = data.sort_values("path_fit_rate", ascending=True)
    color_map = {
        "historically_supportive": "#16a34a",
        "mixed_but_useful": "#f59e0b",
        "weak_or_context_only": "#ef4444",
        "not_reliable": "#991b1b",
        "thin_sample": "#7f8ea3",
    }
    figure = go.Figure(
        go.Bar(
            x=data["path_fit_rate"],
            y=data["path_phase"].astype(str),
            orientation="h",
            marker_color=[
                color_map.get(str(label), "#7f8ea3")
                for label in data.get("reliability_label", pd.Series(dtype=str)).astype(str)
            ],
            customdata=data[["origins", "reliability_label"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Path fit rate: %{x:.1%}<br>"
                "Origins: %{customdata[0]:.0f}<br>Label: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis_title="Path phase-fit rate",
        xaxis_tickformat=".0%",
        yaxis_title="Path phase",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=360,
    )
    return figure


def _crisis_playback_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["horizon_days"] = pd.to_numeric(data.get("horizon_days"), errors="coerce")
    data = data.sort_values(["origin_date", "stage_order"])
    is_nowcast = data["horizon_days"].eq(0).all()
    horizon = (
        data["horizon"].dropna().astype(str).iloc[0]
        if "horizon" in data and not data["horizon"].dropna().empty
        else "selected"
    )
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
    dates = data["origin_date"].drop_duplicates().tolist()
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            "Phase read at each historical origin",
            "Nowcast confidence at each historical origin"
            if is_nowcast
            else f"What happened over the selected {horizon} horizon",
        ),
    )
    for phase in phase_order:
        phase_rows = data[data["phase"].astype(str).eq(phase)]
        if phase_rows.empty:
            continue
        y_values = []
        for date in dates:
            row = phase_rows[phase_rows["origin_date"].eq(date)]
            y_values.append(float(row["phase_probability"].iloc[0]) if not row.empty else 0.0)
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=y_values,
                mode="lines",
                stackgroup="one",
                name=phase.replace("_", " ").title(),
                line={"color": color_map.get(phase, "#7f8ea3"), "width": 1},
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    outcome_columns = [
        column
        for column in [
            "origin_date",
            "qqq_forward_return",
            "spy_forward_return",
            "bil_forward_return",
            "qqq_forward_drawdown",
            "dominant_phase",
            "dominant_phase_probability",
            "phase_fit",
        ]
        if column in data
    ]
    outcomes = data[outcome_columns].drop_duplicates().sort_values("origin_date")
    outcome_series = (
        [("Dominant phase confidence", "dominant_phase_probability", "#0f766e")]
        if is_nowcast
        else [
            ("QQQ forward return", "qqq_forward_return", "#2563eb"),
            ("SPY forward return", "spy_forward_return", "#16a34a"),
            ("BIL forward return", "bil_forward_return", "#64748b"),
            ("QQQ max drawdown", "qqq_forward_drawdown", "#ef4444"),
        ]
    )
    for name, column, color in outcome_series:
        if column not in outcomes:
            continue
        values = pd.to_numeric(outcomes[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=outcomes["origin_date"],
                y=values,
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 5},
                customdata=outcomes.reindex(
                    columns=["dominant_phase", "phase_fit"],
                    fill_value="n/a",
                ).to_numpy(),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"{name}: " + "%{y:.1%}<br>"
                    "Dominant phase: %{customdata[0]}<br>"
                    + (
                        "<extra></extra>"
                        if is_nowcast
                        else "Phase fit: %{customdata[1]}<extra></extra>"
                    )
                ),
            ),
            row=2,
            col=1,
        )
    stage_rows = (
        data[["stage", "stage_order", "origin_date"]]
        .drop_duplicates()
        .groupby(["stage_order", "stage"])
        .agg(start=("origin_date", "min"), end=("origin_date", "max"))
        .reset_index()
        .sort_values("stage_order")
    )
    shapes = []
    annotations = []
    for position, row in enumerate(stage_rows.itertuples(index=False)):
        color = "rgba(245, 158, 11, 0.08)" if position % 2 == 0 else "rgba(37, 99, 235, 0.06)"
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": row.start,
                "x1": row.end,
                "y0": 0,
                "y1": 1,
                "line": {"width": 0},
                "fillcolor": color,
                "layer": "below",
            }
        )
        stage_midpoint = row.start
        try:
            stage_start = pd.to_datetime(row.start)
            stage_end = pd.to_datetime(row.end)
            stage_midpoint = stage_start + (stage_end - stage_start) / 2
        except (TypeError, ValueError):
            stage_midpoint = row.start
        annotations.append(
            {
                "xref": "x",
                "yref": "paper",
                "x": stage_midpoint,
                "y": 1.055,
                "text": str(row.stage).replace("_", " ").title(),
                "showarrow": False,
                "font": {"size": 11},
                "xanchor": "center",
                "yanchor": "bottom",
            }
        )
    subplot_annotations = []
    for annotation in tuple(figure.layout.annotations or ()):
        annotation_config = annotation.to_plotly_json()
        if annotation_config.get("text") == "Phase read at each historical origin":
            annotation_config.update({"y": 1.12, "yanchor": "bottom"})
        subplot_annotations.append(annotation_config)
    figure.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Phase probability",
        yaxis2_tickformat=".0%",
        yaxis2_title="Dominant phase confidence"
        if is_nowcast
        else f"{horizon} return / drawdown",
        xaxis2_title="Historical origin date",
        legend_title_text="Series",
        shapes=shapes,
        annotations=list(subplot_annotations) + annotations,
        margin={"l": 20, "r": 20, "t": 82, "b": 20},
        height=680,
    )
    return figure


def _phase_frontier_figure(
    frame: pd.DataFrame,
    *,
    selected_horizon: str | None = None,
    selected_phase: str | None = None,
) -> go.Figure:
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
    phase_values: dict[str, list[float]] = {}
    for phase in phase_order:
        phase_rows = data[data["phase"].astype(str).eq(phase)]
        if phase_rows.empty:
            continue
        y_values = []
        for horizon in horizons:
            row = phase_rows[phase_rows["horizon"].astype(str).eq(horizon)]
            y_values.append(float(row["probability"].iloc[0]) if not row.empty else 0.0)
        phase_values[phase] = y_values
        line_widths = [
            3
            if selected_horizon
            and selected_phase
            and str(horizon) == str(selected_horizon)
            and phase == selected_phase
            else 0
            for horizon in horizons
        ]
        figure.add_trace(
            go.Bar(
                x=horizons,
                y=y_values,
                name=phase.replace("_", " ").title(),
                marker_color=color_map.get(phase),
                marker_line_color="#f8fafc",
                marker_line_width=line_widths,
                customdata=[
                    [phase.replace("_", " ").title(), horizon]
                    for horizon in horizons
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Horizon: %{customdata[1]}<br>"
                    "Probability: %{y:.1%}<extra></extra>"
                ),
            )
        )
    if (
        selected_horizon
        and selected_phase
        and str(selected_horizon) in horizons
        and selected_phase in phase_values
    ):
        horizon_index = horizons.index(str(selected_horizon))
        selected_value = phase_values[selected_phase][horizon_index]
        if selected_value > 0:
            lower = 0.0
            for phase in phase_order:
                if phase == selected_phase:
                    break
                lower += phase_values.get(phase, [0.0] * len(horizons))[horizon_index]
            midpoint = lower + selected_value / 2
            figure.add_trace(
                go.Scatter(
                    x=[str(selected_horizon)],
                    y=[midpoint],
                    mode="markers+text",
                    name="Selected phase slice",
                    marker={
                        "symbol": "diamond",
                        "size": 14,
                        "color": "#f8fafc",
                        "line": {"color": "#0f172a", "width": 2},
                    },
                    text=[f"{selected_phase.replace('_', ' ')}<br>{selected_value:.1%}"],
                    textposition="middle right",
                    hovertemplate=(
                        "<b>Selected slice</b><br>"
                        f"Phase: {selected_phase}<br>"
                        f"Horizon: {selected_horizon}<br>"
                        "Probability: %{customdata:.1%}<extra></extra>"
                    ),
                    customdata=[selected_value],
                    showlegend=False,
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
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.2%}"


def _fmt_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.2f}"
