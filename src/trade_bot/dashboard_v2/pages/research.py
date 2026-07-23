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
    defensive_signal_audit_frames,
    i111_evidence_frames,
    leadership_frames,
    pbo_frames,
    prebreak_hindsight_frames,
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
            ("Metric Contract", _metric_contract_label(candidates)),
            ("Execution", _execution_contract_label(candidates)),
        ]
    )
    render_callout(
        "Every displayed candidate is evaluated under the single contract shown above. "
        "The page fails closed instead of merging stale scorecards."
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
            include_defensive_judgement=False,
            include_candidate_diagnostics=False,
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


def _metric_contract_label(frame: pd.DataFrame) -> str:
    if frame.empty or "evaluation_contract_sha256" not in frame:
        return "unavailable"
    values = frame["evaluation_contract_sha256"].dropna().astype(str).unique()
    return values[0][:10] if len(values) == 1 else "MIXED - BLOCKED"


def _execution_contract_label(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unavailable"
    lags = pd.to_numeric(
        frame.get("signal_lag_days", pd.Series(dtype=float)), errors="coerce"
    ).dropna().unique()
    rebalances = frame.get("rebalance_frequency", pd.Series(dtype=str)).dropna().astype(str).unique()
    if len(lags) != 1 or len(rebalances) != 1:
        return "MIXED - BLOCKED"
    return f"lag {int(lags[0])} / {rebalances[0]}"


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
        "Candidate Deep Dive uses the canonical replay library when it is available. "
        "Configured strategies are replayed in that same library, so live-snapshot metrics "
        "cannot silently override comparable research results.",
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
    """Return the canonical V2 candidate set, with runtime metrics only as a fallback."""

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
        active = candidates[~candidates["research_status"].astype(str).eq("pruned_dead_end")].copy()
        if not active.empty:
            candidates = active
    return candidates.reset_index(drop=True)


def _render_candidate_artifact_read(strategy_name: str) -> None:
    pbo = pbo_frames()
    selection = pbo.get("selection", pd.DataFrame())
    stats = pbo.get("stats", pd.DataFrame())
    with st.expander("PBO / overfit artifact read", expanded=False):
        st.caption(
            "PBO applies only to the declared candidate shelf in the persisted study. It does "
            "not correct for abandoned families or the full adaptive research history."
        )
        if selection.empty and stats.empty:
            st.info("No PBO artifacts found. Run `poetry run trade-bot audit-backtest-pbo`.")
            return
        matches = []
        for frame in [selection, stats]:
            if not frame.empty and "strategy" in frame:
                matches.append(frame[frame["strategy"].astype(str) == strategy_name])
        combined = (
            pd.concat([match for match in matches if not match.empty], ignore_index=True)
            if matches
            else pd.DataFrame()
        )
        if combined.empty:
            st.info("This candidate is not present in the latest PBO artifact set.")
        else:
            _render_metric_dataframe(_display_metrics(combined))
    if "i111" in strategy_name.lower():
        evidence = i111_evidence_frames()
        with st.expander("I111 evidence dossier", expanded=False):
            matched = False
            for label, key in (
                ("Native metrics", "native_metrics"),
                ("Adversarial robustness", "adversarial_robustness"),
                ("QC headline", "qc_headline"),
            ):
                frame = evidence.get(key, pd.DataFrame())
                if frame.empty:
                    continue
                name_column = next(
                    (column for column in ("result_name", "strategy", "name") if column in frame),
                    None,
                )
                selected = (
                    frame[frame[name_column].astype(str).eq(strategy_name)]
                    if name_column
                    else pd.DataFrame()
                )
                if selected.empty:
                    continue
                matched = True
                st.caption(label)
                _render_metric_dataframe(_display_metrics(selected.head(10)))
            smoothing = evidence.get("smoothing_gates", pd.DataFrame())
            if (
                strategy_name == "i111_native_risk_repair_guard17_relief85_ai85_div"
                and not smoothing.empty
            ):
                matched = True
                st.caption("Fixed-slate execution-smoothing challengers")
                _render_metric_dataframe(_display_metrics(smoothing))
            if not matched:
                st.info("No strategy-specific V2.2/V2.3 evidence rows are persisted yet.")


def _render_validation_artifacts() -> None:
    render_section_header("Validation Artifacts")
    pbo = pbo_frames()
    leadership = leadership_frames()
    i111 = i111_evidence_frames()
    if not pbo["summary"].empty:
        render_section_header("PBO Summary")
        st.caption(
            "Within-shelf selection-risk diagnostic only. Consult the governance trial ledger "
            "before interpreting this as research-wide overfit evidence."
        )
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
    if any(not frame.empty for frame in i111.values()):
        render_section_header("I111 V2.2 / V2.3 Evidence Dossier")
        render_callout(
            "Readiness comes from persisted QC, adversarial, execution, PBO, and provenance evidence. Configuration presence alone does not imply paper or promotion readiness."
        )
        provenance_warning = _provenance_integrity_warning(i111.get("manifests", pd.DataFrame()))
        if provenance_warning:
            st.warning(provenance_warning)
        for label, key in (
            ("Fixed-slate smoothing gates", "smoothing_gates"),
            ("Execution-hardening mechanisms", "execution_mechanisms"),
            ("Adversarial gaps", "adversarial_gaps"),
            ("Family smoothing PBO (15 strategies / 70 splits)", "smoothing_pbo"),
            ("Artifact provenance", "manifests"),
        ):
            frame = i111.get(key, pd.DataFrame())
            if frame.empty:
                continue
            st.caption(label)
            _render_metric_dataframe(_display_metrics(frame.head(40)))
    if all(frame.empty for frame in [*pbo.values(), *leadership.values(), *i111.values()]):
        st.info("No validation artifacts found yet.")


def _provenance_integrity_warning(manifests: pd.DataFrame) -> str | None:
    if manifests.empty:
        return None
    statuses = manifests.get(
        "artifact_integrity_status",
        pd.Series("unverified_no_hashes", index=manifests.index),
    ).astype(str)
    source_statuses = manifests.get(
        "source_tree_status",
        pd.Series("unavailable", index=manifests.index),
    ).astype(str)
    missing = int(
        pd.to_numeric(
            manifests.get("missing_artifact_count", pd.Series(0, index=manifests.index)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    mismatched = int(
        pd.to_numeric(
            manifests.get("artifact_mismatch_count", pd.Series(0, index=manifests.index)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    unverified_manifests = int(statuses.ne("verified").sum())
    stale_manifests = int(source_statuses.eq("stale").sum())
    if not any((missing, mismatched, unverified_manifests, stale_manifests)):
        return None
    parts = []
    if missing:
        parts.append(f"{missing} declared artifact(s) missing")
    if mismatched:
        parts.append(f"{mismatched} artifact hash/size mismatch(es)")
    if unverified_manifests:
        parts.append(f"{unverified_manifests} manifest(s) not fully hash-verified")
    if stale_manifests:
        parts.append(f"{stale_manifests} manifest(s) generated from a different source tree")
    return (
        "Provenance warning: "
        + "; ".join(parts)
        + ". Regenerate the affected study before treating its dossier as current evidence."
    )


def _render_cycle_tracker() -> None:
    render_section_header("Cycle Path Frontier")
    st.caption(
        "Zoom-out research layer for speculative-cycle path phases, horizon probabilities, "
        "and historical analog winners. This view reads persisted artifacts only."
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
    prebreak = prebreak_hindsight_frames()
    defensive_audit = defensive_signal_audit_frames()
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
        "Cycle Tracker is a zoom-out cycle map, not a risk-budget or allocation override. It asks which larger speculative-cycle phase the current market resembles, which phases are plausible by horizon, and which assets historically worked in similar prior states.",
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
        _render_crisis_playback(
            crisis,
            prebreak=prebreak,
            defensive_audit=defensive_audit,
        )
    else:
        st.info(
            "This cycle tracker run does not include crisis playback yet. Re-run `poetry run trade-bot run-cycle-tracker`."
        )

    if path_forecast.empty and forecast.empty and not phase.empty:
        render_section_header("0M Nowcast Phase Probabilities")
        render_chart(
            _phase_frontier_figure(phase),
            title="Cycle Phase Nowcast Frontier",
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
            "exposure_family",
            "candidate_role",
            "candidate_score",
            "phase_distinctiveness",
            "ubiquity_penalty",
            "cycle_leadership_fragility",
            "theme_fragility_penalty",
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
        f"Current-phase analog candidates ({len(candidates):,})",
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
    duration_days = (
        pd.to_numeric(
            pd.Series([latest.get("phase_duration_days", 0)]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )
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
        "Cycle Path Behavior Inspector",
        help_text=(
            "Select a horizon and phase from the path-aware forecast to inspect expected return, "
            "drawdown, benchmark excess, and the historical analog winner set for that slice."
        ),
    )
    st.caption(
        "Use this paired view to ask: if this larger cycle phase dominates, what has usually happened next and which assets ranked best in similar prior states?"
    )
    forecast = path_forecast.copy()
    if "horizon_days" in forecast:
        forecast["horizon_days"] = pd.to_numeric(forecast["horizon_days"], errors="coerce")
    else:
        forecast["horizon_days"] = range(len(forecast))
    if "probability" in forecast:
        forecast["probability"] = pd.to_numeric(forecast["probability"], errors="coerce").fillna(
            0.0
        )
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
        str(
            frontier_slice.sort_values(["rank", "frontier_score"], ascending=[True, False]).iloc[0][
                "ticker"
            ]
        )
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
                "Highest-ranked ticker in the historical analog winner frontier for this phase.",
            ),
        ]
    )
    render_callout(
        f"For {selected_horizon} `{selected_phase}`, the inspector combines the current path probability "
        "with prior-only historical outcome behavior and the current historical analog winner frontier. Treat this as cycle analog research, not an allocation override."
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
                st.info("No historical analog winner frontier rows are available for this slice.")
            else:
                render_chart(
                    _phase_winner_figure(
                        frontier_slice.sort_values(
                            ["rank", "frontier_score"],
                            ascending=[True, False],
                        ).head(8)
                    ),
                    title="Historical Analog Winner Frontier",
                    help_text=(
                        "Ranks assets that historically did best in similar phase/horizon settings, then downgrades thin samples, non-specific winners, and current-cycle leadership fragility."
                    ),
                    key=f"dashboard_v2_cycle_behavior_winner_chart_{selected_horizon}_{selected_phase}",
                )
    elif not frontier_slice.empty:
        render_chart(
            _phase_winner_figure(
                frontier_slice.sort_values(
                    ["rank", "frontier_score"], ascending=[True, False]
                ).head(8)
            ),
            title="Historical Analog Winner Frontier",
            help_text=(
                "Ranks assets that historically did best in similar phase/horizon settings, then downgrades thin samples, non-specific winners, and current-cycle leadership fragility."
            ),
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
            _render_metric_dataframe(
                _display_metrics(validation_slice[validation_columns].head(40))
            )
        if not frontier_slice.empty:
            st.caption(
                "Historical analog winner frontier for the selected phase and horizon, with current-cycle robustness guards."
            )
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
                    "phase_distinctiveness",
                    "ubiquity_penalty",
                    "cycle_leadership_fragility",
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
    render_callout(_path_reliability_callout_text(has_nowcast=bool(has_nowcast)))
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
    is_nowcast = (
        int(
            pd.to_numeric(pd.Series([headline.get("horizon_days")]), errors="coerce")
            .fillna(-1)
            .iloc[0]
        )
        == 0
    )
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
    return "Bars show historical hit rates for the selected forward horizon and each path phase's expected behavior."


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
    is_nowcast = (
        int(
            pd.to_numeric(pd.Series([headline.get("horizon_days")]), errors="coerce")
            .fillna(-1)
            .iloc[0]
        )
        == 0
    )
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


def _render_crisis_playback(
    crisis: pd.DataFrame,
    *,
    prebreak: dict[str, pd.DataFrame] | None = None,
    defensive_audit: dict[str, pd.DataFrame] | None = None,
) -> None:
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
        default=(
            "3m" if "3m" in horizon_options else (horizon_options[0] if horizon_options else None)
        ),
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
        playback_metric = (
            "Avg Confidence",
            _fmt_pct(dominant["dominant_phase_probability"].mean()),
        )
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
    playback_tickers = _available_crisis_playback_tickers(selected)
    if playback_tickers:
        default_tickers = [
            ticker for ticker in ["QQQ", "SPY", "VTI"] if ticker in playback_tickers
        ] or playback_tickers[:3]
        selected_tickers = st.multiselect(
            "Playback market tickers",
            playback_tickers,
            default=default_tickers,
            key=f"dashboard_v2_cycle_crisis_playback_tickers_{selected_crisis}_{selected_horizon}",
        )
        if selected_tickers:
            render_chart(
                _crisis_playback_market_figure(selected, tickers=selected_tickers),
                title="Historical Crisis Market Path",
                key=(
                    "dashboard_v2_cycle_crisis_market_path_chart_"
                    f"{selected_crisis}_{selected_horizon}"
                ),
            )
    with st.expander("Crisis stage summary", expanded=True):
        _render_metric_dataframe(_display_metrics(stage_summary))
    if prebreak:
        _render_prebreak_hindsight_layers(
            selected_crisis=str(selected_crisis),
            prebreak=prebreak,
            defensive_audit=defensive_audit or {},
        )


def _render_prebreak_hindsight_layers(
    *,
    selected_crisis: str,
    prebreak: dict[str, pd.DataFrame],
    defensive_audit: dict[str, pd.DataFrame],
) -> None:
    signal_panel = prebreak.get("snapshot_signal_panel", pd.DataFrame())
    signal_rank = prebreak.get("signal_predictiveness_rank", pd.DataFrame())
    action_timing = prebreak.get("action_timing", pd.DataFrame())
    staged_risk = prebreak.get("staged_risk_behavior", pd.DataFrame())
    late_trigger = prebreak.get("late_trigger_mesh", pd.DataFrame())
    hard_defense_attribution = prebreak.get("hard_defense_attribution", pd.DataFrame())
    policy_variants = prebreak.get("policy_variant_results", pd.DataFrame())
    current_readout = prebreak.get("current_best_signal_readout", pd.DataFrame())
    if all(
        frame.empty
        for frame in [
            signal_panel,
            signal_rank,
            action_timing,
            staged_risk,
            late_trigger,
            hard_defense_attribution,
            policy_variants,
            current_readout,
        ]
    ):
        return

    render_section_header(
        "Pre-Break Behavior And Early Warning",
        help_text=(
            "Historical snapshot layer for checking whether trade-bot became defensive before "
            "known bubble breaks, which signals carried the warning, and whether today's read "
            "resembles normal behavior or pre-break defensive behavior."
        ),
    )
    render_callout(
        "This layer uses hindsight-labeled historical snapshots. It does not predict a crash by "
        "itself; it shows whether the same signals that helped before are quiet, mixed, or elevated now."
    )
    _render_prebreak_population_summary(signal_panel)
    if not current_readout.empty:
        _render_current_prebreak_monitor(current_readout, defensive_audit=defensive_audit)
    if not signal_panel.empty or not action_timing.empty:
        render_section_header("Selected Crisis Trade-Bot Behavior")
        selected_event = _select_prebreak_event(
            selected_crisis=selected_crisis,
            signal_panel=signal_panel,
            action_timing=action_timing,
        )
        _render_selected_prebreak_event_behavior(
            selected_crisis=selected_event,
            signal_panel=signal_panel,
            action_timing=action_timing,
        )
        _render_prebreak_margin_experiment(
            selected_crisis=selected_event,
            staged_risk=staged_risk,
            late_trigger=late_trigger,
            hard_defense_attribution=hard_defense_attribution,
            policy_variants=policy_variants,
        )
    if not signal_rank.empty:
        _render_prebreak_signal_attribution(signal_rank, current_readout=current_readout)


def _render_prebreak_population_summary(signal_panel: pd.DataFrame) -> None:
    if signal_panel.empty:
        st.warning(
            "The pre-break report has no snapshot population. Rebuild the historical event and "
            "ordinary-control origins before interpreting this section."
        )
        return
    population = _prebreak_population_summary(signal_panel)
    render_section_header(
        "Historical Sample Population",
        help_text=(
            "Coverage behind the hindsight layer. Origins overlap at the three-month outcome "
            "horizon, so conservative event/control clusters are more meaningful than the raw row count."
        ),
    )
    render_card_grid(
        [
            ("Historical Origins", population["origins"]),
            ("Mature 3m Outcomes", population["mature_outcomes"]),
            ("Named Events", population["named_events"]),
            ("Ordinary Controls", population["ordinary_controls"]),
            (
                "Conservative Clusters",
                population["population_clusters"],
                "Named events count once; ordinary controls are grouped by calendar quarter.",
            ),
            ("Date Range", population["date_range"]),
        ]
    )
    render_callout(
        f"Population balance: {population['event_origins']} event-window origins and "
        f"{population['ordinary_controls']} ordinary-history controls. Treat the "
        f"{population['population_clusters']} event/quarter clusters as the conservative evidence "
        "count; the raw origins are overlapping observations, not independent trials."
    )


def _prebreak_population_summary(signal_panel: pd.DataFrame) -> dict[str, object]:
    data = signal_panel.copy()
    dates = pd.to_datetime(
        data.get("market_date", pd.Series(index=data.index, dtype=object)),
        errors="coerce",
    )
    event_names = data.get("event_name", pd.Series("", index=data.index)).fillna("").astype(str)
    event_mask = event_names.str.strip().ne("")
    roles = data.get("population_role", pd.Series("", index=data.index)).fillna("").astype(str)
    control_mask = roles.eq("historical_control") | ~event_mask
    if "population_cluster" in data:
        clusters = data["population_cluster"].fillna("").astype(str)
    else:
        clusters = pd.Series("", index=data.index, dtype=object)
        clusters.loc[event_mask] = "event:" + event_names.loc[event_mask]
        clusters.loc[control_mask] = (
            "control:" + dates.loc[control_mask].dt.to_period("Q").astype(str)
        )
    mature = pd.to_numeric(
        data.get("break_severity_3m", pd.Series(index=data.index, dtype=float)),
        errors="coerce",
    ).notna()
    valid_dates = dates.dropna()
    date_range = (
        f"{valid_dates.min().date()} to {valid_dates.max().date()}"
        if not valid_dates.empty
        else "n/a"
    )
    return {
        "origins": int(len(data)),
        "mature_outcomes": int(mature.sum()),
        "named_events": int(event_names.loc[event_mask].nunique()),
        "event_origins": int(event_mask.sum()),
        "ordinary_controls": int(control_mask.sum()),
        "population_clusters": int(clusters[clusters.str.strip().ne("")].nunique()),
        "date_range": date_range,
    }


def _select_prebreak_event(
    *,
    selected_crisis: str,
    signal_panel: pd.DataFrame,
    action_timing: pd.DataFrame,
) -> str:
    event_options = _prebreak_event_options(signal_panel, action_timing)
    if not event_options:
        return selected_crisis
    default_event = _default_prebreak_event_for_crisis(selected_crisis, event_options)
    selected = st.selectbox(
        "Pre-break event to inspect",
        event_options,
        index=event_options.index(default_event),
        format_func=lambda value: str(value).replace("_", " ").title(),
        key="dashboard_v2_prebreak_event_to_inspect",
        help=(
            "This selector controls the hindsight snapshot behavior panel below. "
            "It is separate from the Historical stress window selector above, which controls "
            "the Cycle Tracker crisis playback."
        ),
    )
    return str(selected)


def _prebreak_event_options(
    signal_panel: pd.DataFrame,
    action_timing: pd.DataFrame,
) -> list[str]:
    events: list[str] = []
    for frame in [action_timing, signal_panel]:
        if frame.empty or "event_name" not in frame:
            continue
        for value in frame["event_name"].dropna().astype(str).drop_duplicates().tolist():
            if value.startswith("ALL_SEVERE_"):
                continue
            if value not in events:
                events.append(value)
    return events


def _default_prebreak_event_for_crisis(
    selected_crisis: str,
    event_options: list[str],
) -> str:
    aliases = {
        "global_financial_crisis": "gfc_credit_bubble_peak",
        "q4_2018_tightening": "q4_2018_liquidity_break",
        "covid_liquidity_crash": "covid_crash_peak",
        "inflation_tech_unwind": "inflation_rates_growth_peak",
    }
    preferred = aliases.get(str(selected_crisis), "")
    if preferred in event_options:
        return preferred
    return event_options[0]


def _render_current_prebreak_monitor(
    current_readout: pd.DataFrame,
    *,
    defensive_audit: dict[str, pd.DataFrame],
) -> None:
    render_section_header("Current Early Warning Monitor")
    data = current_readout.copy()
    for column in [
        "latest_value",
        "historical_percentile",
        "predictive_score",
        "spearman_to_break_severity",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    latest_date = (
        str(data["market_date"].dropna().iloc[0])
        if "market_date" in data and not data["market_date"].dropna().empty
        else "n/a"
    )
    read_counts = (
        data["current_risk_read"].fillna("unknown").astype(str).str.lower().value_counts()
        if "current_risk_read" in data
        else pd.Series(dtype=int)
    )
    top = (
        data.sort_values("predictive_score", ascending=False).iloc[0]
        if "predictive_score" in data and not data.empty
        else pd.Series(dtype=object)
    )
    top_read = str(top.get("current_risk_read", "n/a"))
    render_card_grid(
        [
            (
                "Read Date",
                latest_date,
                "Latest market date in the current pre-break signal readout.",
            ),
            (
                "Top Monitor",
                _signal_label(top.get("signal", "n/a")),
                "Highest-ranked current signal from the pre-break hindsight monitor set.",
                _risk_read_tone(top_read),
            ),
            (
                "Top Read",
                top_read.title(),
                "Current interpretation for the highest-ranked pre-break monitor.",
                _risk_read_tone(top_read),
            ),
            (
                "High Risk",
                int(read_counts.get("high_risk", 0)),
                "Count of best historical monitors currently in the top risk percentile band.",
                "critical" if int(read_counts.get("high_risk", 0)) else "neutral",
            ),
            (
                "Elevated",
                int(read_counts.get("elevated", 0)),
                "Count of best historical monitors currently reading elevated.",
                "critical" if int(read_counts.get("elevated", 0)) else "neutral",
            ),
            (
                "Mixed",
                int(read_counts.get("mixed", 0)),
                "Count of best historical monitors currently reading mixed.",
                "warning" if int(read_counts.get("mixed", 0)) else "neutral",
            ),
            (
                "Contained",
                int(read_counts.get("contained", 0)),
                "Count of best historical monitors currently reading contained.",
                "success" if int(read_counts.get("contained", 0)) else "neutral",
            ),
        ]
    )
    if not data.empty:
        columns = [
            column
            for column in [
                "signal",
                "market_date",
                "latest_value",
                "historical_percentile",
                "risk_direction",
                "current_risk_read",
                "predictive_score",
                "spearman_to_break_severity",
            ]
            if column in data
        ]
        display = data[columns].head(12).copy()
        if "signal" in display:
            display["signal"] = display["signal"].map(_signal_label)
        with st.expander("Current monitor detail", expanded=True):
            _render_metric_dataframe(_display_metrics(display))
    _render_defensive_posture_bridge(defensive_audit)


def _render_defensive_posture_bridge(defensive_audit: dict[str, pd.DataFrame]) -> None:
    exposure = defensive_audit.get("current_defensive_exposure", pd.DataFrame())
    scorecards = defensive_audit.get("scorecards", pd.DataFrame())
    if exposure.empty and scorecards.empty:
        return
    render_section_header(
        "Defensive Posture Cross-Check",
        help_text=(
            "Separate audit for whether current high defensive weights historically avoided "
            "drawdowns or acted as false alarms. Use this beside, not instead of, pre-break monitors."
        ),
    )
    exposure_row, score_row = _defensive_posture_rows(exposure, scorecards)
    episode_count = pd.to_numeric(
        pd.Series([score_row.get("defensive_episode_starts")]), errors="coerce"
    ).iloc[0]
    raw_label = str(score_row.get("defensive_judgement_label", "n/a"))
    render_card_grid(
        [
            (
                "Current Risk Weight",
                _fmt_pct(exposure_row.get("current_risk_weight")),
                "Current risk-asset weight from the defensive signal audit.",
            ),
            (
                "Current Defensive",
                _fmt_pct(exposure_row.get("current_defensive_weight")),
                "Current defensive or cash-like weight from the defensive signal audit.",
            ),
            (
                "Historical 65%+ Read",
                _defensive_label_display(raw_label),
                "Historical quality label for the focus strategy's 65%+ defensive episodes.",
            ),
            (
                "Beneficial Under Rule",
                _fmt_pct(score_row.get("defensive_correct_rate")),
                "Share where the benchmark underperformed cash or crossed the declared drawdown threshold; not crash-prediction accuracy.",
            ),
            (
                "Costly False Positive",
                _fmt_pct(score_row.get("defensive_false_alarm_rate")),
                "Share of audited defensive episodes that looked more like missed upside.",
            ),
        ]
    )
    if not score_row.empty:
        threshold = pd.to_numeric(
            pd.Series([score_row.get("defensive_threshold")]), errors="coerce"
        ).iloc[0]
        horizon = str(score_row.get("defensive_judgement_horizon", "1m"))
        benchmark = str(score_row.get("defensive_benchmark_ticker", "SPY"))
        current_defensive = pd.to_numeric(
            pd.Series([exposure_row.get("current_defensive_weight")]), errors="coerce"
        ).iloc[0]
        cohort = (
            f"{int(episode_count)} historical episodes"
            if pd.notna(episode_count)
            else "historical episodes"
        )
        threshold_text = f"{threshold:.0%}+" if pd.notna(threshold) else "high-defensive"
        current_context = ""
        if pd.notna(current_defensive) and pd.notna(threshold) and current_defensive < threshold:
            current_context = (
                f" Today's {current_defensive:.1%} defensive weight is below that audit trigger, "
                "so this is context for stronger historical defenses, not a resolved label for today."
            )
        render_callout(
            f"This read uses {cohort} for the focus strategy at {threshold_text} defense, "
            f"measured over {horizon} against {benchmark}.{current_context}"
        )


def _defensive_posture_rows(
    exposure: pd.DataFrame,
    scorecards: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Pair the focus exposure with its own SPY scorecard instead of row order."""
    exposure_row = exposure.iloc[0] if not exposure.empty else pd.Series(dtype=object)
    if scorecards.empty:
        return exposure_row, pd.Series(dtype=object)

    selected = scorecards.copy()
    focus_strategy = exposure_row.get("strategy")
    if focus_strategy is not None and "strategy" in selected:
        matching = selected[selected["strategy"].astype(str).eq(str(focus_strategy))]
        if not matching.empty:
            selected = matching
    if "defensive_benchmark_ticker" in selected:
        spy = selected[selected["defensive_benchmark_ticker"].astype(str).eq("SPY")]
        if not spy.empty:
            selected = spy
    if "defensive_judgement_horizon" in selected:
        one_month = selected[
            selected["defensive_judgement_horizon"].astype(str).eq("1m")
        ]
        if not one_month.empty:
            selected = one_month
    return exposure_row, selected.iloc[0]


def _defensive_label_display(label: str) -> str:
    labels = {
        "mixed_but_informative": "Mixed / informative",
        "weak_defensive_signal": "Weak defensive signal",
        "not_enough_history": "Insufficient history",
    }
    return labels.get(label, label.replace("_", " ").strip().title() or "n/a")


def _render_selected_prebreak_event_behavior(
    *,
    selected_crisis: str,
    signal_panel: pd.DataFrame,
    action_timing: pd.DataFrame,
) -> None:
    event_rows = _prebreak_event_rows(signal_panel, selected_crisis=selected_crisis)
    timing = _prebreak_timing_row(action_timing, selected_crisis=selected_crisis)
    if event_rows.empty and timing.empty:
        st.info("No pre-break hindsight rows are available for this selected crisis.")
        return
    if not event_rows.empty:
        event_rows["market_date"] = pd.to_datetime(event_rows["market_date"], errors="coerce")
        event_rows["days_to_break"] = pd.to_numeric(event_rows["days_to_break"], errors="coerce")
    worst_drawdown = (
        pd.to_numeric(event_rows.get("forward_min_max_drawdown_3m"), errors="coerce").min()
        if not event_rows.empty and "forward_min_max_drawdown_3m" in event_rows
        else float("nan")
    )
    render_card_grid(
        [
            (
                "Snapshots",
                int(timing.get("snapshots", len(event_rows))),
                "Historical weekly snapshots available for the selected event.",
            ),
            (
                "First Defensive",
                _days_before_label(timing.get("first_defensive_days_before_break")),
                "How early the bot first recommended any defensive action before the break date.",
            ),
            (
                "First Hard Defense",
                _days_before_label(timing.get("first_hard_defensive_days_before_break")),
                "How early the bot first showed hard defensive behavior before the break date.",
            ),
            (
                "Aligned When Severe",
                _fmt_pct(timing.get("aligned_when_severe_share")),
                "Among severe hindsight windows, share where trade-bot was already defensive.",
            ),
            (
                "Median Risk Budget",
                _fmt_pct(timing.get("median_risk_budget_multiplier")),
                "Median risk budget multiplier across this event's snapshots.",
            ),
            (
                "Worst Forward DD",
                _fmt_pct(worst_drawdown),
                "Worst 3-month forward max drawdown seen from the selected event's snapshots.",
            ),
        ]
    )
    if not event_rows.empty:
        render_chart(
            _prebreak_event_behavior_figure(event_rows),
            title="Trade-Bot Behavior Into Break",
            help_text=(
                "Risk budget and action severity are what the bot knew at the snapshot date. "
                "Forward returns and drawdowns are hindsight outcomes used to judge timing."
            ),
            key=f"dashboard_v2_prebreak_event_behavior_{selected_crisis}",
        )
        columns = [
            column
            for column in [
                "market_date",
                "days_to_break",
                "risk_status",
                "recommended_action",
                "risk_budget_multiplier",
                "defensive_action_flag",
                "hard_defensive_action_flag",
                "hindsight_action_aligned",
                "forward_spy_return_3m",
                "forward_qqq_return_3m",
                "forward_smh_return_3m",
                "forward_min_max_drawdown_3m",
                "break_severity_3m",
                "forward_break_label_3m",
            ]
            if column in event_rows
        ]
        with st.expander("Snapshot behavior and what-if outcomes", expanded=True):
            _render_metric_dataframe(_display_metrics(event_rows[columns]))


def _render_prebreak_margin_experiment(
    *,
    selected_crisis: str,
    staged_risk: pd.DataFrame,
    late_trigger: pd.DataFrame,
    hard_defense_attribution: pd.DataFrame,
    policy_variants: pd.DataFrame,
) -> None:
    selected_stages = _selected_prebreak_stages(staged_risk, selected_crisis=selected_crisis)
    selected_mesh = _selected_late_trigger_mesh(late_trigger, selected_crisis=selected_crisis)
    selected_attribution = _selected_hard_defense_attribution(
        hard_defense_attribution,
        selected_crisis=selected_crisis,
    )
    selected_variants = _selected_policy_variants(
        policy_variants,
        selected_crisis=selected_crisis,
    )
    if (
        selected_stages.empty
        and selected_mesh.empty
        and selected_attribution.empty
        and selected_variants.empty
    ):
        return
    render_section_header(
        "Can We Shrink The Margin?",
        help_text=(
            "Research-only audit of whether hard defense could be delayed while preserving "
            "late-stage drawdown coverage. Every stage and gate is measured relative to a "
            "break date known in hindsight, so this is a policy-hypothesis screen—not a live "
            "countdown or allocation rule."
        ),
    )
    render_callout(
        "How to read this: before confirmation, blue below green means Trade Bot carried less "
        "risk capacity than the hypothetical stage floor; after confirmation, blue above green "
        "means it carried more. Green is a research target, not an observed optimal allocation."
    )
    if not selected_mesh.empty:
        mesh = selected_mesh.copy()
        for column in [
            "trigger_days_before_break",
            "hard_defense_lead_cut_days",
            "missed_severe_label_share_if_gated",
            "mean_candidate_risk_budget_lift",
            "pre_trigger_false_alarm_share",
        ]:
            if column in mesh:
                mesh[column] = pd.to_numeric(mesh[column], errors="coerce")
        best = mesh.sort_values(
            [
                "missed_severe_label_share_if_gated",
                "mean_candidate_risk_budget_lift",
                "trigger_days_before_break",
            ],
            ascending=[True, False, True],
        ).iloc[0]
        render_card_grid(
            [
                (
                    "In-Sample Gate",
                    _days_before_label(best.get("trigger_days_before_break")),
                    "Best tested gate for this already-known historical event: lowest missed "
                    "severe-label share, then highest capacity lift. Do not use it as a live "
                    "countdown.",
                    _late_trigger_tone(best.get("mesh_read")),
                ),
                (
                    "Lead Removed",
                    _days_before_label(best.get("hard_defense_lead_cut_days")),
                    "How many calendar days of the event's observed early hard-defense lead time "
                    "the hypothetical gate would remove.",
                ),
                (
                    "Severe Missed",
                    _fmt_pct(best.get("missed_severe_label_share_if_gated")),
                    "Share of this event's severe three-month drawdown labels occurring before "
                    "the gate. Zero is in-sample coverage, not a guarantee for a new event.",
                    (
                        "critical"
                        if pd.to_numeric(
                            pd.Series([best.get("missed_severe_label_share_if_gated")]),
                            errors="coerce",
                        ).iloc[0]
                        > 0.25
                        else "neutral"
                    ),
                ),
                (
                    "Capacity Lift",
                    _fmt_pp(best.get("mean_candidate_risk_budget_lift")),
                    "Average additional risk-budget capacity allowed before the gate, measured "
                    "in percentage points. This is not an estimated return improvement or a "
                    "portfolio weight.",
                    (
                        "success"
                        if pd.to_numeric(
                            pd.Series([best.get("mean_candidate_risk_budget_lift")]),
                            errors="coerce",
                        ).iloc[0]
                        >= 0.10
                        else "neutral"
                    ),
                ),
            ]
        )
        st.caption(_late_trigger_interpretation(best))
    if not selected_stages.empty:
        render_chart(
            _prebreak_staged_risk_figure(selected_stages),
            title="Actual Risk Budget Vs Staged Target",
            help_text=(
                "Blue is the bot's median risk-budget capacity, not its actual risk-asset weight. "
                "Green is a hypothetical hindsight stage floor: long-lead 100%, early watch 75%, "
                "warning 60%, confirmed pre-break 35%, and break/unwind 20%. Red is the share of "
                "snapshots classified hard-defensive. Orange marks early hard-defense snapshots "
                "with positive subsequent upside and no severe three-month drawdown label."
            ),
            key=f"dashboard_v2_prebreak_staged_margin_{selected_crisis}",
        )
        st.caption(
            "Stage windows use calendar days from the known break: Long Lead >120d; Early Watch "
            "60–120d; Warning 46–59d; Confirmed Pre-Break 15–45d; Break / Unwind 0–14d; "
            "Post-Break after the event. Use the shape to diagnose timing, then validate any "
            "candidate rule across other crises and walk-forward data."
        )
        stage_columns = [
            column
            for column in [
                "prebreak_stage",
                "snapshots",
                "min_days_to_break",
                "max_days_to_break",
                "target_staged_risk_budget_multiplier",
                "median_risk_budget_multiplier",
                "mean_candidate_risk_budget_lift",
                "hard_defensive_snapshot_share",
                "early_hard_false_alarm_share",
                "severe_label_share",
            ]
            if column in selected_stages
        ]
        with st.expander("Staged-risk behavior detail", expanded=False):
            display = selected_stages[stage_columns].copy()
            if "prebreak_stage" in display:
                display["prebreak_stage"] = display["prebreak_stage"].map(_stage_label)
            _render_metric_dataframe(_display_metrics(display))
    if not selected_attribution.empty:
        attribution_columns = [
            column
            for column in [
                "prebreak_stage",
                "hard_defense_source",
                "hard_defensive_snapshots",
                "source_share_of_stage_hard_defense",
                "median_risk_budget_multiplier",
                "median_current_risk_asset_weight",
                "median_scenario_event_macro_multiplier",
                "median_portfolio_risk_multiplier",
                "early_hard_false_alarm_share",
            ]
            if column in selected_attribution
        ]
        with st.expander("What caused early hard defense?", expanded=True):
            display = selected_attribution[attribution_columns].copy()
            if "prebreak_stage" in display:
                display["prebreak_stage"] = display["prebreak_stage"].map(_stage_label)
            if "hard_defense_source" in display:
                display["hard_defense_source"] = display["hard_defense_source"].map(_stage_label)
            _render_metric_dataframe(_display_metrics(display))
    if not selected_mesh.empty:
        mesh_columns = [
            column
            for column in [
                "trigger_days_before_break",
                "actual_first_hard_defensive_days_before_break",
                "hard_defense_lead_cut_days",
                "pre_trigger_hard_defensive_share",
                "pre_trigger_false_alarm_share",
                "severe_label_coverage_inside_trigger",
                "missed_severe_label_share_if_gated",
                "mean_candidate_risk_budget_lift",
                "median_forward_return_when_lifted",
                "median_forward_drawdown_when_lifted",
                "mesh_read",
            ]
            if column in selected_mesh
        ]
        with st.expander("Late-trigger mesh detail", expanded=True):
            _render_metric_dataframe(_display_metrics(selected_mesh[mesh_columns]))
    if not selected_variants.empty:
        variant_columns = [
            column
            for column in [
                "policy_name",
                "median_actual_risk_budget_multiplier",
                "median_policy_risk_budget_multiplier",
                "mean_early_risk_budget_lift",
                "mean_false_alarm_risk_budget_lift",
                "mean_severe_label_risk_budget_lift",
                "mean_incremental_return_proxy_3m",
                "mean_incremental_drawdown_proxy_3m",
                "candidate_score",
                "policy_read",
            ]
            if column in selected_variants
        ]
        with st.expander("Research policy variants", expanded=True):
            _render_metric_dataframe(_display_metrics(selected_variants[variant_columns]))


def _render_prebreak_signal_attribution(
    signal_rank: pd.DataFrame,
    *,
    current_readout: pd.DataFrame,
) -> None:
    render_section_header("What Worked To Detect Break Risk")
    data = signal_rank.copy()
    for column in [
        "predictive_score",
        "absolute_spearman",
        "spearman_to_break_severity",
        "event_auc_edge",
        "event_auc",
        "high_minus_low_break_severity",
        "latest_value",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if not current_readout.empty and "signal" in current_readout:
        current = current_readout[["signal", "current_risk_read", "historical_percentile"]].copy()
        data = data.merge(current, on="signal", how="left")
    render_chart(
        _prebreak_signal_rank_figure(data.head(12)),
        title="Most Predictive Pre-Break Signals",
        help_text=(
            "Ranks snapshot signals by correlation with future break severity, event-classification "
            "edge, and high-minus-low outcome spread."
        ),
        key="dashboard_v2_prebreak_signal_rank_chart",
    )
    durable = data.head(8).copy()
    if "signal" in durable:
        durable["signal"] = durable["signal"].map(_signal_label)
    columns = [
        column
        for column in [
            "signal",
            "observations",
            "predictive_score",
            "risk_direction",
            "current_risk_read",
            "historical_percentile",
            "spearman_to_break_severity",
            "event_auc",
            "high_minus_low_break_severity",
            "latest_value",
        ]
        if column in durable
    ]
    with st.expander("Signal attribution table", expanded=True):
        _render_metric_dataframe(_display_metrics(durable[columns]))


def _prebreak_event_rows(
    signal_panel: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.DataFrame:
    if signal_panel.empty or "event_name" not in signal_panel:
        return pd.DataFrame()
    frame = signal_panel[
        signal_panel["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ].copy()
    if frame.empty:
        return frame
    sort_columns = [
        column for column in ["market_date", "days_to_break", "run_id"] if column in frame
    ]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _prebreak_timing_row(
    action_timing: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.Series:
    if action_timing.empty or "event_name" not in action_timing:
        return pd.Series(dtype=object)
    selected = action_timing[
        action_timing["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ]
    return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)


def _selected_prebreak_stages(
    staged_risk: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.DataFrame:
    if staged_risk.empty or "event_name" not in staged_risk:
        return pd.DataFrame()
    selected = staged_risk[
        staged_risk["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ].copy()
    if selected.empty:
        return selected
    if "prebreak_stage_order" in selected:
        selected["prebreak_stage_order"] = pd.to_numeric(
            selected["prebreak_stage_order"],
            errors="coerce",
        )
        selected = selected.sort_values("prebreak_stage_order")
    return selected.reset_index(drop=True)


def _selected_late_trigger_mesh(
    late_trigger: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.DataFrame:
    if late_trigger.empty or "event_name" not in late_trigger:
        return pd.DataFrame()
    selected = late_trigger[
        late_trigger["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ].copy()
    if selected.empty:
        return selected
    if "trigger_days_before_break" in selected:
        selected["trigger_days_before_break"] = pd.to_numeric(
            selected["trigger_days_before_break"],
            errors="coerce",
        )
        selected = selected.sort_values("trigger_days_before_break")
    return selected.reset_index(drop=True)


def _selected_hard_defense_attribution(
    hard_defense_attribution: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.DataFrame:
    if hard_defense_attribution.empty or "event_name" not in hard_defense_attribution:
        return pd.DataFrame()
    selected = hard_defense_attribution[
        hard_defense_attribution["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ].copy()
    if selected.empty:
        return selected
    if "hard_defense_source" in selected:
        selected = selected[
            selected["hard_defense_source"].fillna("").astype(str).ne("not_hard_defensive")
        ].copy()
    for column in [
        "prebreak_stage_order",
        "source_share_of_stage_hard_defense",
        "hard_defensive_snapshots",
    ]:
        if column in selected:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    sort_columns = [
        column
        for column in [
            "prebreak_stage_order",
            "source_share_of_stage_hard_defense",
            "hard_defensive_snapshots",
        ]
        if column in selected
    ]
    if sort_columns:
        selected = selected.sort_values(
            sort_columns,
            ascending=[True, False, False][: len(sort_columns)],
        )
    return selected.reset_index(drop=True)


def _selected_policy_variants(
    policy_variants: pd.DataFrame,
    *,
    selected_crisis: str,
) -> pd.DataFrame:
    if policy_variants.empty or "event_name" not in policy_variants:
        return pd.DataFrame()
    selected = policy_variants[
        policy_variants["event_name"].fillna("").astype(str).eq(str(selected_crisis))
    ].copy()
    if selected.empty:
        return selected
    if "candidate_score" in selected:
        selected["candidate_score"] = pd.to_numeric(selected["candidate_score"], errors="coerce")
        selected = selected.sort_values("candidate_score", ascending=False)
    return selected.reset_index(drop=True)


def _prebreak_staged_risk_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    if data.empty:
        return go.Figure()
    for column in [
        "target_staged_risk_budget_multiplier",
        "median_risk_budget_multiplier",
        "hard_defensive_snapshot_share",
        "early_hard_false_alarm_share",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    stage_labels = [_stage_label(value) for value in data["prebreak_stage"].astype(str)]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=stage_labels,
            y=data["median_risk_budget_multiplier"],
            name="Median actual budget",
            marker_color="#2563eb",
            hovertemplate="<b>%{x}</b><br>Actual budget: %{y:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=stage_labels,
            y=data["target_staged_risk_budget_multiplier"],
            name="Staged target",
            marker_color="#16a34a",
            opacity=0.72,
            hovertemplate="<b>%{x}</b><br>Target budget: %{y:.1%}<extra></extra>",
        )
    )
    if "hard_defensive_snapshot_share" in data:
        figure.add_trace(
            go.Scatter(
                x=stage_labels,
                y=data["hard_defensive_snapshot_share"],
                mode="lines+markers",
                name="Hard-defense share",
                line={"color": "#ef4444", "width": 3},
                marker={"size": 8},
                hovertemplate="<b>%{x}</b><br>Hard-defense share: %{y:.1%}<extra></extra>",
            )
        )
    if "early_hard_false_alarm_share" in data:
        figure.add_trace(
            go.Scatter(
                x=stage_labels,
                y=data["early_hard_false_alarm_share"],
                mode="lines+markers",
                name="Early hard false alarm",
                line={"color": "#f59e0b", "width": 2, "dash": "dash"},
                marker={"size": 7},
                hovertemplate="<b>%{x}</b><br>Early false alarm: %{y:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        yaxis_title="Share / risk budget",
        yaxis_tickformat=".0%",
        xaxis_title="Pre-break stage",
        legend_title_text="Series",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=420,
    )
    return figure


def _prebreak_event_behavior_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy().sort_values("market_date")
    for column in [
        "risk_budget_multiplier",
        "action_severity_score",
        "forward_spy_return_3m",
        "forward_qqq_return_3m",
        "forward_smh_return_3m",
        "forward_min_max_drawdown_3m",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.48, 0.52],
        subplot_titles=(
            "Known-at-the-time behavior",
            "Hindsight 3-month what-if outcomes",
        ),
    )
    x_values = data["market_date"]
    if "risk_budget_multiplier" in data:
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=data["risk_budget_multiplier"],
                mode="lines+markers",
                name="Risk budget multiplier",
                line={"color": "#2563eb", "width": 3},
                marker={"size": 7},
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Risk budget: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "action_severity_score" in data:
        figure.add_trace(
            go.Bar(
                x=x_values,
                y=data["action_severity_score"],
                name="Action severity",
                marker_color="#f59e0b",
                opacity=0.48,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Action severity: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    outcome_series = [
        ("SPY 3m return", "forward_spy_return_3m", "#16a34a"),
        ("QQQ 3m return", "forward_qqq_return_3m", "#2563eb"),
        ("SMH 3m return", "forward_smh_return_3m", "#8b5cf6"),
        ("Worst 3m drawdown", "forward_min_max_drawdown_3m", "#ef4444"),
    ]
    for name, column, color in outcome_series:
        if column not in data or data[column].notna().sum() == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=data[column],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 6},
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{name}: %{{y:.1%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    break_dates = data.get("event_break_date", pd.Series(dtype=object)).dropna().astype(str)
    if not break_dates.empty:
        break_date = pd.to_datetime(break_dates.iloc[0], errors="coerce")
        if pd.notna(break_date):
            figure.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=break_date.isoformat(),
                x1=break_date.isoformat(),
                y0=0,
                y1=1,
                line={"color": "#ef4444", "dash": "dash", "width": 2},
            )
            figure.add_annotation(
                xref="x",
                yref="paper",
                x=break_date.isoformat(),
                y=1.02,
                text="Break",
                showarrow=False,
                font={"size": 11, "color": "#ef4444"},
                xanchor="left",
            )
    figure.add_hline(y=0, line_color="#7f8ea3", line_width=1, row=2, col=1)
    figure.update_layout(
        yaxis_title="Risk budget / severity",
        yaxis_tickformat=".0%",
        yaxis2_title="Forward outcome",
        yaxis2_tickformat=".0%",
        xaxis2_title="Snapshot market date",
        legend_title_text="Series",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        height=560,
    )
    return figure


def _prebreak_signal_rank_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    if data.empty:
        return go.Figure()
    data["predictive_score"] = pd.to_numeric(
        data["predictive_score"],
        errors="coerce",
    ).fillna(0.0)
    data = data.sort_values("predictive_score", ascending=True)
    colors = [
        _risk_read_color(value)
        for value in data.get("current_risk_read", pd.Series("", index=data.index)).astype(str)
    ]
    y_labels = [_signal_label(signal) for signal in data["signal"].astype(str)]
    customdata = data.reindex(
        columns=[
            "risk_direction",
            "current_risk_read",
            "historical_percentile",
            "spearman_to_break_severity",
            "event_auc",
            "high_minus_low_break_severity",
        ],
        fill_value="n/a",
    ).to_numpy()
    figure = go.Figure(
        go.Bar(
            x=data["predictive_score"],
            y=y_labels,
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>Predictive score: %{x:.2f}<br>"
                "Risk direction: %{customdata[0]}<br>"
                "Current read: %{customdata[1]}<br>"
                "Current percentile: %{customdata[2]:.1%}<br>"
                "Spearman: %{customdata[3]:.2f}<br>"
                "Event AUC: %{customdata[4]:.2f}<br>"
                "High-low severity spread: %{customdata[5]:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis_title="Predictive score",
        yaxis_title="Signal",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=420,
    )
    return figure


def _signal_label(value: object) -> str:
    text = str(value or "n/a")
    cleaned = text.replace("cycle_component_", "cycle: ")
    cleaned = cleaned.replace("health_", "market: ")
    cleaned = cleaned.replace("instability_", "instability: ")
    return cleaned.replace("_", " ").title()


def _stage_label(value: object) -> str:
    labels = {
        "long_lead_context": "Long Lead",
        "early_watch": "Early Watch",
        "warning": "Warning",
        "confirmed_prebreak": "Confirmed Pre-Break",
        "break_unwind": "Break / Unwind",
        "postbreak_followthrough": "Post-Break",
    }
    text = str(value or "")
    return labels.get(text, text.replace("_", " ").title() or "n/a")


def _late_trigger_tone(value: object) -> str:
    text = str(value).strip().lower()
    if text == "promising":
        return "success"
    if text == "too_late":
        return "critical"
    if text == "limited_lift":
        return "warning"
    return "neutral"


def _risk_read_tone(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"high_risk", "elevated"}:
        return "critical"
    if text == "mixed":
        return "warning"
    if text == "contained":
        return "success"
    return "neutral"


def _risk_read_color(value: object) -> str:
    tone = _risk_read_tone(value)
    return {
        "critical": "#ef4444",
        "warning": "#f59e0b",
        "success": "#16a34a",
    }.get(tone, "#7f8ea3")


def _days_before_label(value: object) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "n/a"
    if float(parsed) < 0:
        return f"{abs(int(parsed))}d after"
    return f"{int(parsed)}d before"


def _late_trigger_interpretation(row: pd.Series) -> str:
    gate = _days_before_label(row.get("trigger_days_before_break"))
    lead_removed = _days_before_label(row.get("hard_defense_lead_cut_days"))
    missed = _fmt_pct(row.get("missed_severe_label_share_if_gated"))
    lift = _fmt_pp(row.get("mean_candidate_risk_budget_lift"))
    drawdown = _fmt_pct(row.get("median_forward_drawdown_when_lifted"))
    return (
        f"Selected-event hindsight read: a gate {gate} would remove {lead_removed} of early "
        f"hard-defense lead, restore {lift} of average risk-budget capacity, and place {missed} "
        f"of this event's severe labels before the gate. The median three-month forward maximum "
        f"drawdown where capacity would be restored was {drawdown}, so results near the severe "
        "threshold remain economically meaningful."
    )


def _phase_winner_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["frontier_score"] = pd.to_numeric(data["frontier_score"], errors="coerce").fillna(0.0)
    if "origins" not in data:
        data["origins"] = 0
    data["origins"] = pd.to_numeric(data["origins"], errors="coerce").fillna(0).astype(int)
    for column in [
        "median_forward_return",
        "median_forward_drawdown",
        "phase_distinctiveness",
        "ubiquity_penalty",
        "cycle_leadership_fragility",
        "theme_fragility_penalty",
    ]:
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
                    "phase_distinctiveness",
                    "ubiquity_penalty",
                    "cycle_leadership_fragility",
                    "theme_fragility_penalty",
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
                "Median drawdown: %{customdata[6]:.1%}<br>"
                "Phase specificity: %{customdata[7]:.1%}<br>"
                "Ubiquity penalty: %{customdata[8]:.1%}<br>"
                "Leadership fragility: %{customdata[9]:.1%}<br>"
                "Theme penalty: %{customdata[10]:.1%}<extra></extra>"
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
    title_suffix = (
        f" | QQQ hit rate {_fmt_pct(hit_rate)}"
        if hit_rate is not None and not pd.isna(hit_rate)
        else ""
    )
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


def _available_crisis_playback_tickers(frame: pd.DataFrame) -> list[str]:
    tickers: list[str] = []
    for column in frame.columns:
        if not str(column).endswith("_playback_index"):
            continue
        ticker = str(column).removesuffix("_playback_index").upper()
        if f"{ticker.lower()}_playback_drawdown" in frame.columns:
            tickers.append(ticker)
    preferred = ["QQQ", "SPY", "VTI", "SMH", "IWM", "LQD", "TLT", "BIL"]
    ordered = [ticker for ticker in preferred if ticker in tickers]
    ordered.extend(sorted(ticker for ticker in tickers if ticker not in preferred))
    return ordered


def _crisis_playback_market_frame(frame: pd.DataFrame, *, tickers: list[str]) -> pd.DataFrame:
    columns = ["origin_date", "stage", "stage_order"]
    for ticker in tickers:
        key = ticker.lower()
        columns.extend(
            [
                f"{key}_playback_index",
                f"{key}_playback_drawdown",
                f"{key}_playback_return_since_window_start",
                f"{key}_playback_close",
            ]
        )
    available = [column for column in columns if column in frame.columns]
    if "origin_date" not in available:
        return pd.DataFrame()
    data = frame[available].drop_duplicates().copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"], errors="coerce")
    if "stage_order" in data:
        data["stage_order"] = pd.to_numeric(data["stage_order"], errors="coerce")
    return data.sort_values(["origin_date", "stage_order"]).reset_index(drop=True)


def _crisis_playback_market_figure(frame: pd.DataFrame, *, tickers: list[str]) -> go.Figure:
    data = _crisis_playback_market_frame(frame, tickers=tickers)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Ticker path indexed to crisis-window start",
            "Ticker drawdown from prior crisis-window high",
        ),
    )
    colors = {
        "QQQ": "#2563eb",
        "SPY": "#16a34a",
        "VTI": "#06b6d4",
        "SMH": "#8b5cf6",
        "IWM": "#f59e0b",
        "LQD": "#64748b",
        "TLT": "#0f766e",
        "BIL": "#94a3b8",
    }
    for ticker in tickers:
        key = ticker.lower()
        index_column = f"{key}_playback_index"
        drawdown_column = f"{key}_playback_drawdown"
        close_column = f"{key}_playback_close"
        if index_column not in data or drawdown_column not in data:
            continue
        if close_column not in data:
            data[close_column] = float("nan")
        if "stage" not in data:
            data["stage"] = "n/a"
        customdata = data[[close_column, "stage"]].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=data["origin_date"],
                y=pd.to_numeric(data[index_column], errors="coerce"),
                mode="lines+markers",
                name=f"{ticker} indexed price",
                line={"color": colors.get(ticker, "#7f8ea3"), "width": 2},
                marker={"size": 5},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"{ticker} index: "
                    "%{y:.2f}<br>"
                    "Close: %{customdata[0]:.2f}<br>"
                    "Stage: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data["origin_date"],
                y=pd.to_numeric(data[drawdown_column], errors="coerce"),
                mode="lines+markers",
                name=f"{ticker} drawdown",
                line={"color": colors.get(ticker, "#7f8ea3"), "width": 2, "dash": "dot"},
                marker={"size": 5},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"{ticker} drawdown: "
                    "%{y:.1%}<br>"
                    "Close: %{customdata[0]:.2f}<br>"
                    "Stage: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=1.0, line_color="#7f8ea3", line_width=1, row=1, col=1)
    figure.add_hline(y=0.0, line_color="#7f8ea3", line_width=1, row=2, col=1)
    figure.update_layout(
        yaxis_title="Index",
        yaxis_tickformat=".2f",
        yaxis2_title="Drawdown",
        yaxis2_tickformat=".0%",
        xaxis2_title="Historical origin date",
        legend_title_text="Ticker",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        height=520,
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
            (
                "Nowcast confidence at each historical origin"
                if is_nowcast
                else f"What happened over the selected {horizon} horizon"
            ),
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
        yaxis2_title="Dominant phase confidence" if is_nowcast else f"{horizon} return / drawdown",
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
            (
                3
                if selected_horizon
                and selected_phase
                and str(horizon) == str(selected_horizon)
                and phase == selected_phase
                else 0
            )
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
                customdata=[[phase.replace("_", " ").title(), horizon] for horizon in horizons],
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


def _fmt_pp(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric * 100:.2f} pp"


def _fmt_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.2f}"
