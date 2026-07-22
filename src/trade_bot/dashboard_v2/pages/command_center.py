from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.overview import render_operating_overview
from trade_bot.dashboard.trends import (
    compact_metric_line_figure,
    latest_per_market_date,
    load_snapshot_trend_frames,
)
from trade_bot.dashboard_v2.components.cards import (
    help_icon,
    render_callout,
    render_card_grid,
    render_chart,
    render_section_header,
)
from trade_bot.dashboard_v2.components.tones import (
    decision_sanity_tone,
    normalize_tone,
    portfolio_risk_tone,
    posture_calibration_tone,
    probability_pressure_tone,
    risk_budget_tone,
    risk_score_tone,
    risk_status_tone,
    target_defensive_tone,
)
from trade_bot.dashboard_v2.help import metric_help
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.runtime import (
    HISTORICAL_SNAPSHOT_NOTICE,
    DashboardRuntime,
)


def render_today_page(runtime: DashboardRuntime) -> None:
    run = runtime.baseline_run
    current_state = run.current_state
    if runtime.is_historical_snapshot_mode:
        st.warning(HISTORICAL_SNAPSHOT_NOTICE)
        render_card_grid(
            [
                ("Historical Market Date", current_state.market_date),
                (
                    "Snapshot Run",
                    getattr(runtime.snapshot_manifest, "run_id", "live fallback"),
                ),
                ("Mode", "Display only"),
            ]
        )
        st.caption(
            "Use Macro, Performance, Monitoring, Research, or Simulation for point-in-time "
            "inspection. Switch to Latest snapshot or Live pipeline before reviewing current "
            "actions."
        )
        return
    trade_decision = runtime.operating_trade_decision
    operating_error = getattr(runtime, "operating_strategy_error", None)
    if (
        operating_error
        or trade_decision is None
        or runtime.open_ticket_count is None
        or runtime.book_alignment is None
        or runtime.action_headline is None
    ):
        st.error(
            "Operating decision unavailable for promoted book "
            f"'{runtime.promoted_book.book_name}': {operating_error or 'unknown resolution error'}"
        )
        st.caption(
            "Position plans, action headlines, open-ticket counts, and operating risk are "
            "suppressed until the book names a strategy present in this run with an explicit "
            "defensive policy. Use Forward Test to inspect or edit the book."
        )
        return
    summary = _first_row(trade_decision.summary)

    render_card_grid(
        [
            ("Market Date", current_state.market_date),
            (
                "Fragility",
                str(current_state.risk_status).upper(),
                "Broad price-derived fragility diagnostic; it does not size the portfolio unless the timing layer has calibrated authority.",
                risk_status_tone(current_state.risk_status),
            ),
            (
                "Risk Score",
                f"{float(current_state.risk_score):.2f}",
                None,
                risk_score_tone(current_state.risk_score),
            ),
            (
                "Timing Gate",
                str(getattr(current_state, "risk_timing_state", "unassessed")).replace(
                    "_", " "
                ).title(),
                f"Effective sizing authority: {_fmt_pct(summary.get('risk_timing_sizing_authority'))}. Meaningful de-risking requires independent credit, volatility, breadth, and trend confirmation.",
                "warning"
                if str(getattr(current_state, "risk_timing_state", ""))
                in {"warning", "confirmed_break", "severe_break"}
                else "neutral",
            ),
            (
                "Open Tickets",
                runtime.open_ticket_count,
                None,
                "warning" if runtime.open_ticket_count else "success",
            ),
            (
                "Risk Budget",
                _fmt_pct(summary.get("risk_budget_multiplier")),
                None,
                risk_budget_tone(summary.get("risk_budget_multiplier")),
            ),
            (
                "Raw 1M Risk-Off Score",
                _fmt_pct(summary.get("one_month_risk_off_probability")),
                f"Uncalibrated model output; allocation authority {_fmt_pct(summary.get('scenario_sizing_authority'))}. Do not interpret as a literal forecast probability.",
                probability_pressure_tone(
                    summary.get("one_month_risk_off_probability"),
                    warning_at=0.15,
                    critical_at=0.35,
                ),
            ),
        ]
    )
    render_callout(
        str(getattr(runtime.action_headline, "headline", "")) or current_state.risk_summary
    )

    view = st.pills(
        "Today view",
        ["Decision", "Operating brief", "Trends", "Raw evidence"],
        default="Decision",
        selection_mode="single",
        key="dashboard_v2_today_view",
    )
    selected_view = view or "Decision"

    if selected_view == "Decision":
        render_section_header("Trade Decision")
        plan = (
            runtime.execution_book_alignment.position_plan
            if runtime.execution_book_alignment is not None
            and not runtime.execution_book_alignment.position_plan.empty
            else trade_decision.position_plan
        )
        _render_decision_context(runtime, summary, plan)
        _render_metric_dataframe(_display_metrics(plan))
        with st.expander("Scenario links", expanded=False):
            _render_metric_dataframe(_display_metrics(trade_decision.scenario_links))
        with st.expander("Raw decision explanation", expanded=False):
            st.write(str(summary.get("human_explanation", current_state.risk_summary)))
    elif selected_view == "Operating brief":
        render_operating_overview(
            baseline_run=runtime.baseline_run,
            headline=runtime.action_headline,
            open_ticket_count=runtime.open_ticket_count,
            experiment_scorecards=pd.DataFrame(),
            default_book_alignment=runtime.book_alignment,
            previous_run=None,
            execution_book_alignment=runtime.execution_book_alignment,
        )
    elif selected_view == "Trends":
        render_callout(
            "Trend plots read saved snapshots and can take longer on the first load.",
            heavy=True,
        )
        with timed("today.trends"):
            (
                metric_history,
                _component_history,
                _scenario_driver_history,
                _driver_rotation_history,
            ) = load_snapshot_trend_frames(
                str(runtime.paths.run_store_path),
                str(runtime.paths.artifact_dir),
                str(runtime.paths.job_log_dir),
            )
        metric_history = latest_per_market_date(metric_history)
        figure = compact_metric_line_figure(
            metric_history,
            columns=[
                "risk_score",
                "one_month_risk_off_probability",
                "risk_budget_multiplier",
            ],
            labels={
                "risk_score": "Risk score",
                "one_month_risk_off_probability": "1M risk-off probability",
                "risk_budget_multiplier": "Risk budget multiplier",
            },
            title="Risk Score, Risk-Off Odds, and Trade Budget",
            yaxis_title="Score / multiplier",
        )
        if figure is None:
            st.info("No saved trend history is available yet.")
        else:
            render_chart(figure)
    else:
        render_section_header("Raw Evidence")
        st.dataframe(trade_decision.evidence, width="stretch")
        render_section_header("Trading Alerts")
        st.dataframe(current_state.strategy_alerts, width="stretch")
        render_section_header("Scenario Outlook")
        _render_metric_dataframe(_display_metrics(current_state.scenario_outlook.copy()))


def _first_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(str(value)):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _render_decision_context(
    runtime: DashboardRuntime,
    summary: dict[str, object],
    plan: pd.DataFrame,
) -> None:
    run = runtime.baseline_run
    current_state = run.current_state
    allocation_cards = [
        (
            "Fragility Diagnostic",
            f"{str(current_state.risk_status).upper()} ({_fmt_float(current_state.risk_score)})",
            current_state.risk_summary,
            risk_status_tone(current_state.risk_status),
        ),
        (
            "Target Posture",
            _target_posture_range(summary.get("target_defensive_weight")),
            "A planning range is shown here; exact instrument weights remain in the auditable plan below.",
            target_defensive_tone(summary.get("target_defensive_weight")),
        ),
        (
            "Portfolio Risk Engine",
            str(summary.get("portfolio_risk_level", "n/a")).replace("_", " "),
            _portfolio_risk_sentence(summary),
            portfolio_risk_tone(summary.get("portfolio_risk_level")),
        ),
    ]
    research_cards = [
        (
            "Risk Timing",
            str(summary.get("risk_timing_state", "unassessed")).replace("_", " ").title(),
            f"Raw multiplier {_fmt_pct(summary.get('raw_risk_timing_multiplier'))}; effective multiplier {_fmt_pct(summary.get('risk_timing_multiplier'))} at {_fmt_pct(summary.get('risk_timing_sizing_authority'))} authority. Calibration: {summary.get('risk_timing_calibration_status', 'unknown')}.",
            "warning"
            if str(summary.get("risk_timing_state", ""))
            in {"warning", "confirmed_break", "severe_break"}
            else "neutral",
        ),
        (
            "Raw 1M Scenario Scores",
            _scenario_probability_summary(run.trade_decision.scenario_links),
            f"Raw model estimates, not calibrated forecast probabilities. Research-only at {_fmt_pct(summary.get('scenario_sizing_authority'))} sizing authority; calibration status: {summary.get('scenario_calibration_status', 'unknown')}.",
            _scenario_mix_tone(summary),
        ),
        (
            "Current Event Pressure",
            _event_pressure_summary(run.news_monitor.triage),
            f"Raw pressure {_fmt_pct(summary.get('raw_event_pressure'))}; effective sizing pressure {_fmt_pct(summary.get('effective_event_pressure'))}. News is informational unless policy explicitly grants authority.",
            probability_pressure_tone(
                summary.get("raw_event_pressure"), warning_at=0.01, critical_at=0.12
            ),
        ),
        (
            "Macro Inclusion",
            _macro_inclusion_summary(run.signal_inclusion.summary),
            f"Effective macro pressure {_fmt_pct(summary.get('macro_pressure'))} at {_fmt_pct(summary.get('macro_sizing_authority'))} authority. Calibration: {summary.get('macro_calibration_status', 'unknown')}; vintage: {summary.get('macro_data_vintage_status', 'unknown')}.",
            probability_pressure_tone(
                summary.get("macro_pressure"), warning_at=0.04, critical_at=0.12
            ),
        ),
    ]
    governance_cards = [
        (
            "Decision Sanity",
            str(summary.get("decision_sanity_signal", "n/a")).replace("_", " "),
            str(summary.get("decision_sanity_note", "n/a")),
            decision_sanity_tone(summary.get("decision_sanity_signal")),
        ),
        (
            "Posture Calibration",
            str(summary.get("posture_calibration_signal", "n/a")).replace("_", " "),
            str(summary.get("posture_calibration_note", "n/a")),
            posture_calibration_tone(summary.get("posture_calibration_signal")),
        ),
    ]
    for title, cards in (
        ("Allocation-authoritative system", allocation_cards),
        ("Research context — does not imply allocation authority", research_cards),
        ("Governance and calibration", governance_cards),
    ):
        st.markdown(f"#### {title}")
        st.markdown(
            '<div class="v2-decision-grid">'
            + "".join(
                _decision_card_html(label, answer, detail, tone)
                for label, answer, detail, tone in cards
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    _render_attribution_and_counterfactuals(run.trade_decision)
    _render_cost_of_defense()
    _render_recommendation_changes(summary)
    if not plan.empty:
        st.caption("Material target weights and current-book drift are shown below.")


def _target_posture_range(value: object) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "n/a"
    lower = max(0.0, 0.05 * int(parsed / 0.05))
    upper = min(1.0, lower + 0.05)
    return f"Defensive planning range {lower:.0%}–{upper:.0%}"


def _render_attribution_and_counterfactuals(trade_decision: object) -> None:
    attribution = getattr(trade_decision, "attribution", pd.DataFrame())
    counterfactuals = getattr(trade_decision, "counterfactuals", pd.DataFrame())
    st.markdown("#### Why this target changed")
    if attribution.empty:
        st.info("No causal attribution is available for this decision.")
    else:
        display = attribution[
            [
                "layer",
                "role",
                "authority",
                "marginal_defensive_add_pp",
                "defensive_weight",
            ]
        ].copy()
        display["layer"] = display["layer"].str.replace("_", " ").str.title()
        display["marginal_defensive_add_pp"] = display["marginal_defensive_add_pp"].round(1)
        display["authority"] = display["authority"].map(lambda value: f"{value:.0%}")
        display["defensive_weight"] = display["defensive_weight"].map(
            lambda value: f"{value:.0%}"
        )
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption(
            "Marginal effects are sequential and sum to the final defensive weight. A zero "
            "effect is not presented as a reason for the recommendation."
        )
    with st.expander("Permanent news counterfactuals", expanded=False):
        if counterfactuals.empty:
            st.info("No counterfactual table is available.")
        else:
            st.dataframe(counterfactuals, width="stretch", hide_index=True)


def _render_cost_of_defense() -> None:
    path = "reports/defensive_layer_calibration/calibration_summary.csv"
    try:
        calibration = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError):
        return
    selected = calibration[
        (calibration["cohort"] == "all_three")
        & calibration["horizon"].isin(["1m", "3m"])
    ]
    if selected.empty:
        return
    st.markdown("#### Historical cost of defense")
    cards = []
    for _, row in selected.sort_values("horizon").iterrows():
        cards.append(
            (
                f"{str(row['horizon']).upper()} layered-defense history",
                f"Median return regret {float(row['median_regret_vs_base']):.1%}; drawdown improvement {float(row['drawdown_improvement_p50']):.1%}",
                f"{int(row['episode_starts'])} non-overlapping episodes; defense beneficial under the stated rule {float(row['correct_defense_rate']):.0%}, costly false positive {float(row['false_alarm_rate']):.0%}.",
                "warning",
            )
        )
    st.markdown(
        '<div class="v2-decision-grid">'
        + "".join(_decision_card_html(*card) for card in cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Retrospective, small-sample evidence. 'Beneficial' means benchmark underperformed cash or crossed the declared drawdown threshold; it is not a crash-prediction accuracy score. Overlapping weekly observations are not counted as independent episodes.")


def _render_recommendation_changes(summary: dict[str, object]) -> None:
    st.markdown("#### What would change the recommendation?")
    st.write(
        "More risk requires the base market strategy to re-risk and the independent hard "
        "portfolio limits to remain clear. Scenario sizing stays off until walk-forward "
        "calibration earns positive skill with credible uncertainty; news alone cannot change "
        "the target. Less risk requires a worse quantitative risk state or an actual beta, "
        "expected-shortfall, or catastrophic-stress breach."
    )
    st.caption(
        f"Active utility profile: {summary.get('utility_profile', 'unknown')}; "
        f"normal-tail limit {_fmt_pct(summary.get('normal_tail_loss_limit'))}; "
        f"catastrophic-stress limit {_fmt_pct(summary.get('catastrophic_stress_loss_limit'))}; "
        f"scenario calibration: {summary.get('scenario_calibration_status', 'unknown')}."
    )


def _decision_card_html(
    label: str, answer: object, detail: object, tone: object | None = None
) -> str:
    tone_text = normalize_tone(tone)
    tone_class = "" if tone_text == "neutral" else f" v2-decision-card-{html.escape(tone_text)}"
    return (
        f'<div class="v2-decision-card{tone_class}">'
        f'<p class="v2-card-label">{html.escape(str(label))}{help_icon(metric_help(str(label)))}</p>'
        f'<p class="v2-decision-answer">{html.escape(str(answer))}</p>'
        f'<p class="v2-decision-detail">{html.escape(str(detail))}</p>'
        "</div>"
    )


def _scenario_probability_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No scenario links"
    scenario_col = "scenario" if "scenario" in frame else frame.columns[0]
    probability_col = "probability" if "probability" in frame else ""
    bucket_col = "risk_bucket" if "risk_bucket" in frame else ""
    rows: list[str] = []
    sortable = frame.copy()
    if probability_col:
        sortable[probability_col] = pd.to_numeric(sortable[probability_col], errors="coerce")
        sortable = sortable.sort_values(probability_col, ascending=False)
    for _, row in sortable.head(4).iterrows():
        probability = _fmt_pct(row.get(probability_col)) if probability_col else "n/a"
        bucket = f", {row.get(bucket_col)}" if bucket_col else ""
        rows.append(f"{row.get(scenario_col)} ({probability}{bucket})")
    return "; ".join(rows) if rows else "No scenario links"


def _event_pressure_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No active event pressure"
    status_col = "activation_status" if "activation_status" in frame else ""
    working = frame.copy()
    if status_col:
        active = working[
            ~working[status_col].astype(str).str.lower().isin({"inactive", "ignored", "watch_only"})
        ]
        if not active.empty:
            working = active
    title_col = _first_existing_column(working, ("title", "headline", "event", "name"))
    category_col = _first_existing_column(working, ("category", "event_category"))
    direction_col = _first_existing_column(working, ("direction", "event_direction"))
    rows: list[str] = []
    for _, row in working.head(5).iterrows():
        title = str(row.get(title_col, "event")).strip() if title_col else "event"
        tags = [
            str(row.get(column)).strip()
            for column in (category_col, direction_col)
            if column and str(row.get(column, "")).strip()
        ]
        rows.append(f"{title} ({', '.join(tags)})" if tags else title)
    return "; ".join(rows) if rows else "No active event pressure"


def _portfolio_risk_sentence(summary: dict[str, object]) -> str:
    return (
        f"Applied constraints: {summary.get('portfolio_constraints', 'none')}; "
        f"ES95 {_fmt_pct(summary.get('portfolio_expected_shortfall_95'))}; "
        f"max stress loss {_fmt_pct(summary.get('portfolio_max_stress_loss'))}; "
        f"equity beta {_fmt_float(summary.get('portfolio_equity_beta'))}; "
        f"AI beta {_fmt_float(summary.get('portfolio_ai_beta'))}."
    )


def _macro_inclusion_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No macro inclusion rows"
    decision_col = _first_existing_column(
        frame, ("decision", "inclusion_decision", "recommendation")
    )
    category_col = _first_existing_column(frame, ("category", "signal_group", "macro_category"))
    if decision_col is None:
        return f"{len(frame)} macro row(s) available"
    accepted = frame[
        frame[decision_col].astype(str).str.contains("include|accept|authority", case=False)
    ]
    watched = frame[~frame.index.isin(accepted.index)]
    if category_col is not None and not watched.empty:
        watched_categories = ", ".join(watched[category_col].astype(str).head(3).tolist())
        return f"{len(accepted)} authority row(s); {len(watched)} watch/rejected row(s): {watched_categories}"
    return f"{len(accepted)} authority row(s); {len(watched)} watch/rejected row(s)"


def _scenario_mix_tone(summary: dict[str, object]) -> str:
    risk_off = _float_or_none(summary.get("one_month_risk_off_probability")) or 0.0
    transition = _float_or_none(summary.get("one_month_transition_probability")) or 0.0
    risk_on = _float_or_none(summary.get("one_month_risk_on_probability")) or 0.0
    if risk_off >= 0.35:
        return "critical"
    if risk_off + transition >= 0.25:
        return "warning"
    if risk_on >= 0.45:
        return "success"
    return "neutral"


def _float_or_none(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _fmt_float(value: object) -> str:
    try:
        return f"{float(str(value)):.2f}"
    except (TypeError, ValueError):
        return "n/a"
