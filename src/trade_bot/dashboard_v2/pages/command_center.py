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
from trade_bot.dashboard_v2.components.cards import render_callout, render_card_grid
from trade_bot.dashboard_v2.perf import timed
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime


def render_today_page(runtime: DashboardRuntime) -> None:
    run = runtime.baseline_run
    current_state = run.current_state
    trade_decision = run.trade_decision
    summary = _first_row(trade_decision.summary)

    render_card_grid(
        [
            ("Market Date", current_state.market_date),
            ("Risk", str(current_state.risk_status).upper()),
            ("Risk Score", f"{float(current_state.risk_score):.2f}"),
            ("Open Tickets", runtime.open_ticket_count),
            ("Risk Budget", _fmt_pct(summary.get("risk_budget_multiplier"))),
            ("1M Risk-Off", _fmt_pct(summary.get("one_month_risk_off_probability"))),
        ]
    )
    render_callout(str(getattr(runtime.action_headline, "headline", "")) or current_state.risk_summary)

    view = st.pills(
        "Today view",
        ["Decision", "Operating brief", "Trends", "Raw evidence"],
        default="Decision",
        selection_mode="single",
        key="dashboard_v2_today_view",
    )
    selected_view = view or "Decision"

    if selected_view == "Decision":
        st.subheader("Trade Decision")
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
            metric_history, _component_history, _scenario_driver_history, _driver_rotation_history = (
                load_snapshot_trend_frames(
                    str(runtime.paths.run_store_path),
                    str(runtime.paths.artifact_dir),
                    str(runtime.paths.job_log_dir),
                )
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
            st.plotly_chart(figure, use_container_width=True)
    else:
        st.subheader("Raw Evidence")
        st.dataframe(trade_decision.evidence, use_container_width=True)
        st.subheader("Trading Alerts")
        st.dataframe(current_state.strategy_alerts, use_container_width=True)
        st.subheader("Scenario Outlook")
        _render_metric_dataframe(_display_metrics(current_state.scenario_outlook.copy()))


def _first_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _render_decision_context(
    runtime: DashboardRuntime,
    summary: dict[str, object],
    plan: pd.DataFrame,
) -> None:
    run = runtime.baseline_run
    current_state = run.current_state
    cards = [
        (
            "Risk State",
            f"{str(current_state.risk_status).upper()} ({_fmt_float(current_state.risk_score)})",
            current_state.risk_summary,
        ),
        (
            "1M Scenario Mix",
            _scenario_probability_summary(run.trade_decision.scenario_links),
            "These scenarios set the starting risk budget before event, macro, and portfolio-risk clamps.",
        ),
        (
            "Current Event Pressure",
            _event_pressure_summary(run.news_monitor.triage),
            f"Event pressure: {_fmt_pct(summary.get('event_pressure'))}. News can pressure sizing, but it should not override tradable confirmation by itself.",
        ),
        (
            "Target Posture",
            str(summary.get("scenario_adjusted_position", "n/a")),
            f"Base systematic posture: {summary.get('base_position', 'n/a')}.",
        ),
        (
            "Portfolio Risk Engine",
            str(summary.get("portfolio_risk_level", "n/a")).replace("_", " "),
            _portfolio_risk_sentence(summary),
        ),
        (
            "Macro Inclusion",
            _macro_inclusion_summary(run.signal_inclusion.summary),
            f"Macro pressure in the current decision summary: {_fmt_pct(summary.get('macro_pressure'))}.",
        ),
        (
            "Decision Sanity",
            str(summary.get("decision_sanity_signal", "n/a")).replace("_", " "),
            str(summary.get("decision_sanity_note", "n/a")),
        ),
        (
            "Posture Calibration",
            str(summary.get("posture_calibration_signal", "n/a")).replace("_", " "),
            str(summary.get("posture_calibration_note", "n/a")),
        ),
    ]
    st.markdown(
        '<div class="v2-decision-grid">'
        + "".join(_decision_card_html(label, answer, detail) for label, answer, detail in cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    if not plan.empty:
        st.caption("Material target weights and current-book drift are shown below.")


def _decision_card_html(label: str, answer: object, detail: object) -> str:
    return (
        '<div class="v2-decision-card">'
        f'<p class="v2-card-label">{html.escape(str(label))}</p>'
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
    decision_col = _first_existing_column(frame, ("decision", "inclusion_decision", "recommendation"))
    category_col = _first_existing_column(frame, ("category", "signal_group", "macro_category"))
    if decision_col is None:
        return f"{len(frame)} macro row(s) available"
    accepted = frame[frame[decision_col].astype(str).str.contains("include|accept|authority", case=False)]
    watched = frame[~frame.index.isin(accepted.index)]
    if category_col is not None and not watched.empty:
        watched_categories = ", ".join(watched[category_col].astype(str).head(3).tolist())
        return f"{len(accepted)} authority row(s); {len(watched)} watch/rejected row(s): {watched_categories}"
    return f"{len(accepted)} authority row(s); {len(watched)} watch/rejected row(s)"


def _first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
