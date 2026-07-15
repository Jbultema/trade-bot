from __future__ import annotations

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
        st.write(str(summary.get("human_explanation", current_state.risk_summary)))
        plan = (
            runtime.execution_book_alignment.position_plan
            if runtime.execution_book_alignment is not None
            and not runtime.execution_book_alignment.position_plan.empty
            else trade_decision.position_plan
        )
        _render_metric_dataframe(_display_metrics(plan))
        with st.expander("Scenario links", expanded=False):
            _render_metric_dataframe(_display_metrics(trade_decision.scenario_links))
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
