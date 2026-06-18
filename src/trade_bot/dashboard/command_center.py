from __future__ import annotations

import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.research.baselines import BaselineRun


def _render_command_center(baseline_run: BaselineRun) -> None:
    current_state = baseline_run.current_state
    trade_decision = baseline_run.trade_decision

    st.subheader("Current State")
    st.caption(
        "Market context behind the recommendation. Use this to understand the risk regime, not as a standalone trade command."
    )
    metric_cols = st.columns(4)
    _helped_metric(metric_cols[0], "Market Date", current_state.market_date)
    _helped_metric(metric_cols[1], "Risk Status", current_state.risk_status.upper())
    _helped_metric(metric_cols[2], "Risk Score", f"{current_state.risk_score:.2f}")
    _helped_metric(metric_cols[3], "Tracked Tickers", f"{baseline_run.prices.shape[1]:,}")
    st.write(current_state.risk_summary)

    st.subheader("Trade Decision")
    st.caption(
        "Actionable bridge from model posture to reviewable target weights and ticket direction."
    )
    if trade_decision.summary.empty:
        st.write("No trade-decision diagnostics available.")
    else:
        decision_summary = trade_decision.summary.iloc[0]
        decision_cols = st.columns(4)
        _helped_metric(decision_cols[0], "Action", str(decision_summary["recommended_action"]))
        _helped_metric(
            decision_cols[1],
            "Risk Budget",
            f"{float(decision_summary['risk_budget_multiplier']):.2f}",
        )
        _helped_metric(
            decision_cols[2],
            "1M Risk-Off",
            f"{float(decision_summary['one_month_risk_off_probability']):.0%}",
            key="one_month_risk_off_probability",
        )
        _helped_metric(decision_cols[3], "Authority", str(decision_summary["decision_authority"]))
        st.write(str(decision_summary["human_explanation"]))
        _render_metric_dataframe(_display_metrics(trade_decision.position_plan))
        st.dataframe(trade_decision.evidence, use_container_width=True)
        _render_metric_dataframe(_display_metrics(trade_decision.scenario_links))

    st.subheader("Trading Alerts")
    st.dataframe(current_state.strategy_alerts, use_container_width=True)

    st.subheader("Future-State Scenario Rollup")
    _render_metric_dataframe(_display_metrics(current_state.scenario_outlook.copy()))
