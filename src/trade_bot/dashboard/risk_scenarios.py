from __future__ import annotations

import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.research.baselines import BaselineRun


def _render_risk_and_scenarios(baseline_run: BaselineRun) -> None:
    current_state = baseline_run.current_state
    trade_decision = baseline_run.trade_decision

    st.subheader("Portfolio Risk Engine")
    st.caption(
        "Sizing guardrails. These rows explain whether scenarios, stress loss, beta, or concentration changed the target position."
    )
    portfolio_risk = baseline_run.portfolio_risk or trade_decision.portfolio_risk
    if portfolio_risk is None or portfolio_risk.summary.empty:
        st.write("No portfolio risk diagnostics available.")
    else:
        risk_summary = portfolio_risk.summary.iloc[0]
        risk_cols = st.columns(6)
        _helped_metric(risk_cols[0], "Risk Level", str(risk_summary["portfolio_risk_level"]))
        _helped_metric(
            risk_cols[1],
            "Risk Multiplier",
            f"{float(risk_summary['portfolio_risk_multiplier']):.2f}",
            key="portfolio_risk_multiplier",
        )
        _helped_metric(
            risk_cols[2],
            "ES 95",
            f"{float(risk_summary['post_expected_shortfall_95']):.2%}",
            key="post_expected_shortfall_95",
        )
        _helped_metric(
            risk_cols[3],
            "Max Stress Loss",
            f"{float(risk_summary['post_max_stress_loss']):.2%}",
            key="post_max_stress_loss",
        )
        _helped_metric(
            risk_cols[4],
            "Equity Beta",
            f"{float(risk_summary['post_equity_beta']):.2f}",
            key="post_equity_beta",
        )
        _helped_metric(
            risk_cols[5],
            "AI Beta",
            f"{float(risk_summary['post_ai_beta']):.2f}",
            key="post_ai_beta",
        )

        (
            risk_constraints_tab,
            risk_factor_tab,
            risk_tail_tab,
            risk_correlation_tab,
            risk_scenario_tab,
        ) = st.tabs(["Constraints", "Factors / Betas", "Tail / Stress", "Correlation", "Scenarios"])
        with risk_constraints_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.summary))
            _render_metric_dataframe(_display_metrics(portfolio_risk.constraint_report))
            st.caption("Risk-engine sizing bridge")
            _render_metric_dataframe(_display_metrics(portfolio_risk.sizing_adjustments))
        with risk_factor_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.factor_exposures))
            st.caption("Pre- versus post-risk beta decomposition")
            _render_metric_dataframe(_display_metrics(portfolio_risk.beta_decomposition))
        with risk_tail_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.tail_risk))
            _render_metric_dataframe(_display_metrics(portfolio_risk.stress_tests))
        with risk_correlation_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.correlation_regime))
            _render_metric_dataframe(_display_metrics(portfolio_risk.marginal_risk_contribution))
        with risk_scenario_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.scenario_risk_budget))

    st.subheader("Future-State Scenario Lattice")
    _render_metric_dataframe(_display_metrics(current_state.scenario_drivers))

    scenario_lattice = current_state.scenario_lattice
    scenario_horizon = st.radio(
        "Scenario horizon",
        ["1w", "1m", "3m", "6m"],
        index=1,
        horizontal=True,
    )
    scenario_bucket_options = ["all", *sorted(scenario_lattice["risk_bucket"].unique())]
    scenario_bucket = st.selectbox("Risk bucket", scenario_bucket_options)
    scenario_limit = st.slider("Scenarios shown", min_value=5, max_value=20, value=12, step=1)
    scenario_view = scenario_lattice[scenario_lattice["horizon"] == scenario_horizon]
    if scenario_bucket != "all":
        scenario_view = scenario_view[scenario_view["risk_bucket"] == scenario_bucket]
    _render_metric_dataframe(
        _display_metrics(scenario_view.sort_values("rank").head(scenario_limit))
    )

    st.subheader("Risk Confirmation Matrix")
    st.dataframe(current_state.confirmation_matrix, use_container_width=True)

    st.subheader("Market Health")
    _render_metric_dataframe(_display_metrics(current_state.market_health))

    st.subheader("Vol-Adjusted Momentum Signal Table")
    momentum_filter = st.radio(
        "Momentum filter",
        ["all", "bullish", "neutral", "bearish"],
        horizontal=True,
    )
    momentum_state_table = current_state.momentum_state.copy()
    if momentum_filter != "all":
        momentum_state_table = momentum_state_table[momentum_state_table["momentum_state_label"] == momentum_filter]
    _render_metric_dataframe(_display_metrics(momentum_state_table.head(75)))
