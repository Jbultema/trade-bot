from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.trends import (
    compact_metric_line_figure,
    latest_per_market_date,
    load_monitoring_trend_frame,
    load_simulation_validation_trend_frame,
    load_snapshot_trend_frames,
    long_metric_line_figure,
)
from trade_bot.DEFAULTS import (
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.trading.book_alignment import BookAlignmentRun


def _render_command_center(
    baseline_run: BaselineRun,
    book_alignment: BookAlignmentRun | None = None,
    *,
    run_store_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
) -> None:
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
        execution_plan = (
            book_alignment.position_plan
            if book_alignment is not None and not book_alignment.position_plan.empty
            else trade_decision.position_plan
        )
        if book_alignment is not None and not book_alignment.summary.empty:
            st.caption(
                "Book-aware execution bridge for the default paper champion. "
                "This reconciles logged paper holdings to the latest target."
            )
        _render_metric_dataframe(_display_metrics(execution_plan))
        if book_alignment is not None and not book_alignment.summary.empty:
            with st.expander("Raw model position bridge", expanded=False):
                _render_metric_dataframe(_display_metrics(trade_decision.position_plan))
        st.dataframe(trade_decision.evidence, use_container_width=True)
        _render_metric_dataframe(_display_metrics(trade_decision.scenario_links))

    st.subheader("Trading Alerts")
    st.dataframe(current_state.strategy_alerts, use_container_width=True)

    st.subheader("Future-State Scenario Rollup")
    _render_metric_dataframe(_display_metrics(current_state.scenario_outlook.copy()))
    _render_change_over_time_station(
        run_store_path=str(run_store_path),
        artifact_dir=str(artifact_dir),
        job_log_dir=str(job_log_dir),
        warehouse_path=str(warehouse_path),
    )


def _render_change_over_time_station(
    *,
    run_store_path: str,
    artifact_dir: str,
    job_log_dir: str,
    warehouse_path: str,
) -> None:
    st.subheader("Change-Over-Time Station")
    st.caption(
        "Small history plots for the operating metrics where direction matters. Use this to "
        "separate a fresh break from a condition that has persisted across recent snapshots."
    )
    metric_history, component_history, scenario_driver_history, driver_rotation_history = (
        load_snapshot_trend_frames(run_store_path, artifact_dir, job_log_dir)
    )
    metric_history = latest_per_market_date(metric_history)
    component_history = latest_per_market_date(component_history, subset=["component"])
    scenario_driver_history = latest_per_market_date(
        scenario_driver_history,
        subset=["driver"],
    )
    driver_rotation_history = latest_per_market_date(
        driver_rotation_history,
        subset=["driver"],
    )

    trend_tabs = st.tabs(
        [
            "Current State",
            "Risk Constraints",
            "Instability",
            "Macro Drivers",
            "Monitoring",
            "Simulation Quality",
        ]
    )
    with trend_tabs[0]:
        _render_current_state_trends(metric_history)
    with trend_tabs[1]:
        _render_risk_constraint_trends(metric_history)
    with trend_tabs[2]:
        _render_instability_trends(metric_history, component_history)
    with trend_tabs[3]:
        _render_macro_driver_trends(scenario_driver_history, driver_rotation_history)
    with trend_tabs[4]:
        monitoring_history = load_monitoring_trend_frame(warehouse_path)
        _render_monitoring_trends(monitoring_history)
    with trend_tabs[5]:
        validation_history = load_simulation_validation_trend_frame(warehouse_path)
        _render_simulation_quality_trends(validation_history)


def _render_current_state_trends(metric_history: pd.DataFrame) -> None:
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
        height=300,
    )
    _render_trend_or_empty(figure, "No current-state snapshot history is available yet.")


def _render_risk_constraint_trends(metric_history: pd.DataFrame) -> None:
    cols = st.columns(2)
    with cols[0]:
        figure = compact_metric_line_figure(
            metric_history,
            columns=[
                "portfolio_risk_multiplier",
                "post_expected_shortfall_95",
                "post_max_stress_loss",
            ],
            labels={
                "portfolio_risk_multiplier": "Risk multiplier",
                "post_expected_shortfall_95": "ES 95",
                "post_max_stress_loss": "Max stress loss",
            },
            title="Sizing Clamp and Tail Risk",
            yaxis_title="Value",
            height=280,
        )
        _render_trend_or_empty(figure, "No portfolio risk constraint history is available.")
    with cols[1]:
        figure = compact_metric_line_figure(
            metric_history,
            columns=["post_equity_beta", "post_ai_beta", "correlation_shift"],
            labels={
                "post_equity_beta": "Equity beta",
                "post_ai_beta": "AI beta",
                "correlation_shift": "Correlation shift",
            },
            title="Beta and Correlation Pressure",
            yaxis_title="Exposure / shift",
            height=280,
        )
        _render_trend_or_empty(figure, "No beta or correlation history is available.")


def _render_instability_trends(
    metric_history: pd.DataFrame,
    component_history: pd.DataFrame,
) -> None:
    cols = st.columns(2)
    with cols[0]:
        figure = compact_metric_line_figure(
            metric_history,
            columns=["regime_instability_score", "spy_ytd_large_move_share"],
            labels={
                "regime_instability_score": "Instability score",
                "spy_ytd_large_move_share": "SPY +/-1% YTD share",
            },
            title="Regime Instability Over Time",
            yaxis_title="Score / share",
            height=280,
        )
        _render_trend_or_empty(figure, "No regime-instability history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            component_history,
            category_column="component",
            value_column="component_score",
            title="Top Instability Components",
            yaxis_title="Component score",
            top_n=6,
            height=280,
        )
        _render_trend_or_empty(figure, "No regime-instability component history is available.")


def _render_macro_driver_trends(
    scenario_driver_history: pd.DataFrame,
    driver_rotation_history: pd.DataFrame,
) -> None:
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            scenario_driver_history,
            category_column="driver",
            value_column="score",
            title="Scenario Driver Scores",
            yaxis_title="Score",
            top_n=7,
            height=300,
        )
        _render_trend_or_empty(figure, "No scenario driver score history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            driver_rotation_history,
            category_column="driver_label",
            value_column="current_activation",
            title="Driver Rotation Activation",
            yaxis_title="Activation",
            top_n=7,
            height=300,
        )
        _render_trend_or_empty(figure, "No driver-rotation history is available.")


def _render_monitoring_trends(monitoring_history: pd.DataFrame) -> None:
    if monitoring_history.empty:
        st.info("No monitoring valuation history is stored yet.")
        return
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            monitoring_history,
            category_column="window_label",
            value_column="excess_return",
            title="Forward Excess Return by Monitored Window",
            yaxis_title="Excess return",
            percent=True,
            top_n=8,
            height=300,
        )
        _render_trend_or_empty(figure, "No excess-return monitoring history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            monitoring_history,
            category_column="window_label",
            value_column="drawdown_envelope_used",
            title="Drawdown Envelope Used",
            yaxis_title="Envelope used",
            percent=True,
            top_n=8,
            height=300,
        )
        _render_trend_or_empty(figure, "No drawdown-envelope history is available.")
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            monitoring_history,
            category_column="window_label",
            value_column="drawdown",
            title="Forward Drawdown",
            yaxis_title="Drawdown",
            percent=True,
            top_n=8,
            height=280,
        )
        _render_trend_or_empty(figure, "No drawdown history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            monitoring_history,
            category_column="window_label",
            value_column="beta_adjusted_spy_delta",
            title="Beta-Adjusted S&P Delta",
            yaxis_title="Delta",
            percent=True,
            top_n=8,
            height=280,
        )
        _render_trend_or_empty(figure, "No beta-adjusted delta history is available.")


def _render_simulation_quality_trends(validation_history: pd.DataFrame) -> None:
    if validation_history.empty:
        st.info("No simulation validation metrics are stored yet.")
        return
    summary = validation_history[
        validation_history["metric_scope"].astype(str).isin(
            ["primary_summary", "horizon_summary", "ablation_summary"]
        )
    ].copy()
    if summary.empty:
        st.info("No summary-level simulation validation metrics are stored yet.")
        return
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            summary,
            category_column="horizon",
            value_column="coverage_error",
            title="Coverage Error by Horizon",
            yaxis_title="Coverage error",
            percent=True,
            top_n=6,
            height=300,
        )
        _render_trend_or_empty(figure, "No coverage-error history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            summary,
            category_column="horizon",
            value_column="median_abs_error",
            title="Median Forecast Miss by Horizon",
            yaxis_title="Median absolute error",
            percent=True,
            top_n=6,
            height=300,
        )
        _render_trend_or_empty(figure, "No median-error history is available.")
    cols = st.columns(2)
    with cols[0]:
        figure = long_metric_line_figure(
            summary,
            category_column="variant",
            value_column="launch_action_score",
            title="Launch Action Score by Variant",
            yaxis_title="Score",
            percent=True,
            top_n=6,
            height=280,
        )
        _render_trend_or_empty(figure, "No launch-action score history is available.")
    with cols[1]:
        figure = long_metric_line_figure(
            summary,
            category_column="variant",
            value_column="constructive_capture_rate",
            title="Constructive Capture by Variant",
            yaxis_title="Capture rate",
            percent=True,
            top_n=6,
            height=280,
        )
        _render_trend_or_empty(figure, "No constructive-capture history is available.")


def _render_trend_or_empty(figure: object, empty_message: str) -> None:
    if getattr(figure, "data", None):
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info(empty_message)
