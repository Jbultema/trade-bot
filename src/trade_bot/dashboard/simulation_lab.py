from __future__ import annotations

import hashlib
import html
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import (
    _clearable_selectbox,
    _helped_metric,
    _render_metric_dataframe,
    _render_runtime_notice,
)
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _escape_markdown_dollars,
    _format_currency,
    _format_decimal,
    _format_percent,
)
from trade_bot.dashboard.strategy_candidates import outcome_strategy_option_frame
from trade_bot.dashboard.trends import (
    load_simulation_validation_trend_frame,
    long_metric_line_figure,
)
from trade_bot.DEFAULTS import (
    DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW,
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
    DEFAULT_SIMULATION_REFERENCE_STRATEGIES,
)
from trade_bot.research.approach_explorer import (
    build_approach_backtest_result,
    build_approach_catalog,
    decision_sanity_from_catalog_row,
    execution_for_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.forward_simulation import (
    REGIME_BUCKETS,
    ForwardSimulationConfig,
    build_regime_return_library,
    regime_mix_frame,
    scenario_probability_frame,
    simulate_factor_conditioned_paths,
    simulate_regime_conditioned_paths,
    simulation_settings_frame,
    summarize_forward_simulation,
)
from trade_bot.research.strategy_outcome_utility import (
    OutcomeBootstrapConfig,
    bootstrap_outcome_paths,
    contribution_periods_per_year,
    summarize_bootstrap_outcomes,
    terminal_wealth_from_cagr,
)
from trade_bot.storage.warehouse import TradingWarehouse


def _render_simulation_lab(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    *,
    warehouse_path: str = "",
) -> None:
    st.subheader("Simulation Lab")
    st.caption(
        "Forward-looking planning workbench. This section separates deterministic CAGR math, "
        "historical sequence risk, and scenario-conditioned forward paths so the future-state "
        "engine has its own inspection surface."
    )

    _render_simulation_planning_cards()
    _render_simulation_method_guide()
    simulation_view = (
        st.pills(
            "Simulation Lab view",
            [
                "Future-State Map",
                "Strategy Simulations",
                "Validation History",
                "Interpretability",
            ],
            selection_mode="single",
            default="Future-State Map",
            key="simulation_lab_view",
        )
        or "Future-State Map"
    )
    _render_simulation_lab_direct_view(
        simulation_view,
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
        warehouse_path=warehouse_path,
    )


def _render_simulation_lab_direct_view(
    simulation_view: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    warehouse_path: str = "",
) -> None:
    _render_simulation_view_runtime_notice(simulation_view)

    selected_strategy: str | None = None
    selected_scorecard: pd.Series | None = None
    selected_result: BacktestResult | None = None
    if simulation_view in {"Strategy Simulations", "Interpretability"}:
        selected_strategy, selected_scorecard, selected_result = _selected_simulation_strategy(
            bot_config=bot_config,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
        )

    if simulation_view == "Strategy Simulations":
        scenario_source = _scenario_source_frame(baseline_run)
        _render_strategy_simulations(
            selected_strategy=selected_strategy,
            selected_scorecard=selected_scorecard,
            selected_result=selected_result,
            baseline_run=baseline_run,
            scenario_source=scenario_source,
        )
    elif simulation_view == "Validation History":
        _render_simulation_validation_history(
            warehouse_path=warehouse_path,
            selected_strategy=None,
        )
    elif simulation_view == "Interpretability":
        scenario_source = _scenario_source_frame(baseline_run)
        probabilities = scenario_probability_frame(scenario_source)
        _render_simulation_interpretability(
            selected_strategy=selected_strategy,
            selected_scorecard=selected_scorecard,
            selected_result=selected_result,
            baseline_run=baseline_run,
            scenario_source=scenario_source,
            probabilities=probabilities,
        )
    elif simulation_view == "Future-State Map":
        scenario_source = _scenario_source_frame(baseline_run)
        probabilities = scenario_probability_frame(scenario_source)
        _render_future_state_map(baseline_run, scenario_source, probabilities)


def _render_simulation_view_runtime_notice(simulation_view: str) -> None:
    if simulation_view == "Strategy Simulations":
        _render_runtime_notice(
            "Strategy Simulations can be slow on first render",
            (
                "Changing the selected strategy or reference overlays can recompute cached "
                "bootstrap, regime-conditioned, and factor-proxy path summaries. Repeat views "
                "should be faster while the cache is warm."
            ),
            tone="warning",
        )
    elif simulation_view == "Validation History":
        _render_runtime_notice(
            "Validation History reads persisted DuckDB metrics",
            (
                "This is usually faster than rerunning validation, but large origin-level "
                "history tables and charts can still take a moment to render."
            ),
            tone="neutral",
        )
    elif simulation_view == "Interpretability":
        _render_runtime_notice(
            "Interpretability renders several diagnostic tables",
            (
                "This view is meant for deeper review. It avoids rerunning validation, but it "
                "does render resemblance, regime, method, and scenario diagnostics together."
            ),
            tone="neutral",
        )


def _render_simulation_validation_history(
    *,
    warehouse_path: str,
    selected_strategy: str | None,
) -> None:
    st.markdown("**Rolling-Origin Validation History**")
    st.caption(
        "Reads persisted validation runs from the local DuckDB store. Use this to compare "
        "baseline regime blocks, duration-aware transitions, covariate matching, and factor "
        "proxy variants across repeated validation runs."
    )
    if not warehouse_path:
        st.info("No warehouse path was supplied for validation history.")
        return

    warehouse = TradingWarehouse(warehouse_path)
    runs = warehouse.simulation_validation_runs(limit=50)
    if selected_strategy and not runs.empty and "strategy" in runs:
        selected_runs = runs[runs["strategy"].astype(str) == selected_strategy].copy()
        if not selected_runs.empty:
            runs = selected_runs
    if runs.empty:
        st.info(
            "No simulation validation history is stored yet. Run "
            "`poetry run trade-bot validate-simulation-engine --ablation` to populate it."
        )
        return

    latest_run_id = str(runs.iloc[0]["validation_run_id"])
    latest_strategy = str(runs.iloc[0].get("strategy", ""))
    primary_metrics = warehouse.simulation_validation_metrics(
        validation_run_id=latest_run_id,
        metric_scope="primary_summary",
        limit=5,
    )
    ablation_metrics = warehouse.simulation_validation_metrics(
        metric_scope="ablation_summary",
        limit=500,
    )
    latest_ablation = ablation_metrics[
        ablation_metrics["validation_run_id"].astype(str) == latest_run_id
    ]
    origin_metrics = warehouse.simulation_validation_metrics(
        validation_run_id=latest_run_id,
        metric_scope="rolling_origin",
        limit=500,
    )
    horizon_metrics = warehouse.simulation_validation_metrics(
        validation_run_id=latest_run_id,
        metric_scope="horizon_summary",
        limit=50,
    )

    primary_row = (
        primary_metrics.iloc[0] if not primary_metrics.empty else pd.Series(runs.iloc[0].to_dict())
    )
    _render_simulation_validation_conclusion(
        latest_run=runs.iloc[0],
        primary_row=primary_row,
        latest_ablation=latest_ablation,
        origin_metrics=origin_metrics,
    )
    _render_simulation_horizon_summary(horizon_metrics)
    _render_simulation_quality_history(warehouse_path, selected_strategy=selected_strategy)

    visual_cols = st.columns([1.25, 1.0])
    with visual_cols[0]:
        if not origin_metrics.empty:
            st.plotly_chart(
                _simulation_validation_band_figure(origin_metrics),
                width="stretch",
            )
    with visual_cols[1]:
        if not latest_ablation.empty:
            st.plotly_chart(
                _simulation_ablation_comparison_figure(latest_ablation),
                width="stretch",
            )
        elif not origin_metrics.empty:
            st.plotly_chart(
                _simulation_validation_error_figure(origin_metrics),
                width="stretch",
            )

    run_columns = [
        "created_at_utc",
        "strategy",
        "market_date",
        "horizons",
        "paths",
        "interval_low",
        "interval_high",
        "primary_interval_coverage",
        "primary_coverage_error",
        "primary_median_abs_error",
        "primary_launch_decision_accuracy",
        "primary_distribution_calibration_read",
        "primary_action_readiness_read",
        "primary_validity_read",
    ]
    available_run_columns = [column for column in run_columns if column in runs]
    with st.expander("Validation run audit table", expanded=False):
        _render_metric_dataframe(
            _display_metrics(runs[available_run_columns].head(12)),
            hide_index=True,
        )

    if not latest_ablation.empty:
        st.caption("Latest ablation readout")
        ablation_columns = [
            "variant",
            "label",
            "rows",
            "origins",
            "interval_coverage",
            "coverage_error",
            "median_abs_error",
            "severe_drawdown_brier",
            "launch_decision_accuracy",
            "launch_action_score",
            "launch_overrisk_rate",
            "constructive_capture_rate",
            "distribution_calibration_read",
            "action_readiness_read",
            "validity_read",
        ]
        _render_metric_dataframe(
            _display_metrics(
                latest_ablation[
                    [column for column in ablation_columns if column in latest_ablation]
                ]
            ),
            hide_index=True,
        )
    else:
        st.info(
            "This validation run does not include ablation metrics. Re-run "
            "`poetry run trade-bot validate-simulation-engine --ablation` if you need the "
            "model-variant comparison."
        )

    if not ablation_metrics.empty:
        st.caption("Ablation history")
        if selected_strategy and selected_strategy != latest_strategy:
            selected_ablation = ablation_metrics[
                ablation_metrics["strategy"].astype(str) == selected_strategy
            ].copy()
        else:
            selected_ablation = ablation_metrics
        history_columns = [
            "created_at_utc",
            "strategy",
            "variant",
            "coverage_error",
            "median_abs_error",
            "launch_decision_accuracy",
            "launch_action_score",
            "launch_overrisk_rate",
            "distribution_calibration_read",
            "action_readiness_read",
            "validity_read",
        ]
        _render_metric_dataframe(
            _display_metrics(
                selected_ablation[
                    [column for column in history_columns if column in selected_ablation]
                ].head(40)
            ),
            hide_index=True,
        )
    if origin_metrics.empty:
        return
    st.caption("Latest rolling-origin observations")
    origin_columns = [
        "origin_date",
        "horizon",
        "train_days",
        "paths",
        "realized_return",
        "simulated_p10_return",
        "simulated_p50_return",
        "simulated_p90_return",
        "realized_in_interval",
        "p50_error",
        "simulated_launch_decision",
        "realized_launch_decision",
        "launch_action_error",
        "launch_overrisk",
        "captured_constructive_launch",
    ]
    origin_display = origin_metrics[
        [column for column in origin_columns if column in origin_metrics]
    ].rename(
        columns={
            "simulated_p10_return": "simulated_lower_band_return",
            "simulated_p90_return": "simulated_upper_band_return",
        }
    )
    _render_metric_dataframe(
        _display_metrics(origin_display),
        hide_index=True,
    )


def _render_simulation_horizon_summary(horizon_metrics: pd.DataFrame) -> None:
    if horizon_metrics.empty:
        st.info(
            "This validation run does not include per-horizon summary metrics yet. "
            "Re-run validation with the latest CLI to populate them."
        )
        return
    st.caption("Latest per-horizon readout")
    display_columns = [
        "horizon",
        "rows",
        "interval_coverage",
        "target_coverage",
        "coverage_error",
        "median_abs_error",
        "launch_decision_accuracy",
        "launch_action_score",
        "launch_overrisk_rate",
        "constructive_capture_rate",
        "distribution_calibration_read",
        "action_readiness_read",
        "validity_read",
    ]
    _render_metric_dataframe(
        _display_metrics(
            horizon_metrics[[column for column in display_columns if column in horizon_metrics]]
        ),
        hide_index=True,
    )


def _render_simulation_quality_history(
    warehouse_path: str,
    *,
    selected_strategy: str | None,
) -> None:
    history = load_simulation_validation_trend_frame(warehouse_path)
    if selected_strategy and not history.empty and "strategy" in history:
        scoped = history[history["strategy"].astype(str) == selected_strategy].copy()
        if not scoped.empty:
            history = scoped
    summary = history[
        history.get("metric_scope", pd.Series(dtype=str))
        .astype(str)
        .isin(["primary_summary", "horizon_summary", "ablation_summary"])
    ].copy()
    if summary.empty:
        return
    st.caption("Validation quality over time")
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
        if figure.data:
            st.plotly_chart(figure, width="stretch")
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
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    cols = st.columns(3)
    with cols[0]:
        figure = long_metric_line_figure(
            summary,
            category_column="variant",
            value_column="launch_action_score",
            title="Launch Action Score",
            yaxis_title="Score",
            percent=True,
            top_n=6,
            height=280,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    with cols[1]:
        figure = long_metric_line_figure(
            summary,
            category_column="variant",
            value_column="launch_overrisk_rate",
            title="Over-Risk Rate",
            yaxis_title="Rate",
            percent=True,
            top_n=6,
            height=280,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")
    with cols[2]:
        figure = long_metric_line_figure(
            summary,
            category_column="variant",
            value_column="constructive_capture_rate",
            title="Constructive Capture",
            yaxis_title="Rate",
            percent=True,
            top_n=6,
            height=280,
        )
        if figure.data:
            st.plotly_chart(figure, width="stretch")


def _render_simulation_validation_conclusion(
    *,
    latest_run: pd.Series,
    primary_row: pd.Series,
    latest_ablation: pd.DataFrame,
    origin_metrics: pd.DataFrame,
) -> None:
    interval_coverage = _safe_float(primary_row.get("interval_coverage"))
    target_coverage = _safe_float(primary_row.get("target_coverage"))
    coverage_error = _safe_float(primary_row.get("coverage_error"))
    median_abs_error = _safe_float(primary_row.get("median_abs_error"))
    launch_accuracy = _safe_float(primary_row.get("launch_decision_accuracy"))
    launch_action_score = _safe_float(primary_row.get("launch_action_score"))
    launch_overrisk_rate = _safe_float(primary_row.get("launch_overrisk_rate"))
    constructive_capture_rate = _safe_float(primary_row.get("constructive_capture_rate"))
    validity_read = str(
        primary_row.get("validity_read") or latest_run.get("primary_validity_read", "")
    )
    interval_share = _origin_interval_share(origin_metrics)

    plain_english_read = _validation_history_plain_english_read(
        validity_read=validity_read,
        interval_coverage=interval_coverage,
        target_coverage=target_coverage,
        coverage_error=coverage_error,
        median_abs_error=median_abs_error,
        launch_accuracy=launch_accuracy,
        latest_ablation=latest_ablation,
    )
    st.markdown(
        _simulation_validation_verdict_card(
            validity_read=validity_read,
            coverage_error=coverage_error,
            median_abs_error=median_abs_error,
            launch_accuracy=launch_accuracy,
            launch_action_score=launch_action_score,
            launch_overrisk_rate=launch_overrisk_rate,
            detail=plain_english_read,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _simulation_validation_metric_cards(
            interval_coverage=interval_coverage,
            target_coverage=target_coverage,
            coverage_error=coverage_error,
            median_abs_error=median_abs_error,
            launch_accuracy=launch_accuracy,
            launch_action_score=launch_action_score,
            launch_overrisk_rate=launch_overrisk_rate,
            constructive_capture_rate=constructive_capture_rate,
            interval_share=interval_share,
        ),
        unsafe_allow_html=True,
    )


def _validation_history_plain_english_read(
    *,
    validity_read: str,
    interval_coverage: float | None,
    target_coverage: float | None,
    coverage_error: float | None,
    median_abs_error: float | None,
    launch_accuracy: float | None,
    latest_ablation: pd.DataFrame,
) -> str:
    target_label = _format_percent(
        target_coverage if target_coverage is not None else _default_validation_target_coverage()
    )
    coverage_label = _format_percent(interval_coverage)
    miss_label = _format_percent(median_abs_error)
    launch_label = _format_percent(launch_accuracy)
    if coverage_error is not None and abs(coverage_error) <= 0.05:
        calibration = (
            f"The simulated return band is roughly calibrated: realized outcomes landed inside "
            f"the band {coverage_label} of the time versus a target near {target_label}."
        )
    elif coverage_error is not None and coverage_error < -0.05:
        calibration = (
            f"The simulated band is too narrow or too optimistic: realized outcomes only landed "
            f"inside it {coverage_label} of the time versus a target near {target_label}."
        )
    else:
        calibration = (
            f"The simulated band is too wide or too conservative: realized outcomes landed inside "
            f"it {coverage_label} of the time versus a target near {target_label}."
        )

    launch_warning = (
        f" The go/no-go call is still weak at {launch_label}, so use this as a planning range, "
        "not as a launch trigger."
        if launch_accuracy is not None and launch_accuracy < 0.50
        else f" The go/no-go call matched history {launch_label} of the time."
    )
    miss_warning = (
        f" The median simulated return missed realized outcomes by about {miss_label} on "
        "average, so the p50 line should be treated as directional rather than precise."
        if median_abs_error is not None
        else ""
    )
    ablation_read = _ablation_plain_english_read(latest_ablation)
    return f"{calibration}{miss_warning}{launch_warning} {ablation_read} Validity label: {validity_read}."


def _simulation_validation_verdict_card(
    *,
    validity_read: str,
    coverage_error: float | None,
    median_abs_error: float | None,
    launch_accuracy: float | None,
    launch_action_score: float | None,
    launch_overrisk_rate: float | None,
    detail: str,
) -> str:
    verdict = _simulation_validation_verdict(
        coverage_error=coverage_error,
        median_abs_error=median_abs_error,
        launch_accuracy=launch_accuracy,
        launch_action_score=launch_action_score,
        launch_overrisk_rate=launch_overrisk_rate,
    )
    escaped_title = html.escape(verdict["title"])
    escaped_copy = html.escape(verdict["copy"])
    escaped_detail = html.escape(detail)
    escaped_label = html.escape(str(validity_read or "unknown"))
    return (
        f'<div class="simulation-validation-verdict simulation-validation-{verdict["status"]}">'
        f'<div class="simulation-validation-verdict-kicker">Interpretation</div>'
        f"<h4>{escaped_title}</h4>"
        f"<p>{escaped_copy}</p>"
        f'<p class="simulation-validation-detail">{escaped_detail}</p>'
        f'<span class="simulation-validation-pill">{escaped_label}</span>'
        "</div>"
    )


def _simulation_validation_verdict(
    *,
    coverage_error: float | None,
    median_abs_error: float | None,
    launch_accuracy: float | None,
    launch_action_score: float | None,
    launch_overrisk_rate: float | None,
) -> dict[str, str]:
    coverage_status = _simulation_validation_metric_status("coverage", coverage_error)
    median_status = _simulation_validation_metric_status("median", median_abs_error)
    launch_status = _simulation_validation_metric_status("launch", launch_accuracy)
    action_score_status = _simulation_validation_metric_status("action_score", launch_action_score)
    over_risk_status = _simulation_validation_metric_status("overrisk", launch_overrisk_rate)
    if launch_status == "bad" or action_score_status == "bad" or over_risk_status == "bad":
        return {
            "status": "bad",
            "title": "Research-useful, not decision-ready",
            "copy": (
                "The return band can be useful for planning, but the launch/no-launch "
                "classification is too weak to drive trades by itself."
            ),
        }
    if "bad" in {coverage_status, median_status}:
        return {
            "status": "bad",
            "title": "Validation is failing at least one core check",
            "copy": (
                "Treat this run as a model-diagnostics result, not evidence that the "
                "simulation engine is ready to influence allocations."
            ),
        }
    if "warn" in {
        coverage_status,
        median_status,
        launch_status,
        action_score_status,
        over_risk_status,
    }:
        return {
            "status": "warn",
            "title": "Useful, but still only a planning range",
            "copy": (
                "The band is not obviously broken, but at least one accuracy check is "
                "marginal. Use it to compare scenarios rather than approve a strategy."
            ),
        }
    return {
        "status": "good",
        "title": "Calibration checks look healthy",
        "copy": (
            "The band hit rate, median miss, and launch/no-launch check are all inside "
            "the current dashboard thresholds."
        ),
    }


def _simulation_validation_metric_cards(
    *,
    interval_coverage: float | None,
    target_coverage: float | None,
    coverage_error: float | None,
    median_abs_error: float | None,
    launch_accuracy: float | None,
    launch_action_score: float | None,
    launch_overrisk_rate: float | None,
    constructive_capture_rate: float | None,
    interval_share: float | None,
) -> str:
    cards = [
        _simulation_validation_metric_card(
            label="Interval hit rate",
            value=_coverage_pair_label(interval_coverage, target_coverage),
            status=_simulation_validation_metric_status("coverage", coverage_error),
            read=_coverage_status_read(coverage_error),
        ),
        _simulation_validation_metric_card(
            label="Coverage miss",
            value=_format_percent(coverage_error),
            status=_simulation_validation_metric_status("coverage", coverage_error),
            read="Distance from target hit rate",
        ),
        _simulation_validation_metric_card(
            label="Median miss",
            value=_format_percent(median_abs_error),
            status=_simulation_validation_metric_status("median", median_abs_error),
            read="Average p50 forecast error",
        ),
        _simulation_validation_metric_card(
            label="Go/no-go accuracy",
            value=_format_percent(launch_accuracy),
            status=_simulation_validation_metric_status("launch", launch_accuracy),
            read="Exact wait/ramp/full hindsight match",
        ),
        _simulation_validation_metric_card(
            label="Action score",
            value=_format_percent(launch_action_score),
            status=_simulation_validation_metric_status("action_score", launch_action_score),
            read="Partial credit for being one step off",
        ),
        _simulation_validation_metric_card(
            label="Over-risk rate",
            value=_format_percent(launch_overrisk_rate),
            status=_simulation_validation_metric_status("overrisk", launch_overrisk_rate),
            read="How often simulated action was too aggressive",
        ),
        _simulation_validation_metric_card(
            label="Constructive capture",
            value=_format_percent(constructive_capture_rate),
            status=_simulation_validation_metric_status("capture", constructive_capture_rate),
            read="Did it participate when hindsight was constructive",
        ),
        _simulation_validation_metric_card(
            label="Origins in band",
            value=_format_percent(interval_share),
            status=_simulation_validation_metric_status("coverage", coverage_error),
            read="Realized outcomes inside the simulated band",
        ),
    ]
    return f'<div class="simulation-validation-grid">{"".join(cards)}</div>'


def _simulation_validation_metric_card(
    *,
    label: str,
    value: str,
    status: str,
    read: str,
) -> str:
    escaped_label = html.escape(label)
    escaped_value = html.escape(value)
    escaped_read = html.escape(read)
    escaped_status = html.escape(_simulation_validation_status_label(status))
    return (
        f'<div class="simulation-validation-card simulation-validation-{status}">'
        f'<div class="simulation-validation-card-label">{escaped_label}</div>'
        f"<strong>{escaped_value}</strong>"
        f"<p>{escaped_read}</p>"
        f'<span class="simulation-validation-pill">{escaped_status}</span>'
        "</div>"
    )


def _simulation_validation_metric_status(metric: str, value: float | None) -> str:
    if value is None:
        return "neutral"
    if metric == "coverage":
        miss = abs(value)
        if miss <= 0.03:
            return "good"
        if miss <= 0.08:
            return "warn"
        return "bad"
    if metric == "median":
        if value <= 0.03:
            return "good"
        if value <= 0.06:
            return "warn"
        return "bad"
    if metric == "launch":
        if value >= 0.60:
            return "good"
        if value >= 0.40:
            return "warn"
        return "bad"
    if metric == "action_score":
        if value >= 0.75:
            return "good"
        if value >= 0.55:
            return "warn"
        return "bad"
    if metric == "overrisk":
        if value <= 0.25:
            return "good"
        if value <= 0.45:
            return "warn"
        return "bad"
    if metric == "capture":
        if value >= 0.75:
            return "good"
        if value >= 0.55:
            return "warn"
        return "bad"
    return "neutral"


def _coverage_status_read(coverage_error: float | None) -> str:
    if coverage_error is None:
        return "No target coverage comparison"
    if coverage_error < -0.08:
        return "Too many realized outcomes missed the band"
    if coverage_error > 0.08:
        return "Band is likely too generous"
    if abs(coverage_error) > 0.03:
        return "Near target, but not tight"
    return "Close to target"


def _simulation_validation_status_label(status: str) -> str:
    return {
        "good": "Healthy",
        "warn": "Caution",
        "bad": "Weak",
        "neutral": "No read",
    }.get(status, "No read")


def _ablation_plain_english_read(latest_ablation: pd.DataFrame) -> str:
    if latest_ablation.empty:
        return "No ablation comparison is available for this run."
    best_error = _best_ablation_row(latest_ablation, "median_abs_error", absolute=False)
    best_coverage = _best_ablation_row(latest_ablation, "coverage_error", absolute=True)
    if best_error is None and best_coverage is None:
        return "Ablation rows are present, but the comparison metrics are incomplete."
    if best_error is not None and best_coverage is not None:
        if str(best_error.get("variant")) == str(best_coverage.get("variant")):
            return (
                f"The strongest variant on both coverage and median error is "
                f"{best_error.get('label', best_error.get('variant'))}."
            )
        return (
            f"Ablation is mixed: {best_coverage.get('label', best_coverage.get('variant'))} "
            f"has the best coverage, while {best_error.get('label', best_error.get('variant'))} "
            "has the lowest median error."
        )
    row = best_error if best_error is not None else best_coverage
    return f"The strongest available ablation read is {row.get('label', row.get('variant'))}."


def _best_ablation_row(
    frame: pd.DataFrame,
    column: str,
    *,
    absolute: bool,
) -> pd.Series | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.abs() if absolute else values
    values = values.dropna()
    if values.empty:
        return None
    return frame.loc[values.idxmin()]


def _coverage_pair_label(interval_coverage: float | None, target_coverage: float | None) -> str:
    if interval_coverage is None and target_coverage is None:
        return "n/a"
    target = (
        target_coverage if target_coverage is not None else _default_validation_target_coverage()
    )
    return f"{_format_percent(interval_coverage)} / {_format_percent(target)}"


def _default_validation_target_coverage() -> float:
    return (
        DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH
        - DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW
    )


def _origin_interval_share(origin_metrics: pd.DataFrame) -> float | None:
    if origin_metrics.empty or "realized_in_interval" not in origin_metrics:
        return None
    values = origin_metrics["realized_in_interval"].dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _simulation_validation_band_figure(origin_metrics: pd.DataFrame) -> go.Figure:
    frame = _origin_metrics_for_plot(origin_metrics)
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=frame["origin_date"],
            y=frame["simulated_p90_return"],
            mode="lines",
            line={"color": "rgba(37, 99, 235, 0.15)", "width": 1},
            name="Simulated upper band",
            hovertemplate="Upper band %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["origin_date"],
            y=frame["simulated_p10_return"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.14)",
            line={"color": "rgba(37, 99, 235, 0.18)", "width": 1},
            name="Simulated evaluation band",
            hovertemplate="Lower band %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["origin_date"],
            y=frame["simulated_p50_return"],
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            name="Simulated p50",
            hovertemplate="p50 %{y:.1%}<extra></extra>",
        )
    )
    colors = frame["realized_in_interval"].map({True: "#0f766e", False: "#dc2626"})
    fig.add_trace(
        go.Scatter(
            x=frame["origin_date"],
            y=frame["realized_return"],
            mode="markers",
            marker={"color": colors, "size": 8, "line": {"color": "white", "width": 1}},
            name="Realized return",
            hovertemplate="Realized %{y:.1%}<br>%{x|%Y-%m-%d}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Did realized returns land inside the simulated range?",
        margin={"l": 10, "r": 10, "t": 50, "b": 35},
        legend={"orientation": "h", "y": -0.22},
        yaxis={"tickformat": ".0%", "title": "3m return"},
        xaxis={"title": "Origin date"},
        hovermode="x unified",
    )
    return fig


def _simulation_ablation_comparison_figure(latest_ablation: pd.DataFrame) -> go.Figure:
    frame = latest_ablation.copy()
    frame["variant_label"] = frame["label"].fillna(frame["variant"]).astype(str)
    frame["absolute_coverage_error"] = pd.to_numeric(
        frame.get("coverage_error"),
        errors="coerce",
    ).abs()
    frame["median_abs_error"] = pd.to_numeric(frame.get("median_abs_error"), errors="coerce")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["variant_label"],
            y=frame["absolute_coverage_error"],
            name="Coverage miss",
            marker_color="#2563eb",
            hovertemplate="Coverage miss %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["variant_label"],
            y=frame["median_abs_error"],
            name="Median miss",
            marker_color="#f97316",
            hovertemplate="Median miss %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Which simulation variant is least wrong?",
        barmode="group",
        margin={"l": 10, "r": 10, "t": 50, "b": 80},
        legend={"orientation": "h", "y": -0.32},
        yaxis={"tickformat": ".0%", "title": "Error, lower is better"},
        xaxis={"tickangle": -20},
    )
    return fig


def _simulation_validation_error_figure(origin_metrics: pd.DataFrame) -> go.Figure:
    frame = _origin_metrics_for_plot(origin_metrics)
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(
        go.Histogram(
            x=frame["p50_error"],
            nbinsx=24,
            marker_color="#64748b",
            name="p50 error",
            hovertemplate="p50 error %{x:.1%}<br>Origins %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#0f172a")
    fig.update_layout(
        title="Was the simulated median biased high or low?",
        margin={"l": 10, "r": 10, "t": 50, "b": 35},
        xaxis={"tickformat": ".0%", "title": "p50 minus realized return"},
        yaxis={"title": "Origin count"},
        showlegend=False,
    )
    return fig


def _origin_metrics_for_plot(origin_metrics: pd.DataFrame) -> pd.DataFrame:
    required = [
        "origin_date",
        "realized_return",
        "simulated_p10_return",
        "simulated_p50_return",
        "simulated_p90_return",
    ]
    if origin_metrics.empty or any(column not in origin_metrics for column in required):
        return pd.DataFrame()
    frame = origin_metrics.copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="coerce")
    for column in [
        "realized_return",
        "simulated_p10_return",
        "simulated_p50_return",
        "simulated_p90_return",
        "p50_error",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "realized_in_interval" in frame:
        frame["realized_in_interval"] = frame["realized_in_interval"].astype(bool)
    else:
        frame["realized_in_interval"] = frame["realized_return"].ge(
            frame["simulated_p10_return"]
        ) & frame["realized_return"].le(frame["simulated_p90_return"])
    return frame.dropna(subset=required).sort_values("origin_date")


def _render_simulation_planning_cards() -> None:
    contribution_periods = contribution_periods_per_year(DEFAULT_OUTCOME_CONTRIBUTION_TIMING)
    contribution_amount = DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION / contribution_periods
    cadence = DEFAULT_OUTCOME_CONTRIBUTION_TIMING.replace("_", " ").title()
    cols = st.columns(5)
    _helped_metric(
        cols[0], "Starting Account", _format_currency(DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE)
    )
    _helped_metric(
        cols[1],
        "Annual Contribution",
        f"{_format_currency(DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION)} / yr",
    )
    _helped_metric(
        cols[2], "Contribution Cadence", f"{_format_currency(contribution_amount)} {cadence}"
    )
    _helped_metric(cols[3], "Horizon", f"{DEFAULT_OUTCOME_HORIZON_YEARS} years")
    _helped_metric(
        cols[4],
        "Soft / Hard DD",
        (
            f"{abs(DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT):.0%} / "
            f"{abs(DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT):.0%}"
        ),
    )


def _render_simulation_method_guide() -> None:
    with st.expander("How to read the simulation forms", expanded=False):
        _render_metric_dataframe(
            pd.DataFrame(
                [
                    {
                        "form": "Deterministic 15Y",
                        "what_it_does": "Applies historical CAGR to the configured starting balance and monthly contributions.",
                        "best_use": "Fast reference point for the growth frontier.",
                        "do_not_use_for": "Drawdown, path risk, or confidence.",
                    },
                    {
                        "form": "Historical bootstrap",
                        "what_it_does": "Resamples historical daily-return blocks from the selected strategy.",
                        "best_use": "Sequence risk and drawdown ranges if the future looks like the strategy's own past.",
                        "do_not_use_for": "A current-state-aware forecast.",
                    },
                    {
                        "form": "Regime-conditioned forward paths",
                        "what_it_does": "Samples historical return blocks by regime, then biases starting and transition states using today's scenario map.",
                        "best_use": "Forward planning ranges when current risk-off/transition/risk-on probabilities matter.",
                        "do_not_use_for": "A promise that the specific regime sequence will happen.",
                    },
                    {
                        "form": "Reference overlays",
                        "what_it_does": "Runs the same simulation logic on SPY/QQQ reference portfolios when available.",
                        "best_use": "Judging whether the selected strategy earns its complexity versus doing nothing.",
                        "do_not_use_for": "Declaring a strategy superior without Monitoring and out-of-sample evidence.",
                    },
                ]
            ),
            hide_index=True,
        )


def _render_future_state_map(
    baseline_run: BaselineRun,
    scenario_source: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> None:
    st.markdown("**Future-State Simulation Map**")
    st.caption(
        "This is the aggregate state input to the forward engine. Detailed current scenarios are "
        "mapped into broad empirical return buckets, then used to bias simulated starting states "
        "and future regime transitions."
    )
    risk_off = _probability_for_bucket(probabilities, "risk_off")
    transition = _probability_for_bucket(probabilities, "transition")
    fragile = _probability_for_bucket(probabilities, "risk_on_fragile")
    risk_on = _probability_for_bucket(probabilities, "risk_on")
    top = _top_scenario_row(scenario_source)
    cols = st.columns(5)
    _helped_metric(cols[0], "Risk-Off", _format_percent(risk_off))
    _helped_metric(cols[1], "Transition", _format_percent(transition))
    _helped_metric(cols[2], "Fragile Risk-On", _format_percent(fragile))
    _helped_metric(cols[3], "Broad Risk-On", _format_percent(risk_on))
    _helped_metric(cols[4], "Top Scenario", str(top.get("scenario", "n/a"))[:38])

    chart_cols = st.columns([1.05, 1.0])
    with chart_cols[0]:
        st.plotly_chart(_scenario_probability_figure(probabilities), width="stretch")
    with chart_cols[1]:
        st.caption("Current scenario records")
        scenario_columns = [
            "horizon",
            "rank",
            "scenario",
            "probability",
            "risk_bucket",
            "expected_bot_posture",
            "preferred_exposure",
            "confirmation",
            "off_ramp",
        ]
        available = [column for column in scenario_columns if column in scenario_source]
        if available:
            _render_metric_dataframe(
                _display_metrics(scenario_source[available].head(12)),
                hide_index=True,
            )
        else:
            st.write("No detailed scenario rows are available in the loaded run.")

    with st.expander("Simulation settings", expanded=False):
        st.caption("Regime-conditioned forward simulation settings")
        _render_metric_dataframe(
            simulation_settings_frame(ForwardSimulationConfig()), hide_index=True
        )
        st.caption("Current state context")
        _render_metric_dataframe(
            pd.DataFrame(
                [
                    {
                        "input": "market_date",
                        "value": baseline_run.current_state.market_date,
                        "meaning": "Date of the market snapshot used by the current scenario map.",
                    },
                    {
                        "input": "risk_status",
                        "value": str(baseline_run.current_state.risk_status).upper(),
                        "meaning": "Current risk posture that informs scenario interpretation.",
                    },
                    {
                        "input": "risk_score",
                        "value": _format_decimal(baseline_run.current_state.risk_score),
                        "meaning": "Current composite risk score from the operating run.",
                    },
                ]
            ),
            hide_index=True,
        )


def _selected_simulation_strategy(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> tuple[str | None, pd.Series | None, BacktestResult | None]:
    options = _strategy_option_frame(
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    if options.empty:
        st.warning("No reconstructable strategies are available for simulation.")
        return None, None, None

    labels = options["simulation_label"].tolist()
    selected_label = _clearable_selectbox(
        "Strategy to simulate",
        labels,
        key="simulation_lab_selected_strategy",
        placeholder="Search simulation strategies...",
    )
    if selected_label is None:
        st.info("Choose a strategy to run forward simulations.")
        return None, None, None
    row = options[options["simulation_label"] == selected_label].iloc[0]
    strategy_name = str(row["strategy"])
    result = _result_for_strategy(
        strategy_name,
        bot_config=bot_config,
        baseline_run=baseline_run,
    )
    return strategy_name, row, result


def _reference_option_frame(
    baseline_run: BaselineRun,
    selected_strategy: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for strategy_name, label in DEFAULT_SIMULATION_REFERENCE_STRATEGIES:
        if strategy_name == selected_strategy:
            continue
        if strategy_name in baseline_run.results:
            rows.append({"strategy": strategy_name, "label": label})
    return pd.DataFrame(rows)


def _reference_simulations(
    *,
    baseline_run: BaselineRun,
    reference_options: pd.DataFrame,
    selected_reference_labels: list[str],
    scenario_source: pd.DataFrame,
    path_count: int,
) -> list[dict[str, object]]:
    if reference_options.empty or not selected_reference_labels:
        return []
    selected_options = reference_options[
        reference_options["label"].astype(str).isin(selected_reference_labels)
    ].copy()
    colors = ("#0f766e", "#f97316", "#7c3aed", "#64748b")
    simulations: list[dict[str, object]] = []
    for position, (_, row) in enumerate(selected_options.iterrows()):
        strategy_name = str(row["strategy"])
        result = baseline_run.results.get(strategy_name)
        if result is None:
            continue
        returns = _returns_tuple(result)
        if len(returns) < 30:
            continue
        bootstrap_paths = _cached_bootstrap_paths(returns, path_count)
        forward_paths = _cached_regime_forward_paths(
            returns,
            _scenario_records(scenario_source),
            path_count,
        )
        simulations.append(
            {
                "strategy": strategy_name,
                "label": str(row["label"]),
                "deterministic_wealth": _deterministic_wealth(None, result),
                "bootstrap_paths": bootstrap_paths,
                "bootstrap_summary": summarize_bootstrap_outcomes(bootstrap_paths),
                "forward_paths": forward_paths,
                "forward_summary": summarize_forward_simulation(
                    forward_paths,
                    config=ForwardSimulationConfig(paths=path_count),
                ),
                "result": result,
                "color": colors[position % len(colors)],
            }
        )
    return simulations


def _render_strategy_simulations(
    *,
    selected_strategy: str | None,
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult | None,
    baseline_run: BaselineRun,
    scenario_source: pd.DataFrame,
) -> None:
    st.markdown("**Selected-Strategy Simulation Readout**")
    st.caption(
        "Use this to ask whether a candidate's wealth range and drawdown range are tolerable, "
        "not just whether its historical point estimate looked good."
    )
    if selected_strategy is None or selected_result is None:
        st.info("Pick a strategy with reconstructable history to see simulation paths.")
        return

    daily_returns = _daily_returns(selected_result)
    if daily_returns.empty:
        st.warning("The selected strategy has no usable daily return history.")
        return
    if len(daily_returns) < 30:
        st.warning(
            "The selected strategy has fewer than 30 daily observations. "
            "Simulation paths are suppressed until the candidate has enough history "
            "to produce a useful distribution."
        )
        return

    deterministic_wealth = _deterministic_wealth(selected_scorecard, selected_result)
    historical_metrics = _historical_result_metrics(selected_result)
    fast_cols = st.columns(5)
    _helped_metric(fast_cols[0], "Deterministic 15Y", _format_currency(deterministic_wealth))
    _helped_metric(fast_cols[1], "Historical CAGR", _format_percent(historical_metrics["cagr"]))
    _helped_metric(
        fast_cols[2],
        "Historical Max DD",
        _format_percent(historical_metrics["max_drawdown"]),
    )
    _helped_metric(
        fast_cols[3],
        "Historical Ulcer",
        _format_percent(historical_metrics["ulcer_index"]),
    )
    _helped_metric(fast_cols[4], "Return Days", f"{len(daily_returns):,}")

    _render_runtime_notice(
        "Path simulations are explicit-load",
        (
            "The fast read above avoids Monte Carlo work. Turn on path simulations when you "
            "need the bootstrap/regime fan charts; reference overlays and factor-proxy paths "
            "add additional compute."
        ),
        tone="warning",
    )
    load_path_simulations = st.toggle(
        "Load bootstrap and regime path simulations",
        value=False,
        key="simulation_lab_load_path_simulations",
        help=(
            "Runs cached bootstrap and regime-conditioned simulations for the selected strategy. "
            "First render after a strategy change can still take a moment."
        ),
    )
    if not load_path_simulations:
        st.info(
            "Path simulations are paused so this page stays responsive. Turn on the loader above "
            "when you are ready to wait for the distribution charts."
        )
        return

    simulation_profile = st.radio(
        "Simulation path budget",
        ["Quick (150 paths)", "Standard (600 paths)"],
        index=0,
        horizontal=True,
        key="simulation_lab_path_budget",
        help="Quick is intended for interactive review. Standard matches the default planning path count.",
    )
    path_count = 150 if simulation_profile.startswith("Quick") else 600
    include_reference_overlays = st.toggle(
        "Include reference overlays",
        value=False,
        key="simulation_lab_include_reference_overlays",
        help="Adds SPY/QQQ-style reference simulations. Useful, but each selected reference adds more work.",
    )
    include_factor_proxy = st.toggle(
        "Include factor-proxy diagnostics",
        value=False,
        key="simulation_lab_include_factor_proxy",
        help="Fits factor proxies and simulates factor-conditioned paths. This is the slowest optional layer.",
    )

    with st.spinner(f"Running {path_count:,} path bootstrap and regime simulations..."):
        bootstrap_paths = _cached_bootstrap_paths(_returns_tuple(selected_result), path_count)
        forward_paths = _cached_regime_forward_paths(
            _returns_tuple(selected_result),
            _scenario_records(scenario_source),
            path_count,
        )
    bootstrap_summary = summarize_bootstrap_outcomes(bootstrap_paths)
    forward_summary = summarize_forward_simulation(
        forward_paths,
        config=ForwardSimulationConfig(paths=path_count),
    )
    factor_inputs = (
        _factor_simulation_inputs(selected_result, baseline_run.prices)
        if include_factor_proxy
        else None
    )
    factor_paths = (
        _cached_factor_forward_paths(
            factor_inputs[0],
            factor_inputs[1],
            factor_inputs[2],
            _scenario_records(scenario_source),
            path_count,
        )
        if factor_inputs is not None
        else pd.DataFrame()
    )
    factor_summary = (
        summarize_forward_simulation(factor_paths, config=ForwardSimulationConfig(paths=path_count))
        if not factor_paths.empty
        else {}
    )
    reference_options = _reference_option_frame(baseline_run, selected_strategy)
    selected_reference_labels: list[str] = []
    if include_reference_overlays and not reference_options.empty:
        selected_reference_labels = st.multiselect(
            "Reference overlays",
            reference_options["label"].tolist(),
            default=[],
            help=(
                "Compare this strategy against major do-nothing references using the same "
                "bootstrap and regime-conditioned simulation settings."
            ),
            key="simulation_lab_reference_overlays",
        )
    reference_simulations = _reference_simulations(
        baseline_run=baseline_run,
        reference_options=reference_options,
        selected_reference_labels=selected_reference_labels,
        scenario_source=scenario_source,
        path_count=path_count,
    )

    cols = st.columns(6)
    _helped_metric(cols[0], "Deterministic 15Y", _format_currency(deterministic_wealth))
    _helped_metric(
        cols[1],
        "Bootstrap Median",
        _format_currency(bootstrap_summary.get("terminal_wealth_p50")),
    )
    _helped_metric(
        cols[2],
        "Forward Median",
        _format_currency(forward_summary.get("terminal_wealth_p50")),
    )
    _helped_metric(
        cols[3],
        "Forward P10",
        _format_currency(forward_summary.get("terminal_wealth_p10")),
    )
    _helped_metric(
        cols[4],
        "Median Forward DD",
        _format_percent(forward_summary.get("max_drawdown_p50")),
    )
    _helped_metric(
        cols[5],
        "Severe DD Prob",
        _format_percent(forward_summary.get("severe_drawdown_probability")),
    )

    st.caption(
        "Catastrophic-tail utility from the contribution-aware historical block bootstrap; "
        "these are resampled-path frequencies, not crash forecasts."
    )
    tail_cols = st.columns(4)
    _helped_metric(
        tail_cols[0],
        "P(DD > 10%)",
        _format_percent(bootstrap_summary.get("drawdown_over_10_probability")),
    )
    _helped_metric(
        tail_cols[1],
        "P(DD > 20%)",
        _format_percent(bootstrap_summary.get("drawdown_over_20_probability")),
    )
    _helped_metric(
        tail_cols[2],
        "P(DD > 30%)",
        _format_percent(bootstrap_summary.get("drawdown_over_30_probability")),
    )
    _helped_metric(
        tail_cols[3],
        "Avg DD If >20%",
        _format_percent(bootstrap_summary.get("expected_drawdown_if_over_20")),
    )

    st.info(
        _escape_markdown_dollars(
            _simulation_plain_english_read(
                selected_strategy,
                bootstrap_summary,
                forward_summary,
                reference_simulations=reference_simulations,
            )
        )
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            _simulation_histogram(
                bootstrap_paths,
                column="terminal_wealth",
                title="Historical sequence bootstrap",
                xaxis_title="Terminal wealth",
                color="#0f766e",
                hovertemplate="Terminal wealth %{x:$,.0f}<br>Paths %{y}<extra></extra>",
            ),
            width="stretch",
        )
    with chart_cols[1]:
        st.plotly_chart(
            _simulation_overlay_histogram(
                [
                    {
                        "label": selected_strategy,
                        "paths": forward_paths,
                        "color": "#2563eb",
                    },
                    *[
                        {
                            "label": str(reference["label"]),
                            "paths": reference["forward_paths"],
                            "color": reference["color"],
                        }
                        for reference in reference_simulations
                    ],
                ],
                column="terminal_wealth",
                title="Regime-conditioned forward paths vs references",
                xaxis_title="Terminal wealth",
                value_label="Terminal wealth",
                value_format="currency",
            ),
            width="stretch",
        )

    drawdown_cols = st.columns(2)
    with drawdown_cols[0]:
        st.plotly_chart(
            _simulation_histogram(
                forward_paths,
                column="max_drawdown",
                title="Selected strategy forward drawdown distribution",
                xaxis_title="Max drawdown",
                color="#dc2626",
                hovertemplate="Max drawdown %{x:.1%}<br>Paths %{y}<extra></extra>",
            ),
            width="stretch",
        )
    with drawdown_cols[1]:
        st.plotly_chart(
            _simulation_overlay_histogram(
                [
                    {
                        "label": selected_strategy,
                        "paths": forward_paths,
                        "color": "#dc2626",
                    },
                    *[
                        {
                            "label": str(reference["label"]),
                            "paths": reference["forward_paths"],
                            "color": reference["color"],
                        }
                        for reference in reference_simulations
                    ],
                ],
                column="max_drawdown",
                title="Forward drawdown paths vs references",
                xaxis_title="Max drawdown",
                value_label="Max drawdown",
                value_format="percent",
            ),
            width="stretch",
        )

    if reference_simulations:
        st.caption("Reference portfolio comparison")
        selected_forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
        comparison_rows = [
            _simulation_comparison_row(
                label="Selected strategy",
                strategy=selected_strategy,
                deterministic_wealth=deterministic_wealth,
                bootstrap_summary=bootstrap_summary,
                forward_summary=forward_summary,
                selected_forward_median=selected_forward_median,
            )
        ]
        comparison_rows.extend(
            _simulation_comparison_row(
                label=str(reference["label"]),
                strategy=str(reference["strategy"]),
                deterministic_wealth=reference["deterministic_wealth"],
                bootstrap_summary=reference["bootstrap_summary"],
                forward_summary=reference["forward_summary"],
                selected_forward_median=selected_forward_median,
            )
            for reference in reference_simulations
        )
        _render_metric_dataframe(_display_metrics(pd.DataFrame(comparison_rows)), hide_index=True)
    elif reference_options.empty:
        st.caption(
            "SPY/QQQ reference overlays are unavailable in this loaded run because the snapshot "
            "does not include buy-and-hold reference results."
        )

    st.caption("Drawdown distribution detail")
    drawdown_rows = [
        _drawdown_distribution_row(
            label="Selected bootstrap",
            strategy=selected_strategy,
            paths=bootstrap_paths,
        ),
        _drawdown_distribution_row(
            label="Selected forward",
            strategy=selected_strategy,
            paths=forward_paths,
        ),
    ]
    for reference in reference_simulations:
        drawdown_rows.append(
            _drawdown_distribution_row(
                label=f"{reference['label']} forward",
                strategy=str(reference["strategy"]),
                paths=reference["forward_paths"],
            )
        )
    _render_metric_dataframe(_display_metrics(pd.DataFrame(drawdown_rows)), hide_index=True)

    st.caption("Advanced simulation diagnostics")
    advanced_rows = [
        _advanced_simulation_row(
            model="Duration/covariate regime paths",
            summary=forward_summary,
        )
    ]
    if factor_summary:
        advanced_rows.append(
            _advanced_simulation_row(
                model="Factor-proxy paths",
                summary=factor_summary,
            )
        )
    else:
        advanced_rows.append(
            {
                "model": "Factor-proxy paths",
                "paths": 0,
                "terminal_wealth_p50": None,
                "max_drawdown_p50": None,
                "severe_drawdown_probability": None,
                "mean_regime_switches": None,
                "mean_max_risk_off_streak_days": None,
                "mean_covariate_match_distance": None,
                "factor_model_r_squared": None,
                "read": "Unavailable because the loaded snapshot lacks enough factor proxy history.",
            }
        )
    _render_metric_dataframe(_display_metrics(pd.DataFrame(advanced_rows)), hide_index=True)

    st.caption("Current-path resemblance")
    validation_rows = [
        _simulation_validation_row(
            label="Selected strategy",
            strategy=selected_strategy,
            result=selected_result,
            deterministic_wealth=deterministic_wealth,
            bootstrap_summary=bootstrap_summary,
            forward_summary=forward_summary,
        )
    ]
    for reference in reference_simulations:
        result = reference.get("result")
        if isinstance(result, BacktestResult):
            validation_rows.append(
                _simulation_validation_row(
                    label=str(reference["label"]),
                    strategy=str(reference["strategy"]),
                    result=result,
                    deterministic_wealth=reference["deterministic_wealth"],
                    bootstrap_summary=reference["bootstrap_summary"],
                    forward_summary=reference["forward_summary"],
                )
            )
    _render_metric_dataframe(_display_metrics(pd.DataFrame(validation_rows)), hide_index=True)

    st.caption("Outcome distribution summary")
    _render_metric_dataframe(
        _display_metrics(
            pd.DataFrame(
                [
                    _summary_row("Deterministic CAGR", deterministic_wealth, None),
                    _summary_row("Historical bootstrap", None, bootstrap_summary),
                    _summary_row("Regime-conditioned", None, forward_summary),
                ]
            )
        ),
        hide_index=True,
    )


def _drawdown_distribution_row(
    *,
    label: str,
    strategy: str,
    paths: pd.DataFrame,
) -> dict[str, object]:
    drawdowns = _numeric_path_column(paths, "max_drawdown")
    ulcers = _numeric_path_column(paths, "ulcer_index")
    return {
        "simulation": label,
        "strategy": strategy,
        "paths": int(paths.shape[0]),
        "max_drawdown_p10": _series_quantile(drawdowns, 0.10),
        "max_drawdown_p25": _series_quantile(drawdowns, 0.25),
        "max_drawdown_p50": _series_quantile(drawdowns, 0.50),
        "max_drawdown_p75": _series_quantile(drawdowns, 0.75),
        "max_drawdown_p90": _series_quantile(drawdowns, 0.90),
        "ulcer_index_p50": _series_quantile(ulcers, 0.50),
        "breach_soft_band": _series_mean(drawdowns <= DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT),
        "breach_hard_band": _series_mean(drawdowns <= DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT),
    }


def _advanced_simulation_row(
    *,
    model: str,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model,
        "paths": summary.get("paths"),
        "terminal_wealth_p50": summary.get("terminal_wealth_p50"),
        "max_drawdown_p50": summary.get("max_drawdown_p50"),
        "severe_drawdown_probability": summary.get("severe_drawdown_probability"),
        "mean_regime_switches": summary.get("mean_regime_switches"),
        "mean_max_risk_off_streak_days": summary.get("mean_max_risk_off_streak_days"),
        "mean_covariate_match_distance": summary.get("mean_covariate_match_distance"),
        "factor_model_r_squared": summary.get("factor_model_r_squared"),
        "read": _advanced_simulation_read(summary),
    }


def _advanced_simulation_read(summary: dict[str, object]) -> str:
    factor_r2 = _safe_float(summary.get("factor_model_r_squared"))
    match_distance = _safe_float(summary.get("mean_covariate_match_distance"))
    risk_off_streak = _safe_float(summary.get("mean_max_risk_off_streak_days"))
    if factor_r2 is not None:
        if factor_r2 >= 0.60:
            return "Factor proxy explains enough history to treat this as a useful stress lens."
        return "Factor proxy fit is weak; use as a fragility check, not as a strategy oracle."
    if match_distance is not None and match_distance <= 1.0:
        return "Sampled blocks are close to the latest trend/volatility/covariate state."
    if risk_off_streak is not None and risk_off_streak >= 40:
        return "Duration model is allowing long stress regimes; inspect drawdown tolerance."
    return "Uses duration-aware regime transitions and covariate-matched historical blocks."


def _simulation_validation_row(
    *,
    label: str,
    strategy: str,
    result: BacktestResult,
    deterministic_wealth: float | None,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
) -> dict[str, object]:
    historical = _historical_result_metrics(result)
    bootstrap_median = _safe_float(bootstrap_summary.get("terminal_wealth_p50"))
    forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    forward_dd = _safe_float(forward_summary.get("max_drawdown_p50"))
    forward_ulcer = _safe_float(forward_summary.get("ulcer_index_p50"))
    deterministic_delta = _relative_delta(forward_median, deterministic_wealth)
    drawdown_delta = (
        forward_dd - historical["max_drawdown"]
        if forward_dd is not None and historical["max_drawdown"] is not None
        else None
    )
    ulcer_delta = (
        forward_ulcer - historical["ulcer_index"]
        if forward_ulcer is not None and historical["ulcer_index"] is not None
        else None
    )
    return {
        "portfolio": label,
        "strategy": strategy,
        "observed_years": historical["observed_years"],
        "historical_cagr": historical["cagr"],
        "historical_max_drawdown": historical["max_drawdown"],
        "historical_ulcer_index": historical["ulcer_index"],
        "deterministic_15y": deterministic_wealth,
        "bootstrap_median_15y": bootstrap_median,
        "forward_median_15y": forward_median,
        "forward_vs_deterministic": deterministic_delta,
        "forward_median_drawdown": forward_dd,
        "forward_minus_historical_dd": drawdown_delta,
        "forward_median_ulcer": forward_ulcer,
        "forward_minus_historical_ulcer": ulcer_delta,
        "calibration_read": _simulation_calibration_read(
            deterministic_delta=deterministic_delta,
            drawdown_delta=drawdown_delta,
            ulcer_delta=ulcer_delta,
        ),
    }


def _render_simulation_insight_cards(cards: list[dict[str, str]]) -> None:
    st.markdown(
        '<div class="operating-grid">'
        + "".join(_simulation_insight_card_html(card) for card in cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _simulation_insight_card_html(card: dict[str, str]) -> str:
    tone = card.get("tone", "")
    tone_class = f" operating-card-{html.escape(tone)}" if tone else ""
    return (
        f'<div class="operating-card{tone_class}">'
        f'<p class="operating-label">{html.escape(card["label"])}</p>'
        f'<p class="operating-answer">{html.escape(card["answer"])}</p>'
        f'<p class="operating-detail">{html.escape(card["detail"])}</p>'
        "</div>"
    )


def _simulation_interpretability_cards(
    *,
    strategy: str,
    validation_row: dict[str, object],
    drawdown_row: dict[str, object],
    forward_summary: dict[str, object],
    reference_edges: list[dict[str, object]],
    probabilities: pd.DataFrame,
) -> list[dict[str, str]]:
    calibration_read = str(validation_row.get("calibration_read", "insufficient_history"))
    reference_edge = _best_reference_edge(reference_edges)
    p10_wealth = _safe_float(forward_summary.get("terminal_wealth_p10"))
    p50_wealth = _safe_float(forward_summary.get("terminal_wealth_p50"))
    p90_wealth = _safe_float(forward_summary.get("terminal_wealth_p90"))
    median_dd = _safe_float(drawdown_row.get("max_drawdown_p50"))
    severe_prob = _safe_float(forward_summary.get("severe_drawdown_probability"))
    hard_breach = _safe_float(drawdown_row.get("breach_hard_band"))
    risk_off = _probability_sum(probabilities, ("risk_off",))
    transition = _probability_sum(probabilities, ("transition",))
    fragile = _probability_sum(probabilities, ("risk_on_fragile",))
    top_regime, top_probability = _top_probability_label(probabilities)
    return [
        {
            "tone": _calibration_tone(calibration_read),
            "label": "Past Resemblance",
            "answer": _calibration_answer(calibration_read),
            "detail": (
                f"Forward median is {_format_percent(validation_row.get('forward_vs_deterministic'))} "
                "versus deterministic CAGR math; median drawdown delta is "
                f"{_format_percent(validation_row.get('forward_minus_historical_dd'))}; "
                f"ulcer delta is {_format_percent(validation_row.get('forward_minus_historical_ulcer'))}."
            ),
        },
        {
            "tone": _reference_edge_tone(reference_edge),
            "label": "Reference Edge",
            "answer": _reference_edge_answer(reference_edge),
            "detail": (
                f"{strategy} forward range is {_format_currency(p10_wealth)} to "
                f"{_format_currency(p90_wealth)}, with median {_format_currency(p50_wealth)}. "
                f"{_reference_edge_detail(reference_edge)}"
            ),
        },
        {
            "tone": _drawdown_tone(median_dd, severe_prob, hard_breach),
            "label": "Drawdown Pain",
            "answer": f"Median {_format_percent(median_dd)}; hard breach {_format_percent(hard_breach)}",
            "detail": (
                f"The simulated median max drawdown is {_format_percent(median_dd)}. "
                f"Severe drawdown probability is {_format_percent(severe_prob)}; "
                f"hard-band breach share is {_format_percent(hard_breach)}."
            ),
        },
        {
            "tone": "warning" if risk_off + transition >= 0.55 else "success",
            "label": "Scenario Tilt",
            "answer": f"{top_regime} leads at {_format_percent(top_probability)}",
            "detail": (
                f"Current scenario bridge gives risk-off {_format_percent(risk_off)}, "
                f"transition {_format_percent(transition)}, fragile risk-on {_format_percent(fragile)}. "
                "The forward engine uses this to bias starting and transition regimes."
            ),
        },
    ]


def _simulation_verdict_rows(
    *,
    selected_strategy: str,
    validation_row: dict[str, object],
    drawdown_row: dict[str, object],
    forward_summary: dict[str, object],
    reference_edges: list[dict[str, object]],
    probabilities: pd.DataFrame,
) -> list[dict[str, object]]:
    top_regime, top_probability = _top_probability_label(probabilities)
    best_edge = _best_reference_edge(reference_edges)
    return [
        {
            "question": "Do simulated futures resemble the tested past?",
            "read": _calibration_answer(
                str(validation_row.get("calibration_read", "insufficient_history"))
            ),
            "evidence": (
                "Forward vs deterministic "
                f"{_format_percent(validation_row.get('forward_vs_deterministic'))}; "
                "forward DD minus historical DD "
                f"{_format_percent(validation_row.get('forward_minus_historical_dd'))}; "
                "forward ulcer minus historical ulcer "
                f"{_format_percent(validation_row.get('forward_minus_historical_ulcer'))}."
            ),
            "implication": "Use this to judge whether the simulation is roughly past-like or stress-shifted.",
        },
        {
            "question": "Does the strategy earn its complexity versus references?",
            "read": _reference_edge_answer(best_edge),
            "evidence": _reference_edge_detail(best_edge),
            "implication": (
                f"{selected_strategy} should be compared against simple references before it is "
                "treated as an operational improvement."
            ),
        },
        {
            "question": "Where is the drawdown pain?",
            "read": (
                f"Median max drawdown {_format_percent(drawdown_row.get('max_drawdown_p50'))}; "
                f"p10 max drawdown {_format_percent(drawdown_row.get('max_drawdown_p10'))}."
            ),
            "evidence": (
                f"Severe drawdown probability {_format_percent(forward_summary.get('severe_drawdown_probability'))}; "
                f"hard-band breach share {_format_percent(drawdown_row.get('breach_hard_band'))}; "
                f"median ulcer {_format_percent(drawdown_row.get('ulcer_index_p50'))}."
            ),
            "implication": "This is the behavioral tolerance check, not just a return forecast.",
        },
        {
            "question": "What current-state assumption is shaping the paths?",
            "read": f"{top_regime} has the largest scenario share at {_format_percent(top_probability)}.",
            "evidence": (
                f"Risk-off {_format_percent(_probability_sum(probabilities, ('risk_off',)))}; "
                f"transition {_format_percent(_probability_sum(probabilities, ('transition',)))}; "
                f"risk-on {_format_percent(_probability_sum(probabilities, ('risk_on',)))}; "
                f"fragile risk-on {_format_percent(_probability_sum(probabilities, ('risk_on_fragile',)))}."
            ),
            "implication": "If the scenario map is wrong, the regime-conditioned simulation will be wrong in that direction.",
        },
        {
            "question": "What should not be over-read?",
            "read": "This is empirical path planning, not an oracle.",
            "evidence": "The model samples historical regime-return blocks and cannot invent unseen AI, credit, policy, or liquidity events.",
            "implication": "Use the output as a distribution and stress lens alongside paper monitoring.",
        },
    ]


def _regime_resemblance_frame(
    *,
    probabilities: pd.DataFrame,
    library: pd.DataFrame,
    forward_paths: pd.DataFrame,
) -> pd.DataFrame:
    historical = (
        library["regime"].astype(str).value_counts(normalize=True)
        if not library.empty and "regime" in library
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for bucket in REGIME_BUCKETS:
        historical_share = _safe_float(historical.get(bucket, 0.0)) or 0.0
        scenario_share = _probability_sum(probabilities, (bucket,))
        simulated_share = (
            _series_mean(_numeric_path_column(forward_paths, f"share_{bucket}")) or 0.0
        )
        scenario_delta = scenario_share - historical_share
        simulation_delta = simulated_share - historical_share
        rows.append(
            {
                "regime": bucket,
                "historical_return_library_share": historical_share,
                "current_scenario_probability": scenario_share,
                "mean_simulated_path_share": simulated_share,
                "scenario_minus_history": scenario_delta,
                "simulation_minus_history": simulation_delta,
                "read": _regime_resemblance_read(simulation_delta),
            }
        )
    return pd.DataFrame(rows)


def _regime_resemblance_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    bars = [
        ("historical_return_library_share", "Historical library", "#64748b"),
        ("current_scenario_probability", "Current scenario", "#f59e0b"),
        ("mean_simulated_path_share", "Simulated paths", "#0f766e"),
    ]
    for column, label, color in bars:
        fig.add_trace(
            go.Bar(
                x=frame["regime"],
                y=frame[column],
                name=label,
                marker={"color": color},
                hovertemplate=f"{label}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=360,
        title="Regime resemblance: past vs current scenario vs simulated paths",
        yaxis={"title": "Share", "tickformat": ".0%"},
        xaxis={"title": "Regime bucket"},
        margin={"l": 20, "r": 20, "t": 48, "b": 84},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _simulation_missing_rows(
    validation_row: dict[str, object],
    probabilities: pd.DataFrame,
    library: pd.DataFrame,
) -> list[dict[str, object]]:
    regime_counts = (
        library["regime"].astype(str).value_counts().reindex(REGIME_BUCKETS).fillna(0).astype(int)
        if not library.empty and "regime" in library
        else pd.Series(0, index=REGIME_BUCKETS, dtype=int)
    )
    sparse_regimes = [regime for regime, count in regime_counts.items() if int(count) < 50]
    return [
        {
            "gap": "Novel regime risk",
            "current_read": (
                f"Risk-off plus transition probability is "
                f"{_format_percent(_probability_sum(probabilities, ('risk_off', 'transition')))}."
            ),
            "why_it_matters": "The simulator can sample only historical analogues, so a new AI/credit/policy break may not be fully represented.",
            "mitigation": "Compare simulation output with Daily Brief drivers, paper monitoring, and explicit stress tests.",
        },
        {
            "gap": "Sparse regime evidence",
            "current_read": (
                f"Sparse buckets: {', '.join(sparse_regimes) if sparse_regimes else 'none'}; "
                f"smallest bucket count {int(regime_counts.min()) if not regime_counts.empty else 0}."
            ),
            "why_it_matters": "A regime with little history produces less reliable forward-path behavior.",
            "mitigation": "Treat affected regimes as lower-confidence and inspect reference overlays.",
        },
        {
            "gap": "Execution and taxes",
            "current_read": "Path simulations use strategy returns before actual fills, missed trades, and taxable-account frictions.",
            "why_it_matters": "A strategy can look good in ideal paths and still disappoint when execution timing or taxes drag returns.",
            "mitigation": "Use Monitoring, Forward Test, implementation shortfall, and taxable-impact views before live use.",
        },
        {
            "gap": "Point-estimate confidence",
            "current_read": str(validation_row.get("calibration_read", "insufficient_history")),
            "why_it_matters": "Large gaps between deterministic, bootstrap, and regime-conditioned output mean the CAGR point estimate is not enough.",
            "mitigation": "Prioritize candidates that retain an edge across the distribution, not only in deterministic CAGR math.",
        },
    ]


def _simulation_method_rows() -> list[dict[str, str]]:
    return [
        {
            "layer": "Deterministic CAGR",
            "what_it_uses": "One historical CAGR point estimate plus configured starting account and monthly contributions.",
            "best_for": "Fast frontier ranking and simple benchmark comparison.",
            "main_failure_mode": "Ignores sequencing, volatility, and drawdown path pain.",
        },
        {
            "layer": "Historical block bootstrap",
            "what_it_uses": "Resampled blocks of the selected strategy's realized daily returns.",
            "best_for": "Sequence risk, drawdown persistence, and terminal-wealth range.",
            "main_failure_mode": "Assumes the future is drawn from the same realized return mix.",
        },
        {
            "layer": "Regime-conditioned forward paths",
            "what_it_uses": "Current scenario probabilities plus historical regime-labeled return blocks.",
            "best_for": "Current-state-aware planning ranges and downside/tail inspection.",
            "main_failure_mode": "Regime labels are coarse and scenario probabilities can be wrong.",
        },
    ]


def _reference_edge_rows(
    forward_summary: dict[str, object],
    reference_simulations: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    rows: list[dict[str, object]] = []
    for reference in reference_simulations:
        summary = reference.get("forward_summary")
        if not isinstance(summary, dict):
            continue
        reference_median = _safe_float(summary.get("terminal_wealth_p50"))
        edge = (
            selected_median - reference_median
            if selected_median is not None and reference_median is not None
            else None
        )
        rows.append(
            {
                "reference": reference.get("label"),
                "reference_strategy": reference.get("strategy"),
                "selected_forward_median": selected_median,
                "reference_forward_median": reference_median,
                "selected_minus_reference": edge,
                "selected_edge_pct": _relative_delta(selected_median, reference_median),
                "reference_median_forward_drawdown": summary.get("max_drawdown_p50"),
                "reference_severe_drawdown_probability": summary.get("severe_drawdown_probability"),
            }
        )
    return rows


def _simulation_comparison_row(
    *,
    label: str,
    strategy: str,
    deterministic_wealth: float | None,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
    selected_forward_median: float | None,
) -> dict[str, object]:
    forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    selected_delta = (
        selected_forward_median - forward_median
        if selected_forward_median is not None and forward_median is not None
        else None
    )
    return {
        "portfolio": label,
        "strategy": strategy,
        "deterministic_15y": deterministic_wealth,
        "bootstrap_p10": bootstrap_summary.get("terminal_wealth_p10"),
        "bootstrap_median": bootstrap_summary.get("terminal_wealth_p50"),
        "bootstrap_p90": bootstrap_summary.get("terminal_wealth_p90"),
        "forward_p10": forward_summary.get("terminal_wealth_p10"),
        "forward_median": forward_summary.get("terminal_wealth_p50"),
        "forward_p90": forward_summary.get("terminal_wealth_p90"),
        "selected_minus_row_forward_median": selected_delta,
        "median_forward_drawdown": forward_summary.get("max_drawdown_p50"),
        "severe_drawdown_probability": forward_summary.get("severe_drawdown_probability"),
    }


def _render_simulation_interpretability(
    *,
    selected_strategy: str | None,
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult | None,
    baseline_run: BaselineRun,
    scenario_source: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> None:
    st.markdown("**Simulation Interpretability**")
    st.caption(
        "This turns simulation paths into decision intelligence: whether the simulated future "
        "looks like the tested past, whether the selected strategy earns its complexity versus "
        "references, and what can still go wrong."
    )
    if selected_strategy is None or selected_result is None:
        st.info("Pick a strategy with reconstructable history to see simulation interpretation.")
        return

    daily_returns = _daily_returns(selected_result)
    if len(daily_returns) < 30:
        st.info("Selected strategy history is too short for a meaningful regime-return library.")
        return
    library = build_regime_return_library(
        daily_returns,
        config=ForwardSimulationConfig(),
    )
    if library.empty:
        st.info("The selected strategy did not produce a usable historical return library.")
        return

    deterministic_wealth = _deterministic_wealth(selected_scorecard, selected_result)
    return_values = _returns_tuple(selected_result)
    path_count = 150
    bootstrap_paths = _cached_bootstrap_paths(return_values, path_count)
    bootstrap_summary = summarize_bootstrap_outcomes(bootstrap_paths)
    forward_paths = _cached_regime_forward_paths(
        return_values,
        _scenario_records(scenario_source),
        path_count,
    )
    forward_summary = summarize_forward_simulation(
        forward_paths,
        config=ForwardSimulationConfig(paths=path_count),
    )
    reference_options = _reference_option_frame(baseline_run, selected_strategy)
    reference_simulations = _reference_simulations(
        baseline_run=baseline_run,
        reference_options=reference_options,
        selected_reference_labels=(
            reference_options["label"].head(2).tolist() if not reference_options.empty else []
        ),
        scenario_source=scenario_source,
        path_count=path_count,
    )
    validation_row = _simulation_validation_row(
        label="Selected strategy",
        strategy=selected_strategy,
        result=selected_result,
        deterministic_wealth=deterministic_wealth,
        bootstrap_summary=bootstrap_summary,
        forward_summary=forward_summary,
    )
    drawdown_row = _drawdown_distribution_row(
        label="Selected forward",
        strategy=selected_strategy,
        paths=forward_paths,
    )
    reference_edges = _reference_edge_rows(forward_summary, reference_simulations)

    _render_simulation_insight_cards(
        _simulation_interpretability_cards(
            strategy=selected_strategy,
            validation_row=validation_row,
            drawdown_row=drawdown_row,
            forward_summary=forward_summary,
            reference_edges=reference_edges,
            probabilities=probabilities,
        )
    )

    st.caption("Simulation verdict")
    _render_metric_dataframe(
        _display_metrics(
            pd.DataFrame(
                _simulation_verdict_rows(
                    selected_strategy=selected_strategy,
                    validation_row=validation_row,
                    drawdown_row=drawdown_row,
                    forward_summary=forward_summary,
                    reference_edges=reference_edges,
                    probabilities=probabilities,
                )
            )
        ),
        hide_index=True,
    )

    resemblance = _regime_resemblance_frame(
        probabilities=probabilities,
        library=library,
        forward_paths=forward_paths,
    )
    if not resemblance.empty:
        st.caption("How much do the simulated futures resemble the tested past?")
        st.plotly_chart(_regime_resemblance_figure(resemblance), width="stretch")
        _render_metric_dataframe(_display_metrics(resemblance), hide_index=True)

    st.caption("What the simulation may be missing")
    _render_metric_dataframe(
        pd.DataFrame(_simulation_missing_rows(validation_row, probabilities, library)),
        hide_index=True,
    )

    regime_summary = (
        library.groupby("regime")["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_daily_return", "std": "daily_volatility"})
    )
    regime_summary["annualized_mean_return"] = regime_summary["mean_daily_return"] * 252
    regime_summary["annualized_volatility"] = regime_summary["daily_volatility"] * (252**0.5)
    with st.expander(
        "Audit details: model inputs, scenario records, and sampled return libraries",
        expanded=False,
    ):
        _render_metric_dataframe(pd.DataFrame(_simulation_method_rows()), hide_index=True)
        cols = st.columns(2)
        with cols[0]:
            st.caption("Current scenario probability bridge")
            _render_metric_dataframe(_display_metrics(probabilities), hide_index=True)
        with cols[1]:
            st.caption("Scenario records feeding the bridge")
            available = [
                column
                for column in [
                    "horizon",
                    "scenario",
                    "probability",
                    "risk_bucket",
                    "expected_bot_posture",
                ]
                if column in scenario_source
            ]
            if available:
                _render_metric_dataframe(
                    _display_metrics(scenario_source[available].head(10)), hide_index=True
                )
            else:
                st.write("No detailed scenario records available.")

        st.caption(f"Historical return library for {selected_strategy}")
        _render_metric_dataframe(_display_metrics(regime_summary), hide_index=True)
        mix = regime_mix_frame(forward_paths)
        if not mix.empty:
            st.caption("Average simulated regime mix")
            _render_metric_dataframe(_display_metrics(mix), hide_index=True)


def _strategy_option_frame(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    frame = outcome_strategy_option_frame(
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
        include_defensive_judgement=False,
    )
    if not frame.empty:
        return frame
    catalog = build_approach_catalog(bot_config)
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    output = catalog.copy()
    output = output[
        output.get("source", pd.Series("", index=output.index)).astype(str).eq("baseline")
    ]
    if output.empty:
        output = catalog.head(25).copy()
    output["simulation_label"] = output.apply(
        lambda row: f"{row.get('display_name', row.get('strategy', 'Strategy'))} | configured",
        axis=1,
    )
    return output.reset_index(drop=True)


def _result_for_strategy(
    strategy_name: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
) -> BacktestResult | None:
    catalog = build_approach_catalog(bot_config)
    if catalog.empty or "strategy" not in catalog:
        return baseline_run.results.get(strategy_name)
    matches = catalog[catalog["strategy"].astype(str).eq(strategy_name)]
    if matches.empty:
        return baseline_run.results.get(strategy_name)
    row = matches.iloc[0]
    cache = st.session_state.setdefault("_simulation_lab_result_cache", {})
    cache_key = _simulation_result_cache_key(row, baseline_run)
    if cache_key in cache:
        return cache[cache_key]
    try:
        strategy = strategy_from_catalog_row(row)
        execution = execution_for_catalog_row(row, bot_config.execution)
        result, _ = build_approach_backtest_result(
            baseline_run.prices,
            strategy,
            execution,
            scenario_sizing=scenario_sizing_from_catalog_row(row),
            future_state_model=future_state_model_from_catalog_row(row),
            strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
            decision_sanity=decision_sanity_from_catalog_row(row),
            name=strategy_name,
        )
    except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return baseline_run.results.get(strategy_name)
    cache[cache_key] = result
    if len(cache) > 32:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)
    return result


def _simulation_result_cache_key(row: pd.Series, baseline_run: BaselineRun) -> str:
    prices = getattr(baseline_run, "prices", pd.DataFrame())
    price_marker: dict[str, object] | str
    if prices.empty:
        price_marker = "empty"
    else:
        price_marker = {
            "rows": int(len(prices)),
            "columns": list(map(str, prices.columns)),
            "start": str(prices.index.min()),
            "end": str(prices.index.max()),
        }
    payload = {
        "price_marker": price_marker,
        **{
            column: str(row.get(column, ""))
            for column in [
                "approach_id",
                "strategy",
                "strategy_json",
                "execution_json",
                "scenario_sizing_json",
                "future_state_model_json",
                "strategy_drawdown_model_json",
                "decision_sanity_json",
            ]
        },
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_source_frame(baseline_run: BaselineRun) -> pd.DataFrame:
    lattice = baseline_run.current_state.scenario_lattice.copy()
    if not lattice.empty and {"risk_bucket", "probability"}.issubset(lattice.columns):
        one_month = lattice[
            lattice.get("horizon", pd.Series("", index=lattice.index)).astype(str).eq("1m")
        ]
        return one_month.copy() if not one_month.empty else lattice
    outlook = baseline_run.current_state.scenario_outlook.copy()
    if not outlook.empty:
        return outlook
    return pd.DataFrame()


def _probability_for_bucket(probabilities: pd.DataFrame, bucket: str) -> float | None:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return None
    matches = probabilities[probabilities["regime"].astype(str).eq(bucket)]
    if matches.empty:
        return 0.0
    return _safe_float(matches.iloc[0]["probability"])


def _top_scenario_row(scenario_source: pd.DataFrame) -> pd.Series:
    if scenario_source.empty:
        return pd.Series(dtype=object)
    if "probability" in scenario_source:
        values = pd.to_numeric(scenario_source["probability"], errors="coerce").fillna(-1.0)
        return scenario_source.loc[values.idxmax()]
    return scenario_source.iloc[0]


def _scenario_probability_figure(probabilities: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=probabilities["regime"],
            y=probabilities["probability"],
            marker={"color": ["#b91c1c", "#f59e0b", "#2563eb", "#0f766e"][: len(probabilities)]},
            hovertemplate="%{x}<br>Probability %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        yaxis_tickformat=".0%",
        yaxis_title="Probability",
        xaxis_title="Simulation bucket",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


def _deterministic_wealth(
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult,
) -> float | None:
    cagr = _safe_float(selected_scorecard.get("cagr")) if selected_scorecard is not None else None
    if cagr is None:
        cagr = _cagr_from_result(selected_result)
    if cagr is None:
        return None
    return float(
        terminal_wealth_from_cagr(
            cagr,
            years=DEFAULT_OUTCOME_HORIZON_YEARS,
            starting_account_value=DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
            annual_contribution=DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
            contribution_timing=DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
        ).iloc[0]
    )


def _daily_returns(result: BacktestResult) -> pd.Series:
    return (
        result.equity.pct_change()
        .replace([float("inf"), float("-inf")], pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
    )


def _returns_tuple(result: BacktestResult) -> tuple[float, ...]:
    return tuple(float(value) for value in _daily_returns(result).to_numpy(dtype=float))


def _factor_simulation_inputs(
    result: BacktestResult,
    prices: pd.DataFrame,
) -> tuple[tuple[float, ...], tuple[str, ...], tuple[tuple[float, ...], ...]] | None:
    strategy_returns = _daily_returns(result)
    factor_returns = _factor_returns_from_prices(prices)
    if strategy_returns.empty or factor_returns.empty:
        return None
    aligned = pd.concat([strategy_returns.rename("strategy_return"), factor_returns], axis=1)
    aligned = aligned.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if aligned.shape[0] < max(60, aligned.shape[1] * 3):
        return None
    strategy_values = tuple(
        float(value) for value in aligned["strategy_return"].to_numpy(dtype=float)
    )
    factor_columns = tuple(str(column) for column in factor_returns.columns)
    factor_values = tuple(
        tuple(float(value) for value in row)
        for row in aligned.loc[:, factor_columns].to_numpy(dtype=float)
    )
    return strategy_values, factor_columns, factor_values


def _factor_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(index=prices.index)
    price_returns = (
        prices.sort_index().astype(float).pct_change().replace([float("inf"), float("-inf")], pd.NA)
    )
    for factor_name, proxy_ticker, _label, _description in DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS:
        if proxy_ticker in price_returns:
            frame[factor_name] = pd.to_numeric(price_returns[proxy_ticker], errors="coerce")
    return frame.dropna(how="all")


def _scenario_records(scenario_source: pd.DataFrame) -> tuple[tuple[str, float], ...]:
    probabilities = scenario_probability_frame(scenario_source)
    return tuple(
        (str(row["regime"]), float(row["probability"])) for _, row in probabilities.iterrows()
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_bootstrap_paths(return_values: tuple[float, ...], path_count: int) -> pd.DataFrame:
    return bootstrap_outcome_paths(
        pd.Series(return_values, dtype=float), config=OutcomeBootstrapConfig(paths=path_count)
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_regime_forward_paths(
    return_values: tuple[float, ...],
    scenario_records: tuple[tuple[str, float], ...],
    path_count: int,
) -> pd.DataFrame:
    scenario_frame = pd.DataFrame(
        [
            {"risk_bucket": regime, "probability": probability}
            for regime, probability in scenario_records
        ]
    )
    return simulate_regime_conditioned_paths(
        pd.Series(return_values, dtype=float),
        scenario_outlook=scenario_frame,
        config=ForwardSimulationConfig(paths=path_count),
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_factor_forward_paths(
    return_values: tuple[float, ...],
    factor_columns: tuple[str, ...],
    factor_rows: tuple[tuple[float, ...], ...],
    scenario_records: tuple[tuple[str, float], ...],
    path_count: int,
) -> pd.DataFrame:
    if not return_values or not factor_columns or not factor_rows:
        return pd.DataFrame()
    scenario_frame = pd.DataFrame(
        [
            {"risk_bucket": regime, "probability": probability}
            for regime, probability in scenario_records
        ]
    )
    factor_frame = pd.DataFrame(factor_rows, columns=list(factor_columns), dtype=float)
    return simulate_factor_conditioned_paths(
        pd.Series(return_values, dtype=float),
        factor_frame,
        scenario_outlook=scenario_frame,
        config=ForwardSimulationConfig(paths=path_count),
    )


def _simulation_plain_english_read(
    selected_strategy: str,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
    *,
    reference_simulations: list[dict[str, object]] | None = None,
) -> str:
    forward_p10 = _format_currency(forward_summary.get("terminal_wealth_p10"))
    forward_p50 = _format_currency(forward_summary.get("terminal_wealth_p50"))
    forward_p90 = _format_currency(forward_summary.get("terminal_wealth_p90"))
    severe = _format_percent(forward_summary.get("severe_drawdown_probability"))
    bootstrap_median = _format_currency(bootstrap_summary.get("terminal_wealth_p50"))
    reference_sentence = _reference_plain_english_read(forward_summary, reference_simulations or [])
    return (
        f"{selected_strategy}: the historical bootstrap median is {bootstrap_median}. "
        f"The scenario-conditioned central range is {forward_p10} to {forward_p90}, "
        f"with median {forward_p50} and severe-drawdown probability {severe}. "
        f"{reference_sentence}"
        "Read this as a planning distribution, not as a point forecast."
    )


def _reference_plain_english_read(
    selected_forward_summary: dict[str, object],
    reference_simulations: list[dict[str, object]],
) -> str:
    selected_median = _safe_float(selected_forward_summary.get("terminal_wealth_p50"))
    if selected_median is None or not reference_simulations:
        return ""
    parts: list[str] = []
    for reference in reference_simulations:
        summary = reference.get("forward_summary")
        if not isinstance(summary, dict):
            continue
        reference_median = _safe_float(summary.get("terminal_wealth_p50"))
        if reference_median is None:
            continue
        delta = selected_median - reference_median
        relation = "above" if delta >= 0 else "below"
        parts.append(
            f"{reference['label']} median {_format_currency(reference_median)} "
            f"({_format_currency(abs(delta))} {relation})"
        )
    if not parts:
        return ""
    return f"Against references: {'; '.join(parts)}. "


def _summary_row(
    label: str,
    deterministic_wealth: float | None,
    summary: dict[str, object] | None,
) -> dict[str, object]:
    if summary is None:
        return {
            "model": label,
            "terminal_wealth_p10": deterministic_wealth,
            "terminal_wealth_p50": deterministic_wealth,
            "terminal_wealth_p90": deterministic_wealth,
            "max_drawdown_p50": None,
            "ulcer_index_p50": None,
            "severe_drawdown_probability": None,
            "capital_impairment_probability": None,
        }
    return {
        "model": label,
        "terminal_wealth_p10": summary.get("terminal_wealth_p10"),
        "terminal_wealth_p50": summary.get("terminal_wealth_p50"),
        "terminal_wealth_p90": summary.get("terminal_wealth_p90"),
        "max_drawdown_p50": summary.get("max_drawdown_p50"),
        "ulcer_index_p50": summary.get("ulcer_index_p50"),
        "severe_drawdown_probability": summary.get("severe_drawdown_probability"),
        "capital_impairment_probability": summary.get("capital_impairment_probability"),
    }


def _simulation_histogram(
    paths: pd.DataFrame,
    *,
    column: str,
    title: str,
    xaxis_title: str,
    color: str,
    hovertemplate: str,
) -> go.Figure:
    fig = go.Figure()
    if column in paths and not paths.empty:
        fig.add_trace(
            go.Histogram(
                x=paths[column],
                nbinsx=40,
                marker={"color": color, "line": {"color": "#0f172a", "width": 0.5}},
                hovertemplate=hovertemplate,
            )
        )
    fig.update_layout(
        height=300,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Path count",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _simulation_overlay_histogram(
    items: list[dict[str, object]],
    *,
    column: str,
    title: str,
    xaxis_title: str,
    value_label: str = "Value",
    value_format: str = "decimal",
) -> go.Figure:
    fig = go.Figure()
    for item in items:
        paths = item.get("paths")
        if not isinstance(paths, pd.DataFrame) or column not in paths or paths.empty:
            continue
        label = str(item.get("label", "portfolio"))
        hovertemplate = _overlay_hover_template(label, value_label, value_format)
        fig.add_trace(
            go.Histogram(
                x=paths[column],
                nbinsx=40,
                name=label,
                opacity=0.55,
                marker={
                    "color": str(item.get("color", "#2563eb")),
                    "line": {"color": "#0f172a", "width": 0.5},
                },
                hovertemplate=hovertemplate,
            )
        )
    xaxis: dict[str, object] = {"title": xaxis_title}
    if value_format == "percent":
        xaxis["tickformat"] = ".0%"
    fig.update_layout(
        barmode="overlay",
        height=340,
        title=title,
        xaxis=xaxis,
        yaxis_title="Path count",
        margin={"l": 20, "r": 20, "t": 44, "b": 76},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _overlay_hover_template(label: str, value_label: str, value_format: str) -> str:
    if value_format == "currency":
        value = "%{x:$,.0f}"
    elif value_format == "percent":
        value = "%{x:.1%}"
    else:
        value = "%{x}"
    return f"{label}<br>{value_label} {value}<br>Paths %{{y}}<extra></extra>"


def _cagr_from_result(result: BacktestResult) -> float | None:
    if result.equity.empty or len(result.equity) < 2:
        return None
    start = _safe_float(result.equity.iloc[0])
    end = _safe_float(result.equity.iloc[-1])
    if start is None or end is None or start <= 0.0:
        return None
    years = max(len(result.equity) / 252.0, 1.0 / 252.0)
    return (end / start) ** (1.0 / years) - 1.0


def _historical_result_metrics(result: BacktestResult) -> dict[str, float | None]:
    equity = pd.to_numeric(result.equity, errors="coerce").dropna()
    if equity.empty:
        return {
            "observed_years": None,
            "cagr": None,
            "max_drawdown": None,
            "ulcer_index": None,
        }
    drawdown = equity / equity.cummax() - 1.0
    return {
        "observed_years": len(equity) / 252.0,
        "cagr": _cagr_from_result(result),
        "max_drawdown": _series_min(drawdown),
        "ulcer_index": _series_ulcer_index(drawdown),
    }


def _simulation_calibration_read(
    *,
    deterministic_delta: float | None,
    drawdown_delta: float | None,
    ulcer_delta: float | None,
) -> str:
    if deterministic_delta is None and drawdown_delta is None and ulcer_delta is None:
        return "insufficient_history"
    stressed = (
        (deterministic_delta is not None and deterministic_delta <= -0.15)
        or (drawdown_delta is not None and drawdown_delta <= -0.05)
        or (ulcer_delta is not None and ulcer_delta >= 0.03)
    )
    optimistic = (
        deterministic_delta is not None
        and deterministic_delta >= 0.15
        and (drawdown_delta is None or drawdown_delta >= 0.03)
    )
    if stressed:
        return "simulation_more_stressed_than_history"
    if optimistic:
        return "simulation_more_optimistic_than_history"
    return "broadly_past_like"


def _calibration_answer(calibration_read: str) -> str:
    labels = {
        "simulation_more_stressed_than_history": "Forward paths are more stressed than history",
        "simulation_more_optimistic_than_history": "Forward paths look easier than history",
        "broadly_past_like": "Forward paths broadly resemble the tested past",
        "insufficient_history": "Not enough history to judge resemblance",
    }
    return labels.get(calibration_read, calibration_read.replace("_", " ").title())


def _calibration_tone(calibration_read: str) -> str:
    if calibration_read == "simulation_more_stressed_than_history":
        return "warning"
    if calibration_read == "simulation_more_optimistic_than_history":
        return "critical"
    if calibration_read == "broadly_past_like":
        return "success"
    return "warning"


def _best_reference_edge(reference_edges: list[dict[str, object]]) -> dict[str, object] | None:
    scored = [
        row
        for row in reference_edges
        if _safe_float(row.get("selected_minus_reference")) is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda row: _safe_float(row.get("selected_minus_reference")) or 0.0)


def _reference_edge_answer(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "Reference comparison unavailable"
    delta = _safe_float(edge.get("selected_minus_reference"))
    label = str(edge.get("reference", "reference"))
    if delta is None:
        return f"Reference comparison to {label} unavailable"
    if delta >= 0:
        return f"Beats {label} by {_format_currency(delta)}"
    return f"Trails {label} by {_format_currency(abs(delta))}"


def _reference_edge_detail(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "Reference overlays were unavailable for this loaded run."
    return (
        f"Selected median {_format_currency(edge.get('selected_forward_median'))}; "
        f"{edge.get('reference')} median {_format_currency(edge.get('reference_forward_median'))}; "
        f"edge {_format_percent(edge.get('selected_edge_pct'))}; reference median drawdown "
        f"{_format_percent(edge.get('reference_median_forward_drawdown'))}."
    )


def _reference_edge_tone(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "warning"
    delta = _safe_float(edge.get("selected_minus_reference"))
    if delta is None:
        return "warning"
    return "success" if delta >= 0 else "critical"


def _drawdown_tone(
    median_drawdown: float | None,
    severe_probability: float | None,
    hard_breach: float | None,
) -> str:
    if (hard_breach is not None and hard_breach >= 0.10) or (
        severe_probability is not None and severe_probability >= 0.10
    ):
        return "critical"
    if median_drawdown is not None and median_drawdown <= DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT:
        return "warning"
    return "success"


def _probability_sum(probabilities: pd.DataFrame, buckets: tuple[str, ...]) -> float:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return 0.0
    frame = probabilities[probabilities["regime"].astype(str).isin(buckets)]
    if frame.empty:
        return 0.0
    return float(pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0).sum())


def _top_probability_label(probabilities: pd.DataFrame) -> tuple[str, float | None]:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return "n/a", None
    values = pd.to_numeric(probabilities["probability"], errors="coerce")
    if values.dropna().empty:
        return "n/a", None
    idx = values.idxmax()
    return str(probabilities.loc[idx, "regime"]), _safe_float(values.loc[idx])


def _regime_resemblance_read(simulation_delta: float) -> str:
    if simulation_delta >= 0.05:
        return "future_overweights_past"
    if simulation_delta <= -0.05:
        return "future_underweights_past"
    return "past_like"


def _numeric_path_column(paths: pd.DataFrame, column: str) -> pd.Series:
    if paths.empty or column not in paths:
        return pd.Series(dtype=float)
    return pd.to_numeric(paths[column], errors="coerce").dropna()


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(quantile))


def _series_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.mean())


def _series_min(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.min())


def _series_ulcer_index(drawdown: pd.Series) -> float | None:
    clean = pd.to_numeric(drawdown, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean.pow(2).mean()) ** 0.5)


def _relative_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or abs(reference) <= 1e-12:
        return None
    return value / reference - 1.0


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
