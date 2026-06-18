from __future__ import annotations

import html
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

from trade_bot.config import load_config
from trade_bot.dashboard.metric_explainers import (
    metric_categories,
    metric_detail,
    metric_guide_frame,
    metric_help,
)
from trade_bot.DEFAULT import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENT_CACHE_TTL_SECONDS,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_MACRO_PATH,
    DEFAULT_NEWS_PATH,
    DEFAULT_PERFORMANCE_WINDOW,
    DEFAULT_PERFORMANCE_WINDOWS,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
    DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS,
)
from trade_bot.reporting.report import (
    latest_positions_frame,
    make_equity_drawdown_figure,
    window_performance_frame,
)
from trade_bot.research.action_headline import ActionHeadline, build_action_headline
from trade_bot.research.approach_explorer import (
    approach_scorecard_row,
    build_approach_catalog,
    build_approach_mechanics,
    build_approach_risk_notes,
    build_approach_steps,
    build_latest_approach_weights,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun, run_configured_baselines
from trade_bot.research.experiment_monitor import (
    latest_experiment_iteration,
    load_experiment_candidates,
    load_experiment_regime_metrics,
    load_experiment_scorecards,
    load_experiment_walk_forward,
    summarize_experiment_families,
    summarize_experiment_history,
    summarize_experiment_operating_systems,
)
from trade_bot.storage.run_store import RunStore, SnapshotManifest, build_snapshot_fingerprint
from trade_bot.trading.journal import (
    DEFAULT_JOURNAL_PATH,
    TicketSizingConfig,
    TradeJournal,
    build_recommendation_tickets,
)


def _display_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    display = metrics.copy()
    percent_columns = [
        "cagr",
        "median_cagr",
        "worst_cagr",
        "best_cagr",
        "momentum_6m_skip_1w",
        "annualized_volatility",
        "realized_vol_3m",
        "max_drawdown",
        "worst_drawdown",
        "best_day",
        "worst_day",
        "return_1d",
        "return_1w",
        "return_1m",
        "return_3m",
        "drawdown",
        "total_return",
        "current_drawdown",
        "coverage",
        "average_turnover",
        "total_transaction_cost",
        "positive_window_rate",
        "return",
        "risk_asset_return",
        "defensive_return",
        "oil_complex_return",
        "credit_relative_return",
        "vixy_return",
        "spy_return",
        "qqq_return",
        "primary_strategy_return",
        "best_strategy_return",
        "probability",
        "percentile_5y",
        "pct_change_1w",
        "pct_change_2w",
        "pct_change_1m",
        "pct_change_3m",
        "pct_change_12m",
        "range_position_1y",
        "usable_share",
        "best_cagr",
        "best_max_drawdown",
        "active_day_rate",
        "base_cagr",
        "overlay_cagr",
        "delta_cagr",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "max_drawdown_improvement",
        "delta_worst_1y_cagr",
        "delta_worst_3y_cagr",
        "delta_worst_5y_cagr",
        "delta_positive_1y_window_rate",
        "delta_average_turnover",
        "current_weight",
        "scenario_adjusted_weight",
        "delta_weight",
        "target_weight",
        "weight",
        "one_month_risk_off_probability",
        "one_month_transition_probability",
        "one_month_fragile_upside_probability",
        "event_pressure",
        "macro_pressure",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "worst_regime_return",
        "worst_regime_cagr",
        "median_regime_return",
        "median_regime_cagr",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "transition_regime_return",
        "regime_positive_rate",
        "transition_regime_hit_rate",
        "walk_forward_median_cagr",
        "walk_forward_worst_cagr",
        "walk_forward_positive_rate",
        "walk_forward_worst_drawdown",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(_format_percent)
    for column in [
        "sharpe",
        "sortino",
        "calmar",
        "median_sharpe",
        "median_calmar",
        "years",
        "final_equity",
        "windows",
        "score",
        "z_score_5y",
        "change_1w",
        "change_2w",
        "change_1m",
        "change_3m",
        "change_12m",
        "short_move_z_1m",
        "change_acceleration_1m_vs_3m",
        "slope_1m",
        "slope_3m",
        "realized_vol_1m",
        "realized_vol_3m",
        "reversal_pressure",
        "risk_score",
        "mean_risk_score",
        "best_calmar",
        "iteration_rank",
        "promotion_score",
        "best_score",
        "median_score",
        "urgency_score",
        "confidence",
        "source_priority",
        "latest_pressure",
        "pressure_threshold",
        "risk_multiplier",
        "base_sharpe",
        "overlay_sharpe",
        "delta_sharpe",
        "base_calmar",
        "overlay_calmar",
        "delta_calmar",
        "usable_days",
        "active_days",
        "risk_budget_multiplier",
        "scenario_event_macro_multiplier",
        "portfolio_risk_multiplier",
        "robustness_score",
        "holdout_folds",
        "tested_regimes",
        "portfolio_equity_beta",
        "portfolio_ai_beta",
        "pre_equity_beta",
        "post_equity_beta",
        "max_equity_beta",
        "pre_ai_beta",
        "post_ai_beta",
        "max_ai_beta",
        "beta",
        "pre_beta",
        "post_beta",
        "beta_change",
        "correlation",
        "correlation_shift",
        "average_correlation_short",
        "average_correlation_long",
        "correlation_regime_shift",
        "scenario_risk_multiplier",
    ]:
        if column in display:
            display[column] = display[column].map(_format_decimal)
    for column in [
        "pre_risk_target_weight",
        "risk_adjusted_weight",
        "risk_engine_delta",
        "portfolio_expected_shortfall_95",
        "portfolio_max_stress_loss",
        "pre_expected_shortfall_95",
        "post_expected_shortfall_95",
        "max_expected_shortfall_95",
        "pre_max_stress_loss",
        "post_max_stress_loss",
        "max_stress_loss",
        "pre_scenario_weighted_stress_loss",
        "post_scenario_weighted_stress_loss",
        "max_scenario_weighted_stress_loss",
        "scenario_probability_weight",
        "pre_shock_return",
        "pre_loss",
        "post_shock_return",
        "post_loss",
        "risk_engine_delta_loss",
        "confidence_level",
        "value_at_risk",
        "expected_shortfall",
        "worst_day",
        "portfolio_annualized_volatility",
        "factor_annualized_volatility",
        "realized_volatility",
        "risk_contribution_pct",
        "annualized_vol_contribution",
        "post_absolute_beta_share",
        "risk_off_probability",
        "transition_probability",
        "fragile_upside_probability",
        "risk_on_probability",
        "ai_unwind_probability",
        "credit_stress_probability",
        "inflation_oil_probability",
        "max_single_asset_weight",
        "max_concentration_hhi",
        "max_expected_shortfall_95",
        "max_stress_loss",
        "max_scenario_weighted_stress_loss",
        "min_defensive_weight",
        "post_defensive_weight",
    ]:
        if column in display:
            display[column] = display[column].map(_format_percent)
    return display


def _display_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    currency_columns = [
        "reference_price",
        "limit_low",
        "limit_high",
        "target_notional",
        "min_notional",
        "max_notional",
        "price",
        "notional",
        "fees",
        "net_cash_deployed",
    ]
    share_columns = ["min_shares", "max_shares", "quantity", "net_quantity"]
    percent_columns = ["current_weight", "target_weight", "delta_weight"]
    for column in currency_columns:
        if column in display:
            display[column] = display[column].map(_format_currency)
    for column in share_columns:
        if column in display:
            display[column] = display[column].map(_format_shares)
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(_format_percent)
    return display


def _format_percent(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.2%}"


def _format_decimal(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:,.2f}"


def _format_currency(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"${numeric:,.2f}"


def _format_shares(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:,.4f}"


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _safe_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("America/Denver")


def _result_date_bounds(results: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for result in results.values():
        equity = result.equity.dropna()
        if equity.empty:
            continue
        starts.append(pd.Timestamp(equity.index.min()))
        ends.append(pd.Timestamp(equity.index.max()))
    if not starts or not ends:
        today = pd.Timestamp(date.today())
        return today, today
    return min(starts), max(ends)


def _default_strategy_selection(strategy_names: list[str]) -> list[str]:
    preferred = [
        "drawdown_managed_dual_momentum",
        "vol_target_dual_momentum",
        "dual_momentum_core",
        "buy_hold_spy",
        "buy_hold_qqq",
    ]
    selected = [name for name in preferred if name in strategy_names]
    return selected or strategy_names[: min(4, len(strategy_names))]


def _window_start_from_preset(
    preset: str,
    *,
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    custom_start: date | None = None,
) -> pd.Timestamp:
    if preset == "30 days":
        start = latest - pd.DateOffset(days=30)
    elif preset == "90 days":
        start = latest - pd.DateOffset(days=90)
    elif preset == "6 months":
        start = latest - pd.DateOffset(months=6)
    elif preset == "1 year":
        start = latest - pd.DateOffset(years=1)
    elif preset == "3 years":
        start = latest - pd.DateOffset(years=3)
    elif preset == "5 years":
        start = latest - pd.DateOffset(years=5)
    elif preset == "YTD":
        start = pd.Timestamp(year=latest.year, month=1, day=1)
    elif preset == "Custom" and custom_start is not None:
        start = pd.Timestamp(custom_start)
    else:
        start = earliest
    return max(earliest, min(start, latest))


@st.cache_data(show_spinner=False, ttl=DEFAULT_EXPERIMENT_CACHE_TTL_SECONDS)
def load_experiment_dashboard_frames(
    root: str | Path = DEFAULT_EXPERIMENTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_experiment_scorecards(root),
        load_experiment_regime_metrics(root),
        load_experiment_walk_forward(root),
        load_experiment_candidates(root),
    )


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_snapshot_dashboard_run(
    config_path_string: str,
    events_path_string: str,
    macro_path_string: str,
    news_path_string: str,
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
) -> tuple[BaselineRun, SnapshotManifest] | None:
    run_store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    fingerprint = build_snapshot_fingerprint(
        config_path_string,
        events_path_string,
        macro_path_string,
        news_path_string,
    )
    return run_store.load_latest_snapshot(
        fingerprint=fingerprint,
        require_matching_config=True,
    )


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_snapshot_jobs_frame(
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
) -> pd.DataFrame:
    run_store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    return run_store.list_jobs(limit=8)


def _install_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        h1 {
            font-size: 2rem;
            letter-spacing: 0;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        :root {
            --tb-card-bg: var(--secondary-background-color, #ffffff);
            --tb-card-border: rgba(125, 139, 155, 0.35);
            --tb-card-text: var(--text-color, #111827);
            --tb-card-muted: #4b5563;
            --tb-card-muted: color-mix(in srgb, var(--text-color, #111827) 68%, transparent);
            --tb-critical-bg: #fff5f5;
            --tb-critical-bg: color-mix(in srgb, #ef4444 12%, var(--background-color, #ffffff));
            --tb-critical-border: #c53030;
            --tb-warning-bg: #fffaf0;
            --tb-warning-bg: color-mix(in srgb, #f59e0b 12%, var(--background-color, #ffffff));
            --tb-warning-border: #b7791f;
            --tb-success-bg: #f7fbf8;
            --tb-success-bg: color-mix(in srgb, #22c55e 10%, var(--background-color, #ffffff));
            --tb-success-border: #2f855a;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --tb-card-bg: #171b22;
                --tb-card-border: #2d3440;
                --tb-card-text: #f8fafc;
                --tb-card-muted: #cbd5e1;
                --tb-critical-bg: #2a1518;
                --tb-critical-border: #ef4444;
                --tb-warning-bg: #251d10;
                --tb-warning-border: #f59e0b;
                --tb-success-bg: #102018;
                --tb-success-border: #22c55e;
            }
        }
        .action-banner {
            border: 1px solid var(--tb-card-border);
            border-left-width: 8px;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 18px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
        }
        .action-do_nothing {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .action-small_actions {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .action-critical_actions {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .headline-label {
            margin: 0;
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--tb-card-muted);
            font-weight: 700;
        }
        .headline-title {
            margin: 4px 0 6px;
            font-size: 1.45rem;
            line-height: 1.25;
            font-weight: 750;
            color: var(--tb-card-text);
        }
        .headline-copy {
            margin: 0 0 10px;
            color: var(--tb-card-text);
            line-height: 1.45;
        }
        .headline-next {
            margin: 0;
            color: var(--tb-card-text);
            font-weight: 650;
        }
        div[data-testid="stMetric"] {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--tb-card-text);
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            color: var(--tb-card-text) !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {
            color: var(--tb-card-muted) !important;
        }
        .brief-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 16px;
        }
        .brief-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 155px;
        }
        .brief-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .brief-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .brief-card-success {
            border-left-color: var(--tb-success-border);
        }
        .brief-label {
            margin: 0 0 6px;
            color: var(--tb-card-muted);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .brief-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text);
            font-size: 1.05rem;
            line-height: 1.3;
            font-weight: 750;
        }
        .brief-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.42;
            font-size: 0.92rem;
        }
        .operating-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 16px;
        }
        .operating-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 15px 17px;
            min-height: 145px;
        }
        .operating-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .operating-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .operating-card-success {
            border-left-color: var(--tb-success-border);
        }
        .operating-label {
            margin: 0 0 6px;
            color: var(--tb-card-muted);
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .operating-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text);
            font-size: 1.08rem;
            line-height: 1.28;
            font-weight: 760;
        }
        .operating-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.45;
            font-size: 0.93rem;
        }
        @media (max-width: 1100px) {
            .brief-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .operating-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 700px) {
            .brief-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _helped_metric(
    container: Any,
    label: str,
    value: object,
    *,
    key: str | None = None,
) -> None:
    container.metric(label, value, help=metric_help(key or label))


def _metric_column_config(frame: pd.DataFrame) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        help_text = metric_help(column)
        if help_text:
            config[column] = st.column_config.Column(help=help_text)
    return config


def _render_metric_dataframe(
    frame: pd.DataFrame,
    *,
    use_container_width: bool = True,
    hide_index: bool | None = None,
) -> None:
    kwargs: dict[str, Any] = {"use_container_width": use_container_width}
    column_config = _metric_column_config(frame)
    if column_config:
        kwargs["column_config"] = column_config
    if hide_index is not None:
        kwargs["hide_index"] = hide_index
    st.dataframe(frame, **kwargs)


def _render_metric_guide() -> None:
    with st.expander("Metric Guide", expanded=False):
        st.caption(
            "Quick reference for the scorecard, risk, scenario, and performance terms used across the dashboard."
        )
        filter_cols = st.columns([1, 2])
        category_options = ["all", *metric_categories()]
        selected_category = filter_cols[0].selectbox(
            "Metric family",
            category_options,
            key="metric_guide_category",
        )
        search_text = filter_cols[1].text_input(
            "Search",
            "",
            key="metric_guide_search",
        )
        guide = metric_guide_frame(
            category=None if selected_category == "all" else selected_category,
            search=search_text,
        )
        if guide.empty:
            st.write("No matching metric explainers.")
            return

        selected_metric = st.selectbox(
            "Detailed metric",
            list(guide["metric"]),
            key="metric_guide_metric",
        )
        detail = metric_detail(str(selected_metric))
        if detail is not None:
            st.markdown(f"**{detail.metric}**")
            st.write(detail.plain_english)
            st.markdown(f"**Calculation:** {detail.calculation}")
            st.markdown(f"**How to read:** {detail.how_to_read}")
            st.markdown(f"**Watch out:** {detail.caution}")

        guide_columns = ["metric", "category", "plain_english", "how_to_read", "caution"]
        st.dataframe(guide[guide_columns], use_container_width=True, hide_index=True)


def _render_action_headline(headline: ActionHeadline) -> None:
    st.markdown(
        f"""
        <div class="action-banner action-{html.escape(headline.level)}">
            <p class="headline-label">Action Headline</p>
            <div class="headline-title">{html.escape(headline.headline)}</div>
            <p class="headline-copy">{html.escape(headline.explanation)}</p>
            <p class="headline-next">Next: {html.escape(headline.next_action)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row = headline.metrics.iloc[0]
    cols = st.columns(6)
    _helped_metric(cols[0], "Severity", str(headline.severity))
    _helped_metric(cols[1], "Risk", str(metric_row["risk_status"]), key="risk_status")
    _helped_metric(cols[2], "Risk Score", f"{float(metric_row['risk_score']):.2f}")
    _helped_metric(cols[3], "Max Change", f"{float(metric_row['max_position_change']):.1%}")
    _helped_metric(cols[4], "Active News", f"{int(metric_row['active_news_items'])}")
    _helped_metric(cols[5], "Open Tickets", f"{int(metric_row['open_ticket_count'])}")
    if not headline.drivers.empty:
        st.dataframe(headline.drivers, use_container_width=True)


def _render_operating_brief(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
) -> None:
    st.subheader("Operating Brief")
    st.caption(
        "Instruction sheet for the current run: conclusion, action, sizing, evidence, and how scenarios changed the recommendation."
    )
    cards = _operating_brief_cards(baseline_run=baseline_run, headline=headline)
    st.markdown(
        '<div class="operating-grid">'
        + "".join(_operating_card_html(card) for card in cards)
        + "</div>",
        unsafe_allow_html=True,
    )

    sizing_steps = _recommended_sizing_steps(baseline_run.trade_decision.position_plan)
    scenario_bridge = _scenario_bridge_table(baseline_run)
    evidence = _operating_evidence_table(baseline_run, headline)

    detail_tab, sizing_tab, scenario_tab, evidence_tab = st.tabs(
        ["Read This First", "Sizing Steps", "Scenario Bridge", "Why"]
    )
    with detail_tab:
        _render_metric_dataframe(_operating_instruction_table(baseline_run, headline), hide_index=True)
    with sizing_tab:
        st.caption(
            "Sizing is shown in portfolio-weight percentage points. Use the Forward Test section to convert these into ticket dollar/share ranges."
        )
        _render_metric_dataframe(_display_metrics(sizing_steps), hide_index=True)
    with scenario_tab:
        st.caption(
            "Future scenarios do not directly predict one exact future. They adjust the allowed risk budget, defensive minimums, beta limits, and stress-loss guardrails."
        )
        _render_metric_dataframe(_display_metrics(scenario_bridge), hide_index=True)
    with evidence_tab:
        _render_metric_dataframe(evidence, hide_index=True)


def _operating_brief_cards(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
) -> list[dict[str, str]]:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    action = str(trade_summary.get("recommended_action", headline.label))
    target_position = str(
        trade_summary.get("scenario_adjusted_position")
        or trade_summary.get("base_position")
        or "No target position available."
    )
    risk_budget = _format_decimal(trade_summary.get("risk_budget_multiplier", "n/a"))
    scenario_effect = _scenario_effect_sentence(trade_summary, risk_summary)
    primary_sizing = _primary_sizing_sentence(baseline_run.trade_decision.position_plan)
    return [
        {
            "tone": _brief_tone(headline.level),
            "label": "Conclusion",
            "answer": headline.label,
            "detail": headline.explanation,
        },
        {
            "tone": _brief_tone(headline.level),
            "label": "Recommended Action",
            "answer": action.replace("_", " ").title(),
            "detail": (
                f"Move only after review toward target posture: {target_position}. "
                f"Current combined risk budget is {risk_budget}."
            ),
        },
        {
            "tone": "warning" if "REDUCE" in action else "success",
            "label": "Sizing Translation",
            "answer": primary_sizing["answer"],
            "detail": primary_sizing["detail"],
        },
        {
            "tone": "warning",
            "label": "Scenario Incorporation",
            "answer": scenario_effect["answer"],
            "detail": scenario_effect["detail"],
        },
    ]


def _operating_card_html(card: dict[str, str]) -> str:
    return f"""
    <div class="operating-card operating-card-{html.escape(card['tone'])}">
        <p class="operating-label">{html.escape(card['label'])}</p>
        <p class="operating-answer">{html.escape(card['answer'])}</p>
        <p class="operating-detail">{html.escape(card['detail'])}</p>
    </div>
    """


def _operating_instruction_table(
    baseline_run: BaselineRun,
    headline: ActionHeadline,
) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    action = str(trade_summary.get("recommended_action", headline.label))
    target_position = str(
        trade_summary.get("scenario_adjusted_position")
        or trade_summary.get("base_position")
        or "No target position available."
    )
    risk_status = baseline_run.current_state.risk_status.upper()
    risk_budget = _format_decimal(trade_summary.get("risk_budget_multiplier", "n/a"))
    return pd.DataFrame(
        [
            {
                "question": "What is the conclusion?",
                "answer": headline.headline,
                "what_to_do": headline.next_action,
            },
            {
                "question": "What action is recommended?",
                "answer": action.replace("_", " "),
                "what_to_do": f"Review the sizing steps and move toward {target_position} only if the recommendation still fits your execution window.",
            },
            {
                "question": "How aggressive should sizing be?",
                "answer": f"Risk status is {risk_status}; combined risk budget is {risk_budget}.",
                "what_to_do": "Treat the target weights as capped by scenario/event/macro and portfolio-risk constraints, not as a full-risk forecast.",
            },
            {
                "question": "Where do I execute or paper-test it?",
                "answer": "Use Forward Test after reviewing the Command Center and Risk & Scenarios sections.",
                "what_to_do": "Lock the recommendation set, then log paper or live executions with exact time, price, quantity, and notes.",
            },
        ]
    )


def _recommended_sizing_steps(position_plan: pd.DataFrame) -> pd.DataFrame:
    if position_plan.empty:
        return pd.DataFrame(
            [
                {
                    "step": 1,
                    "ticker": "Portfolio",
                    "action": "NO_DATA",
                    "current_weight": 0.0,
                    "target_weight": 0.0,
                    "delta_weight": 0.0,
                    "instruction": "No position-plan rows are available.",
                }
            ]
        )

    target_column = _first_existing_column(
        position_plan,
        (
            "scenario_adjusted_weight",
            "risk_adjusted_weight",
            "target_weight",
            "pre_risk_target_weight",
        ),
    )
    current_column = _first_existing_column(
        position_plan,
        ("current_weight", "base_weight", "weight"),
    )
    if target_column is None or current_column is None:
        return pd.DataFrame(
            [
                {
                    "step": 1,
                    "ticker": "Portfolio",
                    "action": "REVIEW",
                    "current_weight": 0.0,
                    "target_weight": 0.0,
                    "delta_weight": 0.0,
                    "instruction": "Position-plan weights are incomplete; review the raw Command Center table.",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in position_plan.copy().iterrows():
        ticker = str(row.get("ticker", "Portfolio"))
        action = str(row.get("action", "HOLD"))
        current_weight = _as_float(row.get(current_column))
        target_weight = _as_float(row.get(target_column))
        delta_weight = _as_float(row.get("delta_weight", target_weight - current_weight))
        if abs(delta_weight) < 0.005 and action == "HOLD":
            continue
        rows.append(
            {
                "step": len(rows) + 1,
                "ticker": ticker,
                "action": action,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "instruction": _sizing_instruction(
                    ticker,
                    action,
                    current_weight,
                    target_weight,
                    delta_weight,
                ),
            }
        )

    if not rows:
        rows.append(
            {
                "step": 1,
                "ticker": "Portfolio",
                "action": "HOLD",
                "current_weight": 0.0,
                "target_weight": 0.0,
                "delta_weight": 0.0,
                "instruction": "No material sizing change is currently recommended.",
            }
        )
    return pd.DataFrame(rows)


def _scenario_bridge_table(baseline_run: BaselineRun) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    rows = [
        {
            "bridge_step": "Scenario probabilities",
            "current_read": (
                f"1M risk-off {_format_percent(trade_summary.get('one_month_risk_off_probability'))}; "
                f"transition {_format_percent(trade_summary.get('one_month_transition_probability'))}; "
                f"fragile upside {_format_percent(trade_summary.get('one_month_fragile_upside_probability'))}"
            ),
            "how_it_changes_action": "Higher risk-off, transition, or fragile-upside probabilities shrink the allowed risk budget before tickets are created.",
        },
        {
            "bridge_step": "Scenario/event/macro multiplier",
            "current_read": _format_decimal(trade_summary.get("scenario_event_macro_multiplier")),
            "how_it_changes_action": "This scales down the base strategy target when scenarios, current events, or tested macro pressure argue for caution.",
        },
        {
            "bridge_step": "Portfolio risk multiplier",
            "current_read": _format_decimal(trade_summary.get("portfolio_risk_multiplier")),
            "how_it_changes_action": "This applies factor, beta, expected-shortfall, stress-loss, and concentration limits after the scenario overlay.",
        },
        {
            "bridge_step": "Final risk budget",
            "current_read": _format_decimal(trade_summary.get("risk_budget_multiplier")),
            "how_it_changes_action": "This is the final exposure throttle used to move from the base position to the scenario-adjusted target.",
        },
    ]
    if risk_summary:
        rows.append(
            {
                "bridge_step": "Risk constraints",
                "current_read": str(risk_summary.get("applied_constraints", "none")),
                "how_it_changes_action": (
                    f"Post-risk ES95 is {_format_percent(risk_summary.get('post_expected_shortfall_95'))}; "
                    f"max stress loss is {_format_percent(risk_summary.get('post_max_stress_loss'))}; "
                    f"AI beta is {_format_decimal(risk_summary.get('post_ai_beta'))}."
                ),
            }
        )

    scenario_links = baseline_run.trade_decision.scenario_links
    if not scenario_links.empty:
        for _, row in scenario_links.head(3).iterrows():
            rows.append(
                {
                    "bridge_step": f"Top scenario: {row.get('scenario', 'scenario')}",
                    "current_read": (
                        f"{_format_percent(row.get('probability'))}; "
                        f"{row.get('risk_bucket', 'unknown bucket')}"
                    ),
                    "how_it_changes_action": str(
                        row.get("expected_bot_posture")
                        or row.get("off_ramp")
                        or row.get("confirmation")
                        or ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def _operating_evidence_table(
    baseline_run: BaselineRun,
    headline: ActionHeadline,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in headline.drivers.head(6).iterrows():
        rows.append(
            {
                "evidence_type": "headline_driver",
                "evidence": f"{row.get('driver')}: {row.get('signal')}",
                "interpretation": str(row.get("detail", "")),
            }
        )

    decision_evidence = baseline_run.trade_decision.evidence
    if not decision_evidence.empty:
        for _, row in decision_evidence.head(6).iterrows():
            signal = row.get("signal", row.get("evidence", "decision evidence"))
            impact = row.get("impact", row.get("interpretation", ""))
            rows.append(
                {
                    "evidence_type": str(row.get("evidence_type", "trade_decision")),
                    "evidence": str(signal),
                    "interpretation": str(impact),
                }
            )

    if not rows:
        rows.append(
            {
                "evidence_type": "none",
                "evidence": "No evidence rows available.",
                "interpretation": "Review raw diagnostics in the section tabs.",
            }
        )
    return pd.DataFrame(rows)


def _scenario_effect_sentence(
    trade_summary: dict[str, object],
    risk_summary: dict[str, object],
) -> dict[str, str]:
    risk_off = _format_percent(trade_summary.get("one_month_risk_off_probability"))
    transition = _format_percent(trade_summary.get("one_month_transition_probability"))
    final_budget = _format_decimal(trade_summary.get("risk_budget_multiplier"))
    portfolio_multiplier = _format_decimal(trade_summary.get("portfolio_risk_multiplier"))
    constraints = str(risk_summary.get("applied_constraints", "none")) if risk_summary else "none"
    return {
        "answer": f"1M risk-off {risk_off}, transition {transition}",
        "detail": (
            f"Scenarios feed the risk budget first, then the risk engine applies constraints. "
            f"Final risk budget is {final_budget}; portfolio-risk multiplier is {portfolio_multiplier}; "
            f"constraints: {constraints}."
        ),
    }


def _primary_sizing_sentence(position_plan: pd.DataFrame) -> dict[str, str]:
    sizing_steps = _recommended_sizing_steps(position_plan)
    material = sizing_steps[sizing_steps["ticker"] != "Portfolio"]
    if material.empty:
        return {
            "answer": "No Material Change",
            "detail": "The current plan does not require a meaningful target-weight move.",
        }
    largest = material.copy()
    largest["abs_delta"] = largest["delta_weight"].astype(float).abs()
    row = largest.sort_values("abs_delta", ascending=False).iloc[0]
    return {
        "answer": f"{str(row['action']).replace('_', ' ').title()} {row['ticker']}",
        "detail": (
            f"Largest proposed move is {_format_percent(row['delta_weight'])}: "
            f"{_format_percent(row['current_weight'])} to {_format_percent(row['target_weight'])}. "
            "Translate each percentage-point change into dollars using the account value in Forward Test."
        ),
    }


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _as_float(value: object, default: float = 0.0) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _sizing_instruction(
    ticker: str,
    action: str,
    current_weight: float,
    target_weight: float,
    delta_weight: float,
) -> str:
    if abs(delta_weight) < 0.005:
        return f"Hold {ticker} near {_format_percent(target_weight)}."
    verb = "Add to" if delta_weight > 0 else "Reduce"
    if action in {"SELL", "EXIT"}:
        verb = "Exit or reduce"
    return (
        f"{verb} {ticker} by {_format_percent(abs(delta_weight))} of the account, "
        f"moving from {_format_percent(current_weight)} to {_format_percent(target_weight)}."
    )


def _render_decision_brief(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    experiment_scorecards: pd.DataFrame,
) -> None:
    st.subheader("Decision Brief")
    cards = _decision_brief_cards(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=open_ticket_count,
        experiment_scorecards=experiment_scorecards,
    )
    st.markdown(
        '<div class="brief-grid">' + "".join(_brief_card_html(card) for card in cards) + "</div>",
        unsafe_allow_html=True,
    )

    conclusions = _decision_conclusions_table(
        baseline_run=baseline_run,
        headline=headline,
        experiment_scorecards=experiment_scorecards,
    )
    st.caption("Interpretation layer: read this before scanning detailed tables.")
    st.dataframe(conclusions, use_container_width=True, hide_index=True)

    watch_items = _decision_watch_items(baseline_run)
    if not watch_items.empty:
        st.caption("What would change the decision")
        st.dataframe(_display_metrics(watch_items), use_container_width=True, hide_index=True)


def _decision_brief_cards(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    experiment_scorecards: pd.DataFrame,
) -> list[dict[str, str]]:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    strongest_experiment = _strongest_experiment_summary(experiment_scorecards)
    recent_tension = _recent_performance_tension(baseline_run)
    target_position = str(
        trade_summary.get("scenario_adjusted_position")
        or trade_summary.get("base_position")
        or "No target position available."
    )
    risk_budget = _format_decimal(trade_summary.get("risk_budget_multiplier", "n/a"))
    action = str(trade_summary.get("recommended_action", headline.label))
    risk_level = str(risk_summary.get("portfolio_risk_level", "not available"))
    constraints = str(risk_summary.get("applied_constraints", "none"))
    return [
        {
            "tone": _brief_tone(headline.level),
            "label": "What to do now",
            "answer": action.replace("_", " ").title(),
            "detail": (
                f"Target posture is {target_position}. Risk budget is {risk_budget}. "
                f"Open recommendation tickets: {open_ticket_count}."
            ),
        },
        {
            "tone": "warning" if risk_level != "within_limits" else "success",
            "label": "Why",
            "answer": risk_level.replace("_", " ").title(),
            "detail": (
                f"Portfolio risk constraints: {constraints}. "
                f"Current market state is {baseline_run.current_state.risk_status.upper()} "
                f"with score {baseline_run.current_state.risk_score:.2f}."
            ),
        },
        {
            "tone": "warning",
            "label": "What could make this wrong",
            "answer": recent_tension["answer"],
            "detail": recent_tension["detail"],
        },
        {
            "tone": (
                "success" if strongest_experiment["decision"] == "promote_candidate" else "warning"
            ),
            "label": "Research takeaway",
            "answer": strongest_experiment["answer"],
            "detail": strongest_experiment["detail"],
        },
    ]


def _brief_card_html(card: dict[str, str]) -> str:
    return f"""
    <div class="brief-card brief-card-{html.escape(card['tone'])}">
        <p class="brief-label">{html.escape(card['label'])}</p>
        <p class="brief-answer">{html.escape(card['answer'])}</p>
        <p class="brief-detail">{html.escape(card['detail'])}</p>
    </div>
    """


def _decision_conclusions_table(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    strongest_experiment = _strongest_experiment_summary(experiment_scorecards)
    recent_tension = _recent_performance_tension(baseline_run)
    rows = [
        {
            "question": "What is the system asking me to do?",
            "conclusion": str(trade_summary.get("recommended_action", headline.label)).replace(
                "_", " "
            ),
            "evidence": str(trade_summary.get("human_explanation", headline.explanation)),
            "drill_down": "Trade Plan",
        },
        {
            "question": "Is this mostly a signal or a risk-control decision?",
            "conclusion": str(risk_summary.get("portfolio_risk_level", "not available")).replace(
                "_", " "
            ),
            "evidence": (
                f"Constraints: {risk_summary.get('applied_constraints', 'none')}; "
                f"post ES95 { _format_percent(risk_summary.get('post_expected_shortfall_95'))}; "
                f"max stress loss { _format_percent(risk_summary.get('post_max_stress_loss'))}."
            ),
            "drill_down": "Risk Engine",
        },
        {
            "question": "What is the strongest tested approach right now?",
            "conclusion": strongest_experiment["answer"],
            "evidence": strongest_experiment["detail"],
            "drill_down": "Research",
        },
        {
            "question": "What is the main tension?",
            "conclusion": recent_tension["answer"],
            "evidence": recent_tension["detail"],
            "drill_down": "Performance",
        },
    ]
    return pd.DataFrame(rows)


def _decision_watch_items(baseline_run: BaselineRun) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_links = baseline_run.trade_decision.scenario_links
    if not scenario_links.empty:
        for _, row in scenario_links.head(3).iterrows():
            rows.append(
                {
                    "watch_item": str(row.get("scenario", "scenario")),
                    "current_read": _format_percent(row.get("probability")),
                    "why_it_matters": str(row.get("expected_bot_posture", "")),
                    "off_ramp_or_confirmation": str(
                        row.get("off_ramp") or row.get("confirmation") or ""
                    ),
                }
            )
    if baseline_run.portfolio_risk is not None and not baseline_run.portfolio_risk.summary.empty:
        risk_row = baseline_run.portfolio_risk.summary.iloc[0]
        rows.append(
            {
                "watch_item": "Portfolio constraints",
                "current_read": str(risk_row.get("applied_constraints", "none")),
                "why_it_matters": "These constraints directly change target position sizing.",
                "off_ramp_or_confirmation": "Relax only if scenario and stress losses normalize.",
            }
        )
    if not baseline_run.news_monitor.triage.empty:
        active = baseline_run.news_monitor.triage[
            baseline_run.news_monitor.triage["activation_status"]
            .astype(str)
            .str.contains("event_risk", na=False)
        ]
        if not active.empty:
            rows.append(
                {
                    "watch_item": "Active news pressure",
                    "current_read": f"{len(active):,} active items",
                    "why_it_matters": "News has been converted into event-risk context.",
                    "off_ramp_or_confirmation": "Watch whether price, credit, or scenario drivers confirm it.",
                }
            )
    return pd.DataFrame(rows)


def _portfolio_risk_summary(baseline_run: BaselineRun) -> dict[str, object]:
    risk = baseline_run.portfolio_risk or baseline_run.trade_decision.portfolio_risk
    if risk is None or risk.summary.empty:
        return {}
    return risk.summary.iloc[0].to_dict()


def _strongest_experiment_summary(scorecards: pd.DataFrame) -> dict[str, str]:
    if scorecards.empty:
        return {
            "answer": "No experiment result loaded",
            "detail": "Run experiment iterations to populate the research monitor.",
            "decision": "",
        }
    frame = scorecards.copy()
    if "robustness_score" in frame:
        robust = frame[frame["robustness_score"].notna()]
        if not robust.empty:
            frame = robust
    for column in ["promotion_score", "robustness_score", "calmar"]:
        if column not in frame:
            frame[column] = 0.0
    top = frame.sort_values(
        ["promotion_score", "robustness_score", "calmar"],
        ascending=False,
    ).iloc[0]
    strategy = str(top.get("strategy", "unknown"))
    decision = str(top.get("promotion_decision", "unknown"))
    detail = (
        f"{decision.replace('_', ' ')}; CAGR {_format_percent(top.get('cagr'))}; "
        f"max drawdown {_format_percent(top.get('max_drawdown'))}; "
        f"walk-forward positive rate {_format_percent(top.get('walk_forward_positive_rate'))}."
    )
    return {"answer": strategy, "detail": detail, "decision": decision}


def _recent_performance_tension(baseline_run: BaselineRun) -> dict[str, str]:
    latest = baseline_run.prices.index.max()
    start = pd.Timestamp(latest) - pd.DateOffset(days=90)
    window = window_performance_frame(
        baseline_run.results,
        start=start,
        end=latest,
    )
    if window.empty:
        return {
            "answer": "No recent window available",
            "detail": "Recent performance diagnostics are unavailable.",
        }
    qqq = _window_return(window, "buy_hold_qqq")
    primary = _window_return(window, "drawdown_managed_dual_momentum")
    if qqq is not None and primary is not None and qqq > primary + 0.05:
        return {
            "answer": "Momentum is fighting risk control",
            "detail": (
                f"QQQ is up {_format_percent(qqq)} over ~90 days versus "
                f"{_format_percent(primary)} for the primary strategy, but the risk engine is still "
                "throttling exposure."
            ),
        }
    leader = window.sort_values("total_return", ascending=False).iloc[0]
    return {
        "answer": f"Recent leader: {leader['strategy']}",
        "detail": (
            f"Best ~90 day return is {_format_percent(leader['total_return'])}; "
            "compare this against risk status before adding exposure."
        ),
    }


def _window_return(window: pd.DataFrame, strategy: str) -> float | None:
    rows = window[window["strategy"] == strategy]
    if rows.empty:
        return None
    value = rows.iloc[0].get("total_return")
    numeric = _optional_float(value)
    return numeric


def _first_display_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _brief_tone(level: str) -> str:
    if level == "critical_actions":
        return "critical"
    if level == "small_actions":
        return "warning"
    return "success"


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
    _render_metric_dataframe(_display_metrics(scenario_view.sort_values("rank").head(scenario_limit)))

    st.subheader("Risk Confirmation Matrix")
    st.dataframe(current_state.confirmation_matrix, use_container_width=True)

    st.subheader("Market Health")
    _render_metric_dataframe(_display_metrics(current_state.market_health))

    st.subheader("VAMS-Style Signal Table")
    vams_filter = st.radio(
        "VAMS filter",
        ["all", "bullish", "neutral", "bearish"],
        horizontal=True,
    )
    vams_table = current_state.vams.copy()
    if vams_filter != "all":
        vams_table = vams_table[vams_table["vams_state"] == vams_filter]
    _render_metric_dataframe(_display_metrics(vams_table.head(75)))


def _render_approach_explorer(bot_config: Any, baseline_run: BaselineRun) -> None:
    st.subheader("Approach Explorer")
    st.caption("How each approach works and why it is or is not a candidate for operation.")
    approach_catalog = build_approach_catalog(bot_config)
    if approach_catalog.empty:
        st.write("No approaches available.")
        return

    approach_label = st.selectbox(
        "Approach",
        list(approach_catalog["label"]),
        index=0,
    )
    approach_row = approach_catalog[approach_catalog["label"] == approach_label].iloc[0]
    approach_strategy = strategy_from_catalog_row(approach_row)
    overview_cols = st.columns(5)
    _helped_metric(overview_cols[0], "Source", str(approach_row["source"]))
    _helped_metric(overview_cols[1], "Family", str(approach_row["family"]))
    _helped_metric(overview_cols[2], "Role", str(approach_row["role"]))
    _helped_metric(
        overview_cols[3],
        "Decision",
        str(approach_row["promotion_decision"]),
        key="promotion_decision",
    )
    _helped_metric(overview_cols[4], "Type", approach_strategy.type)
    if str(approach_row.get("hypothesis", "")):
        st.write(str(approach_row["hypothesis"]))

    mechanics_tab, steps_tab, position_tab, scorecard_tab, risk_tab = st.tabs(
        ["Mechanics", "Signal Steps", "Current Position", "Scorecard", "Risk Notes"]
    )
    with mechanics_tab:
        st.dataframe(
            build_approach_mechanics(approach_strategy, bot_config),
            use_container_width=True,
        )
    with steps_tab:
        st.dataframe(build_approach_steps(approach_strategy), use_container_width=True)
    with position_tab:
        _render_metric_dataframe(
            _display_metrics(build_latest_approach_weights(baseline_run.prices, approach_strategy))
        )
        if (
            approach_row["source"] == "baseline"
            and str(approach_row["strategy"]) in baseline_run.results
        ):
            baseline_position = latest_positions_frame(
                {str(approach_row["strategy"]): baseline_run.results[str(approach_row["strategy"])]}
            )
            st.dataframe(
                baseline_position.map(lambda value: f"{value:.2%}"),
                use_container_width=True,
            )
    with scorecard_tab:
        if (
            approach_row["source"] == "baseline"
            and str(approach_row["strategy"]) in baseline_run.metrics.index
        ):
            _render_metric_dataframe(
                _display_metrics(baseline_run.metrics.loc[[str(approach_row["strategy"])]])
            )
        else:
            scorecard_row = approach_scorecard_row(approach_row)
            if scorecard_row.empty:
                st.write("No scorecard metrics available for this approach.")
            else:
                _render_metric_dataframe(_display_metrics(scorecard_row))
    with risk_tab:
        st.dataframe(
            build_approach_risk_notes(approach_strategy, approach_row),
            use_container_width=True,
        )


def _render_experiment_monitor(
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    st.subheader("Experiment Monitor")
    if experiment_scorecards.empty:
        st.write("No experiment scorecards have been saved yet.")
        return

    latest_iteration = latest_experiment_iteration(experiment_scorecards)
    promoted_count = int((experiment_scorecards["promotion_decision"] == "promote_candidate").sum())
    rejected_tail_count = int(
        experiment_scorecards["promotion_decision"]
        .isin(["reject_left_tail", "reject_regime_fragility", "reject_walk_forward_fragility"])
        .sum()
    )
    latest_label = f"{latest_iteration:02d}" if latest_iteration is not None else "n/a"
    col_a, col_b, col_c, col_d = st.columns(4)
    _helped_metric(col_a, "Iterations", latest_label)
    _helped_metric(col_b, "Candidates", f"{len(experiment_scorecards):,}")
    _helped_metric(col_c, "Promoted", f"{promoted_count:,}", key="promotion_decision")
    _helped_metric(col_d, "Risk rejects", f"{rejected_tail_count:,}", key="promotion_decision")

    (
        experiment_overview_tab,
        experiment_leaderboard_tab,
        experiment_regime_tab,
        experiment_detail_tab,
        experiment_manifest_tab,
    ) = st.tabs(["Overview", "Leaderboard", "Regime Tests", "Candidate Detail", "Manifests"])

    with experiment_overview_tab:
        experiment_summary = summarize_experiment_history(experiment_scorecards)
        _render_metric_dataframe(_display_metrics(experiment_summary))

        family_summary = summarize_experiment_families(experiment_scorecards)
        if not family_summary.empty:
            st.caption("Research-family leaderboard")
            _render_metric_dataframe(_display_metrics(family_summary))

        operating_systems = summarize_experiment_operating_systems(experiment_scorecards)
        if not operating_systems.empty:
            st.caption("Best current operating-system candidates by family")
            _render_metric_dataframe(_display_metrics(operating_systems))

    with experiment_leaderboard_tab:
        experiment_iterations = sorted(experiment_scorecards["iteration"].unique())
        default_iterations = experiment_iterations[-10:]
        selected_iterations = st.multiselect(
            "Experiment iterations",
            experiment_iterations,
            default=default_iterations,
            key="experiment_leaderboard_iterations",
        )
        filter_col_a, filter_col_b, filter_col_c, filter_col_d = st.columns(4)
        decision_options = ["all", *sorted(experiment_scorecards["promotion_decision"].unique())]
        role_options = ["all", *sorted(experiment_scorecards["role"].unique())]
        phase_options = ["all", *sorted(experiment_scorecards["phase"].dropna().unique())]
        family_options = ["all", *sorted(experiment_scorecards["family"].dropna().unique())]
        decision_filter = filter_col_a.selectbox(
            "Promotion decision",
            decision_options,
            key="experiment_decision_filter",
        )
        role_filter = filter_col_b.selectbox(
            "Research role",
            role_options,
            key="experiment_role_filter",
        )
        phase_filter = filter_col_c.selectbox(
            "Experiment phase",
            phase_options,
            key="experiment_phase_filter",
        )
        family_filter = filter_col_d.selectbox(
            "Research family",
            family_options,
            key="experiment_family_filter",
        )

        experiment_view = experiment_scorecards[
            experiment_scorecards["iteration"].isin(selected_iterations)
        ]
        if decision_filter != "all":
            experiment_view = experiment_view[
                experiment_view["promotion_decision"] == decision_filter
            ]
        if role_filter != "all":
            experiment_view = experiment_view[experiment_view["role"] == role_filter]
        if phase_filter != "all":
            experiment_view = experiment_view[experiment_view["phase"] == phase_filter]
        if family_filter != "all":
            experiment_view = experiment_view[experiment_view["family"] == family_filter]
        leaderboard_columns = [
            "iteration",
            "strategy",
            "phase",
            "family",
            "role",
            "scenario_sizing",
            "promotion_decision",
            "promotion_score",
            "robustness_score",
            "cagr",
            "max_drawdown",
            "calmar",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
            "left_tail_regime_cagr",
            "hypothesis",
        ]
        _render_metric_dataframe(
            _display_metrics(
                experiment_view[
                    [column for column in leaderboard_columns if column in experiment_view.columns]
                ]
            )
        )

    with experiment_regime_tab:
        if experiment_walk_forward.empty and experiment_regimes.empty:
            st.write("New robustness artifacts will appear after the next experiment iteration.")
        else:
            strategy_options = sorted(experiment_scorecards["strategy"].dropna().unique())
            default_strategies = (
                experiment_scorecards.sort_values("promotion_score", ascending=False)
                .head(8)["strategy"]
                .tolist()
            )
            selected_strategies = st.multiselect(
                "Strategies",
                strategy_options,
                default=default_strategies,
                key="experiment_regime_strategies",
            )
            if not experiment_walk_forward.empty:
                walk_view = experiment_walk_forward[
                    experiment_walk_forward["strategy"].isin(selected_strategies)
                ]
                st.caption("Walk-forward holdout summary")
                _render_metric_dataframe(_display_metrics(walk_view))
            if not experiment_regimes.empty:
                regime_view = experiment_regimes[
                    experiment_regimes["strategy"].isin(selected_strategies)
                ]
                regime_options = ["all", *sorted(regime_view["regime"].dropna().unique())]
                regime_filter = st.selectbox(
                    "Regime window",
                    regime_options,
                    key="experiment_regime_filter",
                )
                if regime_filter != "all":
                    regime_view = regime_view[regime_view["regime"] == regime_filter]
                regime_columns = [
                    "iteration",
                    "strategy",
                    "regime",
                    "regime_type",
                    "total_return",
                    "cagr",
                    "max_drawdown",
                    "calmar",
                    "description",
                ]
                st.caption("Named market-transition and left-tail windows")
                _render_metric_dataframe(
                    _display_metrics(
                        regime_view[
                            [column for column in regime_columns if column in regime_view.columns]
                        ]
                    )
                )

    with experiment_detail_tab:
        detail_options = (
            experiment_scorecards.sort_values("promotion_score", ascending=False)["strategy"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_strategy = st.selectbox(
            "Strategy",
            detail_options,
            key="experiment_detail_strategy",
        )
        detail_rows = experiment_scorecards[experiment_scorecards["strategy"] == selected_strategy]
        _render_metric_dataframe(_display_metrics(detail_rows))
        if not detail_rows.empty and "hypothesis" in detail_rows:
            st.markdown(f"**Hypothesis:** {html.escape(str(detail_rows.iloc[0]['hypothesis']))}")
        if not experiment_candidates.empty:
            manifest_rows = experiment_candidates[
                experiment_candidates["strategy"] == selected_strategy
            ]
            st.caption("Candidate manifest")
            st.dataframe(manifest_rows, use_container_width=True)
        if not experiment_regimes.empty:
            st.caption("Regime diagnostics for selected strategy")
            _render_metric_dataframe(
                _display_metrics(
                    experiment_regimes[experiment_regimes["strategy"] == selected_strategy]
                )
            )

    with experiment_manifest_tab:
        if experiment_candidates.empty:
            st.write("No candidate manifests were found.")
        else:
            manifest_iterations = sorted(experiment_candidates["iteration"].unique())
            selected_manifest_iterations = st.multiselect(
                "Manifest iterations",
                manifest_iterations,
                default=manifest_iterations[-5:],
                key="experiment_manifest_iterations",
            )
            manifest_view = experiment_candidates[
                experiment_candidates["iteration"].isin(selected_manifest_iterations)
            ]
            st.dataframe(manifest_view, use_container_width=True)


def _render_signal_inclusion(baseline_run: BaselineRun) -> None:
    st.subheader("Signal Inclusion Tests")
    signal_inclusion = baseline_run.signal_inclusion
    if signal_inclusion.summary.empty:
        st.write("No signal-inclusion diagnostics available.")
        return

    decision_options = ["all", *sorted(signal_inclusion.summary["decision"].unique())]
    test_status_options = ["all", *sorted(signal_inclusion.summary["test_status"].unique())]
    inclusion_decision = st.selectbox("Inclusion decision", decision_options)
    inclusion_status = st.selectbox("Inclusion test status", test_status_options)
    inclusion_view = signal_inclusion.summary.copy()
    if inclusion_decision != "all":
        inclusion_view = inclusion_view[inclusion_view["decision"] == inclusion_decision]
    if inclusion_status != "all":
        inclusion_view = inclusion_view[inclusion_view["test_status"] == inclusion_status]
    inclusion_columns = [
        "signal_group",
        "test_status",
        "decision",
        "latest_pressure_state",
        "latest_pressure",
        "active_day_rate",
        "delta_cagr",
        "delta_sharpe",
        "max_drawdown_improvement",
        "delta_calmar",
        "delta_worst_3y_cagr",
        "revision_safe",
        "rationale",
    ]
    available_inclusion_columns = [
        column for column in inclusion_columns if column in inclusion_view.columns
    ]
    _render_metric_dataframe(_display_metrics(inclusion_view[available_inclusion_columns]))


def _render_research_lab(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    _render_approach_explorer(bot_config, baseline_run)
    st.divider()
    _render_experiment_monitor(
        experiment_scorecards,
        experiment_regimes,
        experiment_walk_forward,
        experiment_candidates,
    )
    st.divider()
    _render_signal_inclusion(baseline_run)


def _render_news_and_macro(baseline_run: BaselineRun) -> None:
    current_state = baseline_run.current_state

    st.subheader("Signal Coverage")
    coverage_cols = st.columns(3)
    _helped_metric(coverage_cols[0], "Market Proxies", f"{baseline_run.prices.shape[1]:,}")
    _helped_metric(coverage_cols[1], "Macro Configured", f"{len(baseline_run.macro_catalog):,}")
    _helped_metric(coverage_cols[2], "Macro Loaded", f"{baseline_run.macro_data.shape[1]:,}")
    st.dataframe(current_state.signal_coverage, use_container_width=True)

    st.subheader("Macro State")
    if current_state.macro_category_summary.empty:
        st.write("No macro diagnostics available.")
    else:
        _render_metric_dataframe(_display_metrics(current_state.macro_category_summary))
        macro_category_options = ["all", *sorted(current_state.macro_signals["category"].unique())]
        macro_category = st.selectbox("Macro category", macro_category_options)
        macro_near_term_options = [
            "all",
            *sorted(current_state.macro_signals["near_term_state"].unique()),
        ]
        macro_near_term = st.selectbox("Near-term macro state", macro_near_term_options)
        macro_signals = current_state.macro_signals
        if macro_category != "all":
            macro_signals = macro_signals[macro_signals["category"] == macro_category]
        if macro_near_term != "all":
            macro_signals = macro_signals[macro_signals["near_term_state"] == macro_near_term]
        _render_metric_dataframe(_display_metrics(macro_signals))

    st.subheader("News Intake Monitor")
    news_monitor = baseline_run.news_monitor
    if news_monitor.source_health.empty:
        st.write("No news sources are configured.")
    else:
        st.dataframe(news_monitor.source_health, use_container_width=True)

    if news_monitor.triage.empty:
        st.write("No recent news items were triaged.")
    else:
        activation_options = ["all", *sorted(news_monitor.triage["activation_status"].unique())]
        category_options = ["all", *sorted(news_monitor.triage["category"].unique())]
        news_activation = st.selectbox("News activation status", activation_options)
        news_category = st.selectbox("News category", category_options)
        news_view = news_monitor.triage.copy()
        if news_activation != "all":
            news_view = news_view[news_view["activation_status"] == news_activation]
        if news_category != "all":
            news_view = news_view[news_view["category"] == news_category]
        news_columns = [
            "title",
            "source",
            "published_at",
            "category",
            "direction",
            "phase",
            "urgency_score",
            "activation_status",
            "event_id",
            "risk_channels",
            "candidate_proxies",
            "confirmation_window",
            "url",
        ]
        available_news_columns = [column for column in news_columns if column in news_view.columns]
        _render_metric_dataframe(_display_metrics(news_view[available_news_columns].head(100)))

    st.subheader("Event-Risk Monitor")
    event_risk = baseline_run.event_risk
    if event_risk.current_event_scenarios.empty:
        st.write("No current-event scenario playbook is configured.")
    else:
        st.dataframe(event_risk.current_event_scenarios, use_container_width=True)

    if event_risk.event_summary.empty:
        st.write("No historical event-window diagnostics are available.")
    else:
        event_window_filter = st.multiselect(
            "Event windows",
            sorted(event_risk.event_summary["window"].unique()),
            default=["post_5d", "post_21d"],
        )
        event_summary = event_risk.event_summary
        if event_window_filter:
            event_summary = event_summary[event_summary["window"].isin(event_window_filter)]
        _render_metric_dataframe(_display_metrics(event_summary))

    if not event_risk.strategy_event_returns.empty:
        complete_strategy_events = event_risk.strategy_event_returns[
            event_risk.strategy_event_returns["complete"]
        ]
        _render_metric_dataframe(_display_metrics(complete_strategy_events))

    st.subheader("Data Quality")
    _render_metric_dataframe(_display_metrics(current_state.data_quality))


def _render_performance(baseline_run: BaselineRun) -> None:
    st.subheader("Performance")
    _render_metric_dataframe(_display_metrics(baseline_run.metrics))

    st.subheader("Windowed Performance")
    strategy_names = list(baseline_run.results)
    earliest_result_date, latest_result_date = _result_date_bounds(baseline_run.results)
    window_columns = st.columns([1, 2])
    window_preset = window_columns[0].selectbox(
        "Window",
        list(DEFAULT_PERFORMANCE_WINDOWS),
        index=list(DEFAULT_PERFORMANCE_WINDOWS).index(DEFAULT_PERFORMANCE_WINDOW),
    )
    selected_performance_strategies = window_columns[1].multiselect(
        "Approaches",
        strategy_names,
        default=_default_strategy_selection(strategy_names),
    )

    custom_start_date: date | None = None
    window_end = latest_result_date
    if window_preset == "Custom":
        custom_columns = st.columns(2)
        custom_start_date = cast(
            date,
            custom_columns[0].date_input(
                "Start",
                value=max(earliest_result_date, latest_result_date - pd.DateOffset(days=90)).date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
            ),
        )
        custom_end_date = cast(
            date,
            custom_columns[1].date_input(
                "End",
                value=latest_result_date.date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
            ),
        )
        window_end = min(latest_result_date, max(earliest_result_date, pd.Timestamp(custom_end_date)))

    window_start = _window_start_from_preset(
        window_preset,
        earliest=earliest_result_date,
        latest=latest_result_date,
        custom_start=custom_start_date,
    )
    if window_start > window_end:
        window_start = window_end

    if selected_performance_strategies:
        st.plotly_chart(
            make_equity_drawdown_figure(
                baseline_run.results,
                strategy_names=selected_performance_strategies,
                start=window_start,
                end=window_end,
                rebase=True,
                title=f"Growth of $1: {window_start.date()} to {window_end.date()}",
            ),
            use_container_width=True,
        )
        window_stats = window_performance_frame(
            baseline_run.results,
            strategy_names=selected_performance_strategies,
            start=window_start,
            end=window_end,
        )
        _render_metric_dataframe(_display_metrics(window_stats))
    else:
        st.warning("Select at least one approach.")

    st.subheader("Rolling Window Summary")
    _render_metric_dataframe(_display_metrics(baseline_run.window_summary))

    st.subheader("Calendar Year Returns")
    st.dataframe(
        baseline_run.calendar_returns.map(lambda value: f"{value:.2%}"),
        use_container_width=True,
    )

    st.subheader("Full-History Equity and Drawdown")
    st.plotly_chart(make_equity_drawdown_figure(baseline_run.results), use_container_width=True)

    st.subheader("Latest Positions")
    positions = latest_positions_frame(baseline_run.results)
    st.dataframe(positions.map(lambda value: f"{value:.2%}"), use_container_width=True)


def _render_forward_test_and_journal(
    journal: TradeJournal,
    baseline_run: BaselineRun,
) -> None:
    trade_decision = baseline_run.trade_decision
    st.subheader("Forward Test / Trade Journal")
    st.caption(
        "Lock recommendations, paper-trade them, and log actual executions so forward performance can be audited."
    )
    journal_cols = st.columns(4)
    journal_mode = journal_cols[0].selectbox("Mode", ["paper", "live"])
    journal_account = journal_cols[1].text_input("Account", "vanguard_rollover_ira_shadow")
    journal_strategy = journal_cols[2].text_input(
        "Strategy label",
        "scenario_adjusted_trade_decision",
    )
    account_value = journal_cols[3].number_input(
        "Account value",
        min_value=1.0,
        value=10000.0,
        step=1000.0,
    )

    sizing_cols = st.columns(4)
    price_band_pct = (
        sizing_cols[0].number_input("Price band %", min_value=0.0, value=0.75, step=0.25) / 100.0
    )
    size_band_pct = (
        sizing_cols[1].number_input("Size band %", min_value=0.0, value=20.0, step=5.0) / 100.0
    )
    min_trade_notional = sizing_cols[2].number_input(
        "Min trade $",
        min_value=0.0,
        value=25.0,
        step=25.0,
    )
    whole_shares = sizing_cols[3].checkbox("Whole shares", value=True)
    sizing = TicketSizingConfig(
        account_value=float(account_value),
        price_band_pct=float(price_band_pct),
        size_band_pct=float(size_band_pct),
        min_trade_notional=float(min_trade_notional),
        whole_shares=bool(whole_shares),
    )
    ticket_preview = build_recommendation_tickets(
        trade_decision,
        baseline_run.prices,
        mode=journal_mode,
        account=journal_account,
        strategy_name=journal_strategy,
        sizing=sizing,
    )
    ticket_columns = [
        "ticker",
        "side",
        "source_action",
        "current_weight",
        "target_weight",
        "delta_weight",
        "reference_price",
        "limit_low",
        "limit_high",
        "target_notional",
        "min_notional",
        "max_notional",
        "min_shares",
        "max_shares",
    ]
    if ticket_preview.empty:
        st.write("No executable recommendation tickets from the current decision and sizing inputs.")
    else:
        st.caption("Preview of tickets that would be locked from the current trade decision.")
        st.dataframe(
            _display_trade_frame(ticket_preview[ticket_columns]),
            use_container_width=True,
        )
        if st.button("Lock Current Recommendation Set"):
            decision_id = journal.save_decision_snapshot(
                mode=journal_mode,
                account=journal_account,
                strategy_name=journal_strategy,
                trade_decision=trade_decision,
                sizing=sizing,
                tickets=ticket_preview,
            )
            st.success(f"Locked {len(ticket_preview):,} recommendation tickets: {decision_id}")

    st.caption("Locked recommendations")
    ticket_status = st.selectbox("Ticket status", ["open", "all", "executed", "skipped", "expired"])
    stored_tickets = journal.load_recommendation_tickets(
        status=None if ticket_status == "all" else ticket_status
    )
    stored_ticket_columns = [
        "created_at_utc",
        "ticket_id",
        "status",
        "mode",
        "account",
        "strategy_name",
        "ticker",
        "side",
        "reference_price",
        "limit_low",
        "limit_high",
        "target_notional",
        "min_shares",
        "max_shares",
        "rationale",
    ]
    if stored_tickets.empty:
        st.write("No locked recommendation tickets yet.")
    else:
        available_columns = [column for column in stored_ticket_columns if column in stored_tickets]
        st.dataframe(
            _display_trade_frame(stored_tickets[available_columns]),
            use_container_width=True,
        )

    st.caption("Execution log")
    open_tickets = journal.load_recommendation_tickets(status="open")
    ticket_options = ["manual"]
    if not open_tickets.empty:
        ticket_options.extend(
            [
                f"{row['ticket_id']} | {row['ticker']} {row['side']} "
                f"{float(row['min_shares']):.2f}-{float(row['max_shares']):.2f}"
                for _, row in open_tickets.iterrows()
            ]
        )

    with st.form("execution_log_form"):
        selected_ticket_label = st.selectbox("Recommendation ticket", ticket_options)
        selected_ticket_id = None
        selected_ticket = None
        if selected_ticket_label != "manual":
            selected_ticket_id = selected_ticket_label.split(" | ", maxsplit=1)[0]
            selected_ticket_rows = open_tickets[open_tickets["ticket_id"] == selected_ticket_id]
            if not selected_ticket_rows.empty:
                selected_ticket = selected_ticket_rows.iloc[0]

        default_ticker = str(selected_ticket["ticker"]) if selected_ticket is not None else "QQQ"
        default_side = str(selected_ticket["side"]) if selected_ticket is not None else "BUY"
        default_price = (
            float(selected_ticket["reference_price"])
            if selected_ticket is not None
            else float(baseline_run.prices.ffill().iloc[-1].get(default_ticker, 0.0))
        )
        default_quantity = float(selected_ticket["min_shares"]) if selected_ticket is not None else 1.0
        execution_cols = st.columns(5)
        execution_ticker = execution_cols[0].text_input("Ticker", default_ticker).upper()
        execution_side = execution_cols[1].selectbox(
            "Side",
            ["BUY", "SELL"],
            index=0 if default_side == "BUY" else 1,
        )
        execution_quantity = execution_cols[2].number_input(
            "Quantity",
            min_value=0.0001,
            value=max(default_quantity, 0.0001),
            step=1.0 if whole_shares else 0.1,
            format="%.4f",
        )
        execution_price = execution_cols[3].number_input(
            "Price",
            min_value=0.01,
            value=max(default_price, 0.01),
            step=0.01,
            format="%.4f",
        )
        execution_fees = execution_cols[4].number_input(
            "Fees",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )

        time_cols = st.columns(3)
        execution_timezone_name = time_cols[0].text_input("Timezone", "America/Denver")
        execution_timezone = _safe_timezone(execution_timezone_name)
        default_execution_time = datetime.now(execution_timezone).replace(microsecond=0)
        execution_date = time_cols[1].date_input("Execution date", default_execution_time.date())
        execution_time = time_cols[2].time_input("Execution time", default_execution_time.time())
        execution_notes = st.text_area("Execution notes", "")
        submitted_execution = st.form_submit_button("Log Execution")

        if submitted_execution:
            local_execution_time = datetime.combine(execution_date, execution_time).replace(
                tzinfo=execution_timezone
            )
            execution_id = journal.log_execution(
                mode=journal_mode,
                account=journal_account,
                ticker=execution_ticker,
                side=execution_side,
                quantity=float(execution_quantity),
                price=float(execution_price),
                executed_at_utc=local_execution_time.astimezone(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                recommendation_id=selected_ticket_id,
                fees=float(execution_fees),
                notes=execution_notes,
            )
            st.success(f"Logged execution: {execution_id}")

    if not open_tickets.empty:
        status_cols = st.columns(2)
        status_ticket = status_cols[0].selectbox(
            "Open ticket to update",
            list(open_tickets["ticket_id"]),
        )
        status_update = status_cols[1].selectbox("New status", ["skipped", "expired", "open"])
        if st.button("Update Ticket Status"):
            journal.update_ticket_status(status_ticket, status_update)
            st.success(f"Updated ticket {status_ticket} to {status_update}.")

    executions = journal.load_executions()
    execution_columns = [
        "executed_at_utc",
        "execution_id",
        "recommendation_id",
        "mode",
        "account",
        "ticker",
        "side",
        "quantity",
        "price",
        "notional",
        "fees",
        "notes",
    ]
    if executions.empty:
        st.write("No executions logged yet.")
    else:
        st.dataframe(
            _display_trade_frame(executions[execution_columns]),
            use_container_width=True,
        )

    position_summary = journal.execution_position_summary()
    if not position_summary.empty:
        st.caption("Execution-derived position summary")
        st.dataframe(_display_trade_frame(position_summary), use_container_width=True)


def _render_dashboard_section(
    section: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    journal: TradeJournal,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    if section == "Command Center":
        _render_command_center(baseline_run)
    elif section == "Risk & Scenarios":
        _render_risk_and_scenarios(baseline_run)
    elif section == "Research Lab":
        _render_research_lab(
            bot_config,
            baseline_run,
            experiment_scorecards,
            experiment_regimes,
            experiment_walk_forward,
            experiment_candidates,
        )
    elif section == "News & Macro":
        _render_news_and_macro(baseline_run)
    elif section == "Performance":
        _render_performance(baseline_run)
    elif section == "Forward Test":
        _render_forward_test_and_journal(journal, baseline_run)


st.set_page_config(page_title="Trade Bot Dashboard", layout="wide")
_install_dashboard_styles()
st.title("Trade Bot Operations")

config_path = Path(st.sidebar.text_input("Config", str(DEFAULT_CONFIG_PATH)))
events_path = Path(st.sidebar.text_input("Events", str(DEFAULT_EVENTS_PATH)))
macro_path = Path(st.sidebar.text_input("Macro", str(DEFAULT_MACRO_PATH)))
news_path = Path(st.sidebar.text_input("News", str(DEFAULT_NEWS_PATH)))
journal_path = Path(st.sidebar.text_input("Trade journal", str(DEFAULT_JOURNAL_PATH)))
run_store_path = Path(st.sidebar.text_input("Run store", str(DEFAULT_RUN_STORE_DB_PATH)))
artifact_dir = Path(
    st.sidebar.text_input("Snapshot artifacts", str(DEFAULT_RUN_STORE_ARTIFACT_DIR))
)
job_log_dir = Path(st.sidebar.text_input("Snapshot job logs", str(DEFAULT_RUN_STORE_JOB_LOG_DIR)))
run_source = st.sidebar.radio("Run source", ["Latest snapshot (fast)", "Live pipeline"], index=0)
refresh_data = st.sidebar.checkbox("Refresh market data", value=False)
refresh_macro = st.sidebar.checkbox("Refresh macro data", value=False)
refresh_news = st.sidebar.checkbox("Refresh news", value=False)
st.sidebar.caption(
    "Fast mode reads the latest precomputed snapshot. Live mode runs the full pipeline."
)
bot_config = load_config(config_path)

run_store = RunStore(run_store_path, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
if st.sidebar.button("Start Background Snapshot Refresh"):
    job = run_store.start_snapshot_build_job(
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
    )
    st.sidebar.success(f"Queued snapshot job: {job.job_id}")

snapshot_jobs = load_snapshot_jobs_frame(str(run_store_path), str(artifact_dir), str(job_log_dir))
if not snapshot_jobs.empty:
    with st.sidebar.expander("Snapshot jobs", expanded=False):
        job_columns = [
            "created_at_utc",
            "status",
            "run_id",
            "completed_at_utc",
            "log_path",
            "error_message",
        ]
        st.dataframe(snapshot_jobs[job_columns], use_container_width=True)


@st.cache_data(show_spinner="Running backtests...")
def load_live_run(
    config_path_string: str,
    events_path_string: str,
    macro_path_string: str,
    news_path_string: str,
    refresh: bool,
    refresh_macro_data: bool,
    refresh_news_data: bool,
) -> BaselineRun:
    config = load_config(config_path_string)
    return run_configured_baselines(
        config,
        refresh_data=refresh,
        refresh_macro=refresh_macro_data,
        refresh_news=refresh_news_data,
        event_config_path=events_path_string,
        macro_config_path=macro_path_string,
        news_config_path=news_path_string,
    )


snapshot_manifest: SnapshotManifest | None = None
snapshot_loaded = False
if run_source == "Latest snapshot (fast)":
    snapshot_payload = load_snapshot_dashboard_run(
        str(config_path),
        str(events_path),
        str(macro_path),
        str(news_path),
        str(run_store_path),
        str(artifact_dir),
        str(job_log_dir),
    )
    if snapshot_payload is None:
        st.warning(
            "No completed snapshot matches the current config files. "
            "Falling back to a live run for this session; build a snapshot to make cold opens fast."
        )
        baseline_run = load_live_run(
            str(config_path),
            str(events_path),
            str(macro_path),
            str(news_path),
            refresh_data,
            refresh_macro,
            refresh_news,
        )
    else:
        baseline_run, snapshot_manifest = snapshot_payload
        snapshot_loaded = True
else:
    baseline_run = load_live_run(
        str(config_path),
        str(events_path),
        str(macro_path),
        str(news_path),
        refresh_data,
        refresh_macro,
        refresh_news,
    )

if snapshot_manifest is not None:
    st.sidebar.success(
        "Snapshot loaded: "
        f"{snapshot_manifest.market_date} | {snapshot_manifest.risk_status.upper()} | "
        f"{snapshot_manifest.created_at_utc}"
    )
elif not snapshot_loaded:
    st.sidebar.info("Dashboard is using a live pipeline run.")

journal = TradeJournal(journal_path)
headline_open_tickets = journal.load_recommendation_tickets(status="open")
action_headline = build_action_headline(
    current_state=baseline_run.current_state,
    trade_decision=baseline_run.trade_decision,
    news_monitor=baseline_run.news_monitor,
    open_ticket_count=len(headline_open_tickets),
)
_render_action_headline(action_headline)
(
    experiment_scorecards,
    experiment_regimes,
    experiment_walk_forward,
    experiment_candidates,
) = load_experiment_dashboard_frames()
_render_operating_brief(
    baseline_run=baseline_run,
    headline=action_headline,
)
_render_decision_brief(
    baseline_run=baseline_run,
    headline=action_headline,
    open_ticket_count=len(headline_open_tickets),
    experiment_scorecards=experiment_scorecards,
)
_render_metric_guide()
st.divider()

DASHBOARD_SECTIONS = (
    "Command Center",
    "Risk & Scenarios",
    "Research Lab",
    "News & Macro",
    "Performance",
    "Forward Test",
)
selected_section = st.radio(
    "Dashboard section",
    DASHBOARD_SECTIONS,
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(
    "Sections render one at a time so the operating view stays focused and dense research tables do not dominate the page."
)
_render_dashboard_section(
    selected_section,
    bot_config=bot_config,
    baseline_run=baseline_run,
    journal=journal,
    experiment_scorecards=experiment_scorecards,
    experiment_regimes=experiment_regimes,
    experiment_walk_forward=experiment_walk_forward,
    experiment_candidates=experiment_candidates,
)
