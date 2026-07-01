from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _format_currency,
    _format_decimal,
    _format_percent,
    _result_date_bounds,
    _window_start_from_preset,
)
from trade_bot.dashboard.metric_explainers import metric_detail
from trade_bot.DEFAULTS import (
    DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS,
    DEFAULT_DECISION_TIMELINE_FORWARD_DAYS,
    DEFAULT_DECISION_TIMELINE_MAX_EVENTS,
    DEFAULT_DEFAULT_APPROACH_RESEARCH_STATUSES,
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_OPERABILITY_MATERIAL_TRADE_TURNOVER_THRESHOLD,
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_PEER_CURVE_METRIC_LIMIT,
    DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
    DEFAULT_PERFORMANCE_WINDOW,
    DEFAULT_PERFORMANCE_WINDOWS,
    DEFAULT_REFERENCE_BASELINE_STRATEGIES,
)
from trade_bot.features.indicators import drawdown, ulcer_index
from trade_bot.reporting.report import make_equity_drawdown_figure, window_performance_frame
from trade_bot.research.approach_explorer import (
    build_approach_allocation_transition_events,
    build_approach_backtest_result,
    build_approach_catalog,
    build_approach_change_log,
    build_approach_decision_events,
    build_approach_explanation,
    build_approach_exposure_history,
    build_approach_holding_stats,
    build_approach_mechanics,
    build_approach_position_summary,
    build_approach_risk_notes,
    build_approach_steps,
    build_approach_weight_history,
    build_latest_weight_frame,
    decision_sanity_from_catalog_row,
    execution_for_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.curation import rank_strategy_candidates, select_curated_strategy_shelf
from trade_bot.research.experiment_monitor import (
    build_strategy_family_map,
    latest_experiment_iteration,
    strategy_family_takeaways,
    summarize_decision_sanity_impacts,
    summarize_experiment_families,
    summarize_experiment_history,
    summarize_experiment_operating_systems,
    summarize_family_clusters,
    summarize_risk_behavior_matrix,
    summarize_strategy_archetypes,
)
from trade_bot.research.factor_attribution import (
    build_factor_attribution,
    build_factor_decay_monitor,
)
from trade_bot.research.signal_evidence import (
    build_signal_family_evidence,
    build_signal_family_marginal_tests,
    signal_evidence_takeaways,
    tag_scorecard_signal_families,
)
from trade_bot.research.strategy_outcome_utility import (
    add_outcome_frontier_flags,
    drawdown_recovery_return,
    enrich_strategy_outcome_utility,
    terminal_wealth_from_cagr,
)

_OUTCOME_FRONTIER_PLOT_KEY = "outcome_frontier_plot"
_OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY = "outcome_frontier_selected_strategy"
_OUTCOME_FRONTIER_SELECTBOX_KEY = "outcome_frontier_strategy_label"


def _render_taxable_estimate_summary(scorecard: pd.DataFrame) -> None:
    tax_columns = [
        "tax_model_status",
        "tax_account_type",
        "after_tax_cagr",
        "after_tax_max_drawdown",
        "after_tax_calmar",
        "tax_drag_bps_per_year",
        "after_tax_growth_constrained_utility_score",
        "after_tax_terminal_wealth_with_contributions_15y",
        "net_estimated_tax_paid",
        "realized_short_term_gain",
        "realized_long_term_gain",
        "realized_loss_harvested",
        "wash_sale_disallowed_loss",
        "loss_carryforward_end",
        "short_term_gain_share",
    ]
    available = [column for column in tax_columns if column in scorecard]
    if not available:
        return
    tax_status = str(scorecard.iloc[0].get("tax_model_status", ""))
    if not tax_status or tax_status == "not_evaluated":
        return
    st.caption("Estimated taxable-account readout")
    _render_metric_dataframe(_display_metrics(scorecard[available]), hide_index=True)


def _render_strategy_explanation(
    *,
    row: pd.Series,
    strategy: Any,
    bot_config: Any,
) -> None:
    scenario_sizing = scenario_sizing_from_catalog_row(row)
    future_state_model = future_state_model_from_catalog_row(row)
    strategy_drawdown_model = strategy_drawdown_model_from_catalog_row(row)
    decision_sanity = decision_sanity_from_catalog_row(row)
    execution = execution_for_catalog_row(row, bot_config.execution)
    st.markdown("**How this approach works**")
    for paragraph in build_approach_explanation(
        strategy,
        row,
        bot_config,
        execution=execution,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
    ):
        st.write(paragraph)


def _config_cache_payload(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return, attr-defined]
    if is_dataclass(value):
        return asdict(value)  # type: ignore[arg-type]
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_config_cache_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _config_cache_payload(item) for key, item in value.items()}
    return str(value)


def _approach_result_cache_key(
    *,
    baseline_run: BaselineRun,
    strategy: Any,
    execution: Any,
    scenario_sizing: Any,
    future_state_model: Any,
    strategy_drawdown_model: Any,
    decision_sanity: Any,
    name: str,
) -> str:
    prices = baseline_run.prices
    if prices.empty:
        price_marker: object = "empty"
    else:
        price_marker = {
            "rows": int(len(prices)),
            "columns": list(map(str, prices.columns)),
            "start": str(prices.index.min()),
            "end": str(prices.index.max()),
        }
    payload = {
        "price_marker": price_marker,
        "strategy": _config_cache_payload(strategy),
        "execution": _config_cache_payload(execution),
        "scenario_sizing": _config_cache_payload(scenario_sizing),
        "future_state_model": _config_cache_payload(future_state_model),
        "strategy_drawdown_model": _config_cache_payload(strategy_drawdown_model),
        "decision_sanity": _config_cache_payload(decision_sanity),
        "name": name,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cached_approach_backtest_result(
    *,
    row: pd.Series,
    strategy: Any,
    execution: Any,
    scenario_sizing: Any,
    future_state_model: Any,
    strategy_drawdown_model: Any,
    decision_sanity: Any,
    baseline_run: BaselineRun,
) -> tuple[BacktestResult | None, list[str]]:
    cache = st.session_state.setdefault("approach_backtest_result_cache", {})
    name = str(row.get("strategy", "approach"))
    cache_key = _approach_result_cache_key(
        baseline_run=baseline_run,
        strategy=strategy,
        execution=execution,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
        name=name,
    )
    if cache_key not in cache:
        with st.spinner("Reconstructing selected strategy history..."):
            cache[cache_key] = build_approach_backtest_result(
                baseline_run.prices,
                strategy,
                execution,
                scenario_sizing=scenario_sizing,
                future_state_model=future_state_model,
                strategy_drawdown_model=strategy_drawdown_model,
                decision_sanity=decision_sanity,
                name=name,
            )
        if len(cache) > 24:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
    result, missing_columns = cache[cache_key]
    if missing_columns:
        st.caption("Missing from loaded prices: " + ", ".join(missing_columns))
    if result is None:
        st.warning("Could not reconstruct historical weights for this approach from loaded prices.")
    return result, missing_columns


def _load_detail_result_if_needed(
    *,
    selected_detail_view: str,
    row: pd.Series,
    strategy: Any,
    execution: Any,
    scenario_sizing: Any,
    future_state_model: Any,
    strategy_drawdown_model: Any,
    decision_sanity: Any,
    baseline_run: BaselineRun,
) -> BacktestResult | None:
    result_views = {
        "Performance + Allocation",
        "Decision Timeline",
        "Performance Over Time",
        "Allocation Behavior",
        "Factor Attribution",
    }
    if selected_detail_view not in result_views:
        return None
    result, _ = _cached_approach_backtest_result(
        row=row,
        strategy=strategy,
        execution=execution,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
        baseline_run=baseline_run,
    )
    return result


def _render_position_behavior(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str | None,
    key_prefix: str,
) -> None:
    if weights.empty:
        st.write("No position history is available for this approach.")
        return

    earliest_weight_date = pd.Timestamp(weights.index.min())
    latest_weight_date = pd.Timestamp(weights.index.max())
    _, window_start, window_end = _select_history_window(
        label="Position-history window",
        earliest=earliest_weight_date,
        latest=latest_weight_date,
        key_prefix=f"{key_prefix}_position",
    )

    exposure_history = build_approach_exposure_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )
    weight_history = build_approach_weight_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Risk vs defensive/cash exposure")
        st.line_chart(exposure_history)
    with chart_cols[1]:
        st.caption("Allocation weights by selected holding")
        st.area_chart(weight_history)

    st.caption("Position behavior summary")
    _render_metric_dataframe(
        build_approach_position_summary(
            weights,
            defensive_ticker=defensive_ticker,
            lookback_days=None,
            start=window_start,
            end=window_end,
        ),
        hide_index=True,
    )

    st.caption("Recent allocation weights")
    _render_metric_dataframe(
        _format_weight_history_table(weight_history.tail(15)),
        hide_index=True,
    )

    change_log = build_approach_change_log(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )
    if change_log.empty:
        st.caption("No material allocation changes in the selected window.")
    else:
        st.caption("Recent material allocation changes")
        _render_metric_dataframe(_display_metrics(change_log), hide_index=True)

    holding_stats = build_approach_holding_stats(
        weights,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )
    if not holding_stats.empty:
        st.caption("Holding behavior by ticker")
        _render_metric_dataframe(_display_metrics(holding_stats), hide_index=True)


def _format_weight_history_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.reset_index().copy()
    for column in display.columns:
        if column == "date":
            display[column] = display[column].astype(str)
            continue
        display[column] = display[column].map(lambda value: f"{float(value):.1%}")
    return display


def _select_history_window(
    *,
    label: str,
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    key_prefix: str,
) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    window_options = list(DEFAULT_PERFORMANCE_WINDOWS)
    default_index = (
        window_options.index(DEFAULT_PERFORMANCE_WINDOW)
        if DEFAULT_PERFORMANCE_WINDOW in window_options
        else 0
    )
    window_columns = st.columns([1, 2])
    window_preset = window_columns[0].selectbox(
        label,
        window_options,
        index=default_index,
        key=f"{key_prefix}_window",
    )
    custom_start_date: date | None = None
    window_end = latest
    if window_preset == "Custom":
        custom_columns = st.columns(2)
        custom_start_date = cast(
            date,
            custom_columns[0].date_input(
                "Start",
                value=max(earliest, latest - pd.DateOffset(days=90)).date(),
                min_value=earliest.date(),
                max_value=latest.date(),
                key=f"{key_prefix}_start",
            ),
        )
        custom_end_date = cast(
            date,
            custom_columns[1].date_input(
                "End",
                value=latest.date(),
                min_value=earliest.date(),
                max_value=latest.date(),
                key=f"{key_prefix}_end",
            ),
        )
        window_end = min(latest, max(earliest, pd.Timestamp(custom_end_date)))

    window_start = _window_start_from_preset(
        window_preset,
        earliest=earliest,
        latest=latest,
        custom_start=custom_start_date,
    )
    if window_start > window_end:
        window_start = window_end
    return window_preset, window_start, window_end


def _render_performance_allocation_context(
    result: BacktestResult,
    *,
    baseline_run: BaselineRun,
    defensive_ticker: str | None,
    key_prefix: str,
) -> None:
    comparison_options = [
        name
        for name in [
            "buy_hold_spy",
            "buy_hold_qqq",
            "buy_hold_bil",
            "drawdown_managed_dual_momentum",
        ]
        if name in baseline_run.results and name != result.name
    ]
    selected_comparisons = st.multiselect(
        "Comparison lines",
        comparison_options,
        default=comparison_options[:2],
        key=f"{key_prefix}_comparisons",
    )
    chart_results = {result.name: result}
    chart_results.update({name: baseline_run.results[name] for name in selected_comparisons})

    earliest_result_date, latest_result_date = _result_date_bounds({result.name: result})
    _, window_start, window_end = _select_history_window(
        label="Shared performance/allocation window",
        earliest=earliest_result_date,
        latest=latest_result_date,
        key_prefix=f"{key_prefix}_shared",
    )

    exposure_history = build_approach_exposure_history(
        result.weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )
    weight_history = build_approach_weight_history(
        result.weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=window_start,
        end=window_end,
    )

    st.plotly_chart(
        _make_performance_allocation_figure(
            chart_results,
            exposure_history,
            start=window_start,
            end=window_end,
            title=f"Performance, drawdown, and allocation: {window_start.date()} to {window_end.date()}",
        ),
        use_container_width=True,
    )

    stats_col, behavior_col = st.columns(2)
    with stats_col:
        st.caption("Window performance stats")
        window_stats = window_performance_frame(
            chart_results,
            start=window_start,
            end=window_end,
        )
        if not window_stats.empty:
            _render_metric_dataframe(_display_metrics(window_stats), hide_index=True)
    with behavior_col:
        st.caption("Window allocation summary")
        _render_metric_dataframe(
            build_approach_position_summary(
                result.weights,
                defensive_ticker=defensive_ticker,
                lookback_days=None,
                start=window_start,
                end=window_end,
            ),
            hide_index=True,
        )

    if not weight_history.empty:
        st.caption("Detailed allocation weights for the same window")
        st.area_chart(weight_history)

    event_frame = build_approach_allocation_transition_events(
        result,
        defensive_ticker=defensive_ticker,
        start=window_start,
        end=window_end,
    )
    if not event_frame.empty:
        st.caption("Transition events inside the selected window")
        _render_metric_dataframe(_display_metrics(event_frame), hide_index=True)


def _make_performance_allocation_figure(
    results: dict[str, BacktestResult],
    exposure_history: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Growth of $1", "Drawdown", "Allocation exposure"),
    )
    for name, result in results.items():
        equity = result.equity.sort_index().dropna()
        equity = equity.loc[(equity.index >= start) & (equity.index <= end)]
        if equity.empty:
            continue
        normalized = equity / equity.iloc[0]
        figure.add_trace(
            go.Scatter(x=normalized.index, y=normalized, mode="lines", name=name),
            row=1,
            col=1,
        )
        strategy_drawdown = drawdown(normalized)
        figure.add_trace(
            go.Scatter(
                x=strategy_drawdown.index,
                y=strategy_drawdown,
                mode="lines",
                name=f"{name} drawdown",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    allocation_colors = {
        "risk_assets": "#0f766e",
        "defensive": "#f59e0b",
        "cash_or_unallocated": "#94a3b8",
    }
    for column in ["risk_assets", "defensive", "cash_or_unallocated"]:
        if column not in exposure_history:
            continue
        figure.add_trace(
            go.Scatter(
                x=exposure_history.index,
                y=exposure_history[column],
                mode="lines",
                stackgroup="allocation",
                name=column.replace("_", " "),
                line={"color": allocation_colors.get(column)},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            ),
            row=3,
            col=1,
        )

    figure.update_yaxes(tickprefix="$", tickformat=".2f", row=1, col=1)
    figure.update_yaxes(tickformat=".0%", row=2, col=1)
    figure.update_yaxes(tickformat=".0%", range=[0, 1], row=3, col=1)
    figure.update_layout(
        template="plotly_white",
        height=950,
        hovermode="x unified",
        title=title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return figure


def _render_decision_timeline(
    result: BacktestResult,
    *,
    defensive_ticker: str | None,
    key_prefix: str,
) -> None:
    earliest_result_date, latest_result_date = _result_date_bounds({result.name: result})
    _, window_start, window_end = _select_history_window(
        label="Decision-timeline window",
        earliest=earliest_result_date,
        latest=latest_result_date,
        key_prefix=f"{key_prefix}_decision_timeline",
    )
    event_frame = build_approach_decision_events(
        result,
        defensive_ticker=defensive_ticker,
        start=window_start,
        end=window_end,
        context_days=DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS,
        forward_days=DEFAULT_DECISION_TIMELINE_FORWARD_DAYS,
        material_change=DEFAULT_OPERABILITY_MATERIAL_TRADE_TURNOVER_THRESHOLD,
        max_events=DEFAULT_DECISION_TIMELINE_MAX_EVENTS,
    )
    landmark_frame = build_approach_allocation_transition_events(
        result,
        defensive_ticker=defensive_ticker,
        start=window_start,
        end=window_end,
        context_days=DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS,
        forward_days=DEFAULT_DECISION_TIMELINE_FORWARD_DAYS,
        material_change=DEFAULT_OPERABILITY_MATERIAL_TRADE_TURNOVER_THRESHOLD,
    )
    st.caption(
        "Decision timeline: markers show the largest material allocation moves inside the selected "
        "window, not just one max/min event. Hover for inferred driver, top adds/reductions, risk "
        "change, drawdown context, and next-window return. Drivers are inferred from reconstructed "
        "weights; use Mechanics for the formal rule set."
    )
    st.plotly_chart(
        _make_decision_timeline_figure(
            result,
            event_frame,
            landmark_frame=landmark_frame,
            defensive_ticker=defensive_ticker,
            start=window_start,
            end=window_end,
        ),
        use_container_width=True,
    )
    if event_frame.empty:
        st.caption("No material allocation decision events were detected in this window.")
    else:
        st.caption("Major allocation decision events")
        _render_metric_dataframe(_display_metrics(event_frame), hide_index=True)


def _make_decision_timeline_figure(
    result: BacktestResult,
    event_frame: pd.DataFrame,
    *,
    landmark_frame: pd.DataFrame | None = None,
    defensive_ticker: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> go.Figure:
    equity = result.equity.sort_index().dropna()
    equity = equity.loc[(equity.index >= start) & (equity.index <= end)]
    weights = result.weights.loc[
        (result.weights.index >= start) & (result.weights.index <= end)
    ].copy()
    exposure_history = build_approach_exposure_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
        start=start,
        end=end,
    )
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Growth of $1", "Drawdown", "Risk / defense posture"),
    )
    if not equity.empty:
        normalized = equity / equity.iloc[0]
        strategy_drawdown = drawdown(normalized)
        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=result.name,
                line={"color": "#2563eb", "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=strategy_drawdown.index,
                y=strategy_drawdown,
                mode="lines",
                name="drawdown",
                line={"color": "#ef4444", "width": 1.8},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        marker_frame = _decision_timeline_marker_frame(event_frame, normalized, strategy_drawdown)
        if not marker_frame.empty:
            figure.add_trace(
                go.Scatter(
                    x=marker_frame["date"],
                    y=marker_frame["equity_marker"],
                    mode="markers",
                    name="decision events",
                    marker={
                        "size": 12,
                        "symbol": marker_frame["symbol"],
                        "color": marker_frame["color"],
                        "line": {"width": 1, "color": "#0f172a"},
                    },
                    customdata=marker_frame[
                        [
                            "event",
                            "signal",
                            "inferred_driver",
                            "risk_weight_at_event",
                            "risk_weight_change",
                            "defensive_weight_change",
                            "total_change",
                            "top_adds",
                            "top_reductions",
                            "forward_return_1m",
                            "forward_return_3m",
                            "drawdown_at_event",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d}"
                        "<br>%{customdata[1]}"
                        "<br>%{customdata[2]}"
                        "<br>Risk weight: %{customdata[3]:.1%}"
                        "<br>Risk change: %{customdata[4]:+.1%}"
                        "<br>Defensive change: %{customdata[5]:+.1%}"
                        "<br>Total move: %{customdata[6]:.1%}"
                        "<br>Adds: %{customdata[7]}"
                        "<br>Reductions: %{customdata[8]}"
                        "<br>Next 1M return: %{customdata[9]:.1%}"
                        "<br>Next 3M return: %{customdata[10]:.1%}"
                        "<br>Drawdown: %{customdata[11]:.1%}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        if landmark_frame is not None and not landmark_frame.empty:
            risk_landmarks = landmark_frame[
                landmark_frame["event"].astype(str).eq("Worst drawdown point")
            ]
            landmark_markers = _decision_timeline_marker_frame(
                risk_landmarks,
                normalized,
                strategy_drawdown,
            )
            if not landmark_markers.empty:
                figure.add_trace(
                    go.Scatter(
                        x=landmark_markers["date"],
                        y=landmark_markers["equity_marker"],
                        mode="markers",
                        name="drawdown landmark",
                        marker={
                            "size": 12,
                            "symbol": landmark_markers["symbol"],
                            "color": landmark_markers["color"],
                            "line": {"width": 1, "color": "#0f172a"},
                        },
                        customdata=landmark_markers[
                            [
                                "event",
                                "signal",
                                "risk_weight_at_event",
                                "forward_return_3m",
                                "drawdown_at_event",
                            ]
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d}"
                            "<br>%{customdata[1]}"
                            "<br>Risk weight: %{customdata[2]:.1%}"
                            "<br>Next 3M return: %{customdata[3]:.1%}"
                            "<br>Drawdown: %{customdata[4]:.1%}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                for _, row in landmark_markers.iterrows():
                    figure.add_vline(
                        x=row["date"],
                        line_color=str(row["color"]),
                        line_width=1,
                        opacity=0.20,
                    )

    exposure_colors = {
        "risk_assets": "#0f766e",
        "defensive": "#f59e0b",
        "cash_or_unallocated": "#94a3b8",
    }
    for column in ["risk_assets", "defensive", "cash_or_unallocated"]:
        if column not in exposure_history:
            continue
        figure.add_trace(
            go.Scatter(
                x=exposure_history.index,
                y=exposure_history[column],
                mode="lines",
                name=column.replace("_", " "),
                line={"color": exposure_colors[column], "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            ),
            row=3,
            col=1,
        )

    figure.update_yaxes(tickprefix="$", tickformat=".2f", row=1, col=1)
    figure.update_yaxes(tickformat=".0%", row=2, col=1)
    figure.update_yaxes(tickformat=".0%", range=[0, 1], row=3, col=1)
    figure.update_layout(
        template="plotly_white",
        height=780,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def _decision_timeline_marker_frame(
    event_frame: pd.DataFrame,
    normalized_equity: pd.Series,
    strategy_drawdown: pd.Series,
) -> pd.DataFrame:
    if event_frame.empty or normalized_equity.empty:
        return pd.DataFrame()
    event_dates = pd.to_datetime(event_frame["date"], errors="coerce")
    events = event_frame.copy()
    events["date"] = event_dates
    events = events.dropna(subset=["date"])
    if events.empty:
        return pd.DataFrame()
    equity_lookup = (
        normalized_equity.reindex(normalized_equity.index.union(events["date"]))
        .sort_index()
        .ffill()
    )
    drawdown_lookup = (
        strategy_drawdown.reindex(strategy_drawdown.index.union(events["date"]))
        .sort_index()
        .ffill()
    )
    events["equity_marker"] = events["date"].map(equity_lookup)
    events["drawdown_at_event"] = events["date"].map(drawdown_lookup).fillna(
        pd.to_numeric(events.get("drawdown_at_event"), errors="coerce")
    )
    events["symbol"] = events["event"].map(
        {
            "Worst drawdown point": "x",
            "Largest de-risking move": "triangle-down",
            "Largest re-risking move": "triangle-up",
            "De-risking move": "triangle-down",
            "Re-risking move": "triangle-up",
            "Risk rotation": "diamond",
            "Defensive add": "triangle-down",
            "Defensive reduce": "triangle-up",
        }
    ).fillna("circle")
    events["color"] = events["event"].map(
        {
            "Worst drawdown point": "#ef4444",
            "Largest de-risking move": "#f59e0b",
            "Largest re-risking move": "#0f766e",
            "De-risking move": "#f59e0b",
            "Re-risking move": "#0f766e",
            "Risk rotation": "#6366f1",
            "Defensive add": "#d97706",
            "Defensive reduce": "#14b8a6",
        }
    ).fillna("#4f46e5")
    numeric_defaults = {
        "risk_weight_at_event": float("nan"),
        "risk_weight_change": float("nan"),
        "defensive_weight_change": float("nan"),
        "total_change": float("nan"),
        "forward_return_1m": float("nan"),
        "forward_return_3m": float("nan"),
    }
    for column, default in numeric_defaults.items():
        if column not in events:
            events[column] = default
        events[column] = pd.to_numeric(events[column], errors="coerce")
    text_defaults = {
        "signal": "",
        "inferred_driver": "Driver not available for this landmark.",
        "top_adds": "n/a",
        "top_reductions": "n/a",
    }
    for column, default in text_defaults.items():
        if column not in events:
            events[column] = default
        events[column] = events[column].fillna(default).astype(str)
    return events


def _render_approach_performance(
    result: BacktestResult,
    *,
    baseline_run: BaselineRun,
    key_prefix: str,
) -> None:
    comparison_options = [
        name
        for name in [
            "buy_hold_spy",
            "buy_hold_qqq",
            "buy_hold_bil",
            "drawdown_managed_dual_momentum",
        ]
        if name in baseline_run.results and name != result.name
    ]
    default_comparisons = comparison_options[:2]
    selected_comparisons = st.multiselect(
        "Comparison lines",
        comparison_options,
        default=default_comparisons,
        key=f"{key_prefix}_performance_comparisons",
    )
    chart_results = {result.name: result}
    chart_results.update({name: baseline_run.results[name] for name in selected_comparisons})

    earliest_result_date, latest_result_date = _result_date_bounds({result.name: result})
    _, window_start, window_end = _select_history_window(
        label="Performance window",
        earliest=earliest_result_date,
        latest=latest_result_date,
        key_prefix=f"{key_prefix}_performance",
    )

    st.plotly_chart(
        make_equity_drawdown_figure(
            chart_results,
            start=window_start,
            end=window_end,
            rebase=True,
            title=f"Growth of $1: {window_start.date()} to {window_end.date()}",
        ),
        use_container_width=True,
    )
    window_stats = window_performance_frame(
        chart_results,
        start=window_start,
        end=window_end,
    )
    if not window_stats.empty:
        st.caption("Window performance stats")
        _render_metric_dataframe(_display_metrics(window_stats), hide_index=True)


def _render_factor_attribution(result: BacktestResult, *, baseline_run: BaselineRun) -> None:
    attribution = build_factor_attribution(result.equity, baseline_run.prices)
    if attribution.summary.empty or attribution.factor_attribution.empty:
        st.write(
            "No factor attribution is available. The selected approach may not have enough "
            "overlapping history with the proxy factor universe."
        )
        return

    summary = attribution.summary.iloc[0]
    cols = st.columns(5)
    _helped_metric(cols[0], "Factor R2", _format_percent(summary["factor_model_r_squared"]))
    _helped_metric(cols[1], "Residual Share", _format_percent(summary["residual_contribution_share"]))
    _helped_metric(cols[2], "Dominant Factor", str(summary["dominant_factor"]))
    _helped_metric(cols[3], "Dominant Share", _format_percent(summary["dominant_factor_share"]))
    _helped_metric(
        cols[4],
        "Residual Vol",
        _format_percent(summary["residual_annualized_volatility"]),
    )

    st.caption(
        "Proxy-factor attribution: cumulative return contribution, factor beta, and variance "
        "contribution. Residual strategy behavior is the part not explained by the current proxy set."
    )
    factor_view = attribution.factor_attribution.copy()
    st.plotly_chart(
        _factor_contribution_waterfall_figure(factor_view),
        use_container_width=True,
    )
    return_risk_similarity = _safe_float(
        factor_view["absolute_contribution_share"].abs().corr(
            factor_view["risk_contribution_pct"].abs()
        )
    )
    contribution_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cumulative return contribution", "Variance contribution"),
    )
    contribution_fig.add_trace(
        go.Bar(
            x=factor_view["return_contribution"],
            y=factor_view["label"],
            orientation="h",
            name="Return",
            marker_color="#0f766e",
        ),
        row=1,
        col=1,
    )
    contribution_fig.add_trace(
        go.Bar(
            x=factor_view["risk_contribution_pct"],
            y=factor_view["label"],
            orientation="h",
            name="Risk",
            marker_color="#b45309",
        ),
        row=1,
        col=2,
    )
    contribution_fig.update_layout(
        height=420,
        template="plotly_white",
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    contribution_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(contribution_fig, use_container_width=True)
    if return_risk_similarity is not None:
        st.caption(
            "These panels can look similar when the same proxy factors both earned the returns and "
            f"explained the variance. Absolute return-share / variance-contribution similarity is "
            f"{return_risk_similarity:.2f}. Variance contribution is covariance with strategy returns, "
            "not a duplicated return-contribution calculation."
        )

    attribution_columns = [
        "label",
        "proxy_ticker",
        "beta",
        "correlation",
        "return_contribution",
        "absolute_contribution_share",
        "risk_contribution_pct",
        "annualized_factor_volatility",
        "description",
    ]
    _render_metric_dataframe(
        _display_metrics(
            factor_view[[column for column in attribution_columns if column in factor_view]]
        ),
        hide_index=True,
    )

    st.caption("Factor decay / behavior drift")
    decay = build_factor_decay_monitor(result.equity, baseline_run.prices)
    if decay.empty:
        st.write("No recent factor-decay diagnostic is available for this approach.")
        return
    flagged = decay[(decay["drift_flag"]) | (decay["model_decay_flag"])]
    if flagged.empty:
        st.success("No major recent factor-decay flags versus the full-history attribution.")
    else:
        st.warning(
            f"{len(flagged):,} factor-decay flag(s): recent behavior is diverging from "
            "the full-history factor profile."
        )
    _render_metric_dataframe(_display_metrics(decay), hide_index=True)


def _factor_contribution_waterfall_figure(factor_view: pd.DataFrame) -> go.Figure:
    if factor_view.empty or "return_contribution" not in factor_view:
        return go.Figure()
    display = factor_view.copy()
    display["return_contribution"] = pd.to_numeric(
        display["return_contribution"],
        errors="coerce",
    )
    display = display.dropna(subset=["return_contribution"])
    if display.empty:
        return go.Figure()
    display = display.sort_values(
        "return_contribution",
        key=lambda series: series.abs(),
        ascending=False,
    )
    labels = display["label"].astype(str).tolist()
    values = display["return_contribution"].astype(float).tolist()
    total = float(sum(values))
    figure = go.Figure(
        go.Waterfall(
            name="Return attribution",
            orientation="v",
            measure=["relative", *["relative"] * len(labels), "total"],
            x=["Start", *labels, "Explained + residual"],
            y=[0.0, *values, total],
            connector={"line": {"color": "#94a3b8"}},
            increasing={"marker": {"color": "#0f766e"}},
            decreasing={"marker": {"color": "#b91c1c"}},
            totals={"marker": {"color": "#2563eb"}},
            hovertemplate="%{x}<br>Contribution %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Factor Attribution Waterfall",
        template="plotly_white",
        yaxis={"title": "Arithmetic return contribution", "tickformat": ".0%"},
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 90},
    )
    return figure


def _scorecard_for_catalog_row(
    row: pd.Series,
    *,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    strategy_name = str(row.get("strategy", ""))
    if str(row.get("source", "")) == "baseline":
        if strategy_name in baseline_run.metrics.index:
            return baseline_run.metrics.loc[[strategy_name]].reset_index()
        return pd.DataFrame()
    if experiment_scorecards.empty or "strategy" not in experiment_scorecards:
        return pd.DataFrame()
    return experiment_scorecards[experiment_scorecards["strategy"] == strategy_name].copy()


def _curated_strategy_rank_map(experiment_scorecards: pd.DataFrame) -> dict[str, int]:
    if experiment_scorecards.empty:
        return {}
    curated = select_curated_strategy_shelf(
        rank_strategy_candidates(experiment_scorecards),
        limit=25,
    )
    if curated.empty:
        return {}
    return {
        str(row["strategy"]): int(row["curation_rank"])
        for _, row in curated.iterrows()
        if "strategy" in row and "curation_rank" in row
    }


def _approach_catalog_for_detail(
    bot_config: Any,
    *,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    catalog = build_approach_catalog(bot_config).copy()
    if catalog.empty:
        return catalog
    curated_rank = _curated_strategy_rank_map(experiment_scorecards)
    catalog["curation_rank"] = catalog["strategy"].map(curated_rank)
    catalog["is_curated"] = catalog["curation_rank"].notna()
    catalog["detail_scope"] = catalog.apply(
        lambda row: (
            "baseline"
            if row.get("source") == "baseline"
            else (
                "curated_top_25"
                if bool(row.get("is_curated"))
                else str(row.get("research_status", "experiment_archive"))
            )
        ),
        axis=1,
    )
    catalog["detail_sort"] = catalog.apply(_approach_detail_sort_key, axis=1)
    return catalog.sort_values("detail_sort").reset_index(drop=True)


def _approach_detail_sort_key(row: pd.Series) -> tuple[int, float, str]:
    if bool(row.get("is_curated", False)):
        rank = row.get("curation_rank")
        try:
            return (0, float(rank), str(row.get("strategy", "")))
        except (TypeError, ValueError):
            return (0, 999.0, str(row.get("strategy", "")))
    if row.get("source") == "baseline":
        return (1, 0.0, str(row.get("strategy", "")))
    score = row.get("promotion_score")
    try:
        sort_score = -float(score)
    except (TypeError, ValueError):
        sort_score = 0.0
    return (2, sort_score, str(row.get("strategy", "")))


def _default_reference_catalog_mask(catalog: pd.DataFrame) -> pd.Series:
    if catalog.empty or "strategy" not in catalog:
        return pd.Series(dtype=bool)
    return catalog["strategy"].astype(str).str.lower().isin(DEFAULT_REFERENCE_BASELINE_STRATEGIES)


def _render_approach_detail_workbench(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    st.caption(
        "Canonical drill-down for strategy research: explanation, historical performance, "
        "allocation behavior, robustness diagnostics, and the raw candidate manifest."
    )
    catalog = _approach_catalog_for_detail(
        bot_config,
        experiment_scorecards=experiment_scorecards,
    )
    if catalog.empty:
        st.write("No approaches are available to inspect.")
        return

    scope_options = [
        "Curated shelf + core baselines",
        "Operational candidates + core baselines",
        "All non-pruned research + core baselines",
        "All approaches and archived rows",
    ]
    selected_scope = st.radio(
        "Approach set",
        scope_options,
        horizontal=True,
        key="approach_detail_scope",
    )
    default_reference = _default_reference_catalog_mask(catalog)
    if selected_scope == "Curated shelf + core baselines":
        visible_catalog = catalog[default_reference | catalog["is_curated"]]
    elif selected_scope == "Operational candidates + core baselines":
        visible_catalog = catalog[
            default_reference
            | catalog["research_status"].isin(DEFAULT_DEFAULT_APPROACH_RESEARCH_STATUSES)
        ]
    elif selected_scope == "All non-pruned research + core baselines":
        visible_catalog = catalog[
            default_reference | ~catalog["research_status"].eq("pruned_dead_end")
        ]
    else:
        visible_catalog = catalog
    if visible_catalog.empty:
        visible_catalog = catalog

    selected_label = st.selectbox(
        "Approach to inspect",
        visible_catalog["label"].tolist(),
        key="approach_detail_label",
    )
    approach_row = visible_catalog[visible_catalog["label"] == selected_label].iloc[0]
    approach_strategy = strategy_from_catalog_row(approach_row)
    approach_execution = execution_for_catalog_row(approach_row, bot_config.execution)
    scenario_sizing = scenario_sizing_from_catalog_row(approach_row)
    future_state_model = future_state_model_from_catalog_row(approach_row)
    strategy_drawdown_model = strategy_drawdown_model_from_catalog_row(approach_row)
    decision_sanity = decision_sanity_from_catalog_row(approach_row)

    overview_cols = st.columns(7)
    _helped_metric(overview_cols[0], "Source", str(approach_row["source"]))
    _helped_metric(overview_cols[1], "Category", str(approach_row["family"]))
    _helped_metric(overview_cols[2], "Role", str(approach_row["role"]))
    _helped_metric(
        overview_cols[3],
        "Decision",
        str(approach_row["promotion_decision"]),
        key="promotion_decision",
    )
    _helped_metric(overview_cols[4], "Type", approach_strategy.type)
    _helped_metric(
        overview_cols[5],
        "Research Status",
        str(approach_row.get("research_status", "unclassified")).replace("_", " "),
    )
    curation_value = (
        "not curated"
        if pd.isna(approach_row.get("curation_rank"))
        else f"#{int(float(approach_row['curation_rank']))}"
    )
    _helped_metric(overview_cols[6], "Curated Rank", curation_value)
    if future_state_model is not None:
        st.caption(
            "Future-state model: "
            f"{future_state_model.model} / {future_state_model.feature_set} / "
            f"{future_state_model.horizon_days} trading days"
        )
    if strategy_drawdown_model is not None:
        st.caption(
            "Strategy drawdown model: "
            f"{strategy_drawdown_model.model} / {strategy_drawdown_model.feature_set} / "
            f"{strategy_drawdown_model.horizon_days} trading days / "
            f"{strategy_drawdown_model.future_drawdown_threshold:.0%} forward drawdown label"
        )

    _render_strategy_explanation(
        row=approach_row,
        strategy=approach_strategy,
        bot_config=bot_config,
    )
    if str(approach_row.get("hypothesis", "")):
        with st.expander("Original research hypothesis", expanded=False):
            st.write(str(approach_row["hypothesis"]))

    detail_views = [
        "Summary",
        "Performance + Allocation",
        "Decision Timeline",
        "Performance Over Time",
        "Allocation Behavior",
        "Factor Attribution",
        "Mechanics",
        "Robustness",
        "Manifest / Risk Notes",
    ]
    selected_detail_view = (
        st.pills(
            "Candidate detail view",
            detail_views,
            selection_mode="single",
            default="Summary",
            key="approach_detail_view",
            label_visibility="collapsed",
            width="stretch",
        )
        or "Summary"
    )
    detail_result = _load_detail_result_if_needed(
        selected_detail_view=selected_detail_view,
        row=approach_row,
        strategy=approach_strategy,
        execution=approach_execution,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
        baseline_run=baseline_run,
    )

    if selected_detail_view == "Summary":
        scorecard = _scorecard_for_catalog_row(
            approach_row,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
        )
        if not scorecard.empty:
            summary_columns = [
                "display_name",
                "strategy",
                "promotion_decision",
                "promotion_score",
                "monitoring_readiness_label",
                "growth_utility_tier",
                "cagr",
                "max_drawdown",
                "calmar",
                "walk_forward_positive_rate",
                "left_tail_regime_return",
                "operability_label",
                "material_trade_days_per_year",
            ]
            available_summary_columns = [column for column in summary_columns if column in scorecard]
            st.caption("Selected candidate scorecard")
            _render_metric_dataframe(
                _display_metrics(scorecard[available_summary_columns]),
                hide_index=True,
            )
        st.caption("Risk notes")
        _render_metric_dataframe(
            build_approach_risk_notes(approach_strategy, approach_row),
            hide_index=True,
        )
        st.info(
            "This summary is intentionally light. Choose a detail view above to build the heavier "
            "performance, decision-timeline, allocation, or factor-attribution charts."
        )

    elif selected_detail_view == "Performance + Allocation":
        if detail_result is None:
            st.write("No performance/allocation curve could be reconstructed for this approach.")
        else:
            st.caption(
                "Use this shared-window view to inspect whether the strategy got defensive before "
                "drawdowns, stayed defensive too long, or re-entered risk after repair."
            )
            _render_performance_allocation_context(
                detail_result,
                baseline_run=baseline_run,
                defensive_ticker=approach_strategy.defensive_ticker,
                key_prefix="approach_combined",
            )

    elif selected_detail_view == "Decision Timeline":
        if detail_result is None:
            st.write("No decision timeline could be reconstructed for this approach.")
        else:
            _render_decision_timeline(
                detail_result,
                defensive_ticker=approach_strategy.defensive_ticker,
                key_prefix="approach_detail",
            )

    elif selected_detail_view == "Performance Over Time":
        scorecard = _scorecard_for_catalog_row(
            approach_row,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
        )
        if not scorecard.empty:
            _render_taxable_estimate_summary(scorecard)
            st.caption("Full-history scorecard")
            _render_metric_dataframe(_display_metrics(scorecard), hide_index=True)
        if detail_result is None:
            st.write("No performance curve could be reconstructed for this approach.")
        else:
            _render_approach_performance(
                detail_result,
                baseline_run=baseline_run,
                key_prefix="approach_detail",
            )

    elif selected_detail_view == "Allocation Behavior":
        if detail_result is None:
            st.write("No allocation history could be reconstructed for this approach.")
        else:
            st.caption("Current reconstructed position")
            _render_metric_dataframe(
                _display_metrics(build_latest_weight_frame(detail_result.weights)),
                hide_index=True,
            )
            st.caption("How positions changed over time")
            _render_position_behavior(
                detail_result.weights,
                defensive_ticker=approach_strategy.defensive_ticker,
                key_prefix="approach_detail",
            )

    elif selected_detail_view == "Factor Attribution":
        if detail_result is None:
            st.write("No factor attribution can be reconstructed for this approach.")
        else:
            _render_factor_attribution(detail_result, baseline_run=baseline_run)

    elif selected_detail_view == "Mechanics":
        st.caption("Mechanics")
        _render_metric_dataframe(
            build_approach_mechanics(approach_strategy, bot_config, execution=approach_execution),
            hide_index=True,
        )
        st.caption("Signal steps")
        _render_metric_dataframe(build_approach_steps(approach_strategy), hide_index=True)
        if scenario_sizing is not None:
            st.caption("Scenario sizing layer")
            st.write(
                f"Profile `{scenario_sizing.profile}` scales risk exposure between "
                f"{scenario_sizing.min_multiplier:.0%} and {scenario_sizing.max_multiplier:.0%}; "
                "removed risk budget is routed to the defensive sleeve."
            )
        if future_state_model is not None:
            st.caption("Future-state ML sizing layer")
            bayesian_note = ""
            if future_state_model.model.startswith("bayesian"):
                bayesian_note = (
                    " It uses posterior smoothing, recency-weighted evidence, and shrinkage "
                    "before converting probabilities into sizing."
                )
            st.write(
                "Predicts regime-bucket probabilities instead of prices, then scales risk exposure. "
                f"Model `{future_state_model.model}` uses `{future_state_model.feature_set}` features, "
                f"a {future_state_model.horizon_days}-day target horizon, "
                f"{future_state_model.train_window_days} training days, and "
                f"{future_state_model.min_train_observations} minimum observations."
                f"{bayesian_note}"
            )
        if strategy_drawdown_model is not None:
            st.caption("Strategy-specific ML drawdown guard")
            st.write(
                "Labels the selected strategy's own future drawdown, not future index return. "
                f"Model `{strategy_drawdown_model.model}` uses `{strategy_drawdown_model.feature_set}` features, "
                f"a {strategy_drawdown_model.horizon_days}-day target horizon, "
                f"{strategy_drawdown_model.train_window_days} training days, and an activation threshold of "
                f"{strategy_drawdown_model.activation_probability:.0%}. If active, the risk sleeve scales toward "
                f"{strategy_drawdown_model.stress_multiplier:.0%} with a floor of "
                f"{strategy_drawdown_model.min_multiplier:.0%}."
            )
        if decision_sanity is not None:
            st.caption("Decision-sanity overlay")
            st.write(
                "Caps extra defensive sizing unless market confirmation breaks are broad enough. "
                f"Required confirmation gates: {decision_sanity.required_confirmation_breaks}; "
                f"event/news-only defensive-add cap: {decision_sanity.max_defensive_add:.0%}."
            )

    elif selected_detail_view == "Robustness":
        selected_strategy = str(approach_row.get("strategy", ""))
        rendered_robustness = False
        if not experiment_walk_forward.empty:
            walk_view = experiment_walk_forward[
                experiment_walk_forward["strategy"] == selected_strategy
            ]
            if not walk_view.empty:
                st.caption("Walk-forward diagnostics")
                _render_metric_dataframe(_display_metrics(walk_view), hide_index=True)
                rendered_robustness = True
        if not experiment_regimes.empty:
            regime_view = experiment_regimes[experiment_regimes["strategy"] == selected_strategy]
            if not regime_view.empty:
                st.caption("Named market-transition and left-tail windows")
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
                _render_metric_dataframe(
                    _display_metrics(
                        regime_view[
                            [column for column in regime_columns if column in regime_view.columns]
                        ]
                    ),
                    hide_index=True,
                )
                rendered_robustness = True
        if not rendered_robustness:
            st.write("No robustness artifacts are available for this approach yet.")

    elif selected_detail_view == "Manifest / Risk Notes":
        st.caption("Risk notes")
        _render_metric_dataframe(
            build_approach_risk_notes(approach_strategy, approach_row),
            hide_index=True,
        )
        selected_strategy = str(approach_row.get("strategy", ""))
        manifest_rows = pd.DataFrame()
        if not experiment_candidates.empty:
            manifest_rows = experiment_candidates[
                experiment_candidates["strategy"] == selected_strategy
            ]
        if not manifest_rows.empty:
            st.caption("Candidate manifest")
            st.dataframe(manifest_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No experiment manifest is available for this approach.")


def _render_curated_strategy_shelf(experiment_scorecards: pd.DataFrame) -> None:
    curated = select_curated_strategy_shelf(
        rank_strategy_candidates(experiment_scorecards),
        limit=25,
    )
    if curated.empty:
        return
    st.markdown("**Curated strategy shelf**")
    st.caption(
        "Top operational candidates to inspect or paper-monitor first. The shelf anchors on score, "
        "then forces diversity across strategy families so one historical winner does not crowd out "
        "different failure-mode coverage."
    )
    columns = [
        "curation_rank",
        "display_name",
        "strategy",
        "curation_bucket",
        "curation_reason",
        "iteration",
        "phase",
        "family",
        "role",
        "promotion_decision",
        "promotion_score",
        "confidence_score",
        "confidence_label",
        "deployment_blockers",
        "benchmark_knockout_label",
        "future_state_model",
        "strategy_drawdown_model",
        "research_status",
        "prune_reason",
        "monitoring_readiness_score",
        "monitoring_readiness_label",
        "robustness_score",
        "cagr",
        "max_drawdown",
        "calmar",
        "average_turnover",
        "operability_label",
        "material_trade_days_per_year",
        "risk_cycle_label",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "hypothesis",
    ]
    view = curated[[column for column in columns if column in curated.columns]].rename(
        columns={"family": "category"}
    )
    _render_metric_dataframe(_display_metrics(view), hide_index=True)


def _render_strategy_family_map(
    experiment_scorecards: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    family_map = build_strategy_family_map(experiment_scorecards, experiment_candidates)
    if family_map.empty:
        st.write("No strategy-family map is available yet.")
        return
    active_family_map = family_map[
        ~family_map.get("research_status", pd.Series("", index=family_map.index)).eq(
            "pruned_dead_end"
        )
    ]
    if active_family_map.empty:
        active_family_map = family_map

    st.caption(
        "High-level navigation for the research archive. This groups strategies by what they "
        "actually express: the risk-on sleeve, the defensive sleeve, and the behavior used to "
        "move between them. Use this before choosing paper candidates so the monitor is not "
        "filled with look-alike variants."
    )

    takeaways = strategy_family_takeaways(family_map)
    if takeaways:
        st.markdown("**Aggregate read**")
        st.info("\n\n".join(f"- {takeaway}" for takeaway in takeaways))

    metric_cols = st.columns(5)
    _helped_metric(metric_cols[0], "Mapped", f"{len(family_map):,}")
    _helped_metric(metric_cols[1], "Archetypes", f"{family_map['strategy_archetype'].nunique():,}")
    _helped_metric(metric_cols[2], "Risk Behaviors", f"{family_map['risk_behavior'].nunique():,}")
    _helped_metric(
        metric_cols[3], "Equity Expressions", f"{family_map['equity_expression'].nunique():,}"
    )
    _helped_metric(
        metric_cols[4],
        "Promoted",
        f"{int((family_map['promotion_decision'] == 'promote_candidate').sum()):,}",
        key="promotion_decision",
    )

    archetype_summary = summarize_strategy_archetypes(active_family_map)
    if not archetype_summary.empty:
        st.markdown("**Strategy archetype summary**")
        st.caption(
            "Summary statistics exclude pruned dead-end rows; the detailed map can still show them."
        )
        chart_columns = [
            column
            for column in ["median_cagr", "median_max_drawdown", "median_turnover"]
            if column in archetype_summary
        ]
        if chart_columns:
            chart_frame = archetype_summary.set_index("strategy_archetype")[chart_columns]
            st.bar_chart(chart_frame.dropna(how="all"))
        _render_metric_dataframe(_display_metrics(archetype_summary), hide_index=True)

    risk_matrix = summarize_risk_behavior_matrix(active_family_map)
    if not risk_matrix.empty:
        st.markdown("**Risk-behavior matrix**")
        st.caption(
            "This shows the actual operating families: what they buy for upside, how they defend, "
            "and whether the risk logic is trend exit, cooldown, dip reentry, sector gating, or a benchmark."
        )
        matrix_columns = [
            "risk_behavior",
            "equity_expression",
            "defensive_expression",
            "candidates",
            "promoted",
            "best_strategy",
            "best_score",
            "median_cagr",
            "median_max_drawdown",
            "median_turnover",
            "interpretation",
        ]
        _render_metric_dataframe(
            _display_metrics(
                risk_matrix[[column for column in matrix_columns if column in risk_matrix]]
            ),
            hide_index=True,
        )

    st.markdown("**Strategy map**")
    filter_cols = st.columns(4)
    archetype_options = ["all", *sorted(family_map["strategy_archetype"].dropna().unique())]
    behavior_options = ["all", *sorted(family_map["risk_behavior"].dropna().unique())]
    equity_options = ["all", *sorted(family_map["equity_expression"].dropna().unique())]
    defensive_options = ["all", *sorted(family_map["defensive_expression"].dropna().unique())]
    archetype_filter = filter_cols[0].selectbox(
        "Archetype",
        archetype_options,
        key="family_map_archetype_filter",
    )
    behavior_filter = filter_cols[1].selectbox(
        "Risk behavior",
        behavior_options,
        key="family_map_behavior_filter",
    )
    equity_filter = filter_cols[2].selectbox(
        "Equity expression",
        equity_options,
        key="family_map_equity_filter",
    )
    defensive_filter = filter_cols[3].selectbox(
        "Defense expression",
        defensive_options,
        key="family_map_defensive_filter",
    )

    strategy_view = family_map.copy()
    if archetype_filter != "all":
        strategy_view = strategy_view[strategy_view["strategy_archetype"] == archetype_filter]
    if behavior_filter != "all":
        strategy_view = strategy_view[strategy_view["risk_behavior"] == behavior_filter]
    if equity_filter != "all":
        strategy_view = strategy_view[strategy_view["equity_expression"] == equity_filter]
    if defensive_filter != "all":
        strategy_view = strategy_view[strategy_view["defensive_expression"] == defensive_filter]

    strategy_columns = [
        "iteration",
        "display_name",
        "strategy",
        "strategy_archetype",
        "risk_behavior",
        "equity_expression",
        "defensive_expression",
        "strategy_type",
        "defensive_ticker",
        "ticker_count",
        "primary_tickers",
        "strategy_drawdown_model",
        "research_status",
        "prune_reason",
        "promotion_decision",
        "promotion_score",
        "cagr",
        "max_drawdown",
        "calmar",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "risk_read",
        "hypothesis",
    ]
    _render_metric_dataframe(
        _display_metrics(
            strategy_view[[column for column in strategy_columns if column in strategy_view]]
        ),
        hide_index=True,
    )

    family_clusters = summarize_family_clusters(family_map)
    if not family_clusters.empty:
        with st.expander("Family cluster details", expanded=False):
            _render_metric_dataframe(_display_metrics(family_clusters), hide_index=True)


def _render_outcome_frontier(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    st.markdown("**Outcome Frontier**")
    st.caption(
        "Growth-constrained research view: this asks whether extra CAGR is worth the additional "
        "drawdown for a 15-year accumulation account. The soft drawdown band starts at "
        f"{abs(DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT):.0%}; the hard review band starts at "
        f"{abs(DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT):.0%}."
    )
    if experiment_scorecards.empty:
        st.write("No experiment scorecards are available for outcome-frontier analysis yet.")
        return

    frame = add_outcome_frontier_flags(enrich_strategy_outcome_utility(experiment_scorecards))
    if "research_status" in frame:
        active = frame[~frame["research_status"].astype(str).eq("pruned_dead_end")].copy()
        if not active.empty:
            frame = active
    required_columns = {"cagr", "max_drawdown", "growth_constrained_utility_score"}
    if not required_columns.issubset(frame.columns):
        st.write(
            "Outcome utility fields are not available yet. Run the daily update stack or migrate experiments."
        )
        return

    plot_frame = frame.dropna(subset=["cagr", "max_drawdown"]).copy()
    if plot_frame.empty:
        st.write("No CAGR/drawdown rows are available for outcome-frontier analysis.")
        return

    plot_frame = plot_frame.sort_values("growth_constrained_utility_score", ascending=False)
    fig = go.Figure()
    for tier, tier_frame in plot_frame.groupby("growth_utility_tier", dropna=False):
        fig.add_trace(
            go.Scatter(
                x=tier_frame["max_drawdown"],
                y=tier_frame["cagr"],
                mode="markers",
                name=str(tier).replace("_", " ").title(),
                marker={
                    "size": _outcome_marker_sizes(tier_frame),
                    "opacity": 0.76,
                    "line": {"width": 1, "color": "#0f172a"},
                },
                text=tier_frame.get("display_name", tier_frame.get("strategy", "")),
                customdata=tier_frame[
                    [
                        column
                        for column in [
                            "strategy",
                            "growth_constrained_utility_score",
                            f"terminal_wealth_with_contributions_{DEFAULT_OUTCOME_HORIZON_YEARS}y",
                            "drawdown_recovery_return",
                            "monitoring_readiness_label",
                        ]
                        if column in tier_frame
                    ]
                ],
                hovertemplate=(
                    "%{text}<br>Max drawdown %{x:.1%}<br>CAGR %{y:.1%}"
                    "<br>Utility %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
    pareto = plot_frame[plot_frame.get("is_growth_pareto_efficient", False).astype(bool)]
    if not pareto.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto["max_drawdown"],
                y=pareto["cagr"],
                mode="markers",
                name="Pareto frontier",
                marker={
                    "symbol": "diamond-open",
                    "size": 16,
                    "line": {"width": 2, "color": "#ef4444"},
                },
                text=pareto.get("display_name", pareto.get("strategy", "")),
                customdata=pareto[["strategy"]],
                hovertemplate="%{text}<br>Pareto efficient<extra></extra>",
            )
        )
    _add_outcome_drawdown_band_trace(
        fig,
        y_values=plot_frame["cagr"],
        x_value=DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
        name="Soft drawdown band",
        color="#f59e0b",
        dash="dash",
        detail=(
            "Drawdowns more negative than this enter the soft penalty band; "
            "high-growth candidates can remain eligible."
        ),
    )
    _add_outcome_drawdown_band_trace(
        fig,
        y_values=plot_frame["cagr"],
        x_value=DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
        name="Hard review band",
        color="#ef4444",
        dash="dot",
        detail="Drawdowns at or beyond this line trigger hard review/rejection behavior.",
    )
    fig.update_layout(
        height=520,
        xaxis_title="Max drawdown",
        yaxis_title="CAGR",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend_title="Growth utility tier",
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key=_OUTCOME_FRONTIER_PLOT_KEY,
        on_select="rerun",
        selection_mode="points",
    )

    selected_options = _outcome_select_options(plot_frame)
    selected_from_chart = _plotly_selected_strategy(selection)
    if selected_from_chart:
        st.session_state[_OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY] = selected_from_chart

    option_labels = selected_options["label"].tolist()
    selected_index = _outcome_label_index_for_strategy(
        selected_options,
        st.session_state.get(_OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY),
    )
    selected_label_from_state = option_labels[selected_index]
    if (
        selected_from_chart
        or st.session_state.get(_OUTCOME_FRONTIER_SELECTBOX_KEY) not in option_labels
    ):
        st.session_state[_OUTCOME_FRONTIER_SELECTBOX_KEY] = selected_label_from_state

    selected_label = st.selectbox(
        "Outcome strategy to inspect",
        option_labels,
        key=_OUTCOME_FRONTIER_SELECTBOX_KEY,
    )
    selected_row = selected_options[selected_options["label"] == selected_label].iloc[0]
    selected_strategy = str(selected_row["strategy"])
    st.session_state[_OUTCOME_FRONTIER_SELECTED_STRATEGY_KEY] = selected_strategy
    selected_scorecard = plot_frame[plot_frame["strategy"].astype(str) == selected_strategy].iloc[0]
    _render_outcome_decision_cards(
        selected_scorecard,
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
        peer_frame=plot_frame,
    )

    comparison_columns = [
        "display_name",
        "strategy",
        "growth_constrained_utility_score",
        "growth_utility_tier",
        f"terminal_wealth_with_contributions_{DEFAULT_OUTCOME_HORIZON_YEARS}y",
        "wealth_multiple_vs_spy",
        "wealth_multiple_vs_qqq",
        "cagr",
        "max_drawdown",
        "drawdown_recovery_return",
        "walk_forward_positive_rate",
        "worst_1y_cagr",
        "worst_3y_cagr",
        "left_tail_regime_return",
        "monitoring_readiness_label",
        "operability_label",
    ]
    st.caption("Top outcome-utility candidates")
    _render_metric_dataframe(
        _display_metrics(
            plot_frame[[column for column in comparison_columns if column in plot_frame]].head(20)
        ),
        hide_index=True,
    )


def _outcome_marker_sizes(frame: pd.DataFrame) -> pd.Series:
    wealth_column = f"terminal_wealth_with_contributions_{DEFAULT_OUTCOME_HORIZON_YEARS}y"
    if wealth_column not in frame:
        return pd.Series(12.0, index=frame.index)
    wealth = pd.to_numeric(frame[wealth_column], errors="coerce")
    if wealth.notna().sum() <= 1 or float(wealth.max()) == float(wealth.min()):
        return pd.Series(13.0, index=frame.index)
    scaled = (wealth - wealth.min()) / max(float(wealth.max() - wealth.min()), 1e-12)
    return 9.0 + 18.0 * scaled.fillna(0.0)


def _add_outcome_drawdown_band_trace(
    fig: go.Figure,
    *,
    y_values: pd.Series,
    x_value: float,
    name: str,
    color: str,
    dash: str,
    detail: str,
) -> None:
    clean = pd.to_numeric(y_values, errors="coerce").dropna()
    if clean.empty:
        y_min, y_max = 0.0, 1.0
    else:
        span = max(float(clean.max() - clean.min()), 0.01)
        y_min = float(clean.min() - span * 0.08)
        y_max = float(clean.max() + span * 0.08)
    fig.add_trace(
        go.Scatter(
            x=[x_value, x_value],
            y=[y_min, y_max],
            mode="lines",
            name=f"{name}: {abs(x_value):.0%}",
            line={"color": color, "dash": dash, "width": 2},
            hovertemplate=(
                f"<b>{name}</b><br>Max drawdown threshold: {x_value:.0%}<br>{detail}"
                "<extra></extra>"
            ),
            showlegend=True,
            customdata=[[""], [""]],
        )
    )


def _outcome_select_options(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["label"] = output.apply(
        lambda row: (
            f"{row.get('display_name', row.get('strategy', 'strategy'))} | "
            f"utility {_format_decimal(row.get('growth_constrained_utility_score'))} | "
            f"CAGR {_format_percent(row.get('cagr'))} | "
            f"DD {_format_percent(row.get('max_drawdown'))}"
        ),
        axis=1,
    )
    return output.sort_values("growth_constrained_utility_score", ascending=False)


def _outcome_label_index_for_strategy(options: pd.DataFrame, strategy: Any) -> int:
    if strategy is None or "strategy" not in options:
        return 0
    matches = options.index[options["strategy"].astype(str).eq(str(strategy))].tolist()
    if not matches:
        return 0
    return int(options.index.get_loc(matches[0]))


def _plotly_selected_strategy(selection_event: Any) -> str | None:
    """Return the first selected strategy id from a Streamlit Plotly selection event."""

    selection = _get_selection_field(selection_event, "selection")
    points = _get_selection_field(selection, "points")
    if not points:
        return None

    first_point = points[0]
    customdata = _get_selection_field(first_point, "customdata")
    if customdata is None:
        return None
    if isinstance(customdata, (list, tuple)):
        if not customdata:
            return None
        return _non_empty_strategy_id(customdata[0])
    if not isinstance(customdata, (str, bytes)) and hasattr(customdata, "__len__"):
        if len(customdata) == 0:
            return None
        return _non_empty_strategy_id(customdata[0])
    return _non_empty_strategy_id(customdata)


def _non_empty_strategy_id(value: object) -> str | None:
    strategy_id = str(value).strip()
    return strategy_id or None


def _get_selection_field(payload: Any, field: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(field)
    return getattr(payload, field, None)


def _render_outcome_decision_cards(
    row: pd.Series,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    peer_frame: pd.DataFrame,
) -> None:
    wealth_column = f"terminal_wealth_with_contributions_{DEFAULT_OUTCOME_HORIZON_YEARS}y"
    wealth = _safe_float(row.get(wealth_column))
    extra_spy = _extra_wealth_from_multiple(wealth, row.get("wealth_multiple_vs_spy"))
    extra_qqq = _extra_wealth_from_multiple(wealth, row.get("wealth_multiple_vs_qqq"))
    result = _selected_outcome_result(
        str(row.get("strategy", "")),
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    underwater_rate = _time_underwater_rate(result)
    pain_index = _ulcer_index(result)

    cols = st.columns(5)
    _helped_metric(cols[0], "15Y Wealth", _format_currency(wealth))
    _helped_metric(cols[1], "Extra vs SPY", _format_currency(extra_spy))
    _helped_metric(cols[2], "Extra vs QQQ", _format_currency(extra_qqq))
    _helped_metric(cols[3], "Recovery Needed", _format_percent(row.get("drawdown_recovery_return")))
    _helped_metric(cols[4], "Ulcer Index", _format_percent(pain_index))

    st.info(_outcome_decision_helper(row, extra_spy=extra_spy, extra_qqq=extra_qqq))

    benchmark_values = _outcome_benchmark_metric_values(baseline_run, experiment_scorecards)
    context = _outcome_selected_benchmark_context(
        row,
        selected_ulcer_index=pain_index,
        selected_underwater_rate=underwater_rate,
        benchmark_values=benchmark_values,
    )
    _render_metric_dataframe(
        context,
        hide_index=True,
        column_help={
            "selected": "Selected strategy value.",
            "spy": "SPY buy-and-hold value calculated from the benchmark curve when available.",
            "qqq": "QQQ buy-and-hold value calculated from the benchmark curve when available.",
            "note": "Why a benchmark value may be missing.",
        },
    )
    st.caption("Selected metric context across displayed outcome candidates")
    _render_metric_dataframe(
        _display_metrics(
            _outcome_metric_peer_context(
                row,
                selected_result=result,
                selected_ulcer_index=pain_index,
                selected_underwater_rate=underwater_rate,
                peer_frame=peer_frame,
                bot_config=bot_config,
                baseline_run=baseline_run,
                experiment_scorecards=experiment_scorecards,
                benchmark_values=benchmark_values,
            )
        ),
        hide_index=True,
        column_help={
            "peer_percentile": (
                "Percent of displayed outcome candidates the selected strategy beats for this metric. "
                "Higher is better after adjusting for metric direction."
            ),
            "peer_min": "Lowest raw value among displayed outcome candidates.",
            "peer_median": "Median raw value among displayed outcome candidates.",
            "peer_max": "Highest raw value among displayed outcome candidates.",
        },
    )


def _outcome_selected_benchmark_context(
    row: pd.Series,
    *,
    selected_ulcer_index: float | None,
    selected_underwater_rate: float | None,
    benchmark_values: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    metric_specs = _outcome_metric_specs()
    rows: list[dict[str, object]] = []
    for spec in metric_specs:
        metric = str(spec["metric"])
        metric_key = str(spec["key"])
        kind = str(spec["kind"])
        if metric_key == "ulcer_index":
            selected = selected_ulcer_index
        elif metric_key == "days_below_prior_peak":
            selected = selected_underwater_rate
        elif metric_key == "extra_wealth_vs_spy":
            selected = _extra_wealth_from_multiple(
                _safe_float(row.get(_outcome_wealth_column())),
                row.get("wealth_multiple_vs_spy"),
            )
        elif metric_key == "extra_wealth_vs_qqq":
            selected = _extra_wealth_from_multiple(
                _safe_float(row.get(_outcome_wealth_column())),
                row.get("wealth_multiple_vs_qqq"),
            )
        else:
            selected = _safe_float(row.get(metric_key))

        spy_value = _benchmark_display_value(
            metric_key,
            benchmark_values=benchmark_values,
            benchmark="SPY",
        )
        qqq_value = _benchmark_display_value(
            metric_key,
            benchmark_values=benchmark_values,
            benchmark="QQQ",
        )
        rows.append(
            {
                "metric": metric,
                "definition": _metric_plain_english(metric),
                "selected": _format_outcome_value(selected, kind),
                "spy": _format_outcome_value(spy_value, kind),
                "qqq": _format_outcome_value(qqq_value, kind),
                "note": _outcome_benchmark_note(metric_key, spy_value, qqq_value),
            }
        )
    return pd.DataFrame(rows)


def _outcome_metric_peer_context(
    row: pd.Series,
    *,
    selected_result: BacktestResult | None,
    selected_ulcer_index: float | None,
    selected_underwater_rate: float | None,
    peer_frame: pd.DataFrame,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    benchmark_values: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    working = _outcome_peer_distribution_frame(
        row,
        selected_result=selected_result,
        selected_ulcer_index=selected_ulcer_index,
        selected_underwater_rate=selected_underwater_rate,
        peer_frame=peer_frame,
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    rows: list[dict[str, object]] = []
    for spec in _outcome_metric_specs():
        metric = str(spec["metric"])
        metric_key = str(spec["key"])
        kind = str(spec["kind"])
        lower_is_better = bool(spec["lower_is_better"])
        selected = _safe_float(working.loc[working["is_selected"], metric_key].iloc[0])
        values = pd.to_numeric(working[metric_key], errors="coerce").dropna()
        if values.empty or selected is None:
            peer_min = peer_median = peer_max = peer_percentile = None
        else:
            peer_min = float(values.min())
            peer_median = float(values.median())
            peer_max = float(values.max())
            peer_percentile = _peer_percentile(
                selected,
                values,
                lower_is_better=lower_is_better,
            )
        rows.append(
            {
                "metric": metric,
                "selected": _format_outcome_value(selected, kind),
                "spy": _format_outcome_value(
                    _benchmark_display_value(
                        metric_key,
                        benchmark_values=benchmark_values,
                        benchmark="SPY",
                    ),
                    kind,
                ),
                "qqq": _format_outcome_value(
                    _benchmark_display_value(
                        metric_key,
                        benchmark_values=benchmark_values,
                        benchmark="QQQ",
                    ),
                    kind,
                ),
                "peer_min": _format_outcome_value(peer_min, kind),
                "peer_median": _format_outcome_value(peer_median, kind),
                "peer_max": _format_outcome_value(peer_max, kind),
                "peer_percentile": _format_percent(peer_percentile),
                "peer_count": int(values.shape[0]),
                "how_to_read": _metric_how_to_read(metric),
            }
        )
    return pd.DataFrame(rows)


def _outcome_peer_distribution_frame(
    row: pd.Series,
    *,
    selected_result: BacktestResult | None,
    selected_ulcer_index: float | None,
    selected_underwater_rate: float | None,
    peer_frame: pd.DataFrame,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    working = peer_frame.copy()
    wealth_column = _outcome_wealth_column()
    if wealth_column in working:
        wealth = pd.to_numeric(working[wealth_column], errors="coerce")
        spy_multiple = pd.to_numeric(
            working["wealth_multiple_vs_spy"]
            if "wealth_multiple_vs_spy" in working
            else pd.Series(pd.NA, index=working.index),
            errors="coerce",
        )
        qqq_multiple = pd.to_numeric(
            working["wealth_multiple_vs_qqq"]
            if "wealth_multiple_vs_qqq" in working
            else pd.Series(pd.NA, index=working.index),
            errors="coerce",
        )
        working["extra_wealth_vs_spy"] = wealth - wealth / spy_multiple.replace(0.0, pd.NA)
        working["extra_wealth_vs_qqq"] = wealth - wealth / qqq_multiple.replace(0.0, pd.NA)
    selected_strategy = str(row.get("strategy", ""))
    working["is_selected"] = working["strategy"].astype(str).eq(selected_strategy)
    curve_metrics = _outcome_peer_curve_metrics(
        working,
        selected_strategy=selected_strategy,
        selected_result=selected_result,
        selected_ulcer_index=selected_ulcer_index,
        selected_underwater_rate=selected_underwater_rate,
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    if not curve_metrics.empty:
        working = working.merge(curve_metrics, on="strategy", how="left")
    else:
        working["ulcer_index"] = pd.NA
        working["days_below_prior_peak"] = pd.NA
    if working["is_selected"].any():
        selected_index = working.index[working["is_selected"]][0]
        working.loc[selected_index, "ulcer_index"] = selected_ulcer_index
        working.loc[selected_index, "days_below_prior_peak"] = selected_underwater_rate
    return working


def _outcome_peer_curve_metrics(
    frame: pd.DataFrame,
    *,
    selected_strategy: str,
    selected_result: BacktestResult | None,
    selected_ulcer_index: float | None,
    selected_underwater_rate: float | None,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    if "strategy" not in frame:
        return pd.DataFrame()
    candidate_strategies = (
        frame.sort_values("growth_constrained_utility_score", ascending=False)["strategy"]
        .astype(str)
        .drop_duplicates()
        .head(DEFAULT_OUTCOME_PEER_CURVE_METRIC_LIMIT)
        .tolist()
    )
    if selected_strategy and selected_strategy not in candidate_strategies:
        candidate_strategies.append(selected_strategy)
    catalog = _approach_catalog_for_detail(bot_config, experiment_scorecards=experiment_scorecards)
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy_name in candidate_strategies:
        if strategy_name == selected_strategy and selected_result is not None:
            result = selected_result
            metric_ulcer = selected_ulcer_index
            metric_underwater = selected_underwater_rate
        else:
            result = _result_for_catalog_strategy(
                strategy_name,
                catalog=catalog,
                bot_config=bot_config,
                baseline_run=baseline_run,
            )
            metric_ulcer = _ulcer_index(result)
            metric_underwater = _time_underwater_rate(result)
        if result is None:
            continue
        rows.append(
            {
                "strategy": strategy_name,
                "ulcer_index": metric_ulcer,
                "days_below_prior_peak": metric_underwater,
            }
        )
    return pd.DataFrame(rows)


def _result_for_catalog_strategy(
    strategy_name: str,
    *,
    catalog: pd.DataFrame,
    bot_config: Any,
    baseline_run: BaselineRun,
) -> BacktestResult | None:
    matches = catalog[catalog["strategy"].astype(str) == strategy_name]
    if matches.empty:
        return None
    row = matches.iloc[0]
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
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return result


def _outcome_benchmark_metric_values(
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for label, strategy_name in {"SPY": "buy_hold_spy", "QQQ": "buy_hold_qqq"}.items():
        result = baseline_run.results.get(strategy_name)
        metrics: dict[str, float | None] = {
            "cagr": _baseline_metric_value(baseline_run.metrics, strategy_name, "cagr"),
            "max_drawdown": _baseline_metric_value(
                baseline_run.metrics,
                strategy_name,
                "max_drawdown",
            ),
            "worst_1y_cagr": _baseline_window_value(
                baseline_run.window_summary,
                strategy_name,
                "1y",
                "worst_cagr",
            ),
            "worst_3y_cagr": _baseline_window_value(
                baseline_run.window_summary,
                strategy_name,
                "3y",
                "worst_cagr",
            ),
            "left_tail_regime_return": _scorecard_strategy_value(
                experiment_scorecards,
                strategy_name,
                "left_tail_regime_return",
            ),
        }
        if result is not None:
            metrics["ulcer_index"] = _ulcer_index(result)
            metrics["days_below_prior_peak"] = _time_underwater_rate(result)
        else:
            metrics["ulcer_index"] = None
            metrics["days_below_prior_peak"] = None
        metrics["drawdown_recovery_return"] = _drawdown_recovery_value(metrics["max_drawdown"])
        metrics[_outcome_wealth_column()] = _terminal_wealth_value(metrics["cagr"])
        output[label] = metrics
    return output


def _baseline_metric_value(metrics: pd.DataFrame, strategy_name: str, column: str) -> float | None:
    if metrics.empty or column not in metrics:
        return None
    if strategy_name not in metrics.index:
        return None
    return _safe_float(metrics.loc[strategy_name, column])


def _baseline_window_value(
    window_summary: pd.DataFrame,
    strategy_name: str,
    window: str,
    column: str,
) -> float | None:
    if window_summary.empty or column not in window_summary:
        return None
    frame = window_summary.reset_index()
    if "name" not in frame or "window" not in frame:
        return None
    matches = frame[
        frame["name"].astype(str).eq(strategy_name) & frame["window"].astype(str).eq(window)
    ]
    if matches.empty:
        return None
    return _safe_float(matches.iloc[0][column])


def _scorecard_strategy_value(
    scorecards: pd.DataFrame,
    strategy_name: str,
    column: str,
) -> float | None:
    if scorecards.empty or column not in scorecards:
        return None
    if "strategy" in scorecards:
        matches = scorecards[scorecards["strategy"].astype(str).eq(strategy_name)]
        if not matches.empty:
            return _safe_float(matches.iloc[0][column])
    if strategy_name in scorecards.index:
        return _safe_float(scorecards.loc[strategy_name, column])
    return None


def _benchmark_display_value(
    metric_key: str,
    *,
    benchmark_values: dict[str, dict[str, float | None]],
    benchmark: str,
) -> float | None:
    if metric_key == "extra_wealth_vs_spy":
        return _benchmark_extra_wealth(benchmark_values, benchmark=benchmark, against="SPY")
    if metric_key == "extra_wealth_vs_qqq":
        return _benchmark_extra_wealth(benchmark_values, benchmark=benchmark, against="QQQ")
    return benchmark_values.get(benchmark, {}).get(metric_key)


def _benchmark_extra_wealth(
    benchmark_values: dict[str, dict[str, float | None]],
    *,
    benchmark: str,
    against: str,
) -> float | None:
    benchmark_wealth = benchmark_values.get(benchmark, {}).get(_outcome_wealth_column())
    against_wealth = benchmark_values.get(against, {}).get(_outcome_wealth_column())
    if benchmark_wealth is None or against_wealth is None:
        return None
    return benchmark_wealth - against_wealth


def _outcome_metric_specs() -> list[dict[str, object]]:
    wealth_column = _outcome_wealth_column()
    return [
        {"metric": "15Y Wealth", "key": wealth_column, "kind": "currency", "lower_is_better": False},
        {
            "metric": "Extra vs SPY",
            "key": "extra_wealth_vs_spy",
            "kind": "currency",
            "lower_is_better": False,
        },
        {
            "metric": "Extra vs QQQ",
            "key": "extra_wealth_vs_qqq",
            "kind": "currency",
            "lower_is_better": False,
        },
        {"metric": "CAGR", "key": "cagr", "kind": "percent", "lower_is_better": False},
        {
            "metric": "Max Drawdown",
            "key": "max_drawdown",
            "kind": "percent",
            "lower_is_better": False,
        },
        {
            "metric": "Recovery Needed",
            "key": "drawdown_recovery_return",
            "kind": "percent",
            "lower_is_better": True,
        },
        {
            "metric": "Worst 1Y CAGR",
            "key": "worst_1y_cagr",
            "kind": "percent",
            "lower_is_better": False,
        },
        {
            "metric": "Worst 3Y CAGR",
            "key": "worst_3y_cagr",
            "kind": "percent",
            "lower_is_better": False,
        },
        {
            "metric": "Left-Tail Regime Return",
            "key": "left_tail_regime_return",
            "kind": "percent",
            "lower_is_better": False,
        },
        {
            "metric": "Ulcer Index",
            "key": "ulcer_index",
            "kind": "percent",
            "lower_is_better": True,
        },
        {
            "metric": "Days Below Prior Peak",
            "key": "days_below_prior_peak",
            "kind": "percent",
            "lower_is_better": True,
        },
    ]


def _outcome_wealth_column() -> str:
    return f"terminal_wealth_with_contributions_{DEFAULT_OUTCOME_HORIZON_YEARS}y"


def _terminal_wealth_value(cagr: float | None) -> float | None:
    if cagr is None:
        return None
    wealth = terminal_wealth_from_cagr(
        cagr,
        years=DEFAULT_OUTCOME_HORIZON_YEARS,
        starting_account_value=DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
        annual_contribution=DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    )
    if wealth.empty:
        return None
    return _safe_float(wealth.iloc[0])


def _drawdown_recovery_value(max_drawdown: float | None) -> float | None:
    if max_drawdown is None:
        return None
    recovery = drawdown_recovery_return(max_drawdown)
    if recovery.empty:
        return None
    return _safe_float(recovery.iloc[0])


def _peer_percentile(
    selected: float,
    values: pd.Series,
    *,
    lower_is_better: bool,
) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    if lower_is_better:
        return float((clean >= selected).mean())
    return float((clean <= selected).mean())


def _format_outcome_value(value: float | None, kind: str) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "not available"
    if kind == "currency":
        return _format_currency(numeric)
    if kind == "percent":
        return _format_percent(numeric)
    return _format_decimal(numeric)


def _metric_plain_english(metric_name: str) -> str:
    detail = metric_detail(metric_name)
    return detail.plain_english if detail else ""


def _metric_how_to_read(metric_name: str) -> str:
    detail = metric_detail(metric_name)
    return detail.how_to_read if detail else ""


def _outcome_benchmark_note(
    metric_key: str,
    spy_value: float | None,
    qqq_value: float | None,
) -> str:
    if spy_value is not None and qqq_value is not None:
        return ""
    if metric_key == "left_tail_regime_return":
        return "Regime metric is available only when benchmark regime summaries are present in the experiment artifacts."
    return "Benchmark curve or derived benchmark metric is not available in the loaded snapshot."


def _selected_outcome_result(
    strategy_name: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> BacktestResult | None:
    catalog = _approach_catalog_for_detail(bot_config, experiment_scorecards=experiment_scorecards)
    if catalog.empty or "strategy" not in catalog:
        return None
    matches = catalog[catalog["strategy"].astype(str) == strategy_name]
    if matches.empty:
        return None
    row = matches.iloc[0]
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
    return result


def _time_underwater_rate(result: BacktestResult | None) -> float | None:
    if result is None or result.equity.empty:
        return None
    strategy_drawdown = drawdown(result.equity)
    return float((strategy_drawdown < 0.0).mean())


def _ulcer_index(result: BacktestResult | None) -> float | None:
    if result is None or result.equity.empty:
        return None
    return ulcer_index(result.equity)


def _extra_wealth_from_multiple(wealth: float | None, multiple: object) -> float | None:
    wealth_value = _safe_float(wealth)
    multiple_value = _safe_float(multiple)
    if wealth_value is None or multiple_value is None or multiple_value <= 0.0:
        return None
    return wealth_value - wealth_value / multiple_value


def _outcome_decision_helper(
    row: pd.Series, *, extra_spy: float | None, extra_qqq: float | None
) -> str:
    max_drawdown = _safe_float(row.get("max_drawdown"))
    cagr = _safe_float(row.get("cagr"))
    utility_tier = str(row.get("growth_utility_tier", "")).replace("_", " ")
    soft = abs(DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT)
    hard = abs(DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT)
    drawdown_depth = abs(max_drawdown or 0.0)
    band = "inside the preferred band"
    if drawdown_depth >= hard:
        band = "outside the hard drawdown band"
    elif drawdown_depth > soft:
        band = "inside the soft penalty band"
    support = (
        "supported"
        if _safe_float(row.get("walk_forward_positive_rate"))
        and _safe_float(row.get("walk_forward_positive_rate")) >= 0.65
        else "not fully supported"
    )
    return (
        f"Outcome read: this strategy compounds at {_format_percent(cagr)} with max drawdown "
        f"{_format_percent(max_drawdown)}, which is {band}. At the configured 15-year accumulation "
        f"assumptions it is {_format_currency(extra_spy)} versus SPY and {_format_currency(extra_qqq)} "
        f"versus QQQ. The growth utility tier is {utility_tier}; walk-forward evidence is {support}."
    )


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(cast(object, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _render_experiment_monitor(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
    decision_sanity_impacts: pd.DataFrame,
) -> None:
    st.subheader("Experiment Monitor")
    if experiment_scorecards.empty:
        st.write(
            "No experiment scorecards have been saved yet. Baseline approaches are still inspectable below."
        )
        _render_approach_detail_workbench(
            bot_config=bot_config,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
            experiment_regimes=experiment_regimes,
            experiment_walk_forward=experiment_walk_forward,
            experiment_candidates=experiment_candidates,
        )
        return

    latest_iteration = latest_experiment_iteration(experiment_scorecards)
    promoted_count = int((experiment_scorecards["promotion_decision"] == "promote_candidate").sum())
    rejected_tail_count = int(
        experiment_scorecards["promotion_decision"]
        .isin(["reject_left_tail", "reject_regime_fragility", "reject_walk_forward_fragility"])
        .sum()
    )
    pruned_count = int(
        experiment_scorecards.get(
            "research_status", pd.Series("", index=experiment_scorecards.index)
        )
        .eq("pruned_dead_end")
        .sum()
    )
    latest_label = f"{latest_iteration:02d}" if latest_iteration is not None else "n/a"
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    _helped_metric(col_a, "Iterations", latest_label)
    _helped_metric(col_b, "Candidates", f"{len(experiment_scorecards):,}")
    _helped_metric(col_c, "Promoted", f"{promoted_count:,}", key="promotion_decision")
    _helped_metric(col_d, "Risk rejects", f"{rejected_tail_count:,}", key="promotion_decision")
    _helped_metric(col_e, "Pruned", f"{pruned_count:,}")

    st.caption(
        "Research flow: the upper area summarizes patterns across experiments; the lower area is "
        "reserved for a deep dive into one selected candidate and its internal strategy tabs."
    )

    st.markdown("**Aggregated Insights Across Experiments**")
    st.caption(
        "Cross-experiment views: rankings, curated lists, frontier tradeoffs, signal/family "
        "patterns, account impacts, and validation/QC. Use this area to decide which candidate "
        "deserves inspection below."
    )
    aggregate_views = [
        "Overview",
        "Leaderboard",
        "Curated Shelf",
        "Outcome Frontier",
        "Signal Evidence",
        "Family Map",
        "Taxable Impact",
        "Validation / QC",
        "Manifests",
    ]
    aggregate_view = (
        st.pills(
            "Aggregate insight view",
            aggregate_views,
            selection_mode="single",
            default="Overview",
            key="experiment_aggregate_view",
            label_visibility="collapsed",
            width="stretch",
        )
        or "Overview"
    )

    if aggregate_view == "Overview":
        experiment_summary = summarize_experiment_history(experiment_scorecards)
        _render_metric_dataframe(_display_metrics(experiment_summary))

        family_summary = summarize_experiment_families(experiment_scorecards)
        if not family_summary.empty:
            st.caption("Research-category leaderboard")
            _render_metric_dataframe(
                _display_metrics(family_summary.rename(columns={"family": "category"}))
            )

        operating_systems = summarize_experiment_operating_systems(experiment_scorecards)
        if not operating_systems.empty:
            st.caption("Best current operating-system candidates by category")
            _render_metric_dataframe(
                _display_metrics(operating_systems.rename(columns={"family": "category"}))
            )

    elif aggregate_view == "Leaderboard":
        st.caption(
            "Aggregate comparison table. Start here when you want to filter the whole experiment "
            "backlog by iteration, status, family, role, and promotion result."
        )
        experiment_iterations = sorted(experiment_scorecards["iteration"].unique())
        default_iterations = experiment_iterations[-10:]
        selected_iterations = st.multiselect(
            "Experiment iterations",
            experiment_iterations,
            default=default_iterations,
            key="experiment_leaderboard_iterations",
        )
        filter_col_a, filter_col_b, filter_col_c, filter_col_d, filter_col_e = st.columns(5)
        decision_options = ["all", *sorted(experiment_scorecards["promotion_decision"].unique())]
        role_options = ["all", *sorted(experiment_scorecards["role"].unique())]
        phase_options = ["all", *sorted(experiment_scorecards["phase"].dropna().unique())]
        family_options = ["all", *sorted(experiment_scorecards["family"].dropna().unique())]
        status_options = [
            "default surface",
            "active research",
            "all",
            *sorted(
                experiment_scorecards.get("research_status", pd.Series(dtype=str)).dropna().unique()
            ),
        ]
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
            "Research category",
            family_options,
            key="experiment_family_filter",
        )
        status_filter = filter_col_e.selectbox(
            "Research status",
            status_options,
            key="experiment_status_filter",
        )

        experiment_view = experiment_scorecards[
            experiment_scorecards["iteration"].isin(selected_iterations)
        ]
        if status_filter == "default surface":
            experiment_view = experiment_view[
                experiment_view.get("research_status", pd.Series("", index=experiment_view.index))
                .astype(str)
                .isin(DEFAULT_DEFAULT_APPROACH_RESEARCH_STATUSES)
            ]
        elif status_filter == "active research":
            experiment_view = experiment_view[
                ~experiment_view.get(
                    "research_status", pd.Series("", index=experiment_view.index)
                ).eq("pruned_dead_end")
            ]
        elif status_filter != "all":
            experiment_view = experiment_view[
                experiment_view.get("research_status", pd.Series("", index=experiment_view.index))
                == status_filter
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
            "display_name",
            "strategy",
            "phase",
            "family",
            "role",
            "scenario_sizing",
            "future_state_model",
            "strategy_drawdown_model",
            "decision_sanity",
            "research_status",
            "prune_reason",
            "promotion_decision",
            "promotion_score",
            "confidence_score",
            "confidence_label",
            "deployment_blockers",
            "benchmark_knockout_label",
            "monitoring_readiness_score",
            "monitoring_readiness_label",
            "robustness_score",
            "cagr",
            "max_drawdown",
            "calmar",
            "operability_label",
            "material_trade_days_per_year",
            "risk_cycle_label",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
            "left_tail_regime_cagr",
            "hypothesis",
        ]
        leaderboard_view = experiment_view[
            [column for column in leaderboard_columns if column in experiment_view.columns]
        ].rename(columns={"family": "category"})
        _render_metric_dataframe(_display_metrics(leaderboard_view))

    elif aggregate_view == "Curated Shelf":
        _render_curated_strategy_shelf(experiment_scorecards)

    elif aggregate_view == "Outcome Frontier":
        _render_outcome_frontier(
            bot_config=bot_config,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
            experiment_candidates=experiment_candidates,
        )

    elif aggregate_view == "Signal Evidence":
        _render_signal_evidence(experiment_scorecards, experiment_candidates)

    elif aggregate_view == "Family Map":
        _render_strategy_family_map(experiment_scorecards, experiment_candidates)

    elif aggregate_view == "Taxable Impact":
        _render_taxable_impact(
            bot_config=bot_config,
            experiment_scorecards=experiment_scorecards,
        )

    elif aggregate_view == "Validation / QC":
        validation_views = [
            "Sanity Impact",
            "Confidence Gauntlet",
            "Paper Readiness",
            "Regime Tests",
        ]
        validation_view = (
            st.pills(
                "Validation view",
                validation_views,
                selection_mode="single",
                default="Sanity Impact",
                key="experiment_validation_view",
                label_visibility="collapsed",
                width="stretch",
            )
            or "Sanity Impact"
        )
        if validation_view == "Sanity Impact":
            _render_decision_sanity_impact(decision_sanity_impacts)
        elif validation_view == "Confidence Gauntlet":
            _render_confidence_gauntlet(experiment_scorecards)
        elif validation_view == "Paper Readiness":
            _render_paper_readiness(experiment_scorecards)
        elif validation_view == "Regime Tests":
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
                                [
                                    column
                                    for column in regime_columns
                                    if column in regime_view.columns
                                ]
                            ]
                        )
                    )

    elif aggregate_view == "Manifests":
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

    st.divider()
    st.subheader("Candidate Deep Dive")
    st.caption(
        "Lower research area for one selected strategy. The upper section compares experiments "
        "across the backlog; this section explains a single candidate in detail."
    )
    deep_dive_col_a, deep_dive_col_b, deep_dive_col_c = st.columns(3)
    deep_dive_col_a.info(
        "Pick the candidate to inspect, then use the internal tabs to study behavior over time."
    )
    deep_dive_col_b.info(
        "Use this before moving a strategy into paper monitoring or changing champion status."
    )
    deep_dive_col_c.info(
        "Look for performance, drawdown, allocation changes, decision events, and factor exposure."
    )
    _render_approach_detail_workbench(
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
        experiment_regimes=experiment_regimes,
        experiment_walk_forward=experiment_walk_forward,
        experiment_candidates=experiment_candidates,
    )


def _render_signal_evidence(
    experiment_scorecards: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
) -> None:
    st.markdown("**Signal Evidence**")
    st.caption(
        "Marginal-contribution infrastructure for pruning and expansion. Paired rows compare "
        "candidate strategies to their parent/control where possible; broader family rows are "
        "association evidence only. Metrics are after the experiment backtest execution-cost assumptions."
    )
    tagged = tag_scorecard_signal_families(experiment_scorecards, experiment_candidates)
    evidence = build_signal_family_evidence(tagged)
    marginal_tests = build_signal_family_marginal_tests(tagged)
    if evidence.empty:
        st.write("No signal-family evidence could be computed from the saved scorecards.")
        return

    validated_count = int(
        evidence["evidence_label"].isin(["validated_contributor", "promising_mixed"]).sum()
    )
    context_only_count = int(evidence["evidence_label"].isin(["context_only", "research_gap"]).sum())
    paired_count = int(evidence["paired_tests"].sum())
    cols = st.columns(4)
    _helped_metric(cols[0], "Signal Families", f"{len(evidence):,}")
    _helped_metric(cols[1], "Paired Tests", f"{paired_count:,}")
    _helped_metric(cols[2], "Useful / Promising", f"{validated_count:,}")
    _helped_metric(cols[3], "Context / Gaps", f"{context_only_count:,}")

    for takeaway in signal_evidence_takeaways(evidence):
        st.write(f"- {takeaway}")

    chart_frame = evidence.sort_values("net_evidence_score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=chart_frame["net_evidence_score"],
            y=chart_frame["signal_label"],
            orientation="h",
            marker_color=chart_frame["evidence_label"].map(
                {
                    "validated_contributor": "#0f766e",
                    "promising_mixed": "#b7791f",
                    "needs_more_ablation": "#4f46e5",
                    "context_only": "#64748b",
                    "not_proven": "#b91c1c",
                    "research_gap": "#9ca3af",
                }
            ),
            customdata=chart_frame[["paired_tests", "candidate_count", "evidence_label"]],
            hovertemplate=(
                "<b>%{y}</b><br>Evidence score: %{x:.0%}<br>"
                "Paired tests: %{customdata[0]}<br>"
                "Candidates: %{customdata[1]}<br>"
                "Label: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Signal-Family Marginal Evidence",
        template="plotly_white",
        xaxis={"title": "Net evidence score", "tickformat": ".0%", "range": [0, 1]},
        yaxis={"title": ""},
        height=max(420, 32 * len(chart_frame) + 120),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    heatmap_frame = _signal_ablation_heatmap_frame(marginal_tests)
    if not heatmap_frame.empty:
        st.caption(
            "Ablation heatmap: median paired child-minus-parent deltas by signal family. "
            "Green means the family improved the metric direction; red means it hurt after controls."
        )
        st.plotly_chart(
            _signal_ablation_heatmap_figure(heatmap_frame),
            use_container_width=True,
        )

    evidence_tab, paired_tab, tagged_tab = st.tabs(
        ["Family Evidence", "Paired Marginal Tests", "Tagged Strategies"]
    )
    with evidence_tab:
        evidence_columns = [
            "signal_label",
            "evidence_label",
            "data_status",
            "evidence_tier",
            "candidate_count",
            "paired_tests",
            "net_evidence_score",
            "median_delta_cagr",
            "median_delta_max_drawdown",
            "median_delta_reentry_score",
            "median_delta_average_turnover",
            "cagr_win_rate",
            "drawdown_win_rate",
            "churn_win_rate",
            "best_strategy",
            "best_cagr",
            "best_max_drawdown",
            "recommendation",
            "caveat",
        ]
        _render_metric_dataframe(
            _display_metrics(evidence[[column for column in evidence_columns if column in evidence]]),
            hide_index=True,
        )
    with paired_tab:
        if marginal_tests.empty:
            st.write("No parent/control pairs were available for marginal signal tests.")
        else:
            family_filter = st.selectbox(
                "Signal family",
                ["all", *sorted(marginal_tests["signal_label"].dropna().unique())],
                key="signal_evidence_family_filter",
            )
            paired_view = marginal_tests
            if family_filter != "all":
                paired_view = paired_view[paired_view["signal_label"] == family_filter]
            paired_columns = [
                "signal_label",
                "child_strategy",
                "parent_strategy",
                "iteration",
                "delta_cagr",
                "delta_max_drawdown",
                "delta_calmar",
                "delta_reentry_score",
                "delta_average_turnover",
                "delta_left_tail_regime_return",
                "hypothesis",
            ]
            _render_metric_dataframe(
                _display_metrics(
                    paired_view[[column for column in paired_columns if column in paired_view]]
                ),
                hide_index=True,
            )
    with tagged_tab:
        tagged_columns = [
            "iteration",
            "strategy",
            "display_name",
            "phase",
            "family",
            "role",
            "signal_families",
            "promotion_decision",
            "promotion_score",
            "cagr",
            "max_drawdown",
            "average_turnover",
            "hypothesis",
        ]
        _render_metric_dataframe(
            _display_metrics(tagged[[column for column in tagged_columns if column in tagged]]),
            hide_index=True,
        )


def _signal_ablation_heatmap_frame(marginal_tests: pd.DataFrame) -> pd.DataFrame:
    if marginal_tests.empty or "signal_label" not in marginal_tests:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in [
            "delta_cagr",
            "delta_max_drawdown",
            "delta_calmar",
            "delta_reentry_score",
            "delta_average_turnover",
            "delta_left_tail_regime_return",
        ]
        if column in marginal_tests
    ]
    if not metric_columns:
        return pd.DataFrame()
    grouped = (
        marginal_tests.groupby("signal_label", dropna=False)[metric_columns]
        .median(numeric_only=True)
        .dropna(how="all")
    )
    if grouped.empty:
        return pd.DataFrame()
    signed = grouped.copy()
    if "delta_average_turnover" in signed:
        signed["delta_average_turnover"] = -signed["delta_average_turnover"]
    return signed.sort_index()


def _signal_ablation_heatmap_figure(heatmap_frame: pd.DataFrame) -> go.Figure:
    labels = {
        "delta_cagr": "CAGR",
        "delta_max_drawdown": "Max drawdown",
        "delta_calmar": "Calmar",
        "delta_reentry_score": "Re-entry",
        "delta_average_turnover": "Lower churn",
        "delta_left_tail_regime_return": "Left-tail",
    }
    display = heatmap_frame.rename(columns=labels)
    z_values = display.to_numpy(dtype=float)
    max_abs = float(pd.Series(z_values.ravel()).abs().dropna().max()) if z_values.size else 0.0
    color_bound = max(max_abs, 0.01)
    figure = go.Figure(
        go.Heatmap(
            z=z_values,
            x=display.columns.tolist(),
            y=display.index.astype(str).tolist(),
            zmid=0.0,
            zmin=-color_bound,
            zmax=color_bound,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.5, "#f8fafc"],
                [1.0, "#0f766e"],
            ],
            colorbar={"title": "Median delta"},
            hovertemplate=(
                "<b>%{y}</b><br>%{x}<br>Median paired delta: %{z:.2%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Signal Ablation Heatmap",
        template="plotly_white",
        height=max(360, 34 * len(display.index) + 140),
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        xaxis={"side": "top"},
        yaxis={"title": ""},
    )
    return figure


def _render_taxable_impact(*, bot_config: Any, experiment_scorecards: pd.DataFrame) -> None:
    st.markdown("**Taxable Impact**")
    st.caption(
        "Estimated taxable-account research view. Use this to check whether an active strategy's "
        "edge survives realized gains, short-term gain mix, wash-sale estimates, loss carryforward, "
        "and configured tax rates. IRA-like/pre-tax selection should still use the Outcome Frontier."
    )
    tax_config = getattr(bot_config, "tax_account", None)
    account_type = str(getattr(tax_config, "account_type", "unknown"))
    short_rate = _safe_float(getattr(tax_config, "federal_short_term_tax_rate", None)) or 0.0
    short_rate += _safe_float(getattr(tax_config, "state_short_term_tax_rate", None)) or 0.0
    long_rate = _safe_float(getattr(tax_config, "federal_long_term_tax_rate", None)) or 0.0
    long_rate += _safe_float(getattr(tax_config, "state_long_term_tax_rate", None)) or 0.0
    niit_applies = bool(getattr(tax_config, "niit_applies", False))
    niit_rate = _safe_float(getattr(tax_config, "niit_rate", None)) or 0.0
    if niit_applies:
        short_rate += niit_rate
        long_rate += niit_rate
    assumption_cols = st.columns(5)
    _helped_metric(assumption_cols[0], "Account", account_type.replace("_", " ").title())
    _helped_metric(assumption_cols[1], "Short Rate", _format_percent(short_rate))
    _helped_metric(assumption_cols[2], "Long Rate", _format_percent(long_rate))
    _helped_metric(
        assumption_cols[3],
        "Lot Method",
        str(getattr(tax_config, "lot_selection_method", "unknown")).replace("_", " ").title(),
    )
    _helped_metric(
        assumption_cols[4],
        "Wash Window",
        f"{int(getattr(tax_config, 'wash_sale_window_days', 0) or 0)} days",
    )

    required_columns = {
        "after_tax_cagr",
        "after_tax_max_drawdown",
        "tax_drag_bps_per_year",
        "after_tax_growth_constrained_utility_score",
    }
    if experiment_scorecards.empty or not required_columns.issubset(experiment_scorecards.columns):
        st.info(
            "No estimated taxable scorecard fields are available yet. Run new experiment iterations "
            "with the current code, then run `migrate-warehouse` so the dashboard can load them."
        )
        return

    frame = experiment_scorecards.copy()
    if "research_status" in frame:
        active = frame[~frame["research_status"].astype(str).eq("pruned_dead_end")].copy()
        if not active.empty:
            frame = active
    frame = frame.dropna(subset=["after_tax_cagr", "after_tax_growth_constrained_utility_score"])
    if frame.empty:
        st.info(
            "Taxable columns exist, but no candidate has complete estimated taxable metrics yet."
        )
        return

    card_cols = st.columns(4)
    _helped_metric(card_cols[0], "Tax-Evaluated", f"{len(frame):,}")
    _helped_metric(
        card_cols[1],
        "Median Tax Drag",
        _format_decimal(frame["tax_drag_bps_per_year"].median()),
        key="tax_drag_bps_per_year",
    )
    _helped_metric(
        card_cols[2],
        "Best After-Tax Utility",
        _format_decimal(frame["after_tax_growth_constrained_utility_score"].max()),
        key="after_tax_growth_constrained_utility_score",
    )
    best_after_tax_cagr = frame.sort_values(
        "after_tax_growth_constrained_utility_score", ascending=False
    ).iloc[0]
    _helped_metric(
        card_cols[3],
        "Top After-Tax CAGR",
        _format_percent(best_after_tax_cagr.get("after_tax_cagr")),
    )

    plot_frame = frame.dropna(subset=["cagr", "after_tax_cagr"]).copy()
    if not plot_frame.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_frame["cagr"],
                y=plot_frame["after_tax_cagr"],
                mode="markers",
                name="Strategies",
                marker={
                    "size": 11,
                    "color": plot_frame["tax_drag_bps_per_year"],
                    "colorscale": "Bluered",
                    "showscale": True,
                    "colorbar": {"title": "Tax drag bps"},
                    "line": {"width": 1, "color": "#0f172a"},
                    "opacity": 0.78,
                },
                text=plot_frame.get("display_name", plot_frame.get("strategy", "")),
                customdata=plot_frame[
                    [
                        column
                        for column in [
                            "strategy",
                            "after_tax_growth_constrained_utility_score",
                            "tax_drag_bps_per_year",
                            "short_term_gain_share",
                        ]
                        if column in plot_frame
                    ]
                ],
                hovertemplate=(
                    "%{text}<br>Pre-tax CAGR %{x:.1%}<br>After-tax CAGR %{y:.1%}"
                    "<br>Tax drag %{customdata[2]:.0f} bps"
                    "<br>After-tax utility %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
        axis_min = float(min(plot_frame["cagr"].min(), plot_frame["after_tax_cagr"].min()))
        axis_max = float(max(plot_frame["cagr"].max(), plot_frame["after_tax_cagr"].max()))
        fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                name="No tax drag",
                line={"dash": "dash", "color": "#64748b"},
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            height=460,
            xaxis_title="Pre-tax CAGR",
            yaxis_title="Estimated after-tax CAGR",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            margin={"l": 20, "r": 20, "t": 35, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

    top_columns = [
        "iteration",
        "display_name",
        "strategy",
        "family",
        "research_status",
        "cagr",
        "after_tax_cagr",
        "tax_drag_bps_per_year",
        "max_drawdown",
        "after_tax_max_drawdown",
        "growth_constrained_utility_score",
        "after_tax_growth_constrained_utility_score",
        "after_tax_terminal_wealth_with_contributions_15y",
        "net_estimated_tax_paid",
        "realized_short_term_gain",
        "realized_long_term_gain",
        "realized_loss_harvested",
        "wash_sale_disallowed_loss",
        "loss_carryforward_end",
        "short_term_gain_share",
        "monitoring_readiness_label",
    ]
    st.caption("Top estimated after-tax candidates")
    top_view = frame.sort_values("after_tax_growth_constrained_utility_score", ascending=False)
    _render_metric_dataframe(
        _display_metrics(
            top_view[[column for column in top_columns if column in top_view]].head(25)
        ),
        hide_index=True,
    )

    with st.expander("Tax drag watchlist", expanded=False):
        st.caption(
            "High tax drag is not automatically fatal, but it means taxable-account monitoring should "
            "prove the after-tax edge instead of relying on pre-tax CAGR."
        )
        drag_columns = [
            "display_name",
            "strategy",
            "family",
            "cagr",
            "after_tax_cagr",
            "tax_drag_bps_per_year",
            "short_term_gain_share",
            "realized_short_term_gain",
            "wash_sale_disallowed_loss",
            "after_tax_growth_constrained_utility_score",
        ]
        drag_view = frame.sort_values("tax_drag_bps_per_year", ascending=False)
        _render_metric_dataframe(
            _display_metrics(
                drag_view[[column for column in drag_columns if column in drag_view]].head(25)
            ),
            hide_index=True,
        )

    st.info(
        "Interpretation: taxable mode is a parallel research lens, not a replacement for risk control. "
        "Favor candidates where after-tax utility and after-tax CAGR remain strong, tax drag is understandable, "
        "and wash-sale/loss-carryforward estimates do not dominate the result."
    )


def _render_decision_sanity_impact(decision_sanity_impacts: pd.DataFrame) -> None:
    st.caption(
        "Backtested raw-versus-capped ablations for the decision-sanity overlay. Positive "
        "delta promotion score, Calmar, walk-forward rate, left-tail return, and max drawdown "
        "are good. Positive delta max drawdown means the drawdown became less negative."
    )
    if decision_sanity_impacts.empty:
        st.write(
            "No decision-sanity impact files were found. Run iteration 77 or 78 to generate "
            "paired raw-versus-capped evidence."
        )
        return

    summary = summarize_decision_sanity_impacts(decision_sanity_impacts)
    if not summary.empty:
        st.markdown("**Profile adoption read**")
        summary_columns = [
            "decision_sanity",
            "adoption_read",
            "pairs",
            "mean_delta_promotion_score",
            "mean_delta_cagr",
            "mean_delta_max_drawdown",
            "mean_delta_calmar",
            "mean_delta_turnover",
            "mean_delta_walk_forward_positive_rate",
            "mean_delta_left_tail_regime_return",
            "promotion_win_rate",
            "drawdown_win_rate",
            "calmar_win_rate",
        ]
        _render_metric_dataframe(
            _display_metrics(summary[[column for column in summary_columns if column in summary]]),
            hide_index=True,
        )

    iterations = sorted(decision_sanity_impacts["iteration"].dropna().unique())
    selected_iterations = st.multiselect(
        "Impact iterations",
        iterations,
        default=iterations[-2:],
        key="decision_sanity_impact_iterations",
    )
    impact_view = decision_sanity_impacts[
        decision_sanity_impacts["iteration"].isin(selected_iterations)
    ].copy()
    if impact_view.empty:
        st.write("No impact rows match the selected iterations.")
        return

    st.markdown("**Paired raw-versus-capped details**")
    detail_columns = [
        "iteration",
        "family",
        "decision_sanity",
        "raw_strategy",
        "capped_strategy",
        "raw_promotion_decision",
        "capped_promotion_decision",
        "delta_promotion_score",
        "delta_cagr",
        "delta_max_drawdown",
        "delta_calmar",
        "delta_average_turnover",
        "delta_walk_forward_positive_rate",
        "delta_left_tail_regime_return",
    ]
    _render_metric_dataframe(
        _display_metrics(
            impact_view[[column for column in detail_columns if column in impact_view.columns]]
        ),
        hide_index=True,
    )


def _render_paper_readiness(experiment_scorecards: pd.DataFrame) -> None:
    if experiment_scorecards.empty or "monitoring_readiness_score" not in experiment_scorecards:
        st.write(
            "Paper-readiness diagnostics will appear after running an experiment iteration with "
            "the updated operability and re-entry metrics."
        )
        return

    st.caption(
        "Ranks candidates for paper monitoring by blending historical performance, robustness, "
        "human-executable trade cadence, and whether the strategy can re-risk after defensive periods."
    )
    frame = experiment_scorecards.copy()
    frame["monitoring_readiness_score"] = pd.to_numeric(
        frame["monitoring_readiness_score"],
        errors="coerce",
    )
    frame = frame.sort_values("monitoring_readiness_score", ascending=False)

    label_summary = (
        frame.groupby("monitoring_readiness_label", as_index=False, dropna=False)
        .agg(
            candidates=("strategy", "count"),
            best_strategy=("strategy", "first"),
            best_score=("monitoring_readiness_score", "max"),
            median_promotion_score=("promotion_score", "median"),
            median_operability_score=("operability_score", "median"),
            median_reentry_score=("reentry_score", "median"),
        )
        .sort_values("best_score", ascending=False)
    )
    st.markdown("**Readiness summary**")
    _render_metric_dataframe(_display_metrics(label_summary), hide_index=True)

    st.markdown("**Top paper-monitoring candidates**")
    columns = [
        "iteration",
        "display_name",
        "strategy",
        "phase",
        "family",
        "monitoring_readiness_label",
        "monitoring_readiness_score",
        "promotion_decision",
        "promotion_score",
        "robustness_score",
        "operability_label",
        "operability_score",
        "material_trade_days_per_year",
        "mean_days_between_material_trades",
        "max_single_day_turnover",
        "risk_cycle_label",
        "reentry_score",
        "median_reentry_days",
        "low_risk_day_rate",
        "cagr",
        "max_drawdown",
        "calmar",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
    ]
    _render_metric_dataframe(
        _display_metrics(frame.head(30)[[column for column in columns if column in frame.columns]]),
        hide_index=True,
    )


def _render_confidence_gauntlet(experiment_scorecards: pd.DataFrame) -> None:
    if experiment_scorecards.empty or "confidence_score" not in experiment_scorecards:
        st.write(
            "Confidence-gauntlet columns will appear after running an experiment iteration with "
            "benchmark knockout, confidence, and deployment-blocker scoring."
        )
        return

    st.caption(
        "Deployment-focused research gate. This blends performance, robustness, benchmark knockout, "
        "paper-readiness, walk-forward stability, left-tail behavior, and operability into one audit view."
    )
    frame = experiment_scorecards.copy()
    frame["confidence_score"] = pd.to_numeric(frame["confidence_score"], errors="coerce")
    frame = frame.sort_values("confidence_score", ascending=False)

    label_summary = (
        frame.groupby("confidence_label", as_index=False, dropna=False)
        .agg(
            candidates=("strategy", "count"),
            best_strategy=("strategy", "first"),
            best_confidence=("confidence_score", "max"),
            median_promotion_score=("promotion_score", "median"),
            median_benchmark_score=("benchmark_knockout_score", "median"),
            median_readiness_score=("monitoring_readiness_score", "median"),
        )
        .sort_values("best_confidence", ascending=False)
    )
    st.markdown("**Confidence summary**")
    _render_metric_dataframe(_display_metrics(label_summary), hide_index=True)

    blocker_counts = _deployment_blocker_counts(frame)
    if not blocker_counts.empty:
        st.markdown("**Most common blockers**")
        st.bar_chart(blocker_counts.set_index("blocker")["candidates"])
        _render_metric_dataframe(blocker_counts, hide_index=True)

    st.markdown("**Highest-confidence candidates**")
    columns = [
        "iteration",
        "display_name",
        "strategy",
        "phase",
        "family",
        "confidence_label",
        "confidence_score",
        "deployment_blockers",
        "benchmark_knockout_label",
        "benchmark_knockout_score",
        "monitoring_readiness_label",
        "monitoring_readiness_score",
        "promotion_decision",
        "promotion_score",
        "robustness_score",
        "operability_label",
        "risk_cycle_label",
        "cagr",
        "max_drawdown",
        "calmar",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
    ]
    _render_metric_dataframe(
        _display_metrics(frame.head(35)[[column for column in columns if column in frame.columns]]),
        hide_index=True,
    )


def _deployment_blocker_counts(scorecards: pd.DataFrame) -> pd.DataFrame:
    if "deployment_blockers" not in scorecards:
        return pd.DataFrame()
    rows = []
    for blockers in scorecards["deployment_blockers"].dropna().astype(str):
        if not blockers or blockers == "none" or blockers == "nan":
            continue
        rows.extend(part.strip() for part in blockers.split(";") if part.strip())
    if not rows:
        return pd.DataFrame()
    return (
        pd.Series(rows, name="blocker")
        .value_counts()
        .rename_axis("blocker")
        .reset_index(name="candidates")
    )


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


@st.cache_data(show_spinner=False, ttl=300)
def _load_ml_diagnostic_frames(
    root: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(root)
    frames = []
    for filename in [
        "metrics.csv",
        "latest_probabilities.csv",
        "family_importance.csv",
        "drift.csv",
    ]:
        file_path = base / filename
        frames.append(pd.read_csv(file_path) if file_path.exists() else pd.DataFrame())
    return cast(tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple(frames))


def _render_ml_diagnostics() -> None:
    st.subheader("ML Diagnostics")
    metrics, latest, family_importance, drift = _load_ml_diagnostic_frames(
        str(DEFAULT_ML_DIAGNOSTICS_DIR)
    )
    if metrics.empty and latest.empty:
        st.info(
            "No ML diagnostic artifacts found. Run `poetry run trade-bot run-ml-diagnostics` "
            "to generate future-state, re-entry, off-ramp, router, churn, feature-importance, and drift diagnostics."
        )
        return

    overview_cols = st.columns(4)
    _helped_metric(
        overview_cols[0], "Tasks", f"{metrics['task'].nunique():,}" if "task" in metrics else "0"
    )
    _helped_metric(overview_cols[1], "Model Rows", f"{len(metrics):,}")
    best_utility = pd.to_numeric(metrics.get("utility_score"), errors="coerce").max()
    _helped_metric(
        overview_cols[2],
        "Best Utility",
        f"{best_utility:.3f}" if best_utility == best_utility else "n/a",
    )
    top_drift = (
        pd.to_numeric(drift.get("drift_score"), errors="coerce").max()
        if not drift.empty
        else float("nan")
    )
    _helped_metric(
        overview_cols[3],
        "Top Drift",
        f"{top_drift:.2f}" if top_drift == top_drift else "n/a",
    )

    model_tab, latest_tab, families_tab, drift_tab = st.tabs(
        ["Model Tasks", "Latest Probabilities", "Feature Families", "Drift"]
    )
    with model_tab:
        st.caption(
            "Walk-forward model diagnostics. Utility combines accuracy, balanced accuracy, positive-class recall, Brier score, and calibration error."
        )
        columns = [
            "task",
            "kind",
            "horizon_days",
            "model",
            "observations",
            "utility_score",
            "accuracy",
            "balanced_accuracy",
            "brier_score",
            "calibration_error",
            "positive_class",
            "positive_recall",
        ]
        available = [column for column in columns if column in metrics.columns]
        _render_metric_dataframe(_display_metrics(metrics[available].head(60)), hide_index=True)
    with latest_tab:
        st.caption(
            "Latest batch inference probabilities by task and model. These are research diagnostics, not trade tickets."
        )
        columns = [
            "task",
            "kind",
            "horizon_days",
            "model",
            "top_class",
            "top_probability",
        ]
        probability_columns = [column for column in latest.columns if column.startswith("prob_")]
        available = [
            column for column in [*columns, *probability_columns] if column in latest.columns
        ]
        _render_metric_dataframe(_display_metrics(latest[available].head(80)), hide_index=True)
    with families_tab:
        st.caption("Feature-family importance from sklearn models, aggregated across folds.")
        columns = [
            "task",
            "kind",
            "horizon_days",
            "model",
            "feature_family",
            "mean_importance",
            "represented_features",
        ]
        available = [column for column in columns if column in family_importance.columns]
        _render_metric_dataframe(
            _display_metrics(family_importance[available].head(80)), hide_index=True
        )
    with drift_tab:
        st.caption(
            "Feature drift compares the most recent year against the prior reference window."
        )
        columns = [
            "feature",
            "feature_family",
            "recent_mean",
            "reference_mean",
            "mean_shift_z",
            "psi",
            "drift_score",
        ]
        available = [column for column in columns if column in drift.columns]
        _render_metric_dataframe(_display_metrics(drift[available].head(80)), hide_index=True)


def _render_research_lab(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
    decision_sanity_impacts: pd.DataFrame,
) -> None:
    _render_experiment_monitor(
        bot_config,
        baseline_run,
        experiment_scorecards,
        experiment_regimes,
        experiment_walk_forward,
        experiment_candidates,
        decision_sanity_impacts,
    )
    st.divider()
    st.subheader("Research Diagnostics / QC")
    st.caption(
        "Lower-frequency diagnostics that support pruning and model governance. These are useful "
        "for audit and expansion work, but they are intentionally collapsed so the main research "
        "flow stays focused on strategy comparison and candidate inspection."
    )
    with st.expander("ML diagnostics", expanded=False):
        _render_ml_diagnostics()
    with st.expander("Signal inclusion tests", expanded=False):
        _render_signal_inclusion(baseline_run)
