from __future__ import annotations

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
from trade_bot.DEFAULTS import (
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
    DEFAULT_PERFORMANCE_WINDOW,
    DEFAULT_PERFORMANCE_WINDOWS,
)
from trade_bot.features.indicators import drawdown
from trade_bot.reporting.report import make_equity_drawdown_figure, window_performance_frame
from trade_bot.research.approach_explorer import (
    build_approach_allocation_transition_events,
    build_approach_backtest_result,
    build_approach_catalog,
    build_approach_change_log,
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
from trade_bot.research.strategy_outcome_utility import (
    add_outcome_frontier_flags,
    enrich_strategy_outcome_utility,
)


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


def _render_strategy_summary_and_behavior(
    *,
    row: pd.Series,
    strategy: Any,
    bot_config: Any,
    baseline_run: BaselineRun,
    key_prefix: str,
) -> Any:
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

    result, missing_columns = build_approach_backtest_result(
        baseline_run.prices,
        strategy,
        execution,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
        name=str(row.get("strategy", "approach")),
    )
    if missing_columns:
        st.caption("Missing from loaded prices: " + ", ".join(missing_columns))
    if result is None:
        st.warning("Could not reconstruct historical weights for this approach from loaded prices.")
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
        "Curated top 25 + baselines",
        "Operational + iteration candidates",
        "All non-pruned approaches",
        "All approaches",
    ]
    selected_scope = st.radio(
        "Approach set",
        scope_options,
        horizontal=True,
        key="approach_detail_scope",
    )
    if selected_scope == "Curated top 25 + baselines":
        visible_catalog = catalog[(catalog["source"] == "baseline") | catalog["is_curated"]]
    elif selected_scope == "Operational + iteration candidates":
        visible_catalog = catalog[
            (catalog["source"] == "baseline")
            | catalog["research_status"].isin(
                ["operational_candidate", "needs_iteration", "reference"]
            )
        ]
    elif selected_scope == "All non-pruned approaches":
        visible_catalog = catalog[
            (catalog["source"] == "baseline") | ~catalog["research_status"].eq("pruned_dead_end")
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

    detail_result = _render_strategy_summary_and_behavior(
        row=approach_row,
        strategy=approach_strategy,
        bot_config=bot_config,
        baseline_run=baseline_run,
        key_prefix="approach_detail",
    )
    if str(approach_row.get("hypothesis", "")):
        with st.expander("Original research hypothesis", expanded=False):
            st.write(str(approach_row["hypothesis"]))

    (
        combined_tab,
        performance_tab,
        allocation_tab,
        mechanics_tab,
        robustness_tab,
        manifest_tab,
    ) = st.tabs(
        [
            "Performance + Allocation",
            "Performance Over Time",
            "Allocation Behavior",
            "Mechanics",
            "Robustness",
            "Manifest / Risk Notes",
        ]
    )

    with combined_tab:
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

    with performance_tab:
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

    with allocation_tab:
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

    with mechanics_tab:
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

    with robustness_tab:
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

    with manifest_tab:
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
                hovertemplate="%{text}<br>Pareto efficient<extra></extra>",
            )
        )
    fig.add_vline(x=DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT, line_dash="dash", line_color="#f59e0b")
    fig.add_vline(x=DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT, line_dash="dot", line_color="#ef4444")
    fig.update_layout(
        height=520,
        xaxis_title="Max drawdown",
        yaxis_title="CAGR",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend_title="Growth utility tier",
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_options = _outcome_select_options(plot_frame)
    selected_label = st.selectbox(
        "Outcome strategy to inspect",
        selected_options["label"].tolist(),
        key="outcome_frontier_strategy",
    )
    selected_row = selected_options[selected_options["label"] == selected_label].iloc[0]
    selected_strategy = str(selected_row["strategy"])
    selected_scorecard = plot_frame[plot_frame["strategy"].astype(str) == selected_strategy].iloc[0]
    _render_outcome_decision_cards(
        selected_scorecard,
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
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


def _render_outcome_decision_cards(
    row: pd.Series,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
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

    cols = st.columns(5)
    _helped_metric(cols[0], "15Y Wealth", _format_currency(wealth))
    _helped_metric(cols[1], "Extra vs SPY", _format_currency(extra_spy))
    _helped_metric(cols[2], "Extra vs QQQ", _format_currency(extra_qqq))
    _helped_metric(cols[3], "Recovery Needed", _format_percent(row.get("drawdown_recovery_return")))
    _helped_metric(cols[4], "Time Underwater", _format_percent(underwater_rate))

    st.info(_outcome_decision_helper(row, extra_spy=extra_spy, extra_qqq=extra_qqq))

    context = pd.DataFrame(
        [
            {
                "metric": "CAGR",
                "selected": row.get("cagr"),
                "spy_delta": row.get("excess_cagr_vs_spy"),
                "qqq_delta": row.get("excess_cagr_vs_qqq"),
            },
            {
                "metric": "Max drawdown",
                "selected": row.get("max_drawdown"),
                "spy_delta": row.get("drawdown_improvement_vs_spy"),
                "qqq_delta": row.get("drawdown_improvement_vs_qqq"),
            },
            {
                "metric": "Worst 1Y CAGR",
                "selected": row.get("worst_1y_cagr"),
                "spy_delta": pd.NA,
                "qqq_delta": pd.NA,
            },
            {
                "metric": "Worst 3Y CAGR",
                "selected": row.get("worst_3y_cagr"),
                "spy_delta": pd.NA,
                "qqq_delta": pd.NA,
            },
            {
                "metric": "Left-tail regime return",
                "selected": row.get("left_tail_regime_return"),
                "spy_delta": pd.NA,
                "qqq_delta": pd.NA,
            },
        ]
    )
    _render_metric_dataframe(_display_metrics(context), hide_index=True)


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

    (
        experiment_detail_tab,
        experiment_outcome_tab,
        experiment_taxable_tab,
        experiment_sanity_tab,
        experiment_confidence_tab,
        experiment_readiness_tab,
        experiment_shelf_tab,
        experiment_family_tab,
        experiment_leaderboard_tab,
        experiment_regime_tab,
        experiment_overview_tab,
        experiment_manifest_tab,
    ) = st.tabs(
        [
            "Candidate Details",
            "Outcome Frontier",
            "Taxable Impact",
            "Sanity Impact",
            "Confidence Gauntlet",
            "Paper Readiness",
            "Curated Shelf",
            "Family Map",
            "Leaderboard",
            "Regime Tests",
            "Overview",
            "Manifests",
        ]
    )

    with experiment_detail_tab:
        st.caption(
            "Primary research workbench. Use this before paper-monitoring a strategy: it shows "
            "the explanation, performance-over-time charts, allocation behavior, mechanics, "
            "robustness diagnostics, and manifest/risk notes."
        )
        _render_approach_detail_workbench(
            bot_config=bot_config,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
            experiment_regimes=experiment_regimes,
            experiment_walk_forward=experiment_walk_forward,
            experiment_candidates=experiment_candidates,
        )

    with experiment_outcome_tab:
        _render_outcome_frontier(
            bot_config=bot_config,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
            experiment_candidates=experiment_candidates,
        )

    with experiment_taxable_tab:
        _render_taxable_impact(
            bot_config=bot_config,
            experiment_scorecards=experiment_scorecards,
        )

    with experiment_sanity_tab:
        _render_decision_sanity_impact(decision_sanity_impacts)

    with experiment_confidence_tab:
        _render_confidence_gauntlet(experiment_scorecards)

    with experiment_readiness_tab:
        _render_paper_readiness(experiment_scorecards)

    with experiment_shelf_tab:
        _render_curated_strategy_shelf(experiment_scorecards)

    with experiment_family_tab:
        _render_strategy_family_map(experiment_scorecards, experiment_candidates)

    with experiment_overview_tab:
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

    with experiment_leaderboard_tab:
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
        if status_filter == "active research":
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
    _render_ml_diagnostics()
    st.divider()
    _render_signal_inclusion(baseline_run)
