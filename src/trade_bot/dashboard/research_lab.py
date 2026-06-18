from __future__ import annotations

from datetime import date
from typing import Any, cast

import pandas as pd
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _result_date_bounds,
    _window_start_from_preset,
)
from trade_bot.DEFAULT import DEFAULT_PERFORMANCE_WINDOW, DEFAULT_PERFORMANCE_WINDOWS
from trade_bot.reporting.report import make_equity_drawdown_figure, window_performance_frame
from trade_bot.research.approach_explorer import (
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
    execution_for_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.curation import rank_strategy_candidates, select_curated_strategy_shelf
from trade_bot.research.experiment_monitor import (
    build_strategy_family_map,
    latest_experiment_iteration,
    strategy_family_takeaways,
    summarize_experiment_families,
    summarize_experiment_history,
    summarize_experiment_operating_systems,
    summarize_family_clusters,
    summarize_risk_behavior_matrix,
    summarize_strategy_archetypes,
)


def _render_strategy_summary_and_behavior(
    *,
    row: pd.Series,
    strategy: Any,
    bot_config: Any,
    baseline_run: BaselineRun,
    key_prefix: str,
) -> Any:
    scenario_sizing = scenario_sizing_from_catalog_row(row)
    execution = execution_for_catalog_row(row, bot_config.execution)
    st.markdown("**How this approach works**")
    for paragraph in build_approach_explanation(
        strategy,
        row,
        bot_config,
        execution=execution,
        scenario_sizing=scenario_sizing,
    ):
        st.write(paragraph)

    result, missing_columns = build_approach_backtest_result(
        baseline_run.prices,
        strategy,
        execution,
        scenario_sizing=scenario_sizing,
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

    window_options = {
        "3 months": 63,
        "6 months": 126,
        "1 year": 252,
        "2 years": 504,
        "5 years": 1260,
    }
    selected_window = st.selectbox(
        "Position-history window",
        list(window_options),
        index=2,
        key=f"{key_prefix}_position_window",
    )
    lookback_days = window_options[selected_window]

    exposure_history = build_approach_exposure_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=lookback_days,
    )
    weight_history = build_approach_weight_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=lookback_days,
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
            lookback_days=lookback_days,
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
        lookback_days=lookback_days,
    )
    if change_log.empty:
        st.caption("No material allocation changes in the selected window.")
    else:
        st.caption("Recent material allocation changes")
        _render_metric_dataframe(_display_metrics(change_log), hide_index=True)

    holding_stats = build_approach_holding_stats(weights, lookback_days=lookback_days)
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
    window_columns = st.columns([1, 2])
    window_options = list(DEFAULT_PERFORMANCE_WINDOWS)
    default_index = (
        window_options.index(DEFAULT_PERFORMANCE_WINDOW)
        if DEFAULT_PERFORMANCE_WINDOW in window_options
        else 0
    )
    window_preset = window_columns[0].selectbox(
        "Performance window",
        window_options,
        index=default_index,
        key=f"{key_prefix}_performance_window",
    )
    custom_start_date: date | None = None
    window_end = latest_result_date
    if window_preset == "Custom":
        custom_columns = st.columns(2)
        custom_start_date = cast(
            date,
            custom_columns[0].date_input(
                "Start",
                value=max(
                    earliest_result_date,
                    latest_result_date - pd.DateOffset(days=90),
                ).date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
                key=f"{key_prefix}_performance_start",
            ),
        )
        custom_end_date = cast(
            date,
            custom_columns[1].date_input(
                "End",
                value=latest_result_date.date(),
                min_value=earliest_result_date.date(),
                max_value=latest_result_date.date(),
                key=f"{key_prefix}_performance_end",
            ),
        )
        window_end = min(
            latest_result_date, max(earliest_result_date, pd.Timestamp(custom_end_date))
        )

    window_start = _window_start_from_preset(
        window_preset,
        earliest=earliest_result_date,
        latest=latest_result_date,
        custom_start=custom_start_date,
    )
    if window_start > window_end:
        window_start = window_end

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
            else ("curated_top_25" if bool(row.get("is_curated")) else "experiment_archive")
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

    scope_options = ["Curated top 25 + baselines", "All approaches"]
    selected_scope = st.radio(
        "Approach set",
        scope_options,
        horizontal=True,
        key="approach_detail_scope",
    )
    if selected_scope == "Curated top 25 + baselines":
        visible_catalog = catalog[(catalog["source"] == "baseline") | catalog["is_curated"]]
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

    overview_cols = st.columns(6)
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
    curation_value = (
        "not curated"
        if pd.isna(approach_row.get("curation_rank"))
        else f"#{int(float(approach_row['curation_rank']))}"
    )
    _helped_metric(overview_cols[5], "Curated Rank", curation_value)

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
        performance_tab,
        allocation_tab,
        mechanics_tab,
        robustness_tab,
        manifest_tab,
    ) = st.tabs(
        [
            "Performance Over Time",
            "Allocation Behavior",
            "Mechanics",
            "Robustness",
            "Manifest / Risk Notes",
        ]
    )

    with performance_tab:
        scorecard = _scorecard_for_catalog_row(
            approach_row,
            baseline_run=baseline_run,
            experiment_scorecards=experiment_scorecards,
        )
        if not scorecard.empty:
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
        "strategy",
        "curation_bucket",
        "curation_reason",
        "iteration",
        "phase",
        "family",
        "role",
        "promotion_decision",
        "promotion_score",
        "robustness_score",
        "cagr",
        "max_drawdown",
        "calmar",
        "average_turnover",
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

    archetype_summary = summarize_strategy_archetypes(family_map)
    if not archetype_summary.empty:
        st.markdown("**Strategy archetype summary**")
        chart_columns = [
            column
            for column in ["median_cagr", "median_max_drawdown", "median_turnover"]
            if column in archetype_summary
        ]
        if chart_columns:
            chart_frame = archetype_summary.set_index("strategy_archetype")[chart_columns]
            st.bar_chart(chart_frame.dropna(how="all"))
        _render_metric_dataframe(_display_metrics(archetype_summary), hide_index=True)

    risk_matrix = summarize_risk_behavior_matrix(family_map)
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
        "strategy",
        "strategy_archetype",
        "risk_behavior",
        "equity_expression",
        "defensive_expression",
        "strategy_type",
        "defensive_ticker",
        "ticker_count",
        "primary_tickers",
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


def _render_experiment_monitor(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
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
    latest_label = f"{latest_iteration:02d}" if latest_iteration is not None else "n/a"
    col_a, col_b, col_c, col_d = st.columns(4)
    _helped_metric(col_a, "Iterations", latest_label)
    _helped_metric(col_b, "Candidates", f"{len(experiment_scorecards):,}")
    _helped_metric(col_c, "Promoted", f"{promoted_count:,}", key="promotion_decision")
    _helped_metric(col_d, "Risk rejects", f"{rejected_tail_count:,}", key="promotion_decision")

    (
        experiment_detail_tab,
        experiment_shelf_tab,
        experiment_family_tab,
        experiment_leaderboard_tab,
        experiment_regime_tab,
        experiment_overview_tab,
        experiment_manifest_tab,
    ) = st.tabs(
        [
            "Candidate Details",
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
            "Research category",
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
    _render_experiment_monitor(
        bot_config,
        baseline_run,
        experiment_scorecards,
        experiment_regimes,
        experiment_walk_forward,
        experiment_candidates,
    )
    st.divider()
    _render_signal_inclusion(baseline_run)
