from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, cast

import pandas as pd
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _default_strategy_selection,
    _display_metrics,
    _result_date_bounds,
    _window_start_from_preset,
)
from trade_bot.DEFAULTS import (
    DEFAULT_CURATED_SHELF_LIMIT,
    DEFAULT_PERFORMANCE_WINDOW,
    DEFAULT_PERFORMANCE_WINDOWS,
)
from trade_bot.reporting.report import (
    latest_positions_frame,
    make_equity_drawdown_figure,
    window_performance_frame,
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
from trade_bot.research.curation import rank_strategy_candidates, select_curated_strategy_shelf


def _render_performance(
    baseline_run: BaselineRun,
    *,
    bot_config: Any | None = None,
    experiment_scorecards: pd.DataFrame | None = None,
) -> None:
    st.subheader("Performance")
    _render_metric_dataframe(_display_metrics(baseline_run.metrics))

    st.subheader("Windowed Performance")
    option_frame = _performance_option_frame(
        baseline_run,
        bot_config=bot_config,
        experiment_scorecards=experiment_scorecards,
    )
    strategy_names = option_frame["strategy"].tolist()
    label_lookup = dict(zip(option_frame["strategy"], option_frame["label"], strict=False))
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
        format_func=lambda strategy: label_lookup.get(strategy, strategy),
    )
    selected_results, missing_results = _selected_performance_results(
        selected_performance_strategies,
        baseline_run=baseline_run,
        bot_config=bot_config,
        option_frame=option_frame,
    )
    if missing_results:
        st.caption(
            "Could not reconstruct: "
            + ", ".join(missing_results)
            + ". These usually need missing ticker history or stale manifests."
        )
    if not selected_results:
        st.warning("Select at least one approach with a reconstructable history.")
        return
    earliest_result_date, latest_result_date = _result_date_bounds(selected_results)

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

    if selected_performance_strategies:
        st.plotly_chart(
            make_equity_drawdown_figure(
                selected_results,
                strategy_names=list(selected_results),
                start=window_start,
                end=window_end,
                rebase=True,
                title=f"Growth of $1: {window_start.date()} to {window_end.date()}",
            ),
            use_container_width=True,
        )
        window_stats = window_performance_frame(
            selected_results,
            strategy_names=list(selected_results),
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

    st.subheader("Selected Full-History Equity and Drawdown")
    st.plotly_chart(make_equity_drawdown_figure(selected_results), use_container_width=True)

    st.subheader("Latest Positions")
    positions = latest_positions_frame(selected_results)
    st.dataframe(positions.map(lambda value: f"{value:.2%}"), use_container_width=True)


def _performance_option_frame(
    baseline_run: BaselineRun,
    *,
    bot_config: Any | None,
    experiment_scorecards: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    catalog = _safe_approach_catalog(bot_config)
    catalog_by_strategy = _catalog_by_strategy(catalog)

    for strategy_name in baseline_run.results:
        seen.add(strategy_name)
        rows.append(
            {
                "strategy": strategy_name,
                "label": _baseline_strategy_label(strategy_name, catalog_by_strategy),
                "source": "baseline",
                "curation_rank": pd.NA,
                "approach_id": f"baseline::{strategy_name}",
            }
        )

    curated_rank = _curated_strategy_rank_map(experiment_scorecards)
    for strategy_name, rank in sorted(curated_rank.items(), key=lambda item: item[1]):
        if strategy_name in seen:
            continue
        row = catalog_by_strategy.get(strategy_name)
        if row is None:
            continue
        seen.add(strategy_name)
        rows.append(
            {
                "strategy": strategy_name,
                "label": _curated_strategy_label(row, rank),
                "source": str(row.get("source", "experiment")),
                "curation_rank": rank,
                "approach_id": str(row.get("approach_id", f"experiment::{strategy_name}")),
            }
        )

    return pd.DataFrame(rows)


def _safe_approach_catalog(bot_config: Any | None) -> pd.DataFrame:
    if bot_config is None:
        return pd.DataFrame()
    try:
        catalog = build_approach_catalog(bot_config)
    except (OSError, ValueError, TypeError, AttributeError):
        return pd.DataFrame()
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    return catalog


def _catalog_by_strategy(catalog: pd.DataFrame) -> dict[str, pd.Series]:
    if catalog.empty or "strategy" not in catalog:
        return {}
    rows: dict[str, pd.Series] = {}
    for _, row in catalog.iterrows():
        strategy_name = str(row.get("strategy", ""))
        if strategy_name and strategy_name not in rows:
            rows[strategy_name] = row
    return rows


def _curated_strategy_rank_map(experiment_scorecards: pd.DataFrame | None) -> dict[str, int]:
    if experiment_scorecards is None or experiment_scorecards.empty:
        return {}
    curated = select_curated_strategy_shelf(
        rank_strategy_candidates(experiment_scorecards),
        limit=DEFAULT_CURATED_SHELF_LIMIT,
    )
    if curated.empty or "strategy" not in curated:
        return {}
    return {
        str(row["strategy"]): int(row["curation_rank"])
        for _, row in curated.iterrows()
        if pd.notna(row.get("strategy")) and pd.notna(row.get("curation_rank"))
    }


def _baseline_strategy_label(
    strategy_name: str,
    catalog_by_strategy: dict[str, pd.Series],
) -> str:
    row = catalog_by_strategy.get(strategy_name)
    display_name = str(row.get("display_name", strategy_name)) if row is not None else strategy_name
    return f"configured | {display_name}"


def _curated_strategy_label(row: pd.Series, rank: int) -> str:
    strategy_name = str(row.get("strategy", ""))
    display_name = str(row.get("display_name", strategy_name) or strategy_name)
    decision = str(row.get("promotion_decision", "candidate") or "candidate")
    return f"curated #{rank:02d} | {display_name} | {decision}"


def _selected_performance_results(
    selected_strategy_names: list[str],
    *,
    baseline_run: BaselineRun,
    bot_config: Any | None,
    option_frame: pd.DataFrame,
) -> tuple[dict[str, BacktestResult], list[str]]:
    results: dict[str, BacktestResult] = {}
    missing: list[str] = []
    if not selected_strategy_names:
        return results, missing

    catalog = _safe_approach_catalog(bot_config)
    catalog_by_strategy = _catalog_by_strategy(catalog)
    selected = option_frame[option_frame["strategy"].isin(selected_strategy_names)]
    for _, option in selected.iterrows():
        strategy_name = str(option["strategy"])
        if strategy_name in baseline_run.results:
            results[strategy_name] = baseline_run.results[strategy_name]
            continue
        row = catalog_by_strategy.get(strategy_name)
        if row is None or bot_config is None:
            missing.append(strategy_name)
            continue
        result = _cached_catalog_result(strategy_name, row, baseline_run, bot_config)
        if result is None:
            missing.append(strategy_name)
        else:
            results[strategy_name] = result
    return results, missing


def _cached_catalog_result(
    strategy_name: str,
    row: pd.Series,
    baseline_run: BaselineRun,
    bot_config: Any,
) -> BacktestResult | None:
    cache_key = _catalog_result_cache_key(row)
    cache = st.session_state.setdefault("_performance_result_cache", {})
    if cache_key not in cache:
        cache[cache_key] = _catalog_result(strategy_name, row, baseline_run, bot_config)
        if len(cache) > 32:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
    return cache[cache_key]


def _catalog_result_cache_key(row: pd.Series) -> str:
    payload = {
        column: str(row.get(column, ""))
        for column in [
            "approach_id",
            "strategy",
            "strategy_json",
            "scenario_sizing_json",
            "future_state_model_json",
            "strategy_drawdown_model_json",
            "decision_sanity_json",
        ]
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _catalog_result(
    strategy_name: str,
    row: pd.Series,
    baseline_run: BaselineRun,
    bot_config: Any,
) -> BacktestResult | None:
    try:
        strategy = strategy_from_catalog_row(row)
        execution = execution_for_catalog_row(row, bot_config.execution)
        result, _missing_columns = build_approach_backtest_result(
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
        return None
    return result
