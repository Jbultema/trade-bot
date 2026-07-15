from __future__ import annotations

from typing import Any

import pandas as pd

from trade_bot.dashboard.formatting import _format_decimal, _format_percent
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_judgement import defensive_judgement_scorecard
from trade_bot.research.strategy_naming import strategy_display_name
from trade_bot.research.strategy_outcome_utility import (
    add_outcome_frontier_flags,
    enrich_strategy_outcome_utility,
)


def outcome_candidate_scorecards(
    *,
    baseline_run: BaselineRun,
    bot_config: Any,
    experiment_scorecards: pd.DataFrame,
    include_defensive_judgement: bool = True,
) -> pd.DataFrame:
    runtime_scorecards = runtime_outcome_scorecards(
        baseline_run=baseline_run,
        bot_config=bot_config,
        include_defensive_judgement=include_defensive_judgement,
    )
    frames = [frame for frame in [runtime_scorecards, experiment_scorecards] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "strategy" not in combined:
        return pd.DataFrame()
    combined["_runtime_source_rank"] = (
        combined.get("source", pd.Series("", index=combined.index))
        .astype(str)
        .eq("latest_runtime_snapshot")
        .map({True: 0, False: 1})
    )
    return (
        combined.sort_values(["_runtime_source_rank", "strategy"], na_position="last")
        .drop_duplicates("strategy", keep="first")
        .drop(columns=["_runtime_source_rank"], errors="ignore")
        .reset_index(drop=True)
    )


def runtime_outcome_scorecards(
    *,
    baseline_run: BaselineRun,
    bot_config: Any,
    include_defensive_judgement: bool = True,
) -> pd.DataFrame:
    metrics = getattr(baseline_run, "metrics", pd.DataFrame())
    if metrics.empty:
        return pd.DataFrame()
    configured_strategies = set(getattr(bot_config, "strategies", {}) or {})
    runtime = metrics.reset_index().copy()
    if "name" in runtime and "strategy" not in runtime:
        runtime = runtime.rename(columns={"name": "strategy"})
    elif "strategy" not in runtime:
        runtime = runtime.rename(columns={runtime.columns[0]: "strategy"})
    if configured_strategies:
        runtime = runtime[runtime["strategy"].astype(str).isin(configured_strategies)].copy()
    if runtime.empty:
        return runtime
    runtime["source"] = "latest_runtime_snapshot"
    runtime["phase"] = "configured"
    runtime["family"] = "baseline_runtime"
    runtime["role"] = "configured_strategy"
    runtime["promotion_decision"] = "runtime_snapshot"
    runtime["research_status"] = "runtime_snapshot"
    runtime["monitoring_readiness_label"] = "snapshot_ready"
    runtime["operability_label"] = "paper_operable"
    runtime["hypothesis"] = (
        "Configured strategy from the latest runtime snapshot; included so the current app "
        "run and newest frontier variants can be compared against migrated experiments."
    )
    runtime["display_name"] = runtime["strategy"].astype(str).map(
        lambda strategy: strategy_display_name(
            strategy,
            family="baseline_runtime",
            phase="configured",
        )
    )
    for window, output_column in {
        "1y": "worst_1y_cagr",
        "3y": "worst_3y_cagr",
        "5y": "worst_5y_cagr",
    }.items():
        window_values = _runtime_window_values(
            baseline_run,
            window=window,
            column="worst_cagr",
        )
        if not window_values.empty:
            runtime[output_column] = runtime["strategy"].astype(str).map(window_values)
    for window, output_column in {
        "1y": "positive_1y_window_rate",
        "3y": "positive_3y_window_rate",
        "5y": "positive_5y_window_rate",
    }.items():
        window_values = _runtime_window_values(
            baseline_run,
            window=window,
            column="positive_window_rate",
        )
        if not window_values.empty:
            runtime[output_column] = runtime["strategy"].astype(str).map(window_values)
    if include_defensive_judgement:
        defensive_scorecards = _runtime_defensive_judgement_values(baseline_run)
        if not defensive_scorecards.empty:
            runtime = runtime.merge(defensive_scorecards, on="strategy", how="left")
    return runtime


def runtime_benchmark_metrics(baseline_run: BaselineRun) -> pd.DataFrame:
    """Return runtime benchmark rows using the names expected by outcome utility."""

    metrics = getattr(baseline_run, "metrics", pd.DataFrame())
    if metrics.empty:
        return pd.DataFrame()
    metrics = _indexed_metrics(metrics)
    benchmark_rows: dict[str, pd.Series] = {}
    for ticker, runtime_name in {"spy": "buy_hold_spy", "qqq": "buy_hold_qqq"}.items():
        if runtime_name in metrics.index:
            benchmark_rows[f"benchmark_{ticker}"] = metrics.loc[runtime_name].copy()
    if not benchmark_rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(benchmark_rows, orient="index")


def _indexed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    if "name" in metrics.columns:
        return metrics.set_index("name", drop=False)
    if "strategy" in metrics.columns:
        return metrics.set_index("strategy", drop=False)
    return metrics


def _runtime_window_values(
    baseline_run: BaselineRun,
    *,
    window: str,
    column: str,
) -> pd.Series:
    window_summary = getattr(baseline_run, "window_summary", pd.DataFrame())
    if window_summary.empty or column not in window_summary:
        return pd.Series(dtype=float)
    frame = window_summary.reset_index()
    strategy_column = "strategy" if "strategy" in frame.columns else "name"
    if strategy_column not in frame or "window" not in frame:
        return pd.Series(dtype=float)
    selected = frame[frame["window"].astype(str).eq(window)].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    values = pd.to_numeric(selected.set_index(strategy_column)[column], errors="coerce")
    return values[~values.index.duplicated(keep="last")]


def _runtime_defensive_judgement_values(baseline_run: BaselineRun) -> pd.DataFrame:
    prices = getattr(baseline_run, "prices", pd.DataFrame())
    results = getattr(baseline_run, "results", {}) or {}
    if prices.empty or not results:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        scorecard = defensive_judgement_scorecard(result, prices)
        if "QQQ" in prices:
            qqq_scorecard = defensive_judgement_scorecard(
                result,
                prices,
                benchmark_ticker="QQQ",
            )
            for column, value in qqq_scorecard.items():
                if column in {
                    "defensive_correct_rate",
                    "defensive_false_alarm_rate",
                    "defensive_mixed_rate",
                    "defensive_avg_benchmark_excess_vs_cash",
                    "defensive_median_forward_drawdown",
                    "defensive_episode_starts",
                    "defensive_judgement_label",
                }:
                    scorecard[f"qqq_{column}"] = value
        scorecard["strategy"] = strategy
        rows.append(scorecard)
    return pd.DataFrame(rows)


def outcome_strategy_option_frame(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    limit: int = 80,
    include_defensive_judgement: bool = True,
) -> pd.DataFrame:
    scorecards = outcome_candidate_scorecards(
        baseline_run=baseline_run,
        bot_config=bot_config,
        experiment_scorecards=experiment_scorecards,
        include_defensive_judgement=include_defensive_judgement,
    )
    if scorecards.empty or not {"strategy", "cagr", "max_drawdown"}.issubset(
        scorecards.columns
    ):
        return pd.DataFrame()
    frame = add_outcome_frontier_flags(
        enrich_strategy_outcome_utility(
            scorecards,
            benchmark_metrics=runtime_benchmark_metrics(baseline_run),
        )
    ).copy()
    if "research_status" in frame:
        active = frame[~frame["research_status"].astype(str).eq("pruned_dead_end")].copy()
        if not active.empty:
            frame = active
    frame = frame.sort_values("growth_constrained_utility_score", ascending=False).head(limit)
    frame["simulation_label"] = frame.apply(scorecard_option_label, axis=1)
    return frame.reset_index(drop=True)


def scorecard_option_label(row: pd.Series) -> str:
    return (
        f"{row.get('display_name', row.get('strategy', 'Strategy'))} | "
        f"utility {_format_decimal(row.get('growth_constrained_utility_score'))} | "
        f"CAGR {_format_percent(row.get('cagr'))} | "
        f"DD {_format_percent(row.get('max_drawdown'))}"
    )
