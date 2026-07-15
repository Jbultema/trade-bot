from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.DEFAULTS import DEFAULT_EXPERIMENTS_DIR
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
from trade_bot.research.forward_simulation import build_regime_return_library

TECH_LEADERSHIP_TICKERS = {
    "QQQ",
    "SMH",
    "SOXX",
    "IGV",
    "XLK",
    "VGT",
    "NVDA",
    "AVGO",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "GOOG",
    "GOOGL",
    "TSLA",
    "AMD",
    "ARM",
    "ASML",
    "TSM",
    "PLTR",
}
TECH_BUCKET_TICKERS = ("QQQ", "SMH", "SOXX", "IGV")
MEGA_CAP_TECH_TICKERS = {
    "NVDA",
    "AVGO",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "GOOG",
    "GOOGL",
    "TSLA",
    "AMD",
    "ARM",
    "ASML",
    "TSM",
    "PLTR",
}
DEFAULT_BETA_TICKERS = ("QQQ", "SMH", "SOXX", "SPY", "VEA", "IWM", "GLD", "TLT")
DEFAULT_ROUTER_HORIZONS = (21, 63, 126)
DEFAULT_ROUTER_TOP_K = 3
DEFAULT_ROUTER_SHRINKAGE = 0.35
DEFAULT_ROUTER_NEIGHBORS = 80


@dataclass(frozen=True)
class LeadershipDiagnosticsRun:
    output_dir: Path
    artifacts: dict[str, Path]
    selected_strategies: tuple[str, ...]
    tech_dependence: pd.DataFrame
    factor_betas: pd.DataFrame
    impairment: pd.DataFrame
    scenario_heatmap: pd.DataFrame
    router_summary: pd.DataFrame
    router_selection: pd.DataFrame
    router_scenarios: pd.DataFrame
    router_comparison: pd.DataFrame
    router_scores: pd.DataFrame
    readout: str


def run_leadership_diagnostics(
    *,
    config: BotConfig,
    prices: pd.DataFrame,
    output_dir: str | Path = "reports/leadership_diagnostics",
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    strategies: tuple[str, ...] = (),
    top_n: int = 12,
    router_horizons: tuple[int, ...] = DEFAULT_ROUTER_HORIZONS,
    min_train_days: int = 756,
    origin_step_days: int = 126,
) -> LeadershipDiagnosticsRun:
    """Audit whether top strategies depend too much on one tech leadership regime.

    The diagnostics are intentionally interpretability and validation outputs.
    They do not alter strategy weights or daily trade decisions.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_approach_catalog(config, experiment_root=experiment_root)
    selected_rows = select_leadership_diagnostic_rows(
        config,
        catalog,
        strategies=strategies,
        top_n=top_n,
    )
    results = _build_results_for_rows(config, prices, selected_rows)
    if not results:
        raise ValueError("No strategy results could be built for leadership diagnostics.")

    selected_names = tuple(results)
    benchmark_returns = _benchmark_returns(prices)
    tech_dependence = build_tech_dependence_frame(results)
    factor_betas = build_factor_beta_frame(results, benchmark_returns)
    contribution = build_return_contribution_frame(results, prices)
    underperformance = build_qqq_underperformance_frame(results, prices)
    impairment = run_leadership_impairment(results, prices)
    scenario_heatmap = build_scenario_strategy_heatmap(results, prices)
    router_folds, router_summary, router_scores = run_walk_forward_strategy_router(
        results,
        prices,
        horizons=router_horizons,
        min_train_days=min_train_days,
        origin_step_days=origin_step_days,
    )
    router_selection = build_router_selection_summary(router_folds)
    router_scenarios = build_router_scenario_summary(router_folds)
    router_comparison = build_router_model_comparison(router_folds)

    artifacts = {
        "tech_dependence": output / "strategy_tech_dependence.csv",
        "factor_betas": output / "strategy_factor_betas.csv",
        "return_contribution": output / "strategy_return_contribution.csv",
        "qqq_underperformance": output / "qqq_underperformance_periods.csv",
        "leadership_impairment": output / "leadership_impairment.csv",
        "scenario_heatmap": output / "scenario_strategy_heatmap.csv",
        "router_folds": output / "walk_forward_router_folds.csv",
        "router_summary": output / "walk_forward_router_summary.csv",
        "router_selection": output / "walk_forward_router_selection.csv",
        "router_scenarios": output / "walk_forward_router_scenarios.csv",
        "router_comparison": output / "walk_forward_router_comparison.csv",
        "router_scores": output / "walk_forward_router_scores.csv",
    }
    frames = {
        "tech_dependence": tech_dependence,
        "factor_betas": factor_betas,
        "return_contribution": contribution,
        "qqq_underperformance": underperformance,
        "leadership_impairment": impairment,
        "scenario_heatmap": scenario_heatmap,
        "router_folds": router_folds,
        "router_summary": router_summary,
        "router_selection": router_selection,
        "router_scenarios": router_scenarios,
        "router_comparison": router_comparison,
        "router_scores": router_scores,
    }
    for name, frame in frames.items():
        frame.to_csv(artifacts[name], index=False)

    readout = _markdown_readout(
        selected_names=selected_names,
        tech_dependence=tech_dependence,
        factor_betas=factor_betas,
        impairment=impairment,
        scenario_heatmap=scenario_heatmap,
        router_summary=router_summary,
        router_selection=router_selection,
        router_scenarios=router_scenarios,
        router_comparison=router_comparison,
    )
    summary_path = output / "summary.md"
    summary_path.write_text(readout, encoding="utf-8")
    artifacts["summary"] = summary_path
    return LeadershipDiagnosticsRun(
        output_dir=output,
        artifacts=artifacts,
        selected_strategies=selected_names,
        tech_dependence=tech_dependence,
        factor_betas=factor_betas,
        impairment=impairment,
        scenario_heatmap=scenario_heatmap,
        router_summary=router_summary,
        router_selection=router_selection,
        router_scenarios=router_scenarios,
        router_comparison=router_comparison,
        router_scores=router_scores,
        readout=readout,
    )


def select_leadership_diagnostic_rows(
    config: BotConfig,
    catalog: pd.DataFrame,
    *,
    strategies: tuple[str, ...] = (),
    top_n: int = 12,
) -> pd.DataFrame:
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    frame = catalog.copy()
    frame["strategy"] = frame["strategy"].astype(str)
    if strategies:
        requested = list(dict.fromkeys(str(strategy) for strategy in strategies))
        return frame[frame["strategy"].isin(requested)].drop_duplicates("strategy")

    priority: list[str] = [config.primary_strategy] if config.primary_strategy else []

    scored = frame.copy()
    sort_column = _first_available_column(
        scored,
        (
            "growth_constrained_utility_score",
            "promotion_score",
            "cagr",
            "calmar",
        ),
    )
    if sort_column:
        scored["_diagnostic_sort"] = pd.to_numeric(scored[sort_column], errors="coerce")
        scored = scored.sort_values("_diagnostic_sort", ascending=False)
    top_scored = [
        name for name in scored["strategy"].astype(str).tolist() if name not in set(priority)
    ][:top_n]
    selected_names = list(dict.fromkeys([*priority, *top_scored]))
    return frame[frame["strategy"].isin(selected_names)].drop_duplicates("strategy")


def required_tickers_for_catalog_rows(rows: pd.DataFrame) -> set[str]:
    tickers: set[str] = set()
    for _, row in rows.iterrows():
        try:
            strategy = strategy_from_catalog_row(row)
        except (TypeError, ValueError):
            continue
        tickers.update(strategy.tickers)
        tickers.update(strategy.satellite_tickers)
        if strategy.defensive_ticker:
            tickers.add(strategy.defensive_ticker)
    return tickers


def leadership_candidate_tickers(
    config: BotConfig,
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    strategies: tuple[str, ...] = (),
    top_n: int = 12,
) -> set[str]:
    catalog = build_approach_catalog(config, experiment_root=experiment_root)
    selected = select_leadership_diagnostic_rows(
        config,
        catalog,
        strategies=strategies,
        top_n=top_n,
    )
    return set(configured_tickers(config)) | required_tickers_for_catalog_rows(selected)


def build_tech_dependence_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        weights = result.weights.copy()
        tech_cols = [column for column in weights.columns if column in TECH_LEADERSHIP_TICKERS]
        mega_cols = [column for column in weights.columns if column in MEGA_CAP_TECH_TICKERS]
        row: dict[str, object] = {
            "strategy": strategy,
            "avg_tech_ai_weight": _row_sum(weights, tech_cols).mean(),
            "current_tech_ai_weight": _last_value(_row_sum(weights, tech_cols)),
            "max_tech_ai_weight": _row_sum(weights, tech_cols).max(),
            "avg_mega_cap_tech_weight": _row_sum(weights, mega_cols).mean(),
            "current_mega_cap_tech_weight": _last_value(_row_sum(weights, mega_cols)),
            "avg_non_tech_weight": 1.0 - _row_sum(weights, tech_cols).mean(),
            "current_non_tech_weight": 1.0 - _last_value(_row_sum(weights, tech_cols)),
        }
        for ticker in TECH_BUCKET_TICKERS:
            if ticker in weights:
                row[f"avg_{ticker.lower()}_weight"] = float(weights[ticker].mean())
                row[f"current_{ticker.lower()}_weight"] = float(weights[ticker].iloc[-1])
            else:
                row[f"avg_{ticker.lower()}_weight"] = 0.0
                row[f"current_{ticker.lower()}_weight"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_tech_ai_weight", ascending=False)


def build_factor_beta_frame(
    results: dict[str, BacktestResult],
    benchmark_returns: pd.DataFrame,
    *,
    beta_tickers: tuple[str, ...] = DEFAULT_BETA_TICKERS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        strategy_returns = pd.to_numeric(result.returns, errors="coerce").dropna()
        for ticker in beta_tickers:
            if ticker not in benchmark_returns:
                continue
            aligned = pd.concat(
                [strategy_returns.rename("strategy"), benchmark_returns[ticker].rename("benchmark")],
                axis=1,
                join="inner",
            ).dropna()
            if len(aligned) < 30:
                beta = float("nan")
                correlation = float("nan")
            else:
                variance = float(aligned["benchmark"].var())
                beta = (
                    float(aligned["strategy"].cov(aligned["benchmark"]) / variance)
                    if abs(variance) > 1e-12
                    else float("nan")
                )
                correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
            rows.append(
                {
                    "strategy": strategy,
                    "factor": ticker,
                    "beta": beta,
                    "correlation": correlation,
                    "observations": int(len(aligned)),
                }
            )
    return pd.DataFrame(rows)


def build_return_contribution_frame(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        contributions = _asset_contributions(result, asset_returns)
        if contributions.empty:
            continue
        total = float(contributions.sum())
        abs_total = float(contributions.abs().sum())
        positive_total = float(contributions.clip(lower=0.0).sum())
        for ticker, contribution in contributions.sort_values(key=lambda s: s.abs(), ascending=False).items():
            is_tech = ticker in TECH_LEADERSHIP_TICKERS
            rows.append(
                {
                    "strategy": strategy,
                    "ticker": ticker,
                    "is_tech_ai": is_tech,
                    "return_contribution": float(contribution),
                    "share_of_total_contribution": _safe_ratio(float(contribution), total),
                    "share_of_abs_contribution": _safe_ratio(abs(float(contribution)), abs_total),
                    "share_of_positive_contribution": _safe_ratio(
                        max(float(contribution), 0.0),
                        positive_total,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_qqq_underperformance_frame(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    *,
    lookback_days: int = 63,
    peers: tuple[str, ...] = ("SPY", "RSP", "VEA"),
) -> pd.DataFrame:
    if "QQQ" not in prices:
        return pd.DataFrame()
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    qqq_trend = _rolling_compound(asset_returns["QQQ"], lookback_days)
    rows: list[dict[str, object]] = []
    for peer in peers:
        if peer not in asset_returns:
            continue
        condition = qqq_trend < _rolling_compound(asset_returns[peer], lookback_days)
        for strategy, result in results.items():
            strategy_returns = result.returns.reindex(condition.index).fillna(0.0)
            benchmark_returns = asset_returns[peer].reindex(condition.index).fillna(0.0)
            rows.append(
                _filtered_performance_row(
                    strategy=strategy,
                    scenario=f"QQQ underperformed {peer} over trailing {lookback_days}d",
                    returns=strategy_returns,
                    benchmark_returns=benchmark_returns,
                    mask=condition.fillna(False),
                )
            )
    return pd.DataFrame(rows)


def run_leadership_impairment(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    scenarios = {
        "tech_returns_haircut_25pct": {
            "tech_return_multiplier": 0.75,
            "tech_weight_multiplier": 1.0,
            "boost": (),
        },
        "half_tech_weight_to_global_breadth": {
            "tech_return_multiplier": 1.0,
            "tech_weight_multiplier": 0.50,
            "boost": ("VEA", "RSP", "IWM", "ACWX"),
        },
        "half_tech_weight_to_real_assets": {
            "tech_return_multiplier": 1.0,
            "tech_weight_multiplier": 0.50,
            "boost": ("GLD", "DBC", "XLE", "TLT"),
        },
        "half_tech_weight_to_global_blend": {
            "tech_return_multiplier": 1.0,
            "tech_weight_multiplier": 0.50,
            "boost": ("VEA", "VWO", "ACWX", "IWM", "GLD", "RSP"),
        },
    }
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        base_metrics = _metrics_from_returns(strategy, result.returns)
        rows.append(_metrics_output_row(strategy, "native", base_metrics, base_metrics))
        for scenario, settings in scenarios.items():
            stressed_returns = _stressed_strategy_returns(
                result,
                asset_returns,
                tech_return_multiplier=float(settings["tech_return_multiplier"]),
                tech_weight_multiplier=float(settings["tech_weight_multiplier"]),
                boost_tickers=tuple(settings["boost"]),
            )
            if stressed_returns.empty:
                continue
            metrics = _metrics_from_returns(f"{strategy}:{scenario}", stressed_returns)
            rows.append(_metrics_output_row(strategy, scenario, metrics, base_metrics))
    return pd.DataFrame(rows)


def build_scenario_strategy_heatmap(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    *,
    benchmark_ticker: str = "QQQ",
) -> pd.DataFrame:
    labels = _scenario_labels(prices)
    if labels.empty:
        return pd.DataFrame()
    benchmark = _benchmark_returns(prices)
    benchmark_returns = (
        benchmark[benchmark_ticker]
        if benchmark_ticker in benchmark
        else pd.Series(0.0, index=labels.index)
    )
    rows: list[dict[str, object]] = []
    for strategy, result in results.items():
        returns = result.returns.reindex(labels.index).fillna(0.0)
        for scenario in sorted(labels.dropna().unique()):
            mask = labels.eq(scenario)
            row = _filtered_performance_row(
                strategy=strategy,
                scenario=str(scenario),
                returns=returns,
                benchmark_returns=benchmark_returns.reindex(labels.index).fillna(0.0),
                mask=mask,
            )
            row["benchmark"] = benchmark_ticker
            rows.append(row)
    return pd.DataFrame(rows)


def run_walk_forward_strategy_router(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_ROUTER_HORIZONS,
    min_train_days: int = 756,
    origin_step_days: int = 63,
    top_k_blend: int = DEFAULT_ROUTER_TOP_K,
    shrinkage: float = DEFAULT_ROUTER_SHRINKAGE,
    max_state_neighbors: int = DEFAULT_ROUTER_NEIGHBORS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = _scenario_labels(prices)
    if labels.empty or not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    state_features = _router_state_features(prices).reindex(labels.index).ffill()
    benchmark_ticker = "QQQ" if "QQQ" in prices else "SPY" if "SPY" in prices else ""
    benchmark_returns = (
        prices[benchmark_ticker].pct_change(fill_method=None).fillna(0.0)
        if benchmark_ticker
        else pd.Series(0.0, index=labels.index)
    )
    common_dates = labels.index
    rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for horizon in horizons:
        if len(common_dates) <= min_train_days + horizon:
            continue
        embargo_days = max(5, horizon)
        for origin_pos in range(min_train_days, len(common_dates) - horizon, origin_step_days):
            origin_date = common_dates[origin_pos]
            scenario = str(labels.iloc[origin_pos])
            candidate_scores = _router_candidate_scores(
                results,
                benchmark_returns,
                labels,
                state_features,
                origin_date=origin_date,
                scenario=scenario,
                horizon_days=horizon,
                embargo_days=embargo_days,
                max_state_neighbors=max_state_neighbors,
            )
            if candidate_scores.empty:
                continue
            ranked = candidate_scores.sort_values("score", ascending=False)
            selected = str(ranked.iloc[0]["strategy"])
            blend_names = ranked.head(top_k_blend)["strategy"].astype(str).tolist()
            shrink_weights = _router_shrinkage_weights(
                ranked,
                all_names=tuple(results),
                top_k=top_k_blend,
                shrinkage=shrinkage,
            )
            prior_scores = _prior_static_scores(
                results,
                benchmark_returns,
                origin_date=origin_date,
                horizon_days=horizon,
                embargo_days=embargo_days,
            )
            prior_best = _prior_best_from_scores(prior_scores)
            prior_top3_weights = _prior_top_k_weights(prior_scores, top_k=top_k_blend)
            selected_return = _forward_return(results[selected].returns, origin_date, horizon)
            blend_return = float(
                np.nanmean(
                    [
                        _forward_return(results[name].returns, origin_date, horizon)
                        for name in blend_names
                    ]
                )
            )
            shrink_blend_return = _weighted_forward_return(
                results,
                shrink_weights,
                origin_date=origin_date,
                horizon_days=horizon,
            )
            equal_return = float(
                np.nanmean(
                    [
                        _forward_return(result.returns, origin_date, horizon)
                        for result in results.values()
                    ]
                )
            )
            prior_best_return = (
                _forward_return(results[prior_best].returns, origin_date, horizon)
                if prior_best
                else float("nan")
            )
            prior_top3_return = _weighted_forward_return(
                results,
                prior_top3_weights,
                origin_date=origin_date,
                horizon_days=horizon,
            )
            benchmark_forward = _forward_return(benchmark_returns, origin_date, horizon)
            for rank, (_, score_row) in enumerate(ranked.iterrows(), start=1):
                strategy = str(score_row["strategy"])
                score_payload = score_row.to_dict()
                score_payload.update(
                    {
                        "origin_date": str(origin_date.date()),
                        "horizon_days": horizon,
                        "scenario_bucket": scenario,
                        "rank": rank,
                        "selected": strategy == selected,
                        "shrinkage_weight": shrink_weights.get(strategy, 0.0),
                        "prior_best_baseline": strategy == prior_best,
                    }
                )
                score_rows.append(score_payload)
            rows.append(
                {
                    "origin_date": str(origin_date.date()),
                    "horizon_days": horizon,
                    "scenario_bucket": scenario,
                    "selected_strategy": selected,
                    "selected_score": float(ranked.iloc[0]["score"]),
                    "score_source": str(ranked.iloc[0].get("score_source", "")),
                    "selected_forward_return": selected_return,
                    "selected_excess_vs_benchmark": selected_return - benchmark_forward,
                    "top3_blend_forward_return": blend_return,
                    "top3_blend_excess_vs_benchmark": blend_return - benchmark_forward,
                    "shrinkage_blend_forward_return": shrink_blend_return,
                    "shrinkage_blend_excess_vs_benchmark": shrink_blend_return - benchmark_forward,
                    "prior_best_strategy": prior_best,
                    "prior_best_forward_return": prior_best_return,
                    "prior_best_excess_vs_benchmark": prior_best_return - benchmark_forward,
                    "prior_top3_blend_forward_return": prior_top3_return,
                    "prior_top3_blend_excess_vs_benchmark": prior_top3_return - benchmark_forward,
                    "equal_candidate_forward_return": equal_return,
                    "equal_candidate_excess_vs_benchmark": equal_return - benchmark_forward,
                    "benchmark_ticker": benchmark_ticker,
                    "benchmark_forward_return": benchmark_forward,
                    "candidate_count": int(len(candidate_scores)),
                    "similar_prior_windows": int(ranked.iloc[0]["similar_windows"]),
                    "scenario_prior_windows": int(ranked.iloc[0].get("scenario_windows", 0)),
                    "fallback_windows": int(ranked.iloc[0]["fallback_windows"]),
                    "embargo_days": embargo_days,
                }
            )
    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame(), pd.DataFrame(score_rows)
    summary = (
        folds.groupby("horizon_days")
        .agg(
            folds=("origin_date", "count"),
            selected_mean_forward_return=("selected_forward_return", "mean"),
            selected_hit_rate=("selected_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            selected_mean_excess_vs_benchmark=("selected_excess_vs_benchmark", "mean"),
            top3_blend_mean_forward_return=("top3_blend_forward_return", "mean"),
            top3_blend_hit_rate=("top3_blend_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            top3_blend_mean_excess_vs_benchmark=("top3_blend_excess_vs_benchmark", "mean"),
            shrinkage_blend_mean_forward_return=("shrinkage_blend_forward_return", "mean"),
            shrinkage_blend_hit_rate=(
                "shrinkage_blend_excess_vs_benchmark",
                lambda s: float((s > 0).mean()),
            ),
            shrinkage_blend_mean_excess_vs_benchmark=(
                "shrinkage_blend_excess_vs_benchmark",
                "mean",
            ),
            prior_best_mean_forward_return=("prior_best_forward_return", "mean"),
            prior_best_hit_rate=("prior_best_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            prior_best_mean_excess_vs_benchmark=("prior_best_excess_vs_benchmark", "mean"),
            prior_top3_mean_forward_return=("prior_top3_blend_forward_return", "mean"),
            prior_top3_hit_rate=(
                "prior_top3_blend_excess_vs_benchmark",
                lambda s: float((s > 0).mean()),
            ),
            prior_top3_mean_excess_vs_benchmark=("prior_top3_blend_excess_vs_benchmark", "mean"),
            equal_candidate_mean_forward_return=("equal_candidate_forward_return", "mean"),
            equal_candidate_mean_excess_vs_benchmark=("equal_candidate_excess_vs_benchmark", "mean"),
            benchmark_mean_forward_return=("benchmark_forward_return", "mean"),
            mean_similar_prior_windows=("similar_prior_windows", "mean"),
            fallback_share=("fallback_windows", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return folds, summary, pd.DataFrame(score_rows)


def build_router_selection_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Summarize which strategy the prior-only router preferred at each horizon."""

    if folds.empty or "selected_strategy" not in folds:
        return pd.DataFrame()
    grouped = (
        folds.groupby(["horizon_days", "selected_strategy"])
        .agg(
            selected_count=("origin_date", "count"),
            mean_selected_score=("selected_score", "mean"),
            mean_forward_return=("selected_forward_return", "mean"),
            hit_rate=("selected_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            mean_excess_vs_benchmark=("selected_excess_vs_benchmark", "mean"),
            mean_benchmark_forward_return=("benchmark_forward_return", "mean"),
            mean_similar_prior_windows=("similar_prior_windows", "mean"),
            fallback_share=("fallback_windows", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .rename(columns={"selected_strategy": "strategy"})
    )
    horizon_counts = folds.groupby("horizon_days")["origin_date"].count().rename("horizon_folds")
    grouped = grouped.merge(horizon_counts, on="horizon_days", how="left")
    grouped["selection_rate"] = grouped["selected_count"] / grouped["horizon_folds"].replace(0, np.nan)
    return grouped.sort_values(
        ["horizon_days", "selected_count", "mean_excess_vs_benchmark"],
        ascending=[True, False, False],
    )


def build_router_scenario_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Summarize router utility by the scenario bucket visible at each origin."""

    if folds.empty or "scenario_bucket" not in folds:
        return pd.DataFrame()
    return (
        folds.groupby(["horizon_days", "scenario_bucket"])
        .agg(
            folds=("origin_date", "count"),
            selected_mean_forward_return=("selected_forward_return", "mean"),
            selected_hit_rate=("selected_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            selected_mean_excess_vs_benchmark=("selected_excess_vs_benchmark", "mean"),
            top3_blend_mean_forward_return=("top3_blend_forward_return", "mean"),
            top3_blend_hit_rate=("top3_blend_excess_vs_benchmark", lambda s: float((s > 0).mean())),
            top3_blend_mean_excess_vs_benchmark=("top3_blend_excess_vs_benchmark", "mean"),
            shrinkage_blend_mean_forward_return=("shrinkage_blend_forward_return", "mean"),
            shrinkage_blend_hit_rate=(
                "shrinkage_blend_excess_vs_benchmark",
                lambda s: float((s > 0).mean()),
            ),
            shrinkage_blend_mean_excess_vs_benchmark=(
                "shrinkage_blend_excess_vs_benchmark",
                "mean",
            ),
            equal_candidate_mean_forward_return=("equal_candidate_forward_return", "mean"),
            benchmark_mean_forward_return=("benchmark_forward_return", "mean"),
            mean_similar_prior_windows=("similar_prior_windows", "mean"),
            fallback_share=("fallback_windows", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["horizon_days", "selected_mean_excess_vs_benchmark"], ascending=[True, False])
    )


def build_router_model_comparison(folds: pd.DataFrame) -> pd.DataFrame:
    """Compare router variants against static baselines by horizon."""

    if folds.empty:
        return pd.DataFrame()
    model_columns = {
        "state_router_pick": "selected_forward_return",
        "state_router_top3_equal": "top3_blend_forward_return",
        "state_router_shrinkage_blend": "shrinkage_blend_forward_return",
        "prior_best_static": "prior_best_forward_return",
        "prior_top3_static": "prior_top3_blend_forward_return",
        "equal_candidate_shelf": "equal_candidate_forward_return",
        "benchmark": "benchmark_forward_return",
    }
    rows: list[dict[str, object]] = []
    for horizon, group in folds.groupby("horizon_days"):
        benchmark = pd.to_numeric(group["benchmark_forward_return"], errors="coerce")
        for model, column in model_columns.items():
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            excess = values - benchmark
            rows.append(
                {
                    "horizon_days": int(horizon),
                    "model": model,
                    "folds": int(values.notna().sum()),
                    "mean_forward_return": float(values.mean()),
                    "mean_excess_vs_benchmark": float(excess.mean()),
                    "hit_rate_vs_benchmark": float((excess > 0).mean()),
                    "median_forward_return": float(values.median()),
                    "q25_forward_return": float(values.quantile(0.25)),
                    "q75_forward_return": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def _build_results_for_rows(
    config: BotConfig,
    prices: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, BacktestResult]:
    results: dict[str, BacktestResult] = {}
    for _, row in rows.iterrows():
        strategy_name = str(row.get("strategy", ""))
        if not strategy_name or strategy_name in results:
            continue
        try:
            strategy = strategy_from_catalog_row(row)
            execution = execution_for_catalog_row(row, config.execution)
            result, _metrics = build_approach_backtest_result(
                prices,
                strategy,
                execution,
                scenario_sizing=scenario_sizing_from_catalog_row(row),
                future_state_model=future_state_model_from_catalog_row(row),
                strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
                decision_sanity=decision_sanity_from_catalog_row(row),
                name=strategy_name,
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        results[strategy_name] = result
    return results


def _benchmark_returns(prices: pd.DataFrame) -> pd.DataFrame:
    available = [ticker for ticker in DEFAULT_BETA_TICKERS if ticker in prices]
    if not available:
        return pd.DataFrame(index=prices.index)
    return prices[available].pct_change(fill_method=None).fillna(0.0)


def _row_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return frame[columns].sum(axis=1).fillna(0.0)


def _last_value(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _asset_contributions(result: BacktestResult, asset_returns: pd.DataFrame) -> pd.Series:
    columns = [column for column in result.weights.columns if column in asset_returns]
    if not columns:
        return pd.Series(dtype=float)
    aligned_returns = asset_returns[columns].reindex(result.weights.index).fillna(0.0)
    aligned_weights = result.weights[columns].reindex(aligned_returns.index).fillna(0.0)
    return (aligned_weights * aligned_returns).sum(axis=0).sort_values(ascending=False)


def _rolling_compound(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).rolling(window, min_periods=max(5, window // 3)).apply(
        np.prod,
        raw=True,
    ) - 1.0


def _filtered_performance_row(
    *,
    strategy: str,
    scenario: str,
    returns: pd.Series,
    benchmark_returns: pd.Series,
    mask: pd.Series,
) -> dict[str, object]:
    selected_returns = pd.to_numeric(returns[mask], errors="coerce").dropna()
    selected_benchmark = pd.to_numeric(benchmark_returns[mask], errors="coerce").dropna()
    selected_benchmark = selected_benchmark.reindex(selected_returns.index).fillna(0.0)
    if selected_returns.empty:
        return {
            "strategy": strategy,
            "scenario_bucket": scenario,
            "observations": 0,
            "state_cagr": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "benchmark_excess": float("nan"),
        }
    equity = (1.0 + selected_returns).cumprod()
    benchmark_equity = (1.0 + selected_benchmark).cumprod()
    periods = max(len(selected_returns), 1)
    state_cagr = float(equity.iloc[-1] ** (252.0 / periods) - 1.0)
    benchmark_cagr = float(benchmark_equity.iloc[-1] ** (252.0 / periods) - 1.0)
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "strategy": strategy,
        "scenario_bucket": scenario,
        "observations": int(len(selected_returns)),
        "state_cagr": state_cagr,
        "max_drawdown": drawdown,
        "hit_rate": float((selected_returns > 0.0).mean()),
        "benchmark_excess": state_cagr - benchmark_cagr,
    }


def _stressed_strategy_returns(
    result: BacktestResult,
    asset_returns: pd.DataFrame,
    *,
    tech_return_multiplier: float,
    tech_weight_multiplier: float,
    boost_tickers: tuple[str, ...],
) -> pd.Series:
    columns = [column for column in result.weights.columns if column in asset_returns]
    if not columns:
        return pd.Series(dtype=float)
    weights = result.weights[columns].copy()
    returns = asset_returns[columns].reindex(weights.index).fillna(0.0).copy()
    tech_cols = [column for column in columns if column in TECH_LEADERSHIP_TICKERS]
    if tech_cols:
        returns.loc[:, tech_cols] = returns[tech_cols] * tech_return_multiplier
        if tech_weight_multiplier < 1.0:
            original_tech = weights[tech_cols].sum(axis=1)
            weights.loc[:, tech_cols] = weights[tech_cols] * tech_weight_multiplier
            freed = (original_tech - weights[tech_cols].sum(axis=1)).clip(lower=0.0)
            available_boost = [ticker for ticker in boost_tickers if ticker in asset_returns]
            if available_boost:
                boost_returns = asset_returns[available_boost].reindex(weights.index).fillna(0.0)
                for ticker in available_boost:
                    if ticker not in weights:
                        weights[ticker] = 0.0
                        returns[ticker] = boost_returns[ticker]
                add_each = freed / len(available_boost)
                for ticker in available_boost:
                    weights[ticker] = weights[ticker].add(add_each, fill_value=0.0)
    weights = weights.clip(lower=0.0)
    row_sum = weights.sum(axis=1)
    over = row_sum > 1.0
    weights.loc[over] = weights.loc[over].div(row_sum.loc[over], axis=0)
    return (weights.reindex(returns.index).fillna(0.0) * returns).sum(axis=1)


def _metrics_from_returns(name: str, returns: pd.Series) -> PerformanceMetrics:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    equity = (1.0 + clean).cumprod()
    turnover = pd.Series(0.0, index=clean.index)
    costs = pd.Series(0.0, index=clean.index)
    return calculate_metrics(name, clean, equity, turnover, costs)


def _metrics_output_row(
    strategy: str,
    scenario: str,
    metrics: PerformanceMetrics,
    base: PerformanceMetrics,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "scenario": scenario,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "calmar": metrics.calmar,
        "delta_cagr_vs_native": metrics.cagr - base.cagr,
        "delta_drawdown_vs_native": metrics.max_drawdown - base.max_drawdown,
    }


def _scenario_labels(prices: pd.DataFrame) -> pd.Series:
    market_ticker = "QQQ" if "QQQ" in prices else "SPY" if "SPY" in prices else ""
    if not market_ticker:
        return pd.Series(dtype=object)
    returns = prices[market_ticker].pct_change(fill_method=None).fillna(0.0)
    regime = build_regime_return_library(returns)["regime"].reindex(prices.index).ffill()
    labels = regime.fillna("transition").astype(str)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    if "QQQ" in asset_returns:
        qqq_trend = _rolling_compound(asset_returns["QQQ"], 63)
        peers = [ticker for ticker in ("SPY", "RSP", "VEA") if ticker in asset_returns]
        if peers:
            peer_trends = pd.concat(
                [_rolling_compound(asset_returns[ticker], 63).rename(ticker) for ticker in peers],
                axis=1,
            )
            labels.loc[qqq_trend < peer_trends.min(axis=1)] = "tech_leadership_impaired"
            labels.loc[qqq_trend > peer_trends.max(axis=1)] = "tech_leadership"
    return labels


def _router_state_features(prices: pd.DataFrame) -> pd.DataFrame:
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    market = "QQQ" if "QQQ" in asset_returns else "SPY" if "SPY" in asset_returns else ""
    if not market:
        return pd.DataFrame(index=prices.index)
    features = pd.DataFrame(index=prices.index)
    features["market_21d_return"] = _rolling_compound(asset_returns[market], 21)
    features["market_63d_return"] = _rolling_compound(asset_returns[market], 63)
    features["market_126d_return"] = _rolling_compound(asset_returns[market], 126)
    features["market_21d_vol"] = asset_returns[market].rolling(21, min_periods=10).std() * np.sqrt(252.0)
    market_price = prices[market].ffill()
    features["market_126d_drawdown"] = market_price / market_price.rolling(126, min_periods=21).max() - 1.0
    for ticker in ("SPY", "RSP", "VEA", "IWM", "SMH", "SOXX", "GLD", "TLT", "XLE"):
        if ticker in asset_returns:
            features[f"{ticker.lower()}_63d_return"] = _rolling_compound(asset_returns[ticker], 63)
    if "QQQ" in asset_returns:
        qqq = _rolling_compound(asset_returns["QQQ"], 63)
        for peer in ("SPY", "RSP", "VEA", "IWM"):
            if peer in asset_returns:
                features[f"qqq_vs_{peer.lower()}_63d"] = qqq - _rolling_compound(
                    asset_returns[peer],
                    63,
                )
    if "SMH" in asset_returns and "SPY" in asset_returns:
        features["semis_vs_spy_63d"] = _rolling_compound(asset_returns["SMH"], 63) - _rolling_compound(
            asset_returns["SPY"],
            63,
        )
    if "GLD" in asset_returns and "TLT" in asset_returns:
        features["gold_vs_tlt_63d"] = _rolling_compound(asset_returns["GLD"], 63) - _rolling_compound(
            asset_returns["TLT"],
            63,
        )
    return features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _router_candidate_scores(
    results: dict[str, BacktestResult],
    benchmark_returns: pd.Series,
    labels: pd.Series,
    state_features: pd.DataFrame,
    *,
    origin_date: pd.Timestamp,
    scenario: str,
    horizon_days: int,
    embargo_days: int,
    max_state_neighbors: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    state_distances = _state_distances(state_features, origin_date)
    for strategy, result in results.items():
        returns = result.returns.sort_index()
        prior_returns = returns[returns.index < origin_date]
        if len(prior_returns) <= horizon_days * 3:
            continue
        samples = _prior_forward_samples(
            prior_returns,
            benchmark_returns,
            labels,
            origin_date=origin_date,
            scenario=scenario,
            horizon_days=horizon_days,
            embargo_days=embargo_days,
            same_scenario=True,
        )
        state_samples = _prior_forward_samples(
            prior_returns,
            benchmark_returns,
            labels,
            origin_date=origin_date,
            scenario=scenario,
            horizon_days=horizon_days,
            embargo_days=embargo_days,
            same_scenario=False,
            state_distances=state_distances,
            max_state_neighbors=max_state_neighbors,
        )
        fallback_samples = pd.DataFrame()
        score_source = "state_similarity"
        if len(state_samples) >= 5 and len(samples) >= 5:
            score = 0.70 * _score_router_samples(state_samples) + 0.30 * _score_router_samples(samples)
            score_source = "state_similarity_plus_scenario"
            scoring_samples = state_samples
        elif len(state_samples) >= 5:
            score = _score_router_samples(state_samples)
            scoring_samples = state_samples
        elif len(samples) >= 5:
            score = _score_router_samples(samples)
            score_source = "scenario_bucket"
            scoring_samples = samples
        else:
            fallback_samples = _prior_forward_samples(
                prior_returns,
                benchmark_returns,
                labels,
                origin_date=origin_date,
                scenario=scenario,
                horizon_days=horizon_days,
                embargo_days=embargo_days,
                same_scenario=False,
            )
            if fallback_samples.empty:
                continue
            score = _score_router_samples(fallback_samples)
            score_source = "global_prior_fallback"
            scoring_samples = fallback_samples
        if scoring_samples.empty:
            continue
        rows.append(
            {
                "strategy": strategy,
                "score": float(score),
                "score_source": score_source,
                "median_forward_return": float(scoring_samples["forward_return"].median()),
                "median_excess_return": float(scoring_samples["excess_return"].median()),
                "q25_drawdown": float(scoring_samples["max_drawdown"].quantile(0.25)),
                "similar_windows": int(len(state_samples)),
                "scenario_windows": int(len(samples)),
                "fallback_windows": int(len(fallback_samples)),
                "mean_state_distance": float(
                    scoring_samples["state_distance"].mean()
                    if "state_distance" in scoring_samples
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _score_router_samples(samples: pd.DataFrame) -> float:
    if samples.empty:
        return float("nan")
    score = float(samples["excess_return"].median() + 0.35 * samples["forward_return"].median())
    return score - 0.35 * abs(float(samples["max_drawdown"].quantile(0.25)))


def _state_distances(features: pd.DataFrame, origin_date: pd.Timestamp) -> pd.Series:
    if features.empty or origin_date not in features.index:
        return pd.Series(dtype=float)
    prior = features[features.index < origin_date].copy()
    if prior.empty:
        return pd.Series(dtype=float)
    origin = features.loc[origin_date]
    mean = prior.mean()
    std = prior.std().replace(0.0, np.nan).fillna(1.0)
    z_prior = (prior - mean) / std
    z_origin = (origin - mean) / std
    distance = ((z_prior - z_origin) ** 2).mean(axis=1) ** 0.5
    return distance.replace([np.inf, -np.inf], np.nan).dropna()


def _prior_forward_samples(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    labels: pd.Series,
    *,
    origin_date: pd.Timestamp,
    scenario: str,
    horizon_days: int,
    embargo_days: int,
    same_scenario: bool,
    state_distances: pd.Series | None = None,
    max_state_neighbors: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    dates = returns.index
    origin_pos = int(dates.searchsorted(origin_date))
    for pos in range(0, len(dates) - horizon_days, max(5, horizon_days // 3)):
        start = dates[pos]
        if pos + horizon_days - 1 >= origin_pos - embargo_days:
            continue
        if same_scenario and str(labels.reindex([start]).iloc[0]) != scenario:
            continue
        window = returns.iloc[pos : pos + horizon_days]
        benchmark = benchmark_returns.reindex(window.index).fillna(0.0)
        equity = (1.0 + window).cumprod()
        row = {
            "start_date": str(start.date()),
            "forward_return": float(equity.iloc[-1] - 1.0),
            "benchmark_return": float((1.0 + benchmark).prod() - 1.0),
            "excess_return": float(equity.iloc[-1] - 1.0 - ((1.0 + benchmark).prod() - 1.0)),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        }
        if state_distances is not None and start in state_distances.index:
            row["state_distance"] = float(state_distances.loc[start])
        elif state_distances is not None:
            continue
        rows.append(row)
    frame = pd.DataFrame(rows)
    if state_distances is not None and not frame.empty and "state_distance" in frame:
        frame = frame.sort_values("state_distance")
        if max_state_neighbors is not None and max_state_neighbors > 0:
            frame = frame.head(max_state_neighbors)
    return frame


def _router_shrinkage_weights(
    ranked: pd.DataFrame,
    *,
    all_names: tuple[str, ...],
    top_k: int,
    shrinkage: float,
) -> dict[str, float]:
    names = [str(name) for name in all_names]
    if not names:
        return {}
    equal = 1.0 / len(names)
    weights = dict.fromkeys(names, shrinkage * equal)
    top = ranked.head(top_k).copy()
    if top.empty:
        return weights
    scores = pd.to_numeric(top["score"], errors="coerce").fillna(top["score"].median())
    raw = _softmax(scores)
    for strategy, weight in zip(top["strategy"].astype(str), raw, strict=False):
        weights[strategy] = weights.get(strategy, 0.0) + (1.0 - shrinkage) * float(weight)
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()} if total > 0 else weights


def _prior_top_k_weights(scores: pd.DataFrame, *, top_k: int) -> dict[str, float]:
    if scores.empty:
        return {}
    top = scores.sort_values("score", ascending=False).head(top_k)
    weight = 1.0 / len(top)
    return {str(strategy): weight for strategy in top["strategy"]}


def _prior_best_from_scores(scores: pd.DataFrame) -> str:
    if scores.empty:
        return ""
    return str(scores.sort_values("score", ascending=False).iloc[0]["strategy"])


def _prior_static_scores(
    results: dict[str, BacktestResult],
    benchmark_returns: pd.Series,
    *,
    origin_date: pd.Timestamp,
    horizon_days: int,
    embargo_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = pd.Series("all_prior", index=benchmark_returns.index)
    for strategy, result in results.items():
        samples = _prior_forward_samples(
            result.returns.sort_index(),
            benchmark_returns,
            labels,
            origin_date=origin_date,
            scenario="all_prior",
            horizon_days=horizon_days,
            embargo_days=embargo_days,
            same_scenario=False,
        )
        if samples.empty:
            continue
        rows.append(
            {
                "strategy": strategy,
                "score": _score_router_samples(samples),
                "windows": len(samples),
            }
        )
    return pd.DataFrame(rows)


def _weighted_forward_return(
    results: dict[str, BacktestResult],
    weights: dict[str, float],
    *,
    origin_date: pd.Timestamp,
    horizon_days: int,
) -> float:
    returns: list[float] = []
    weight_values: list[float] = []
    for strategy, weight in weights.items():
        if strategy not in results or weight <= 0:
            continue
        value = _forward_return(results[strategy].returns, origin_date, horizon_days)
        if np.isfinite(value):
            returns.append(value)
            weight_values.append(weight)
    if not returns:
        return float("nan")
    total = float(sum(weight_values))
    if total <= 0:
        return float("nan")
    return float(np.dot(np.asarray(returns), np.asarray(weight_values) / total))


def _softmax(values: pd.Series, *, temperature: float = 0.03) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return np.asarray([])
    arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0)
    scaled = (arr - np.nanmax(arr)) / max(temperature, 1e-6)
    exp = np.exp(np.clip(scaled, -60.0, 0.0))
    total = exp.sum()
    return exp / total if total > 0 else np.full(arr.shape, 1.0 / len(arr))


def _forward_return(returns: pd.Series, origin_date: pd.Timestamp, horizon_days: int) -> float:
    future = returns[returns.index > origin_date].head(horizon_days)
    if future.empty:
        return float("nan")
    return float((1.0 + future.fillna(0.0)).prod() - 1.0)


def _first_available_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _markdown_readout(
    *,
    selected_names: tuple[str, ...],
    tech_dependence: pd.DataFrame,
    factor_betas: pd.DataFrame,
    impairment: pd.DataFrame,
    scenario_heatmap: pd.DataFrame,
    router_summary: pd.DataFrame,
    router_selection: pd.DataFrame,
    router_scenarios: pd.DataFrame,
    router_comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Leadership Diagnostics",
        "",
        f"Strategies evaluated: {len(selected_names)}",
        "",
    ]
    if not tech_dependence.empty:
        top = tech_dependence.sort_values("current_tech_ai_weight", ascending=False).iloc[0]
        lines.append(
            "Highest current tech/AI exposure: "
            f"`{top['strategy']}` at {float(top['current_tech_ai_weight']):.1%}."
        )
    if not factor_betas.empty:
        qqq = factor_betas[factor_betas["factor"].eq("QQQ")].copy()
        if not qqq.empty:
            top_beta = qqq.sort_values("beta", ascending=False).iloc[0]
            lines.append(
                "Highest QQQ beta: "
                f"`{top_beta['strategy']}` beta {float(top_beta['beta']):.2f}."
            )
    if not impairment.empty:
        stressed = impairment[~impairment["scenario"].eq("native")]
        if not stressed.empty:
            worst = stressed.sort_values("delta_cagr_vs_native").iloc[0]
            lines.append(
                "Largest leadership impairment hit: "
                f"`{worst['strategy']}` under `{worst['scenario']}` "
                f"({float(worst['delta_cagr_vs_native']):.1%} CAGR delta)."
            )
    if not scenario_heatmap.empty:
        tech_rows = scenario_heatmap[
            scenario_heatmap["scenario_bucket"].astype(str).eq("tech_leadership_impaired")
        ]
        if not tech_rows.empty:
            best = tech_rows.sort_values("benchmark_excess", ascending=False).iloc[0]
            lines.append(
                "Best strategy during tech-leadership impairment states: "
                f"`{best['strategy']}` with {float(best['benchmark_excess']):.1%} "
                "annualized benchmark excess."
            )
    if not router_summary.empty:
        best_horizon = router_summary.sort_values(
            "top3_blend_mean_excess_vs_benchmark",
            ascending=False,
        ).iloc[0]
        lines.append(
            "Walk-forward router best horizon: "
            f"{int(best_horizon['horizon_days'])} trading days, top-3 blend excess "
            f"{float(best_horizon['top3_blend_mean_excess_vs_benchmark']):.1%}."
        )
        selected_best = router_summary.sort_values(
            "selected_mean_excess_vs_benchmark",
            ascending=False,
        ).iloc[0]
        selected_worst = router_summary.sort_values("selected_mean_excess_vs_benchmark").iloc[0]
        lines.extend(
            [
                "",
                "Router interpretation:",
                "- The router asks whether the current scenario state should change which strategy from today's shelf gets preference.",
                "- It is prior-only at each origin, so the selection score uses only older folds; the candidate shelf is still today's shelf.",
                "- It is most useful when 3-month and 6-month selections beat the benchmark and the equal-candidate shelf; 1-month reads are more noise-sensitive.",
                "- Current report result: "
                f"best selected horizon was {int(selected_best['horizon_days'])}d "
                f"at {float(selected_best['selected_mean_excess_vs_benchmark']):.1%} mean excess; "
                f"weakest was {int(selected_worst['horizon_days'])}d "
                f"at {float(selected_worst['selected_mean_excess_vs_benchmark']):.1%}.",
            ]
        )
    if not router_selection.empty:
        frequent = router_selection.sort_values(["selected_count", "mean_excess_vs_benchmark"], ascending=False).iloc[0]
        lines.append(
            "- Most frequent selected strategy: "
            f"`{frequent['strategy']}` at {int(frequent['horizon_days'])}d "
            f"({float(frequent['selection_rate']):.1%} of folds, "
            f"{float(frequent['mean_excess_vs_benchmark']):.1%} mean excess)."
        )
    if not router_scenarios.empty:
        scenario_best = router_scenarios.sort_values(
            "selected_mean_excess_vs_benchmark",
            ascending=False,
        ).iloc[0]
        lines.append(
            "- Best scenario bucket for selected routing: "
            f"`{scenario_best['scenario_bucket']}` at {int(scenario_best['horizon_days'])}d "
            f"({float(scenario_best['selected_mean_excess_vs_benchmark']):.1%} mean excess, "
            f"{int(scenario_best['folds'])} folds)."
        )
    if not router_comparison.empty:
        comparison = router_comparison[
            router_comparison["model"].astype(str).ne("benchmark")
        ].copy()
        if not comparison.empty:
            best_model = comparison.sort_values(
                "mean_excess_vs_benchmark",
                ascending=False,
            ).iloc[0]
            lines.append(
                "- Best evaluated router/baseline model: "
                f"`{best_model['model']}` at {int(best_model['horizon_days'])}d "
                f"({float(best_model['mean_excess_vs_benchmark']):.1%} mean excess, "
                f"{float(best_model['hit_rate_vs_benchmark']):.1%} hit rate)."
            )
    lines.extend(
        [
            "",
            "Artifacts:",
            "- `strategy_tech_dependence.csv`",
            "- `strategy_factor_betas.csv`",
            "- `strategy_return_contribution.csv`",
            "- `qqq_underperformance_periods.csv`",
            "- `leadership_impairment.csv`",
            "- `scenario_strategy_heatmap.csv`",
            "- `walk_forward_router_folds.csv`",
            "- `walk_forward_router_summary.csv`",
            "- `walk_forward_router_selection.csv`",
            "- `walk_forward_router_scenarios.csv`",
            "- `walk_forward_router_comparison.csv`",
            "- `walk_forward_router_scores.csv`",
            "",
            "Interpretation rule: this is a purged, prior-only validation and routing audit. It is not a live allocation override.",
        ]
    )
    return "\n".join(lines) + "\n"
