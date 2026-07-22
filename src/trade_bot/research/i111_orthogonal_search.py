from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.backtest.windows import (
    calendar_year_metrics,
    rolling_window_metrics,
    summarize_walk_forward,
    summarize_windows,
    walk_forward_holdout_metrics,
)
from trade_bot.config import (
    BotConfig,
    DrawdownControlConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    configured_tickers,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_ORTHOGONAL_SEARCH_OUTPUT_DIR = Path("reports/i111_orthogonal_search")
DEFAULT_I111_NATIVE_CHALLENGER = "i111_native_risk_repair_guard17_relief85_ai85_div"
DEFAULT_MAX_NEW_COMBINATIONS = 50

CORE_AI_TICKERS = ("QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN", "PLTR")
SIGNAL_TICKERS = ("SPY", "QQQ", "SMH", "RSP", "HYG", "LQD", "BIL", "GLD", "TLT")


@dataclass(frozen=True)
class SourceRecipe:
    name: str
    description: str
    updates: dict[str, Any]


@dataclass(frozen=True)
class MechanismRecipe:
    name: str
    family: str
    hypothesis: str
    updates: dict[str, Any]


@dataclass(frozen=True)
class OrthogonalCandidate:
    name: str
    source_recipe: str
    mechanism: str
    mechanism_family: str
    hypothesis: str
    strategy: StrategyConfig


@dataclass(frozen=True)
class OrthogonalSearchResult:
    strategy_metrics: pd.DataFrame
    candidate_roster: pd.DataFrame
    variant_summary: pd.DataFrame
    rolling_windows: pd.DataFrame
    walk_forward: pd.DataFrame
    calendar_years: pd.DataFrame
    summary: str


def run_i111_orthogonal_search(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_ORTHOGONAL_SEARCH_OUTPUT_DIR,
    max_new_combinations: int = DEFAULT_MAX_NEW_COMBINATIONS,
    refresh_data: bool = False,
) -> OrthogonalSearchResult:
    baselines, candidates = build_i111_orthogonal_candidates(
        config,
        max_new_combinations=max_new_combinations,
    )
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(tuple(candidate.strategy for candidate in candidates))
        | _candidate_tickers(tuple(baselines.values()))
        | set(AI_GROWTH_TICKERS)
        | set(SIGNAL_TICKERS)
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    baseline_results = {
        name: _run_strategy(config, name, strategy, prices) for name, strategy in baselines.items()
    }
    baseline_metrics = {name: _metrics(result) for name, result in baseline_results.items()}
    primary_metrics = baseline_metrics.get("baseline_primary")
    native_metrics = baseline_metrics.get("baseline_native_challenger")
    metric_rows: list[dict[str, object]] = []
    candidate_results: dict[str, BacktestResult] = dict(baseline_results)
    for name, result in baseline_results.items():
        metrics = baseline_metrics[name]
        metric_rows.append(
            _metric_row(
                name=name,
                role="baseline",
                source_recipe="baseline",
                mechanism="baseline",
                mechanism_family="baseline",
                hypothesis="Reference strategy.",
                metrics=metrics,
                result=result,
                primary_metrics=primary_metrics,
                native_metrics=native_metrics,
            )
        )

    for candidate in candidates:
        result = _run_strategy(config, candidate.name, candidate.strategy, prices)
        candidate_results[candidate.name] = result
        metrics = _metrics(result)
        metric_rows.append(
            _metric_row(
                name=candidate.name,
                role="new_combination",
                source_recipe=candidate.source_recipe,
                mechanism=candidate.mechanism,
                mechanism_family=candidate.mechanism_family,
                hypothesis=candidate.hypothesis,
                metrics=metrics,
                result=result,
                primary_metrics=primary_metrics,
                native_metrics=native_metrics,
            )
        )

    strategy_metrics = pd.DataFrame(metric_rows)
    rolling = summarize_windows(rolling_window_metrics(candidate_results)).reset_index()
    walk_forward = summarize_walk_forward(
        walk_forward_holdout_metrics(candidate_results)
    ).reset_index()
    calendar = calendar_year_metrics(candidate_results)
    strategy_metrics = _add_robustness(strategy_metrics, rolling, walk_forward, calendar)
    strategy_metrics = _score(strategy_metrics)
    candidate_roster = _candidate_roster(candidates)
    variant_summary = _variant_summary(strategy_metrics)
    summary = _summary_markdown(
        strategy_metrics,
        variant_summary,
        new_combination_count=len(candidates),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    candidate_roster.to_csv(output_path / "candidate_roster.csv", index=False)
    variant_summary.to_csv(output_path / "variant_summary.csv", index=False)
    rolling.to_csv(output_path / "rolling_windows.csv", index=False)
    walk_forward.to_csv(output_path / "walk_forward.csv", index=False)
    calendar.to_csv(output_path / "calendar_years.csv", index=False)
    (output_path / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output_path,
        study="i111_orthogonal_search",
        config=config,
        prices=prices,
        parameters={
            "max_new_combinations": max_new_combinations,
            "refresh_data": refresh_data,
            "candidate_count": len(candidates),
        },
        artifacts=[
            "strategy_metrics.csv",
            "candidate_roster.csv",
            "variant_summary.csv",
            "rolling_windows.csv",
            "walk_forward.csv",
            "calendar_years.csv",
            "summary.md",
        ],
    )
    return OrthogonalSearchResult(
        strategy_metrics=strategy_metrics,
        candidate_roster=candidate_roster,
        variant_summary=variant_summary,
        rolling_windows=rolling,
        walk_forward=walk_forward,
        calendar_years=calendar,
        summary=summary,
    )


def build_i111_orthogonal_candidates(
    config: BotConfig,
    *,
    max_new_combinations: int = DEFAULT_MAX_NEW_COMBINATIONS,
) -> tuple[dict[str, StrategyConfig], tuple[OrthogonalCandidate, ...]]:
    primary = config.strategies[config.primary_strategy]
    baselines = {"baseline_primary": primary}
    if DEFAULT_I111_NATIVE_CHALLENGER in config.strategies:
        baselines["baseline_native_challenger"] = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]

    candidates: list[OrthogonalCandidate] = []
    for source in _source_recipes():
        source_strategy = _strategy_from_recipe(primary, source.updates)
        for mechanism in _mechanism_recipes():
            strategy = _strategy_from_recipe(source_strategy, mechanism.updates)
            name = f"{source.name}__{mechanism.name}"
            candidates.append(
                OrthogonalCandidate(
                    name=name,
                    source_recipe=source.name,
                    mechanism=mechanism.name,
                    mechanism_family=mechanism.family,
                    hypothesis=mechanism.hypothesis,
                    strategy=strategy,
                )
            )
            if len(candidates) >= max_new_combinations:
                return baselines, tuple(candidates)
    return baselines, tuple(candidates)


def _source_recipes() -> tuple[SourceRecipe, ...]:
    return (
        SourceRecipe(
            name="src_guard16_vol185_mult60",
            description="Moderately later guard than current primary.",
            updates=_source_updates(max_drawdown=-0.16, risk_multiplier=0.60, vol=0.185),
        ),
        SourceRecipe(
            name="src_guard17_vol185_mult65",
            description="Late guard family that produced the current native challenger.",
            updates=_source_updates(max_drawdown=-0.17, risk_multiplier=0.65, vol=0.185),
        ),
        SourceRecipe(
            name="src_guard18_vol185_mult70",
            description="Latest tested guard with higher post-trigger risk budget.",
            updates=_source_updates(max_drawdown=-0.18, risk_multiplier=0.70, vol=0.185),
        ),
        SourceRecipe(
            name="src_guard15_vol19_mult60",
            description="Higher volatility target with a tighter guard.",
            updates=_source_updates(max_drawdown=-0.15, risk_multiplier=0.60, vol=0.190),
        ),
        SourceRecipe(
            name="src_guard17_vol20_mult65",
            description="Late guard with more upside volatility budget.",
            updates=_source_updates(max_drawdown=-0.17, risk_multiplier=0.65, vol=0.200),
        ),
    )


def _mechanism_recipes() -> tuple[MechanismRecipe, ...]:
    repair = _native_repair_updates()
    return (
        MechanismRecipe(
            name="fast42_repair",
            family="signal_speed",
            hypothesis="A faster 42-day signal may catch re-risk windows earlier without changing the risk engine.",
            updates={**repair, "lookback_days": 42},
        ),
        MechanismRecipe(
            name="slow84_repair",
            family="signal_speed",
            hypothesis="A slower 84-day signal may avoid early false defensive rotations.",
            updates={**repair, "lookback_days": 84},
        ),
        MechanismRecipe(
            name="skip0_fast42_repair",
            family="signal_speed",
            hypothesis="Removing the momentum skip may improve participation when leadership repair is abrupt.",
            updates={**repair, "lookback_days": 42, "skip_days": 0},
        ),
        MechanismRecipe(
            name="trend_quality_rank",
            family="ranking_quality",
            hypothesis="Trend-quality ranking may favor durable leadership over short-lived momentum spikes.",
            updates={
                **repair,
                "ranking_metric": "return_trend_quality",
                "weighting": "momentum_score",
            },
        ),
        MechanismRecipe(
            name="top3_max40",
            family="concentration_shape",
            hypothesis="Fewer winners with a higher single-name cap may preserve the AI engine with cleaner concentration.",
            updates={**repair, "top_n": 3, "max_asset_weight": 0.40},
        ),
        MechanismRecipe(
            name="top5_max30",
            family="concentration_shape",
            hypothesis="More winners with a lower single-name cap may reduce idiosyncratic AI drawdown without blunt sector caps.",
            updates={**repair, "top_n": 5, "max_asset_weight": 0.30},
        ),
        MechanismRecipe(
            name="broad_us_universe",
            family="universe_breadth",
            hypothesis="Adding broad US equity alternatives may keep risk-on exposure when AI leadership narrows or repairs.",
            updates={**repair, "tickers": [*CORE_AI_TICKERS, "SPY", "RSP", "IWM", "VTI"]},
        ),
        MechanismRecipe(
            name="factor_quality_universe",
            family="universe_breadth",
            hypothesis="Quality, momentum, and low-volatility factor ETFs may provide non-AI risk-on substitutes.",
            updates={**repair, "tickers": [*CORE_AI_TICKERS, "QUAL", "MTUM", "USMV", "COWZ"]},
        ),
        MechanismRecipe(
            name="global_real_universe",
            family="diversifier_universe",
            hypothesis="Global equity, gold, and Treasuries may create endogenous non-AI escape routes.",
            updates={
                **repair,
                "tickers": [*CORE_AI_TICKERS, "SPY", "RSP", "VEA", "VWO", "GLD", "TLT"],
            },
        ),
        MechanismRecipe(
            name="ai_leadership_relief90",
            family="confirmation_gate",
            hypothesis="Requiring AI leadership confirmation for BIL relief may reduce false early re-risking.",
            updates={
                **repair,
                "risk_repair_signal": "ai_leadership",
                "risk_repair_defensive_cap": 0.90,
                "risk_repair_defensive_release": 0.25,
            },
        ),
    )


def _source_updates(*, max_drawdown: float, risk_multiplier: float, vol: float) -> dict[str, Any]:
    return {
        "type": "dual_momentum_risk_repair",
        "tickers": list(CORE_AI_TICKERS),
        "lookback_days": 63,
        "skip_days": 5,
        "top_n": 4,
        "min_return": 0.025,
        "ranking_metric": "risk_adjusted_return",
        "weighting": "risk_adjusted_score",
        "volatility_lookback_days": 63,
        "trend_filter_days": None,
        "max_asset_weight": 0.35,
        "volatility_target": VolatilityTargetConfig(
            annualized_volatility=vol,
            lookback_days=21,
            max_leverage=1.0,
        ),
        "drawdown_control": DrawdownControlConfig(
            equity_lookback_days=84,
            max_drawdown=max_drawdown,
            risk_multiplier=risk_multiplier,
        ),
    }


def _native_repair_updates() -> dict[str, Any]:
    return {
        "type": "dual_momentum_risk_repair",
        "risk_repair_signal": "balanced",
        "risk_repair_defensive_cap": 0.85,
        "risk_repair_defensive_release": 0.15,
        "risk_repair_ai_soft_cap": 0.85,
        "risk_repair_ai_soft_threshold": 0.90,
        "risk_repair_ai_excess_destination": "diversifier_mix",
        "risk_repair_ai_diversifier_tickers": ["SPY", "RSP", "GLD", "TLT"],
        "risk_repair_lookback_days": 42,
    }


def _strategy_from_recipe(strategy: StrategyConfig, updates: dict[str, Any]) -> StrategyConfig:
    return strategy.model_copy(update=updates)


def _run_strategy(
    config: BotConfig,
    name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
) -> BacktestResult:
    candidate_prices = _strategy_prices(prices, strategy)
    weights = build_strategy_weights(candidate_prices, strategy)
    return run_backtest(
        name,
        candidate_prices,
        weights,
        config.execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )


def _strategy_prices(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = list(dict.fromkeys([*required_strategy_tickers(strategy), *SIGNAL_TICKERS]))
    missing = unusable_required_price_columns(prices, columns)
    if missing:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing}")
    frame = prices[columns].sort_index()
    return frame.loc[frame.notna().any(axis=1)]


def _candidate_tickers(strategies: tuple[StrategyConfig, ...]) -> set[str]:
    tickers: set[str] = set(SIGNAL_TICKERS)
    for strategy in strategies:
        tickers.update(required_strategy_tickers(strategy))
    return tickers


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _metric_row(
    *,
    name: str,
    role: str,
    source_recipe: str,
    mechanism: str,
    mechanism_family: str,
    hypothesis: str,
    metrics: PerformanceMetrics,
    result: BacktestResult,
    primary_metrics: PerformanceMetrics | None,
    native_metrics: PerformanceMetrics | None,
) -> dict[str, object]:
    average_ai_weight = _average_ai_weight(result.weights)
    defensive = (
        result.weights["BIL"]
        if "BIL" in result.weights.columns
        else pd.Series(0.0, index=result.weights.index)
    )
    row = {
        "result_name": name,
        "role": role,
        "source_recipe": source_recipe,
        "mechanism": mechanism,
        "mechanism_family": mechanism_family,
        "hypothesis": hypothesis,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "average_ai_growth_weight": average_ai_weight,
        "average_defensive_weight": float(defensive.mean()),
        "hard_defensive_day_rate": float((defensive >= 0.50).mean()),
    }
    if primary_metrics is not None:
        row["delta_vs_primary_cagr"] = metrics.cagr - primary_metrics.cagr
        row["delta_vs_primary_max_drawdown"] = metrics.max_drawdown - primary_metrics.max_drawdown
        row["delta_vs_primary_calmar"] = metrics.calmar - primary_metrics.calmar
    if native_metrics is not None:
        row["delta_vs_native_cagr"] = metrics.cagr - native_metrics.cagr
        row["delta_vs_native_max_drawdown"] = metrics.max_drawdown - native_metrics.max_drawdown
        row["delta_vs_native_calmar"] = metrics.calmar - native_metrics.calmar
    return row


def _average_ai_weight(weights: pd.DataFrame) -> float:
    columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    if not columns:
        return 0.0
    return float(weights[columns].sum(axis=1).mean())


def _add_robustness(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    walk_forward: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    output = metrics.copy()
    if not rolling.empty:
        three_year = rolling[rolling["window"].eq("3y")][
            ["name", "worst_cagr", "positive_window_rate", "worst_drawdown"]
        ].rename(
            columns={
                "name": "result_name",
                "worst_cagr": "worst_3y_cagr",
                "positive_window_rate": "positive_3y_window_rate",
                "worst_drawdown": "worst_3y_drawdown",
            }
        )
        output = output.merge(three_year, on="result_name", how="left")
    if not walk_forward.empty:
        output = output.merge(
            walk_forward[
                [
                    "name",
                    "walk_forward_median_cagr",
                    "walk_forward_worst_cagr",
                    "walk_forward_positive_rate",
                    "walk_forward_worst_drawdown",
                ]
            ].rename(columns={"name": "result_name"}),
            on="result_name",
            how="left",
        )
    if not calendar.empty:
        year_summary = (
            calendar.groupby("name", observed=True)
            .agg(
                calendar_years=("window", "count"),
                negative_calendar_years=("total_return", lambda values: int((values < 0).sum())),
            )
            .reset_index()
            .rename(columns={"name": "result_name"})
        )
        output = output.merge(year_summary, on="result_name", how="left")
    return output


def _score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["drawdown_penalty"] = (output["max_drawdown"].abs() - 0.205).clip(lower=0.0) * 2.5
    output["native_cagr_shortfall_penalty"] = (
        output.get("delta_vs_native_cagr", 0.0).clip(upper=0.0).abs() * 1.5
    )
    output["stale_defense_penalty"] = output["hard_defensive_day_rate"].clip(lower=0.08) * 0.12
    output["low_ai_penalty"] = (0.60 - output["average_ai_growth_weight"]).clip(lower=0.0) * 0.20
    output["orthogonal_score"] = (
        output["cagr"]
        + 0.40 * output["calmar"]
        + 0.15 * output.get("delta_vs_native_max_drawdown", 0.0)
        - output["drawdown_penalty"]
        - output["native_cagr_shortfall_penalty"]
        - output["stale_defense_penalty"]
        - output["low_ai_penalty"]
    )
    output["considerable_improvement"] = (
        output["role"].eq("new_combination")
        & (output["cagr"] >= 0.225)
        & (output["max_drawdown"] >= -0.205)
        & (output.get("delta_vs_native_cagr", 0.0) >= 0.003)
    ) | (
        output["role"].eq("new_combination")
        & (output["cagr"] >= 0.220)
        & (output["max_drawdown"] >= -0.190)
        & (output.get("delta_vs_native_max_drawdown", 0.0) >= 0.006)
    )
    output["promotion_candidate"] = (
        output["role"].eq("new_combination")
        & (output["cagr"] >= 0.220)
        & (output["max_drawdown"] >= -0.205)
        & (output["average_ai_growth_weight"] >= 0.60)
    )
    return output.sort_values("orthogonal_score", ascending=False)


def _candidate_roster(candidates: tuple[OrthogonalCandidate, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_name": candidate.name,
                "source_recipe": candidate.source_recipe,
                "mechanism": candidate.mechanism,
                "mechanism_family": candidate.mechanism_family,
                "hypothesis": candidate.hypothesis,
                "tickers": ",".join(candidate.strategy.tickers),
            }
            for candidate in candidates
        ]
    )


def _variant_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    new_rows = metrics[metrics["role"].eq("new_combination")]
    if new_rows.empty:
        return pd.DataFrame()
    return (
        new_rows.groupby("mechanism_family", observed=True)
        .agg(
            candidates=("result_name", "count"),
            best_cagr=("cagr", "max"),
            median_cagr=("cagr", "median"),
            best_max_drawdown=("max_drawdown", "max"),
            median_max_drawdown=("max_drawdown", "median"),
            best_calmar=("calmar", "max"),
            promotion_rate=("promotion_candidate", "mean"),
            considerable_count=("considerable_improvement", "sum"),
        )
        .reset_index()
        .sort_values(["considerable_count", "best_calmar"], ascending=False)
    )


def _summary_markdown(
    metrics: pd.DataFrame,
    variant_summary: pd.DataFrame,
    *,
    new_combination_count: int,
) -> str:
    baselines = metrics[metrics["role"].eq("baseline")]
    new_rows = metrics[metrics["role"].eq("new_combination")]
    top = new_rows.head(12)
    considerable = new_rows[new_rows["considerable_improvement"]]
    lines = [
        "# I111 Orthogonal Search",
        "",
        "## Goal",
        "",
        (
            f"Search for orthogonal improvement mechanisms and stop after "
            f"{new_combination_count} new combinations unless a considerable improvement is found."
        ),
        "",
        "## Baselines",
        "",
        "| baseline | CAGR | max DD | Calmar | AI/growth wt |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in baselines.iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['calmar']:.2f} | "
            f"{row['average_ai_growth_weight']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Best New Combinations",
            "",
            "| result | family | CAGR | max DD | native CAGR delta | native DD delta | AI/growth wt | considerable? |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['mechanism_family']} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row.get('delta_vs_native_cagr', 0.0):.2%} | "
            f"{row.get('delta_vs_native_max_drawdown', 0.0):.2%} | "
            f"{row['average_ai_growth_weight']:.2%} | {bool(row['considerable_improvement'])} |"
        )
    lines.extend(["", "## Family Summary", ""])
    if variant_summary.empty:
        lines.append("No new combinations were tested.")
    else:
        lines.append(
            "| family | candidates | best CAGR | median CAGR | best max DD | promotion rate | considerable |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in variant_summary.iterrows():
            lines.append(
                f"| {row['mechanism_family']} | {int(row['candidates'])} | "
                f"{row['best_cagr']:.2%} | {row['median_cagr']:.2%} | "
                f"{row['best_max_drawdown']:.2%} | {row['promotion_rate']:.0%} | "
                f"{int(row['considerable_count'])} |"
            )
    lines.extend(["", "## Readout", ""])
    if considerable.empty:
        lines.append(
            "No candidate cleared the considerable-improvement gate. The best results are useful "
            "survey evidence, but they do not displace the current native challenger yet."
        )
    else:
        winner = considerable.sort_values("orthogonal_score", ascending=False).iloc[0]
        lines.append(
            f"Considerable improvement found: `{winner['result_name']}` with CAGR "
            f"{winner['cagr']:.2%}, max DD {winner['max_drawdown']:.2%}, and Calmar "
            f"{winner['calmar']:.2f}."
        )
    return "\n".join(lines) + "\n"
