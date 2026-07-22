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
    StrategyConfig,
    configured_tickers,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.i111_candidates import (
    DEFAULT_I111_PREFIX,
    DEFAULT_UPSIDE_TOP_NAMES,
    I111Candidate,
    build_i111_candidates,
)
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_NATIVE_I111_RISK_REPAIR_OUTPUT_DIR = Path("reports/native_i111_risk_repair")


@dataclass(frozen=True)
class NativeRiskRepairSpec:
    name: str
    description: str
    updates: dict[str, Any]


@dataclass(frozen=True)
class NativeRiskRepairResult:
    strategy_metrics: pd.DataFrame
    variant_summary: pd.DataFrame
    rolling_windows: pd.DataFrame
    walk_forward: pd.DataFrame
    calendar_years: pd.DataFrame
    summary: str


def run_native_i111_risk_repair_lab(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_NATIVE_I111_RISK_REPAIR_OUTPUT_DIR,
    strategy_prefix: str = DEFAULT_I111_PREFIX,
    include_upside_research: bool = True,
    upside_top_names: tuple[str, ...] = DEFAULT_UPSIDE_TOP_NAMES,
    specs: tuple[NativeRiskRepairSpec, ...] | None = None,
    refresh_data: bool = False,
) -> NativeRiskRepairResult:
    source_candidates = build_i111_candidates(
        config,
        strategy_prefix=strategy_prefix,
        include_upside_research=include_upside_research,
        upside_top_names=upside_top_names,
    )
    repair_specs = specs or default_native_risk_repair_specs()
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(source_candidates)
        | set(AI_GROWTH_TICKERS)
        | {"SPY", "QQQ", "SMH", "HYG", "LQD", "RSP", "BIL", "GLD", "TLT"}
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    results: dict[str, BacktestResult] = {}
    metric_rows: list[dict[str, object]] = []
    for source in source_candidates:
        source_key = f"{source.name}__base"
        source_result = _run_candidate(config, source_key, source.strategy, prices)
        source_metrics = _metrics(source_result)
        source_behavior = _behavior(source_result)
        source_ai = _average_ai_weight(source_result.weights)
        results[source_key] = source_result
        metric_rows.append(
            _metric_row(
                result_name=source_key,
                source=source,
                variant_name="base",
                description="Unmodified source strategy.",
                metrics=source_metrics,
                source_metrics=source_metrics,
                primary_metrics=None,
                behavior=source_behavior,
                source_behavior=source_behavior,
                average_ai_growth_weight=source_ai,
            )
        )
        for spec in repair_specs:
            strategy = _native_strategy(source.strategy, spec)
            result_name = f"{source.name}__{spec.name}"
            result = _run_candidate(config, result_name, strategy, prices)
            metrics = _metrics(result)
            behavior = _behavior(result)
            results[result_name] = result
            metric_rows.append(
                _metric_row(
                    result_name=result_name,
                    source=source,
                    variant_name=spec.name,
                    description=spec.description,
                    metrics=metrics,
                    source_metrics=source_metrics,
                    primary_metrics=None,
                    behavior=behavior,
                    source_behavior=source_behavior,
                    average_ai_growth_weight=_average_ai_weight(result.weights),
                )
            )

    strategy_metrics = pd.DataFrame(metric_rows)
    primary_name = _primary_base_name(config, strategy_metrics)
    if primary_name:
        primary = strategy_metrics[strategy_metrics["result_name"].eq(primary_name)].iloc[0]
        strategy_metrics["delta_vs_primary_cagr"] = strategy_metrics["cagr"] - float(
            primary["cagr"]
        )
        strategy_metrics["delta_vs_primary_max_drawdown"] = strategy_metrics[
            "max_drawdown"
        ] - float(primary["max_drawdown"])
    else:
        strategy_metrics["delta_vs_primary_cagr"] = 0.0
        strategy_metrics["delta_vs_primary_max_drawdown"] = 0.0

    rolling = summarize_windows(rolling_window_metrics(results)).reset_index()
    walk_forward = summarize_walk_forward(walk_forward_holdout_metrics(results)).reset_index()
    calendar = calendar_year_metrics(results)
    strategy_metrics = _add_robustness(strategy_metrics, rolling, walk_forward, calendar)
    strategy_metrics = _score(strategy_metrics)
    variant_summary = _variant_summary(strategy_metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    strategy_metrics.sort_values("research_score", ascending=False).to_csv(
        output_path / "strategy_metrics.csv",
        index=False,
    )
    variant_summary.to_csv(output_path / "variant_summary.csv", index=False)
    rolling.to_csv(output_path / "rolling_windows.csv", index=False)
    walk_forward.to_csv(output_path / "walk_forward.csv", index=False)
    calendar.to_csv(output_path / "calendar_years.csv", index=False)
    summary = _summary_markdown(strategy_metrics, variant_summary, primary_name)
    (output_path / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output_path,
        study="native_i111_risk_repair",
        config=config,
        prices=prices,
        parameters={
            "strategy_prefix": strategy_prefix,
            "include_upside_research": include_upside_research,
            "upside_top_names": list(upside_top_names),
            "refresh_data": refresh_data,
            "repair_specs": [spec.name for spec in repair_specs],
        },
        artifacts=[
            "strategy_metrics.csv",
            "variant_summary.csv",
            "rolling_windows.csv",
            "walk_forward.csv",
            "calendar_years.csv",
            "summary.md",
        ],
    )
    return NativeRiskRepairResult(
        strategy_metrics=strategy_metrics,
        variant_summary=variant_summary,
        rolling_windows=rolling,
        walk_forward=walk_forward,
        calendar_years=calendar,
        summary=summary,
    )


def default_native_risk_repair_specs() -> tuple[NativeRiskRepairSpec, ...]:
    return (
        NativeRiskRepairSpec(
            name="relief_cap80_rel25",
            description="Release 25% of BIL above 80% when broad repair evidence is constructive.",
            updates={
                "risk_repair_defensive_cap": 0.80,
                "risk_repair_defensive_release": 0.25,
            },
        ),
        NativeRiskRepairSpec(
            name="relief_cap85_rel15",
            description="Release 15% of BIL above 85% only when broad repair evidence is constructive.",
            updates={
                "risk_repair_defensive_cap": 0.85,
                "risk_repair_defensive_release": 0.15,
            },
        ),
        NativeRiskRepairSpec(
            name="relief_cap80_rel15_credit",
            description="Release 15% of BIL above 80% only with trend, breadth, and credit confirmation.",
            updates={
                "risk_repair_signal": "credit_breadth",
                "risk_repair_defensive_cap": 0.80,
                "risk_repair_defensive_release": 0.15,
            },
        ),
        NativeRiskRepairSpec(
            name="relief_cap75_rel25",
            description="Release 25% of BIL above 75% when broad repair evidence is constructive.",
            updates={
                "risk_repair_defensive_cap": 0.75,
                "risk_repair_defensive_release": 0.25,
            },
        ),
        NativeRiskRepairSpec(
            name="relief_cap75_rel50",
            description="Release 50% of BIL above 75% when broad repair evidence is constructive.",
            updates={
                "risk_repair_defensive_cap": 0.75,
                "risk_repair_defensive_release": 0.50,
            },
        ),
        NativeRiskRepairSpec(
            name="floor60_balanced",
            description="Hold at least 60% risk in constructive regimes unless late AI stress is high.",
            updates={"risk_repair_constructive_floor": 0.60},
        ),
        NativeRiskRepairSpec(
            name="ai_soft75_s085",
            description="Cap aggregate AI/growth at 75% only when AI stress score is at least 85%.",
            updates={
                "risk_repair_ai_soft_cap": 0.75,
                "risk_repair_ai_soft_threshold": 0.85,
            },
        ),
        NativeRiskRepairSpec(
            name="ai_soft85_s090_div",
            description="Cap AI/growth at 85% only in very high stress and rotate excess to diversifiers.",
            updates={
                "risk_repair_ai_soft_cap": 0.85,
                "risk_repair_ai_soft_threshold": 0.90,
                "risk_repair_ai_excess_destination": "diversifier_mix",
            },
        ),
        NativeRiskRepairSpec(
            name="ai_soft80_s090_div",
            description="Cap AI/growth at 80% only in very high stress and rotate excess to diversifiers.",
            updates={
                "risk_repair_ai_soft_cap": 0.80,
                "risk_repair_ai_soft_threshold": 0.90,
                "risk_repair_ai_excess_destination": "diversifier_mix",
            },
        ),
        NativeRiskRepairSpec(
            name="ai_soft70_s080",
            description="Cap aggregate AI/growth at 70% only when AI stress score is at least 80%.",
            updates={
                "risk_repair_ai_soft_cap": 0.70,
                "risk_repair_ai_soft_threshold": 0.80,
            },
        ),
        NativeRiskRepairSpec(
            name="ai_stage75_60",
            description="Cap AI/growth at 75% in high stress and 60% in extreme stress.",
            updates={
                "risk_repair_ai_soft_cap": 0.75,
                "risk_repair_ai_soft_threshold": 0.80,
                "risk_repair_ai_hard_cap": 0.60,
                "risk_repair_ai_hard_threshold": 0.92,
            },
        ),
        NativeRiskRepairSpec(
            name="ai_stage85_70_div",
            description="Rotate excess AI/growth to diversifiers at 85% in high stress and 70% in extremes.",
            updates={
                "risk_repair_ai_soft_cap": 0.85,
                "risk_repair_ai_soft_threshold": 0.86,
                "risk_repair_ai_hard_cap": 0.70,
                "risk_repair_ai_hard_threshold": 0.94,
                "risk_repair_ai_excess_destination": "diversifier_mix",
            },
        ),
        NativeRiskRepairSpec(
            name="balanced_relief85_ai85_div",
            description="Very light defensive relief plus very high-threshold diversifier AI cap.",
            updates={
                "risk_repair_defensive_cap": 0.85,
                "risk_repair_defensive_release": 0.15,
                "risk_repair_ai_soft_cap": 0.85,
                "risk_repair_ai_soft_threshold": 0.90,
                "risk_repair_ai_excess_destination": "diversifier_mix",
            },
        ),
        NativeRiskRepairSpec(
            name="balanced_relief75_ai75",
            description="Modest defensive relief plus high-threshold AI/growth cap.",
            updates={
                "risk_repair_defensive_cap": 0.75,
                "risk_repair_defensive_release": 0.25,
                "risk_repair_ai_soft_cap": 0.75,
                "risk_repair_ai_soft_threshold": 0.85,
            },
        ),
        NativeRiskRepairSpec(
            name="balanced_relief75_ai70",
            description="More assertive relief plus 70% high-stress AI/growth cap.",
            updates={
                "risk_repair_defensive_cap": 0.75,
                "risk_repair_defensive_release": 0.50,
                "risk_repair_ai_soft_cap": 0.70,
                "risk_repair_ai_soft_threshold": 0.82,
            },
        ),
        NativeRiskRepairSpec(
            name="floor60_ai75",
            description="Constructive 60% risk floor plus high-threshold AI/growth cap.",
            updates={
                "risk_repair_constructive_floor": 0.60,
                "risk_repair_ai_soft_cap": 0.75,
                "risk_repair_ai_soft_threshold": 0.85,
            },
        ),
    )


def _native_strategy(strategy: StrategyConfig, spec: NativeRiskRepairSpec) -> StrategyConfig:
    updates = {
        "type": "dual_momentum_risk_repair",
        "risk_repair_signal": "balanced",
        "risk_repair_lookback_days": 42,
        **spec.updates,
    }
    return strategy.model_copy(update=updates)


def _run_candidate(
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
    columns = required_strategy_tickers(strategy)
    missing_required = unusable_required_price_columns(prices, columns)
    if missing_required:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing_required}")
    frame = prices[columns].sort_index()
    return frame.loc[frame.notna().any(axis=1)]


def _candidate_tickers(candidates: tuple[I111Candidate, ...]) -> set[str]:
    tickers: set[str] = set()
    for candidate in candidates:
        tickers.update(required_strategy_tickers(candidate.strategy))
    return tickers


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _behavior(result: BacktestResult) -> dict[str, float]:
    weights = result.weights
    defensive = weights["BIL"] if "BIL" in weights.columns else pd.Series(0.0, index=weights.index)
    return {
        "average_defensive_weight": float(defensive.mean()),
        "hard_defensive_day_rate": float((defensive >= 0.50).mean()),
        "average_risk_weight": float((1.0 - defensive).mean()),
        "average_turnover": float(result.turnover.mean()),
    }


def _average_ai_weight(weights: pd.DataFrame) -> float:
    columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    if not columns:
        return 0.0
    return float(weights[columns].sum(axis=1).mean())


def _metric_row(
    *,
    result_name: str,
    source: I111Candidate,
    variant_name: str,
    description: str,
    metrics: PerformanceMetrics,
    source_metrics: PerformanceMetrics,
    primary_metrics: PerformanceMetrics | None,
    behavior: dict[str, float],
    source_behavior: dict[str, float],
    average_ai_growth_weight: float,
) -> dict[str, object]:
    del primary_metrics
    row = {
        "result_name": result_name,
        "source_strategy": source.name,
        "source_group": source.source_group,
        "variant_name": variant_name,
        "description": description,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "average_ai_growth_weight": average_ai_growth_weight,
        "delta_vs_source_cagr": metrics.cagr - source_metrics.cagr,
        "delta_vs_source_max_drawdown": metrics.max_drawdown - source_metrics.max_drawdown,
        "delta_vs_source_calmar": metrics.calmar - source_metrics.calmar,
    }
    row.update(behavior)
    row["delta_average_defensive_weight"] = (
        behavior["average_defensive_weight"] - source_behavior["average_defensive_weight"]
    )
    row["delta_hard_defensive_day_rate"] = (
        behavior["hard_defensive_day_rate"] - source_behavior["hard_defensive_day_rate"]
    )
    return row


def _primary_base_name(config: BotConfig, metrics: pd.DataFrame) -> str | None:
    preferred = f"{config.primary_strategy}__base"
    if preferred in set(metrics["result_name"]):
        return preferred
    base_rows = metrics[metrics["variant_name"].eq("base")]
    if base_rows.empty:
        return None
    return str(base_rows.sort_values("cagr", ascending=False).iloc[0]["result_name"])


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
    output["drawdown_penalty"] = (output["max_drawdown"].abs() - 0.205).clip(lower=0.0) * 2.0
    output["cagr_shortfall_penalty"] = output["delta_vs_primary_cagr"].clip(upper=0.0).abs() * 1.5
    output["stale_defense_penalty"] = output["hard_defensive_day_rate"].clip(lower=0.08) * 0.15
    output["ai_cap_penalty"] = (0.55 - output["average_ai_growth_weight"]).clip(lower=0.0) * 0.15
    output["research_score"] = (
        output["cagr"]
        + 0.40 * output["calmar"]
        + 0.25 * output["delta_vs_source_cagr"]
        + 0.10 * output["delta_vs_source_max_drawdown"]
        - output["drawdown_penalty"]
        - output["cagr_shortfall_penalty"]
        - output["stale_defense_penalty"]
        - output["ai_cap_penalty"]
    )
    output["near_22_cagr"] = output["cagr"] >= 0.215
    output["near_20_dd"] = output["max_drawdown"] >= -0.205
    output["improves_source_cagr"] = output["delta_vs_source_cagr"] > 0.0
    output["improves_source_drawdown"] = output["delta_vs_source_max_drawdown"] > 0.0
    output["promotion_candidate"] = (
        output["near_22_cagr"]
        & output["near_20_dd"]
        & (output["delta_vs_source_cagr"] >= -0.001)
        & (output["delta_hard_defensive_day_rate"] <= 0.01)
    )
    return output.sort_values("research_score", ascending=False)


def _variant_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("variant_name", observed=True)
    return (
        grouped.agg(
            candidates=("result_name", "count"),
            median_cagr=("cagr", "median"),
            best_cagr=("cagr", "max"),
            median_max_drawdown=("max_drawdown", "median"),
            best_calmar=("calmar", "max"),
            median_delta_vs_source_cagr=("delta_vs_source_cagr", "median"),
            median_delta_vs_source_max_drawdown=("delta_vs_source_max_drawdown", "median"),
            median_average_ai_growth_weight=("average_ai_growth_weight", "median"),
            promotion_rate=("promotion_candidate", "mean"),
        )
        .reset_index()
        .sort_values(["promotion_rate", "median_cagr"], ascending=False)
    )


def _summary_markdown(
    metrics: pd.DataFrame,
    variant_summary: pd.DataFrame,
    primary_name: str | None,
) -> str:
    lines = [
        "# Native I111 Risk-Repair Lab",
        "",
        "## Goal",
        "",
        (
            "Test strategy-native defensive repair and conditional AI-concentration controls "
            "against the i111 22% CAGR strategy family."
        ),
        "",
    ]
    if primary_name:
        primary = metrics[metrics["result_name"].eq(primary_name)].iloc[0]
        lines.extend(
            [
                "## Configured Champion Baseline",
                "",
                (
                    f"`{primary_name}`: CAGR {primary['cagr']:.2%}, "
                    f"max DD {primary['max_drawdown']:.2%}, Calmar {primary['calmar']:.2f}, "
                    f"hard-defense rate {primary['hard_defensive_day_rate']:.2%}, "
                    f"AI/growth weight {primary['average_ai_growth_weight']:.2%}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Best Native Candidates",
            "",
            (
                "| result | variant | CAGR | max DD | source CAGR delta | source DD delta | "
                "hard-defense delta | AI/growth wt | promote? |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in metrics.head(12).iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['variant_name']} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['delta_vs_source_cagr']:.2%} | "
            f"{row['delta_vs_source_max_drawdown']:.2%} | "
            f"{row['delta_hard_defensive_day_rate']:.2%} | "
            f"{row['average_ai_growth_weight']:.2%} | {bool(row['promotion_candidate'])} |"
        )
    lines.extend(["", "## Variant Summary", ""])
    lines.append(
        "| variant | candidates | median CAGR | median max DD | median source CAGR delta | promotion rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in variant_summary.iterrows():
        lines.append(
            f"| {row['variant_name']} | {int(row['candidates'])} | "
            f"{row['median_cagr']:.2%} | {row['median_max_drawdown']:.2%} | "
            f"{row['median_delta_vs_source_cagr']:.2%} | {row['promotion_rate']:.0%} |"
        )
    promoted = metrics[metrics["promotion_candidate"]]
    lines.extend(["", "## Readout", ""])
    if promoted.empty:
        lines.append(
            "No native variant cleared the preservation gate. Treat any apparent improvement as "
            "research-only until the next broader strategy-design iteration."
        )
    else:
        winner = promoted.sort_values("research_score", ascending=False).iloc[0]
        lines.append(
            f"Best gate-clearing native candidate: `{winner['result_name']}` with "
            f"CAGR {winner['cagr']:.2%}, max DD {winner['max_drawdown']:.2%}, and "
            f"AI/growth weight {winner['average_ai_growth_weight']:.2%}."
        )
    return "\n".join(lines) + "\n"
