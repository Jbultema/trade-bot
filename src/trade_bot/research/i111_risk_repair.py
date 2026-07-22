from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import (
    RISK_REPAIR_SIGNAL_TICKERS,
    BotConfig,
    StrategyConfig,
    configured_tickers,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.ai_concentration_repair import AiRepairSpec, _average_ai_weight
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.i111_candidates import (
    DEFAULT_I111_PREFIX,
    DEFAULT_UPSIDE_TOP_NAMES,
    I111Candidate,
    build_i111_candidates,
)
from trade_bot.research.prebreak_hindsight import _safe_float
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.research.top_tier_risk_repair import (
    DEFAULT_PREBREAK_SIGNAL_PANEL,
    DefensiveReliefSpec,
    TopTierRepairSpec,
    _behavior_metrics,
    _metric_row,
    _prebreak_behavior_rows,
    _prebreak_stage_dates,
    apply_top_tier_repair_variant,
    summarize_behavior,
    summarize_prebreak_stage_behavior,
    summarize_top_tier_repair,
)
from trade_bot.research.upside_capture import _apply_constructive_overlay
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_RISK_REPAIR_OUTPUT_DIR = Path("reports/i111_risk_repair")
I111RiskRepairCandidate = I111Candidate


@dataclass(frozen=True)
class I111RiskRepairResult:
    strategy_metrics: pd.DataFrame
    variant_summary: pd.DataFrame
    behavior_summary: pd.DataFrame
    prebreak_stage_summary: pd.DataFrame
    candidate_roster: pd.DataFrame
    summary: str


def run_i111_risk_repair_lab(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_RISK_REPAIR_OUTPUT_DIR,
    prebreak_signal_panel: str | Path = DEFAULT_PREBREAK_SIGNAL_PANEL,
    strategy_prefix: str = DEFAULT_I111_PREFIX,
    include_upside_research: bool = True,
    upside_top_names: tuple[str, ...] = DEFAULT_UPSIDE_TOP_NAMES,
    specs: tuple[TopTierRepairSpec, ...] | None = None,
    refresh_data: bool = False,
) -> I111RiskRepairResult:
    candidates = build_i111_risk_repair_candidates(
        config,
        strategy_prefix=strategy_prefix,
        include_upside_research=include_upside_research,
        upside_top_names=upside_top_names,
    )
    repair_specs = specs or default_i111_repair_specs()
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(candidates)
        | set(AI_GROWTH_TICKERS)
        | {"SPY", "QQQ", "SMH", "HYG", "LQD", "BIL", "GLD", "TLT"}
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    prebreak_dates = _prebreak_stage_dates(prebreak_signal_panel)
    metric_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    prebreak_rows: list[dict[str, object]] = []
    roster_rows: list[dict[str, object]] = []
    for candidate in candidates:
        base_result = _run_configured_candidate(config, candidate, prices)
        base_metrics = _metrics(base_result)
        base_behavior = _behavior_metrics(base_result)
        roster_rows.append(_candidate_roster_row(candidate, base_metrics, base_behavior))
        metric_rows.append(
            _metric_row(
                strategy=candidate.name,
                family=candidate.source_group,
                variant_name="base",
                metrics=base_metrics,
                base_metrics=base_metrics,
                base_behavior=base_behavior,
                behavior=base_behavior,
                average_ai_growth_weight=_average_ai_weight(base_result.weights),
            )
        )
        behavior_rows.append(
            _behavior_row(
                candidate.name, candidate.source_group, "base", base_behavior, base_behavior
            )
        )
        prebreak_rows.extend(
            _prebreak_behavior_rows(
                candidate.name, "base", base_result, base_result, prebreak_dates
            )
        )
        for spec in repair_specs:
            variant_result = apply_top_tier_repair_variant(
                base_result,
                prices,
                spec,
                transaction_cost_bps=config.execution.transaction_cost_bps,
            )
            variant_metrics = _metrics(variant_result)
            variant_behavior = _behavior_metrics(variant_result)
            metric_rows.append(
                _metric_row(
                    strategy=candidate.name,
                    family=candidate.source_group,
                    variant_name=spec.name,
                    metrics=variant_metrics,
                    base_metrics=base_metrics,
                    base_behavior=base_behavior,
                    behavior=variant_behavior,
                    average_ai_growth_weight=_average_ai_weight(base_result.weights),
                )
            )
            behavior_rows.append(
                _behavior_row(
                    candidate.name,
                    candidate.source_group,
                    spec.name,
                    base_behavior,
                    variant_behavior,
                )
            )
            prebreak_rows.extend(
                _prebreak_behavior_rows(
                    candidate.name,
                    spec.name,
                    base_result,
                    variant_result,
                    prebreak_dates,
                )
            )

    strategy_metrics = pd.DataFrame(metric_rows)
    behavior_summary = summarize_behavior(pd.DataFrame(behavior_rows))
    prebreak_stage_summary = summarize_prebreak_stage_behavior(pd.DataFrame(prebreak_rows))
    variant_summary = summarize_top_tier_repair(
        strategy_metrics,
        behavior_summary,
        prebreak_stage_summary,
    )
    candidate_roster = pd.DataFrame(roster_rows).sort_values("base_cagr", ascending=False)
    result = I111RiskRepairResult(
        strategy_metrics=strategy_metrics,
        variant_summary=variant_summary,
        behavior_summary=behavior_summary,
        prebreak_stage_summary=prebreak_stage_summary,
        candidate_roster=candidate_roster,
        summary=_summary_markdown(candidate_roster, variant_summary, prebreak_stage_summary),
    )
    write_i111_risk_repair_outputs(result, output_dir=output_dir)
    write_research_manifest(
        output_dir,
        study="i111_risk_repair",
        config=config,
        prices=prices,
        parameters={
            "strategy_prefix": strategy_prefix,
            "include_upside_research": include_upside_research,
            "upside_top_names": list(upside_top_names),
            "prebreak_signal_panel": str(prebreak_signal_panel),
            "refresh_data": refresh_data,
            "repair_specs": [spec.name for spec in repair_specs],
        },
        artifacts=[
            "candidate_roster.csv",
            "strategy_metrics.csv",
            "variant_summary.csv",
            "behavior_summary.csv",
            "prebreak_stage_summary.csv",
            "summary.md",
        ],
    )
    return result


def build_i111_risk_repair_candidates(
    config: BotConfig,
    *,
    strategy_prefix: str = DEFAULT_I111_PREFIX,
    include_upside_research: bool = True,
    upside_top_names: tuple[str, ...] = DEFAULT_UPSIDE_TOP_NAMES,
) -> tuple[I111RiskRepairCandidate, ...]:
    return build_i111_candidates(
        config,
        strategy_prefix=strategy_prefix,
        include_upside_research=include_upside_research,
        upside_top_names=upside_top_names,
    )


def default_i111_repair_specs() -> tuple[TopTierRepairSpec, ...]:
    ai_cap45 = AiRepairSpec(
        name="ai_dual_confirm_break_cap45_bil",
        stress_signal="ai_dual_confirm_break",
        ai_cap=0.45,
        destination="bil",
    )
    ai_cap35 = AiRepairSpec(
        name="ai_dual_confirm_break_cap35_bil",
        stress_signal="ai_dual_confirm_break",
        ai_cap=0.35,
        destination="bil",
    )
    relief75 = DefensiveReliefSpec("defensive_relief_cap75", 0.75)
    relief65 = DefensiveReliefSpec("defensive_relief_cap65", 0.65)
    relief55 = DefensiveReliefSpec("defensive_relief_cap55", 0.55)
    return (
        TopTierRepairSpec(name="ai_cap45_bil", ai_repair=ai_cap45),
        TopTierRepairSpec(name="ai_cap35_bil", ai_repair=ai_cap35),
        TopTierRepairSpec(name="defensive_relief_cap75", defensive_relief=relief75),
        TopTierRepairSpec(name="defensive_relief_cap65", defensive_relief=relief65),
        TopTierRepairSpec(name="defensive_relief_cap55", defensive_relief=relief55),
        TopTierRepairSpec(
            name="ai_cap45_plus_relief75", ai_repair=ai_cap45, defensive_relief=relief75
        ),
        TopTierRepairSpec(
            name="ai_cap45_plus_relief65", ai_repair=ai_cap45, defensive_relief=relief65
        ),
        TopTierRepairSpec(
            name="ai_cap45_plus_relief55", ai_repair=ai_cap45, defensive_relief=relief55
        ),
        TopTierRepairSpec(
            name="ai_cap35_plus_relief65", ai_repair=ai_cap35, defensive_relief=relief65
        ),
    )


def write_i111_risk_repair_outputs(
    result: I111RiskRepairResult,
    *,
    output_dir: str | Path = DEFAULT_I111_RISK_REPAIR_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.candidate_roster.to_csv(output_path / "candidate_roster.csv", index=False)
    result.strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    result.variant_summary.to_csv(output_path / "variant_summary.csv", index=False)
    result.behavior_summary.to_csv(output_path / "behavior_summary.csv", index=False)
    result.prebreak_stage_summary.to_csv(output_path / "prebreak_stage_summary.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _run_configured_candidate(
    config: BotConfig,
    candidate: I111RiskRepairCandidate,
    prices: pd.DataFrame,
) -> BacktestResult:
    candidate_prices = _strategy_prices(prices, candidate.strategy)
    weights = build_strategy_weights(candidate_prices, candidate.strategy)
    if candidate.overlay:
        weights = _apply_constructive_overlay(candidate_prices, weights, candidate.overlay)
    return run_backtest(
        candidate.name,
        candidate_prices,
        weights,
        config.execution,
        volatility_target=candidate.strategy.volatility_target,
        drawdown_control=candidate.strategy.drawdown_control,
    )


def _candidate_tickers(candidates: tuple[I111RiskRepairCandidate, ...]) -> set[str]:
    tickers: set[str] = set(RISK_REPAIR_SIGNAL_TICKERS)
    for candidate in candidates:
        tickers.update(required_strategy_tickers(candidate.strategy))
    return tickers


def _strategy_prices(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = list(
        dict.fromkeys([*required_strategy_tickers(strategy), *RISK_REPAIR_SIGNAL_TICKERS])
    )
    missing = unusable_required_price_columns(prices, columns)
    if missing:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing}")
    frame = prices[columns].sort_index()
    return frame.loc[frame.notna().any(axis=1)]


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _candidate_roster_row(
    candidate: I111RiskRepairCandidate,
    metrics: PerformanceMetrics,
    behavior: dict[str, float],
) -> dict[str, object]:
    return {
        "strategy": candidate.name,
        "source_group": candidate.source_group,
        "base_cagr": metrics.cagr,
        "base_max_drawdown": metrics.max_drawdown,
        "base_calmar": metrics.calmar,
        "base_sharpe": metrics.sharpe,
        "base_average_defensive_weight": behavior["average_defensive_weight"],
        "base_hard_defensive_day_rate": behavior["hard_defensive_day_rate"],
        "base_max_hard_defensive_run_days": behavior["max_hard_defensive_run_days"],
    }


def _behavior_row(
    strategy: str,
    family: str,
    variant_name: str,
    base_behavior: dict[str, float],
    behavior: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {"strategy": strategy, "family": family, "variant_name": variant_name}
    row.update(behavior)
    row["delta_hard_defensive_day_rate"] = (
        behavior["hard_defensive_day_rate"] - base_behavior["hard_defensive_day_rate"]
    )
    row["delta_max_hard_defensive_run_days"] = (
        behavior["max_hard_defensive_run_days"] - base_behavior["max_hard_defensive_run_days"]
    )
    return row


def _summary_markdown(
    candidate_roster: pd.DataFrame,
    variant_summary: pd.DataFrame,
    prebreak_stage_summary: pd.DataFrame,
) -> str:
    lines = [
        "# I111 Risk-Repair Lab",
        "",
        "This report applies the AI concentration and defensive-relief repair overlays",
        "directly to the configured i111 runtime family plus the strongest nearby",
        "upside-capture research variants.",
        "",
        "## Candidate Roster",
        "",
    ]
    if candidate_roster.empty:
        lines.append("- no candidates were tested")
    else:
        for _, row in candidate_roster.iterrows():
            lines.append(
                "- "
                f"{row['strategy']} ({row['source_group']}): "
                f"CAGR {_safe_float(row['base_cagr']):.2%}, "
                f"max DD {_safe_float(row['base_max_drawdown']):.2%}, "
                f"Calmar {_safe_float(row['base_calmar']):.2f}, "
                f"hard-defense rate {_safe_float(row['base_hard_defensive_day_rate']):.2%}"
            )
    lines.extend(["", "## Variant Read", ""])
    if variant_summary.empty:
        lines.append("- no variant rows were available")
    else:
        for _, row in variant_summary.iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: gate {row['promotion_gate']}; "
                f"median CAGR {_safe_float(row['median_cagr']):.2%} "
                f"({_safe_float(row['median_delta_cagr']):+.2%}), "
                f"median max DD {_safe_float(row['median_max_drawdown']):.2%} "
                f"({_safe_float(row['median_delta_max_drawdown']):+.2%}), "
                f"CAGR win rate {_safe_float(row['cagr_win_rate']):.0%}, "
                f"DD win rate {_safe_float(row['drawdown_win_rate']):.0%}, "
                f"early prebreak defensive delta "
                f"{_safe_float(row['early_prebreak_delta_defensive_weight']):+.2%}"
            )
    lines.extend(["", "## Pre-Break Behavior", ""])
    if prebreak_stage_summary.empty:
        lines.append("- no pre-break stage rows were available")
    else:
        for _, row in prebreak_stage_summary.iterrows():
            lines.append(
                "- "
                f"{row['variant_name']} / {row['stage']}: "
                f"defensive delta {_safe_float(row['delta_average_defensive_weight']):+.2%}, "
                f"hard-defense delta {_safe_float(row['delta_hard_defensive_day_rate']):+.2%}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "A promoted repair must improve median CAGR, avoid worsening median max",
            "drawdown, win on both metrics across most candidates, and not increase",
            "early pre-break defensiveness.",
        ]
    )
    return "\n".join(lines) + "\n"
