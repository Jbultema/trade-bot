from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import moving_average
from trade_bot.research.ai_concentration_repair import (
    AiRepairSpec,
    _average_ai_weight,
    _stress_signal,
    apply_ai_repair_variant,
)
from trade_bot.research.experiments import (
    _candidate_tickers,
    _load_previous_candidates,
    _load_previous_scorecards,
    _strategy_prices,
)
from trade_bot.research.prebreak_hindsight import _safe_float
from trade_bot.research.risk_landscape_survey import (
    AI_GROWTH_TICKERS,
    _credit_weak,
    _result_from_weights,
)
from trade_bot.research.risk_policy_backtest import (
    _active_experiment_root,
    _run_candidate_backtest,
    _selected_candidates,
)

DEFAULT_TOP_TIER_REPAIR_OUTPUT_DIR = Path("reports/top_tier_risk_repair")
DEFAULT_PREBREAK_SIGNAL_PANEL = Path("reports/prebreak_hindsight/snapshot_signal_panel.csv")
DEFENSIVE_TICKER = "BIL"


@dataclass(frozen=True)
class DefensiveReliefSpec:
    name: str
    max_defensive_weight: float
    destination: str = "spy_qqq"


@dataclass(frozen=True)
class TopTierRepairSpec:
    name: str
    ai_repair: AiRepairSpec | None = None
    defensive_relief: DefensiveReliefSpec | None = None


@dataclass(frozen=True)
class TopTierRiskRepairResult:
    strategy_metrics: pd.DataFrame
    variant_summary: pd.DataFrame
    behavior_summary: pd.DataFrame
    prebreak_stage_summary: pd.DataFrame
    summary: str


def run_top_tier_risk_repair_lab(
    config: BotConfig,
    *,
    iteration: int = 164,
    top_n: int = 20,
    specs: tuple[TopTierRepairSpec, ...] | None = None,
    experiment_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_TOP_TIER_REPAIR_OUTPUT_DIR,
    prebreak_signal_panel: str | Path = DEFAULT_PREBREAK_SIGNAL_PANEL,
    refresh_data: bool = False,
) -> TopTierRiskRepairResult:
    experiment_root = Path(experiment_root) if experiment_root else _active_experiment_root()
    candidates = _selected_candidates(
        iteration,
        scorecards=_load_previous_scorecards(experiment_root, iteration + 1),
        candidates_manifest=_load_previous_candidates(experiment_root, iteration + 1),
        top_n=top_n,
        experiment_root=experiment_root,
    )
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
    )
    repair_specs = specs or default_top_tier_repair_specs()
    prebreak_dates = _prebreak_stage_dates(prebreak_signal_panel)
    metric_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    prebreak_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_result = _run_candidate_backtest(config, candidate, prices, candidate_prices)
        base_metrics = _metrics(base_result)
        base_behavior = _behavior_metrics(base_result)
        metric_rows.append(
            _metric_row(
                strategy=candidate.name,
                family=candidate.family,
                variant_name="base",
                metrics=base_metrics,
                base_metrics=base_metrics,
                base_behavior=base_behavior,
                behavior=base_behavior,
                average_ai_growth_weight=_average_ai_weight(base_result.weights),
            )
        )
        behavior_rows.append(
            _behavior_row(candidate.name, candidate.family, "base", base_behavior, base_behavior)
        )
        prebreak_rows.extend(
            _prebreak_behavior_rows(candidate.name, "base", base_result, base_result, prebreak_dates)
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
                    family=candidate.family,
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
                    candidate.family,
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
    result = TopTierRiskRepairResult(
        strategy_metrics=strategy_metrics,
        variant_summary=variant_summary,
        behavior_summary=behavior_summary,
        prebreak_stage_summary=prebreak_stage_summary,
        summary=build_top_tier_repair_summary(variant_summary, prebreak_stage_summary),
    )
    write_top_tier_risk_repair_outputs(result, output_dir=output_dir)
    return result


def default_top_tier_repair_specs() -> tuple[TopTierRepairSpec, ...]:
    ai_cap45 = AiRepairSpec(
        name="ai_dual_confirm_break_cap45_bil",
        stress_signal="ai_dual_confirm_break",
        ai_cap=0.45,
        destination="bil",
    )
    return (
        TopTierRepairSpec(name="ai_cap45_bil", ai_repair=ai_cap45),
        TopTierRepairSpec(
            name="defensive_relief_cap65",
            defensive_relief=DefensiveReliefSpec("defensive_relief_cap65", 0.65),
        ),
        TopTierRepairSpec(
            name="defensive_relief_cap55",
            defensive_relief=DefensiveReliefSpec("defensive_relief_cap55", 0.55),
        ),
        TopTierRepairSpec(
            name="ai_cap45_plus_relief65",
            ai_repair=ai_cap45,
            defensive_relief=DefensiveReliefSpec("defensive_relief_cap65", 0.65),
        ),
        TopTierRepairSpec(
            name="ai_cap45_plus_relief55",
            ai_repair=ai_cap45,
            defensive_relief=DefensiveReliefSpec("defensive_relief_cap55", 0.55),
        ),
    )


def apply_top_tier_repair_variant(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    spec: TopTierRepairSpec,
    *,
    transaction_cost_bps: float,
) -> BacktestResult:
    result = base_result
    if spec.defensive_relief is not None:
        result = apply_defensive_relief_variant(
            result,
            prices,
            spec.defensive_relief,
            transaction_cost_bps=transaction_cost_bps,
        )
    if spec.ai_repair is not None:
        result = apply_ai_repair_variant(
            result,
            prices,
            spec.ai_repair,
            transaction_cost_bps=transaction_cost_bps,
        )
    return BacktestResult(
        name=f"{base_result.name}__{spec.name}",
        equity=result.equity.rename(f"{base_result.name}__{spec.name}"),
        returns=result.returns.rename(f"{base_result.name}__{spec.name}"),
        gross_returns=result.gross_returns.rename(f"{base_result.name}__{spec.name}"),
        weights=result.weights,
        target_weights=result.target_weights,
        turnover=result.turnover.rename(f"{base_result.name}__{spec.name}"),
        transaction_costs=result.transaction_costs.rename(f"{base_result.name}__{spec.name}"),
    )


def apply_defensive_relief_variant(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    spec: DefensiveReliefSpec,
    *,
    transaction_cost_bps: float,
) -> BacktestResult:
    aligned_prices = prices.reindex(base_result.weights.index).ffill()
    weights = base_result.weights.reindex(aligned_prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=aligned_prices.columns, fill_value=0.0)
    if DEFENSIVE_TICKER not in weights:
        return base_result
    relief = _defensive_relief_signal(aligned_prices, base_result)
    defensive = weights[DEFENSIVE_TICKER].clip(lower=0.0, upper=1.0)
    excess_defense = (defensive - spec.max_defensive_weight).clip(lower=0.0)
    release = pd.Series(0.0, index=weights.index)
    release.loc[relief] = excess_defense.loc[relief]
    if release.abs().sum() <= 0.0:
        return base_result
    adjusted = weights.copy()
    adjusted[DEFENSIVE_TICKER] = (adjusted[DEFENSIVE_TICKER] - release).clip(lower=0.0)
    destination_weights = _relief_destination_weights(spec.destination, adjusted, release)
    for ticker, addition in destination_weights.items():
        adjusted[ticker] = adjusted[ticker] + addition
    return _result_from_weights(
        base_result,
        aligned_prices,
        adjusted,
        transaction_cost_bps,
        f"{base_result.name}__{spec.name}",
    )


def summarize_top_tier_repair(
    strategy_metrics: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    prebreak_stage_summary: pd.DataFrame,
) -> pd.DataFrame:
    variants = strategy_metrics[strategy_metrics["variant_name"].ne("base")].copy()
    if variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant_name, group in variants.groupby("variant_name", sort=False):
        behavior = behavior_summary[behavior_summary["variant_name"].eq(variant_name)]
        prebreak = prebreak_stage_summary[
            prebreak_stage_summary["variant_name"].eq(variant_name)
        ]
        early = prebreak[prebreak["stage"].isin(["long_lead", "early_watch"])]
        confirmed = prebreak[prebreak["stage"].eq("confirmed_prebreak")]
        row = {
            "variant_name": variant_name,
            "strategies": len(group),
            "median_cagr": _median(group, "cagr"),
            "median_max_drawdown": _median(group, "max_drawdown"),
            "median_calmar": _median(group, "calmar"),
            "median_delta_cagr": _median(group, "delta_cagr_vs_base"),
            "median_delta_max_drawdown": _median(group, "delta_max_drawdown_vs_base"),
            "median_delta_calmar": _median(group, "delta_calmar_vs_base"),
            "cagr_win_rate": _positive_rate(group, "delta_cagr_vs_base"),
            "drawdown_win_rate": _positive_rate(group, "delta_max_drawdown_vs_base"),
            "median_average_ai_growth_weight": _median(group, "average_ai_growth_weight"),
            "median_delta_hard_defensive_day_rate": _median(
                behavior,
                "median_delta_hard_defensive_day_rate",
            ),
            "median_delta_max_hard_defensive_run_days": _median(
                behavior,
                "median_delta_max_hard_defensive_run_days",
            ),
            "early_prebreak_delta_defensive_weight": _median(
                early,
                "delta_average_defensive_weight",
            ),
            "confirmed_prebreak_delta_defensive_weight": _median(
                confirmed,
                "delta_average_defensive_weight",
            ),
        }
        row["promotion_gate"] = _promotion_gate(row)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "promotion_gate",
                "median_delta_cagr",
                "median_delta_max_drawdown",
                "early_prebreak_delta_defensive_weight",
            ],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_behavior(behavior_rows: pd.DataFrame) -> pd.DataFrame:
    if behavior_rows.empty:
        return pd.DataFrame()
    rows = []
    for variant_name, group in behavior_rows[behavior_rows["variant_name"].ne("base")].groupby(
        "variant_name",
        sort=False,
    ):
        rows.append(
            {
                "variant_name": variant_name,
                "strategies": len(group),
                "median_average_defensive_weight": _median(group, "average_defensive_weight"),
                "median_hard_defensive_day_rate": _median(group, "hard_defensive_day_rate"),
                "median_max_hard_defensive_run_days": _median(
                    group,
                    "max_hard_defensive_run_days",
                ),
                "median_delta_hard_defensive_day_rate": _median(
                    group,
                    "delta_hard_defensive_day_rate",
                ),
                "median_delta_max_hard_defensive_run_days": _median(
                    group,
                    "delta_max_hard_defensive_run_days",
                ),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_prebreak_stage_behavior(prebreak_rows: pd.DataFrame) -> pd.DataFrame:
    if prebreak_rows.empty:
        return pd.DataFrame()
    rows = []
    variants = prebreak_rows[prebreak_rows["variant_name"].ne("base")]
    for (variant_name, stage), group in variants.groupby(["variant_name", "stage"], sort=False):
        rows.append(
            {
                "variant_name": variant_name,
                "stage": stage,
                "observations": int(group["observations"].sum()),
                "median_average_defensive_weight": _median(group, "average_defensive_weight"),
                "median_hard_defensive_day_rate": _median(group, "hard_defensive_day_rate"),
                "delta_average_defensive_weight": _median(
                    group,
                    "delta_average_defensive_weight",
                ),
                "delta_hard_defensive_day_rate": _median(
                    group,
                    "delta_hard_defensive_day_rate",
                ),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_top_tier_repair_summary(
    variant_summary: pd.DataFrame,
    prebreak_stage_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Top-Tier Risk Repair Lab",
        "",
        "This lab integrates the promoted AI concentration repair with the previous",
        "top-tier strategy candidates and separately tests defensive-relief overlays",
        "that are designed to reduce early or persistent BIL exposure.",
        "",
        "## Variant Read",
        "",
    ]
    if variant_summary.empty:
        lines.append("- no variant rows were available")
    else:
        for _, row in variant_summary.iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: gate {row['promotion_gate']}; "
                f"CAGR {_safe_float(row['median_cagr']):.2%} "
                f"({_safe_float(row['median_delta_cagr']):+.2%}), "
                f"max DD {_safe_float(row['median_max_drawdown']):.2%} "
                f"({_safe_float(row['median_delta_max_drawdown']):+.2%}), "
                f"hard-defense rate delta "
                f"{_safe_float(row['median_delta_hard_defensive_day_rate']):+.2%}, "
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
            "## Gate Definition",
            "",
            "- promote: positive median CAGR, non-worse median max drawdown, at least",
            "  80% CAGR and drawdown win rates, and no increase in early pre-break",
            "  defensiveness.",
            "- watchlist: directionally useful but misses one robustness or behavior bar.",
            "- reject: worsens performance or fails to address the behavior problem.",
        ]
    )
    return "\n".join(lines)


def write_top_tier_risk_repair_outputs(
    result: TopTierRiskRepairResult,
    *,
    output_dir: str | Path = DEFAULT_TOP_TIER_REPAIR_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    result.variant_summary.to_csv(output_path / "variant_summary.csv", index=False)
    result.behavior_summary.to_csv(output_path / "behavior_summary.csv", index=False)
    result.prebreak_stage_summary.to_csv(output_path / "prebreak_stage_summary.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _defensive_relief_signal(prices: pd.DataFrame, result: BacktestResult) -> pd.Series:
    spy_trend = prices["SPY"].ffill() > moving_average(prices[["SPY"]], 200)["SPY"]
    qqq_trend = prices["QQQ"].ffill() > moving_average(prices[["QQQ"]], 100)["QQQ"]
    credit_ok = ~_credit_weak(prices)
    dual_confirm = _stress_signal("ai_dual_confirm_break", prices, result)
    return (spy_trend & qqq_trend & credit_ok & ~dual_confirm).shift(1, fill_value=False).astype(
        bool
    )


def _relief_destination_weights(
    destination: str,
    weights: pd.DataFrame,
    release: pd.Series,
) -> dict[str, pd.Series]:
    if destination == "spy_qqq":
        return {"SPY": release * 0.70, "QQQ": release * 0.30}
    if destination == "spy":
        return {"SPY": release}
    risk_columns = [
        column
        for column in weights.columns
        if column != DEFENSIVE_TICKER and pd.to_numeric(weights[column], errors="coerce").sum() > 0
    ]
    if not risk_columns:
        return {"SPY": release * 0.70, "QQQ": release * 0.30}
    risk_weight = weights[risk_columns].sum(axis=1)
    allocation = {}
    for column in risk_columns:
        allocation[column] = release * weights[column].div(risk_weight.where(risk_weight > 0.0)).fillna(0.0)
    return allocation


def _metric_row(
    *,
    strategy: str,
    family: str,
    variant_name: str,
    metrics: PerformanceMetrics,
    base_metrics: PerformanceMetrics,
    base_behavior: dict[str, float],
    behavior: dict[str, float],
    average_ai_growth_weight: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "variant_name": variant_name,
        "average_ai_growth_weight": average_ai_growth_weight,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "delta_cagr_vs_base": metrics.cagr - base_metrics.cagr,
        "delta_max_drawdown_vs_base": metrics.max_drawdown - base_metrics.max_drawdown,
        "delta_calmar_vs_base": metrics.calmar - base_metrics.calmar,
        "average_defensive_weight": behavior["average_defensive_weight"],
        "hard_defensive_day_rate": behavior["hard_defensive_day_rate"],
        "delta_hard_defensive_day_rate": (
            behavior["hard_defensive_day_rate"] - base_behavior["hard_defensive_day_rate"]
        ),
        "max_hard_defensive_run_days": behavior["max_hard_defensive_run_days"],
        "delta_max_hard_defensive_run_days": (
            behavior["max_hard_defensive_run_days"]
            - base_behavior["max_hard_defensive_run_days"]
        ),
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


def _prebreak_behavior_rows(
    strategy: str,
    variant_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
    prebreak_dates: pd.DataFrame,
) -> list[dict[str, object]]:
    if prebreak_dates.empty:
        return []
    rows = []
    base_defensive = _defensive_weight(base_result).reindex(prebreak_dates["market_date"]).dropna()
    variant_defensive = _defensive_weight(variant_result).reindex(prebreak_dates["market_date"]).dropna()
    for stage, stage_dates in prebreak_dates.groupby("stage", sort=False):
        dates = pd.DatetimeIndex(stage_dates["market_date"])
        base_stage = base_defensive.reindex(dates).dropna()
        variant_stage = variant_defensive.reindex(dates).dropna()
        if len(base_stage) < 2 or len(variant_stage) < 2:
            continue
        rows.append(
            {
                "strategy": strategy,
                "variant_name": variant_name,
                "stage": stage,
                "observations": min(len(base_stage), len(variant_stage)),
                "average_defensive_weight": float(variant_stage.mean()),
                "hard_defensive_day_rate": float((variant_stage >= 0.75).mean()),
                "base_average_defensive_weight": float(base_stage.mean()),
                "base_hard_defensive_day_rate": float((base_stage >= 0.75).mean()),
                "delta_average_defensive_weight": float(
                    variant_stage.mean() - base_stage.mean()
                ),
                "delta_hard_defensive_day_rate": float(
                    (variant_stage >= 0.75).mean() - (base_stage >= 0.75).mean()
                ),
            }
        )
    return rows


def _behavior_metrics(result: BacktestResult) -> dict[str, float]:
    defensive = _defensive_weight(result)
    hard = defensive >= 0.75
    return {
        "average_defensive_weight": float(defensive.mean()),
        "high_defensive_day_rate": float((defensive >= 0.50).mean()),
        "hard_defensive_day_rate": float(hard.mean()),
        "max_hard_defensive_run_days": float(_max_true_run(hard)),
    }


def _defensive_weight(result: BacktestResult) -> pd.Series:
    if DEFENSIVE_TICKER not in result.weights:
        return pd.Series(0.0, index=result.weights.index)
    return result.weights[DEFENSIVE_TICKER].clip(lower=0.0, upper=1.0).fillna(0.0)


def _prebreak_stage_dates(path: str | Path) -> pd.DataFrame:
    signal_path = Path(path)
    if not signal_path.exists():
        return pd.DataFrame(columns=["market_date", "stage"])
    frame = pd.read_csv(signal_path, usecols=["market_date", "days_to_break"])
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame["days_to_break"] = pd.to_numeric(frame["days_to_break"], errors="coerce")
    frame = frame.dropna(subset=["market_date", "days_to_break"]).drop_duplicates()
    frame["stage"] = frame["days_to_break"].map(_stage_from_days_to_break)
    return frame.dropna(subset=["stage"])[["market_date", "stage"]].reset_index(drop=True)


def _stage_from_days_to_break(days: float) -> str | None:
    if 120 <= days <= 365:
        return "long_lead"
    if 60 <= days < 120:
        return "early_watch"
    if 15 <= days < 60:
        return "confirmed_prebreak"
    if 0 <= days < 15:
        return "break_window"
    return None


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _promotion_gate(row: dict[str, object]) -> str:
    if (
        _safe_float(row.get("median_delta_cagr")) > 0.0
        and _safe_float(row.get("median_delta_max_drawdown")) >= 0.0
        and _safe_float(row.get("cagr_win_rate")) >= 0.80
        and _safe_float(row.get("drawdown_win_rate")) >= 0.80
        and _safe_float(row.get("early_prebreak_delta_defensive_weight")) <= 0.001
    ):
        return "0_promote_candidate"
    if (
        _safe_float(row.get("median_delta_cagr")) > 0.0
        and _safe_float(row.get("median_delta_max_drawdown")) >= -0.005
    ):
        return "1_watchlist"
    return "2_reject"


def _max_true_run(values: pd.Series) -> int:
    max_run = 0
    current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        max_run = max(max_run, current)
    return max_run


def _positive_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float((values > 0.0).mean())


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").median())
