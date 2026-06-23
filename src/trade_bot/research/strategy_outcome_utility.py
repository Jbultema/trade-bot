from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_CHURN_PENALTY_WEIGHT,
    DEFAULT_OUTCOME_FLOOR_CAGR,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_MIN_LEFT_TAIL_REGIME_RETURN,
    DEFAULT_OUTCOME_MIN_WALK_FORWARD_POSITIVE_RATE,
    DEFAULT_OUTCOME_MIN_WORST_3Y_CAGR,
    DEFAULT_OUTCOME_OVERFIT_PENALTY_WEIGHT,
    DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
    DEFAULT_OUTCOME_TARGET_CAGR,
)


@dataclass(frozen=True)
class StrategyOutcomeUtilityConfig:
    horizon_years: int = DEFAULT_OUTCOME_HORIZON_YEARS
    starting_account_value: float = DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE
    annual_contribution: float = DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION
    soft_drawdown_limit: float = DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT
    hard_drawdown_limit: float = DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT
    floor_cagr: float = DEFAULT_OUTCOME_FLOOR_CAGR
    target_cagr: float = DEFAULT_OUTCOME_TARGET_CAGR
    min_walk_forward_positive_rate: float = DEFAULT_OUTCOME_MIN_WALK_FORWARD_POSITIVE_RATE
    min_worst_3y_cagr: float = DEFAULT_OUTCOME_MIN_WORST_3Y_CAGR
    min_left_tail_regime_return: float = DEFAULT_OUTCOME_MIN_LEFT_TAIL_REGIME_RETURN
    overfit_penalty_weight: float = DEFAULT_OUTCOME_OVERFIT_PENALTY_WEIGHT
    churn_penalty_weight: float = DEFAULT_OUTCOME_CHURN_PENALTY_WEIGHT


def enrich_strategy_outcome_utility(
    frame: pd.DataFrame,
    *,
    benchmark_metrics: pd.DataFrame | None = None,
    config: StrategyOutcomeUtilityConfig | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cfg = config or StrategyOutcomeUtilityConfig()
    output = frame.copy()
    cagr = _numeric_series(output, "cagr")
    max_drawdown = _numeric_series(output, "max_drawdown")

    terminal_wealth = terminal_wealth_from_cagr(
        cagr,
        years=cfg.horizon_years,
        starting_account_value=cfg.starting_account_value,
    )
    terminal_wealth_with_contributions = terminal_wealth_from_cagr(
        cagr,
        years=cfg.horizon_years,
        starting_account_value=cfg.starting_account_value,
        annual_contribution=cfg.annual_contribution,
    )
    output[f"terminal_wealth_{cfg.horizon_years}y"] = terminal_wealth
    output[f"terminal_wealth_with_contributions_{cfg.horizon_years}y"] = (
        terminal_wealth_with_contributions
    )

    for ticker in ("spy", "qqq"):
        benchmark_cagr = _benchmark_cagr(output, ticker, benchmark_metrics)
        benchmark_wealth = terminal_wealth_from_cagr(
            benchmark_cagr,
            years=cfg.horizon_years,
            starting_account_value=cfg.starting_account_value,
            annual_contribution=cfg.annual_contribution,
        )
        output[f"wealth_multiple_vs_{ticker}"] = (
            terminal_wealth_with_contributions
            / benchmark_wealth.replace(
                0.0,
                np.nan,
            )
        )

    output["drawdown_recovery_return"] = drawdown_recovery_return(max_drawdown)
    output["drawdown_soft_penalty"] = drawdown_soft_penalty(max_drawdown, config=cfg)
    output["drawdown_hard_penalty"] = drawdown_hard_penalty(max_drawdown, config=cfg)

    wealth_score = _wealth_score(terminal_wealth_with_contributions, config=cfg)
    validation_score = _validation_score(output, config=cfg)
    churn_penalty = _churn_penalty(output)
    outcome_score = (
        0.78 * wealth_score
        + 0.22 * validation_score
        - 0.18 * output["drawdown_soft_penalty"].fillna(0.0)
        - 0.35 * output["drawdown_hard_penalty"].fillna(0.0)
        - cfg.churn_penalty_weight * churn_penalty
    ).clip(0.0, 1.0)
    output["growth_constrained_utility_score"] = outcome_score
    output["growth_utility_tier"] = output.apply(_growth_utility_tier, axis=1)
    return add_outcome_frontier_flags(output)


def enrich_after_tax_outcome_utility(
    frame: pd.DataFrame,
    *,
    benchmark_metrics: pd.DataFrame | None = None,
    config: StrategyOutcomeUtilityConfig | None = None,
) -> pd.DataFrame:
    """Add growth utility columns based on estimated after-tax CAGR/drawdown."""

    if frame.empty or "after_tax_cagr" not in frame or "after_tax_max_drawdown" not in frame:
        return frame.copy()
    output = frame.copy()
    tax_input = output.copy()
    tax_input["cagr"] = _numeric_series(output, "after_tax_cagr")
    tax_input["max_drawdown"] = _numeric_series(output, "after_tax_max_drawdown")
    enriched = enrich_strategy_outcome_utility(
        tax_input,
        benchmark_metrics=benchmark_metrics,
        config=config,
    )
    cfg = config or StrategyOutcomeUtilityConfig()
    mapping = {
        f"terminal_wealth_{cfg.horizon_years}y": f"after_tax_terminal_wealth_{cfg.horizon_years}y",
        f"terminal_wealth_with_contributions_{cfg.horizon_years}y": f"after_tax_terminal_wealth_with_contributions_{cfg.horizon_years}y",
        "wealth_multiple_vs_spy": "after_tax_wealth_multiple_vs_spy",
        "wealth_multiple_vs_qqq": "after_tax_wealth_multiple_vs_qqq",
        "drawdown_recovery_return": "after_tax_drawdown_recovery_return",
        "drawdown_soft_penalty": "after_tax_drawdown_soft_penalty",
        "drawdown_hard_penalty": "after_tax_drawdown_hard_penalty",
        "growth_constrained_utility_score": "after_tax_growth_constrained_utility_score",
        "growth_utility_tier": "after_tax_growth_utility_tier",
        "is_growth_pareto_efficient": "after_tax_is_growth_pareto_efficient",
    }
    for source, target in mapping.items():
        if source in enriched:
            output[target] = enriched[source]
    return output


def terminal_wealth_from_cagr(
    cagr: pd.Series | float,
    *,
    years: int,
    starting_account_value: float,
    annual_contribution: float = 0.0,
) -> pd.Series:
    rates = _as_series(cagr).astype(float)
    growth_base = (1.0 + rates).clip(lower=0.0) ** float(years)
    starting_wealth = starting_account_value * growth_base
    if annual_contribution == 0.0:
        return starting_wealth
    annuity_factor = pd.Series(float(years), index=rates.index, dtype=float)
    non_zero = rates.abs() > 1e-12
    annuity_factor.loc[non_zero] = ((1.0 + rates.loc[non_zero]) ** float(years) - 1.0) / rates.loc[
        non_zero
    ]
    return starting_wealth + annual_contribution * annuity_factor


def drawdown_recovery_return(max_drawdown: pd.Series | float) -> pd.Series:
    values = _as_series(max_drawdown).astype(float)
    drawdown_depth = (-values.clip(upper=0.0)).clip(lower=0.0)
    recovery = 1.0 / (1.0 - drawdown_depth.replace(1.0, np.nan)) - 1.0
    return recovery.where(drawdown_depth < 1.0, np.inf)


def drawdown_soft_penalty(
    max_drawdown: pd.Series | float,
    *,
    config: StrategyOutcomeUtilityConfig | None = None,
) -> pd.Series:
    cfg = config or StrategyOutcomeUtilityConfig()
    values = _as_series(max_drawdown).astype(float)
    depth = (-values.clip(upper=0.0)).clip(lower=0.0)
    soft = abs(cfg.soft_drawdown_limit)
    hard = abs(cfg.hard_drawdown_limit)
    band = max(hard - soft, 1e-12)
    return ((depth - soft) / band).clip(0.0, 1.0).where(values.notna())


def drawdown_hard_penalty(
    max_drawdown: pd.Series | float,
    *,
    config: StrategyOutcomeUtilityConfig | None = None,
) -> pd.Series:
    cfg = config or StrategyOutcomeUtilityConfig()
    values = _as_series(max_drawdown).astype(float)
    depth = (-values.clip(upper=0.0)).clip(lower=0.0)
    return (depth >= abs(cfg.hard_drawdown_limit)).astype(float).where(values.notna())


def add_outcome_frontier_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    if "cagr" not in output or "max_drawdown" not in output:
        output["is_growth_pareto_efficient"] = False
        return output
    cagr = pd.to_numeric(output["cagr"], errors="coerce")
    drawdown = pd.to_numeric(output["max_drawdown"], errors="coerce")
    efficient: list[bool] = []
    for idx in output.index:
        if pd.isna(cagr.loc[idx]) or pd.isna(drawdown.loc[idx]):
            efficient.append(False)
            continue
        dominated = (
            (cagr >= cagr.loc[idx])
            & (drawdown >= drawdown.loc[idx])
            & ((cagr > cagr.loc[idx]) | (drawdown > drawdown.loc[idx]))
        ).fillna(False)
        efficient.append(not bool(dominated.any()))
    output["is_growth_pareto_efficient"] = efficient
    return output


def _wealth_score(
    terminal_wealth_with_contributions: pd.Series,
    *,
    config: StrategyOutcomeUtilityConfig,
) -> pd.Series:
    floor_wealth = terminal_wealth_from_cagr(
        config.floor_cagr,
        years=config.horizon_years,
        starting_account_value=config.starting_account_value,
        annual_contribution=config.annual_contribution,
    ).iloc[0]
    target_wealth = terminal_wealth_from_cagr(
        config.target_cagr,
        years=config.horizon_years,
        starting_account_value=config.starting_account_value,
        annual_contribution=config.annual_contribution,
    ).iloc[0]
    denominator = max(float(np.log(target_wealth) - np.log(floor_wealth)), 1e-12)
    return (
        (np.log(terminal_wealth_with_contributions.clip(lower=1.0)) - np.log(floor_wealth))
        / denominator
    ).clip(
        0.0,
        1.0,
    )


def _validation_score(frame: pd.DataFrame, *, config: StrategyOutcomeUtilityConfig) -> pd.Series:
    index = frame.index
    walk = _numeric_series(frame, "walk_forward_positive_rate").fillna(
        config.min_walk_forward_positive_rate
    )
    worst_3y = _numeric_series(frame, "worst_3y_cagr").fillna(config.min_worst_3y_cagr)
    left_tail = _numeric_series(frame, "left_tail_regime_return").fillna(
        config.min_left_tail_regime_return
    )
    overfit = _numeric_series(frame, "overfit_risk_score").fillna(0.0).clip(0.0, 1.0)

    walk_penalty = ((config.min_walk_forward_positive_rate - walk) / 0.35).clip(0.0, 1.0)
    worst_3y_penalty = ((config.min_worst_3y_cagr - worst_3y) / 0.20).clip(0.0, 1.0)
    left_tail_penalty = ((config.min_left_tail_regime_return - left_tail) / 0.25).clip(0.0, 1.0)
    validation_penalty = (
        0.35 * walk_penalty
        + 0.25 * worst_3y_penalty
        + 0.25 * left_tail_penalty
        + config.overfit_penalty_weight * overfit
    )
    return pd.Series(1.0, index=index).sub(validation_penalty).clip(0.0, 1.0)


def _churn_penalty(frame: pd.DataFrame) -> pd.Series:
    labels = frame.get("operability_label", pd.Series("", index=frame.index)).fillna("").astype(str)
    label_penalty = labels.map(
        {
            "paper_operable": 0.0,
            "weekly_cadence": 0.05,
            "weekly_large_moves": 0.15,
            "review_churn": 0.35,
            "review_large_moves": 0.55,
            "too_twitchy": 1.0,
        }
    ).fillna(0.10)
    material_days = _numeric_series(frame, "material_trade_days_per_year")
    cadence_penalty = ((material_days - 60.0) / 80.0).clip(0.0, 1.0).fillna(0.0)
    return pd.concat([label_penalty, cadence_penalty], axis=1).max(axis=1)


def _growth_utility_tier(row: pd.Series) -> str:
    score = _optional_float(row.get("growth_constrained_utility_score")) or 0.0
    hard_penalty = _optional_float(row.get("drawdown_hard_penalty")) or 0.0
    soft_penalty = _optional_float(row.get("drawdown_soft_penalty")) or 0.0
    walk_rate = _optional_float(row.get("walk_forward_positive_rate"))
    if hard_penalty >= 1.0:
        return "growth_reject_hard_drawdown"
    if score >= 0.82 and soft_penalty <= 0.65 and (walk_rate is None or walk_rate >= 0.65):
        return "growth_champion_candidate"
    if score >= 0.70:
        return "growth_challenger_candidate"
    if score >= 0.55:
        return "growth_watchlist"
    return "growth_research_only"


def _benchmark_cagr(
    frame: pd.DataFrame,
    ticker: str,
    benchmark_metrics: pd.DataFrame | None,
) -> pd.Series:
    benchmark_name = f"benchmark_{ticker}"
    if benchmark_metrics is not None and benchmark_name in benchmark_metrics.index:
        return pd.Series(float(benchmark_metrics.loc[benchmark_name, "cagr"]), index=frame.index)
    excess_column = f"excess_cagr_vs_{ticker}"
    if excess_column in frame and "cagr" in frame:
        return _numeric_series(frame, "cagr") - _numeric_series(frame, excess_column)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _as_series(value: pd.Series | float) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series([cast(float, value)], dtype=float)


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(object, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
