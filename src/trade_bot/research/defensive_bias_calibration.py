from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import BotConfig
from trade_bot.features.indicators import daily_returns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_layer_calibration import (
    _point_in_time_scenario_state,
    frozen_weight_outcome,
)

DEFAULT_DEFENSIVE_BIAS_CALIBRATION_DIR = Path("reports/defensive_bias_calibration")
DEFAULT_DEFENSE_THRESHOLD = 0.60
DEFAULT_LOW_DEFENSE_THRESHOLD = 0.20
DEFAULT_SHIFT_CAP = 0.05
DEFAULT_MIN_HISTORY_DAYS = 756
DEFAULT_PRIMARY_HORIZON_DAYS = 21
DEFAULT_SECONDARY_HORIZON_DAYS = 63
DEFAULT_MINIMUM_EFFECT = 0.0005
DEFAULT_PRIOR_STRENGTHS = (24.0, 18.0, 12.0)

POLICY_ROSTER = (
    "base",
    "fixed_symmetric_5pp",
    "confirmation_fixed_symmetric_5pp",
    "hierarchical_confirmation_symmetric_5pp",
    "hierarchical_confirmation_defense_relief_5pp",
)

COMPARABLE_DYNAMIC_STRATEGIES = {
    "absolute_momentum_spy",
    "relative_momentum_sector",
    "dual_momentum_core",
    "vol_target_dual_momentum",
    "drawdown_managed_dual_momentum",
}

CRISIS_WINDOWS = (
    ("global_financial_crisis", "2007-10-01", "2009-06-30"),
    ("euro_debt_us_downgrade", "2011-04-01", "2011-12-30"),
    ("china_oil_growth_scare", "2015-06-01", "2016-03-31"),
    ("volatility_q4_liquidity", "2018-01-01", "2018-12-31"),
    ("covid_shock", "2020-02-01", "2020-06-30"),
    ("inflation_rates_growth_peak", "2021-03-01", "2022-12-30"),
    ("regional_bank_stress", "2023-02-01", "2023-06-30"),
    ("yen_carry_growth_scare", "2024-07-01", "2024-09-30"),
)

ERA_WINDOWS = (
    ("gfc_and_recovery", "2006-01-01", "2012-12-31"),
    ("post_gfc_expansion", "2013-01-01", "2017-12-31"),
    ("late_cycle_pandemic_inflation", "2018-01-01", "2022-12-31"),
    ("recent", "2023-01-01", "2099-12-31"),
)


@dataclass(frozen=True)
class DefensiveBiasCalibrationRun:
    origin_states: pd.DataFrame
    origin_outcomes: pd.DataFrame
    population_summary: pd.DataFrame
    online_estimates: pd.DataFrame
    strategy_metrics: pd.DataFrame
    era_metrics: pd.DataFrame
    crisis_holdouts: pd.DataFrame
    sensitivity: pd.DataFrame
    current_read: pd.DataFrame
    promotion_gates: pd.DataFrame
    output_paths: dict[str, Path]


def run_defensive_bias_calibration(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_DEFENSIVE_BIAS_CALIBRATION_DIR,
    defense_threshold: float = DEFAULT_DEFENSE_THRESHOLD,
    low_defense_threshold: float = DEFAULT_LOW_DEFENSE_THRESHOLD,
    shift_cap: float = DEFAULT_SHIFT_CAP,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    primary_horizon_days: int = DEFAULT_PRIMARY_HORIZON_DAYS,
    secondary_horizon_days: int = DEFAULT_SECONDARY_HORIZON_DAYS,
    defensive_ticker: str = "BIL",
) -> DefensiveBiasCalibrationRun:
    """Test a bounded, point-in-time correction for persistent allocation bias.

    This is deliberately separate from hard portfolio-risk constraints. It learns
    only from matured frozen-weight counterfactuals and never changes live policy.
    """

    prices = baseline_run.prices.sort_index()
    strategy_families = {
        name: family
        for name in baseline_run.results
        if (family := comparable_strategy_family(name)) is not None
    }
    if not strategy_families:
        raise ValueError("No comparable dynamic risk-managed strategies are available.")
    common_dates = prices.index
    for name in strategy_families:
        common_dates = common_dates.intersection(baseline_run.results[name].weights.index)
    origins = _month_end_origins(common_dates.sort_values(), min_history_days=min_history_days)
    if not origins:
        raise ValueError("No eligible month-end origins are available.")

    origin_states = build_origin_states(prices, origins)
    origin_outcomes = build_origin_outcomes(
        baseline_run,
        strategy_families,
        origins,
        defense_threshold=defense_threshold,
        low_defense_threshold=low_defense_threshold,
        shift_cap=shift_cap,
        horizons=(primary_horizon_days, secondary_horizon_days),
        defensive_ticker=defensive_ticker,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
    )
    population_summary = summarize_counterfactual_population(origin_outcomes)
    online_estimates = build_online_estimates(
        origin_outcomes,
        origin_states,
        strategy_families,
        origins,
        primary_horizon_days=primary_horizon_days,
    )
    adjusted_results = build_policy_results(
        baseline_run,
        strategy_families,
        origin_states,
        online_estimates,
        defense_threshold=defense_threshold,
        low_defense_threshold=low_defense_threshold,
        shift_cap=shift_cap,
        defensive_ticker=defensive_ticker,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
    )
    strategy_metrics = summarize_policy_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    era_metrics = summarize_era_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    crisis_holdouts = build_crisis_holdouts(
        baseline_run,
        strategy_families,
        origin_states,
        origin_outcomes,
        online_estimates,
        adjusted_results,
        defense_threshold=defense_threshold,
        low_defense_threshold=low_defense_threshold,
        shift_cap=shift_cap,
        defensive_ticker=defensive_ticker,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
        primary_horizon_days=primary_horizon_days,
    )
    sensitivity = build_sensitivity_summary(
        baseline_run,
        strategy_families,
        origin_states,
        online_estimates,
        defensive_ticker=defensive_ticker,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
    )
    current_read = build_current_read(
        baseline_run,
        strategy_families,
        origin_states,
        online_estimates,
        config.primary_strategy,
        defense_threshold=defense_threshold,
        low_defense_threshold=low_defense_threshold,
        shift_cap=shift_cap,
        defensive_ticker=defensive_ticker,
    )
    promotion_gates = build_promotion_gates(
        strategy_metrics,
        era_metrics,
        crisis_holdouts,
        focus_strategy=config.primary_strategy,
    )
    output_paths = write_defensive_bias_outputs(
        output_dir=output_dir,
        config=config,
        prices=prices,
        frames={
            "origin_states": origin_states,
            "origin_outcomes": origin_outcomes,
            "population_summary": population_summary,
            "online_estimates": online_estimates,
            "strategy_metrics": strategy_metrics,
            "era_metrics": era_metrics,
            "crisis_holdouts": crisis_holdouts,
            "sensitivity": sensitivity,
            "current_read": current_read,
            "promotion_gates": promotion_gates,
        },
        parameters={
            "population": "all_eligible_month_ends",
            "min_history_days": min_history_days,
            "defense_threshold": defense_threshold,
            "low_defense_threshold": low_defense_threshold,
            "shift_cap": shift_cap,
            "primary_horizon_days": primary_horizon_days,
            "secondary_horizon_days": secondary_horizon_days,
            "utility": {
                "return_weight": 1.0,
                "drawdown_improvement_weight": 0.75,
                "drawdown_deterioration_weight": 1.50,
            },
            "prior_strengths": {
                "global": DEFAULT_PRIOR_STRENGTHS[0],
                "family": DEFAULT_PRIOR_STRENGTHS[1],
                "strategy": DEFAULT_PRIOR_STRENGTHS[2],
            },
            "minimum_effect": DEFAULT_MINIMUM_EFFECT,
            "trial_roster": list(POLICY_ROSTER),
            "hard_portfolio_constraints": "excluded_not_bias_correctable",
            "automatic_allocation_authority": 0.0,
        },
    )
    return DefensiveBiasCalibrationRun(
        origin_states=origin_states,
        origin_outcomes=origin_outcomes,
        population_summary=population_summary,
        online_estimates=online_estimates,
        strategy_metrics=strategy_metrics,
        era_metrics=era_metrics,
        crisis_holdouts=crisis_holdouts,
        sensitivity=sensitivity,
        current_read=current_read,
        promotion_gates=promotion_gates,
        output_paths=output_paths,
    )


def comparable_strategy_family(strategy: str) -> str | None:
    normalized = str(strategy).lower()
    if normalized.startswith("i111"):
        return "i111"
    if normalized in COMPARABLE_DYNAMIC_STRATEGIES:
        return "dynamic_risk_managed"
    return None


def bounded_sleeve_shift(
    weights: pd.Series,
    action: str,
    *,
    cap: float,
    defensive_ticker: str = "BIL",
) -> tuple[pd.Series, float]:
    """Move a bounded amount without inventing a risky holding."""

    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if clean.sum() > 1.0 + 1e-10:
        clean = clean / float(clean.sum())
    if defensive_ticker not in clean.index:
        clean.loc[defensive_ticker] = 0.0
    residual = max(0.0, 1.0 - float(clean.sum()))
    risk_columns = [
        column
        for column in clean.index
        if column != defensive_ticker and float(clean.loc[column]) > 1e-12
    ]
    if not risk_columns:
        return clean, 0.0
    risk_weight = float(clean[risk_columns].sum())
    defensive_weight = float(clean.loc[defensive_ticker])
    if action == "defense_relief":
        shift = min(float(cap), defensive_weight + residual)
        if shift <= 0.0 or risk_weight <= 0.0:
            return clean, 0.0
        bil_reduction = min(defensive_weight, shift)
        clean.loc[defensive_ticker] = defensive_weight - bil_reduction
        clean.loc[risk_columns] = clean[risk_columns] * ((risk_weight + shift) / risk_weight)
    elif action == "risk_restraint":
        shift = min(float(cap), risk_weight)
        if shift <= 0.0:
            return clean, 0.0
        clean.loc[risk_columns] = clean[risk_columns] * ((risk_weight - shift) / risk_weight)
        clean.loc[defensive_ticker] = defensive_weight + shift
    else:
        return clean, 0.0
    return clean, float(shift)


def defensive_weight(weights: pd.Series, defensive_ticker: str = "BIL") -> float:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    explicit = float(clean.get(defensive_ticker, 0.0))
    return float(np.clip(explicit + max(0.0, 1.0 - float(clean.sum())), 0.0, 1.0))


def build_origin_states(prices: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin in origins:
        risk_score, risk_status, _lattice, timing = _point_in_time_scenario_state(
            prices.loc[:origin]
        )
        rows.append(
            {
                "origin_date": origin,
                "risk_score": risk_score,
                "risk_status": risk_status,
                "risk_timing_state": timing.state,
                "confirmation_break_count": len(timing.confirmation_breaks),
                "confirmation_breaks": ", ".join(timing.confirmation_breaks),
                "defense_relief_allowed": timing.state
                not in {"confirmed_break", "severe_break"},
            }
        )
    return pd.DataFrame(rows)


def build_origin_outcomes(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origins: list[pd.Timestamp],
    *,
    defense_threshold: float,
    low_defense_threshold: float,
    shift_cap: float,
    horizons: tuple[int, ...],
    defensive_ticker: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    prices = baseline_run.prices.sort_index()
    positions = {pd.Timestamp(date): position for position, date in enumerate(prices.index)}
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        result = baseline_run.results[strategy]
        for origin in origins:
            base = result.weights.loc[:origin].iloc[-1].astype(float)
            defense = defensive_weight(base, defensive_ticker)
            action = (
                "defense_relief"
                if defense >= defense_threshold
                else ("risk_restraint" if defense <= low_defense_threshold else "none")
            )
            if action == "none":
                continue
            candidate, applied_shift = bounded_sleeve_shift(
                base,
                action,
                cap=shift_cap,
                defensive_ticker=defensive_ticker,
            )
            if applied_shift <= 0.0:
                continue
            universe = base.index.union(candidate.index)
            turnover = float(
                candidate.reindex(universe, fill_value=0.0)
                .sub(base.reindex(universe, fill_value=0.0))
                .abs()
                .sum()
            )
            initial_cost = turnover * transaction_cost_bps / 10_000.0
            position = positions[pd.Timestamp(origin)]
            for horizon_days in horizons:
                base_return, base_drawdown = frozen_weight_outcome(
                    prices,
                    base,
                    position,
                    horizon_days,
                    initial_cost=0.0,
                )
                candidate_return, candidate_drawdown = frozen_weight_outcome(
                    prices,
                    candidate,
                    position,
                    horizon_days,
                    initial_cost=initial_cost,
                )
                if any(
                    pd.isna(value)
                    for value in (
                        base_return,
                        base_drawdown,
                        candidate_return,
                        candidate_drawdown,
                    )
                ):
                    continue
                maturity_position = position + horizon_days
                rows.append(
                    {
                        "origin_date": origin,
                        "maturity_date": prices.index[maturity_position],
                        "strategy": strategy,
                        "family": family,
                        "action": action,
                        "horizon_days": horizon_days,
                        "base_defensive_weight": defense,
                        "applied_shift": applied_shift,
                        "incremental_turnover": turnover,
                        "incremental_cost": initial_cost,
                        "base_forward_return": float(base_return),
                        "candidate_forward_return": float(candidate_return),
                        "return_delta": float(candidate_return) - float(base_return),
                        "base_forward_max_drawdown": float(base_drawdown),
                        "candidate_forward_max_drawdown": float(candidate_drawdown),
                        "drawdown_delta": float(candidate_drawdown) - float(base_drawdown),
                        "utility_delta": counterfactual_utility_delta(
                            float(candidate_return) - float(base_return),
                            float(candidate_drawdown) - float(base_drawdown),
                        ),
                        "evaluation_window": _window_label(origin),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["origin_date", "strategy", "horizon_days"]
    ).reset_index(drop=True)


def counterfactual_utility_delta(return_delta: float, drawdown_delta: float) -> float:
    drawdown_weight = 0.75 if drawdown_delta >= 0.0 else 1.50
    return float(return_delta + drawdown_weight * drawdown_delta)


def summarize_counterfactual_population(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    data = outcomes.copy()
    data["market_type"] = data["evaluation_window"].where(
        data["evaluation_window"].eq("ordinary_market"),
        "named_stress_window",
    )
    rows: list[dict[str, object]] = []
    for (market_type, action, horizon), group in data.groupby(
        ["market_type", "action", "horizon_days"],
        sort=False,
    ):
        rows.append(
            {
                "market_type": market_type,
                "action": action,
                "horizon_days": horizon,
                "rows": len(group),
                "unique_origins": group["origin_date"].nunique(),
                "strategies": group["strategy"].nunique(),
                "mean_utility_delta": group["utility_delta"].mean(),
                "median_utility_delta": group["utility_delta"].median(),
                "positive_utility_rate": group["utility_delta"].gt(0.0).mean(),
                "mean_return_delta": group["return_delta"].mean(),
                "mean_drawdown_delta": group["drawdown_delta"].mean(),
            }
        )
    return pd.DataFrame(rows)


def hierarchical_posterior(
    history: pd.DataFrame,
    *,
    strategy: str,
    family: str,
    action: str,
    origin: pd.Timestamp,
    excluded_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> dict[str, object]:
    eligible = history[
        history["action"].eq(action)
        & pd.to_datetime(history["maturity_date"]).lt(pd.Timestamp(origin))
    ].copy()
    if excluded_window is not None:
        start, end = excluded_window
        dates = pd.to_datetime(eligible["origin_date"])
        eligible = eligible[~dates.between(start, end)]
    if eligible.empty:
        return _empty_posterior()

    origin_global = eligible.groupby("origin_date", as_index=False)["utility_delta"].mean()
    global_n = int(len(origin_global))
    global_raw = float(origin_global["utility_delta"].mean())
    global_strength, family_strength, strategy_strength = DEFAULT_PRIOR_STRENGTHS
    global_mean = global_n / (global_n + global_strength) * global_raw

    family_rows = eligible[eligible["family"].eq(family)]
    family_by_origin = family_rows.groupby("origin_date", as_index=False)["utility_delta"].mean()
    family_n = int(len(family_by_origin))
    family_raw = (
        float(family_by_origin["utility_delta"].mean()) if family_n else global_mean
    )
    family_mean = (
        family_n * family_raw + family_strength * global_mean
    ) / (family_n + family_strength)

    strategy_rows = eligible[eligible["strategy"].eq(strategy)]
    strategy_n = int(len(strategy_rows))
    strategy_raw = (
        float(strategy_rows["utility_delta"].mean()) if strategy_n else family_mean
    )
    strategy_mean = (
        strategy_n * strategy_raw + strategy_strength * family_mean
    ) / (strategy_n + strategy_strength)
    eligible_action = (
        global_n >= 24
        and family_n >= 12
        and strategy_n >= 8
        and min(global_mean, family_mean, strategy_mean) > DEFAULT_MINIMUM_EFFECT
    )
    return {
        "global_origins": global_n,
        "family_origins": family_n,
        "strategy_observations": strategy_n,
        "global_posterior_mean": global_mean,
        "family_posterior_mean": family_mean,
        "strategy_posterior_mean": strategy_mean,
        "minimum_posterior_mean": min(global_mean, family_mean, strategy_mean),
        "hierarchical_action_eligible": bool(eligible_action),
    }


def build_online_estimates(
    outcomes: pd.DataFrame,
    origin_states: pd.DataFrame,
    strategy_families: dict[str, str],
    origins: list[pd.Timestamp],
    *,
    primary_horizon_days: int,
    excluded_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    history = outcomes[outcomes["horizon_days"].eq(primary_horizon_days)].copy()
    state_by_origin = origin_states.set_index("origin_date")
    rows: list[dict[str, object]] = []
    for origin in origins:
        state = state_by_origin.loc[origin]
        for strategy, family in strategy_families.items():
            for action in ("defense_relief", "risk_restraint"):
                posterior = hierarchical_posterior(
                    history,
                    strategy=strategy,
                    family=family,
                    action=action,
                    origin=origin,
                    excluded_window=excluded_window,
                )
                confirmation_allowed = bool(
                    action != "defense_relief" or state["defense_relief_allowed"]
                )
                rows.append(
                    {
                        "origin_date": origin,
                        "strategy": strategy,
                        "family": family,
                        "action": action,
                        "risk_score": state["risk_score"],
                        "risk_status": state["risk_status"],
                        "risk_timing_state": state["risk_timing_state"],
                        "confirmation_allowed": confirmation_allowed,
                        **posterior,
                        "action_allowed": bool(
                            confirmation_allowed
                            and posterior["hierarchical_action_eligible"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_policy_results(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origin_states: pd.DataFrame,
    online_estimates: pd.DataFrame,
    *,
    defense_threshold: float,
    low_defense_threshold: float,
    shift_cap: float,
    defensive_ticker: str,
    transaction_cost_bps: float,
) -> dict[tuple[str, str], BacktestResult]:
    results: dict[tuple[str, str], BacktestResult] = {}
    for strategy in strategy_families:
        base = baseline_run.results[strategy]
        for policy in POLICY_ROSTER[1:]:
            adjusted = build_adjusted_weight_path(
                base.weights,
                origin_states,
                online_estimates[online_estimates["strategy"].eq(strategy)],
                policy=policy,
                defense_threshold=defense_threshold,
                low_defense_threshold=low_defense_threshold,
                shift_cap=shift_cap,
                defensive_ticker=defensive_ticker,
            )
            results[(strategy, policy)] = _result_from_execution_weights(
                base,
                baseline_run.prices,
                adjusted,
                transaction_cost_bps=transaction_cost_bps,
                name=f"{strategy}__{policy}",
            )
    return results


def build_adjusted_weight_path(
    base_weights: pd.DataFrame,
    origin_states: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    policy: str,
    defense_threshold: float,
    low_defense_threshold: float,
    shift_cap: float,
    defensive_ticker: str,
) -> pd.DataFrame:
    adjusted = base_weights.copy().astype(float).clip(lower=0.0)
    if adjusted.empty or origin_states.empty:
        return adjusted
    if defensive_ticker not in adjusted:
        adjusted[defensive_ticker] = 0.0
    row_sums = adjusted.sum(axis=1)
    over_allocated = row_sums.gt(1.0 + 1e-10)
    if over_allocated.any():
        adjusted.loc[over_allocated] = adjusted.loc[over_allocated].div(
            row_sums.loc[over_allocated],
            axis=0,
        )
    risk_columns = [
        column for column in adjusted.columns if column != defensive_ticker
    ]
    if not risk_columns:
        return adjusted
    state_lookup = origin_states.set_index("origin_date")
    estimate_lookup = estimates.set_index(["origin_date", "action"])
    origins = [pd.Timestamp(value) for value in origin_states["origin_date"]]
    for number, origin in enumerate(origins):
        end = origins[number + 1] if number + 1 < len(origins) else adjusted.index[-1]
        mask = adjusted.index.to_series().ge(origin) & adjusted.index.to_series().lt(end)
        if number + 1 == len(origins):
            mask = adjusted.index.to_series().ge(origin)
        block_index = adjusted.index[mask.to_numpy()]
        if block_index.empty:
            continue
        state = state_lookup.loc[origin]
        block = adjusted.loc[block_index]
        residual = (1.0 - block.sum(axis=1)).clip(lower=0.0)
        defense = block[defensive_ticker] + residual
        risk = block[risk_columns].sum(axis=1)
        if _policy_allows_action(
            policy,
            "defense_relief",
            origin,
            state,
            estimate_lookup,
        ):
            relief = pd.Series(
                np.minimum(float(shift_cap), defense),
                index=block_index,
            ).where(defense.ge(defense_threshold) & risk.gt(1e-12), 0.0)
            relief_scale = ((risk + relief) / risk.where(risk.gt(1e-12))).fillna(1.0)
            adjusted.loc[block_index, risk_columns] = block[risk_columns].mul(
                relief_scale,
                axis=0,
            )
            bil_reduction = pd.concat(
                [block[defensive_ticker], relief],
                axis=1,
            ).min(axis=1)
            adjusted.loc[block_index, defensive_ticker] = (
                block[defensive_ticker] - bil_reduction
            )
        if _policy_allows_action(
            policy,
            "risk_restraint",
            origin,
            state,
            estimate_lookup,
        ):
            block = adjusted.loc[block_index]
            residual = (1.0 - block.sum(axis=1)).clip(lower=0.0)
            defense = block[defensive_ticker] + residual
            risk = block[risk_columns].sum(axis=1)
            restraint = pd.Series(
                np.minimum(float(shift_cap), risk),
                index=block_index,
            ).where(defense.le(low_defense_threshold) & risk.gt(1e-12), 0.0)
            restraint_scale = ((risk - restraint) / risk.where(risk.gt(1e-12))).fillna(
                1.0
            )
            adjusted.loc[block_index, risk_columns] = block[risk_columns].mul(
                restraint_scale,
                axis=0,
            )
            adjusted.loc[block_index, defensive_ticker] = defense + restraint
    return adjusted


def _policy_allows_action(
    policy: str,
    action: str,
    origin: pd.Timestamp,
    state: pd.Series,
    estimate_lookup: pd.DataFrame,
) -> bool:
    if policy == "fixed_symmetric_5pp":
        return True
    if action == "defense_relief" and not bool(state["defense_relief_allowed"]):
        return False
    if policy == "confirmation_fixed_symmetric_5pp":
        return True
    if (
        policy == "hierarchical_confirmation_defense_relief_5pp"
        and action != "defense_relief"
    ):
        return False
    if policy.startswith("hierarchical_"):
        key = (origin, action)
        if key not in estimate_lookup.index:
            return False
        value = estimate_lookup.loc[key, "action_allowed"]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return bool(value)
    return False


def summarize_policy_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        policy_results = {"base": base} | {
            policy: adjusted_results[(strategy, policy)]
            for policy in POLICY_ROSTER[1:]
        }
        for policy, result in policy_results.items():
            metric = calculate_metrics(
                result.name,
                result.returns,
                result.equity,
                result.turnover,
                result.transaction_costs,
                historical_evidence_basis="point_in_time_full_history_overlay",
            )
            base_weights = base.weights.reindex(result.weights.index).fillna(0.0)
            aligned = result.weights.reindex(columns=base_weights.columns, fill_value=0.0)
            adjustment = aligned.sub(base_weights).abs().sum(axis=1) / 2.0
            rows.append(
                {
                    "strategy": strategy,
                    "family": family,
                    "policy": policy,
                    **metric.__dict__,
                    "active_day_rate": float(adjustment.gt(1e-10).mean()),
                    "mean_absolute_sleeve_shift": float(adjustment.mean()),
                    "max_absolute_sleeve_shift": float(adjustment.max()),
                }
            )
    frame = pd.DataFrame(rows)
    base = frame[frame["policy"].eq("base")].set_index("strategy")
    frame["cagr_delta_vs_base"] = frame.apply(
        lambda row: float(row["cagr"]) - float(base.loc[row["strategy"], "cagr"]),
        axis=1,
    )
    frame["max_drawdown_delta_vs_base"] = frame.apply(
        lambda row: float(row["max_drawdown"])
        - float(base.loc[row["strategy"], "max_drawdown"]),
        axis=1,
    )
    frame["sharpe_delta_vs_base"] = frame.apply(
        lambda row: float(row["sharpe"]) - float(base.loc[row["strategy"], "sharpe"]),
        axis=1,
    )
    return frame


def summarize_era_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        for policy in POLICY_ROSTER[1:]:
            candidate = adjusted_results[(strategy, policy)]
            for era, start, end in ERA_WINDOWS:
                base_stats = _return_slice_stats(base.returns, start, end)
                candidate_stats = _return_slice_stats(candidate.returns, start, end)
                if base_stats["observations"] == 0:
                    continue
                rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "era": era,
                        **{f"base_{key}": value for key, value in base_stats.items()},
                        **{
                            f"candidate_{key}": value
                            for key, value in candidate_stats.items()
                        },
                        "annualized_return_delta": candidate_stats["annualized_return"]
                        - base_stats["annualized_return"],
                        "max_drawdown_delta": candidate_stats["max_drawdown"]
                        - base_stats["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows)


def build_crisis_holdouts(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origin_states: pd.DataFrame,
    outcomes: pd.DataFrame,
    online_estimates: pd.DataFrame,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    *,
    defense_threshold: float,
    low_defense_threshold: float,
    shift_cap: float,
    defensive_ticker: str,
    transaction_cost_bps: float,
    primary_horizon_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origins = [pd.Timestamp(value) for value in origin_states["origin_date"]]
    for crisis, start_text, end_text in CRISIS_WINDOWS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        scoped_origins = [origin for origin in origins if start <= origin <= end]
        if not scoped_origins:
            continue
        excluded_estimates = build_online_estimates(
            outcomes,
            origin_states,
            strategy_families,
            scoped_origins,
            primary_horizon_days=primary_horizon_days,
            excluded_window=(start, end),
        )
        estimates = pd.concat(
            [
                online_estimates[
                    ~online_estimates["origin_date"].isin(scoped_origins)
                ],
                excluded_estimates,
            ],
            ignore_index=True,
        )
        for strategy, family in strategy_families.items():
            base = baseline_run.results[strategy]
            for policy in POLICY_ROSTER[1:]:
                if policy.startswith("hierarchical_"):
                    adjusted_weights = build_adjusted_weight_path(
                        base.weights,
                        origin_states,
                        estimates[estimates["strategy"].eq(strategy)],
                        policy=policy,
                        defense_threshold=defense_threshold,
                        low_defense_threshold=low_defense_threshold,
                        shift_cap=shift_cap,
                        defensive_ticker=defensive_ticker,
                    )
                    candidate = _result_from_execution_weights(
                        base,
                        baseline_run.prices,
                        adjusted_weights,
                        transaction_cost_bps=transaction_cost_bps,
                        name=f"{strategy}__loo__{crisis}__{policy}",
                    )
                    training_exclusion = "all_origins_in_test_window"
                else:
                    candidate = adjusted_results[(strategy, policy)]
                    training_exclusion = "not_applicable_fixed_policy"
                base_stats = _return_slice_stats(base.returns, start_text, end_text)
                candidate_stats = _return_slice_stats(
                    candidate.returns, start_text, end_text
                )
                if base_stats["observations"] == 0:
                    continue
                rows.append(
                    {
                        "crisis": crisis,
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "test_start": start,
                        "test_end": end,
                        "training_exclusion": training_exclusion,
                        "test_origins": len(scoped_origins),
                        "base_cumulative_return": base_stats["cumulative_return"],
                        "candidate_cumulative_return": candidate_stats[
                            "cumulative_return"
                        ],
                        "return_delta": candidate_stats["cumulative_return"]
                        - base_stats["cumulative_return"],
                        "base_max_drawdown": base_stats["max_drawdown"],
                        "candidate_max_drawdown": candidate_stats["max_drawdown"],
                        "max_drawdown_delta": candidate_stats["max_drawdown"]
                        - base_stats["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows)


def build_sensitivity_summary(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origin_states: pd.DataFrame,
    online_estimates: pd.DataFrame,
    *,
    defensive_ticker: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    relief_policy = "hierarchical_confirmation_defense_relief_5pp"
    symmetric_policy = "hierarchical_confirmation_symmetric_5pp"
    settings = {
        (relief_policy, 0.025, 0.60, 0.20),
        (relief_policy, 0.050, 0.60, 0.20),
        (relief_policy, 0.075, 0.60, 0.20),
        (relief_policy, 0.050, 0.55, 0.20),
        (relief_policy, 0.050, 0.65, 0.20),
        (symmetric_policy, 0.050, 0.60, 0.15),
        (symmetric_policy, 0.050, 0.60, 0.20),
        (symmetric_policy, 0.050, 0.60, 0.25),
    }
    rows: list[dict[str, object]] = []
    for policy, cap, defense_threshold, low_threshold in sorted(settings):
        strategy_deltas: list[float] = []
        drawdown_deltas: list[float] = []
        for strategy, family in strategy_families.items():
            base = baseline_run.results[strategy]
            weights = build_adjusted_weight_path(
                base.weights,
                origin_states,
                online_estimates[online_estimates["strategy"].eq(strategy)],
                policy=policy,
                defense_threshold=defense_threshold,
                low_defense_threshold=low_threshold,
                shift_cap=cap,
                defensive_ticker=defensive_ticker,
            )
            result = _result_from_execution_weights(
                base,
                baseline_run.prices,
                weights,
                transaction_cost_bps=transaction_cost_bps,
                name=f"{strategy}__sensitivity",
            )
            base_metric = calculate_metrics(
                base.name,
                base.returns,
                base.equity,
                base.turnover,
                base.transaction_costs,
            )
            candidate_metric = calculate_metrics(
                result.name,
                result.returns,
                result.equity,
                result.turnover,
                result.transaction_costs,
            )
            cagr_delta = candidate_metric.cagr - base_metric.cagr
            drawdown_delta = candidate_metric.max_drawdown - base_metric.max_drawdown
            strategy_deltas.append(cagr_delta)
            drawdown_deltas.append(drawdown_delta)
            rows.append(
                {
                    "policy": policy,
                    "shift_cap": cap,
                    "defense_threshold": defense_threshold,
                    "low_defense_threshold": low_threshold,
                    "strategy": strategy,
                    "family": family,
                    "cagr_delta_vs_base": cagr_delta,
                    "max_drawdown_delta_vs_base": drawdown_delta,
                }
            )
        rows.append(
            {
                "policy": policy,
                "shift_cap": cap,
                "defense_threshold": defense_threshold,
                "low_defense_threshold": low_threshold,
                "strategy": "__cross_strategy_summary__",
                "family": "all",
                "cagr_delta_vs_base": float(np.median(strategy_deltas)),
                "max_drawdown_delta_vs_base": float(np.median(drawdown_deltas)),
                "positive_cagr_rate": float(np.mean(np.asarray(strategy_deltas) > 0.0)),
                "nonworse_drawdown_rate": float(
                    np.mean(np.asarray(drawdown_deltas) >= 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_current_read(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origin_states: pd.DataFrame,
    estimates: pd.DataFrame,
    focus_strategy: str,
    *,
    defense_threshold: float,
    low_defense_threshold: float,
    shift_cap: float,
    defensive_ticker: str,
) -> pd.DataFrame:
    if focus_strategy not in strategy_families:
        return pd.DataFrame()
    latest_date = baseline_run.results[focus_strategy].weights.index[-1]
    weights = baseline_run.results[focus_strategy].weights.iloc[-1].astype(float)
    defense = defensive_weight(weights, defensive_ticker)
    action = (
        "defense_relief"
        if defense >= defense_threshold
        else ("risk_restraint" if defense <= low_defense_threshold else "none")
    )
    evidence_action = (
        action
        if action != "none"
        else (
            "defense_relief"
            if abs(defense - defense_threshold)
            <= abs(defense - low_defense_threshold)
            else "risk_restraint"
        )
    )
    latest_origin = pd.Timestamp(origin_states["origin_date"].max())
    state = origin_states.set_index("origin_date").loc[latest_origin]
    estimate = pd.Series(dtype=object)
    selected = estimates[
        estimates["strategy"].eq(focus_strategy)
        & estimates["origin_date"].eq(latest_origin)
        & estimates["action"].eq(evidence_action)
    ]
    if not selected.empty:
        estimate = selected.iloc[-1]
    allowed = bool(action != "none" and estimate.get("action_allowed", False))
    candidate, applied = (
        bounded_sleeve_shift(
            weights,
            action,
            cap=shift_cap,
            defensive_ticker=defensive_ticker,
        )
        if allowed
        else (weights.copy(), 0.0)
    )
    return pd.DataFrame(
        [
            {
                "market_date": latest_date,
                "evidence_origin": latest_origin,
                "strategy": focus_strategy,
                "family": strategy_families[focus_strategy],
                "current_defensive_weight": defense,
                "current_risk_weight": 1.0 - defense,
                "triggered_action": action,
                "nearest_evidence_action": evidence_action,
                "confirmation_state": state["risk_timing_state"],
                "confirmation_allowed": bool(
                    action != "defense_relief" or state["defense_relief_allowed"]
                ),
                "hierarchical_action_eligible": bool(
                    estimate.get("hierarchical_action_eligible", False)
                ),
                "research_action_allowed": allowed,
                "research_only_shift": applied,
                "research_adjusted_defensive_weight": defensive_weight(
                    candidate, defensive_ticker
                ),
                "global_origins": estimate.get("global_origins", 0),
                "family_origins": estimate.get("family_origins", 0),
                "strategy_observations": estimate.get("strategy_observations", 0),
                "global_posterior_mean": estimate.get(
                    "global_posterior_mean", np.nan
                ),
                "family_posterior_mean": estimate.get(
                    "family_posterior_mean", np.nan
                ),
                "strategy_posterior_mean": estimate.get(
                    "strategy_posterior_mean", np.nan
                ),
                "allocation_authority": 0.0,
                "status": "retrospective_research_only",
            }
        ]
    )


def build_promotion_gates(
    strategy_metrics: pd.DataFrame,
    era_metrics: pd.DataFrame,
    crisis_holdouts: pd.DataFrame,
    *,
    focus_strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in POLICY_ROSTER[1:]:
        candidate = strategy_metrics[strategy_metrics["policy"].eq(policy)]
        focus = candidate[candidate["strategy"].eq(focus_strategy)]
        focus_cagr = (
            float(focus["cagr_delta_vs_base"].iloc[0]) if not focus.empty else np.nan
        )
        focus_dd = (
            float(focus["max_drawdown_delta_vs_base"].iloc[0])
            if not focus.empty
            else np.nan
        )
        strategy_positive = float(candidate["cagr_delta_vs_base"].gt(0.0).mean())
        strategy_dd = float(candidate["max_drawdown_delta_vs_base"].ge(0.0).mean())
        policy_eras = era_metrics[era_metrics["policy"].eq(policy)]
        era_positive = (
            float(
                policy_eras.groupby("era")["annualized_return_delta"]
                .median()
                .gt(0.0)
                .mean()
            )
            if not policy_eras.empty
            else 0.0
        )
        policy_crises = crisis_holdouts[crisis_holdouts["policy"].eq(policy)]
        crisis_nonworse = (
            float(policy_crises["max_drawdown_delta"].ge(0.0).mean())
            if not policy_crises.empty
            else 0.0
        )
        checks = (
            ("focus_cagr_nonnegative", focus_cagr >= 0.0, focus_cagr, 0.0),
            ("focus_drawdown_not_worse_by_1pp", focus_dd >= -0.01, focus_dd, -0.01),
            (
                "strategy_positive_cagr_rate",
                strategy_positive >= 0.60,
                strategy_positive,
                0.60,
            ),
            (
                "strategy_nonworse_drawdown_rate",
                strategy_dd >= 0.60,
                strategy_dd,
                0.60,
            ),
            ("era_positive_rate", era_positive >= 0.60, era_positive, 0.60),
            (
                "crisis_holdout_nonworse_drawdown_rate",
                crisis_nonworse >= 0.80,
                crisis_nonworse,
                0.80,
            ),
            ("prospective_shadow_evidence", False, 0.0, 1.0),
        )
        policy_rows = [
            {
                "policy": policy,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
            for gate, passed, observed, required in checks
        ]
        retrospective = all(row["passed"] for row in policy_rows[:-1])
        for row in policy_rows:
            row["retrospective_gate_passed"] = retrospective
            row["promotion_allowed"] = False
            row["authority"] = 0.0
        rows.extend(policy_rows)
    return pd.DataFrame(rows)


def write_defensive_bias_outputs(
    *,
    output_dir: str | Path,
    config: BotConfig,
    prices: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    parameters: dict[str, object],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    summary_path = root / "summary.md"
    summary_path.write_text(_build_summary(frames) + "\n", encoding="utf-8")
    paths["summary"] = summary_path
    artifacts = [path.name for path in paths.values()]
    manifest = write_research_manifest(
        root,
        study="defensive_bias_calibration",
        config=config,
        prices=prices,
        parameters=parameters,
        artifacts=artifacts,
    )
    paths["manifest"] = manifest
    return paths


def _build_summary(frames: dict[str, pd.DataFrame]) -> str:
    metrics = frames["strategy_metrics"]
    current = frames["current_read"]
    gates = frames["promotion_gates"]
    policy = "hierarchical_confirmation_defense_relief_5pp"
    candidate = metrics[metrics["policy"].eq(policy)]
    lines = [
        "# Defensive Bias Calibration",
        "",
        "Status: retrospective research only; allocation authority remains 0%.",
        "",
        "## Broad result",
        "",
        f"- Strategies tested: {candidate['strategy'].nunique():,}.",
        f"- Eligible month-end counterfactuals: {len(frames['origin_outcomes']):,}.",
        f"- Positive full-history CAGR delta: {candidate['cagr_delta_vs_base'].gt(0).mean():.1%} of strategies.",
        f"- Non-worse full-history drawdown: {candidate['max_drawdown_delta_vs_base'].ge(0).mean():.1%} of strategies.",
    ]
    if not current.empty:
        row = current.iloc[0]
        lines.extend(
            [
                "",
                "## Current research read",
                "",
                f"- Current defense: {float(row['current_defensive_weight']):.2%}.",
                f"- Triggered action: `{row['triggered_action']}`.",
                f"- Research action allowed: {bool(row['research_action_allowed'])}.",
                f"- Hypothetical bounded shift: {float(row['research_only_shift']):.2%}.",
                "- Live allocation effect: none.",
            ]
        )
    primary_policy = "hierarchical_confirmation_defense_relief_5pp"
    primary_gates = gates[gates["policy"].eq(primary_policy)]
    failed = primary_gates[~primary_gates["passed"].astype(bool)]["gate"].astype(str).tolist()
    lines.extend(
        [
            "",
            "## Promotion decision",
            "",
            "- Automatic promotion is prohibited.",
            f"- Failed gates: {', '.join(failed) if failed else 'none retrospective; prospective evidence still required'}.",
            "- Hard portfolio-risk constraints were not bias-corrected.",
        ]
    )
    return "\n".join(lines)


def _result_from_execution_weights(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    name: str,
) -> BacktestResult:
    aligned_prices = prices.reindex(weights.index)
    asset_returns = daily_returns(aligned_prices).reindex(
        index=weights.index, columns=weights.columns
    ).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    transaction_costs = turnover * transaction_cost_bps / 10_000.0
    gross_returns = (weights * asset_returns).sum(axis=1)
    returns = gross_returns - transaction_costs
    first_growth = 1.0 + float(base_result.returns.iloc[0])
    initial = (
        float(base_result.equity.iloc[0]) / first_growth
        if abs(first_growth) > 1e-12
        else float(base_result.equity.iloc[0])
    )
    equity = initial * (1.0 + returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=returns.rename(name),
        gross_returns=gross_returns.rename(name),
        weights=weights,
        target_weights=weights,
        turnover=turnover.rename(name),
        transaction_costs=transaction_costs.rename(name),
    )


def _month_end_origins(
    dates: pd.DatetimeIndex,
    *,
    min_history_days: int,
) -> list[pd.Timestamp]:
    eligible = pd.DatetimeIndex(dates[min_history_days:])
    if eligible.empty:
        return []
    frame = pd.Series(eligible, index=eligible)
    return [
        pd.Timestamp(value)
        for value in frame.groupby(eligible.to_period("M")).max().tolist()
    ]


def _empty_posterior() -> dict[str, object]:
    return {
        "global_origins": 0,
        "family_origins": 0,
        "strategy_observations": 0,
        "global_posterior_mean": 0.0,
        "family_posterior_mean": 0.0,
        "strategy_posterior_mean": 0.0,
        "minimum_posterior_mean": 0.0,
        "hierarchical_action_eligible": False,
    }


def _window_label(date: pd.Timestamp) -> str:
    point = pd.Timestamp(date)
    for label, start, end in CRISIS_WINDOWS:
        if pd.Timestamp(start) <= point <= pd.Timestamp(end):
            return label
    return "ordinary_market"


def _return_slice_stats(
    returns: pd.Series,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, float | int]:
    selected = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if selected.empty:
        return {
            "observations": 0,
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
        }
    equity = (1.0 + selected).cumprod()
    annualized = float(equity.iloc[-1] ** (252.0 / len(selected)) - 1.0)
    return {
        "observations": int(len(selected)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annualized,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }
