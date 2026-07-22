from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_bot.config import BotConfig
from trade_bot.features.indicators import bounded_forward_fill
from trade_bot.portfolio.risk import PortfolioRiskConfig, build_portfolio_risk
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import (
    _risk_status,
    build_confirmation_matrix,
    build_market_health,
    calculate_risk_score,
    momentum_state_table,
)
from trade_bot.research.defensive_judgement import (
    DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS,
    DefensiveJudgementHorizon,
    classify_defensive_judgement,
)
from trade_bot.research.future_scenarios import build_scenario_lattice
from trade_bot.research.operating_history import _sample_history_dates
from trade_bot.research.risk_timing import RiskTimingAssessment, assess_risk_timing
from trade_bot.research.trade_decision import (
    _risk_status_multiplier,
    _scenario_adjusted_weights,
    _scenario_context,
    _weights_with_defensive_residual,
)

DEFAULT_DEFENSIVE_LAYER_CALIBRATION_DIR = Path("reports/defensive_layer_calibration")
DEFAULT_BASE_DEFENSIVE_THRESHOLD = 0.55
DEFAULT_SCENARIO_DEFENSIVE_ADD_THRESHOLD = 0.05
DEFAULT_PORTFOLIO_DEFENSIVE_ADD_THRESHOLD = 0.01
DEFAULT_SCENARIO_PRICE_COLUMNS = (
    "SPY",
    "QQQ",
    "RSP",
    "IWM",
    "HYG",
    "LQD",
    "TLT",
    "GLD",
    "SMH",
    "VIXY",
    "UUP",
    "SPHB",
    "SPLV",
    "XLY",
    "XLP",
    "MGC",
    "VTV",
    "VUG",
    "CPER",
    "NVDA",
    "AVGO",
    "MSFT",
    "USO",
    "XLE",
    "DBC",
    "IEF",
    "VGIT",
    "XLI",
    "XLF",
    "BIL",
)


@dataclass(frozen=True)
class DefensiveLayerCalibrationRun:
    origin_states: pd.DataFrame
    episode_outcomes: pd.DataFrame
    calibration_summary: pd.DataFrame
    incremental_comparison: pd.DataFrame
    policy_backtest: pd.DataFrame
    output_paths: dict[str, Path]


def run_defensive_layer_calibration(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: Path = DEFAULT_DEFENSIVE_LAYER_CALIBRATION_DIR,
    primary_strategy: str | None = None,
    frequency: str = "W-WED",
    min_history_days: int = 504,
    max_points: int = 0,
    base_defensive_threshold: float = DEFAULT_BASE_DEFENSIVE_THRESHOLD,
    scenario_defensive_add_threshold: float = DEFAULT_SCENARIO_DEFENSIVE_ADD_THRESHOLD,
    portfolio_defensive_add_threshold: float = DEFAULT_PORTFOLIO_DEFENSIVE_ADD_THRESHOLD,
    horizons: tuple[DefensiveJudgementHorizon, ...] = (
        DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS
    ),
    defensive_ticker: str = "BIL",
    benchmark_ticker: str = "SPY",
    cash_ticker: str = "BIL",
) -> DefensiveLayerCalibrationRun:
    """Replay the quantitative defense layers at point-in-time historical origins.

    News, events, and external macro inputs are excluded. The quantitative sizing
    layer uses the live risk-status rule plus only the configured, calibration-gated
    scenario authority. The portfolio layer then applies the live hard risk limits.
    """

    strategy_name = primary_strategy or config.primary_strategy
    if strategy_name not in baseline_run.results:
        raise ValueError(f"Primary strategy {strategy_name!r} is absent from the snapshot.")
    prices = baseline_run.prices.sort_index()
    result = baseline_run.results[strategy_name]
    common_dates = prices.index.intersection(result.weights.index).sort_values()
    dates = _sample_history_dates(
        common_dates,
        start_date=None,
        end_date=None,
        frequency=frequency,
        max_points=max_points,
        daily_tail_market_days=0,
        min_history_days=min_history_days,
    )
    state_rows: list[dict[str, Any]] = []
    weight_states: dict[
        pd.Timestamp,
        tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    ] = {}
    policy = config.allocation_policy
    risk_config = PortfolioRiskConfig(
        defensive_ticker=defensive_ticker,
        base_max_expected_shortfall_95=policy.normal_tail_loss_limit,
        base_max_stress_loss=policy.catastrophic_stress_loss_limit,
        scenario_budget_authority=policy.scenario_budget_authority,
        scenario_weighted_stress_authority=policy.scenario_weighted_stress_authority,
    )

    for origin in dates:
        price_history = prices.loc[:origin]
        raw_base = result.weights.loc[:origin].iloc[-1].astype(float)
        base_weights = _weights_with_defensive_residual(
            raw_base,
            defensive_ticker=defensive_ticker,
        )
        risk_score, risk_status, lattice, timing = _point_in_time_scenario_state(price_history)
        scenario_context = _scenario_context(
            lattice,
            sizing_authority=policy.scenario_sizing_authority,
        )
        legacy_status_multiplier = float(_risk_status_multiplier(risk_status))
        status_multiplier = float(timing.multiplier)
        scenario_multiplier = float(scenario_context["risk_multiplier"])
        raw_scenario_multiplier = float(scenario_context["raw_risk_multiplier"])
        status_weights = _scenario_adjusted_weights(
            base_weights,
            risk_multiplier=status_multiplier,
            defensive_ticker=defensive_ticker,
        )
        legacy_status_weights = _scenario_adjusted_weights(
            base_weights,
            risk_multiplier=legacy_status_multiplier,
            defensive_ticker=defensive_ticker,
        )
        quant_multiplier = min(status_multiplier, scenario_multiplier)
        scenario_weights = _scenario_adjusted_weights(
            base_weights,
            risk_multiplier=quant_multiplier,
            defensive_ticker=defensive_ticker,
        )
        risk_history_days = max(
            risk_config.factor_lookback_days,
            risk_config.covariance_lookback_days,
            risk_config.correlation_long_lookback_days,
            risk_config.tail_lookback_days,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            portfolio_risk = build_portfolio_risk(
                price_history.tail(risk_history_days + 8),
                scenario_weights,
                lattice,
                current_weights=base_weights,
                config=risk_config,
            )
        final_weights = (
            portfolio_risk.risk_adjusted_weights
            if not portfolio_risk.risk_adjusted_weights.empty
            else scenario_weights
        )
        base_defensive = _defensive_weight(base_weights, defensive_ticker)
        status_defensive = _defensive_weight(status_weights, defensive_ticker)
        scenario_defensive = _defensive_weight(scenario_weights, defensive_ticker)
        final_defensive = _defensive_weight(final_weights, defensive_ticker)
        risk_status_add = max(0.0, status_defensive - base_defensive)
        scenario_probability_add = max(0.0, scenario_defensive - status_defensive)
        scenario_add = max(0.0, scenario_defensive - base_defensive)
        portfolio_add = max(0.0, final_defensive - scenario_defensive)
        flags = layer_flags(
            base_defensive=base_defensive,
            scenario_defensive_add=scenario_add,
            portfolio_defensive_add=portfolio_add,
            base_threshold=base_defensive_threshold,
            scenario_add_threshold=scenario_defensive_add_threshold,
            portfolio_add_threshold=portfolio_defensive_add_threshold,
        )
        portfolio_summary = _first_row(portfolio_risk.summary)
        state_rows.append(
            {
                "origin_date": origin,
                "strategy": strategy_name,
                "risk_score": risk_score,
                "risk_status": risk_status,
                "legacy_risk_status_multiplier": legacy_status_multiplier,
                "risk_status_multiplier": status_multiplier,
                "risk_timing_state": timing.state,
                "risk_timing_break_count": len(timing.confirmation_breaks),
                "risk_timing_breaks": ", ".join(timing.confirmation_breaks),
                "risk_timing_recovery_count": len(timing.recovery_confirmations),
                "risk_timing_recoveries": ", ".join(timing.recovery_confirmations),
                "raw_scenario_multiplier": raw_scenario_multiplier,
                "scenario_multiplier": scenario_multiplier,
                "scenario_sizing_authority": policy.scenario_sizing_authority,
                "scenario_budget_authority": policy.scenario_budget_authority,
                "scenario_weighted_stress_authority": (
                    policy.scenario_weighted_stress_authority
                ),
                "quant_multiplier": quant_multiplier,
                "quant_clamp_source": (
                    "risk_status"
                    if status_multiplier < scenario_multiplier
                    else (
                        "scenario_probability"
                        if scenario_multiplier < status_multiplier
                        else "none"
                    )
                ),
                "risk_off_probability": scenario_context["risk_off_probability"],
                "transition_probability": scenario_context["transition_probability"],
                "fragile_upside_probability": scenario_context[
                    "fragile_upside_probability"
                ],
                "risk_on_probability": scenario_context["risk_on_probability"],
                "base_defensive_weight": base_defensive,
                "risk_status_defensive_weight": status_defensive,
                "scenario_defensive_weight": scenario_defensive,
                "final_defensive_weight": final_defensive,
                "risk_status_defensive_add": risk_status_add,
                "scenario_probability_defensive_add": scenario_probability_add,
                "scenario_defensive_add": scenario_add,
                "portfolio_defensive_add": portfolio_add,
                "portfolio_risk_multiplier": portfolio_summary.get(
                    "portfolio_risk_multiplier", np.nan
                ),
                "portfolio_constraints": portfolio_summary.get("applied_constraints", ""),
                **flags,
            }
        )
        weight_states[pd.Timestamp(origin)] = (
            base_weights,
            scenario_weights,
            final_weights,
            legacy_status_weights,
        )

    states = pd.DataFrame(state_rows).sort_values("origin_date").reset_index(drop=True)
    episodes = _build_episode_outcomes(
        states,
        weight_states,
        prices,
        horizons=horizons,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
        benchmark_ticker=benchmark_ticker,
        cash_ticker=cash_ticker,
    )
    summary = summarize_layer_calibration(episodes)
    comparisons = build_incremental_comparison(summary)
    policy_backtest = build_layer_policy_backtest(
        states,
        weight_states,
        prices,
        transaction_cost_bps=float(config.execution.transaction_cost_bps),
        cash_ticker=cash_ticker,
    )
    output_paths = _write_outputs(
        output_dir=output_dir,
        states=states,
        episodes=episodes,
        summary=summary,
        comparisons=comparisons,
        policy_backtest=policy_backtest,
        baseline_run=baseline_run,
        config=config,
        parameters={
            "primary_strategy": strategy_name,
            "frequency": frequency,
            "min_history_days": min_history_days,
            "max_points": max_points,
            "base_defensive_threshold": base_defensive_threshold,
            "scenario_defensive_add_threshold": scenario_defensive_add_threshold,
            "portfolio_defensive_add_threshold": portfolio_defensive_add_threshold,
            "horizons": [horizon.__dict__ for horizon in horizons],
            "news_events_macro": "excluded",
            "scenario_sizing_authority": policy.scenario_sizing_authority,
            "risk_timing_sizing_authority": policy.risk_timing_sizing_authority,
            "risk_timing_calibration_status": policy.risk_timing_calibration_status,
            "scenario_budget_authority": policy.scenario_budget_authority,
            "scenario_weighted_stress_authority": policy.scenario_weighted_stress_authority,
            "scenario_portfolio_independence": (
                "scenario_probabilities_non_authoritative"
                if max(
                    policy.scenario_sizing_authority,
                    policy.scenario_budget_authority,
                    policy.scenario_weighted_stress_authority,
                )
                == 0
                else "portfolio_limits_partly_reuse_scenario_lattice"
            ),
            "trial_roster": [
                "base_only",
                "quantitative_only",
                "portfolio_only",
                "base_quantitative",
                "base_portfolio",
                "quantitative_portfolio",
                "all_three",
                "one_layer",
                "two_layers",
                "base_any",
                "base_without_all_three",
            ],
        },
    )
    return DefensiveLayerCalibrationRun(
        states,
        episodes,
        summary,
        comparisons,
        policy_backtest,
        output_paths,
    )


def layer_flags(
    *,
    base_defensive: float,
    scenario_defensive_add: float,
    portfolio_defensive_add: float,
    base_threshold: float,
    scenario_add_threshold: float,
    portfolio_add_threshold: float,
) -> dict[str, Any]:
    base = bool(base_defensive >= base_threshold)
    scenario = bool(scenario_defensive_add >= scenario_add_threshold)
    portfolio = bool(portfolio_defensive_add >= portfolio_add_threshold)
    bits = f"{int(base)}{int(scenario)}{int(portfolio)}"
    names = {
        "000": "no_layers",
        "100": "base_only",
        "010": "quantitative_only",
        "001": "portfolio_only",
        "110": "base_quantitative",
        "101": "base_portfolio",
        "011": "quantitative_portfolio",
        "111": "all_three",
    }
    return {
        "base_layer": base,
        "scenario_layer": scenario,
        "quantitative_sizing_layer": scenario,
        "portfolio_layer": portfolio,
        "layer_count": int(base) + int(scenario) + int(portfolio),
        "layer_combination": names[bits],
    }


def _point_in_time_scenario_state(
    price_history: pd.DataFrame,
) -> tuple[float, str, pd.DataFrame, RiskTimingAssessment]:
    available = [column for column in DEFAULT_SCENARIO_PRICE_COLUMNS if column in price_history]
    # Every rolling indicator needs at most 131 observations. Extra rows preserve
    # bounded-forward-fill behavior around the left edge of the calculation window.
    window = price_history[available].tail(145)
    momentum = momentum_state_table(window)
    confirmation = build_confirmation_matrix(window, momentum)
    health = build_market_health(window, momentum)
    full_focus = price_history.reindex(columns=health.index).ffill()
    for ticker in health.index:
        series = full_focus[ticker].dropna()
        if ticker == "VIXY":
            health.loc[ticker, "drawdown"] = np.nan
        elif not series.empty:
            health.loc[ticker, "drawdown"] = float(series.iloc[-1] / series.max() - 1.0)
    risk_score = calculate_risk_score(confirmation, health)
    risk_status = _risk_status(risk_score)
    timing = assess_risk_timing(risk_score, confirmation, health)
    lattice, _drivers = build_scenario_lattice(
        confirmation,
        health,
        momentum,
        risk_score,
        risk_status,
    )
    return risk_score, risk_status, lattice, timing


def _defensive_weight(weights: pd.Series, defensive_ticker: str) -> float:
    explicit = float(weights.get(defensive_ticker, 0.0))
    residual = max(0.0, 1.0 - float(weights.clip(lower=0.0).sum()))
    return float(np.clip(explicit + residual, 0.0, 1.0))


def _cohort_masks(states: pd.DataFrame) -> dict[str, pd.Series]:
    combination = states["layer_combination"].astype(str)
    masks = {
        name: combination.eq(name)
        for name in (
            "base_only",
            "quantitative_only",
            "portfolio_only",
            "base_quantitative",
            "base_portfolio",
            "quantitative_portfolio",
            "all_three",
        )
    }
    masks.update(
        {
            "one_layer": states["layer_count"].eq(1),
            "two_layers": states["layer_count"].eq(2),
            "base_any": states["base_layer"].astype(bool),
            "base_without_all_three": states["base_layer"].astype(bool)
            & ~combination.eq("all_three"),
        }
    )
    return masks


def _build_episode_outcomes(
    states: pd.DataFrame,
    weight_states: dict[
        pd.Timestamp,
        tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    ],
    prices: pd.DataFrame,
    *,
    horizons: tuple[DefensiveJudgementHorizon, ...],
    transaction_cost_bps: float,
    benchmark_ticker: str,
    cash_ticker: str,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    price_index = prices.index
    position_by_date = {pd.Timestamp(date): position for position, date in enumerate(price_index)}
    for cohort, mask in _cohort_masks(states).items():
        starts = mask & ~mask.shift(1, fill_value=False)
        for state_position in np.flatnonzero(starts.to_numpy()):
            state = states.iloc[int(state_position)]
            origin = pd.Timestamp(state["origin_date"])
            price_position = position_by_date.get(origin)
            if price_position is None:
                continue
            base_weights, scenario_weights, final_weights, _legacy_weights = weight_states[origin]
            turnover = float(
                final_weights.reindex(base_weights.index.union(final_weights.index), fill_value=0.0)
                .sub(base_weights.reindex(base_weights.index.union(final_weights.index), fill_value=0.0))
                .abs()
                .sum()
            )
            layered_cost = turnover * transaction_cost_bps / 10_000.0
            for horizon in horizons:
                benchmark_return = _forward_return(
                    prices[benchmark_ticker], price_position, horizon.trading_days
                )
                cash_return = _forward_return(
                    prices[cash_ticker], price_position, horizon.trading_days
                )
                benchmark_drawdown = _forward_max_drawdown(
                    prices[benchmark_ticker], price_position, horizon.trading_days
                )
                base_return, base_drawdown = frozen_weight_outcome(
                    prices,
                    base_weights,
                    price_position,
                    horizon.trading_days,
                    initial_cost=0.0,
                )
                final_return, final_drawdown = frozen_weight_outcome(
                    prices,
                    final_weights,
                    price_position,
                    horizon.trading_days,
                    initial_cost=layered_cost,
                )
                scenario_turnover = float(
                    scenario_weights.reindex(
                        base_weights.index.union(scenario_weights.index), fill_value=0.0
                    )
                    .sub(
                        base_weights.reindex(
                            base_weights.index.union(scenario_weights.index), fill_value=0.0
                        )
                    )
                    .abs()
                    .sum()
                )
                scenario_cost = scenario_turnover * transaction_cost_bps / 10_000.0
                scenario_return, scenario_drawdown = frozen_weight_outcome(
                    prices,
                    scenario_weights,
                    price_position,
                    horizon.trading_days,
                    initial_cost=scenario_cost,
                )
                judgement = classify_defensive_judgement(
                    benchmark_forward_return=benchmark_return,
                    cash_forward_return=cash_return,
                    benchmark_forward_max_drawdown=benchmark_drawdown,
                    horizon=horizon,
                )
                layered_excess = _subtract(final_return, base_return)
                drawdown_improvement = _subtract(final_drawdown, base_drawdown)
                rows.append(
                    {
                        **state.to_dict(),
                        "cohort": cohort,
                        "horizon": horizon.label,
                        "forward_days": horizon.trading_days,
                        "judgement": judgement,
                        "benchmark_forward_return": benchmark_return,
                        "cash_forward_return": cash_return,
                        "benchmark_excess_vs_cash": _subtract(
                            benchmark_return, cash_return
                        ),
                        "benchmark_forward_max_drawdown": benchmark_drawdown,
                        "base_frozen_return": base_return,
                        "scenario_frozen_return_net": scenario_return,
                        "final_frozen_return_net": final_return,
                        "scenario_excess_vs_base": _subtract(scenario_return, base_return),
                        "portfolio_excess_vs_scenario": _subtract(
                            final_return, scenario_return
                        ),
                        "layered_excess_vs_base": layered_excess,
                        "base_frozen_max_drawdown": base_drawdown,
                        "scenario_frozen_max_drawdown_net": scenario_drawdown,
                        "final_frozen_max_drawdown_net": final_drawdown,
                        "scenario_drawdown_improvement_vs_base": _subtract(
                            scenario_drawdown, base_drawdown
                        ),
                        "portfolio_drawdown_improvement_vs_scenario": _subtract(
                            final_drawdown, scenario_drawdown
                        ),
                        "drawdown_improvement_vs_base": drawdown_improvement,
                        "regret_vs_base": (
                            max(0.0, -float(layered_excess))
                            if pd.notna(layered_excess)
                            else np.nan
                        ),
                        "avoided_loss_vs_base": (
                            max(0.0, float(layered_excess))
                            if pd.notna(layered_excess)
                            else np.nan
                        ),
                        "layered_turnover_vs_base": turnover,
                        "layered_initial_cost": layered_cost,
                    }
                )
    return pd.DataFrame(rows)


def frozen_weight_outcome(
    prices: pd.DataFrame,
    weights: pd.Series,
    position: int,
    days: int,
    *,
    initial_cost: float,
) -> tuple[float | pd.NA, float | pd.NA]:
    end = position + days
    if position < 0 or end >= len(prices):
        return pd.NA, pd.NA
    clean_weights = weights[weights.abs() > 1e-12].astype(float)
    if clean_weights.empty or any(ticker not in prices for ticker in clean_weights.index):
        return pd.NA, pd.NA
    path = bounded_forward_fill(prices.iloc[position : end + 1][clean_weights.index])
    start = path.iloc[0]
    if start.isna().any() or (start == 0.0).any() or path.isna().any().any():
        return pd.NA, pd.NA
    relatives = path.divide(start, axis=1)
    equity = relatives.mul(clean_weights, axis=1).sum(axis=1) * (1.0 - initial_cost)
    equity = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)], ignore_index=True)
    drawdowns = equity / equity.cummax() - 1.0
    return float(equity.iloc[-1] - 1.0), float(drawdowns.min())


def summarize_layer_calibration(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    eligible = episodes[episodes["judgement"].ne("insufficient_forward_data")].copy()
    if eligible.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (cohort, horizon), group in eligible.groupby(["cohort", "horizon"], sort=False):
        judgement = group["judgement"].astype(str)
        correct = int(judgement.eq("correct_defense").sum())
        false = int(judgement.eq("false_alarm").sum())
        n = int(len(group))
        correct_low, correct_high = _wilson_interval(correct, n)
        false_low, false_high = _wilson_interval(false, n)
        rows.append(
            {
                "cohort": cohort,
                "horizon": horizon,
                "episode_starts": n,
                "correct_defense_rate": correct / n,
                "correct_defense_ci_low": correct_low,
                "correct_defense_ci_high": correct_high,
                "false_alarm_rate": false / n,
                "false_alarm_ci_low": false_low,
                "false_alarm_ci_high": false_high,
                "mixed_rate": float(judgement.eq("mixed_or_early").mean()),
                **_distribution_fields(group, "benchmark_forward_return", "spy_return"),
                **_distribution_fields(
                    group, "benchmark_forward_max_drawdown", "spy_max_drawdown"
                ),
                **_distribution_fields(group, "layered_excess_vs_base", "layered_excess"),
                **_distribution_fields(
                    group, "scenario_excess_vs_base", "scenario_excess"
                ),
                **_distribution_fields(
                    group, "portfolio_excess_vs_scenario", "portfolio_excess"
                ),
                **_distribution_fields(
                    group, "drawdown_improvement_vs_base", "drawdown_improvement"
                ),
                **_distribution_fields(
                    group,
                    "scenario_drawdown_improvement_vs_base",
                    "scenario_drawdown_improvement",
                ),
                **_distribution_fields(
                    group,
                    "portfolio_drawdown_improvement_vs_scenario",
                    "portfolio_drawdown_improvement",
                ),
                "layered_beats_base_rate": float(
                    pd.to_numeric(group["layered_excess_vs_base"], errors="coerce").gt(0).mean()
                ),
                "layered_improves_drawdown_rate": float(
                    pd.to_numeric(group["drawdown_improvement_vs_base"], errors="coerce")
                    .gt(0)
                    .mean()
                ),
                "median_regret_vs_base": float(
                    pd.to_numeric(group["regret_vs_base"], errors="coerce").median()
                ),
                "median_avoided_loss_vs_base": float(
                    pd.to_numeric(group["avoided_loss_vs_base"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "cohort"]).reset_index(drop=True)


def build_incremental_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    comparison_pairs = (
        ("base_quantitative", "base_only"),
        ("quantitative_portfolio", "quantitative_only"),
        ("two_layers", "one_layer"),
        ("all_three", "base_only"),
        ("all_three", "one_layer"),
        ("all_three", "two_layers"),
    )
    for horizon in summary["horizon"].unique():
        scoped = summary[summary["horizon"].eq(horizon)].set_index("cohort")
        for target, comparator in comparison_pairs:
            if target not in scoped.index or comparator not in scoped.index:
                continue
            target_row = scoped.loc[target]
            other = scoped.loc[comparator]
            rows.append(
                {
                    "horizon": horizon,
                    "comparison": f"{target}_minus_{comparator}",
                    "target_cohort": target,
                    "comparator_cohort": comparator,
                    "target_episodes": int(target_row["episode_starts"]),
                    "comparator_episodes": int(other["episode_starts"]),
                    "delta_correct_defense_rate": float(
                        target_row["correct_defense_rate"]
                        - other["correct_defense_rate"]
                    ),
                    "delta_false_alarm_rate": float(
                        target_row["false_alarm_rate"] - other["false_alarm_rate"]
                    ),
                    "delta_median_layered_excess_vs_base": float(
                        target_row["layered_excess_p50"] - other["layered_excess_p50"]
                    ),
                    "delta_median_drawdown_improvement": float(
                        target_row["drawdown_improvement_p50"]
                        - other["drawdown_improvement_p50"]
                    ),
                    "evidence_grade": _sample_grade(
                        int(target_row["episode_starts"]),
                        int(other["episode_starts"]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_layer_policy_backtest(
    states: pd.DataFrame,
    weight_states: dict[
        pd.Timestamp,
        tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    ],
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    cash_ticker: str,
) -> pd.DataFrame:
    """Compare weekly base and layered policies without overlapping forward windows."""

    if states.empty or cash_ticker not in prices:
        return pd.DataFrame()
    valid_cash = prices[cash_ticker].dropna()
    if valid_cash.empty:
        return pd.DataFrame()
    first_cash_date = pd.Timestamp(valid_cash.index.min())
    origin_rows = states[states["origin_date"].ge(first_cash_date)].copy()
    if origin_rows.empty:
        return pd.DataFrame()
    origins = [pd.Timestamp(value) for value in origin_rows["origin_date"]]
    current_combination = str(origin_rows.iloc[-1]["layer_combination"])
    current_configuration = dict(
        zip(
            origins,
            origin_rows["layer_combination"].eq(current_combination),
            strict=True,
        )
    )
    policies = {
        "base_weekly": lambda date: weight_states[date][0],
        "legacy_risk_status_weekly": lambda date: weight_states[date][3],
        "quantitative_sizing_weekly": lambda date: weight_states[date][1],
        "full_layers_weekly": lambda date: weight_states[date][2],
        "current_configuration_overlay_only": lambda date: (
            weight_states[date][2]
            if bool(current_configuration[date])
            else weight_states[date][0]
        ),
    }
    rows = []
    for policy_name, target_for_date in policies.items():
        metrics = _simulate_weekly_policy(
            origins,
            prices,
            target_for_date,
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.append({"policy": policy_name, **metrics})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    base = frame.loc[frame["policy"].eq("base_weekly")].iloc[0]
    frame["cagr_delta_vs_base"] = frame["cagr"] - float(base["cagr"])
    frame["max_drawdown_delta_vs_base"] = (
        frame["max_drawdown"] - float(base["max_drawdown"])
    )
    frame["terminal_wealth_delta_vs_base"] = (
        frame["terminal_wealth"] - float(base["terminal_wealth"])
    )
    return frame


def _simulate_weekly_policy(
    origins: list[pd.Timestamp],
    prices: pd.DataFrame,
    target_for_date: Any,
    *,
    transaction_cost_bps: float,
) -> dict[str, float | int | str]:
    positions = {pd.Timestamp(date): position for position, date in enumerate(prices.index)}
    wealth = 1.0
    equity_values = [wealth]
    equity_dates = [origins[0]]
    current_weights: pd.Series | None = None
    turnovers: list[float] = []
    total_cost_fraction = 0.0
    for origin_number, origin in enumerate(origins):
        target = target_for_date(origin).astype(float)
        target = target[target.abs() > 1e-12]
        target = target / float(target.sum())
        if current_weights is None:
            turnover = 0.0
        else:
            universe = current_weights.index.union(target.index)
            turnover = float(
                current_weights.reindex(universe, fill_value=0.0)
                .sub(target.reindex(universe, fill_value=0.0))
                .abs()
                .sum()
            )
        cost_fraction = turnover * transaction_cost_bps / 10_000.0
        wealth *= 1.0 - cost_fraction
        total_cost_fraction += cost_fraction
        turnovers.append(turnover)
        origin_position = positions[origin]
        next_position = (
            positions[origins[origin_number + 1]]
            if origin_number + 1 < len(origins)
            else len(prices) - 1
        )
        held = target.copy()
        held_prices = bounded_forward_fill(
            prices.iloc[origin_position : next_position + 1].reindex(columns=held.index)
        )
        for step in range(1, len(held_prices)):
            previous = held_prices.iloc[step - 1]
            latest = held_prices.iloc[step]
            asset_returns = (latest / previous - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            portfolio_return = float(held.mul(asset_returns).sum())
            wealth *= 1.0 + portfolio_return
            denominator = 1.0 + portfolio_return
            if denominator > 0.0:
                held = held.mul(1.0 + asset_returns) / denominator
            equity_values.append(wealth)
            equity_dates.append(pd.Timestamp(held_prices.index[step]))
        current_weights = held
    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 365.25)
    daily_returns = equity.pct_change().dropna()
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    volatility = float(daily_returns.std() * math.sqrt(252.0)) if not daily_returns.empty else 0.0
    return {
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
        "years": years,
        "terminal_wealth": float(equity.iloc[-1]),
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "annualized_volatility": volatility,
        "average_rebalance_turnover": float(np.mean(turnovers)),
        "total_cost_fraction": total_cost_fraction,
        "rebalance_origins": len(origins),
    }


def _distribution_fields(group: pd.DataFrame, column: str, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return {
        f"{prefix}_p10": float(values.quantile(0.10)) if not values.empty else np.nan,
        f"{prefix}_p50": float(values.quantile(0.50)) if not values.empty else np.nan,
        f"{prefix}_p90": float(values.quantile(0.90)) if not values.empty else np.nan,
    }


def _wilson_interval(successes: int, observations: int, z: float = 1.96) -> tuple[float, float]:
    if observations <= 0:
        return np.nan, np.nan
    p = successes / observations
    denominator = 1.0 + z**2 / observations
    center = (p + z**2 / (2.0 * observations)) / denominator
    margin = z * math.sqrt(
        (p * (1.0 - p) + z**2 / (4.0 * observations)) / observations
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _sample_grade(left: int, right: int) -> str:
    minimum = min(left, right)
    if minimum >= 30:
        return "moderate_sample"
    if minimum >= 10:
        return "small_sample"
    return "insufficient_sample"


def _forward_return(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    path = bounded_forward_fill(series.iloc[position : end + 1])
    if path.isna().any() or float(path.iloc[0]) == 0.0:
        return pd.NA
    return float(path.iloc[-1] / path.iloc[0] - 1.0)


def _forward_max_drawdown(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    path = bounded_forward_fill(series.iloc[position : end + 1]).dropna()
    if path.empty:
        return pd.NA
    relative = path / path.iloc[0]
    return float((relative / relative.cummax() - 1.0).min())


def _subtract(left: object, right: object) -> float | pd.NA:
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return float(left) - float(right)


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if frame is not None and not frame.empty else {}


def _write_outputs(
    *,
    output_dir: Path,
    states: pd.DataFrame,
    episodes: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    policy_backtest: pd.DataFrame,
    baseline_run: BaselineRun,
    config: BotConfig,
    parameters: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "origin_states": output_dir / "origin_states.csv",
        "episode_outcomes": output_dir / "episode_outcomes.csv",
        "calibration_summary": output_dir / "calibration_summary.csv",
        "incremental_comparison": output_dir / "incremental_comparison.csv",
        "policy_backtest": output_dir / "policy_backtest.csv",
        "summary": output_dir / "summary.md",
    }
    states.to_csv(paths["origin_states"], index=False)
    episodes.to_csv(paths["episode_outcomes"], index=False)
    summary.to_csv(paths["calibration_summary"], index=False)
    comparisons.to_csv(paths["incremental_comparison"], index=False)
    policy_backtest.to_csv(paths["policy_backtest"], index=False)
    paths["summary"].write_text(
        _markdown_summary(states, summary, comparisons, policy_backtest, parameters),
        encoding="utf-8",
    )
    manifest_path = write_research_manifest(
        output_dir,
        study="defensive_layer_calibration",
        config=config,
        prices=baseline_run.prices,
        parameters=parameters,
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest_path
    return paths


def _markdown_summary(
    states: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    policy_backtest: pd.DataFrame,
    parameters: dict[str, Any],
) -> str:
    lines = [
        "# Defensive Layer Calibration",
        "",
        "## Scope",
        "",
        f"- Point-in-time origins: {len(states):,} ({parameters['frequency']})",
        f"- Base layer: defensive weight >= {parameters['base_defensive_threshold']:.0%}",
        (
            "- Quantitative sizing layer: confirmation-timed risk state plus any calibration-authorized "
            "scenario clamp adds at least "
            f"{parameters['scenario_defensive_add_threshold']:.0%} defense"
        ),
        (
            "- Risk-timing operating authority: "
            f"{parameters['risk_timing_sizing_authority']:.0%} "
            f"({parameters['risk_timing_calibration_status']}); the raw candidate is replayed "
            "below for research but cannot size the live book."
        ),
        (
            "- Portfolio layer: stress constraints add at least "
            f"{parameters['portfolio_defensive_add_threshold']:.0%} defense"
        ),
        (
            "- Latest classified combination: "
            f"{states.iloc[-1]['layer_combination'] if not states.empty else 'unavailable'}."
        ),
        "- News, events, and external macro inputs are excluded.",
        (
            "- Scenario probability authority: sizing "
            f"{parameters['scenario_sizing_authority']:.0%}, portfolio budget "
            f"{parameters['scenario_budget_authority']:.0%}, scenario-weighted stress "
            f"{parameters['scenario_weighted_stress_authority']:.0%}."
        ),
        "- Episode starts, not every persistent weekly observation, are the unit of analysis.",
        "",
        "## Logic caveat",
        "",
        (
            "With all scenario authorities at zero, scenario probabilities remain diagnostic "
            "and cannot create defense. The active three-layer agreement test is "
            "base strategy defense plus confirmation-timed price state plus independent hard "
            "portfolio limits. These layers still share market-price inputs, so they are "
            "distinct causal pathways rather than statistically independent votes."
            if parameters["scenario_portfolio_independence"]
            == "scenario_probabilities_non_authoritative"
            else (
                "The portfolio layer is downstream and partly reuses scenario probabilities "
                "when it sets stress budgets. Agreement is incremental confirmation from a "
                "distinct constraint calculation, not statistically independent votes."
            )
        ),
        "",
        "## Results",
        "",
    ]
    if summary.empty:
        lines.append("No eligible completed episodes were available.")
    else:
        display_columns = [
            "cohort",
            "horizon",
            "episode_starts",
            "correct_defense_rate",
            "false_alarm_rate",
            "spy_return_p50",
            "spy_max_drawdown_p50",
            "layered_excess_p50",
            "drawdown_improvement_p50",
        ]
        lines.extend(_markdown_table(summary[display_columns]))
    lines.extend(["", "## Incremental comparisons", ""])
    if comparisons.empty:
        lines.append("No multi-layer cohort had an eligible comparator.")
    else:
        lines.extend(_markdown_table(comparisons))
    lines.extend(["", "## Non-overlapping weekly policy replay", ""])
    if policy_backtest.empty:
        lines.append("No eligible weekly policy replay was available.")
    else:
        lines.extend(_markdown_table(policy_backtest))
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "- Retrospective research only; this is not prospective validation.",
            "- The current ticker universe is replayed backward and is not point-in-time membership safe.",
            "- Episode counts can be small, and forward horizons can overlap across distinct cohorts.",
            "- Episode-level portfolios are frozen at origin to isolate sizing consequences.",
            "- The dynamic policy replay rebalances weekly and is not the live daily execution path.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.3f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return rows
