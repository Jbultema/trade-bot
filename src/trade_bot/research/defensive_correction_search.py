from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import BotConfig
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_bias_calibration import (
    CRISIS_WINDOWS,
    ERA_WINDOWS,
    _result_from_execution_weights,
    _return_slice_stats,
    comparable_strategy_family,
)

DEFAULT_DEFENSIVE_CORRECTION_SEARCH_DIR = Path(
    "reports/defensive_correction_search"
)
DEFENSIVE_TICKER = "BIL"
DEFENSE_TRIGGER = 0.60
MAX_RELIEF = 0.20

MECHANISM_ROSTER = (
    "fixed_existing_sleeve_relief",
    "dual_trend_intact_relief",
    "credit_volatility_intact_relief",
    "breadth_intact_relief",
    "positive_momentum_relief",
    "low_volatility_relief",
    "shallow_drawdown_relief",
    "defense_duration_decay",
    "rapid_ramp_damper",
    "recovery_cross_accelerator",
    "intact_risk_floor",
    "spy_beta_bridge",
    "splv_low_beta_bridge",
    "rsp_breadth_bridge",
    "family_disagreement_relief",
    "family_gap_proportional_relief",
    "opportunity_cost_feedback",
    "native_reentry_accelerator",
    "health_score_proportional_relief",
    "ramp_duration_composite",
    "breadth_break_veto_relief",
    "breadth_dual_trend_relief",
    "breadth_shallow_drawdown_relief",
    "breadth_low_volatility_relief",
    "breadth_positive_momentum_relief",
    "breadth_credit_volatility_relief",
    "breadth_duration_confirmed_relief",
    "breadth_splv_bridge",
    "breadth_split_existing_splv",
    "breadth_state_adaptive_bridge",
)

MECHANISM_DESCRIPTIONS = {
    "fixed_existing_sleeve_relief": "Unconditional five-point relief to the existing risk sleeve.",
    "dual_trend_intact_relief": "Five-point relief only when SPY and QQQ 200-day trends are intact.",
    "credit_volatility_intact_relief": "Five-point relief when credit and volatility do not confirm stress.",
    "breadth_intact_relief": "Five-point relief when the RSP/SPY relative trend is intact.",
    "positive_momentum_relief": "Five-point relief when SPY and QQQ one-month momentum is positive.",
    "low_volatility_relief": "Five-point relief below the trailing realized-volatility threshold.",
    "shallow_drawdown_relief": "Five-point relief while SPY drawdown is shallower than five percent.",
    "defense_duration_decay": "Ten-point release after twenty consecutive high-defense sessions.",
    "rapid_ramp_damper": "Reverse five-session defensive increases above ten points.",
    "recovery_cross_accelerator": "Ten-point release for twenty sessions after SPY recovers its 50-day average.",
    "intact_risk_floor": "Maintain at least 45 percent existing risk exposure in intact dual trends.",
    "spy_beta_bridge": "Bridge up to ten defensive points into SPY.",
    "splv_low_beta_bridge": "Bridge up to ten defensive points into SPLV.",
    "rsp_breadth_bridge": "Bridge up to ten defensive points into RSP.",
    "family_disagreement_relief": "Release ten points when strategy defense disagrees with its family median.",
    "family_gap_proportional_relief": "Release the strategy-family defensive gap, capped at fifteen points.",
    "opportunity_cost_feedback": "Release ten points after SPY gains five percent during a defensive episode.",
    "native_reentry_accelerator": "Add five points to an already-started native re-entry.",
    "health_score_proportional_relief": "Scale relief with five independent intact market confirmations.",
    "ramp_duration_composite": "Take the larger of rapid-ramp and duration corrections.",
    "breadth_break_veto_relief": "Breadth relief with a multi-group break veto.",
    "breadth_dual_trend_relief": "Breadth relief requiring intact SPY and QQQ long trends.",
    "breadth_shallow_drawdown_relief": "Breadth relief only during shallow SPY drawdowns.",
    "breadth_low_volatility_relief": "Breadth relief only during low realized volatility.",
    "breadth_positive_momentum_relief": "Breadth relief requiring positive SPY and QQQ momentum.",
    "breadth_credit_volatility_relief": "Breadth relief requiring intact credit and volatility.",
    "breadth_duration_confirmed_relief": "Ten-point breadth relief after persistent high defense.",
    "breadth_splv_bridge": "Five-point breadth-gated bridge into SPLV.",
    "breadth_split_existing_splv": "Split breadth relief between native risk and SPLV.",
    "breadth_state_adaptive_bridge": "Choose native risk, SPLV, or no relief by break count.",
}


@dataclass(frozen=True)
class DefensiveCorrectionSearchRun:
    mechanism_scorecard: pd.DataFrame
    strategy_metrics: pd.DataFrame
    era_metrics: pd.DataFrame
    ordinary_metrics: pd.DataFrame
    rolling_metrics: pd.DataFrame
    crisis_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    current_effects: pd.DataFrame
    output_paths: dict[str, Path]


def run_defensive_correction_search(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_DEFENSIVE_CORRECTION_SEARCH_DIR,
) -> DefensiveCorrectionSearchRun:
    prices = baseline_run.prices.sort_index()
    strategy_families = {
        name: family
        for name in baseline_run.results
        if (family := comparable_strategy_family(name)) is not None
    }
    if not strategy_families:
        raise ValueError("No comparable dynamic strategy paths are available.")
    signals = build_point_in_time_correction_signals(prices)
    family_defense = build_family_defense_paths(
        baseline_run,
        strategy_families,
    )
    transaction_cost_bps = float(config.execution.transaction_cost_bps)

    adjusted_results: dict[tuple[str, str], BacktestResult] = {}
    strategy_rows: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        base_metrics = calculate_metrics(
            base.name,
            base.returns,
            base.equity,
            base.turnover,
            base.transaction_costs,
        )
        for mechanism in MECHANISM_ROSTER:
            adjusted_weights, relief = build_mechanism_weight_path(
                mechanism,
                base.weights,
                prices,
                signals,
                family_median_defense=family_defense[family],
            )
            candidate = _result_from_execution_weights(
                base,
                prices,
                adjusted_weights,
                transaction_cost_bps=transaction_cost_bps,
                name=f"{strategy}__{mechanism}",
            )
            adjusted_results[(strategy, mechanism)] = candidate
            candidate_metrics = calculate_metrics(
                candidate.name,
                candidate.returns,
                candidate.equity,
                candidate.turnover,
                candidate.transaction_costs,
            )
            active = relief.abs().gt(1e-10)
            active_relief = relief.loc[active].abs()
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "family": family,
                    "mechanism": mechanism,
                    "description": MECHANISM_DESCRIPTIONS[mechanism],
                    "cagr": candidate_metrics.cagr,
                    "max_drawdown": candidate_metrics.max_drawdown,
                    "sharpe": candidate_metrics.sharpe,
                    "average_turnover": candidate_metrics.average_turnover,
                    "total_transaction_cost": candidate_metrics.total_transaction_cost,
                    "cagr_delta_vs_base": candidate_metrics.cagr - base_metrics.cagr,
                    "max_drawdown_delta_vs_base": (
                        candidate_metrics.max_drawdown - base_metrics.max_drawdown
                    ),
                    "sharpe_delta_vs_base": candidate_metrics.sharpe - base_metrics.sharpe,
                    "active_day_rate": float(active.mean()),
                    "mean_active_defense_reduction": (
                        float(active_relief.mean()) if not active_relief.empty else 0.0
                    ),
                    "median_active_defense_reduction": (
                        float(active_relief.median()) if not active_relief.empty else 0.0
                    ),
                    "max_defense_reduction": (
                        float(active_relief.max()) if not active_relief.empty else 0.0
                    ),
                }
            )
            current_rows.append(
                {
                    "market_date": prices.index[-1],
                    "strategy": strategy,
                    "family": family,
                    "mechanism": mechanism,
                    "base_defensive_weight": float(
                        effective_defensive_weight_path(base.weights).iloc[-1]
                    ),
                    "adjusted_defensive_weight": float(
                        effective_defensive_weight_path(adjusted_weights).iloc[-1]
                    ),
                    "current_defense_reduction": float(relief.iloc[-1]),
                    "allocation_authority": 0.0,
                }
            )

    strategy_metrics = pd.DataFrame(strategy_rows)
    current_effects = pd.DataFrame(current_rows)
    era_metrics = build_era_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    crisis_metrics = build_crisis_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    ordinary_metrics = build_ordinary_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    rolling_metrics = build_rolling_metrics(
        baseline_run,
        adjusted_results,
        strategy_families,
    )
    cost_sensitivity = build_focus_cost_sensitivity(
        baseline_run,
        config.primary_strategy,
        prices,
        signals,
        family_defense[strategy_families[config.primary_strategy]],
    )
    mechanism_scorecard = build_mechanism_scorecard(
        strategy_metrics,
        era_metrics,
        ordinary_metrics,
        rolling_metrics,
        crisis_metrics,
        cost_sensitivity,
        current_effects,
        focus_strategy=config.primary_strategy,
    )
    paths = write_search_outputs(
        output_dir=output_dir,
        config=config,
        prices=prices,
        frames={
            "mechanism_scorecard": mechanism_scorecard,
            "strategy_metrics": strategy_metrics,
            "era_metrics": era_metrics,
            "ordinary_metrics": ordinary_metrics,
            "rolling_metrics": rolling_metrics,
            "crisis_metrics": crisis_metrics,
            "cost_sensitivity": cost_sensitivity,
            "current_effects": current_effects,
        },
    )
    return DefensiveCorrectionSearchRun(
        mechanism_scorecard=mechanism_scorecard,
        strategy_metrics=strategy_metrics,
        era_metrics=era_metrics,
        ordinary_metrics=ordinary_metrics,
        rolling_metrics=rolling_metrics,
        crisis_metrics=crisis_metrics,
        cost_sensitivity=cost_sensitivity,
        current_effects=current_effects,
        output_paths=paths,
    )


def build_point_in_time_correction_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Build prior-close-only states for correction rules."""

    index = prices.index
    spy = prices["SPY"].ffill()
    qqq = prices["QQQ"].ffill()
    rsp = prices["RSP"].ffill()
    hyg = prices["HYG"].ffill()
    lqd = prices["LQD"].ffill()
    vixy = prices["VIXY"].ffill()

    spy_ma50 = spy.rolling(50, min_periods=50).mean()
    spy_ma200 = spy.rolling(200, min_periods=200).mean()
    qqq_ma200 = qqq.rolling(200, min_periods=200).mean()
    credit = hyg / lqd
    credit_ma100 = credit.rolling(100, min_periods=100).mean()
    breadth = rsp / spy
    breadth_ma100 = breadth.rolling(100, min_periods=100).mean()
    spy_return_21 = spy.pct_change(21)
    qqq_return_21 = qqq.pct_change(21)
    vixy_return_21 = vixy.pct_change(21)
    spy_daily = spy.pct_change()
    realized_vol = spy_daily.rolling(21, min_periods=21).std() * np.sqrt(252.0)
    vol_threshold = realized_vol.rolling(756, min_periods=252).quantile(0.60)
    spy_drawdown = spy / spy.rolling(252, min_periods=63).max() - 1.0

    raw = pd.DataFrame(index=index)
    raw["spy_above_50"] = spy.gt(spy_ma50)
    raw["spy_above_200"] = spy.gt(spy_ma200)
    raw["qqq_above_200"] = qqq.gt(qqq_ma200)
    raw["dual_trend_intact"] = raw["spy_above_200"] & raw["qqq_above_200"]
    raw["credit_intact"] = credit.ge(credit_ma100)
    raw["volatility_intact"] = vixy_return_21.le(0.10)
    raw["breadth_intact"] = breadth.ge(breadth_ma100)
    raw["positive_momentum"] = spy_return_21.gt(0.0) & qqq_return_21.gt(0.0)
    raw["low_volatility"] = realized_vol.le(vol_threshold)
    raw["shallow_drawdown"] = spy_drawdown.gt(-0.05)
    raw["trend_break"] = ~raw["dual_trend_intact"]
    raw["credit_break"] = ~raw["credit_intact"]
    raw["volatility_break"] = ~raw["volatility_intact"]
    raw["breadth_break"] = ~raw["breadth_intact"]
    raw["break_count"] = raw[
        ["trend_break", "credit_break", "volatility_break", "breadth_break"]
    ].sum(axis=1)
    raw["confirmed_break"] = raw["break_count"].ge(2)
    raw["recovery_cross"] = raw["spy_above_50"] & ~raw["spy_above_50"].shift(
        1, fill_value=False
    )
    raw["health_score"] = raw[
        [
            "dual_trend_intact",
            "credit_intact",
            "volatility_intact",
            "breadth_intact",
            "positive_momentum",
        ]
    ].sum(axis=1)
    raw["spy_level"] = spy
    raw["spy_available"] = spy.notna()
    raw["splv_available"] = (
        prices["SPLV"].notna() if "SPLV" in prices else pd.Series(False, index=index)
    )
    raw["rsp_available"] = rsp.notna()

    prior = raw.shift(1)
    bool_columns = [
        column
        for column in prior.columns
        if column
        not in {
            "break_count",
            "health_score",
            "spy_level",
        }
    ]
    for column in bool_columns:
        prior[column] = raw[column].astype(bool).shift(1, fill_value=False)
    prior["break_count"] = pd.to_numeric(prior["break_count"], errors="coerce").fillna(4.0)
    prior["health_score"] = pd.to_numeric(
        prior["health_score"], errors="coerce"
    ).fillna(0.0)
    prior["recovery_window"] = (
        prior["recovery_cross"].astype(int).rolling(20, min_periods=1).max().gt(0)
    )
    return prior


def effective_defensive_weight_path(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str = DEFENSIVE_TICKER,
) -> pd.Series:
    clean = weights.astype(float).clip(lower=0.0)
    explicit = (
        clean[defensive_ticker]
        if defensive_ticker in clean
        else pd.Series(0.0, index=clean.index)
    )
    residual = (1.0 - clean.sum(axis=1)).clip(lower=0.0)
    return (explicit + residual).clip(0.0, 1.0)


def build_family_defense_paths(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
) -> dict[str, pd.Series]:
    rows: dict[str, list[pd.Series]] = {}
    for strategy, family in strategy_families.items():
        defense = effective_defensive_weight_path(
            baseline_run.results[strategy].weights
        ).rename(strategy)
        rows.setdefault(family, []).append(defense)
    return {
        family: pd.concat(series, axis=1).median(axis=1)
        for family, series in rows.items()
    }


def build_mechanism_weight_path(
    mechanism: str,
    base_weights: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    family_median_defense: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if mechanism not in MECHANISM_ROSTER:
        raise ValueError(f"Unknown correction mechanism: {mechanism}")
    index = base_weights.index
    state = signals.reindex(index).ffill()
    defense = effective_defensive_weight_path(base_weights)
    high = defense.ge(DEFENSE_TRIGGER)
    no_break = ~state["confirmed_break"].fillna(True).astype(bool)
    fixed = pd.Series(0.05, index=index).where(high, 0.0)
    duration = consecutive_true_count(high)

    if mechanism == "fixed_existing_sleeve_relief":
        desired = fixed
        destination = "existing"
    elif mechanism == "dual_trend_intact_relief":
        desired = fixed.where(state["dual_trend_intact"].astype(bool), 0.0)
        destination = "existing"
    elif mechanism == "credit_volatility_intact_relief":
        intact = state["credit_intact"].astype(bool) & state[
            "volatility_intact"
        ].astype(bool)
        desired = fixed.where(intact, 0.0)
        destination = "existing"
    elif mechanism == "breadth_intact_relief":
        desired = fixed.where(state["breadth_intact"].astype(bool), 0.0)
        destination = "existing"
    elif mechanism == "positive_momentum_relief":
        desired = fixed.where(state["positive_momentum"].astype(bool), 0.0)
        destination = "existing"
    elif mechanism == "low_volatility_relief":
        desired = fixed.where(state["low_volatility"].astype(bool), 0.0)
        destination = "existing"
    elif mechanism == "shallow_drawdown_relief":
        desired = fixed.where(state["shallow_drawdown"].astype(bool), 0.0)
        destination = "existing"
    elif mechanism == "defense_duration_decay":
        desired = pd.Series(0.10, index=index).where(
            high & duration.ge(20) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "rapid_ramp_damper":
        five_day_increase = defense - defense.shift(5)
        desired = (five_day_increase - 0.10).clip(lower=0.0, upper=MAX_RELIEF).where(
            high & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "recovery_cross_accelerator":
        desired = pd.Series(0.10, index=index).where(
            high & state["recovery_window"].astype(bool) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "intact_risk_floor":
        desired = (0.45 - (1.0 - defense)).clip(
            lower=0.0, upper=MAX_RELIEF
        ).where(state["dual_trend_intact"].astype(bool) & no_break, 0.0)
        destination = "existing"
    elif mechanism == "spy_beta_bridge":
        desired = pd.Series(0.10, index=index).where(
            high
            & state["dual_trend_intact"].astype(bool)
            & state["spy_available"].astype(bool)
            & no_break,
            0.0,
        )
        destination = "SPY"
    elif mechanism == "splv_low_beta_bridge":
        desired = pd.Series(0.10, index=index).where(
            high
            & state["dual_trend_intact"].astype(bool)
            & state["splv_available"].astype(bool)
            & no_break,
            0.0,
        )
        destination = "SPLV"
    elif mechanism == "rsp_breadth_bridge":
        desired = pd.Series(0.10, index=index).where(
            high
            & state["spy_above_200"].astype(bool)
            & state["breadth_intact"].astype(bool)
            & state["rsp_available"].astype(bool)
            & no_break,
            0.0,
        )
        destination = "RSP"
    elif mechanism == "family_disagreement_relief":
        family = family_median_defense.reindex(index).ffill()
        desired = pd.Series(0.10, index=index).where(
            high & family.lt(0.50) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "family_gap_proportional_relief":
        family = family_median_defense.reindex(index).ffill()
        desired = (defense - family).clip(lower=0.0, upper=0.15).where(
            high & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "opportunity_cost_feedback":
        spy_prior = state["spy_level"]
        episode = (~high).cumsum()
        episode_start = spy_prior.groupby(episode).transform("first")
        opportunity_gain = spy_prior / episode_start - 1.0
        desired = pd.Series(0.10, index=index).where(
            high & opportunity_gain.ge(0.05) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "native_reentry_accelerator":
        native_reentry = defense.shift(5) - defense
        desired = pd.Series(0.05, index=index).where(
            defense.ge(0.30) & native_reentry.ge(0.05) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "health_score_proportional_relief":
        score = state["health_score"]
        desired = ((score - 2.0).clip(lower=0.0) * 0.02).clip(
            upper=0.10
        ).where(high & no_break, 0.0)
        destination = "existing"
    elif mechanism == "ramp_duration_composite":
        five_day_increase = defense - defense.shift(5)
        ramp = (five_day_increase - 0.10).clip(
            lower=0.0, upper=MAX_RELIEF
        )
        decay = pd.Series(0.10, index=index).where(duration.ge(20), 0.0)
        desired = pd.concat([ramp, decay], axis=1).max(axis=1).where(
            high & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "breadth_break_veto_relief":
        desired = fixed.where(state["breadth_intact"].astype(bool) & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_dual_trend_relief":
        intact = state["breadth_intact"].astype(bool) & state[
            "dual_trend_intact"
        ].astype(bool)
        desired = fixed.where(intact & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_shallow_drawdown_relief":
        intact = state["breadth_intact"].astype(bool) & state[
            "shallow_drawdown"
        ].astype(bool)
        desired = fixed.where(intact & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_low_volatility_relief":
        intact = state["breadth_intact"].astype(bool) & state[
            "low_volatility"
        ].astype(bool)
        desired = fixed.where(intact & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_positive_momentum_relief":
        intact = state["breadth_intact"].astype(bool) & state[
            "positive_momentum"
        ].astype(bool)
        desired = fixed.where(intact & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_credit_volatility_relief":
        intact = (
            state["breadth_intact"].astype(bool)
            & state["credit_intact"].astype(bool)
            & state["volatility_intact"].astype(bool)
        )
        desired = fixed.where(intact & no_break, 0.0)
        destination = "existing"
    elif mechanism == "breadth_duration_confirmed_relief":
        desired = pd.Series(0.10, index=index).where(
            high & duration.ge(20) & state["breadth_intact"].astype(bool) & no_break,
            0.0,
        )
        destination = "existing"
    elif mechanism == "breadth_splv_bridge":
        desired = fixed.where(
            state["breadth_intact"].astype(bool)
            & state["splv_available"].astype(bool)
            & no_break,
            0.0,
        )
        destination = "SPLV"
    elif mechanism == "breadth_split_existing_splv":
        intact = (
            high
            & state["breadth_intact"].astype(bool)
            & state["splv_available"].astype(bool)
            & no_break
        )
        half = pd.Series(0.025, index=index).where(intact, 0.0)
        native_weights, native_relief = apply_existing_sleeve_relief(
            base_weights,
            half,
        )
        adjusted, splv_relief = apply_satellite_bridge(
            native_weights,
            half,
            destination="SPLV",
            price_available=prices["SPLV"].notna().reindex(index).fillna(False),
        )
        return adjusted, native_relief + splv_relief
    else:
        break_count = state["break_count"]
        breadth = state["breadth_intact"].astype(bool)
        native_desired = pd.Series(0.05, index=index).where(
            high & breadth & break_count.eq(0),
            0.0,
        )
        splv_desired = pd.Series(0.05, index=index).where(
            high
            & breadth
            & break_count.eq(1)
            & state["splv_available"].astype(bool),
            0.0,
        )
        native_weights, native_relief = apply_existing_sleeve_relief(
            base_weights,
            native_desired,
        )
        adjusted, splv_relief = apply_satellite_bridge(
            native_weights,
            splv_desired,
            destination="SPLV",
            price_available=prices["SPLV"].notna().reindex(index).fillna(False),
        )
        return adjusted, native_relief + splv_relief

    desired = pd.to_numeric(desired, errors="coerce").fillna(0.0).clip(
        lower=0.0,
        upper=MAX_RELIEF,
    )
    if destination == "existing":
        return apply_existing_sleeve_relief(base_weights, desired)
    return apply_satellite_bridge(
        base_weights,
        desired,
        destination=destination,
        price_available=prices[destination].notna().reindex(index).fillna(False),
    )


def apply_existing_sleeve_relief(
    base_weights: pd.DataFrame,
    desired_relief: pd.Series,
    *,
    defensive_ticker: str = DEFENSIVE_TICKER,
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = base_weights.astype(float).clip(lower=0.0).copy()
    if defensive_ticker not in adjusted:
        adjusted[defensive_ticker] = 0.0
    risk_columns = [
        column for column in adjusted.columns if column != defensive_ticker
    ]
    risk = adjusted[risk_columns].sum(axis=1)
    defense = effective_defensive_weight_path(
        adjusted,
        defensive_ticker=defensive_ticker,
    )
    actual = pd.concat(
        [
            desired_relief.reindex(adjusted.index).fillna(0.0),
            defense,
        ],
        axis=1,
    ).min(axis=1)
    actual = actual.where(risk.gt(1e-12), 0.0)
    scale = ((risk + actual) / risk.where(risk.gt(1e-12))).fillna(1.0)
    adjusted[risk_columns] = adjusted[risk_columns].mul(scale, axis=0)
    bil_reduction = pd.concat(
        [adjusted[defensive_ticker], actual],
        axis=1,
    ).min(axis=1)
    adjusted[defensive_ticker] = adjusted[defensive_ticker] - bil_reduction
    return adjusted, actual


def apply_satellite_bridge(
    base_weights: pd.DataFrame,
    desired_relief: pd.Series,
    *,
    destination: str,
    price_available: pd.Series,
    defensive_ticker: str = DEFENSIVE_TICKER,
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = base_weights.astype(float).clip(lower=0.0).copy()
    if defensive_ticker not in adjusted:
        adjusted[defensive_ticker] = 0.0
    if destination not in adjusted:
        adjusted[destination] = 0.0
    defense = effective_defensive_weight_path(
        adjusted,
        defensive_ticker=defensive_ticker,
    )
    requested = desired_relief.reindex(adjusted.index).fillna(0.0).where(
        price_available.reindex(adjusted.index).fillna(False),
        0.0,
    )
    actual = pd.concat([requested, defense], axis=1).min(axis=1)
    bil_reduction = pd.concat(
        [adjusted[defensive_ticker], actual],
        axis=1,
    ).min(axis=1)
    adjusted[defensive_ticker] = adjusted[defensive_ticker] - bil_reduction
    adjusted[destination] = adjusted[destination] + actual
    return adjusted, actual


def consecutive_true_count(values: pd.Series) -> pd.Series:
    clean = values.fillna(False).astype(bool)
    groups = (~clean).cumsum()
    counts = clean.astype(int).groupby(groups).cumsum()
    return counts.astype(int)


def build_era_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        for mechanism in MECHANISM_ROSTER:
            candidate = adjusted_results[(strategy, mechanism)]
            for era, start, end in ERA_WINDOWS:
                base_stats = _return_slice_stats(base.returns, start, end)
                candidate_stats = _return_slice_stats(candidate.returns, start, end)
                if base_stats["observations"] == 0:
                    continue
                rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "mechanism": mechanism,
                        "era": era,
                        "annualized_return_delta": candidate_stats[
                            "annualized_return"
                        ]
                        - base_stats["annualized_return"],
                        "max_drawdown_delta": candidate_stats["max_drawdown"]
                        - base_stats["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows)


def build_ordinary_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
) -> pd.DataFrame:
    """Measure candidate behavior after removing every named crisis session."""

    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        ordinary = ordinary_session_mask(base.returns.index)
        base_stats = _selected_return_stats(base.returns.loc[ordinary])
        for mechanism in MECHANISM_ROSTER:
            candidate = adjusted_results[(strategy, mechanism)]
            candidate_stats = _selected_return_stats(candidate.returns.loc[ordinary])
            rows.append(
                {
                    "strategy": strategy,
                    "family": family,
                    "mechanism": mechanism,
                    "observations": base_stats["observations"],
                    "annualized_return_delta": candidate_stats["annualized_return"]
                    - base_stats["annualized_return"],
                    "max_drawdown_delta": candidate_stats["max_drawdown"]
                    - base_stats["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def ordinary_session_mask(index: pd.Index) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    crisis = np.zeros(len(dates), dtype=bool)
    for _name, start, end in CRISIS_WINDOWS:
        crisis |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return pd.Series(~crisis, index=index, dtype=bool)


def build_rolling_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
    *,
    horizons: tuple[int, ...] = (252, 756),
    step: int = 63,
) -> pd.DataFrame:
    """Summarize overlapping one- and three-year windows sampled quarterly."""

    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy].returns.dropna()
        for mechanism in MECHANISM_ROSTER:
            candidate = adjusted_results[(strategy, mechanism)].returns.reindex(
                base.index
            )
            for horizon in horizons:
                return_deltas: list[float] = []
                drawdown_deltas: list[float] = []
                for end in range(horizon, len(base) + 1, step):
                    base_stats = _selected_return_stats(base.iloc[end - horizon : end])
                    candidate_stats = _selected_return_stats(
                        candidate.iloc[end - horizon : end]
                    )
                    return_deltas.append(
                        float(candidate_stats["cumulative_return"])
                        - float(base_stats["cumulative_return"])
                    )
                    drawdown_deltas.append(
                        float(candidate_stats["max_drawdown"])
                        - float(base_stats["max_drawdown"])
                    )
                rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "mechanism": mechanism,
                        "horizon_sessions": horizon,
                        "sample_step_sessions": step,
                        "windows": len(return_deltas),
                        "positive_return_delta_rate": float(
                            np.mean(np.asarray(return_deltas) > 0.0)
                        ),
                        "median_return_delta": float(np.median(return_deltas)),
                        "return_delta_p10": float(
                            np.quantile(return_deltas, 0.10)
                        ),
                        "return_delta_p90": float(
                            np.quantile(return_deltas, 0.90)
                        ),
                        "nonworse_drawdown_rate": float(
                            np.mean(np.asarray(drawdown_deltas) >= -1e-10)
                        ),
                        "median_max_drawdown_delta": float(
                            np.median(drawdown_deltas)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _selected_return_stats(returns: pd.Series) -> dict[str, float | int]:
    selected = returns.dropna()
    if selected.empty:
        return {
            "observations": 0,
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
        }
    equity = (1.0 + selected).cumprod()
    return {
        "observations": int(len(selected)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(equity.iloc[-1] ** (252.0 / len(selected)) - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }


def build_crisis_metrics(
    baseline_run: BaselineRun,
    adjusted_results: dict[tuple[str, str], BacktestResult],
    strategy_families: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        for mechanism in MECHANISM_ROSTER:
            candidate = adjusted_results[(strategy, mechanism)]
            for crisis, start, end in CRISIS_WINDOWS:
                base_stats = _return_slice_stats(base.returns, start, end)
                candidate_stats = _return_slice_stats(candidate.returns, start, end)
                if base_stats["observations"] == 0:
                    continue
                rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "mechanism": mechanism,
                        "crisis": crisis,
                        "return_delta": candidate_stats["cumulative_return"]
                        - base_stats["cumulative_return"],
                        "max_drawdown_delta": candidate_stats["max_drawdown"]
                        - base_stats["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows)


def build_focus_cost_sensitivity(
    baseline_run: BaselineRun,
    focus_strategy: str,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    family_median_defense: pd.Series,
) -> pd.DataFrame:
    base = baseline_run.results[focus_strategy]
    rows: list[dict[str, object]] = []
    for mechanism in MECHANISM_ROSTER:
        weights, _relief = build_mechanism_weight_path(
            mechanism,
            base.weights,
            prices,
            signals,
            family_median_defense=family_median_defense,
        )
        for cost_bps in (5.0, 10.0, 20.0):
            cost_base = _result_from_execution_weights(
                base,
                prices,
                base.weights,
                transaction_cost_bps=cost_bps,
                name=f"{focus_strategy}__base__{cost_bps:g}bps",
            )
            candidate = _result_from_execution_weights(
                base,
                prices,
                weights,
                transaction_cost_bps=cost_bps,
                name=f"{focus_strategy}__{mechanism}__{cost_bps:g}bps",
            )
            base_metrics = calculate_metrics(
                cost_base.name,
                cost_base.returns,
                cost_base.equity,
                cost_base.turnover,
                cost_base.transaction_costs,
            )
            candidate_metrics = calculate_metrics(
                candidate.name,
                candidate.returns,
                candidate.equity,
                candidate.turnover,
                candidate.transaction_costs,
            )
            rows.append(
                {
                    "mechanism": mechanism,
                    "cost_bps": cost_bps,
                    "cagr_delta_vs_base": candidate_metrics.cagr - base_metrics.cagr,
                    "max_drawdown_delta_vs_base": (
                        candidate_metrics.max_drawdown - base_metrics.max_drawdown
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_mechanism_scorecard(
    strategy_metrics: pd.DataFrame,
    era_metrics: pd.DataFrame,
    ordinary_metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    crisis_metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    current_effects: pd.DataFrame,
    *,
    focus_strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mechanism in MECHANISM_ROSTER:
        strategy = strategy_metrics[strategy_metrics["mechanism"].eq(mechanism)]
        focus = strategy[strategy["strategy"].eq(focus_strategy)].iloc[0]
        eras = era_metrics[era_metrics["mechanism"].eq(mechanism)]
        era_medians = eras.groupby("era")["annualized_return_delta"].median()
        ordinary = ordinary_metrics[
            ordinary_metrics["mechanism"].eq(mechanism)
        ]
        focus_ordinary = ordinary[
            ordinary["strategy"].eq(focus_strategy)
        ].iloc[0]
        rolling = rolling_metrics[
            rolling_metrics["mechanism"].eq(mechanism)
            & rolling_metrics["strategy"].eq(focus_strategy)
        ].set_index("horizon_sessions")
        crises = crisis_metrics[crisis_metrics["mechanism"].eq(mechanism)]
        costs = cost_sensitivity[cost_sensitivity["mechanism"].eq(mechanism)]
        current = current_effects[
            current_effects["mechanism"].eq(mechanism)
            & current_effects["strategy"].eq(focus_strategy)
        ].iloc[0]
        focus_cagr_positive = float(focus["cagr_delta_vs_base"]) > 0.0
        focus_drawdown_ok = float(focus["max_drawdown_delta_vs_base"]) >= -0.01
        strategy_positive_rate = float(strategy["cagr_delta_vs_base"].gt(0.0).mean())
        strategy_nonworse_dd_rate = float(
            strategy["max_drawdown_delta_vs_base"].ge(-1e-10).mean()
        )
        positive_era_rate = float(era_medians.gt(0.0).mean())
        crisis_nonworse_dd_rate = float(
            crises["max_drawdown_delta"].ge(-1e-10).mean()
        )
        material = bool(
            float(focus["active_day_rate"]) >= 0.05
            and float(focus["mean_active_defense_reduction"]) >= 0.025
        )
        high_cost = costs[costs["cost_bps"].isin([10.0, 20.0])]
        cost_robust = bool(
            not high_cost.empty
            and high_cost["cagr_delta_vs_base"].gt(0.0).all()
            and high_cost["max_drawdown_delta_vs_base"].ge(-0.01).all()
        )
        gate_values = {
            "focus_cagr_positive": focus_cagr_positive,
            "focus_drawdown_within_1pp": focus_drawdown_ok,
            "strategy_positive_rate_pass": strategy_positive_rate >= 0.60,
            "strategy_nonworse_dd_rate_pass": strategy_nonworse_dd_rate >= 0.60,
            "era_positive_rate_pass": positive_era_rate >= 0.75,
            "crisis_nonworse_dd_rate_pass": crisis_nonworse_dd_rate >= 0.75,
            "material_allocation_effect": material,
            "higher_cost_robust": cost_robust,
        }
        failed_gates = [name for name, passed in gate_values.items() if not passed]
        rows.append(
            {
                "mechanism": mechanism,
                "description": MECHANISM_DESCRIPTIONS[mechanism],
                "focus_cagr_delta": focus["cagr_delta_vs_base"],
                "focus_max_drawdown_delta": focus["max_drawdown_delta_vs_base"],
                "focus_active_day_rate": focus["active_day_rate"],
                "focus_mean_active_defense_reduction": focus[
                    "mean_active_defense_reduction"
                ],
                "focus_current_defense_reduction": current[
                    "current_defense_reduction"
                ],
                "strategy_positive_cagr_rate": strategy_positive_rate,
                "strategy_nonworse_drawdown_rate": strategy_nonworse_dd_rate,
                "positive_era_rate": positive_era_rate,
                "focus_ordinary_annualized_return_delta": focus_ordinary[
                    "annualized_return_delta"
                ],
                "focus_ordinary_max_drawdown_delta": focus_ordinary[
                    "max_drawdown_delta"
                ],
                "ordinary_strategy_positive_return_rate": float(
                    ordinary["annualized_return_delta"].gt(0.0).mean()
                ),
                "ordinary_strategy_nonworse_drawdown_rate": float(
                    ordinary["max_drawdown_delta"].ge(-1e-10).mean()
                ),
                "focus_rolling_1y_positive_return_rate": rolling.loc[
                    252, "positive_return_delta_rate"
                ],
                "focus_rolling_1y_nonworse_drawdown_rate": rolling.loc[
                    252, "nonworse_drawdown_rate"
                ],
                "focus_rolling_3y_positive_return_rate": rolling.loc[
                    756, "positive_return_delta_rate"
                ],
                "focus_rolling_3y_nonworse_drawdown_rate": rolling.loc[
                    756, "nonworse_drawdown_rate"
                ],
                "crisis_nonworse_drawdown_rate": crisis_nonworse_dd_rate,
                **gate_values,
                "gate_pass_count": sum(gate_values.values()),
                "gate_fail_count": len(failed_gates),
                "failed_gates": ", ".join(failed_gates),
                "retrospective_gate_passed": all(gate_values.values()),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "retrospective_gate_passed",
            "gate_fail_count",
            "focus_cagr_delta",
            "strategy_positive_cagr_rate",
        ],
        ascending=[False, True, False, False],
    )


def write_search_outputs(
    *,
    output_dir: str | Path,
    config: BotConfig,
    prices: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    summary = root / "summary.md"
    summary.write_text(build_search_summary(frames) + "\n", encoding="utf-8")
    paths["summary"] = summary
    manifest = write_research_manifest(
        root,
        study="defensive_correction_architecture_search",
        config=config,
        prices=prices,
        parameters={
            "architecture_count": len(MECHANISM_ROSTER),
            "mechanisms": list(MECHANISM_ROSTER),
            "threshold_variants_count_as_architectures": False,
            "point_in_time_market_inputs": "prior_close_only",
            "hard_portfolio_constraints": "excluded",
            "news_events_scenarios_macro": "excluded",
            "automatic_promotion_allowed": False,
            "allocation_authority": 0.0,
            "trial_roster": list(MECHANISM_ROSTER),
        },
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest
    return paths


def build_search_summary(frames: dict[str, pd.DataFrame]) -> str:
    scorecard = frames["mechanism_scorecard"]
    passed = scorecard[scorecard["retrospective_gate_passed"].astype(bool)]
    top = scorecard.iloc[0]
    return "\n".join(
        [
            "# Defensive Correction Architecture Search",
            "",
            "Status: retrospective research only; allocation authority is 0%.",
            "",
            f"- Distinct architectures tested: {len(scorecard):,}.",
            f"- Full retrospective passes: {len(passed):,}.",
            f"- Closest mechanism: `{top['mechanism']}`.",
            f"- Gates passed: {int(top['gate_pass_count'])}/8.",
            f"- Failed gates: {top['failed_gates']}.",
            f"- Focus CAGR delta: {float(top['focus_cagr_delta']):.2%}.",
            f"- Focus max-drawdown delta: {float(top['focus_max_drawdown_delta']):.2%}.",
            f"- Focus active-day rate: {float(top['focus_active_day_rate']):.2%}.",
            f"- Mean active defense reduction: {float(top['focus_mean_active_defense_reduction']):.2%}.",
            f"- Current defense reduction: {float(top['focus_current_defense_reduction']):.2%}.",
            f"- Crisis-excluded focus annualized-return delta: {float(top['focus_ordinary_annualized_return_delta']):.2%}.",
            f"- Crisis-excluded strategy positive-return rate: {float(top['ordinary_strategy_positive_return_rate']):.2%}.",
            f"- Focus rolling 1Y positive-window rate: {float(top['focus_rolling_1y_positive_return_rate']):.2%}.",
            f"- Focus rolling 3Y positive-window rate: {float(top['focus_rolling_3y_positive_return_rate']):.2%}.",
            "",
            (
                "At least one architecture merits deeper adversarial testing."
                if not passed.empty
                else "No architecture cleared the full retrospective gate."
            ),
        ]
    )
