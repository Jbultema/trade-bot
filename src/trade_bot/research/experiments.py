from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics, metrics_frame
from trade_bot.backtest.windows import (
    regime_window_metrics,
    rolling_window_metrics,
    summarize_regimes,
    summarize_walk_forward,
    summarize_windows,
    walk_forward_holdout_metrics,
)
from trade_bot.config import (
    BotConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    configured_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULT import (
    DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS,
    DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER,
    DEFAULT_SCENARIO_MAX_MULTIPLIER,
    DEFAULT_SCENARIO_MIN_MULTIPLIER,
    DEFAULT_SCENARIO_RISK_ON_MULTIPLIER,
    DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS,
    DEFAULT_SCENARIO_STRESS_MULTIPLIER,
    DEFAULT_SCENARIO_TRANSITION_MULTIPLIER,
)
from trade_bot.research.curation import add_research_status
from trade_bot.research.future_state_ml import (
    FutureStateModelConfig,
    StrategyDrawdownModelConfig,
    apply_future_state_position_sizing,
    apply_strategy_drawdown_position_sizing,
)
from trade_bot.research.strategy_naming import strategy_display_name
from trade_bot.strategies.momentum import build_strategy_weights


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    hypothesis: str
    role: str
    strategy: StrategyConfig
    scenario_sizing: ScenarioSizingConfig | None = None
    future_state_model: FutureStateModelConfig | None = None
    strategy_drawdown_model: StrategyDrawdownModelConfig | None = None
    decision_sanity: DecisionSanityConfig | None = None
    phase: str = "broad"
    family: str = "general"
    parent: str | None = None


@dataclass(frozen=True)
class ScenarioSizingConfig:
    profile: str
    stress_multiplier: float = DEFAULT_SCENARIO_STRESS_MULTIPLIER
    transition_multiplier: float = DEFAULT_SCENARIO_TRANSITION_MULTIPLIER
    fragile_upside_multiplier: float = DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER
    risk_on_multiplier: float = DEFAULT_SCENARIO_RISK_ON_MULTIPLIER
    min_multiplier: float = DEFAULT_SCENARIO_MIN_MULTIPLIER
    max_multiplier: float = DEFAULT_SCENARIO_MAX_MULTIPLIER
    lookback_days: int = DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS


@dataclass(frozen=True)
class DecisionSanityConfig:
    profile: str = "confirmation_cap"
    max_defensive_add: float = DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD
    required_confirmation_breaks: int = DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS
    confirmation_threshold: float = 0.25
    left_tail_pressure_threshold: float = 0.35
    lookback_days: int = DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS


@dataclass(frozen=True)
class ExperimentBatchRun:
    iteration: int
    prices: pd.DataFrame
    candidates: tuple[ExperimentCandidate, ...]
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    window_summary: pd.DataFrame
    regime_metrics: pd.DataFrame
    regime_summary: pd.DataFrame
    walk_forward_folds: pd.DataFrame
    walk_forward_summary: pd.DataFrame
    operability_metrics: pd.DataFrame
    transition_metrics: pd.DataFrame
    scorecard: pd.DataFrame


def run_experiment_iteration(
    config: BotConfig,
    *,
    iteration: int = 1,
    refresh_data: bool = False,
    output_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
) -> ExperimentBatchRun:
    previous_scorecards = _load_previous_scorecards(output_dir, iteration)
    previous_candidates = _load_previous_candidates(output_dir, iteration)
    candidates = generate_iteration_candidates(
        iteration,
        previous_scorecards=previous_scorecards,
        previous_candidates=previous_candidates,
    )
    tickers = sorted(set(configured_tickers(config)) | _candidate_tickers(candidates))
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )

    results: dict[str, BacktestResult] = {}
    calculated_metrics: list[PerformanceMetrics] = []
    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_target_weights = build_strategy_weights(candidate_prices, candidate.strategy)
        target_weights = base_target_weights
        if candidate.future_state_model is not None:
            target_weights = apply_future_state_position_sizing(
                target_weights,
                prices,
                candidate.future_state_model,
                defensive_ticker=candidate.strategy.defensive_ticker,
            )
        if candidate.scenario_sizing is not None:
            target_weights = apply_scenario_position_sizing(
                target_weights,
                candidate_prices,
                candidate.scenario_sizing,
                defensive_ticker=candidate.strategy.defensive_ticker,
            )
        if candidate.strategy_drawdown_model is not None:
            target_weights = apply_strategy_drawdown_position_sizing(
                target_weights,
                prices,
                candidate.strategy_drawdown_model,
                defensive_ticker=candidate.strategy.defensive_ticker,
            )
        if candidate.decision_sanity is not None:
            target_weights = apply_decision_sanity_overlay(
                base_target_weights,
                target_weights,
                candidate_prices,
                candidate.decision_sanity,
                defensive_ticker=candidate.strategy.defensive_ticker,
            )
        target_weights = apply_operability_hysteresis(target_weights, candidate.strategy)
        result = run_backtest(
            candidate.name,
            candidate_prices,
            target_weights,
            config.execution,
            volatility_target=candidate.strategy.volatility_target,
            drawdown_control=candidate.strategy.drawdown_control,
        )
        results[candidate.name] = result
        calculated_metrics.append(
            calculate_metrics(
                name=result.name,
                returns=result.returns,
                equity=result.equity,
                turnover=result.turnover,
                transaction_costs=result.transaction_costs,
            )
        )

    metrics = metrics_frame(calculated_metrics)
    window_summary = summarize_windows(rolling_window_metrics(results))
    regime_metrics = regime_window_metrics(results)
    regime_summary = summarize_regimes(regime_metrics)
    walk_forward_folds = walk_forward_holdout_metrics(results)
    walk_forward_summary = summarize_walk_forward(walk_forward_folds)
    operability_metrics = _operability_metrics_frame(results)
    transition_metrics = _transition_metrics_frame(candidates, results)
    benchmark_metrics = _benchmark_metrics(prices, config.execution)
    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        regime_summary=regime_summary,
        walk_forward_summary=walk_forward_summary,
        benchmark_metrics=benchmark_metrics,
        operability_metrics=operability_metrics,
        transition_metrics=transition_metrics,
    )
    _write_experiment_outputs(
        iteration,
        candidates,
        scorecard,
        metrics,
        window_summary,
        regime_metrics,
        regime_summary,
        walk_forward_folds,
        walk_forward_summary,
        operability_metrics,
        transition_metrics,
        output_dir,
    )

    return ExperimentBatchRun(
        iteration=iteration,
        prices=prices,
        candidates=candidates,
        results=results,
        metrics=metrics.sort_values("calmar", ascending=False),
        window_summary=window_summary,
        regime_metrics=regime_metrics,
        regime_summary=regime_summary,
        walk_forward_folds=walk_forward_folds,
        walk_forward_summary=walk_forward_summary,
        operability_metrics=operability_metrics,
        transition_metrics=transition_metrics,
        scorecard=scorecard,
    )


def generate_iteration_candidates(
    iteration: int,
    *,
    previous_scorecards: pd.DataFrame | None = None,
    previous_candidates: pd.DataFrame | None = None,
) -> tuple[ExperimentCandidate, ...]:
    preset = _preset_iteration_candidates(iteration)
    if preset is not None:
        return preset

    evolved = _evolve_from_previous_iteration(
        iteration,
        previous_scorecards=previous_scorecards,
        previous_candidates=previous_candidates,
    )
    if evolved:
        return tuple(evolved[:10])

    fallback_iteration = ((iteration - 1) % 3) + 1
    fallback = _preset_iteration_candidates(fallback_iteration)
    if fallback is None:
        msg = f"Could not generate experiment candidates for iteration {iteration}."
        raise ValueError(msg)
    return _retag_candidates(fallback, iteration)


def _preset_iteration_candidates(iteration: int) -> tuple[ExperimentCandidate, ...] | None:
    if iteration == 1:
        return _broad_core_candidates()
    if iteration == 2:
        return _broad_risk_weighting_candidates()
    if iteration == 3:
        return _broad_scenario_proxy_candidates()
    if iteration == 21:
        return _operating_system_candidates()
    if iteration == 41:
        return _reference_portfolio_candidates()
    if 42 <= iteration <= 49:
        return _active_trading_candidates(iteration)
    if 50 <= iteration <= 54:
        return _final_deep_wide_candidates(iteration)
    if 55 <= iteration <= 60:
        return _dip_reentry_candidates(iteration)
    if 61 <= iteration <= 65:
        return _dip_reentry_overlay_candidates(iteration)
    if 66 <= iteration <= 71:
        return _ai_risk_cycle_candidates(iteration)
    if 72 <= iteration <= 76:
        return _sector_regime_rotation_candidates(iteration)
    if iteration == 77:
        return _decision_sanity_overlay_candidates()
    if iteration == 78:
        return _decision_sanity_tuning_candidates()
    if iteration == 79:
        return _paper_readiness_tuning_candidates()
    if iteration == 80:
        return _operability_gauntlet_candidates()
    if iteration == 81:
        return _future_state_ml_candidates()
    if iteration == 82:
        return _bayesian_future_state_candidates()
    if iteration == 83:
        return _sklearn_future_state_candidates()
    if iteration == 84:
        return _high_cagr_ml_guardrail_candidates()
    if iteration == 85:
        return _strategy_drawdown_ml_guardrail_candidates()
    if iteration == 86:
        return _aggressive_drawdown_ml_hybrid_candidates()
    if 101 <= iteration <= 105:
        return _macro_reset_candidates(iteration)
    return None


def _broad_core_candidates() -> tuple[ExperimentCandidate, ...]:
    return (
        _candidate(
            name="i01_fast_dual_core",
            role="candidate_core",
            phase="broad",
            family="core_cross_asset",
            hypothesis="Faster 3-month dual momentum may adapt quicker to regime turns.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                lookback_days=63,
                skip_days=5,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_slow_dual_core",
            role="candidate_core",
            phase="broad",
            family="core_cross_asset",
            hypothesis="Slower 12-month dual momentum may reduce whipsaw and turnover.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                lookback_days=252,
                skip_days=21,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_single_winner_dual",
            role="candidate_core",
            phase="broad",
            family="core_cross_asset",
            hypothesis="Concentrating in the strongest cross-asset winner may improve CAGR at acceptable risk.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                lookback_days=126,
                skip_days=21,
                top_n=1,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_diversified_cross_asset_dual",
            role="candidate_core",
            phase="broad",
            family="core_cross_asset",
            hypothesis="A broader cross-asset set may reduce dependence on QQQ and improve transition behavior.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "RSP", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_sector_plus_defense_rotation",
            role="candidate_core",
            phase="broad",
            family="sector_rotation",
            hypothesis="Sector rotation with defensive assets may capture leadership while preserving off-ramps.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                    "GLD",
                    "TLT",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_ai_beta_with_escape",
            role="satellite",
            phase="broad",
            family="ai_beta",
            hypothesis="AI leadership can be traded as a satellite only with strict absolute momentum escape.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.02,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.16,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
            ),
        ),
        _candidate(
            name="i01_quality_factor_rotation",
            role="candidate_core",
            phase="broad",
            family="factor_rotation",
            hypothesis="Quality/value/low-vol factors may survive regime shifts better than sector-only rotation.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "MTUM",
                    "VTV",
                    "VUG",
                    "SCHD",
                    "COWZ",
                    "MOAT",
                    "SPMO",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
            ),
        ),
        _candidate(
            name="i01_low_turnover_absolute_trend",
            role="overlay_candidate",
            phase="broad",
            family="trend_following",
            hypothesis="Simple absolute trend across broad assets may be a robust low-turnover operating system.",
            strategy=StrategyConfig(
                type="absolute_momentum",
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT"],
                moving_average_days=200,
                defensive_ticker="BIL",
            ),
        ),
        _candidate(
            name="i01_vol_target_cross_asset",
            role="overlay_candidate",
            phase="broad",
            family="risk_control",
            hypothesis="Volatility targeting may reduce left-tail risk without needing many asset switches.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT", "IEF"],
                lookback_days=126,
                skip_days=21,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
            ),
        ),
        _candidate(
            name="i01_tight_drawdown_cross_asset",
            role="overlay_candidate",
            phase="broad",
            family="risk_control",
            hypothesis="A tighter drawdown throttle may improve survival even if it gives up upside.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                lookback_days=126,
                skip_days=21,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
                drawdown_control=DrawdownControlConfig(
                    equity_lookback_days=126,
                    max_drawdown=-0.07,
                    risk_multiplier=0.25,
                ),
            ),
        ),
    )


def _broad_risk_weighting_candidates() -> tuple[ExperimentCandidate, ...]:
    return (
        _candidate(
            name="i02_cross_asset_risk_adjusted",
            role="candidate_core",
            phase="broad",
            family="risk_adjusted_momentum",
            hypothesis="Ranking by return per unit of realized volatility may reduce transition fragility.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "RSP", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
            ),
        ),
        _candidate(
            name="i02_cross_asset_inverse_vol",
            role="candidate_core",
            phase="broad",
            family="risk_adjusted_momentum",
            hypothesis="Equal selection with inverse-vol sizing may keep winners while muting left-tail assets.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "RSP", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                weighting="inverse_volatility",
            ),
        ),
        _candidate(
            name="i02_trend_confirmed_core",
            role="candidate_core",
            phase="broad",
            family="off_ramp",
            hypothesis="Requiring selected assets to remain above a 200-day trend may improve off-ramp timing.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT", "IEF"],
                lookback_days=126,
                skip_days=21,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.0,
                trend_filter_days=200,
            ),
        ),
        _candidate(
            name="i02_capped_single_winner",
            role="candidate_core",
            phase="broad",
            family="position_sizing",
            hypothesis="Single-winner momentum may need a hard asset cap with residual T-bill exposure.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                lookback_days=126,
                skip_days=21,
                top_n=1,
                defensive_ticker="BIL",
                min_return=0.0,
                max_asset_weight=0.65,
            ),
        ),
        _candidate(
            name="i02_factor_risk_adjusted",
            role="candidate_core",
            phase="broad",
            family="factor_rotation",
            hypothesis="Risk-adjusted factor rotation may avoid crowded momentum crashes better than raw returns.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "MTUM",
                    "VTV",
                    "VUG",
                    "SCHD",
                    "COWZ",
                    "MOAT",
                    "SPMO",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                max_asset_weight=0.4,
            ),
        ),
        _candidate(
            name="i02_sector_inverse_vol",
            role="candidate_core",
            phase="broad",
            family="sector_rotation",
            hypothesis="Sector rotation may be more tradable when volatile sectors receive smaller sizes.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                weighting="inverse_volatility",
            ),
        ),
        _candidate(
            name="i02_global_escape",
            role="satellite",
            phase="broad",
            family="global_rotation",
            hypothesis="Global equity leadership can be held only while absolute and trend confirmation agree.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.01,
                trend_filter_days=200,
                max_asset_weight=0.4,
            ),
        ),
        _candidate(
            name="i02_credit_rates_barbell",
            role="overlay_candidate",
            phase="broad",
            family="credit_rates",
            hypothesis="Credit and duration rotation can identify risk appetite shifts without relying on QQQ.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "IEF", "TLT", "TIP", "GLD"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                weighting="inverse_volatility",
            ),
        ),
        _candidate(
            name="i02_commodity_shock_rotation",
            role="satellite",
            phase="broad",
            family="commodity_shock",
            hypothesis="Commodity and dollar leadership may become the right hedge during inflationary transitions.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["GLD", "IAU", "SLV", "CPER", "USO", "BNO", "DBC", "DBA", "UUP", "TLT"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                max_asset_weight=0.4,
            ),
        ),
        _candidate(
            name="i02_ai_infra_risk_adjusted",
            role="satellite",
            phase="broad",
            family="ai_infrastructure",
            hypothesis="AI infrastructure winners may be safer than AI-beta when sized by risk-adjusted momentum.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX", "QQQ"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.16,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
            ),
        ),
    )


def _broad_scenario_proxy_candidates() -> tuple[ExperimentCandidate, ...]:
    return (
        _candidate(
            name="i03_ai_beta_bubble_escape",
            role="satellite",
            phase="broad",
            family="ai_beta",
            hypothesis="AI beta is only acceptable when strong momentum, trend, and capped sizing all agree.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.03,
                trend_filter_days=100,
                max_asset_weight=0.35,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
        ),
        _candidate(
            name="i03_ai_capex_infra_rotation",
            role="satellite",
            phase="broad",
            family="ai_infrastructure",
            hypothesis="AI capex concern may rotate into power, grid, and infrastructure rather than pure software.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "IGV", "XLI"],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.02,
                weighting="inverse_volatility",
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i03_private_credit_early_stress",
            role="overlay_candidate",
            phase="broad",
            family="private_credit",
            hypothesis="BDC/loan/credit proxy weakness may act as an early off-ramp for equity risk.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["BIZD", "SRLN", "BKLN", "JAAA", "JBBB", "HYG", "LQD", "KRE", "KBE", "IEF"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
            ),
        ),
        _candidate(
            name="i03_policy_whipsaw_barbell",
            role="overlay_candidate",
            phase="broad",
            family="policy_whipsaw",
            hypothesis="Policy/headline whipsaw may favor a barbell of trend equities, gold, duration, and cash.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "RSP", "GLD", "TLT", "IEF", "UUP", "USO", "BNO"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i03_oil_shock_basket",
            role="satellite",
            phase="broad",
            family="oil_shock",
            hypothesis="Oil-shock regimes should show up as energy, commodity, and dollar leadership quickly.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["XLE", "XOP", "OIH", "USO", "BNO", "DBC", "UUP", "GLD", "TLT"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.01,
                weighting="inverse_volatility",
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i03_quality_lowvol_defensive",
            role="candidate_core",
            phase="broad",
            family="defensive_equity",
            hypothesis="Quality, dividends, and low-volatility equities may keep risk-on exposure during late-cycle chop.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "SCHD",
                    "VIG",
                    "MOAT",
                    "COWZ",
                    "XLV",
                    "XLP",
                    "XLU",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
            ),
        ),
        _candidate(
            name="i03_mega_cap_platform_cap",
            role="satellite",
            phase="broad",
            family="mega_cap_platform",
            hypothesis="Mega-cap platform leadership can be tested without letting one crowded name dominate.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "AAPL",
                    "MSFT",
                    "NVDA",
                    "GOOGL",
                    "AMZN",
                    "META",
                    "AVGO",
                    "TSLA",
                    "BRK-B",
                    "JPM",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.02,
                max_asset_weight=0.3,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
            ),
        ),
        _candidate(
            name="i03_crypto_proxy_escape",
            role="wild_probe",
            phase="broad",
            family="crypto_liquidity",
            hypothesis="Crypto ETFs may proxy speculative liquidity but require strict trend and size gates.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["IBIT", "FBTC", "BITB", "ETHE", "QQQ", "GLD", "TLT"],
                lookback_days=42,
                skip_days=5,
                top_n=2,
                defensive_ticker="BIL",
                min_return=0.04,
                trend_filter_days=100,
                max_asset_weight=0.25,
            ),
        ),
        _candidate(
            name="i03_small_value_reflation",
            role="candidate_core",
            phase="broad",
            family="reflation_rotation",
            hypothesis="Small-cap, value, banks, industrials, and materials should be tested as a broadening/reflation path.",
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["IWM", "RSP", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "DBC", "GLD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                weighting="inverse_volatility",
            ),
        ),
        _candidate(
            name="i03_cash_gold_duration_trend",
            role="overlay_candidate",
            phase="broad",
            family="defensive_barbell",
            hypothesis="A defensive trend sleeve can become the off-ramp when equities lose trend support.",
            strategy=StrategyConfig(
                type="absolute_momentum",
                tickers=["GLD", "IAU", "TLT", "IEF", "SHY", "TIP", "UUP"],
                moving_average_days=126,
                defensive_ticker="BIL",
            ),
        ),
    )


def _active_trading_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    batches = {
        42: (
            _active_dual_candidate(
                name="i42_active_fast_core_single",
                family="active_cross_asset",
                hypothesis=(
                    "Daily single-winner cross-asset rotation tests whether very fast leadership "
                    "changes can beat slower weekly systems after next-day lag and costs."
                ),
                tickers=["SPY", "QQQ", "IWM", "RSP", "GLD", "TLT", "IEF", "DBC", "UUP"],
                lookback_days=21,
                skip_days=0,
                top_n=1,
                trend_filter_days=42,
                max_asset_weight=0.65,
            ),
            _active_dual_candidate(
                name="i42_active_fast_core_pair_invvol",
                family="active_cross_asset",
                hypothesis=(
                    "A two-asset fast cross-asset sleeve may keep responsiveness while reducing "
                    "the whipsaw of single-winner rotation."
                ),
                tickers=["SPY", "QQQ", "IWM", "RSP", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=30,
                skip_days=2,
                top_n=2,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.50,
            ),
            _active_dual_candidate(
                name="i42_active_fast_sector_top3",
                family="active_sector_rotation",
                hypothesis=(
                    "Short-horizon sector leadership may catch transitions before broad indexes "
                    "confirm the move."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                    "GLD",
                    "TLT",
                ],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.40,
            ),
            _active_dual_candidate(
                name="i42_active_fast_factor_top2",
                family="active_factor_rotation",
                hypothesis=(
                    "Fast factor rotation tests whether momentum, quality, value, dividend, and "
                    "low-volatility leadership changes are more stable than single sectors."
                ),
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "MTUM",
                    "VTV",
                    "VUG",
                    "SCHD",
                    "VIG",
                    "COWZ",
                    "SPMO",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=2,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=63,
                max_asset_weight=0.45,
            ),
            _active_dual_candidate(
                name="i42_active_policy_barbell",
                family="active_policy_shock",
                hypothesis=(
                    "A fast policy-shock barbell can move among equities, gold, duration, energy, "
                    "commodities, and dollar strength without forcing equity exposure."
                ),
                tickers=["SPY", "QQQ", "RSP", "GLD", "TLT", "IEF", "USO", "BNO", "DBC", "UUP"],
                lookback_days=21,
                skip_days=2,
                top_n=3,
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_absolute_candidate(
                name="i42_active_fast_absolute_trend",
                family="active_trend_following",
                hypothesis=(
                    "A 42-day absolute trend basket is the simple active off-ramp benchmark for "
                    "daily monitoring."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                moving_average_days=42,
            ),
        ),
        43: (
            _active_dual_candidate(
                name="i43_active_ai_beta_sprint",
                family="active_ai_beta",
                hypothesis=(
                    "AI beta may still be tradable if treated as a fast satellite with strict "
                    "trend, return, volatility, and concentration gates."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                    "ARM",
                ],
                lookback_days=21,
                skip_days=0,
                top_n=2,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=42,
                max_asset_weight=0.35,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.16,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
            _active_dual_candidate(
                name="i43_active_ai_infra_switch",
                family="active_ai_infrastructure",
                hypothesis=(
                    "If AI software economics sour but capex continues, power, grid, and hardware "
                    "beneficiaries may become the active expression."
                ),
                tickers=[
                    "VRT",
                    "ETN",
                    "PWR",
                    "CEG",
                    "GEV",
                    "NRG",
                    "CCJ",
                    "SMH",
                    "SOXX",
                    "XLI",
                    "XLU",
                ],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i43_active_hardware_software_switch",
                family="active_ai_beta",
                hypothesis=(
                    "A hardware/software switch tests whether semis, cloud, software, or broad QQQ "
                    "leadership is the better active AI proxy."
                ),
                tickers=[
                    "SMH",
                    "SOXX",
                    "IGV",
                    "SKYY",
                    "CLOU",
                    "QQQ",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "ORCL",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i43_active_mega_cap_escape",
                family="active_mega_cap_platform",
                hypothesis=(
                    "Mega-cap platform rotation can keep exposure to winners while escaping to cash "
                    "when platform leadership fails."
                ),
                tickers=[
                    "AAPL",
                    "MSFT",
                    "NVDA",
                    "GOOGL",
                    "AMZN",
                    "META",
                    "AVGO",
                    "TSLA",
                    "BRK-B",
                    "JPM",
                    "NFLX",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                min_return=0.015,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=63,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
            _active_dual_candidate(
                name="i43_active_spec_liquidity_proxy",
                family="active_speculative_liquidity",
                hypothesis=(
                    "Speculative liquidity can be tested as a small, strict satellite through crypto "
                    "and innovation proxies, but only with tight trend gates."
                ),
                tickers=["ARKK", "IBIT", "FBTC", "BITB", "ETHE", "TSLA", "QQQ", "GLD", "TLT"],
                lookback_days=21,
                skip_days=0,
                top_n=2,
                min_return=0.04,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=42,
                max_asset_weight=0.25,
            ),
            _active_dual_candidate(
                name="i43_active_ai_fragile_scenario",
                family="active_ai_beta",
                hypothesis=(
                    "Fast AI beta should be tested with scenario sizing so crowded upside gets cut "
                    "when breadth, credit, dollar, or volatility pressure rises."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=21,
                skip_days=2,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=42,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        44: (
            _active_dual_candidate(
                name="i44_active_credit_stress_rotation",
                family="active_credit_gate",
                hypothesis=(
                    "Fast credit and rates rotation tests whether loan, high-yield, bank, duration, "
                    "and gold leadership gives an earlier off-ramp than equities."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "KRE", "KBE", "IEF", "TLT", "GLD"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i44_active_defensive_assets_fast",
                family="active_defensive_barbell",
                hypothesis=(
                    "A defensive-only active sleeve tests whether gold, duration, T-bill-like bonds, "
                    "tips, and dollar strength can preserve capital during sharp transitions."
                ),
                tickers=["GLD", "IAU", "TLT", "IEF", "SHY", "TIP", "UUP", "FXY"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.40,
            ),
            _active_dual_candidate(
                name="i44_active_credit_gate_equity",
                family="active_credit_gate",
                hypothesis=(
                    "Equity exposure should shrink quickly if credit/rates proxies beat equity risk "
                    "assets on short-horizon risk-adjusted momentum."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "KRE", "IEF", "TLT", "GLD"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i44_active_drawdown_throttle_core",
                family="active_risk_control",
                hypothesis=(
                    "Fast cross-asset rotation may become viable only if strategy drawdown quickly "
                    "forces a lower-risk posture."
                ),
                tickers=["SPY", "QQQ", "IWM", "RSP", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=42,
                skip_days=5,
                top_n=2,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.40,
                drawdown_control=DrawdownControlConfig(
                    equity_lookback_days=63,
                    max_drawdown=-0.05,
                    risk_multiplier=0.25,
                ),
            ),
            _active_dual_candidate(
                name="i44_active_vol10_core_fast",
                family="active_risk_control",
                hypothesis=(
                    "Daily active rotation with a 10% volatility target tests whether turnover can be "
                    "converted into smoother compounding rather than more left-tail risk."
                ),
                tickers=["SPY", "QQQ", "IWM", "RSP", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=42,
                skip_days=2,
                top_n=2,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.40,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.10,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
            _active_dual_candidate(
                name="i44_active_defensive_scenario_core",
                family="active_risk_control",
                hypothesis=(
                    "Scenario sizing should improve active trading only if it cuts risk before "
                    "price-confirmed drawdowns are fully visible."
                ),
                tickers=["SPY", "QQQ", "RSP", "GLD", "TLT", "IEF", "UUP", "DBC"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        45: (
            _active_dual_candidate(
                name="i45_active_broadening_small_value",
                family="active_reflation_breadth",
                hypothesis=(
                    "Fast small-cap, value, bank, industrial, material, and energy rotation tests "
                    "whether broadening regimes can beat QQQ dependence."
                ),
                tickers=["IWM", "RSP", "VTV", "XLF", "KRE", "KBE", "XLI", "XLB", "XLE", "DBC"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i45_active_bank_industrial_materials",
                family="active_reflation_breadth",
                hypothesis=(
                    "Cyclical breadth may show up first in banks, transports, industrials, housing, "
                    "and materials before broad equity indexes re-rate."
                ),
                tickers=["KRE", "KBE", "XLF", "XLI", "IYT", "XHB", "XLB", "XME", "RSP", "IWM"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i45_active_energy_inflation_shock",
                family="active_commodity_shock",
                hypothesis=(
                    "Oil and inflation shocks may be tradable through fast energy, commodity, gold, "
                    "and dollar leadership."
                ),
                tickers=["XLE", "XOP", "OIH", "USO", "BNO", "DBC", "DBA", "GLD", "UUP", "TLT"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i45_active_consumer_housing_pivot",
                family="active_cyclical_pivot",
                hypothesis=(
                    "Housing, retail, discretionary, transports, and equal-weight breadth can act as "
                    "a fast risk-on/risk-off economic pulse."
                ),
                tickers=["XHB", "XRT", "XLY", "IYT", "XLI", "RSP", "IWM", "GLD", "IEF"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i45_active_quality_or_broadening",
                family="active_breadth_quality_switch",
                hypothesis=(
                    "If growth leadership fades, the system should choose between quality defense "
                    "and cyclical broadening instead of defaulting to bearish cash."
                ),
                tickers=["QUAL", "USMV", "SCHD", "VIG", "COWZ", "IWM", "RSP", "VTV", "XLF", "XLI"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i45_active_reflation_scenario_sized",
                family="active_reflation_breadth",
                hypothesis=(
                    "A reflation/broadening sleeve should be allowed to risk up, but scenario sizing "
                    "must cap it when credit or liquidity pressure contradicts the move."
                ),
                tickers=["IWM", "RSP", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "DBC", "GLD"],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        46: (
            _active_dual_candidate(
                name="i46_active_global_equity_rotation",
                family="active_global_rotation",
                hypothesis=(
                    "Fast global equity rotation tests whether ex-U.S. and country leadership can "
                    "replace U.S. mega-cap dependence during regime shifts."
                ),
                tickers=[
                    "SPY",
                    "RSP",
                    "EFA",
                    "EEM",
                    "VEA",
                    "VWO",
                    "VGK",
                    "EWJ",
                    "INDA",
                    "EWZ",
                    "EWC",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i46_active_dollar_shock_defense",
                family="active_global_macro",
                hypothesis=(
                    "Dollar strength, yen strength, gold, and duration may be the active defense "
                    "when global equity risk deteriorates."
                ),
                tickers=["UUP", "FXY", "FXF", "FXE", "GLD", "TLT", "IEF", "TIP", "SHY"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i46_active_commodity_currency_cross",
                family="active_global_macro",
                hypothesis=(
                    "Commodity, gold, oil, dollar, and duration leadership can capture macro "
                    "transition trades without shorting."
                ),
                tickers=["GLD", "IAU", "SLV", "CPER", "USO", "BNO", "DBC", "DBA", "UUP", "TLT"],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i46_active_em_rebound_probe",
                family="active_global_rotation",
                hypothesis=(
                    "Emerging market and international rebounds may require faster recognition than "
                    "a 6-12 month momentum system can provide."
                ),
                tickers=["EEM", "VWO", "MCHI", "INDA", "EWZ", "EWW", "EWA", "EWJ", "SPY", "UUP"],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i46_active_global_shock_barbell",
                family="active_global_macro",
                hypothesis=(
                    "A global shock barbell should move among U.S. risk, ex-U.S. risk, energy, gold, "
                    "dollar, and duration depending on current leadership."
                ),
                tickers=["SPY", "QQQ", "EFA", "EEM", "GLD", "USO", "DBC", "UUP", "TLT", "IEF"],
                lookback_days=21,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=42,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i46_active_global_scenario_sized",
                family="active_global_rotation",
                hypothesis=(
                    "Global active rotation should only be promoted if scenario sizing improves "
                    "left-tail behavior when dollar, oil, credit, or breadth pressure rises."
                ),
                tickers=[
                    "SPY",
                    "RSP",
                    "EFA",
                    "EEM",
                    "VWO",
                    "VGK",
                    "INDA",
                    "EWZ",
                    "GLD",
                    "UUP",
                    "DBC",
                    "TLT",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        47: (
            _active_dual_candidate(
                name="i47_active_sector_breadth_top4",
                family="active_sector_rotation",
                hypothesis=(
                    "A broader top-four sector basket may capture active sector leadership without "
                    "excessive single-sector churn."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i47_active_factor_breadth_top4",
                family="active_factor_rotation",
                hypothesis=(
                    "A broader factor basket tests whether active trading can be less brittle when "
                    "the system spreads across multiple factor winners."
                ),
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "MTUM",
                    "VTV",
                    "VUG",
                    "SCHD",
                    "VIG",
                    "COWZ",
                    "MOAT",
                    "SPMO",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=63,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i47_active_sector_factor_combo",
                family="active_breadth_combo",
                hypothesis=(
                    "Combining sectors and factors tests whether active breadth signals are stronger "
                    "than either taxonomy alone."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLP",
                    "QUAL",
                    "USMV",
                    "MTUM",
                    "VTV",
                    "COWZ",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i47_active_lowvol_dividend_defense",
                family="active_defensive_equity",
                hypothesis=(
                    "Low-volatility, dividend, quality, and defensive sector leadership can keep the "
                    "system invested without taking full QQQ-style beta."
                ),
                tickers=[
                    "USMV",
                    "SPLV",
                    "SCHD",
                    "VIG",
                    "QUAL",
                    "MOAT",
                    "XLV",
                    "XLP",
                    "XLU",
                    "GLD",
                    "IEF",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i47_active_high_beta_with_escape",
                family="active_high_beta_rotation",
                hypothesis=(
                    "High-beta leadership can be tested only with strict return and trend gates so it "
                    "does not become blind risk chasing."
                ),
                tickers=[
                    "SPHB",
                    "MTUM",
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "ARKK",
                    "IWM",
                    "XLY",
                    "XLK",
                    "GLD",
                    "TLT",
                ],
                lookback_days=21,
                skip_days=0,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=42,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
            _active_dual_candidate(
                name="i47_active_breadth_scenario_sized",
                family="active_breadth_combo",
                hypothesis=(
                    "Scenario sizing should cut broad active sector/factor exposure when breadth or "
                    "credit contradicts the apparent leadership."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLE",
                    "XLV",
                    "XLI",
                    "QUAL",
                    "USMV",
                    "MTUM",
                    "VTV",
                    "COWZ",
                    "GLD",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        48: (
            _active_dual_candidate(
                name="i48_active_whipsaw_core_42d",
                family="active_whipsaw_control",
                hypothesis=(
                    "A slightly slower 42-day active core tests whether daily monitoring can be useful "
                    "without reacting to every short-lived move."
                ),
                tickers=["SPY", "QQQ", "IWM", "RSP", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.35,
            ),
            _active_dual_candidate(
                name="i48_active_whipsaw_sector_63d",
                family="active_whipsaw_control",
                hypothesis=(
                    "A 63-day sector system is the medium-active alternative to daily fast sector "
                    "chasing."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
            ),
            _active_dual_candidate(
                name="i48_active_whipsaw_ai_42d_vol",
                family="active_whipsaw_control",
                hypothesis=(
                    "AI beta may need enough speed to exit bubbles but enough smoothing to avoid "
                    "daily narrative whipsaw."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
            _active_dual_candidate(
                name="i48_active_whipsaw_credit_gate",
                family="active_whipsaw_control",
                hypothesis=(
                    "Credit-gated active rotation should avoid overtrading by requiring a 42-day "
                    "risk-adjusted signal and trend confirmation."
                ),
                tickers=["SPY", "QQQ", "RSP", "HYG", "LQD", "KRE", "GLD", "TLT", "IEF", "UUP"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.35,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i48_active_whipsaw_drawdown_guard",
                family="active_whipsaw_control",
                hypothesis=(
                    "A drawdown guard may allow active systems to stay responsive while limiting "
                    "damage after repeated false breaks."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.35,
                drawdown_control=DrawdownControlConfig(
                    equity_lookback_days=84,
                    max_drawdown=-0.06,
                    risk_multiplier=0.30,
                ),
            ),
            _active_dual_candidate(
                name="i48_active_whipsaw_low_effort",
                family="active_whipsaw_control",
                hypothesis=(
                    "A lower-effort active strategy should still be evaluated because it may deliver "
                    "most of the benefit with fewer trade decisions."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=63,
                skip_days=10,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.35,
            ),
        ),
        49: (
            _active_dual_candidate(
                name="i49_active_os_cross_asset_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The candidate active core combines daily monitoring, short momentum, inverse-vol "
                    "positioning, trend confirmation, scenario sizing, and volatility throttling."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "IWM",
                    "EFA",
                    "EEM",
                    "GLD",
                    "TLT",
                    "IEF",
                    "DBC",
                    "UUP",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i49_active_os_ai_escape_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The active AI operating candidate only works if fast AI leadership survives "
                    "trend, risk-adjusted rank, size caps, and fragile-upside scenario cuts."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=63,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _active_dual_candidate(
                name="i49_active_os_credit_gate_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The active credit-gated candidate gives the system permission to pivot among "
                    "equity, credit, duration, gold, dollar, or cash as risk appetite changes."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "HYG",
                    "LQD",
                    "KRE",
                    "BKLN",
                    "GLD",
                    "TLT",
                    "IEF",
                    "UUP",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i49_active_os_reflation_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The active reflation candidate tests whether broadening can be bought quickly "
                    "without ignoring scenario pressure."
                ),
                tickers=[
                    "IWM",
                    "RSP",
                    "VTV",
                    "XLF",
                    "KRE",
                    "XLI",
                    "XLB",
                    "XLE",
                    "DBC",
                    "GLD",
                    "IEF",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i49_active_os_global_shock_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The active global shock candidate tests whether global equities, commodities, "
                    "gold, dollar, and duration create a more transition-aware operating sleeve."
                ),
                tickers=[
                    "SPY",
                    "EFA",
                    "EEM",
                    "VWO",
                    "INDA",
                    "EWZ",
                    "GLD",
                    "DBC",
                    "UUP",
                    "TLT",
                    "IEF",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i49_active_os_sector_breadth_scenario",
                family="active_operating_system",
                hypothesis=(
                    "The active sector/factor candidate tests whether breadth-aware leadership can "
                    "be a practical challenger to current core systems."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLE",
                    "XLV",
                    "XLI",
                    "QUAL",
                    "USMV",
                    "MTUM",
                    "VTV",
                    "COWZ",
                    "GLD",
                ],
                lookback_days=30,
                skip_days=2,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=63,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_absolute_candidate(
                name="i49_active_os_defensive_barbell",
                family="active_operating_system",
                hypothesis=(
                    "The defensive active operating candidate provides a simple daily trend benchmark "
                    "for capital preservation sleeves."
                ),
                tickers=["GLD", "IAU", "TLT", "IEF", "SHY", "TIP", "UUP", "FXY", "DBC"],
                moving_average_days=63,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i49_active_os_low_effort_active",
                family="active_operating_system",
                hypothesis=(
                    "The low-effort active candidate is included to test whether a slower active "
                    "system captures most benefits with fewer trade changes."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=63,
                skip_days=10,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
    }
    return batches[iteration]


def _final_deep_wide_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    role = "final_candidate"
    phase = "final_deep_dive"
    batches = {
        50: (
            _active_dual_candidate(
                name="i50_final_canary_core_63d",
                role=role,
                phase=phase,
                family="final_canary_core",
                hypothesis=(
                    "A not-too-twitchy canary core uses 63-day risk-adjusted leadership, trend "
                    "confirmation, inverse-vol sizing, and scenario cuts to avoid QQQ forever risk."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "IWM",
                    "EFA",
                    "EEM",
                    "GLD",
                    "TLT",
                    "IEF",
                    "DBC",
                    "UUP",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=126,
                max_asset_weight=0.35,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i50_final_weekly_cross_asset_low_churn",
                role=role,
                phase=phase,
                family="final_low_churn_core",
                hypothesis=(
                    "A slower active cross-asset policy tests whether most active benefit survives "
                    "with fewer trade changes and less human execution burden."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=126,
                max_asset_weight=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i50_final_credit_canary_equity_gate",
                role=role,
                phase=phase,
                family="final_credit_gate",
                hypothesis=(
                    "Credit and rates proxies compete directly with equities so worsening risk appetite "
                    "can pull the system out before broad indexes fully break."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "HYG",
                    "JNK",
                    "LQD",
                    "BKLN",
                    "KRE",
                    "GLD",
                    "TLT",
                    "IEF",
                    "UUP",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_absolute_candidate(
                name="i50_final_absolute_multitrend_barbell",
                role=role,
                phase=phase,
                family="final_trend_defense",
                hypothesis=(
                    "A multi-asset absolute trend barbell is the low-complexity off-ramp benchmark for "
                    "the final curated shelf."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "IWM",
                    "EFA",
                    "EEM",
                    "GLD",
                    "TLT",
                    "IEF",
                    "DBC",
                    "UUP",
                ],
                moving_average_days=84,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i50_final_svxy_risk_appetite_probe",
                role=role,
                phase=phase,
                family="final_risk_appetite_probe",
                hypothesis=(
                    "A constrained risk-appetite probe tests whether inverse-volatility, high-beta, "
                    "and broad equity leadership adds useful signal without becoming a volatility bet."
                ),
                tickers=["SPY", "QQQ", "RSP", "SPHB", "MTUM", "SVXY", "GLD", "TLT", "IEF", "BIL"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.25,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.10,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i50_final_drawdown_guarded_core",
                role=role,
                phase=phase,
                family="final_drawdown_guard",
                hypothesis=(
                    "Drawdown throttling is retested on a medium-active core to see whether it catches "
                    "market transitions without overreacting to routine pullbacks."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "GLD", "TLT", "IEF", "DBC", "UUP"],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.35,
                drawdown_control=DrawdownControlConfig(
                    equity_lookback_days=84,
                    max_drawdown=-0.06,
                    risk_multiplier=0.30,
                ),
            ),
        ),
        51: (
            _active_dual_candidate(
                name="i51_final_ai_escape_quality_gate",
                role=role,
                phase=phase,
                family="final_ai_escape",
                hypothesis=(
                    "AI leadership is only allowed when it beats quality, low-vol, gold, and duration "
                    "after risk adjustment and fragile-upside scenario cuts."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "QUAL",
                    "USMV",
                    "GLD",
                    "TLT",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _active_dual_candidate(
                name="i51_final_ai_infra_power_grid",
                role=role,
                phase=phase,
                family="final_ai_infrastructure",
                hypothesis=(
                    "If AI capex remains real but software economics wobble, power, grid, nuclear, "
                    "industrial, and hardware beneficiaries should compete for the satellite sleeve."
                ),
                tickers=[
                    "VRT",
                    "ETN",
                    "PWR",
                    "CEG",
                    "GEV",
                    "NRG",
                    "CCJ",
                    "SMH",
                    "SOXX",
                    "XLI",
                    "XLU",
                    "GLD",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i51_final_hardware_software_platform_switch",
                role=role,
                phase=phase,
                family="final_ai_platform_switch",
                hypothesis=(
                    "Hardware, cloud, software, mega-cap platforms, and QQQ are forced to compete so "
                    "the system does not assume all AI beta is equivalent."
                ),
                tickers=[
                    "SMH",
                    "SOXX",
                    "IGV",
                    "SKYY",
                    "CLOU",
                    "QQQ",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "ORCL",
                    "META",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _active_dual_candidate(
                name="i51_final_broadening_vs_ai_switch",
                role=role,
                phase=phase,
                family="final_ai_broadening_switch",
                hypothesis=(
                    "A direct AI-versus-broadening contest tests whether leadership is leaving QQQ for "
                    "small-cap, equal-weight, value, banks, industrials, or materials."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "RSP",
                    "IWM",
                    "VTV",
                    "XLF",
                    "KRE",
                    "XLI",
                    "XLB",
                    "COWZ",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i51_final_quality_income_escape",
                role=role,
                phase=phase,
                family="final_defensive_equity",
                hypothesis=(
                    "Quality, dividends, cash-flow, moat, and low-volatility factors are tested as a "
                    "less bearish way to de-risk without going entirely to cash."
                ),
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "SCHD",
                    "VIG",
                    "COWZ",
                    "MOAT",
                    "XLV",
                    "XLP",
                    "XLU",
                    "GLD",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i51_final_spec_liquidity_strict",
                role=role,
                phase=phase,
                family="final_speculative_liquidity",
                hypothesis=(
                    "Crypto and innovation proxies are allowed only as a small strict satellite when "
                    "risk-adjusted trend is strong enough to justify the effort."
                ),
                tickers=["ARKK", "IBIT", "FBTC", "BITB", "ETHE", "TSLA", "QQQ", "GLD", "TLT"],
                lookback_days=42,
                skip_days=5,
                top_n=2,
                min_return=0.04,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        52: (
            _active_dual_candidate(
                name="i52_final_policy_oil_hormuz_barbell",
                role=role,
                phase=phase,
                family="final_policy_oil_shock",
                hypothesis=(
                    "Geopolitical/oil shocks should be tradable only through a barbell that can choose "
                    "energy, commodities, gold, dollar, duration, broad equities, or cash."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "XLE",
                    "XOP",
                    "OIH",
                    "USO",
                    "BNO",
                    "DBC",
                    "GLD",
                    "UUP",
                    "TLT",
                    "IEF",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i52_final_dollar_liquidity_defense",
                role=role,
                phase=phase,
                family="final_global_liquidity",
                hypothesis=(
                    "Dollar, yen, gold, duration, T-bill-like, and inflation-linked assets test a pure "
                    "liquidity-defense sleeve for global stress."
                ),
                tickers=[
                    "UUP",
                    "FXY",
                    "FXF",
                    "FXE",
                    "GLD",
                    "TLT",
                    "IEF",
                    "TIP",
                    "SHY",
                    "SGOV",
                    "USFR",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i52_final_private_credit_bdc_warning",
                role=role,
                phase=phase,
                family="final_private_credit",
                hypothesis=(
                    "BDC, senior-loan, CLO, high-yield, bank, and duration proxies test whether private "
                    "credit stress can become an actionable early-warning sleeve."
                ),
                tickers=[
                    "BIZD",
                    "ARCC",
                    "MAIN",
                    "BXSL",
                    "OBDC",
                    "SRLN",
                    "BKLN",
                    "JAAA",
                    "JBBB",
                    "HYG",
                    "LQD",
                    "KRE",
                    "IEF",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i52_final_rates_credit_equity_triangle",
                role=role,
                phase=phase,
                family="final_rates_credit_triangle",
                hypothesis=(
                    "Equities, high yield, investment grade, duration, TIPS, and gold compete directly "
                    "to identify whether the dominant regime is growth, credit stress, or rates relief."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "HYG",
                    "LQD",
                    "VCIT",
                    "VCSH",
                    "TLT",
                    "IEF",
                    "TIP",
                    "GLD",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i52_final_global_ex_us_replacement",
                role=role,
                phase=phase,
                family="final_global_rotation",
                hypothesis=(
                    "Ex-U.S. and country leadership is retested as a possible replacement for crowded "
                    "U.S. mega-cap beta during market transitions."
                ),
                tickers=[
                    "SPY",
                    "RSP",
                    "EFA",
                    "EEM",
                    "VEA",
                    "VWO",
                    "VGK",
                    "EWJ",
                    "INDA",
                    "EWZ",
                    "EWC",
                    "UUP",
                    "GLD",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i52_final_market_plumbing_probe",
                role=role,
                phase=phase,
                family="final_market_plumbing",
                hypothesis=(
                    "A market-plumbing proxy tests whether short duration, credit quality, munis, gold, "
                    "and volatility-linked risk appetite provide useful signals around liquidity shocks."
                ),
                tickers=[
                    "SGOV",
                    "USFR",
                    "SHY",
                    "VCSH",
                    "VCIT",
                    "MUB",
                    "HYG",
                    "LQD",
                    "GLD",
                    "SVXY",
                    "SPY",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=84,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        53: (
            _active_dual_candidate(
                name="i53_final_sector_factor_blend_low_churn",
                role=role,
                phase=phase,
                family="final_sector_factor_blend",
                hypothesis=(
                    "A sector/factor blend tests whether leadership breadth can be harvested with "
                    "less single-sector concentration and fewer trade flips."
                ),
                tickers=[
                    "XLK",
                    "XLF",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLP",
                    "QUAL",
                    "USMV",
                    "MTUM",
                    "VTV",
                    "COWZ",
                    "GLD",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=126,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i53_final_breadth_reflation_quality_switch",
                role=role,
                phase=phase,
                family="final_breadth_reflation",
                hypothesis=(
                    "The system should be able to switch among quality defense, cyclical broadening, "
                    "and cash-like assets without making a binary bearish call."
                ),
                tickers=[
                    "QUAL",
                    "USMV",
                    "SCHD",
                    "VIG",
                    "COWZ",
                    "RSP",
                    "IWM",
                    "VTV",
                    "XLF",
                    "XLI",
                    "XLB",
                    "GLD",
                    "IEF",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i53_final_lowvol_dividend_cashflow",
                role=role,
                phase=phase,
                family="final_defensive_equity",
                hypothesis=(
                    "Low-vol, dividend, quality, cash-flow, and moat factors are tested as a practical "
                    "medium-term defensive equity sleeve."
                ),
                tickers=[
                    "USMV",
                    "SPLV",
                    "SCHD",
                    "VIG",
                    "QUAL",
                    "COWZ",
                    "MOAT",
                    "VTV",
                    "XLV",
                    "XLP",
                    "XLU",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=126,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i53_final_barbell_growth_defense",
                role=role,
                phase=phase,
                family="final_growth_defense_barbell",
                hypothesis=(
                    "A growth/defense barbell tests whether QQQ and semis can coexist with gold, "
                    "duration, quality, and cash-like protection in one operating policy."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "SPY",
                    "RSP",
                    "QUAL",
                    "USMV",
                    "GLD",
                    "TLT",
                    "IEF",
                    "BIL",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.01,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.11,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i53_final_vol_target_factor_core",
                role=role,
                phase=phase,
                family="final_vol_target_core",
                hypothesis=(
                    "Volatility targeting is retested on a diversified factor/sector core to see if "
                    "smooth compounding beats raw tactical return."
                ),
                tickers=[
                    "SPY",
                    "RSP",
                    "QUAL",
                    "USMV",
                    "MTUM",
                    "VTV",
                    "COWZ",
                    "XLK",
                    "XLF",
                    "XLV",
                    "GLD",
                    "IEF",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.10,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i53_final_cash_plus_risk_reentry",
                role=role,
                phase=phase,
                family="final_reentry_system",
                hypothesis=(
                    "A cash-plus-risk reentry system tests whether the bot can sit in T-bill-like assets "
                    "and re-risk only when broad leadership earns it."
                ),
                tickers=[
                    "BIL",
                    "SGOV",
                    "USFR",
                    "SHY",
                    "SPY",
                    "QQQ",
                    "RSP",
                    "IWM",
                    "GLD",
                    "IEF",
                    "DBC",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        54: (
            _active_dual_candidate(
                name="i54_final_curated_candidate_core",
                role=role,
                phase=phase,
                family="final_curated_operating_system",
                hypothesis=(
                    "A final candidate core combines medium-active cross-asset selection, trend, "
                    "inverse-vol sizing, volatility targeting, and balanced scenario sizing."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "IWM",
                    "EFA",
                    "EEM",
                    "GLD",
                    "TLT",
                    "IEF",
                    "DBC",
                    "UUP",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=63,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i54_final_curated_candidate_satellite_ai",
                role=role,
                phase=phase,
                family="final_curated_operating_system",
                hypothesis=(
                    "A final AI satellite candidate keeps upside participation but cuts size through "
                    "trend, concentration caps, volatility targeting, and fragile-AI scenario sizing."
                ),
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                    "GLD",
                    "TLT",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.25,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _active_dual_candidate(
                name="i54_final_curated_candidate_credit_macro",
                role=role,
                phase=phase,
                family="final_curated_operating_system",
                hypothesis=(
                    "A final credit/macro candidate is included as the explicit off-ramp and non-QQQ "
                    "transition sleeve."
                ),
                tickers=[
                    "SPY",
                    "QQQ",
                    "RSP",
                    "HYG",
                    "LQD",
                    "BKLN",
                    "KRE",
                    "GLD",
                    "TLT",
                    "IEF",
                    "UUP",
                    "DBC",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i54_final_curated_candidate_breadth",
                role=role,
                phase=phase,
                family="final_curated_operating_system",
                hypothesis=(
                    "A final breadth candidate gives the system a non-mega-cap path if leadership rotates "
                    "toward equal weight, small caps, value, banks, industrials, and materials."
                ),
                tickers=[
                    "RSP",
                    "IWM",
                    "VTV",
                    "XLF",
                    "KRE",
                    "XLI",
                    "XLB",
                    "XLE",
                    "COWZ",
                    "QUAL",
                    "GLD",
                    "IEF",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="i54_final_curated_candidate_defense",
                role=role,
                phase=phase,
                family="final_curated_operating_system",
                hypothesis=(
                    "A final defensive candidate tests whether gold, duration, dollar, T-bill-like, and "
                    "defensive equity factors can provide a practical left-tail sleeve."
                ),
                tickers=[
                    "GLD",
                    "IAU",
                    "TLT",
                    "IEF",
                    "TIP",
                    "SHY",
                    "SGOV",
                    "USFR",
                    "UUP",
                    "USMV",
                    "SPLV",
                    "XLU",
                    "XLP",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="i54_final_wild_spec_liquidity_micro_sleeve",
                role=role,
                phase=phase,
                family="final_wild_probe",
                hypothesis=(
                    "A deliberately wild but capped liquidity sleeve tests whether crypto, innovation, "
                    "solar, robotics, biotech, and high-beta proxies add useful early-cycle optionality."
                ),
                tickers=[
                    "ARKK",
                    "IBIT",
                    "FBTC",
                    "ETHE",
                    "TAN",
                    "BOTZ",
                    "ROBO",
                    "XBI",
                    "SPHB",
                    "QQQ",
                    "GLD",
                    "TLT",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.04,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=84,
                max_asset_weight=0.20,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.12,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
    }
    return batches[iteration]


def _dip_reentry_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    batches = {
        55: (
            _dip_reentry_candidate(
                name="i55_dip_broad_market_discount_ladder",
                family="dip_broad_market",
                hypothesis=(
                    "Broad equity reentry ladders back from BIL only after SPY/QQQ/RSP/IWM are "
                    "meaningfully discounted and recovery, volatility, credit, and breadth stop "
                    "looking like an active falling knife."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "IEF"],
                trigger=-0.10,
                deep=-0.22,
                starter=0.18,
                step=0.18,
                max_risk=0.72,
                top_n=4,
                max_asset_weight=0.28,
            ),
            _dip_reentry_candidate(
                name="i55_dip_strict_confirmed_repair",
                family="dip_broad_market",
                hypothesis=(
                    "A strict version waits for a deeper discount and stronger 21-day repair before "
                    "adding equity risk, prioritizing avoiding falling knives over catching the exact bottom."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "QUAL", "USMV", "HYG", "LQD", "GLD", "IEF"],
                trigger=-0.14,
                deep=-0.28,
                min_recovery=0.025,
                starter=0.12,
                step=0.16,
                max_risk=0.60,
                top_n=4,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i55_dip_aggressive_panic_starter",
                family="dip_broad_market",
                hypothesis=(
                    "A more aggressive panic-starter tests whether a small early allocation after "
                    "large drawdowns captures near-bottom returns without letting the ladder get all-in."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "SPHB", "MTUM", "HYG", "LQD", "GLD", "TLT"],
                trigger=-0.08,
                deep=-0.20,
                min_recovery=0.010,
                starter=0.25,
                step=0.22,
                max_risk=0.85,
                top_n=4,
                max_asset_weight=0.30,
                credit_confirmation=True,
                breadth_confirmation=True,
            ),
            _dip_reentry_candidate(
                name="i55_dip_weekly_low_churn_reentry",
                family="dip_low_churn",
                hypothesis=(
                    "A slower low-churn reentry system tests whether discount buying can be operated "
                    "with weekly human decisions rather than daily twitch."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                trigger=-0.12,
                deep=-0.26,
                recovery_days=42,
                confirmation_days=10,
                min_recovery=0.025,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                top_n=4,
                max_asset_weight=0.28,
                trend_filter_days=126,
            ),
            _dip_reentry_candidate(
                name="i55_dip_value_quality_reentry",
                family="dip_fundamental_proxy",
                hypothesis=(
                    "Value, quality, cash-flow, dividend, and low-volatility ETFs act as fundamental "
                    "valuation proxies: buy the dip only in resilient equity cohorts, not the whole index."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "RSP", "SPY"],
                trigger=-0.09,
                deep=-0.20,
                starter=0.20,
                step=0.18,
                max_risk=0.75,
                top_n=5,
                max_asset_weight=0.25,
                breadth_confirmation=False,
            ),
            _dip_reentry_candidate(
                name="i55_dip_credit_repair_first",
                family="dip_credit_repair",
                hypothesis=(
                    "Credit repair may be the confirmation signal for buying broad-market dips; this "
                    "basket can hold credit, duration, gold, or equities depending on repair quality."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "KRE", "SPY", "RSP", "GLD", "IEF"],
                trigger=-0.08,
                deep=-0.18,
                min_recovery=0.010,
                starter=0.18,
                step=0.17,
                max_risk=0.68,
                top_n=4,
                max_asset_weight=0.28,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        56: (
            _dip_reentry_candidate(
                name="i56_dip_ai_semis_crash_repair",
                family="dip_ai_beta_reentry",
                hypothesis=(
                    "AI beta dip buying is only allowed after semis/software/platforms are deeply "
                    "discounted and repair; otherwise this stays mostly in BIL."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN", "GLD", "TLT"],
                trigger=-0.14,
                deep=-0.30,
                min_recovery=0.030,
                starter=0.10,
                step=0.16,
                max_risk=0.55,
                top_n=3,
                max_asset_weight=0.22,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_reentry_candidate(
                name="i56_dip_ai_infra_resilient_reentry",
                family="dip_ai_infrastructure",
                hypothesis=(
                    "AI-infrastructure dip buying tests whether power, grid, nuclear, and hardware "
                    "beneficiaries are better reentry vehicles than pure software beta."
                ),
                tickers=["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX", "XLI", "XLU", "GLD"],
                trigger=-0.12,
                deep=-0.26,
                min_recovery=0.025,
                starter=0.14,
                step=0.17,
                max_risk=0.62,
                top_n=4,
                max_asset_weight=0.22,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i56_dip_mega_cap_platform_repair",
                family="dip_mega_cap_platform",
                hypothesis=(
                    "Mega-cap dip reentry tests whether platform leaders recover first after market "
                    "stress while avoiding single-name concentration."
                ),
                tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "JPM", "GLD"],
                trigger=-0.12,
                deep=-0.25,
                min_recovery=0.025,
                starter=0.14,
                step=0.18,
                max_risk=0.65,
                top_n=4,
                max_asset_weight=0.22,
                vol_ceiling=0.38,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_reentry_candidate(
                name="i56_dip_high_beta_micro_sleeve",
                family="dip_high_beta_reentry",
                hypothesis=(
                    "A deliberately capped high-beta dip sleeve tests whether speculative beta rebounds "
                    "are useful only after extreme discounts and repair."
                ),
                tickers=["SPHB", "ARKK", "IBIT", "FBTC", "XBI", "TAN", "BOTZ", "QQQ", "GLD", "TLT"],
                trigger=-0.18,
                deep=-0.35,
                min_recovery=0.045,
                starter=0.06,
                step=0.12,
                max_risk=0.35,
                top_n=2,
                max_asset_weight=0.16,
                vol_ceiling=0.58,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i56_dip_growth_defense_barbell_repair",
                family="dip_growth_defense_barbell",
                hypothesis=(
                    "Growth dip reentry competes directly with gold, duration, quality, and low-vol so "
                    "the bot can buy recovery without assuming QQQ must lead."
                ),
                tickers=["QQQ", "SMH", "SOXX", "SPY", "RSP", "QUAL", "USMV", "GLD", "TLT", "IEF"],
                trigger=-0.11,
                deep=-0.24,
                min_recovery=0.020,
                starter=0.16,
                step=0.18,
                max_risk=0.68,
                top_n=4,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i56_dip_ai_strict_no_credit_no_buy",
                family="dip_ai_beta_reentry",
                hypothesis=(
                    "This harsh AI reentry variant requires credit and breadth repair; if both fail, "
                    "it intentionally refuses to buy the AI dip."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "HYG", "LQD", "RSP", "GLD"],
                trigger=-0.16,
                deep=-0.32,
                min_recovery=0.035,
                starter=0.08,
                step=0.14,
                max_risk=0.48,
                top_n=3,
                max_asset_weight=0.20,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        57: (
            _dip_reentry_candidate(
                name="i57_dip_breadth_repair_reflation",
                family="dip_breadth_repair",
                hypothesis=(
                    "Reentry after drawdowns may work best when equal-weight, small-cap, value, banks, "
                    "industrials, and materials confirm broadening rather than mega-cap-only bounces."
                ),
                tickers=["RSP", "IWM", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "COWZ", "GLD", "IEF"],
                trigger=-0.10,
                deep=-0.22,
                min_recovery=0.020,
                starter=0.18,
                step=0.18,
                max_risk=0.72,
                top_n=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i57_dip_small_value_washout",
                family="dip_small_value",
                hypothesis=(
                    "Small/value washouts can be powerful if bought after breadth and credit repair, "
                    "but the ladder caps concentration because these assets can keep falling."
                ),
                tickers=["IWM", "MDY", "VTV", "IWD", "XLF", "KRE", "XLI", "XLB", "XHB", "XRT", "GLD"],
                trigger=-0.13,
                deep=-0.28,
                min_recovery=0.030,
                starter=0.12,
                step=0.16,
                max_risk=0.58,
                top_n=4,
                max_asset_weight=0.20,
                vol_ceiling=0.38,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i57_dip_quality_income_discount",
                family="dip_fundamental_proxy",
                hypothesis=(
                    "A quality/income discount ladder represents the fundamentals-aware version: buy "
                    "cash-flow, moat, dividend, and low-vol drawdowns before chasing high beta."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "VFQY", "QVAL", "GLD"],
                trigger=-0.08,
                deep=-0.18,
                min_recovery=0.012,
                starter=0.22,
                step=0.18,
                max_risk=0.78,
                top_n=5,
                max_asset_weight=0.22,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i57_dip_sector_washout_rotation",
                family="dip_sector_reentry",
                hypothesis=(
                    "Sector washout reentry buys only sectors that are both discounted and repairing, "
                    "rather than buying the whole market after every dip."
                ),
                tickers=["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC", "GLD"],
                trigger=-0.10,
                deep=-0.24,
                min_recovery=0.020,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                top_n=5,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i57_dip_global_ex_us_discount",
                family="dip_global_reentry",
                hypothesis=(
                    "Global equity dip buying tests whether cheap ex-U.S. and country ETFs can be "
                    "better rebound vehicles than crowded U.S. mega-cap exposure."
                ),
                tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC", "GLD", "UUP"],
                trigger=-0.11,
                deep=-0.24,
                min_recovery=0.020,
                starter=0.16,
                step=0.18,
                max_risk=0.68,
                top_n=5,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i57_dip_cyclical_repair_strict",
                family="dip_cyclical_repair",
                hypothesis=(
                    "A strict cyclical reentry variant refuses to buy small/value/cyclicals unless the "
                    "short rebound is visible and volatility is settling."
                ),
                tickers=["IWM", "RSP", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "XHB", "IYT", "GLD"],
                trigger=-0.14,
                deep=-0.30,
                min_recovery=0.035,
                starter=0.10,
                step=0.15,
                max_risk=0.52,
                top_n=4,
                max_asset_weight=0.18,
                vol_ceiling=0.36,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        58: (
            _dip_reentry_candidate(
                name="i58_dip_credit_spread_repair_ladder",
                family="dip_credit_repair",
                hypothesis=(
                    "Credit-led reentry waits for high yield, loans, banks, and investment-grade credit "
                    "to repair before letting equity risk scale up."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "KRE", "SPY", "RSP", "GLD", "IEF"],
                trigger=-0.08,
                deep=-0.18,
                min_recovery=0.012,
                starter=0.18,
                step=0.17,
                max_risk=0.66,
                top_n=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i58_dip_liquidity_vol_crush_reentry",
                family="dip_liquidity_reentry",
                hypothesis=(
                    "This tests the classic volatility-crush reentry: buy discounted risk only after "
                    "volatility, dollar, credit, and breadth pressure stop worsening."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "UUP", "GLD", "TLT", "IEF", "SVXY"],
                trigger=-0.10,
                deep=-0.22,
                min_recovery=0.020,
                starter=0.14,
                step=0.18,
                max_risk=0.62,
                top_n=4,
                max_asset_weight=0.22,
                vol_ceiling=0.34,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i58_dip_duration_rates_relief",
                family="dip_rates_relief",
                hypothesis=(
                    "If market cheapness is caused by rates stress, reentry should prefer duration, "
                    "quality credit, gold, and equities only after rates relief is visible."
                ),
                tickers=["SPY", "QQQ", "RSP", "HYG", "LQD", "VCIT", "VCSH", "TLT", "IEF", "TIP", "GLD"],
                trigger=-0.09,
                deep=-0.20,
                min_recovery=0.015,
                starter=0.18,
                step=0.17,
                max_risk=0.68,
                top_n=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i58_dip_private_credit_repair",
                family="dip_private_credit",
                hypothesis=(
                    "Private-credit and BDC proxies test whether the bot should avoid dip buying when "
                    "illiquid credit stress is still leaking into public markets."
                ),
                tickers=["BIZD", "ARCC", "MAIN", "BXSL", "OBDC", "SRLN", "BKLN", "HYG", "LQD", "KRE", "IEF"],
                trigger=-0.10,
                deep=-0.22,
                min_recovery=0.018,
                starter=0.12,
                step=0.15,
                max_risk=0.55,
                top_n=4,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i58_dip_gold_duration_then_risk",
                family="dip_defensive_bridge",
                hypothesis=(
                    "A defensive bridge lets gold/duration/T-bill-like assets win first, then risk assets "
                    "must earn their way back through repair."
                ),
                tickers=["GLD", "IAU", "TLT", "IEF", "TIP", "SHY", "SGOV", "USFR", "SPY", "RSP", "QQQ"],
                trigger=-0.08,
                deep=-0.18,
                min_recovery=0.012,
                starter=0.20,
                step=0.16,
                max_risk=0.62,
                top_n=5,
                max_asset_weight=0.24,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i58_dip_macro_repair_triangle",
                family="dip_macro_repair",
                hypothesis=(
                    "Equities, credit, duration, gold, dollar, and commodities compete as a macro "
                    "repair triangle after deep discounts."
                ),
                tickers=["SPY", "QQQ", "RSP", "HYG", "LQD", "GLD", "TLT", "IEF", "UUP", "DBC", "USO"],
                trigger=-0.10,
                deep=-0.22,
                min_recovery=0.018,
                starter=0.16,
                step=0.18,
                max_risk=0.66,
                top_n=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        59: (
            _dip_reentry_candidate(
                name="i59_dip_capitulation_fast_rebound",
                family="dip_capitulation",
                hypothesis=(
                    "A fast capitulation strategy tests whether steep selloffs plus immediate 5-day "
                    "repair deserve a small but meaningful risk ladder."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "SPHB", "MTUM", "HYG", "LQD", "GLD", "TLT"],
                trigger=-0.12,
                deep=-0.25,
                recovery_days=10,
                confirmation_days=3,
                min_recovery=0.018,
                starter=0.18,
                step=0.18,
                max_risk=0.60,
                top_n=3,
                max_asset_weight=0.24,
                vol_ceiling=0.45,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i59_dip_slow_bottoming_base",
                family="dip_bottoming_base",
                hypothesis=(
                    "A slow bottoming-base variant requires 42-day repair and lower volatility, aiming "
                    "to buy after durable stabilization rather than near the exact bottom."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "QUAL", "USMV", "GLD", "IEF"],
                trigger=-0.11,
                deep=-0.24,
                recovery_days=42,
                confirmation_days=10,
                min_recovery=0.025,
                starter=0.14,
                step=0.16,
                max_risk=0.64,
                top_n=4,
                max_asset_weight=0.24,
                vol_ceiling=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i59_dip_no_breadth_contrarian_probe",
                family="dip_contrarian_probe",
                hypothesis=(
                    "This contrarian probe deliberately relaxes breadth confirmation to test whether "
                    "waiting for breadth repair gives up too much near-bottom upside."
                ),
                tickers=["SPY", "QQQ", "SMH", "SOXX", "RSP", "IWM", "HYG", "LQD", "GLD", "TLT"],
                trigger=-0.10,
                deep=-0.24,
                min_recovery=0.012,
                starter=0.22,
                step=0.18,
                max_risk=0.74,
                top_n=4,
                max_asset_weight=0.26,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i59_dip_deep_value_only",
                family="dip_deep_value_only",
                hypothesis=(
                    "A deep-value-only ladder should rarely trade, but when it does it tests whether "
                    "large discounts plus modest repair are enough to improve long-run compounding."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "VTV", "COWZ", "QUAL", "GLD", "IEF", "HYG"],
                trigger=-0.20,
                deep=-0.35,
                min_recovery=0.020,
                starter=0.20,
                step=0.22,
                max_risk=0.78,
                top_n=4,
                max_asset_weight=0.26,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i59_dip_stair_step_recovery",
                family="dip_stair_step",
                hypothesis=(
                    "A stair-step recovery ladder uses modest starter and repeated confirmation steps, "
                    "trying to capture rebounds while limiting regret if the first bounce fails."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "IEF"],
                trigger=-0.09,
                deep=-0.22,
                min_recovery=0.018,
                starter=0.10,
                step=0.24,
                max_risk=0.82,
                top_n=5,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i59_dip_defensive_to_risk_rotation",
                family="dip_defensive_bridge",
                hypothesis=(
                    "This version starts with defensive assets in the opportunity set and forces risk "
                    "assets to outperform them before the ladder meaningfully re-risks."
                ),
                tickers=["GLD", "TLT", "IEF", "SGOV", "USFR", "SPY", "QQQ", "RSP", "IWM", "HYG"],
                trigger=-0.08,
                deep=-0.20,
                min_recovery=0.014,
                starter=0.16,
                step=0.18,
                max_risk=0.64,
                top_n=5,
                max_asset_weight=0.24,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        60: (
            _dip_reentry_candidate(
                name="i60_dip_final_core_discount_repair",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final core candidate: broad-market discount ladder with credit/breadth/volatility "
                    "repair, designed to complement off-ramp systems that otherwise stay too defensive."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "IEF", "DBC"],
                trigger=-0.11,
                deep=-0.24,
                min_recovery=0.018,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                top_n=5,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i60_dip_final_quality_value_repair",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final quality/value candidate: buy discounted fundamental-proxy cohorts first, "
                    "then let broader beta in only when repair is broad enough."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "RSP", "SPY", "GLD"],
                trigger=-0.08,
                deep=-0.20,
                min_recovery=0.012,
                starter=0.20,
                step=0.18,
                max_risk=0.76,
                top_n=5,
                max_asset_weight=0.22,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i60_dip_final_ai_repair_micro",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final AI micro candidate: participate in AI crash rebounds only through small, "
                    "confirmed, scenario-cut allocations."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "GLD", "TLT", "HYG", "LQD"],
                trigger=-0.15,
                deep=-0.32,
                min_recovery=0.035,
                starter=0.08,
                step=0.14,
                max_risk=0.45,
                top_n=3,
                max_asset_weight=0.18,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_reentry_candidate(
                name="i60_dip_final_credit_breadth_gate",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final credit/breadth gate: refuses to buy equity discounts unless credit and "
                    "equal-weight breadth are repairing, targeting fewer false bottoms."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "KRE", "SPY", "RSP", "IWM", "GLD", "IEF"],
                trigger=-0.09,
                deep=-0.20,
                min_recovery=0.016,
                starter=0.14,
                step=0.17,
                max_risk=0.62,
                top_n=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_reentry_candidate(
                name="i60_dip_final_cyclical_broadening",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final cyclical candidate: buy small/value/cyclical discounts only when the rebound "
                    "is broadening beyond mega-cap growth."
                ),
                tickers=["RSP", "IWM", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "COWZ", "QUAL", "GLD", "IEF"],
                trigger=-0.11,
                deep=-0.25,
                min_recovery=0.022,
                starter=0.14,
                step=0.17,
                max_risk=0.64,
                top_n=5,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="i60_dip_final_deep_value_rare",
                family="dip_final_operating_system",
                hypothesis=(
                    "Final rare-event candidate: only acts in deep discounts, intended as a challenger "
                    "to staying over-defensive after major market washouts."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "VTV", "COWZ", "QUAL", "HYG", "LQD", "GLD", "IEF"],
                trigger=-0.18,
                deep=-0.34,
                min_recovery=0.022,
                starter=0.18,
                step=0.20,
                max_risk=0.72,
                top_n=4,
                max_asset_weight=0.24,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
    }
    return batches[iteration]


def _dip_reentry_overlay_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    batches = {
        61: (
            _dip_overlay_candidate(
                name="i61_dip_overlay_core_cash_redeploy",
                family="dip_overlay_core",
                hypothesis=(
                    "Core overlay starts with a dual-momentum off-ramp, then replaces BIL with "
                    "discounted equities only when credit, breadth, volatility, and repair confirm."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "IEF"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                min_return=0.02,
                trigger=-0.10,
                deep=-0.22,
                starter=0.22,
                step=0.22,
                max_risk=0.82,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i61_dip_overlay_strict_repair_gate",
                family="dip_overlay_core",
                hypothesis=(
                    "Strict overlay tests whether the bot should wait for a deeper discount and "
                    "stronger repair before redeploying defensive cash."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "QUAL", "USMV", "HYG", "LQD", "GLD", "IEF"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                min_return=0.03,
                trigger=-0.14,
                deep=-0.28,
                min_recovery=0.028,
                starter=0.16,
                step=0.18,
                max_risk=0.68,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i61_dip_overlay_aggressive_panic_probe",
                family="dip_overlay_core",
                hypothesis=(
                    "Aggressive panic probe tests whether a larger starter allocation after a crash "
                    "improves rebound capture without abandoning the falling-knife throttle."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "SPHB", "MTUM", "HYG", "LQD", "GLD", "TLT"],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                min_return=0.04,
                trigger=-0.09,
                deep=-0.20,
                min_recovery=0.014,
                starter=0.30,
                step=0.24,
                max_risk=0.92,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i61_dip_overlay_weekly_low_churn",
                family="dip_overlay_low_churn",
                hypothesis=(
                    "Weekly low-churn overlay favors durable basing over exact-bottom timing, keeping "
                    "the expected human trading cadence reasonable."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.02,
                trigger=-0.12,
                deep=-0.26,
                recovery_days=42,
                confirmation_days=10,
                min_recovery=0.025,
                starter=0.18,
                step=0.20,
                max_risk=0.78,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i61_dip_overlay_quality_first",
                family="dip_overlay_fundamental_proxy",
                hypothesis=(
                    "Quality/value/income ETFs act as fundamentals proxies, so cash redeploys first "
                    "into resilient cohorts rather than the highest-beta losers."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "RSP", "SPY"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.01,
                trigger=-0.08,
                deep=-0.18,
                starter=0.24,
                step=0.20,
                max_risk=0.84,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i61_dip_overlay_credit_first",
                family="dip_overlay_credit_repair",
                hypothesis=(
                    "Credit-first overlay uses high-yield and loan repair as permission to replace "
                    "BIL with risk after a drawdown."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "KRE", "SPY", "RSP", "GLD", "IEF"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.01,
                trigger=-0.08,
                deep=-0.18,
                starter=0.22,
                step=0.18,
                max_risk=0.76,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        62: (
            _dip_overlay_candidate(
                name="i62_dip_overlay_ai_beta_escape_reentry",
                family="dip_overlay_ai_beta",
                hypothesis=(
                    "AI-beta overlay tests whether an existing AI off-ramp can buy back only after "
                    "semis/platforms are discounted and repairing."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN", "HYG", "LQD", "GLD"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.04,
                trigger=-0.14,
                deep=-0.30,
                min_recovery=0.030,
                starter=0.12,
                step=0.18,
                max_risk=0.62,
                vol_ceiling=0.44,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_overlay_candidate(
                name="i62_dip_overlay_ai_infra_repair",
                family="dip_overlay_ai_infra",
                hypothesis=(
                    "AI-infrastructure overlay buys power/grid/hardware drawdowns when those assets "
                    "repair before pure software beta."
                ),
                tickers=["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX", "XLI", "XLU", "GLD"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.03,
                trigger=-0.12,
                deep=-0.26,
                min_recovery=0.026,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i62_dip_overlay_mega_cap_platforms",
                family="dip_overlay_mega_cap",
                hypothesis=(
                    "Mega-cap overlay tests whether platforms can be bought after stress without "
                    "letting single-name concentration dominate the rebound."
                ),
                tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "JPM", "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.03,
                trigger=-0.12,
                deep=-0.25,
                min_recovery=0.025,
                starter=0.16,
                step=0.20,
                max_risk=0.74,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_overlay_candidate(
                name="i62_dip_overlay_high_beta_small_probe",
                family="dip_overlay_high_beta",
                hypothesis=(
                    "Small high-beta probe buys only extreme speculative washouts with tight caps, "
                    "testing whether rebounds are worth the operational complexity."
                ),
                tickers=["SPHB", "ARKK", "IBIT", "FBTC", "XBI", "TAN", "BOTZ", "QQQ", "HYG", "LQD", "GLD"],
                lookback_days=42,
                skip_days=5,
                top_n=2,
                min_return=0.05,
                trigger=-0.20,
                deep=-0.38,
                min_recovery=0.050,
                starter=0.08,
                step=0.12,
                max_risk=0.42,
                max_asset_weight=0.16,
                vol_ceiling=0.60,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i62_dip_overlay_growth_defense_barbell",
                family="dip_overlay_growth_defense",
                hypothesis=(
                    "Growth-defense overlay lets gold/duration/quality compete with QQQ and semis "
                    "during recovery, avoiding a one-note AI re-risk."
                ),
                tickers=["QQQ", "SMH", "SOXX", "SPY", "RSP", "QUAL", "USMV", "GLD", "TLT", "IEF", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.02,
                trigger=-0.11,
                deep=-0.24,
                min_recovery=0.020,
                starter=0.18,
                step=0.20,
                max_risk=0.78,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i62_dip_overlay_ai_no_credit_no_buy",
                family="dip_overlay_ai_beta",
                hypothesis=(
                    "Harsh AI overlay refuses to redeploy cash into AI unless credit and breadth "
                    "confirm that the selloff is no longer widening."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "HYG", "LQD", "RSP", "GLD"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                min_return=0.04,
                trigger=-0.16,
                deep=-0.32,
                min_recovery=0.035,
                starter=0.10,
                step=0.16,
                max_risk=0.54,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        63: (
            _dip_overlay_candidate(
                name="i63_dip_overlay_breadth_reflation",
                family="dip_overlay_breadth",
                hypothesis=(
                    "Breadth overlay redeploys cash when equal-weight, small-cap, value, banks, and "
                    "industrials start confirming a market-wide recovery."
                ),
                tickers=["RSP", "IWM", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "COWZ", "HYG", "LQD", "GLD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.02,
                trigger=-0.10,
                deep=-0.22,
                starter=0.20,
                step=0.20,
                max_risk=0.82,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i63_dip_overlay_small_value_washout",
                family="dip_overlay_small_value",
                hypothesis=(
                    "Small/value overlay tests whether the strongest post-washout returns come from "
                    "breadth-sensitive assets after repair rather than QQQ."
                ),
                tickers=["IWM", "MDY", "VTV", "IWD", "XLF", "KRE", "XLI", "XLB", "XHB", "XRT", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.025,
                trigger=-0.13,
                deep=-0.28,
                min_recovery=0.030,
                starter=0.14,
                step=0.18,
                max_risk=0.64,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i63_dip_overlay_quality_income",
                family="dip_overlay_fundamental_proxy",
                hypothesis=(
                    "Quality/income overlay is the fundamentals-proxy version of buy-the-dip: cash "
                    "reenters through profitable, dividend, moat, and low-volatility cohorts."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "VFQY", "QVAL", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.01,
                trigger=-0.08,
                deep=-0.18,
                starter=0.24,
                step=0.20,
                max_risk=0.84,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i63_dip_overlay_sector_washout",
                family="dip_overlay_sector",
                hypothesis=(
                    "Sector washout overlay buys only sectors that are simultaneously cheap, repairing, "
                    "and not too volatile."
                ),
                tickers=["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC", "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.015,
                trigger=-0.10,
                deep=-0.24,
                starter=0.18,
                step=0.20,
                max_risk=0.78,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i63_dip_overlay_global_discount",
                family="dip_overlay_global",
                hypothesis=(
                    "Global discount overlay tests whether ex-U.S. or country ETFs offer better "
                    "post-drawdown value than expensive U.S. mega-cap beta."
                ),
                tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC", "HYG", "LQD", "UUP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.015,
                trigger=-0.11,
                deep=-0.24,
                starter=0.18,
                step=0.20,
                max_risk=0.78,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i63_dip_overlay_cyclical_strict",
                family="dip_overlay_cyclical",
                hypothesis=(
                    "Strict cyclical overlay redeploys cash only when small/value/cyclical assets "
                    "show repair and volatility is no longer expanding."
                ),
                tickers=["IWM", "RSP", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "XHB", "IYT", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.03,
                trigger=-0.14,
                deep=-0.30,
                min_recovery=0.035,
                starter=0.12,
                step=0.16,
                max_risk=0.58,
                vol_ceiling=0.38,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        64: (
            _dip_overlay_candidate(
                name="i64_dip_overlay_liquidity_vol_crush",
                family="dip_overlay_liquidity",
                hypothesis=(
                    "Liquidity overlay buys the dip only after volatility, dollar, credit, and breadth "
                    "pressure stop worsening."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "UUP", "GLD", "TLT", "IEF", "SVXY"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.015,
                trigger=-0.10,
                deep=-0.22,
                starter=0.18,
                step=0.20,
                max_risk=0.76,
                vol_ceiling=0.36,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i64_dip_overlay_credit_spread_repair",
                family="dip_overlay_credit_repair",
                hypothesis=(
                    "Credit-spread overlay redeploys cash into risk when high-yield, loans, banks, and "
                    "investment-grade credit repair together."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "KRE", "SPY", "RSP", "GLD", "IEF"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.01,
                trigger=-0.08,
                deep=-0.18,
                starter=0.22,
                step=0.18,
                max_risk=0.76,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i64_dip_overlay_rates_relief",
                family="dip_overlay_rates",
                hypothesis=(
                    "Rates-relief overlay assumes some selloffs are duration shocks, so cash can "
                    "redeploy into duration, credit, gold, or equities as rates pressure fades."
                ),
                tickers=["SPY", "QQQ", "RSP", "HYG", "LQD", "VCIT", "VCSH", "TLT", "IEF", "TIP", "GLD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.01,
                trigger=-0.09,
                deep=-0.20,
                starter=0.20,
                step=0.18,
                max_risk=0.78,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i64_dip_overlay_private_credit_gate",
                family="dip_overlay_private_credit",
                hypothesis=(
                    "Private-credit gate avoids buying equity dips while BDCs, loans, and regional "
                    "banks are still deteriorating."
                ),
                tickers=["BIZD", "ARCC", "MAIN", "BXSL", "OBDC", "SRLN", "BKLN", "HYG", "LQD", "KRE", "IEF"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.012,
                trigger=-0.10,
                deep=-0.22,
                starter=0.14,
                step=0.16,
                max_risk=0.60,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i64_dip_overlay_gold_duration_bridge",
                family="dip_overlay_defensive_bridge",
                hypothesis=(
                    "Defensive bridge lets gold/duration/cash-like ETFs hold the line, then replaces "
                    "BIL as risk assets repair."
                ),
                tickers=["GLD", "IAU", "TLT", "IEF", "TIP", "SHY", "SGOV", "USFR", "SPY", "RSP", "QQQ"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                trigger=-0.08,
                deep=-0.18,
                starter=0.22,
                step=0.18,
                max_risk=0.72,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i64_dip_overlay_macro_repair_triangle",
                family="dip_overlay_macro",
                hypothesis=(
                    "Macro triangle overlay chooses among equities, credit, duration, gold, dollar, "
                    "and commodities after deep discounts instead of assuming equities lead."
                ),
                tickers=["SPY", "QQQ", "RSP", "HYG", "LQD", "GLD", "TLT", "IEF", "UUP", "DBC", "USO"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.01,
                trigger=-0.10,
                deep=-0.22,
                starter=0.18,
                step=0.20,
                max_risk=0.76,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        65: (
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_core_redeploy",
                family="dip_overlay_final",
                hypothesis=(
                    "Final core overlay: off-ramp first, then measured cash redeployment after broad "
                    "discount plus credit, breadth, volatility, and repair confirmation."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "IEF", "DBC"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.018,
                trigger=-0.10,
                deep=-0.24,
                starter=0.20,
                step=0.20,
                max_risk=0.80,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_quality_value",
                family="dip_overlay_final",
                hypothesis=(
                    "Final quality/value overlay redeploys cash into resilient valuation proxies before "
                    "high-beta market beta."
                ),
                tickers=["QUAL", "USMV", "SPLV", "SCHD", "VIG", "COWZ", "MOAT", "VTV", "RSP", "SPY", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.010,
                trigger=-0.08,
                deep=-0.20,
                starter=0.24,
                step=0.20,
                max_risk=0.84,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_ai_micro",
                family="dip_overlay_final",
                hypothesis=(
                    "Final AI micro overlay participates in AI crash rebounds only through small, "
                    "confirmed, scenario-cut allocations."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "HYG", "LQD", "GLD", "TLT"],
                lookback_days=63,
                skip_days=5,
                top_n=3,
                min_return=0.04,
                trigger=-0.15,
                deep=-0.32,
                min_recovery=0.035,
                starter=0.10,
                step=0.16,
                max_risk=0.50,
                max_asset_weight=0.18,
                vol_ceiling=0.42,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_credit_breadth",
                family="dip_overlay_final",
                hypothesis=(
                    "Final credit/breadth overlay refuses equity re-risking unless public credit and "
                    "equal-weight breadth are repairing."
                ),
                tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "KRE", "SPY", "RSP", "IWM", "GLD", "IEF"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.012,
                trigger=-0.09,
                deep=-0.20,
                starter=0.18,
                step=0.18,
                max_risk=0.72,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_cyclical_broadening",
                family="dip_overlay_final",
                hypothesis=(
                    "Final cyclical overlay buys broadening rebounds only when small/value/cyclical "
                    "discounts begin repairing beyond mega-cap leadership."
                ),
                tickers=["RSP", "IWM", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "COWZ", "QUAL", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.020,
                trigger=-0.11,
                deep=-0.25,
                min_recovery=0.024,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_overlay_candidate(
                name="i65_dip_overlay_final_deep_value_rare",
                family="dip_overlay_final",
                hypothesis=(
                    "Final rare-event overlay only replaces cash after very deep discounts and repair, "
                    "testing whether rare deployment improves compounding without twitchy trading."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "VTV", "COWZ", "QUAL", "HYG", "LQD", "GLD", "IEF"],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.020,
                trigger=-0.18,
                deep=-0.34,
                min_recovery=0.024,
                starter=0.20,
                step=0.22,
                max_risk=0.78,
                vol_ceiling=0.44,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
    }
    return batches[iteration]


def _ai_risk_cycle_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    ai_core = ["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN"]
    ai_concentrated = ["SMH", "SOXX", "NVDA", "AVGO", "MSFT", "META"]
    ai_infra = ["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX"]
    batches = {
        66: (
            _ai_cycle_candidate(
                name="i66_cycle_ai_weekly_low_churn_core",
                family="cycle_ai_plus_low_churn_core",
                hypothesis=(
                    "Layer aggressive AI satellite risk onto the best low-churn reentry posture: "
                    "stay broadly diversified, then let AI replace BIL only after repair is visible."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.02,
                satellite_max=0.42,
                satellite_risk_on=0.32,
                satellite_reentry=0.48,
                trigger=-0.12,
                deep=-0.28,
                recovery_days=42,
                confirmation_days=10,
                min_change=0.04,
                max_step=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i66_cycle_ai_credit_first_defense",
                family="cycle_ai_plus_credit_defense",
                hypothesis=(
                    "Use credit/rates defense as the core; AI can reenter only when high-yield, "
                    "breadth, and AI price repair agree."
                ),
                core_tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "IEF", "TLT", "GLD", "SPY", "RSP"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.01,
                satellite_max=0.35,
                satellite_risk_on=0.25,
                satellite_reentry=0.40,
                trigger=-0.13,
                deep=-0.30,
                min_change=0.04,
                max_step=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i66_cycle_ai_off_ramp_core",
                family="cycle_ai_plus_off_ramp",
                hypothesis=(
                    "Start from the historical off-ramp core, then test whether an AI satellite fixes "
                    "the classic failure of staying defensive too long after selloffs."
                ),
                core_tickers=["SPY", "RSP", "IWM", "GLD", "TLT", "IEF", "DBC", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=3,
                min_return=0.02,
                satellite_max=0.45,
                satellite_risk_on=0.35,
                satellite_reentry=0.50,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.03,
                max_step=0.40,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i66_cycle_ai_macro_triangle",
                family="cycle_ai_plus_macro_triangle",
                hypothesis=(
                    "Let equities, credit, duration, gold, dollar, and commodities decide the core "
                    "regime while AI only receives a satellite budget during confirmed repair."
                ),
                core_tickers=["SPY", "RSP", "HYG", "LQD", "GLD", "TLT", "IEF", "UUP", "DBC", "USO"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.01,
                satellite_max=0.40,
                satellite_risk_on=0.30,
                satellite_reentry=0.48,
                trigger=-0.10,
                deep=-0.24,
                min_change=0.03,
                max_step=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i66_cycle_ai_global_discount",
                family="cycle_ai_plus_global_discount",
                hypothesis=(
                    "Pair AI upside with a global discount core so the system can re-risk outside "
                    "U.S. mega-cap tech when the U.S. setup is crowded."
                ),
                core_tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC", "HYG", "LQD", "UUP"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.015,
                satellite_max=0.35,
                satellite_risk_on=0.25,
                satellite_reentry=0.38,
                trigger=-0.11,
                deep=-0.24,
                min_change=0.04,
                max_step=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i66_cycle_ai_sector_factor_blend",
                family="cycle_ai_plus_sector_factor",
                hypothesis=(
                    "Use sector/factor breadth as the core and allow AI to become the satellite only "
                    "when the market confirms growth leadership is repairing."
                ),
                core_tickers=["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC", "QUAL", "COWZ", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.015,
                satellite_max=0.38,
                satellite_risk_on=0.28,
                satellite_reentry=0.42,
                trigger=-0.10,
                deep=-0.24,
                min_change=0.04,
                max_step=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        67: (
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_ai_slow_confirm",
                family="cycle_state_machine_hysteresis",
                hypothesis=(
                    "A high-hysteresis state machine tests whether the bot can avoid risk-off traps "
                    "by requiring durable signals before moving exposure materially."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.02,
                satellite_max=0.40,
                satellite_risk_on=0.25,
                satellite_reentry=0.45,
                trigger=-0.13,
                deep=-0.29,
                recovery_days=42,
                confirmation_days=10,
                min_change=0.08,
                max_step=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_ai_fast_repair",
                family="cycle_state_machine_hysteresis",
                hypothesis=(
                    "Fast repair state machine tests whether a smaller hysteresis band captures "
                    "earlier rebound juice without turning the system twitchy."
                ),
                core_tickers=["SPY", "RSP", "IWM", "HYG", "LQD", "GLD", "TLT", "IEF"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.02,
                satellite_max=0.48,
                satellite_risk_on=0.35,
                satellite_reentry=0.55,
                trigger=-0.10,
                deep=-0.23,
                recovery_days=21,
                confirmation_days=5,
                min_change=0.025,
                max_step=0.45,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_no_credit_no_ai",
                family="cycle_state_machine_credit_gate",
                hypothesis=(
                    "AI re-risking is forbidden unless credit repair confirms; this directly tests "
                    "whether credit gates prevent false-bottom AI buying."
                ),
                core_tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "SPY", "RSP", "GLD", "IEF"],
                satellite_tickers=ai_concentrated,
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.02,
                satellite_max=0.36,
                satellite_risk_on=0.20,
                satellite_reentry=0.42,
                trigger=-0.15,
                deep=-0.32,
                min_recovery=0.035,
                min_change=0.05,
                max_step=0.25,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_asymmetric_reentry",
                family="cycle_state_machine_asymmetric",
                hypothesis=(
                    "Asymmetric state machine lets AI reenter faster from cash than it adds during "
                    "normal risk-on, testing whether rebounds deserve special treatment."
                ),
                core_tickers=["SPY", "RSP", "IWM", "QUAL", "USMV", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.02,
                satellite_max=0.44,
                satellite_risk_on=0.20,
                satellite_reentry=0.62,
                trigger=-0.12,
                deep=-0.26,
                min_change=0.04,
                max_step=0.32,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_whipsaw_control",
                family="cycle_state_machine_whipsaw",
                hypothesis=(
                    "Whipsaw-control variant deliberately slows all exposure changes to see whether "
                    "less trading improves left-tail and operating quality."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.015,
                satellite_max=0.38,
                satellite_risk_on=0.28,
                satellite_reentry=0.40,
                trigger=-0.12,
                deep=-0.27,
                recovery_days=42,
                confirmation_days=10,
                min_change=0.10,
                max_step=0.18,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i67_cycle_hysteresis_breadth_unlock",
                family="cycle_state_machine_breadth_unlock",
                hypothesis=(
                    "Breadth-unlock state machine tests whether equal-weight and cyclicals should "
                    "unlock AI reentry after broad selloffs."
                ),
                core_tickers=["RSP", "IWM", "VTV", "XLF", "KRE", "XLI", "XLB", "XLE", "COWZ", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.02,
                satellite_max=0.42,
                satellite_risk_on=0.25,
                satellite_reentry=0.50,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.05,
                max_step=0.28,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        68: (
            _ai_cycle_candidate(
                name="i68_cycle_barbell_credit_ai",
                family="cycle_barbell_credit_ai",
                hypothesis=(
                    "Barbell allocator holds credit/rates defense until AI earns a satellite budget "
                    "through price repair and credit confirmation."
                ),
                core_tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "IEF", "TLT", "GLD", "SHY", "SGOV"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                satellite_max=0.36,
                satellite_risk_on=0.20,
                satellite_reentry=0.45,
                trigger=-0.12,
                deep=-0.28,
                min_change=0.04,
                max_step=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i68_cycle_barbell_gold_duration_ai",
                family="cycle_barbell_gold_duration_ai",
                hypothesis=(
                    "Gold/duration bridge protects the portfolio, then AI can take a satellite budget "
                    "when volatility and repair conditions improve."
                ),
                core_tickers=["GLD", "IAU", "TLT", "IEF", "TIP", "SHY", "SGOV", "USFR", "SPY", "RSP", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                satellite_max=0.34,
                satellite_risk_on=0.18,
                satellite_reentry=0.42,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.04,
                max_step=0.24,
                breadth_confirmation=False,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i68_cycle_barbell_policy_oil_ai",
                family="cycle_barbell_policy_oil_ai",
                hypothesis=(
                    "Policy/oil shock barbell competes energy, gold, duration, dollar, and AI, testing "
                    "whether AI should reenter when geopolitical shock fades."
                ),
                core_tickers=["SPY", "RSP", "XLE", "XOP", "USO", "BNO", "DBC", "GLD", "UUP", "TLT", "IEF", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.01,
                satellite_max=0.38,
                satellite_risk_on=0.25,
                satellite_reentry=0.42,
                trigger=-0.12,
                deep=-0.27,
                min_change=0.04,
                max_step=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i68_cycle_barbell_private_credit_gate",
                family="cycle_barbell_private_credit_ai",
                hypothesis=(
                    "Private-credit gate blocks AI reentry while BDCs, loans, and regional banks are "
                    "still breaking, then allows AI once credit stress repairs."
                ),
                core_tickers=["BIZD", "ARCC", "MAIN", "BXSL", "OBDC", "SRLN", "BKLN", "HYG", "LQD", "KRE", "IEF"],
                satellite_tickers=ai_concentrated,
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.012,
                satellite_max=0.32,
                satellite_risk_on=0.16,
                satellite_reentry=0.38,
                trigger=-0.13,
                deep=-0.30,
                min_change=0.05,
                max_step=0.22,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i68_cycle_barbell_rates_relief_ai",
                family="cycle_barbell_rates_relief_ai",
                hypothesis=(
                    "Rates-relief core tests if duration and credit should turn first, then AI follows "
                    "as a satellite only after repair."
                ),
                core_tickers=["SPY", "RSP", "HYG", "LQD", "VCIT", "VCSH", "TLT", "IEF", "TIP", "GLD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.01,
                satellite_max=0.36,
                satellite_risk_on=0.22,
                satellite_reentry=0.42,
                trigger=-0.10,
                deep=-0.23,
                min_change=0.04,
                max_step=0.28,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i68_cycle_barbell_liquidity_ai",
                family="cycle_barbell_liquidity_ai",
                hypothesis=(
                    "Liquidity-volatility barbell uses SVXY, credit, dollar, gold, and duration as "
                    "the regime core before giving AI a reentry budget."
                ),
                core_tickers=["SPY", "RSP", "HYG", "LQD", "UUP", "GLD", "TLT", "IEF", "SVXY", "SHY", "SGOV"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.01,
                satellite_max=0.40,
                satellite_risk_on=0.25,
                satellite_reentry=0.48,
                trigger=-0.10,
                deep=-0.23,
                min_change=0.035,
                max_step=0.30,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        69: (
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_ai_semis_reentry",
                family="cycle_aggressive_ai_reentry",
                hypothesis=(
                    "Aggressive semis reentry tests whether the system can buy high-convexity AI "
                    "after deep discounts without overriding risk controls."
                ),
                core_tickers=["SPY", "RSP", "QUAL", "USMV", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_concentrated,
                lookback_days=63,
                skip_days=5,
                top_n=3,
                min_return=0.03,
                satellite_max=0.58,
                satellite_risk_on=0.42,
                satellite_reentry=0.68,
                trigger=-0.16,
                deep=-0.34,
                min_recovery=0.040,
                vol_ceiling=0.46,
                min_change=0.04,
                max_step=0.42,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_ai_mega_platform",
                family="cycle_aggressive_ai_platform",
                hypothesis=(
                    "Mega-platform reentry tests whether the best AI rebound is concentrated in "
                    "platform leaders rather than broad QQQ."
                ),
                core_tickers=["SPY", "RSP", "QUAL", "COWZ", "GLD", "IEF", "HYG", "LQD"],
                satellite_tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.03,
                satellite_max=0.55,
                satellite_risk_on=0.40,
                satellite_reentry=0.62,
                trigger=-0.13,
                deep=-0.28,
                min_recovery=0.030,
                vol_ceiling=0.42,
                min_change=0.04,
                max_step=0.38,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_ai_infra_reentry",
                family="cycle_aggressive_ai_infra",
                hypothesis=(
                    "AI infrastructure reentry tests whether power/grid/hardware beneficiaries offer "
                    "better post-risk-off reentry than pure software or mega-cap beta."
                ),
                core_tickers=["SPY", "RSP", "XLI", "XLU", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_infra,
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.025,
                satellite_max=0.48,
                satellite_risk_on=0.35,
                satellite_reentry=0.56,
                trigger=-0.12,
                deep=-0.27,
                min_recovery=0.028,
                vol_ceiling=0.44,
                min_change=0.04,
                max_step=0.35,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_high_beta_micro",
                family="cycle_aggressive_high_beta_probe",
                hypothesis=(
                    "High-beta micro sleeve tests whether speculative rebound juice is worth a small, "
                    "strictly capped allocation after deep washouts."
                ),
                core_tickers=["SPY", "RSP", "GLD", "TLT", "HYG", "LQD", "QUAL", "USMV"],
                satellite_tickers=["SPHB", "ARKK", "IBIT", "FBTC", "XBI", "TAN", "BOTZ", "QQQ"],
                lookback_days=42,
                skip_days=5,
                top_n=2,
                min_return=0.05,
                satellite_max=0.28,
                satellite_risk_on=0.16,
                satellite_reentry=0.34,
                trigger=-0.20,
                deep=-0.40,
                min_recovery=0.055,
                vol_ceiling=0.62,
                min_change=0.05,
                max_step=0.22,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_ai_escape_fast",
                family="cycle_aggressive_ai_escape",
                hypothesis=(
                    "Fast AI escape reentry tests the explicit V1-V3 failure mode: do not stay risk-off "
                    "once AI leadership and credit repair return."
                ),
                core_tickers=["SPY", "RSP", "IWM", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.035,
                satellite_max=0.60,
                satellite_risk_on=0.45,
                satellite_reentry=0.70,
                trigger=-0.12,
                deep=-0.27,
                min_recovery=0.025,
                vol_ceiling=0.48,
                min_change=0.025,
                max_step=0.48,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i69_cycle_aggressive_ai_escape_strict",
                family="cycle_aggressive_ai_escape",
                hypothesis=(
                    "Strict AI escape reentry tests whether slower confirmation avoids false starts while "
                    "still solving the stuck-in-cash problem."
                ),
                core_tickers=["SPY", "RSP", "IWM", "QUAL", "USMV", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.035,
                satellite_max=0.48,
                satellite_risk_on=0.30,
                satellite_reentry=0.56,
                trigger=-0.16,
                deep=-0.32,
                recovery_days=42,
                confirmation_days=10,
                min_recovery=0.035,
                vol_ceiling=0.42,
                min_change=0.06,
                max_step=0.30,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        70: (
            _ai_cycle_candidate(
                name="i70_cycle_diverse_active_mega_cap_escape",
                family="cycle_diverse_active_mega_cap",
                hypothesis=(
                    "Active mega-cap escape becomes a parent core while AI satellite reentry is gated "
                    "by drawdown repair rather than raw recent winner chasing."
                ),
                core_tickers=["SPY", "RSP", "QUAL", "COWZ", "MTUM", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA"],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.03,
                satellite_max=0.50,
                satellite_risk_on=0.38,
                satellite_reentry=0.55,
                trigger=-0.12,
                deep=-0.27,
                min_change=0.04,
                max_step=0.35,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i70_cycle_diverse_spec_liquidity_ai",
                family="cycle_diverse_spec_liquidity",
                hypothesis=(
                    "Speculative liquidity parent tests whether AI and high-beta assets should only "
                    "activate after liquidity/volatility repair."
                ),
                core_tickers=["SVXY", "HYG", "LQD", "UUP", "GLD", "TLT", "SPY", "RSP", "SHY", "SGOV"],
                satellite_tickers=["QQQ", "SMH", "SOXX", "ARKK", "SPHB", "IBIT", "FBTC", "XBI"],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                min_return=0.04,
                satellite_max=0.38,
                satellite_risk_on=0.26,
                satellite_reentry=0.44,
                trigger=-0.14,
                deep=-0.30,
                min_recovery=0.035,
                vol_ceiling=0.55,
                min_change=0.05,
                max_step=0.28,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i70_cycle_diverse_sector_breadth_ai",
                family="cycle_diverse_sector_breadth",
                hypothesis=(
                    "Sector breadth parent checks whether AI should reenter only after broader sector "
                    "leadership confirms the move."
                ),
                core_tickers=["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC", "RSP", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.015,
                satellite_max=0.38,
                satellite_risk_on=0.25,
                satellite_reentry=0.42,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.04,
                max_step=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i70_cycle_diverse_policy_oil_ai",
                family="cycle_diverse_policy_oil",
                hypothesis=(
                    "Policy/oil parent tests whether AI reentry can coexist with shock-aware energy, "
                    "gold, dollar, and duration allocations."
                ),
                core_tickers=["SPY", "RSP", "XLE", "XOP", "USO", "BNO", "DBC", "GLD", "UUP", "TLT", "IEF", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.01,
                satellite_max=0.36,
                satellite_risk_on=0.22,
                satellite_reentry=0.40,
                trigger=-0.12,
                deep=-0.27,
                min_change=0.05,
                max_step=0.26,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i70_cycle_diverse_global_macro_ai",
                family="cycle_diverse_global_macro",
                hypothesis=(
                    "Global macro parent tests whether AI satellite risk is useful even when the best "
                    "core regime expression is outside U.S. equities."
                ),
                core_tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "INDA", "EWZ", "GLD", "UUP", "DBC", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.012,
                satellite_max=0.34,
                satellite_risk_on=0.22,
                satellite_reentry=0.38,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.05,
                max_step=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i70_cycle_diverse_final_core_ai",
                family="cycle_diverse_final_core",
                hypothesis=(
                    "Final diverse core combines broad equity, credit, duration, commodities, and AI "
                    "satellite reentry as a candidate operating architecture."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "TLT", "IEF", "DBC"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.015,
                satellite_max=0.42,
                satellite_risk_on=0.30,
                satellite_reentry=0.48,
                trigger=-0.11,
                deep=-0.26,
                min_change=0.04,
                max_step=0.32,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        71: (
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_whipsaw_control",
                family="cycle_cooldown_whipsaw",
                hypothesis=(
                    "Cooldown version of the best hysteresis candidate: require a minimum hold period "
                    "unless a risk-off override fires, aiming to reduce noisy re-risk/de-risk cycles."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.015,
                satellite_max=0.38,
                satellite_risk_on=0.28,
                satellite_reentry=0.40,
                trigger=-0.12,
                deep=-0.27,
                recovery_days=42,
                confirmation_days=10,
                min_change=0.10,
                max_step=0.18,
                min_hold_days=10,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_macro_triangle",
                family="cycle_cooldown_macro_triangle",
                hypothesis=(
                    "Cooldown macro triangle keeps the strong macro/AI blend but forces signals to "
                    "persist before target weights move again."
                ),
                core_tickers=["SPY", "RSP", "HYG", "LQD", "GLD", "TLT", "IEF", "UUP", "DBC", "USO"],
                satellite_tickers=ai_core,
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.01,
                satellite_max=0.40,
                satellite_risk_on=0.30,
                satellite_reentry=0.48,
                trigger=-0.10,
                deep=-0.24,
                min_change=0.08,
                max_step=0.20,
                min_hold_days=8,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_final_core",
                family="cycle_cooldown_final_core",
                hypothesis=(
                    "Cooldown final-core variant tests whether a top diverse operating architecture "
                    "can keep most return while trading less often."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "TLT", "IEF", "DBC"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.015,
                satellite_max=0.42,
                satellite_risk_on=0.30,
                satellite_reentry=0.48,
                trigger=-0.11,
                deep=-0.26,
                min_change=0.08,
                max_step=0.22,
                min_hold_days=8,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_credit_barbell",
                family="cycle_cooldown_credit_barbell",
                hypothesis=(
                    "Cooldown credit barbell prioritizes stability and lets AI reenter slowly from a "
                    "credit/rates defensive core."
                ),
                core_tickers=["HYG", "JNK", "LQD", "BKLN", "SRLN", "JAAA", "JBBB", "IEF", "TLT", "GLD", "SHY", "SGOV"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                satellite_max=0.36,
                satellite_risk_on=0.20,
                satellite_reentry=0.45,
                trigger=-0.12,
                deep=-0.28,
                min_change=0.08,
                max_step=0.18,
                min_hold_days=10,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_ai_escape",
                family="cycle_cooldown_ai_escape",
                hypothesis=(
                    "Cooldown AI escape tests whether aggressive reentry can stay viable when every "
                    "target move must persist long enough to be human-operable."
                ),
                core_tickers=["SPY", "RSP", "IWM", "QUAL", "USMV", "GLD", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.035,
                satellite_max=0.48,
                satellite_risk_on=0.30,
                satellite_reentry=0.56,
                trigger=-0.16,
                deep=-0.32,
                recovery_days=42,
                confirmation_days=10,
                min_recovery=0.035,
                vol_ceiling=0.42,
                min_change=0.08,
                max_step=0.20,
                min_hold_days=8,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _ai_cycle_candidate(
                name="i71_cycle_cooldown_global_macro",
                family="cycle_cooldown_global_macro",
                hypothesis=(
                    "Cooldown global macro tests whether global diversification plus AI satellite can "
                    "solve reentry without frequent target churn."
                ),
                core_tickers=["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "INDA", "EWZ", "GLD", "UUP", "DBC", "TLT", "HYG", "LQD"],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.012,
                satellite_max=0.34,
                satellite_risk_on=0.22,
                satellite_reentry=0.38,
                trigger=-0.11,
                deep=-0.25,
                min_change=0.08,
                max_step=0.18,
                min_hold_days=10,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
    }
    return batches[iteration]



def _sector_regime_rotation_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    sector_spdrs = ["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC"]
    sector_expanded = [
        "XLK",
        "XLF",
        "KRE",
        "XLY",
        "XLP",
        "XLE",
        "XLV",
        "XLI",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        "XME",
        "XRT",
        "XHB",
        "XBI",
        "RSP",
        "IWM",
    ]
    defensive_assets = ["BIL", "SGOV", "SHY", "IEF", "TLT", "GLD", "IAU", "UUP", "LQD"]
    ai_themes = ["QQQ", "SMH", "SOXX", "IGV", "CLOU", "SKYY", "BOTZ", "ARKK", "XLK", "XLC"]
    ai_infra = ["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX", "XLU", "XLI"]
    reflation = ["XLE", "XOP", "USO", "BNO", "DBC", "XLB", "XLI", "XLF", "KRE", "IWM", "RSP", "COWZ"]
    factors = ["VUG", "VTV", "IWF", "IWD", "MTUM", "QUAL", "USMV", "SPLV", "SCHD", "VIG", "MOAT", "COWZ"]
    global_assets = ["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC"]
    credit_assets = ["HYG", "JNK", "LQD", "VCIT", "VCSH", "BKLN", "SRLN", "JAAA", "JBBB"]
    batches = {
        72: (
            _sector_regime_candidate(
                name="i72_sector_regime_classic_spdr",
                family="sector_regime_spdr",
                hypothesis=(
                    "Classic sector rotation should not be only momentum: sector leadership competes "
                    "with credit, breadth, and defensive assets before total risk is allowed."
                ),
                tickers=[*sector_spdrs, *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.005,
                max_asset_weight=0.30,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i72_sector_regime_expanded_industries",
                family="sector_regime_expanded",
                hypothesis=(
                    "Expanded industries test whether homebuilders, retail, biotech, metals, and banks "
                    "carry useful early-cycle or late-cycle signals beyond broad SPDR sectors."
                ),
                tickers=[*sector_expanded, *defensive_assets, "HYG", "LQD", "SPY"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i72_sector_regime_factor_sector_blend",
                family="sector_regime_factor_blend",
                hypothesis=(
                    "Sector plus factor rotation should distinguish defensive quality/low-vol leadership "
                    "from genuine risk-on growth or cyclical leadership."
                ),
                tickers=[*sector_spdrs, *factors, *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i72_sector_regime_ai_theme",
                family="sector_regime_ai_theme",
                hypothesis=(
                    "AI theme rotation tests whether the system can own semis/software/cloud/robotics "
                    "when AI leadership is confirmed and route away when breadth or credit disagrees."
                ),
                tickers=[*ai_themes, *sector_spdrs, *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.010,
                max_asset_weight=0.28,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i72_sector_regime_reflation_cycle",
                family="sector_regime_reflation",
                hypothesis=(
                    "Reflation/cyclical rotation tests whether energy, materials, banks, and industrials "
                    "should replace QQQ/SPY leadership when commodity or breadth regimes shift."
                ),
                tickers=[*reflation, *sector_spdrs, *defensive_assets, "HYG", "LQD", "SPY"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.005,
                max_asset_weight=0.28,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i72_sector_regime_global_us_mix",
                family="sector_regime_global",
                hypothesis=(
                    "Global plus sector rotation tests whether U.S. mega-cap concentration should be "
                    "escaped into international or non-tech sectors when relative leadership changes."
                ),
                tickers=[*global_assets, *sector_spdrs, *defensive_assets, "HYG", "LQD", "UUP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        73: (
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_spdr",
                family="sector_regime_low_churn",
                hypothesis=(
                    "Low-churn SPDR rotation tests whether sector regimes are tradable for a human only "
                    "when target changes clear a material threshold and persist for two weeks."
                ),
                tickers=[*sector_spdrs, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                min_return=0.005,
                min_change=0.08,
                max_step=0.20,
                min_hold_days=10,
                max_asset_weight=0.28,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_factor",
                family="sector_regime_low_churn_factor",
                hypothesis=(
                    "Low-churn factor/sector blend checks whether quality, dividends, and low-volatility "
                    "can reduce sector whipsaw without simply hiding in cash."
                ),
                tickers=[*sector_spdrs, *factors, *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=126,
                skip_days=21,
                top_n=5,
                min_return=0.005,
                min_change=0.08,
                max_step=0.20,
                min_hold_days=10,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_ai_infra",
                family="sector_regime_low_churn_ai_infra",
                hypothesis=(
                    "AI-infrastructure sector rotation should not be twitchy: power/grid/semis only win "
                    "when the broader regime allows risk and the signal persists."
                ),
                tickers=[*ai_infra, *sector_spdrs, *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.010,
                min_change=0.07,
                max_step=0.22,
                min_hold_days=8,
                max_asset_weight=0.26,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_reflation",
                family="sector_regime_low_churn_reflation",
                hypothesis=(
                    "Reflation leadership is often episodic; this tests whether slow confirmation can "
                    "capture oil/banks/materials without chasing every price spike."
                ),
                tickers=[*reflation, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.005,
                min_change=0.08,
                max_step=0.20,
                min_hold_days=10,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_global",
                family="sector_regime_low_churn_global",
                hypothesis=(
                    "Global low-churn rotation tests whether regime changes can move outside U.S. equity "
                    "leadership without increasing operating burden."
                ),
                tickers=[*global_assets, *sector_spdrs, *defensive_assets, "HYG", "LQD", "UUP"],
                lookback_days=126,
                skip_days=21,
                top_n=5,
                min_return=0.005,
                min_change=0.08,
                max_step=0.18,
                min_hold_days=10,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i73_sector_regime_low_churn_credit_sector",
                family="sector_regime_low_churn_credit",
                hypothesis=(
                    "Credit-sector rotation checks whether HYG/LQD/loans can serve as intermediate risk "
                    "rather than forcing a binary cash versus equity decision."
                ),
                tickers=[*credit_assets, *sector_spdrs, *defensive_assets, "SPY", "RSP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.003,
                min_change=0.07,
                max_step=0.18,
                min_hold_days=8,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        74: (
            _sector_regime_candidate(
                name="i74_sector_regime_ai_capex_basket",
                family="sector_regime_ai_capex",
                hypothesis=(
                    "AI capex basket tests whether the right response to AI-bubble risk is rotation among "
                    "semis, software, power, industrials, and utilities rather than only QQQ versus cash."
                ),
                tickers=[*ai_themes, *ai_infra, "XLI", "XLU", "XLF", *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.010,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i74_sector_regime_ai_escape_without_mega",
                family="sector_regime_ai_ex_mega",
                hypothesis=(
                    "Ex-mega-cap AI rotation tests whether AI leadership can be expressed through themes "
                    "and infrastructure without defaulting to QQQ or mega-cap platforms."
                ),
                tickers=["SMH", "SOXX", "IGV", "CLOU", "SKYY", "BOTZ", "ROBO", "VRT", "ETN", "PWR", "CEG", "GEV", "XLU", "XLI", *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.012,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i74_sector_regime_power_grid_barbell",
                family="sector_regime_power_grid",
                hypothesis=(
                    "Power/grid barbell tests whether AI infrastructure creates tradable leadership in "
                    "utilities, industrials, uranium, and grid names before or after semis lead."
                ),
                tickers=["XLU", "XLI", "VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "URA", "SMH", "SOXX", *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.010,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i74_sector_regime_software_hardware_split",
                family="sector_regime_software_hardware",
                hypothesis=(
                    "Software versus hardware split tests whether AI unit-economics pressure should rotate "
                    "away from software/cloud and into semis or infrastructure when leadership diverges."
                ),
                tickers=["IGV", "CLOU", "SKYY", "SMH", "SOXX", "BOTZ", "ROBO", "XLK", "XLC", "XLI", "XLU", *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.010,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i74_sector_regime_ai_defensive_quality",
                family="sector_regime_ai_quality",
                hypothesis=(
                    "AI with quality/defensive ballast tests whether the system can stay invested in "
                    "durable compounders while reducing pure AI beta during fragile regimes."
                ),
                tickers=[*ai_themes, "QUAL", "USMV", "SPLV", "SCHD", "VIG", "MOAT", "XLV", "XLP", "XLU", *defensive_assets, "SPY", "RSP", "HYG"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.008,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i74_sector_regime_ai_reentry_fast",
                family="sector_regime_ai_reentry",
                hypothesis=(
                    "Fast AI sector reentry tests the stuck-in-cash failure mode by allowing AI themes "
                    "to reclaim risk when repair, breadth, and credit improve together."
                ),
                tickers=[*ai_themes, *ai_infra, *sector_expanded, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=42,
                skip_days=5,
                top_n=5,
                min_return=0.012,
                min_change=0.04,
                max_step=0.35,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        75: (
            _sector_regime_candidate(
                name="i75_sector_regime_policy_oil_shock",
                family="sector_regime_policy_oil",
                hypothesis=(
                    "Policy/oil shock rotation tests whether energy, gold, dollar, and duration can win "
                    "while growth sectors step aside during geopolitical or inflation shocks."
                ),
                tickers=[*reflation, "GLD", "IAU", "UUP", "TLT", "IEF", "SHY", "BIL", "XLP", "XLU", "XLV", "SPY", "RSP", "HYG", "LQD"],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.005,
                max_asset_weight=0.26,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="i75_sector_regime_rates_relief",
                family="sector_regime_rates_relief",
                hypothesis=(
                    "Rates-relief rotation tests whether duration, real estate, growth, and credit should "
                    "lead re-risking when inflation and rates pressure eases."
                ),
                tickers=["TLT", "IEF", "TIP", "LQD", "HYG", "XLRE", "XLU", "XLK", "XLY", "QQQ", "SMH", "SPY", "RSP", "BIL", "SGOV", "SHY", "GLD"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.005,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i75_sector_regime_private_credit_warning",
                family="sector_regime_private_credit",
                hypothesis=(
                    "Private-credit warning rotation tests whether BDCs, loans, banks, and credit ETFs "
                    "warn before equity sectors, and whether the system should route to cash/duration."
                ),
                tickers=["BIZD", "ARCC", "MAIN", "BXSL", "OBDC", "SRLN", "BKLN", "JAAA", "JBBB", "KRE", "XLF", "HYG", "LQD", "IEF", "TLT", "BIL", "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.004,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="i75_sector_regime_inflation_barbell",
                family="sector_regime_inflation_barbell",
                hypothesis=(
                    "Inflation barbell rotates among commodities, energy, materials, dollar, gold, and "
                    "defensive equity instead of treating inflation shocks as a simple sell signal."
                ),
                tickers=["DBC", "DBA", "USO", "BNO", "XLE", "XOP", "XLB", "XME", "GLD", "UUP", "XLP", "XLU", "XLV", "BIL", "SHY", "IEF", "SPY", "RSP", "HYG"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.005,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="i75_sector_regime_defensive_equity_bridge",
                family="sector_regime_defensive_bridge",
                hypothesis=(
                    "Defensive-equity bridge tests whether healthcare, staples, utilities, quality, and "
                    "low-vol can keep compounding when risk is yellow but not fully red."
                ),
                tickers=["XLV", "XLP", "XLU", "QUAL", "USMV", "SPLV", "SCHD", "VIG", "MOAT", "GLD", "TLT", "IEF", "BIL", "SPY", "RSP", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.004,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i75_sector_regime_credit_then_sector",
                family="sector_regime_credit_then_sector",
                hypothesis=(
                    "Credit-then-sector rotation tests whether credit repair should unlock sector risk "
                    "before broad equity momentum fully confirms."
                ),
                tickers=[*credit_assets, *sector_expanded, *defensive_assets, "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.004,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        76: (
            _sector_regime_candidate(
                name="i76_sector_regime_final_core",
                family="sector_regime_final_core",
                hypothesis=(
                    "Final core sector-regime system combines sector breadth, factors, credit, duration, "
                    "gold, and cash with explicit regime-controlled risk sizing."
                ),
                tickers=[*sector_spdrs, *factors, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                min_change=0.06,
                max_step=0.25,
                min_hold_days=5,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i76_sector_regime_final_low_churn",
                family="sector_regime_final_low_churn",
                hypothesis=(
                    "Final low-churn sector-regime system sacrifices some responsiveness to improve human "
                    "operability and reduce small frequent reallocations."
                ),
                tickers=[*sector_spdrs, *factors, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=126,
                skip_days=21,
                top_n=5,
                min_return=0.005,
                min_change=0.09,
                max_step=0.18,
                min_hold_days=10,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i76_sector_regime_final_ai_capex",
                family="sector_regime_final_ai_capex",
                hypothesis=(
                    "Final AI-capex sector-regime system allows semis, software, power/grid, quality, "
                    "and defensive assets to compete rather than hard-coding QQQ exposure."
                ),
                tickers=[*ai_themes, *ai_infra, *factors, "XLV", "XLP", "XLU", *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.010,
                min_change=0.06,
                max_step=0.25,
                min_hold_days=5,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="i76_sector_regime_final_macro_barbell",
                family="sector_regime_final_macro_barbell",
                hypothesis=(
                    "Final macro barbell tests sector rotation under oil, dollar, duration, credit, and "
                    "defensive-equity regimes for non-AI market transitions."
                ),
                tickers=[*reflation, "XLV", "XLP", "XLU", "GLD", "IAU", "UUP", "TLT", "IEF", "SHY", "BIL", "HYG", "LQD", "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.005,
                min_change=0.06,
                max_step=0.24,
                min_hold_days=5,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="i76_sector_regime_final_global_rotation",
                family="sector_regime_final_global",
                hypothesis=(
                    "Final global sector-regime system tests whether leadership can migrate outside U.S. "
                    "mega-cap technology while preserving off-ramp discipline."
                ),
                tickers=[*global_assets, *sector_spdrs, *factors, *defensive_assets, "HYG", "LQD", "UUP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                min_change=0.06,
                max_step=0.22,
                min_hold_days=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="i76_sector_regime_final_credit_reentry",
                family="sector_regime_final_credit_reentry",
                hypothesis=(
                    "Final credit-reentry system lets credit and sectors rebuild together after stress, "
                    "targeting the gap between cash defense and early risk-on participation."
                ),
                tickers=[*credit_assets, *sector_expanded, *factors, *defensive_assets, "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=6,
                min_return=0.004,
                min_change=0.06,
                max_step=0.23,
                min_hold_days=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
    }
    return batches[iteration]

def _dip_reentry_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    tickers: list[str],
    trigger: float,
    deep: float,
    min_recovery: float = 0.015,
    starter: float = 0.20,
    step: float = 0.20,
    max_risk: float = 0.75,
    top_n: int = 4,
    max_asset_weight: float = 0.25,
    recovery_days: int = 21,
    confirmation_days: int = 5,
    trend_filter_days: int | None = 100,
    vol_ceiling: float = 0.34,
    credit_confirmation: bool = True,
    breadth_confirmation: bool = True,
    scenario_sizing: ScenarioSizingConfig | None = None,
    role: str = "reentry_candidate",
    phase: str = "dip_reentry",
) -> ExperimentCandidate:
    return _candidate(
        name=name,
        role=role,
        phase=phase,
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="dip_reentry",
            tickers=tickers,
            defensive_ticker="BIL",
            top_n=top_n,
            weighting="risk_adjusted_score",
            volatility_lookback_days=63,
            trend_filter_days=trend_filter_days,
            max_asset_weight=max_asset_weight,
            dip_trigger_drawdown=trigger,
            dip_deep_drawdown=deep,
            dip_recovery_days=recovery_days,
            dip_confirmation_days=confirmation_days,
            dip_min_recovery_return=min_recovery,
            dip_starter_weight=starter,
            dip_step_weight=step,
            dip_max_risk_weight=max_risk,
            dip_volatility_ceiling=vol_ceiling,
            dip_credit_confirmation=credit_confirmation,
            dip_breadth_confirmation=breadth_confirmation,
        ),
    )


def _dip_overlay_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    tickers: list[str],
    lookback_days: int,
    skip_days: int,
    top_n: int,
    min_return: float,
    trigger: float,
    deep: float,
    min_recovery: float = 0.015,
    starter: float = 0.20,
    step: float = 0.20,
    max_risk: float = 0.80,
    max_asset_weight: float = 0.25,
    recovery_days: int = 21,
    confirmation_days: int = 5,
    trend_filter_days: int | None = 100,
    vol_ceiling: float = 0.34,
    credit_confirmation: bool = True,
    breadth_confirmation: bool = True,
    scenario_sizing: ScenarioSizingConfig | None = None,
) -> ExperimentCandidate:
    return _candidate(
        name=name,
        role="reentry_overlay_candidate",
        phase="dip_reentry_overlay",
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="dip_reentry_overlay",
            tickers=tickers,
            defensive_ticker="BIL",
            lookback_days=lookback_days,
            skip_days=skip_days,
            top_n=top_n,
            min_return=min_return,
            ranking_metric="risk_adjusted_return",
            weighting="risk_adjusted_score",
            volatility_lookback_days=63,
            trend_filter_days=trend_filter_days,
            max_asset_weight=max_asset_weight,
            dip_trigger_drawdown=trigger,
            dip_deep_drawdown=deep,
            dip_recovery_days=recovery_days,
            dip_confirmation_days=confirmation_days,
            dip_min_recovery_return=min_recovery,
            dip_starter_weight=starter,
            dip_step_weight=step,
            dip_max_risk_weight=max_risk,
            dip_volatility_ceiling=vol_ceiling,
            dip_credit_confirmation=credit_confirmation,
            dip_breadth_confirmation=breadth_confirmation,
        ),
    )



def _sector_regime_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    tickers: list[str],
    lookback_days: int,
    skip_days: int,
    top_n: int,
    min_return: float,
    min_change: float = 0.04,
    max_step: float = 0.32,
    min_hold_days: int = 0,
    max_asset_weight: float = 0.25,
    trigger: float = -0.10,
    deep: float = -0.24,
    min_recovery: float = 0.018,
    max_risk: float = 0.95,
    vol_ceiling: float = 0.38,
    scenario_sizing: ScenarioSizingConfig | None = None,
) -> ExperimentCandidate:
    proxy_tickers = [
        "BIL",
        "SPY",
        "RSP",
        "QQQ",
        "SMH",
        "XLK",
        "XLE",
        "XLI",
        "XLF",
        "HYG",
        "LQD",
        "TLT",
        "IEF",
        "SHY",
        "DBC",
    ]
    strategy_tickers = list(dict.fromkeys([*tickers, *proxy_tickers]))
    return _candidate(
        name=name,
        role="sector_regime_candidate",
        phase="sector_regime_rotation",
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="sector_regime_rotation",
            tickers=strategy_tickers,
            defensive_ticker="BIL",
            lookback_days=lookback_days,
            skip_days=skip_days,
            top_n=top_n,
            min_return=min_return,
            ranking_metric="risk_adjusted_return",
            weighting="risk_adjusted_score",
            volatility_lookback_days=63,
            trend_filter_days=100,
            max_asset_weight=max_asset_weight,
            dip_lookback_days=126,
            dip_trigger_drawdown=trigger,
            dip_deep_drawdown=deep,
            dip_recovery_days=21,
            dip_confirmation_days=5,
            dip_min_recovery_return=min_recovery,
            dip_starter_weight=0.25,
            dip_step_weight=0.35,
            dip_max_risk_weight=max_risk,
            dip_volatility_ceiling=vol_ceiling,
            dip_credit_confirmation=True,
            dip_breadth_confirmation=True,
            cycle_min_rebalance_change=min_change,
            cycle_max_step_change=max_step,
            cycle_min_hold_days=min_hold_days,
        ),
    )

def _ai_cycle_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    core_tickers: list[str],
    satellite_tickers: list[str],
    lookback_days: int,
    skip_days: int,
    top_n: int,
    min_return: float,
    satellite_max: float,
    satellite_risk_on: float,
    satellite_reentry: float,
    trigger: float,
    deep: float,
    min_recovery: float = 0.020,
    recovery_days: int = 21,
    confirmation_days: int = 5,
    min_change: float = 0.04,
    max_step: float = 0.35,
    min_hold_days: int = 0,
    max_asset_weight: float = 0.30,
    vol_ceiling: float = 0.40,
    credit_confirmation: bool = True,
    breadth_confirmation: bool = True,
    scenario_sizing: ScenarioSizingConfig | None = None,
) -> ExperimentCandidate:
    tickers = list(
        dict.fromkeys(
            [
                *core_tickers,
                *satellite_tickers,
                "SPY",
                "RSP",
                "HYG",
                "LQD",
            ]
        )
    )
    return _candidate(
        name=name,
        role="risk_cycle_candidate",
        phase="ai_risk_cycle",
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="ai_risk_cycle_overlay",
            tickers=tickers,
            satellite_tickers=list(dict.fromkeys(satellite_tickers)),
            defensive_ticker="BIL",
            lookback_days=lookback_days,
            skip_days=skip_days,
            top_n=top_n,
            min_return=min_return,
            ranking_metric="risk_adjusted_return",
            weighting="risk_adjusted_score",
            volatility_lookback_days=63,
            trend_filter_days=100,
            max_asset_weight=max_asset_weight,
            dip_trigger_drawdown=trigger,
            dip_deep_drawdown=deep,
            dip_recovery_days=recovery_days,
            dip_confirmation_days=confirmation_days,
            dip_min_recovery_return=min_recovery,
            dip_starter_weight=0.18,
            dip_step_weight=0.22,
            dip_max_risk_weight=min(0.95, satellite_max + 0.20),
            dip_volatility_ceiling=vol_ceiling,
            dip_credit_confirmation=credit_confirmation,
            dip_breadth_confirmation=breadth_confirmation,
            cycle_satellite_max_weight=satellite_max,
            cycle_satellite_risk_on_weight=satellite_risk_on,
            cycle_satellite_reentry_weight=satellite_reentry,
            cycle_min_rebalance_change=min_change,
            cycle_max_step_change=max_step,
            cycle_min_hold_days=min_hold_days,
        ),
    )


def _active_dual_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    tickers: list[str],
    lookback_days: int,
    skip_days: int,
    top_n: int,
    min_return: float = 0.0,
    ranking_metric: str = "return",
    weighting: str = "equal",
    trend_filter_days: int | None = None,
    max_asset_weight: float | None = None,
    volatility_target: VolatilityTargetConfig | None = None,
    drawdown_control: DrawdownControlConfig | None = None,
    scenario_sizing: ScenarioSizingConfig | None = None,
    role: str = "active_candidate",
    phase: str = "active_trading",
) -> ExperimentCandidate:
    return _candidate(
        name=name,
        role=role,
        phase=phase,
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="dual_momentum",
            tickers=tickers,
            lookback_days=lookback_days,
            skip_days=skip_days,
            top_n=top_n,
            defensive_ticker="BIL",
            min_return=min_return,
            ranking_metric=ranking_metric,
            weighting=weighting,
            trend_filter_days=trend_filter_days,
            max_asset_weight=max_asset_weight,
            volatility_target=volatility_target,
            drawdown_control=drawdown_control,
        ),
    )


def _active_absolute_candidate(
    *,
    name: str,
    family: str,
    hypothesis: str,
    tickers: list[str],
    moving_average_days: int,
    scenario_sizing: ScenarioSizingConfig | None = None,
    role: str = "active_candidate",
    phase: str = "active_trading",
) -> ExperimentCandidate:
    return _candidate(
        name=name,
        role=role,
        phase=phase,
        family=family,
        hypothesis=hypothesis,
        scenario_sizing=scenario_sizing,
        strategy=StrategyConfig(
            type="absolute_momentum",
            tickers=tickers,
            moving_average_days=moving_average_days,
            defensive_ticker="BIL",
            trend_filter_days=None,
            max_asset_weight=None,
        ),
    )



def _macro_reset_candidates(iteration: int) -> tuple[ExperimentCandidate, ...]:
    """Reset-era macro-framework-inspired candidate batches with human-readable names."""
    sector_spdrs = ["XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC"]
    defensive_assets = ["BIL", "SGOV", "SHY", "IEF", "TLT", "GLD", "IAU", "UUP", "LQD"]
    factors = ["VUG", "VTV", "IWF", "IWD", "MTUM", "QUAL", "USMV", "SPLV", "SCHD", "VIG", "MOAT", "COWZ"]
    ai_core = ["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN"]
    ai_infra = ["VRT", "ETN", "PWR", "CEG", "GEV", "NRG", "CCJ", "SMH", "SOXX", "XLU", "XLI"]
    reflation = ["XLE", "XOP", "USO", "BNO", "DBC", "XLB", "XLI", "XLF", "KRE", "IWM", "RSP", "COWZ"]
    global_assets = ["SPY", "RSP", "EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC"]
    credit_assets = ["HYG", "JNK", "LQD", "VCIT", "VCSH", "BKLN", "SRLN", "JAAA", "JBBB"]

    batches = {
        101: (
            _sector_regime_candidate(
                name="regime_pulse_growth_liquidity_01_risk_on_core",
                family="regime_pulse_growth_liquidity",
                hypothesis=(
                    "Regime Pulse reset core: let growth, liquidity, credit, breadth, gold, "
                    "duration, and commodities compete before adding broad equity risk."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "GLD", "IEF", "TLT", "DBC", "UUP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="regime_pulse_growth_liquidity_02_low_churn_core",
                family="regime_pulse_growth_liquidity",
                hypothesis=(
                    "Low-churn Regime Pulse core: same cross-asset decision layer, but target "
                    "changes must be large and persistent enough for human execution."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=5,
                min_return=0.005,
                min_change=0.08,
                max_step=0.18,
                min_hold_days=10,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="regime_pulse_inflation_policy_03_barbell",
                family="regime_pulse_inflation_policy",
                hypothesis=(
                    "Inflation and policy barbell: when inflation/rates pressure dominates, the "
                    "strategy can own energy, commodities, gold, dollar, duration, or defensive cash."
                ),
                tickers=["DBC", "DBA", "USO", "BNO", "XLE", "XLB", "GLD", "UUP", "TLT", "IEF", "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.005,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _active_dual_candidate(
                name="regime_pulse_positioning_04_crowding_escape",
                family="regime_pulse_positioning",
                hypothesis=(
                    "Positioning-aware escape: own growth and AI beta only when trend is confirmed; "
                    "otherwise compete against quality, low-volatility, gold, and duration."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "SPY", "RSP", "QUAL", "USMV", "GLD", "TLT"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.015,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
        ),
        102: (
            _sector_regime_candidate(
                name="growth_inflation_rotation_01_growth_disinflation",
                family="growth_inflation_rotation",
                hypothesis=(
                    "Growth-disinflation proxy: when growth assets lead without inflation stress, favor "
                    "growth, momentum, broad equities, and credit-sensitive risk."
                ),
                tickers=["SPY", "QQQ", "SMH", "MTUM", "VUG", "SPHB", "RSP", "IWM", "HYG", "LQD", *defensive_assets],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.008,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="growth_inflation_rotation_02_reflation_broadening",
                family="growth_inflation_rotation",
                hypothesis=(
                    "Reflation-broadening proxy: test whether small caps, value, banks, energy, "
                    "materials, and industrials can replace mega-cap growth leadership."
                ),
                tickers=[*reflation, "VTV", "IWD", "MDY", "XHB", *defensive_assets, "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="growth_inflation_rotation_03_inflation_defense",
                family="growth_inflation_rotation",
                hypothesis=(
                    "Inflation-defense proxy: route away from duration-sensitive growth when energy, "
                    "commodities, gold, dollar, and defensive equity lead."
                ),
                tickers=["DBC", "DBA", "USO", "BNO", "XLE", "XOP", "XLB", "GLD", "UUP", "XLP", "XLU", "XLV", *defensive_assets, "SPY", "RSP"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="growth_inflation_rotation_04_deflation_quality_duration",
                family="growth_inflation_rotation",
                hypothesis=(
                    "Deflation-quality proxy: test whether duration, quality, low-volatility, gold, "
                    "and T-bills protect better than a simple equity/cash switch."
                ),
                tickers=["TLT", "IEF", "SHY", "SGOV", "GLD", "IAU", "QUAL", "USMV", "SPLV", "XLV", "XLP", "LQD", "SPY", "RSP"],
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.003,
                max_asset_weight=0.24,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
        103: (
            _active_dual_candidate(
                name="positioning_crowding_01_crowded_upside_trim",
                family="positioning_crowding",
                hypothesis=(
                    "Crowded upside trim: keep risk-on participation but force high-beta and AI "
                    "exposure to pass stricter return, trend, and risk-adjusted hurdles."
                ),
                tickers=["QQQ", "SMH", "SOXX", "SPHB", "MTUM", "SPY", "RSP", "QUAL", "USMV", "GLD", "TLT"],
                lookback_days=42,
                skip_days=5,
                top_n=4,
                min_return=0.020,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _dip_reentry_candidate(
                name="positioning_crowding_02_washed_out_reentry",
                family="positioning_crowding",
                hypothesis=(
                    "Washed-out broad-market reentry: buy discounts in measured steps only when "
                    "price repair, credit, breadth, and volatility stop signaling falling-knife risk."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "QUAL", "USMV", "GLD", "IEF"],
                trigger=-0.10,
                deep=-0.24,
                min_recovery=0.018,
                starter=0.16,
                step=0.18,
                max_risk=0.70,
                top_n=5,
                max_asset_weight=0.23,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _dip_reentry_candidate(
                name="positioning_crowding_03_ai_washout_micro",
                family="positioning_crowding_ai",
                hypothesis=(
                    "AI washout micro-sleeve: participate in sharp AI rebounds only with small, "
                    "confirmed allocations after deep discount and repair."
                ),
                tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "GLD", "TLT", "HYG", "LQD"],
                trigger=-0.15,
                deep=-0.32,
                min_recovery=0.035,
                starter=0.08,
                step=0.14,
                max_risk=0.45,
                top_n=3,
                max_asset_weight=0.18,
                vol_ceiling=0.40,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="positioning_crowding_04_sector_washout_rotation",
                family="positioning_crowding_sector",
                hypothesis=(
                    "Sector washout rotation: buy only sectors and factors that are both discounted "
                    "and repairing instead of treating every dip as a broad-market buy."
                ),
                tickers=[*sector_spdrs, *factors, *defensive_assets, "SPY", "RSP", "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.005,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        104: (
            _active_dual_candidate(
                name="exposure_alignment_long_only_01_us_equity",
                family="exposure_alignment_long_only",
                hypothesis=(
                    "Exposure Alignment long-only proxy for U.S. equities: max long when equity momentum state agrees, "
                    "half-sized/cash-like when leadership weakens, never short."
                ),
                tickers=["SPY", "QQQ", "RSP", "IWM", "QUAL", "USMV", "GLD", "TLT", "IEF"],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                min_return=0.010,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.34,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="exposure_alignment_long_only_02_global_equity",
                family="exposure_alignment_long_only",
                hypothesis=(
                    "Exposure Alignment global-equity proxy: global risk assets must beat defensive assets and "
                    "clear absolute trend gates before receiving full exposure."
                ),
                tickers=[*global_assets, "QQQ", "GLD", "TLT", "IEF", "UUP", "HYG", "LQD"],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                min_return=0.008,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _active_dual_candidate(
                name="exposure_alignment_long_only_03_commodity_gold_dollar",
                family="exposure_alignment_long_only",
                hypothesis=(
                    "Exposure Alignment macro-exposure proxy: commodities, oil, gold, dollar, and duration compete "
                    "for long-only exposure based on current vol-adjusted momentum leadership."
                ),
                tickers=["DBC", "DBA", "USO", "BNO", "XLE", "XLB", "GLD", "UUP", "TLT", "IEF", "SHY"],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                min_return=0.003,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.25,
                scenario_sizing=_scenario_profile("defensive"),
            ),
            _sector_regime_candidate(
                name="exposure_alignment_long_only_04_sector_asset_class_blend",
                family="exposure_alignment_long_only",
                hypothesis=(
                    "Exposure Alignment sector/asset-class blend: hold max/half/no-position style long exposure "
                    "across sectors, factors, credit, duration, gold, and cash-like defense."
                ),
                tickers=[*sector_spdrs, *factors, *credit_assets, *defensive_assets, "SPY", "RSP"],
                lookback_days=84,
                skip_days=10,
                top_n=6,
                min_return=0.004,
                min_change=0.06,
                max_step=0.24,
                min_hold_days=5,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("balanced"),
            ),
        ),
        105: (
            _ai_cycle_candidate(
                name="integrated_operating_system_01_retirement_core",
                family="integrated_operating_system",
                hypothesis=(
                    "Retirement-core operating system: broad cross-asset defense plus metered AI "
                    "satellite reentry, tuned for low churn and left-tail control."
                ),
                core_tickers=["SPY", "RSP", "IWM", "EFA", "EEM", "HYG", "LQD", "GLD", "TLT", "IEF", "DBC"],
                satellite_tickers=ai_core,
                lookback_days=126,
                skip_days=21,
                top_n=5,
                min_return=0.015,
                satellite_max=0.36,
                satellite_risk_on=0.24,
                satellite_reentry=0.42,
                trigger=-0.12,
                deep=-0.27,
                recovery_days=42,
                confirmation_days=10,
                min_change=0.08,
                max_step=0.18,
                min_hold_days=10,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _sector_regime_candidate(
                name="integrated_operating_system_02_balanced_macro_swing",
                family="integrated_operating_system",
                hypothesis=(
                    "Balanced macro swing system: sector, factor, credit, duration, gold, dollar, "
                    "and commodities all compete under scenario sizing."
                ),
                tickers=[*sector_spdrs, *factors, *credit_assets, *defensive_assets, "SPY", "RSP", "QQQ", "DBC", "UUP"],
                lookback_days=84,
                skip_days=10,
                top_n=6,
                min_return=0.004,
                min_change=0.06,
                max_step=0.23,
                min_hold_days=5,
                max_asset_weight=0.20,
                scenario_sizing=_scenario_profile("balanced"),
            ),
            _ai_cycle_candidate(
                name="integrated_operating_system_03_aggressive_ai_guarded",
                family="integrated_operating_system_ai",
                hypothesis=(
                    "Aggressive AI guarded system: larger AI upside budget, but only after trend, "
                    "credit, breadth, discount repair, and scenario pressure agree."
                ),
                core_tickers=["SPY", "RSP", "QUAL", "USMV", "GLD", "TLT", "HYG", "LQD", *ai_infra],
                satellite_tickers=ai_core,
                lookback_days=84,
                skip_days=10,
                top_n=5,
                min_return=0.025,
                satellite_max=0.48,
                satellite_risk_on=0.32,
                satellite_reentry=0.56,
                trigger=-0.15,
                deep=-0.32,
                min_recovery=0.035,
                min_change=0.06,
                max_step=0.24,
                min_hold_days=8,
                scenario_sizing=_scenario_profile("fragile_ai"),
            ),
            _sector_regime_candidate(
                name="integrated_operating_system_04_policy_oil_inflation_guard",
                family="integrated_operating_system_policy",
                hypothesis=(
                    "Policy/oil/inflation guard: keep the system from treating inflation shocks as "
                    "generic bearishness by letting energy, commodities, gold, dollar, and defense win."
                ),
                tickers=[*reflation, "GLD", "IAU", "UUP", "TLT", "IEF", "SHY", "XLP", "XLU", "XLV", "QUAL", "USMV", "SPY", "RSP", "HYG", "LQD"],
                lookback_days=63,
                skip_days=5,
                top_n=5,
                min_return=0.004,
                min_change=0.06,
                max_step=0.22,
                min_hold_days=5,
                max_asset_weight=0.22,
                scenario_sizing=_scenario_profile("defensive"),
            ),
        ),
    }
    return batches[iteration]

def _decision_sanity_overlay_candidates() -> tuple[ExperimentCandidate, ...]:
    selected = [
        _operating_system_candidates()[0],
        _operating_system_candidates()[1],
        _operating_system_candidates()[4],
        _operating_system_candidates()[7],
    ]
    names = ["ai_escape", "cross_asset", "sector_rotation", "oil_policy"]
    candidates: list[ExperimentCandidate] = []
    for base, short_name in zip(selected, names, strict=True):
        raw_name = f"i77_sanity_raw_{short_name}"
        capped_name = f"i77_sanity_cap_{short_name}"
        candidates.append(
            _candidate(
                name=raw_name,
                role="sanity_ablation",
                phase="decision_sanity_ablation",
                family=base.family,
                parent=base.name,
                hypothesis=(
                    "Raw scenario-sized benchmark for decision-sanity ablation. This candidate keeps "
                    "the original scenario/risk sizing so the paired capped variant has a clean control."
                ),
                scenario_sizing=base.scenario_sizing,
                strategy=base.strategy,
            )
        )
        candidates.append(
            _candidate(
                name=capped_name,
                role="sanity_ablation",
                phase="decision_sanity_ablation",
                family=base.family,
                parent=raw_name,
                hypothesis=(
                    "Decision-sanity variant: cap extra defensive weight unless at least two of "
                    "credit, volatility/liquidity, breadth, or trend confirm deterioration, or left-tail "
                    "pressure is already severe."
                ),
                scenario_sizing=base.scenario_sizing,
                decision_sanity=_decision_sanity_profile("confirmation_cap"),
                strategy=base.strategy,
            )
        )
    return tuple(candidates)


def _decision_sanity_tuning_candidates() -> tuple[ExperimentCandidate, ...]:
    selected = [
        _operating_system_candidates()[0],
        _operating_system_candidates()[4],
        _operating_system_candidates()[7],
    ]
    names = ["ai_escape", "sector_rotation", "oil_policy"]
    profiles = ["modest_cap", "confirmation_cap", "wide_cap", "strict_gate", "loose_gate"]
    candidates: list[ExperimentCandidate] = []
    for base, short_name in zip(selected, names, strict=True):
        raw_name = f"i78_sanity_raw_{short_name}"
        candidates.append(
            _candidate(
                name=raw_name,
                role="sanity_tuning_control",
                phase="decision_sanity_tuning",
                family=base.family,
                parent=base.name,
                hypothesis=(
                    "Raw scenario-sized benchmark for decision-sanity tuning. This keeps the "
                    "original scenario/risk sizing so each gated variant has a clean control."
                ),
                scenario_sizing=base.scenario_sizing,
                strategy=base.strategy,
            )
        )
        for profile in profiles:
            candidates.append(
                _candidate(
                    name=f"i78_sanity_{profile}_{short_name}",
                    role="sanity_tuning_variant",
                    phase="decision_sanity_tuning",
                    family=base.family,
                    parent=raw_name,
                    hypothesis=(
                        f"Decision-sanity tuning variant using the {profile} profile. It tests "
                        "whether event/news-only de-risking should be capped until market "
                        "confirmation arrives."
                    ),
                    scenario_sizing=base.scenario_sizing,
                    decision_sanity=_decision_sanity_profile(profile),
                    strategy=base.strategy,
                )
            )
    return tuple(candidates)


def _paper_readiness_tuning_candidates() -> tuple[ExperimentCandidate, ...]:
    selected = [
        _operating_system_candidates()[0],
        _operating_system_candidates()[1],
        _operating_system_candidates()[4],
        _operating_system_candidates()[7],
        _operating_system_candidates()[8],
    ]
    variants = [
        (
            "low_churn",
            {
                "skip_days": 21,
                "cycle_min_rebalance_change": 0.08,
                "cycle_max_step_change": 0.18,
                "cycle_min_hold_days": 10,
            },
            "Lower churn variant: larger change threshold, slower execution steps, and a minimum hold period.",
        ),
        (
            "metered_reentry",
            {
                "dip_starter_weight": 0.10,
                "dip_step_weight": 0.12,
                "dip_max_risk_weight": 0.65,
                "dip_confirmation_days": 10,
                "cycle_min_rebalance_change": 0.06,
                "cycle_max_step_change": 0.20,
                "cycle_min_hold_days": 8,
            },
            "Metered re-entry variant: buy back risk more slowly after drawdowns and require longer confirmation.",
        ),
    ]
    candidates: list[ExperimentCandidate] = []
    for base in selected:
        for suffix, updates, hypothesis in variants:
            candidates.append(
                _candidate(
                    name=f"i79_paper_ready_{_short_name(base.name)}_{suffix}",
                    role="paper_readiness_candidate",
                    phase="paper_readiness_tuning",
                    family=base.family,
                    parent=base.name,
                    hypothesis=(
                        f"{hypothesis} Parent: {base.name}. Optimizes for paper-monitorable "
                        "trade cadence, re-entry behavior, and monitoring-readiness score."
                    ),
                    scenario_sizing=base.scenario_sizing,
                    decision_sanity=base.decision_sanity,
                    strategy=_clone_strategy(base.strategy, **updates),
                )
            )
    return tuple(candidates)


def _operability_gauntlet_candidates() -> tuple[ExperimentCandidate, ...]:
    selected = [
        _operating_system_candidates()[0],
        _operating_system_candidates()[1],
        _operating_system_candidates()[4],
        _operating_system_candidates()[8],
        _operating_system_candidates()[9],
    ]
    candidates: list[ExperimentCandidate] = []
    for base in selected:
        slow_strategy = _clone_strategy(
            base.strategy,
            lookback_days=max(base.strategy.lookback_days, 126),
            skip_days=max(base.strategy.skip_days, 21),
            top_n=max(1, min(base.strategy.top_n, 2)),
            weighting="equal",
            volatility_target=None,
            drawdown_control=None,
            cycle_min_rebalance_change=0.12,
            cycle_max_step_change=0.12,
            cycle_min_hold_days=21,
            cycle_risk_off_override_change=0.18,
        )
        candidates.append(
            _candidate(
                name=f"i80_operability_{_short_name(base.name)}_slow_hysteresis",
                role="operability_gauntlet",
                phase="operability_gauntlet",
                family=base.family,
                parent=base.name,
                hypothesis=(
                    "Slow-hysteresis gauntlet: preserve the parent universe but remove daily vol-target "
                    "churn, use equal weights, require larger target changes, cap each step, and hold "
                    f"for at least 21 trading days. Parent: {base.name}."
                ),
                scenario_sizing=base.scenario_sizing,
                decision_sanity=base.decision_sanity,
                strategy=slow_strategy,
            )
        )

        high_conviction_strategy = _clone_strategy(
            base.strategy,
            lookback_days=max(base.strategy.lookback_days, 189),
            skip_days=max(base.strategy.skip_days, 42),
            top_n=1,
            weighting="equal",
            min_return=max(base.strategy.min_return, 0.01),
            trend_filter_days=200,
            max_asset_weight=1.0,
            volatility_target=None,
            drawdown_control=None,
            cycle_min_rebalance_change=0.15,
            cycle_max_step_change=0.15,
            cycle_min_hold_days=42,
            cycle_risk_off_override_change=0.20,
        )
        candidates.append(
            _candidate(
                name=f"i80_operability_{_short_name(base.name)}_high_conviction",
                role="operability_gauntlet",
                phase="operability_gauntlet",
                family=base.family,
                parent=base.name,
                hypothesis=(
                    "High-conviction gauntlet: intentionally reduce degrees of freedom and scenario-size "
                    "churn to test whether a slower one-position operating system can keep enough edge "
                    f"while becoming human-executable. Parent: {base.name}."
                ),
                scenario_sizing=None,
                decision_sanity=base.decision_sanity,
                strategy=high_conviction_strategy,
            )
        )
    return tuple(candidates)


def _future_state_ml_candidates() -> tuple[ExperimentCandidate, ...]:
    base_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    ai_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "SMH", "XLK", "IGV", "RSP", "IWM", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    sector_core = StrategyConfig(
        type="dual_momentum",
        tickers=["XLK", "XLI", "XLF", "XLE", "XLV", "XLP", "XLY", "RSP", "SPY", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=3,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.35,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    controls = (
        _candidate(
            name="i81_ml_state_control_core_no_ml",
            role="future_state_ml_control",
            phase="future_state_ml",
            family="future_state_control",
            hypothesis="No-ML control for the core cross-asset strategy so future-state overlays can be judged against the same base allocation.",
            strategy=base_core,
        ),
        _candidate(
            name="i81_ml_state_control_ai_no_ml",
            role="future_state_ml_control",
            phase="future_state_ml",
            family="future_state_control",
            hypothesis="No-ML control for the AI-sensitive strategy so short-horizon AI regime probabilities have a fair benchmark.",
            strategy=ai_core,
        ),
        _candidate(
            name="i81_ml_state_control_sector_no_ml",
            role="future_state_ml_control",
            phase="future_state_ml",
            family="future_state_control",
            hypothesis="No-ML control for the sector-rotation strategy so three-month future-state overlays have a fair benchmark.",
            strategy=sector_core,
        ),
    )
    specs: tuple[tuple[str, StrategyConfig, str, str, FutureStateModelConfig], ...] = (
        (
            "base_rate_1m_core",
            base_core,
            "future_state_baseline",
            "Rolling historical base-rate scenario probabilities should be a humility benchmark for ML overlays.",
            _future_state_profile("base_rate", horizon_days=21, feature_set="core"),
        ),
        (
            "transition_1m_core",
            base_core,
            "future_state_transition",
            "Conditional transition tables should improve over unconditional base rates when current market state is persistent.",
            _future_state_profile("transition", horizon_days=21, feature_set="core"),
        ),
        (
            "knn_1m_core",
            base_core,
            "future_state_analog",
            "Distance-weighted historical analogs should capture nonlinear risk-off and re-entry setups without price targets.",
            _future_state_profile("knn", horizon_days=21, feature_set="core", k_neighbors=65),
        ),
        (
            "bagged_knn_1m_all",
            base_core,
            "future_state_analog",
            "Feature-bagged analogs should reduce single-feature overfit while preserving nonlinear regime matching.",
            _future_state_profile("feature_bag_knn", horizon_days=21, feature_set="all", k_neighbors=75),
        ),
        (
            "centroid_1m_cross_asset",
            base_core,
            "future_state_distance",
            "Shrunken state centroids should provide a stable low-variance regime classifier for cross-asset allocation.",
            _future_state_profile("centroid", horizon_days=21, feature_set="cross_asset"),
        ),
        (
            "naive_bayes_1m_cross_asset",
            base_core,
            "future_state_probabilistic",
            "Gaussian naive Bayes should handle sparse regime samples with explicit class priors and probability outputs.",
            _future_state_profile("naive_bayes", horizon_days=21, feature_set="cross_asset"),
        ),
        (
            "ridge_logit_1m_core",
            base_core,
            "future_state_regularized",
            "Regularized multinomial logistic state probabilities should test a more classical supervised ML regime model.",
            _future_state_profile("ridge_logit", horizon_days=21, feature_set="core"),
        ),
        (
            "tail_specialist_1m_core",
            base_core,
            "future_state_tail",
            "A risk-off specialist should prioritize left-tail classification, then distribute remaining probability across tradable states.",
            _future_state_profile("tail_specialist", horizon_days=21, feature_set="core", k_neighbors=90),
        ),
        (
            "ensemble_1m_all",
            base_core,
            "future_state_ensemble",
            "A blended state ensemble should be less brittle than any one classifier when market regimes shift.",
            _future_state_profile("ensemble", horizon_days=21, feature_set="all", k_neighbors=75),
        ),
        (
            "knn_1w_ai",
            ai_core,
            "future_state_ai",
            "A short-horizon AI feature analog should catch fragile melt-up versus AI unwind pressure before slow signals react.",
            _future_state_profile("knn", horizon_days=5, feature_set="ai", k_neighbors=55),
        ),
        (
            "ridge_logit_3m_cross_asset",
            base_core,
            "future_state_regularized",
            "A three-month regularized classifier should test slower allocation state prediction for swing-horizon sizing.",
            _future_state_profile("ridge_logit", horizon_days=63, feature_set="cross_asset", train_window_days=1008),
        ),
        (
            "ensemble_3m_sector",
            sector_core,
            "future_state_sector_rotation",
            "A three-month ensemble should help sector rotation avoid late-cycle traps while re-entering when state probabilities improve.",
            _future_state_profile("ensemble", horizon_days=63, feature_set="all", train_window_days=1008, k_neighbors=95),
        ),
    )
    ml_candidates = tuple(
        _candidate(
            name=f"i81_ml_state_{slug}",
            role="future_state_ml_candidate",
            phase="future_state_ml",
            family=family,
            hypothesis=f"{hypothesis} Uses learned future-state probabilities for risk-budget sizing, not price prediction.",
            future_state_model=model_config,
            strategy=strategy,
        )
        for slug, strategy, family, hypothesis, model_config in specs
    )
    return (*controls, *ml_candidates)


def _bayesian_future_state_candidates() -> tuple[ExperimentCandidate, ...]:
    base_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    ai_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "SMH", "XLK", "IGV", "RSP", "IWM", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    sector_core = StrategyConfig(
        type="dual_momentum",
        tickers=["XLK", "XLI", "XLF", "XLE", "XLV", "XLP", "XLY", "RSP", "SPY", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=3,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.35,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    controls = (
        _candidate(
            name="i82_bayes_state_control_core_no_bayes",
            role="bayesian_future_state_control",
            phase="bayesian_future_state",
            family="future_state_control",
            hypothesis="No-Bayesian control for the core cross-asset strategy so posterior probability overlays are judged against the same base allocation.",
            strategy=base_core,
        ),
        _candidate(
            name="i82_bayes_state_control_ai_no_bayes",
            role="bayesian_future_state_control",
            phase="bayesian_future_state",
            family="future_state_control",
            hypothesis="No-Bayesian control for the AI-sensitive strategy so fragile-upside and unwind probabilities have a fair benchmark.",
            strategy=ai_core,
        ),
        _candidate(
            name="i82_bayes_state_control_sector_no_bayes",
            role="bayesian_future_state_control",
            phase="bayesian_future_state",
            family="future_state_control",
            hypothesis="No-Bayesian control for the sector-rotation strategy so sector re-risking overlays have a fair benchmark.",
            strategy=sector_core,
        ),
    )
    specs: tuple[tuple[str, StrategyConfig, str, str, FutureStateModelConfig], ...] = (
        (
            "base_rate_1m_core",
            base_core,
            "bayesian_base_rate",
            "Dirichlet-smoothed rolling class priors test whether Bayesian humility improves scenario sizing before feature models are trusted.",
            _future_state_profile("bayesian_base_rate", horizon_days=21, feature_set="core"),
        ),
        (
            "transition_1m_core",
            base_core,
            "bayesian_transition",
            "A Dirichlet-smoothed state-transition table should avoid overreacting to sparse analogs while still adapting to the current instant regime.",
            _future_state_profile("bayesian_transition", horizon_days=21, feature_set="core"),
        ),
        (
            "naive_bayes_1m_core",
            base_core,
            "bayesian_feature_model",
            "Bayesian Gaussian naive Bayes tests feature-conditioned posterior probabilities with mean and variance shrinkage.",
            _future_state_profile("bayesian_naive_bayes", horizon_days=21, feature_set="core"),
        ),
        (
            "ensemble_1m_all",
            base_core,
            "bayesian_ensemble",
            "A Bayesian ensemble blends priors, transition evidence, feature likelihoods, and tail specialization for a low-variance scenario overlay.",
            _future_state_profile("bayesian_ensemble", horizon_days=21, feature_set="all", k_neighbors=75),
        ),
        (
            "fast_transition_1m_core",
            base_core,
            "bayesian_transition",
            "Short half-life transition priors test whether the model can re-risk faster after market repair without becoming a daily-trading system.",
            _future_state_profile(
                "bayesian_transition",
                horizon_days=21,
                feature_set="core",
                recency_half_life_days=84,
                dirichlet_prior_strength=6.0,
            ),
        ),
        (
            "slow_ensemble_3m_cross_asset",
            base_core,
            "bayesian_ensemble",
            "Slow Bayesian priors test whether three-month state forecasts are better as a stable allocation throttle than as an aggressive tactical trigger.",
            _future_state_profile(
                "bayesian_ensemble",
                horizon_days=63,
                feature_set="cross_asset",
                train_window_days=1008,
                recency_half_life_days=504,
                dirichlet_prior_strength=18.0,
                k_neighbors=95,
            ),
        ),
        (
            "transition_1w_ai",
            ai_core,
            "bayesian_ai_cycle",
            "Short-horizon Bayesian transition probabilities test whether AI leadership should stay risk-on or de-risk during fragile concentration.",
            _future_state_profile("bayesian_transition", horizon_days=5, feature_set="ai", recency_half_life_days=63),
        ),
        (
            "naive_bayes_1w_ai",
            ai_core,
            "bayesian_ai_cycle",
            "Bayesian feature likelihoods on AI proxies test whether QQQ/RSP, SMH/SPY, credit, and volatility features add signal beyond transition priors.",
            _future_state_profile("bayesian_naive_bayes", horizon_days=5, feature_set="ai", recency_half_life_days=63),
        ),
        (
            "ensemble_1m_ai",
            ai_core,
            "bayesian_ai_cycle",
            "A one-month Bayesian ensemble tests whether AI-sensitive strategies can keep upside while avoiding fragile melt-up reversals.",
            _future_state_profile("bayesian_ensemble", horizon_days=21, feature_set="ai", k_neighbors=75),
        ),
        (
            "transition_1m_sector",
            sector_core,
            "bayesian_sector_rotation",
            "Bayesian transition sizing tests whether sector rotation should lower aggregate risk when leadership looks narrow or unstable.",
            _future_state_profile("bayesian_transition", horizon_days=21, feature_set="cross_asset"),
        ),
        (
            "naive_bayes_3m_sector",
            sector_core,
            "bayesian_sector_rotation",
            "Three-month Bayesian feature likelihoods test slower sector-cycle probabilities for rotation without relying on one price-trend window.",
            _future_state_profile(
                "bayesian_naive_bayes",
                horizon_days=63,
                feature_set="all",
                train_window_days=1008,
                bayesian_feature_shrinkage=20.0,
            ),
        ),
        (
            "ensemble_3m_sector",
            sector_core,
            "bayesian_sector_rotation",
            "A three-month Bayesian ensemble tests whether sector rotation benefits from posterior smoothing across transition, tail, and feature evidence.",
            _future_state_profile(
                "bayesian_ensemble",
                horizon_days=63,
                feature_set="all",
                train_window_days=1008,
                recency_half_life_days=378,
                dirichlet_prior_strength=14.0,
                k_neighbors=95,
            ),
        ),
    )
    bayesian_candidates = tuple(
        _candidate(
            name=f"i82_bayes_state_{slug}",
            role="bayesian_future_state_candidate",
            phase="bayesian_future_state",
            family=family,
            hypothesis=f"{hypothesis} The posterior probabilities resize risk exposure; they do not directly forecast price targets.",
            future_state_model=model_config,
            strategy=strategy,
        )
        for slug, strategy, family, hypothesis, model_config in specs
    )
    return (*controls, *bayesian_candidates)


def _sklearn_future_state_candidates() -> tuple[ExperimentCandidate, ...]:
    base_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    ai_core = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "SMH", "XLK", "IGV", "RSP", "IWM", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.45,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    sector_core = StrategyConfig(
        type="dual_momentum",
        tickers=["XLK", "XLI", "XLF", "XLE", "XLV", "XLP", "XLY", "RSP", "SPY", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=3,
        defensive_ticker="BIL",
        ranking_metric="risk_adjusted_return",
        weighting="inverse_volatility",
        trend_filter_days=100,
        max_asset_weight=0.35,
        cycle_min_rebalance_change=0.08,
        cycle_max_step_change=0.20,
        cycle_min_hold_days=10,
    )
    controls = (
        _candidate(
            name="i83_sklearn_state_control_core_no_ml",
            role="sklearn_future_state_control",
            phase="sklearn_future_state",
            family="future_state_control",
            hypothesis="No-sklearn control for the core cross-asset strategy so supervised probability overlays are judged against the same base allocation.",
            strategy=base_core,
        ),
        _candidate(
            name="i83_sklearn_state_control_ai_no_ml",
            role="sklearn_future_state_control",
            phase="sklearn_future_state",
            family="future_state_control",
            hypothesis="No-sklearn control for the AI-sensitive strategy so short-horizon model overlays have a fair benchmark.",
            strategy=ai_core,
        ),
        _candidate(
            name="i83_sklearn_state_control_sector_no_ml",
            role="sklearn_future_state_control",
            phase="sklearn_future_state",
            family="future_state_control",
            hypothesis="No-sklearn control for the sector-rotation strategy so sector regime overlays have a fair benchmark.",
            strategy=sector_core,
        ),
    )
    specs: tuple[tuple[str, StrategyConfig, str, str, FutureStateModelConfig], ...] = (
        (
            "logit_l2_1m_core",
            base_core,
            "sklearn_regularized_state",
            "Regularized logistic probabilities test a simple, calibrated-ish linear state model before nonlinear models are trusted.",
            _future_state_profile("sk_logit_l2", horizon_days=21, feature_set="core"),
        ),
        (
            "logit_l1_1m_core",
            base_core,
            "sklearn_feature_selection",
            "Sparse logistic probabilities test whether implicit feature selection improves future-state sizing and interpretability.",
            _future_state_profile("sk_logit_l1", horizon_days=21, feature_set="core", sklearn_regularization_c=0.35),
        ),
        (
            "forest_1m_all",
            base_core,
            "sklearn_tree_state",
            "Random forests test nonlinear state interactions across trend, breadth, credit, vol, duration, dollar, and commodity features.",
            _future_state_profile("sk_random_forest", horizon_days=21, feature_set="all", sklearn_max_depth=4),
        ),
        (
            "extra_trees_1m_all",
            base_core,
            "sklearn_tree_state",
            "Extra-trees probabilities test a faster nonlinear state model that is less brittle than one fitted decision tree path.",
            _future_state_profile("sk_extra_trees", horizon_days=21, feature_set="all", sklearn_max_depth=4),
        ),
        (
            "gb_1m_core",
            base_core,
            "sklearn_tree_state",
            "A bounded gradient-boosting state model tests whether sequential nonlinear learners add signal without dominating runtime.",
            _future_state_profile("sk_gradient_boosting", horizon_days=21, feature_set="core", sklearn_max_depth=3),
        ),
        (
            "logit_l2_3m_core",
            base_core,
            "sklearn_regularized_state",
            "A three-month regularized logistic model tests slower swing-horizon probabilities with a cheap, interpretable estimator.",
            _future_state_profile("sk_logit_l2", horizon_days=63, feature_set="core", train_window_days=1008),
        ),
        (
            "forest_3m_cross_asset",
            base_core,
            "sklearn_tree_state",
            "A three-month forest tests slower future-state sizing for swing-horizon allocation, not next-day timing.",
            _future_state_profile("sk_random_forest", horizon_days=63, feature_set="cross_asset", train_window_days=1008),
        ),
        (
            "extra_trees_3m_all",
            base_core,
            "sklearn_tree_state",
            "Three-month extra-trees probabilities test slower nonlinear state sizing across the full signal stack.",
            _future_state_profile("sk_extra_trees", horizon_days=63, feature_set="all", train_window_days=1008, sklearn_max_depth=4),
        ),
        (
            "forest_1w_ai",
            ai_core,
            "sklearn_ai_cycle",
            "Short-horizon forests test whether AI concentration can be detected early enough to resize without overtrading.",
            _future_state_profile("sk_random_forest", horizon_days=5, feature_set="ai", sklearn_max_depth=3),
        ),
        (
            "extra_trees_1m_ai",
            ai_core,
            "sklearn_ai_cycle",
            "An AI-focused extra-trees model tests whether supervised probabilities retain upside while cutting fragile melt-up reversals.",
            _future_state_profile("sk_extra_trees", horizon_days=21, feature_set="ai", sklearn_max_depth=4),
        ),
        (
            "forest_1m_sector",
            sector_core,
            "sklearn_sector_rotation",
            "A sector-rotation forest tests nonlinear regime signals for sizing the sector sleeve without forcing binary cash/equity behavior.",
            _future_state_profile("sk_random_forest", horizon_days=21, feature_set="cross_asset", sklearn_max_depth=4),
        ),
        (
            "extra_trees_3m_sector",
            sector_core,
            "sklearn_sector_rotation",
            "Three-month extra-trees tests slower sector-cycle probabilities for rotation, defense, and re-entry discipline.",
            _future_state_profile("sk_extra_trees", horizon_days=63, feature_set="all", train_window_days=1008, sklearn_max_depth=4),
        ),
    )
    sklearn_candidates = tuple(
        _candidate(
            name=f"i83_sklearn_state_{slug}",
            role="sklearn_future_state_candidate",
            phase="sklearn_future_state",
            family=family,
            hypothesis=f"{hypothesis} The model feeds constrained future-state sizing and never directly chooses a trade by itself.",
            future_state_model=model_config,
            strategy=strategy,
        )
        for slug, strategy, family, hypothesis, model_config in specs
    )
    return (*controls, *sklearn_candidates)


def _high_cagr_ml_guardrail_candidates() -> tuple[ExperimentCandidate, ...]:
    base = _operating_system_candidates()[0]
    assert base.name == "i21_os_ai_escape_scenario_sized"
    high_octane = _clone_strategy(
        base.strategy,
        volatility_target=VolatilityTargetConfig(
            annualized_volatility=0.16,
            lookback_days=42,
            max_leverage=1.0,
        ),
    )
    candidates: list[ExperimentCandidate] = [
        _candidate(
            name="i84_high_cagr_control_raw_ai_escape",
            role="high_cagr_ml_control",
            phase="high_cagr_ml_guardrail",
            family="high_cagr_ai_escape",
            parent=base.name,
            hypothesis=(
                "High-CAGR control: keep the historically strong AI escape engine unchanged so ML "
                "guardrails are judged against return preservation, not just drawdown reduction."
            ),
            scenario_sizing=base.scenario_sizing,
            strategy=base.strategy,
        ),
        _candidate(
            name="i84_high_cagr_control_wide_cap_ai_escape",
            role="high_cagr_ml_control",
            phase="high_cagr_ml_guardrail",
            family="high_cagr_ai_escape",
            parent=base.name,
            hypothesis=(
                "Decision-sanity control: preserve the high-CAGR engine but cap event/news-only "
                "de-risking with the wide-cap profile that previously retained most upside."
            ),
            scenario_sizing=base.scenario_sizing,
            decision_sanity=_decision_sanity_profile("wide_cap"),
            strategy=base.strategy,
        ),
        _candidate(
            name="i84_high_cagr_control_high_octane_ai_escape",
            role="high_cagr_ml_control",
            phase="high_cagr_ml_guardrail",
            family="high_cagr_ai_escape",
            parent=base.name,
            hypothesis=(
                "High-octane control: allow less volatility downscaling before adding ML, so the "
                "guardrail can be tested against a higher-return posture."
            ),
            scenario_sizing=base.scenario_sizing,
            decision_sanity=_decision_sanity_profile("wide_cap"),
            strategy=high_octane,
        ),
    ]
    specs: tuple[
        tuple[str, str, ScenarioSizingConfig | None, StrategyConfig, FutureStateModelConfig, DecisionSanityConfig | None],
        ...,
    ] = (
        (
            "rf_1m_tail_gate_fragile",
            "Random-forest 1M tail gate: only reduce the AI engine after risk-off probability crosses a material threshold.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_random_forest", horizon_days=21, feature_set="ai", threshold=0.35),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_1m_tail_gate_fragile",
            "Extra-trees 1M tail gate: test a faster nonlinear guardrail while preserving almost all transition/upside exposure.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_extra_trees", horizon_days=21, feature_set="ai", threshold=0.35),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "logit_1m_tail_gate_fragile",
            "Regularized-logit 1M tail gate: simple linear ML should be hard to beat if the signal is robust.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_logit_l2", horizon_days=21, feature_set="ai", threshold=0.35),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_3m_tail_gate_fragile",
            "Random-forest 3M tail gate: slower swing-horizon risk probabilities may avoid overreacting to noise.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_random_forest", horizon_days=63, feature_set="ai", threshold=0.40),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_3m_tail_gate_fragile",
            "Extra-trees 3M tail gate: nonlinear slower-horizon ML with a higher activation threshold.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_extra_trees", horizon_days=63, feature_set="all", threshold=0.40),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_1w_fast_tail_gate_fragile",
            "Fast 1W AI tail gate: test whether short-horizon ML can catch abrupt AI-beta breaks without staying bearish.",
            base.scenario_sizing,
            base.strategy,
            _return_preserving_ml_profile("sk_random_forest", horizon_days=5, feature_set="ai", threshold=0.45, refit_every_days=63),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_1m_tail_gate_aggressive_scenario",
            "Aggressive scenario plus ML tail gate: let scenario sizing stay risk-on unless ML assigns clear left-tail pressure.",
            _scenario_profile("aggressive"),
            base.strategy,
            _return_preserving_ml_profile("sk_random_forest", horizon_days=21, feature_set="ai", threshold=0.40, stress=0.68),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_1m_tail_gate_no_scenario",
            "ML-only tail gate: remove hand-built scenario sizing to test whether ML can cut drawdown without permanently shrinking CAGR.",
            None,
            base.strategy,
            _return_preserving_ml_profile("sk_random_forest", horizon_days=21, feature_set="ai", threshold=0.45, stress=0.55),
            None,
        ),
        (
            "extra_trees_1m_tail_gate_high_octane",
            "High-octane ML guardrail: raise the volatility target and rely on thresholded ML plus wide-cap sanity to control the left tail.",
            base.scenario_sizing,
            high_octane,
            _return_preserving_ml_profile("sk_extra_trees", horizon_days=21, feature_set="ai", threshold=0.38, stress=0.70),
            _decision_sanity_profile("wide_cap"),
        ),
    )
    candidates.extend(
        _candidate(
            name=f"i84_high_cagr_ml_{slug}",
            role="high_cagr_ml_guardrail",
            phase="high_cagr_ml_guardrail",
            family="high_cagr_ai_escape_ml",
            parent=base.name,
            hypothesis=(
                f"{hypothesis} The ML model can only alter position sizing; the high-CAGR AI engine "
                "still chooses the risk assets."
            ),
            scenario_sizing=scenario_sizing,
            future_state_model=model_config,
            decision_sanity=decision_sanity,
            strategy=strategy,
        )
        for slug, hypothesis, scenario_sizing, strategy, model_config, decision_sanity in specs
    )
    return tuple(candidates)


def _return_preserving_ml_profile(
    model: str,
    *,
    horizon_days: int,
    feature_set: str,
    threshold: float,
    stress: float = 0.72,
    refit_every_days: int = 126,
) -> FutureStateModelConfig:
    return _future_state_profile(
        model,
        horizon_days=horizon_days,
        feature_set=feature_set,
        train_window_days=1008 if horizon_days >= 63 else 756,
        refit_every_days=refit_every_days,
        sklearn_n_estimators=32,
        sklearn_max_depth=4,
        sklearn_min_samples_leaf=22,
        stress_multiplier=stress,
        transition_multiplier=1.0,
        fragile_upside_multiplier=1.0,
        min_multiplier=max(0.55, stress),
        probability_smoothing=0.04,
        risk_off_activation_probability=threshold,
        transition_activation_probability=0.95,
        fragile_activation_probability=0.95,
    )


def _strategy_drawdown_ml_guardrail_candidates() -> tuple[ExperimentCandidate, ...]:
    base = _operating_system_candidates()[0]
    assert base.name == "i21_os_ai_escape_scenario_sized"
    high_octane = _clone_strategy(
        base.strategy,
        volatility_target=VolatilityTargetConfig(
            annualized_volatility=0.16,
            lookback_days=42,
            max_leverage=1.0,
        ),
    )
    controls: list[ExperimentCandidate] = [
        _candidate(
            name="i85_drawdown_ml_control_raw_ai_escape",
            role="strategy_drawdown_ml_control",
            phase="strategy_drawdown_ml_guardrail",
            family="high_cagr_ai_escape",
            parent=base.name,
            hypothesis=(
                "Raw high-CAGR AI escape control for strategy-specific ML drawdown guards. "
                "A candidate only matters if it preserves most of this return while improving tail risk."
            ),
            scenario_sizing=base.scenario_sizing,
            strategy=base.strategy,
        ),
        _candidate(
            name="i85_drawdown_ml_control_wide_cap_ai_escape",
            role="strategy_drawdown_ml_control",
            phase="strategy_drawdown_ml_guardrail",
            family="high_cagr_ai_escape",
            parent=base.name,
            hypothesis=(
                "Wide-cap decision-sanity control keeps news/event-only de-risking from overwhelming "
                "the historically strong AI escape engine."
            ),
            scenario_sizing=base.scenario_sizing,
            decision_sanity=_decision_sanity_profile("wide_cap"),
            strategy=base.strategy,
        ),
        _candidate(
            name="i85_drawdown_ml_control_future_state_tail_gate",
            role="strategy_drawdown_ml_control",
            phase="strategy_drawdown_ml_guardrail",
            family="high_cagr_ai_escape_ml",
            parent=base.name,
            hypothesis=(
                "Best prior high-CAGR future-state ML shape: preserve transition/upside exposure and "
                "only throttle after clear model risk-off pressure."
            ),
            scenario_sizing=base.scenario_sizing,
            future_state_model=_return_preserving_ml_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=0.35,
            ),
            decision_sanity=_decision_sanity_profile("wide_cap"),
            strategy=base.strategy,
        ),
    ]
    specs: tuple[
        tuple[
            str,
            str,
            StrategyConfig,
            StrategyDrawdownModelConfig,
            FutureStateModelConfig | None,
            DecisionSanityConfig | None,
        ],
        ...,
    ] = (
        (
            "rf_1m_dd8_ai",
            "Random forest predicts whether the current AI escape posture faces an 8% forward strategy drawdown over the next month.",
            base.strategy,
            _strategy_drawdown_profile("sk_random_forest", horizon_days=21, feature_set="ai", threshold=-0.08),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_1m_dd8_ai",
            "Extra trees tests a more variance-tolerant nonlinear drawdown guard on AI, credit, breadth, and vol features.",
            base.strategy,
            _strategy_drawdown_profile("sk_extra_trees", horizon_days=21, feature_set="ai", threshold=-0.08),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "logit_1m_dd8_ai",
            "Regularized logistic drawdown guard is the transparent baseline: it should compete if the feature relationship is stable.",
            base.strategy,
            _strategy_drawdown_profile("sk_logit_l2", horizon_days=21, feature_set="ai", threshold=-0.08),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "gradient_1m_dd8_ai",
            "Gradient boosting tests nonlinear interactions while staying shallower than a full tree ensemble.",
            base.strategy,
            _strategy_drawdown_profile("sk_gradient_boosting", horizon_days=21, feature_set="ai", threshold=-0.08, estimators=64),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "ensemble_1m_dd8_ai",
            "A simple sklearn ensemble tests whether model averaging improves drawdown probability stability.",
            base.strategy,
            _strategy_drawdown_profile("sk_ensemble", horizon_days=21, feature_set="ai", threshold=-0.08, estimators=40),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_3m_dd10_all",
            "Three-month forest uses the full cross-asset feature stack to catch slower strategy failure regimes.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_random_forest",
                horizon_days=63,
                feature_set="all",
                threshold=-0.10,
                activation=0.38,
                stress=0.58,
                train_window_days=1008,
            ),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_3m_dd10_cross_asset",
            "Three-month extra trees focuses on cross-asset stress, rates, dollar, oil, credit, and volatility features.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=63,
                feature_set="cross_asset",
                threshold=-0.10,
                activation=0.38,
                stress=0.58,
                train_window_days=1008,
            ),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "rf_1m_dd12_late_hard_gate",
            "Late hard gate only acts when the model sees elevated odds of a severe 12% strategy drawdown, preserving upside otherwise.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_random_forest",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.12,
                activation=0.45,
                stress=0.52,
                min_multiplier=0.50,
            ),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_1m_dd6_soft_gate",
            "Soft gate reacts to smaller 6% drawdown risk, but only trims modestly so it cannot collapse CAGR into cash-like returns.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.06,
                activation=0.42,
                stress=0.75,
                min_multiplier=0.72,
            ),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "extra_trees_future_state_combo",
            "Combine broad future-state risk with strategy-specific drawdown risk to test whether two independent ML views reduce false confidence.",
            base.strategy,
            _strategy_drawdown_profile("sk_extra_trees", horizon_days=21, feature_set="ai", threshold=-0.08, stress=0.70),
            _return_preserving_ml_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=0.40,
                stress=0.78,
            ),
            _decision_sanity_profile("wide_cap"),
        ),
        (
            "high_octane_extra_trees_dd8",
            "High-octane variant asks whether ML can support a higher-volatility target without letting drawdowns expand too far.",
            high_octane,
            _strategy_drawdown_profile("sk_extra_trees", horizon_days=21, feature_set="ai", threshold=-0.08, stress=0.58),
            None,
            _decision_sanity_profile("wide_cap"),
        ),
    )
    candidates = controls.copy()
    candidates.extend(
        _candidate(
            name=f"i85_strategy_drawdown_ml_{slug}",
            role="strategy_drawdown_ml_guardrail",
            phase="strategy_drawdown_ml_guardrail",
            family="high_cagr_ai_escape_strategy_drawdown_ml",
            parent=base.name,
            hypothesis=(
                f"{hypothesis} The model does not choose assets; it only meters the risk sleeve "
                "when the strategy-specific drawdown probability is high."
            ),
            scenario_sizing=base.scenario_sizing,
            future_state_model=future_state_model,
            strategy_drawdown_model=drawdown_model,
            decision_sanity=decision_sanity,
            strategy=strategy,
        )
        for slug, hypothesis, strategy, drawdown_model, future_state_model, decision_sanity in specs
    )
    return tuple(candidates)


def _strategy_drawdown_profile(
    model: str,
    *,
    horizon_days: int,
    feature_set: str,
    threshold: float,
    activation: float = 0.42,
    stress: float = 0.62,
    min_multiplier: float = 0.55,
    train_window_days: int = 756,
    estimators: int = 48,
    refit_every_days: int = 126,
) -> StrategyDrawdownModelConfig:
    return StrategyDrawdownModelConfig(
        model=cast(Any, model),
        horizon_days=horizon_days,
        feature_set=cast(Any, feature_set),
        train_window_days=train_window_days,
        refit_every_days=refit_every_days,
        future_drawdown_threshold=threshold,
        activation_probability=activation,
        stress_multiplier=stress,
        min_multiplier=min_multiplier,
        probability_smoothing=0.06,
        sklearn_n_estimators=estimators,
        sklearn_max_depth=4,
        sklearn_min_samples_leaf=24,
    )


def _aggressive_drawdown_ml_hybrid_candidates() -> tuple[ExperimentCandidate, ...]:
    base = _operating_system_candidates()[0]
    assert base.name == "i21_os_ai_escape_scenario_sized"
    dd6 = _clone_strategy(
        base.strategy,
        drawdown_control=DrawdownControlConfig(
            equity_lookback_days=84,
            max_drawdown=-0.06,
            risk_multiplier=0.45,
        ),
    )
    dd8 = _clone_strategy(
        base.strategy,
        drawdown_control=DrawdownControlConfig(
            equity_lookback_days=126,
            max_drawdown=-0.08,
            risk_multiplier=0.45,
        ),
    )
    dd10 = _clone_strategy(
        base.strategy,
        drawdown_control=DrawdownControlConfig(
            equity_lookback_days=126,
            max_drawdown=-0.10,
            risk_multiplier=0.55,
        ),
    )
    specs: tuple[
        tuple[str, str, StrategyConfig, StrategyDrawdownModelConfig | None, FutureStateModelConfig | None],
        ...,
    ] = (
        (
            "raw_control",
            "Raw high-CAGR control anchors whether any added guardrail earns its complexity.",
            base.strategy,
            None,
            None,
        ),
        (
            "classic_dd6_control",
            "Classic 6% rolling drawdown control tests the non-ML ceiling for drawdown reduction and CAGR sacrifice.",
            dd6,
            None,
            None,
        ),
        (
            "classic_dd8_control",
            "Classic 8% rolling drawdown control is the moderate non-ML benchmark.",
            dd8,
            None,
            None,
        ),
        (
            "classic_dd10_control",
            "Classic 10% rolling drawdown control is the slower non-ML benchmark.",
            dd10,
            None,
            None,
        ),
        (
            "extra_trees_1m_dd8_aggressive",
            "Aggressive extra-trees guard trims earlier and deeper when 1M strategy drawdown odds rise.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.08,
                activation=0.30,
                stress=0.38,
                min_multiplier=0.35,
                estimators=72,
            ),
            None,
        ),
        (
            "rf_1m_dd8_aggressive",
            "Aggressive random forest is the comparable tree-bagging drawdown throttle.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_random_forest",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.08,
                activation=0.30,
                stress=0.38,
                min_multiplier=0.35,
                estimators=72,
            ),
            None,
        ),
        (
            "logit_1m_dd8_aggressive",
            "Aggressive logistic guard tests whether a simpler calibrated surface can cut earlier without overfitting trees.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_logit_l2",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.08,
                activation=0.30,
                stress=0.40,
                min_multiplier=0.35,
            ),
            None,
        ),
        (
            "extra_trees_1m_dd5_early",
            "Early 5% drawdown label tests whether catching smaller strategy cracks reduces the eventual left tail.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.05,
                activation=0.30,
                stress=0.48,
                min_multiplier=0.42,
                estimators=72,
            ),
            None,
        ),
        (
            "extra_trees_3m_dd8_all_aggressive",
            "Three-month full-stack extra trees asks whether slower macro/cross-asset stress is the missing warning layer.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=63,
                feature_set="all",
                threshold=-0.08,
                activation=0.32,
                stress=0.38,
                min_multiplier=0.35,
                train_window_days=1008,
                estimators=72,
            ),
            None,
        ),
        (
            "classic_dd8_plus_extra_trees",
            "Hybrid: ML tries to pre-empt risk, and classic drawdown control limits damage if ML is late.",
            dd8,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.08,
                activation=0.34,
                stress=0.55,
                min_multiplier=0.45,
                estimators=72,
            ),
            None,
        ),
        (
            "future_state_plus_aggressive_dd",
            "Hybrid: broad future-state risk and strategy-specific drawdown risk both need to agree before a larger throttle is applied.",
            base.strategy,
            _strategy_drawdown_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=-0.08,
                activation=0.34,
                stress=0.55,
                min_multiplier=0.45,
                estimators=72,
            ),
            _return_preserving_ml_profile(
                "sk_extra_trees",
                horizon_days=21,
                feature_set="ai",
                threshold=0.38,
                stress=0.78,
            ),
        ),
    )
    return tuple(
        _candidate(
            name=f"i86_aggressive_drawdown_ml_{slug}",
            role="aggressive_drawdown_ml_hybrid",
            phase="aggressive_drawdown_ml_hybrid",
            family="high_cagr_ai_escape_drawdown_hybrid",
            parent=base.name,
            hypothesis=(
                f"{hypothesis} Promotion should require a materially better drawdown/Calmar tradeoff, "
                "not just preserved CAGR."
            ),
            scenario_sizing=base.scenario_sizing,
            future_state_model=future_state_model,
            strategy_drawdown_model=drawdown_model,
            decision_sanity=_decision_sanity_profile("wide_cap"),
            strategy=strategy,
        )
        for slug, hypothesis, strategy, drawdown_model, future_state_model in specs
    )


def _reference_portfolio_candidates() -> tuple[ExperimentCandidate, ...]:
    return (
        _fixed_allocation_candidate(
            name="i41_ref_us_60_40",
            hypothesis="Classic 60/40 U.S. stock/bond allocation should be the simple balanced-policy reference.",
            allocation_weights={"SPY": 0.60, "AGG": 0.40},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_us_80_20",
            hypothesis="Classic 80/20 growth allocation tests whether more equity risk beats tactical complexity after drawdowns.",
            allocation_weights={"SPY": 0.80, "AGG": 0.20},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_us_90_10",
            hypothesis="Aggressive 90/10 allocation is a high-equity reference for long-horizon retirement growth.",
            allocation_weights={"SPY": 0.90, "AGG": 0.10},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_global_three_fund_80_20",
            hypothesis="Bogleheads-style global 80/20 allocation tests diversified equity beta plus bonds.",
            allocation_weights={"SPY": 0.48, "EFA": 0.24, "EEM": 0.08, "AGG": 0.20},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_global_three_fund_60_40",
            hypothesis="Bogleheads-style global 60/40 allocation is the diversified moderate-risk policy reference.",
            allocation_weights={"SPY": 0.36, "EFA": 0.18, "EEM": 0.06, "AGG": 0.40},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_permanent_portfolio",
            hypothesis="Permanent Portfolio tests equal risk assets, long duration, gold, and T-bill-style defense.",
            allocation_weights={"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "SHY": 0.25},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_golden_butterfly",
            hypothesis="Golden Butterfly tests total equity, small-cap, duration, cash-like bonds, and gold diversification.",
            allocation_weights={"SPY": 0.20, "IWM": 0.20, "TLT": 0.20, "SHY": 0.20, "GLD": 0.20},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_all_weather",
            hypothesis="All-weather-style allocation tests equity, duration, intermediate bonds, gold, and commodities.",
            allocation_weights={"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075},
        ),
        _fixed_allocation_candidate(
            name="i41_ref_growth_cash_barbell",
            hypothesis="Growth/cash barbell tests whether a simple QQQ/SPY/BIL policy is competitive with tactical off-ramps.",
            allocation_weights={"QQQ": 0.50, "SPY": 0.30, "BIL": 0.20},
        ),
    )


def _operating_system_candidates() -> tuple[ExperimentCandidate, ...]:
    return (
        _candidate(
            name="i21_os_ai_escape_scenario_sized",
            role="operating_system",
            phase="operating_system",
            family="ai_beta",
            hypothesis=(
                "AI beta can remain a satellite only when scenario pressure dynamically moves "
                "crowded growth risk into T-bills."
            ),
            scenario_sizing=_scenario_profile("fragile_ai"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QQQ",
                    "SMH",
                    "SOXX",
                    "IGV",
                    "NVDA",
                    "AVGO",
                    "MSFT",
                    "META",
                    "AMZN",
                    "PLTR",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.03,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                trend_filter_days=100,
                max_asset_weight=0.35,
                volatility_target=VolatilityTargetConfig(
                    annualized_volatility=0.14,
                    lookback_days=42,
                    max_leverage=1.0,
                ),
            ),
        ),
        _candidate(
            name="i21_os_cross_asset_guardrail",
            role="operating_system",
            phase="operating_system",
            family="core_cross_asset",
            hypothesis=(
                "A cross-asset core with scenario-driven risk budgeting should be less fragile "
                "than a QQQ-heavy winner-take-all approach."
            ),
            scenario_sizing=_scenario_profile("balanced"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                lookback_days=126,
                skip_days=21,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.40,
            ),
        ),
        _candidate(
            name="i21_os_quality_lowvol_credit_gate",
            role="operating_system",
            phase="operating_system",
            family="defensive_equity",
            hypothesis=(
                "Quality, dividend, and low-volatility equity exposure should survive transition "
                "regimes better when credit/volatility pressure sizes exposure down."
            ),
            scenario_sizing=_scenario_profile("defensive"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "QUAL",
                    "USMV",
                    "SPLV",
                    "SCHD",
                    "VIG",
                    "MOAT",
                    "COWZ",
                    "XLV",
                    "XLP",
                    "XLU",
                ],
                lookback_days=126,
                skip_days=21,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i21_os_credit_first_defense",
            role="operating_system",
            phase="operating_system",
            family="credit_rates",
            hypothesis=(
                "A credit/rates sleeve may become the warning system and capital-preservation "
                "system when equity momentum is too concentrated."
            ),
            scenario_sizing=_scenario_profile("defensive"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "HYG",
                    "JNK",
                    "LQD",
                    "BKLN",
                    "SRLN",
                    "JAAA",
                    "JBBB",
                    "IEF",
                    "TLT",
                    "TIP",
                    "GLD",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i21_os_sector_breadth_rotation",
            role="operating_system",
            phase="operating_system",
            family="sector_rotation",
            hypothesis=(
                "Sector breadth rotation can be an equity core if scenario pressure trims exposure "
                "during credit and volatility deterioration."
            ),
            scenario_sizing=_scenario_profile("balanced"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "XLK",
                    "XLF",
                    "XLY",
                    "XLP",
                    "XLE",
                    "XLV",
                    "XLI",
                    "XLU",
                    "XLB",
                    "XLRE",
                    "XLC",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.35,
            ),
        ),
        _candidate(
            name="i21_os_global_macro_rotation",
            role="operating_system",
            phase="operating_system",
            family="global_rotation",
            hypothesis=(
                "Global equity, dollar, commodity, gold, and duration leadership can reduce home "
                "bias and QQQ dependency during regime shifts."
            ),
            scenario_sizing=_scenario_profile("balanced"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "SPY",
                    "RSP",
                    "EFA",
                    "EEM",
                    "VEA",
                    "VWO",
                    "VGK",
                    "INDA",
                    "EWZ",
                    "GLD",
                    "UUP",
                    "DBC",
                    "TLT",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                max_asset_weight=0.30,
            ),
        ),
        _candidate(
            name="i21_os_private_credit_warning",
            role="operating_system",
            phase="operating_system",
            family="private_credit",
            hypothesis=(
                "Private-credit proxies and bank/loan weakness may warn before broad equity "
                "indexes fully price stress."
            ),
            scenario_sizing=_scenario_profile("defensive"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "BIZD",
                    "SRLN",
                    "BKLN",
                    "JAAA",
                    "JBBB",
                    "ARCC",
                    "MAIN",
                    "BXSL",
                    "OBDC",
                    "KRE",
                    "IEF",
                ],
                lookback_days=84,
                skip_days=10,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                max_asset_weight=0.30,
            ),
        ),
        _candidate(
            name="i21_os_oil_policy_shock_barbell",
            role="operating_system",
            phase="operating_system",
            family="policy_oil_shock",
            hypothesis=(
                "Policy and oil-shock regimes may require a barbell that can hold energy, gold, "
                "duration, dollar, or cash rather than forcing equity exposure."
            ),
            scenario_sizing=_scenario_profile("defensive"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "SPY",
                    "QQQ",
                    "XLE",
                    "XOP",
                    "USO",
                    "BNO",
                    "DBC",
                    "GLD",
                    "UUP",
                    "TLT",
                    "IEF",
                ],
                lookback_days=42,
                skip_days=5,
                top_n=3,
                defensive_ticker="BIL",
                min_return=0.01,
                weighting="inverse_volatility",
                max_asset_weight=0.30,
            ),
        ),
        _candidate(
            name="i21_os_ai_infra_real_assets",
            role="operating_system",
            phase="operating_system",
            family="ai_infrastructure",
            hypothesis=(
                "If AI capex persists but software/unit economics crack, infrastructure, power, "
                "and real-asset beneficiaries may be the better expression."
            ),
            scenario_sizing=_scenario_profile("balanced"),
            strategy=StrategyConfig(
                type="dual_momentum",
                tickers=[
                    "VRT",
                    "ETN",
                    "PWR",
                    "CEG",
                    "GEV",
                    "NRG",
                    "CCJ",
                    "SMH",
                    "SOXX",
                    "XLI",
                    "XLU",
                    "GLD",
                ],
                lookback_days=63,
                skip_days=5,
                top_n=4,
                defensive_ticker="BIL",
                min_return=0.02,
                ranking_metric="risk_adjusted_return",
                weighting="inverse_volatility",
                trend_filter_days=100,
                max_asset_weight=0.30,
            ),
        ),
        _candidate(
            name="i21_os_low_turnover_trend_defense",
            role="operating_system",
            phase="operating_system",
            family="trend_following",
            hypothesis=(
                "A low-turnover absolute trend system is the robustness benchmark: it should not "
                "win every year, but it must handle transitions cleanly."
            ),
            scenario_sizing=_scenario_profile("balanced"),
            strategy=StrategyConfig(
                type="absolute_momentum",
                tickers=["SPY", "QQQ", "RSP", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "DBC"],
                moving_average_days=126,
                defensive_ticker="BIL",
            ),
        ),
    )


def _candidate(
    *,
    name: str,
    role: str,
    phase: str,
    family: str,
    hypothesis: str,
    strategy: StrategyConfig,
    scenario_sizing: ScenarioSizingConfig | None = None,
    future_state_model: FutureStateModelConfig | None = None,
    strategy_drawdown_model: StrategyDrawdownModelConfig | None = None,
    decision_sanity: DecisionSanityConfig | None = None,
    parent: str | None = None,
) -> ExperimentCandidate:
    return ExperimentCandidate(
        name=name,
        role=role,
        phase=phase,
        family=family,
        parent=parent,
        hypothesis=hypothesis,
        strategy=strategy,
        scenario_sizing=scenario_sizing,
        future_state_model=future_state_model,
        strategy_drawdown_model=strategy_drawdown_model,
        decision_sanity=decision_sanity,
    )


def _fixed_allocation_candidate(
    *,
    name: str,
    hypothesis: str,
    allocation_weights: dict[str, float],
) -> ExperimentCandidate:
    return _candidate(
        name=name,
        role="reference_portfolio",
        phase="reference",
        family="reference_portfolio",
        hypothesis=hypothesis,
        strategy=StrategyConfig(
            type="fixed_allocation",
            tickers=list(allocation_weights),
            allocation_weights=allocation_weights,
            trend_filter_days=None,
            max_asset_weight=None,
        ),
    )


def _decision_sanity_profile(profile: str) -> DecisionSanityConfig:
    profiles = {
        "confirmation_cap": DecisionSanityConfig(profile="confirmation_cap"),
        "modest_cap": DecisionSanityConfig(profile="modest_cap", max_defensive_add=0.15),
        "wide_cap": DecisionSanityConfig(profile="wide_cap", max_defensive_add=0.30),
        "strict_gate": DecisionSanityConfig(profile="strict_gate", required_confirmation_breaks=3),
        "loose_gate": DecisionSanityConfig(profile="loose_gate", required_confirmation_breaks=1),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown decision-sanity profile: {profile}")
    return profiles[profile]


def _scenario_profile(profile: str) -> ScenarioSizingConfig:
    profiles = {
        "balanced": ScenarioSizingConfig(
            profile="balanced",
            stress_multiplier=0.35,
            transition_multiplier=0.65,
            fragile_upside_multiplier=0.80,
            min_multiplier=0.25,
        ),
        "defensive": ScenarioSizingConfig(
            profile="defensive",
            stress_multiplier=0.20,
            transition_multiplier=0.50,
            fragile_upside_multiplier=0.65,
            min_multiplier=0.15,
        ),
        "fragile_ai": ScenarioSizingConfig(
            profile="fragile_ai",
            stress_multiplier=0.25,
            transition_multiplier=0.55,
            fragile_upside_multiplier=0.50,
            min_multiplier=0.15,
            lookback_days=42,
        ),
        "aggressive": ScenarioSizingConfig(
            profile="aggressive",
            stress_multiplier=0.50,
            transition_multiplier=0.75,
            fragile_upside_multiplier=0.90,
            min_multiplier=0.35,
        ),
    }
    return profiles[profile]


def _future_state_profile(
    model: str,
    *,
    horizon_days: int,
    feature_set: str,
    train_window_days: int = 756,
    refit_every_days: int | None = None,
    k_neighbors: int = 80,
    dirichlet_prior_strength: float = 8.0,
    recency_half_life_days: int = 252,
    bayesian_feature_shrinkage: float = 12.0,
    sklearn_n_estimators: int = 24,
    sklearn_max_depth: int = 5,
    sklearn_min_samples_leaf: int = 20,
    sklearn_regularization_c: float = 0.70,
    stress_multiplier: float = 0.28,
    transition_multiplier: float = 0.62,
    fragile_upside_multiplier: float = 0.78,
    min_multiplier: float = 0.18,
    probability_smoothing: float = 0.10,
    risk_off_activation_probability: float = 0.0,
    transition_activation_probability: float = 0.0,
    fragile_activation_probability: float = 0.0,
) -> FutureStateModelConfig:
    resolved_refit = refit_every_days or (126 if model.startswith("sk_") else 21)
    return FutureStateModelConfig(
        model=cast(Any, model),
        horizon_days=horizon_days,
        feature_set=cast(Any, feature_set),
        train_window_days=train_window_days,
        refit_every_days=resolved_refit,
        k_neighbors=k_neighbors,
        dirichlet_prior_strength=dirichlet_prior_strength,
        recency_half_life_days=recency_half_life_days,
        bayesian_feature_shrinkage=bayesian_feature_shrinkage,
        sklearn_n_estimators=sklearn_n_estimators,
        sklearn_max_depth=sklearn_max_depth,
        sklearn_min_samples_leaf=sklearn_min_samples_leaf,
        sklearn_regularization_c=sklearn_regularization_c,
        stress_multiplier=stress_multiplier,
        transition_multiplier=transition_multiplier,
        fragile_upside_multiplier=fragile_upside_multiplier,
        min_multiplier=min_multiplier,
        probability_smoothing=probability_smoothing,
        risk_off_activation_probability=risk_off_activation_probability,
        transition_activation_probability=transition_activation_probability,
        fragile_activation_probability=fragile_activation_probability,
    )


def _evolve_from_previous_iteration(
    iteration: int,
    *,
    previous_scorecards: pd.DataFrame | None,
    previous_candidates: pd.DataFrame | None,
) -> list[ExperimentCandidate]:
    if previous_scorecards is None or previous_scorecards.empty:
        return []
    if previous_candidates is None or previous_candidates.empty:
        return []

    eligible = _diverse_parent_rows(previous_scorecards, iteration)

    manifest = previous_candidates.drop_duplicates("strategy", keep="last").set_index("strategy")
    candidates: list[ExperimentCandidate] = []
    phase = _phase_for_iteration(iteration)
    for parent_index, (_, row) in enumerate(eligible.iterrows()):
        parent_name = str(row["strategy"])
        if parent_name not in manifest.index:
            continue
        strategy = _strategy_from_manifest(manifest.loc[parent_name])
        if strategy is None or strategy.type not in {"relative_momentum", "dual_momentum"}:
            continue
        scenario_sizing = _scenario_sizing_from_manifest(manifest.loc[parent_name])
        future_state_model = _future_state_model_from_manifest(manifest.loc[parent_name])
        strategy_drawdown_model = _strategy_drawdown_model_from_manifest(manifest.loc[parent_name])
        decision_sanity = _decision_sanity_from_manifest(manifest.loc[parent_name])
        family = str(row.get("family", "evolved"))
        candidates.extend(
            _strategy_variants(
                iteration,
                parent_index=parent_index,
                parent_name=parent_name,
                parent_family=family,
                parent_role=str(row.get("role", "candidate_core")),
                parent_strategy=strategy,
                parent_scenario_sizing=scenario_sizing,
                parent_future_state_model=future_state_model,
                parent_strategy_drawdown_model=strategy_drawdown_model,
                parent_decision_sanity=decision_sanity,
                phase=phase,
            )
        )
    return _dedupe_candidates(candidates)


def _diverse_parent_rows(previous_scorecards: pd.DataFrame, iteration: int) -> pd.DataFrame:
    promoted = previous_scorecards[
        previous_scorecards["promotion_decision"].isin(
            ["promote_candidate", "evolve_next_iteration"]
        )
    ].copy()
    if promoted.empty:
        promoted = previous_scorecards.copy()

    for column, default in {
        "family": "unknown",
        "calmar": 0.0,
        "worst_3y_cagr": -1.0,
    }.items():
        if column not in promoted.columns:
            promoted[column] = default

    promoted = promoted.sort_values(
        ["promotion_score", "calmar", "worst_3y_cagr"],
        ascending=False,
    )
    family_champions = promoted.drop_duplicates("family", keep="first")
    if family_champions.empty:
        return promoted.head(3)

    anchor = family_champions.head(1)
    rotating_pool = family_champions.iloc[1:]
    if rotating_pool.empty:
        return anchor

    offset = (iteration - 4) % len(rotating_pool)
    rotated = pd.concat([rotating_pool.iloc[offset:], rotating_pool.iloc[:offset]])
    return pd.concat([anchor, rotated.head(2)], ignore_index=True)


def _strategy_variants(
    iteration: int,
    *,
    parent_index: int,
    parent_name: str,
    parent_family: str,
    parent_role: str,
    parent_strategy: StrategyConfig,
    parent_scenario_sizing: ScenarioSizingConfig | None,
    parent_future_state_model: FutureStateModelConfig | None,
    parent_strategy_drawdown_model: StrategyDrawdownModelConfig | None,
    parent_decision_sanity: DecisionSanityConfig | None,
    phase: str,
) -> list[ExperimentCandidate]:
    lookback = parent_strategy.lookback_days
    skip = parent_strategy.skip_days
    top_n = parent_strategy.top_n
    variant_specs: list[tuple[str, str, dict[str, object], ScenarioSizingConfig | None]] = [
        (
            "faster",
            "Increase responsiveness to market transitions by shortening lookback and skip windows.",
            {
                "lookback_days": max(42, int(lookback * 0.70)),
                "skip_days": max(0, int(skip * 0.50)),
            },
            parent_scenario_sizing,
        ),
        (
            "slower",
            "Reduce whipsaw by extending lookback while keeping the same universe.",
            {
                "lookback_days": min(315, int(lookback * 1.35)),
                "skip_days": max(skip, 21),
            },
            parent_scenario_sizing,
        ),
        (
            "riskadj",
            "Promote smoother winners by ranking and sizing on risk-adjusted momentum.",
            {
                "ranking_metric": "risk_adjusted_return",
                "weighting": "risk_adjusted_score",
            },
            parent_scenario_sizing,
        ),
        (
            "invvol",
            "Keep the same selection but allocate less to high-volatility winners.",
            {
                "weighting": "inverse_volatility",
            },
            parent_scenario_sizing,
        ),
        (
            "trend",
            "Require 200-day trend confirmation before holding selected risk assets.",
            {
                "trend_filter_days": 200,
            },
            parent_scenario_sizing,
        ),
        (
            "escape",
            "Demand a stronger absolute return hurdle before holding risk assets.",
            {
                "min_return": max(parent_strategy.min_return, 0.02),
                "trend_filter_days": 100,
            },
            parent_scenario_sizing,
        ),
        (
            "vol10",
            "Throttle the strategy to a 10% annualized realized-volatility target.",
            {
                "volatility_target": {
                    "annualized_volatility": 0.10,
                    "lookback_days": 63,
                    "max_leverage": 1.0,
                },
            },
            parent_scenario_sizing,
        ),
        (
            "dd07",
            "Cut exposure aggressively after a 7% rolling strategy drawdown.",
            {
                "drawdown_control": {
                    "equity_lookback_days": 126,
                    "max_drawdown": -0.07,
                    "risk_multiplier": 0.25,
                },
            },
            parent_scenario_sizing,
        ),
        (
            "concentrated",
            "Test whether fewer holdings improve upside enough to justify concentration.",
            {
                "top_n": max(1, top_n - 1),
                "max_asset_weight": 0.65,
            },
            parent_scenario_sizing,
        ),
        (
            "broader",
            "Test whether one extra holding improves regime durability.",
            {
                "top_n": min(len(parent_strategy.tickers), top_n + 1),
                "max_asset_weight": 0.40,
            },
            parent_scenario_sizing,
        ),
        (
            "scenario_def",
            "Add defensive scenario position sizing to test whether the strategy survives regime transitions.",
            {},
            _scenario_profile("defensive"),
        ),
        (
            "scenario_bal",
            "Add balanced scenario position sizing to turn the strategy into a full operating-system candidate.",
            {},
            _scenario_profile("balanced"),
        ),
    ]
    start = (iteration + parent_index) % len(variant_specs)
    selected_specs = [variant_specs[(start + offset) % len(variant_specs)] for offset in range(3)]

    candidates: list[ExperimentCandidate] = []
    for suffix, hypothesis, updates, scenario_sizing in selected_specs:
        strategy = _clone_strategy(parent_strategy, **updates)
        name = f"i{iteration:02d}_{_short_name(parent_name)}_{suffix}"
        candidates.append(
            _candidate(
                name=name,
                role=parent_role,
                phase=phase,
                family=parent_family,
                parent=parent_name,
                hypothesis=f"{hypothesis} Parent: {parent_name}.",
                strategy=strategy,
                scenario_sizing=scenario_sizing,
                future_state_model=parent_future_state_model,
                strategy_drawdown_model=parent_strategy_drawdown_model,
                decision_sanity=parent_decision_sanity,
            )
        )
    return candidates


def _clone_strategy(strategy: StrategyConfig, **updates: object) -> StrategyConfig:
    data = strategy.model_dump(mode="json")
    data.update(updates)
    return StrategyConfig.model_validate(data)


def _strategy_from_manifest(row: pd.Series) -> StrategyConfig | None:
    raw = row.get("strategy_json")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return StrategyConfig.model_validate(json.loads(raw))
    except (TypeError, ValueError):
        return None


def _decision_sanity_from_manifest(row: pd.Series) -> DecisionSanityConfig | None:
    raw = row.get("decision_sanity_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return DecisionSanityConfig(**values)
    except TypeError:
        return None


def _scenario_sizing_from_manifest(row: pd.Series) -> ScenarioSizingConfig | None:
    raw = row.get("scenario_sizing_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return ScenarioSizingConfig(**values)
    except TypeError:
        return None


def _future_state_model_from_manifest(row: pd.Series) -> FutureStateModelConfig | None:
    raw = row.get("future_state_model_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return FutureStateModelConfig(**values)
    except TypeError:
        return None


def _strategy_drawdown_model_from_manifest(row: pd.Series) -> StrategyDrawdownModelConfig | None:
    raw = row.get("strategy_drawdown_model_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return StrategyDrawdownModelConfig(**values)
    except TypeError:
        return None


def _future_state_label(config: FutureStateModelConfig | None) -> str:
    if config is None:
        return ""
    return f"{config.model}_{config.feature_set}_{config.horizon_days}d"


def _strategy_drawdown_label(config: StrategyDrawdownModelConfig | None) -> str:
    if config is None:
        return ""
    threshold = abs(float(config.future_drawdown_threshold))
    return f"{config.model}_{config.feature_set}_{config.horizon_days}d_dd{threshold:.0%}"


def _phase_for_iteration(iteration: int) -> str:
    if iteration <= 3:
        return "broad"
    if iteration <= 12:
        return "deep"
    if iteration <= 16:
        return "stress"
    if iteration <= 20:
        return "creative"
    if iteration <= 40:
        return "operating_system"
    return "creative"


def _short_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in name.lower())
    parts = [part for part in cleaned.split("_") if part and not part.startswith("i")]
    return "_".join(parts[:4])[:38]


def _dedupe_candidates(candidates: list[ExperimentCandidate]) -> list[ExperimentCandidate]:
    seen: set[str] = set()
    deduped: list[ExperimentCandidate] = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        deduped.append(candidate)
    return deduped


def _retag_candidates(
    candidates: tuple[ExperimentCandidate, ...],
    iteration: int,
) -> tuple[ExperimentCandidate, ...]:
    phase = _phase_for_iteration(iteration)
    retagged = []
    for candidate in candidates:
        name = f"i{iteration:02d}_{_short_name(candidate.name)}"
        retagged.append(
            ExperimentCandidate(
                name=name,
                role=candidate.role,
                phase=phase,
                family=candidate.family,
                parent=candidate.parent,
                hypothesis=candidate.hypothesis,
                strategy=candidate.strategy,
                scenario_sizing=candidate.scenario_sizing,
                future_state_model=candidate.future_state_model,
                strategy_drawdown_model=candidate.strategy_drawdown_model,
                decision_sanity=candidate.decision_sanity,
            )
        )
    return tuple(retagged)


def build_experiment_scorecard(
    candidates: tuple[ExperimentCandidate, ...],
    metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
    regime_summary: pd.DataFrame | None = None,
    walk_forward_summary: pd.DataFrame | None = None,
    benchmark_metrics: pd.DataFrame | None = None,
    operability_metrics: pd.DataFrame | None = None,
    transition_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidate_meta = pd.DataFrame(
        [
            {
                "strategy": candidate.name,
                "display_name": strategy_display_name(
                    candidate.name,
                    family=candidate.family,
                    phase=candidate.phase,
                ),
                "phase": candidate.phase,
                "family": candidate.family,
                "role": candidate.role,
                "parent": candidate.parent or "",
                "scenario_sizing": (
                    candidate.scenario_sizing.profile if candidate.scenario_sizing else ""
                ),
                "future_state_model": (
                    _future_state_label(candidate.future_state_model)
                    if candidate.future_state_model
                    else ""
                ),
                "strategy_drawdown_model": (
                    _strategy_drawdown_label(candidate.strategy_drawdown_model)
                    if candidate.strategy_drawdown_model
                    else ""
                ),
                "decision_sanity": (
                    candidate.decision_sanity.profile if candidate.decision_sanity else ""
                ),
                "hypothesis": candidate.hypothesis,
            }
            for candidate in candidates
        ]
    ).set_index("strategy")
    summary = metrics.join(candidate_meta)
    summary.index.name = "strategy"
    summary["worst_1y_cagr"] = _window_stat(window_summary, "1y", "worst_cagr")
    summary["worst_3y_cagr"] = _window_stat(window_summary, "3y", "worst_cagr")
    summary["worst_5y_cagr"] = _window_stat(window_summary, "5y", "worst_cagr")
    summary["positive_1y_window_rate"] = _window_stat(window_summary, "1y", "positive_window_rate")
    summary = _add_regime_context(summary, regime_summary)
    summary = _add_walk_forward_context(summary, walk_forward_summary)
    summary = _add_benchmark_context(summary, benchmark_metrics)
    summary = _add_operability_context(summary, operability_metrics)
    summary = _add_transition_context(summary, transition_metrics)
    summary["robustness_score"] = _robustness_score(summary)
    summary["promotion_score"] = _promotion_score(summary)
    summary["promotion_decision"] = summary.apply(_promotion_decision, axis=1)
    summary["monitoring_readiness_score"] = _monitoring_readiness_score(summary)
    summary["monitoring_readiness_label"] = summary.apply(_monitoring_readiness_label, axis=1)
    summary["benchmark_knockout_score"] = _benchmark_knockout_score(summary)
    summary["benchmark_knockout_label"] = summary.apply(_benchmark_knockout_label, axis=1)
    summary["confidence_score"] = _confidence_score(summary)
    summary["confidence_label"] = summary.apply(_confidence_label, axis=1)
    summary["deployment_blockers"] = summary.apply(_deployment_blockers, axis=1)
    summary = add_research_status(summary)
    columns = [
        "display_name",
        "phase",
        "family",
        "role",
        "parent",
        "scenario_sizing",
        "future_state_model",
        "strategy_drawdown_model",
        "decision_sanity",
        "research_status",
        "prune_reason",
        "promotion_decision",
        "promotion_score",
        "monitoring_readiness_score",
        "monitoring_readiness_label",
        "confidence_score",
        "confidence_label",
        "deployment_blockers",
        "benchmark_knockout_score",
        "benchmark_knockout_label",
        "robustness_score",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "calmar_excess_vs_spy",
        "calmar_excess_vs_qqq",
        "average_turnover",
        "material_trade_days_per_year",
        "mean_days_between_material_trades",
        "median_material_turnover",
        "max_single_day_turnover",
        "operability_score",
        "operability_label",
        "average_risk_weight",
        "low_risk_day_rate",
        "median_reentry_days",
        "reentry_cycles",
        "reentry_score",
        "risk_cycle_label",
        "worst_1y_cagr",
        "worst_3y_cagr",
        "worst_5y_cagr",
        "positive_1y_window_rate",
        "walk_forward_median_cagr",
        "walk_forward_worst_cagr",
        "walk_forward_positive_rate",
        "walk_forward_median_calmar",
        "worst_regime_return",
        "worst_regime_cagr",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "transition_regime_hit_rate",
        "transition_regime_return",
        "regime_positive_rate",
        "hypothesis",
    ]
    return summary[columns].sort_values("promotion_score", ascending=False)


def apply_scenario_position_sizing(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: ScenarioSizingConfig,
    *,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    signals = build_scenario_sizing_signals(prices, config)
    multiplier = signals["risk_multiplier"].reindex(target_weights.index).fillna(1.0)
    adjusted = target_weights.copy().astype(float)
    if defensive_ticker and defensive_ticker not in adjusted.columns:
        adjusted[defensive_ticker] = 0.0

    risk_columns = [
        column for column in adjusted.columns if not defensive_ticker or column != defensive_ticker
    ]
    adjusted.loc[:, risk_columns] = adjusted[risk_columns].mul(multiplier, axis=0)
    if defensive_ticker:
        residual = (1.0 - adjusted.sum(axis=1)).clip(lower=0.0)
        adjusted.loc[:, defensive_ticker] = adjusted[defensive_ticker] + residual
    return adjusted.clip(lower=0.0)


def apply_decision_sanity_overlay(
    base_target_weights: pd.DataFrame,
    adjusted_target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: DecisionSanityConfig,
    *,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    if not defensive_ticker:
        return adjusted_target_weights.clip(lower=0.0)

    tickers = sorted(
        set(base_target_weights.columns) | set(adjusted_target_weights.columns) | {defensive_ticker}
    )
    index = adjusted_target_weights.index
    base = (
        base_target_weights.reindex(index=index, columns=tickers)
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    adjusted = (
        adjusted_target_weights.reindex(index=index, columns=tickers)
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    signals = build_decision_sanity_signals(prices, config).reindex(index).ffill().fillna(0.0)
    cap_active = signals["sanity_cap_active"].astype(bool)

    base_defensive = base[defensive_ticker].clip(lower=0.0, upper=1.0)
    adjusted_defensive = adjusted[defensive_ticker].clip(lower=0.0, upper=1.0)
    max_defensive = (base_defensive + config.max_defensive_add).clip(lower=0.0, upper=1.0)
    capped_defensive = adjusted_defensive.mask(
        cap_active & (adjusted_defensive > max_defensive),
        max_defensive,
    )
    freed_weight = (adjusted_defensive - capped_defensive).clip(lower=0.0)
    adjusted.loc[:, defensive_ticker] = capped_defensive

    risk_columns = [column for column in tickers if column != defensive_ticker]
    if not risk_columns or float(freed_weight.max()) <= 0.0:
        return _normalize_weight_frame(adjusted)

    candidate_basis = adjusted[risk_columns].clip(lower=0.0)
    base_basis = base[risk_columns].clip(lower=0.0)
    candidate_sum = candidate_basis.sum(axis=1)
    base_sum = base_basis.sum(axis=1)
    equal_basis = pd.DataFrame(
        1.0 / len(risk_columns),
        index=adjusted.index,
        columns=risk_columns,
    )
    allocation_basis = candidate_basis.div(candidate_sum.replace(0.0, float("nan")), axis=0)
    fallback_basis = base_basis.div(base_sum.replace(0.0, float("nan")), axis=0)
    allocation_basis = allocation_basis.fillna(fallback_basis).fillna(equal_basis).astype(float)
    adjusted.loc[:, risk_columns] = (
        adjusted[risk_columns].astype(float)
        + allocation_basis.mul(
            freed_weight.astype(float),
            axis=0,
        )
    ).astype(float)
    return _normalize_weight_frame(adjusted)


def apply_operability_hysteresis(
    target_weights: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    min_change = float(strategy.cycle_min_rebalance_change or 0.0)
    max_step = float(strategy.cycle_max_step_change or 1.0)
    min_hold_days = int(strategy.cycle_min_hold_days or 0)
    if min_change < 0.05 and max_step >= 0.999 and min_hold_days <= 0:
        return target_weights

    desired = _normalize_weight_frame(target_weights).sort_index()
    if desired.empty:
        return desired

    current = desired.iloc[0].astype(float).clip(lower=0.0)
    last_trade_position = 0
    rows = [current.copy()]
    for position, (_, target) in enumerate(desired.iloc[1:].iterrows(), start=1):
        target = target.astype(float).clip(lower=0.0)
        delta = target - current
        turnover = float(delta.abs().sum())
        risk_off_override = _risk_off_override_active(
            current,
            target,
            defensive_ticker=strategy.defensive_ticker,
            threshold=float(strategy.cycle_risk_off_override_change or 1.0),
        )
        held_long_enough = position - last_trade_position >= min_hold_days
        if turnover < min_change or (not held_long_enough and not risk_off_override):
            rows.append(current.copy())
            continue

        if max_step > 0.0 and turnover > max_step:
            delta = delta * (max_step / turnover)
        current = (current + delta).clip(lower=0.0)
        total = float(current.sum())
        if total > 1.0:
            current = current / total
        last_trade_position = position
        rows.append(current.copy())

    smoothed = pd.DataFrame(rows, index=desired.index, columns=desired.columns)
    return _normalize_weight_frame(smoothed)


def _risk_off_override_active(
    current: pd.Series,
    target: pd.Series,
    *,
    defensive_ticker: str | None,
    threshold: float,
) -> bool:
    if not defensive_ticker or defensive_ticker not in target.index or threshold <= 0.0:
        return False
    current_risk = 1.0 - float(current.get(defensive_ticker, 0.0))
    target_risk = 1.0 - float(target.get(defensive_ticker, 0.0))
    return current_risk - target_risk >= threshold


def build_decision_sanity_signals(
    prices: pd.DataFrame,
    config: DecisionSanityConfig,
) -> pd.DataFrame:
    signal_config = ScenarioSizingConfig(
        profile=config.profile,
        lookback_days=config.lookback_days,
    )
    signals = build_scenario_sizing_signals(prices, signal_config)
    threshold = config.confirmation_threshold
    credit_break = signals["credit"] <= -threshold
    volatility_break = signals["liquidity_pressure"] >= threshold
    breadth_break = signals["breadth"] <= -threshold
    trend_break = signals["market_trend"] <= -threshold
    confirmation_break_count = (
        credit_break.astype(int)
        + volatility_break.astype(int)
        + breadth_break.astype(int)
        + trend_break.astype(int)
    )
    left_tail_confirmed = signals["risk_off_pressure"] >= config.left_tail_pressure_threshold
    sanity_cap_active = (
        confirmation_break_count < config.required_confirmation_breaks
    ) & ~left_tail_confirmed
    output = signals.copy()
    output["credit_break"] = credit_break
    output["volatility_break"] = volatility_break
    output["breadth_break"] = breadth_break
    output["trend_break"] = trend_break
    output["confirmation_break_count"] = confirmation_break_count
    output["left_tail_confirmed"] = left_tail_confirmed
    output["sanity_cap_active"] = sanity_cap_active
    return output


def _normalize_weight_frame(weights: pd.DataFrame) -> pd.DataFrame:
    clipped = weights.astype(float).clip(lower=0.0).fillna(0.0)
    row_sum = clipped.sum(axis=1)
    over = row_sum > 1.0
    if over.any():
        clipped.loc[over] = clipped.loc[over].div(row_sum.loc[over], axis=0)
    return clipped


def build_scenario_sizing_signals(
    prices: pd.DataFrame,
    config: ScenarioSizingConfig,
) -> pd.DataFrame:
    filled = prices.ffill().sort_index()
    lookback = config.lookback_days
    market_trend = _mean_signal(
        [
            _trend_score(filled, "SPY", lookback),
            _trend_score(filled, "QQQ", lookback),
            _trend_score(filled, "RSP", lookback),
        ],
        filled.index,
    )
    breadth = _mean_signal(
        [
            _relative_score(filled, "RSP", "SPY", lookback, scale=0.08),
            _relative_score(filled, "IWM", "SPY", lookback, scale=0.10),
        ],
        filled.index,
    )
    credit = _mean_signal(
        [
            _relative_score(filled, "HYG", "LQD", lookback, scale=0.06),
            _momentum_score(filled, "HYG", lookback, scale=0.08),
        ],
        filled.index,
    )
    liquidity_pressure = _mean_signal(
        [
            _momentum_score(filled, "VIXY", lookback, scale=0.35),
            _momentum_score(filled, "UUP", lookback, scale=0.08),
        ],
        filled.index,
    ).clip(lower=0.0)
    oil_inflation_pressure = _mean_signal(
        [
            _momentum_score(filled, "USO", lookback, scale=0.20),
            _momentum_score(filled, "DBC", lookback, scale=0.12),
        ],
        filled.index,
    ).clip(lower=0.0)
    ai_concentration = _mean_signal(
        [
            _relative_score(filled, "QQQ", "RSP", lookback, scale=0.08),
            _relative_score(filled, "SMH", "SPY", lookback, scale=0.12),
        ],
        filled.index,
    ).clip(lower=0.0)
    drawdown_pressure = _drawdown_pressure(filled, "SPY", lookback)

    adverse_market = _adverse(market_trend)
    adverse_breadth = _adverse(breadth)
    adverse_credit = _adverse(credit)
    risk_off_pressure = (
        0.30 * adverse_market
        + 0.25 * adverse_credit
        + 0.20 * liquidity_pressure
        + 0.15 * drawdown_pressure
        + 0.10 * oil_inflation_pressure
    ).clip(lower=0.0, upper=1.0)
    transition_pressure = (
        0.35 * adverse_breadth
        + 0.25 * adverse_credit
        + 0.20 * liquidity_pressure
        + 0.20 * oil_inflation_pressure
    ).clip(lower=0.0, upper=1.0)
    fragile_upside_pressure = (
        ai_concentration * adverse_breadth * (1.0 - 0.50 * risk_off_pressure)
    ).clip(lower=0.0, upper=1.0)

    multiplier = pd.Series(config.risk_on_multiplier, index=filled.index)
    multiplier -= risk_off_pressure * (config.risk_on_multiplier - config.stress_multiplier)
    multiplier -= transition_pressure * (config.risk_on_multiplier - config.transition_multiplier)
    multiplier -= fragile_upside_pressure * (
        config.risk_on_multiplier - config.fragile_upside_multiplier
    )
    multiplier = multiplier.clip(lower=config.min_multiplier, upper=config.max_multiplier)

    return pd.DataFrame(
        {
            "market_trend": market_trend,
            "breadth": breadth,
            "credit": credit,
            "liquidity_pressure": liquidity_pressure,
            "oil_inflation_pressure": oil_inflation_pressure,
            "ai_concentration": ai_concentration,
            "drawdown_pressure": drawdown_pressure,
            "risk_off_pressure": risk_off_pressure,
            "transition_pressure": transition_pressure,
            "fragile_upside_pressure": fragile_upside_pressure,
            "risk_multiplier": multiplier,
        },
        index=filled.index,
    ).fillna(
        {
            "market_trend": 0.0,
            "breadth": 0.0,
            "credit": 0.0,
            "liquidity_pressure": 0.0,
            "oil_inflation_pressure": 0.0,
            "ai_concentration": 0.0,
            "drawdown_pressure": 0.0,
            "risk_off_pressure": 0.0,
            "transition_pressure": 0.0,
            "fragile_upside_pressure": 0.0,
            "risk_multiplier": 1.0,
        }
    )


def _mean_signal(series: list[pd.Series], index: pd.Index) -> pd.Series:
    available = [value.reindex(index) for value in series if not value.empty]
    if not available:
        return pd.Series(0.0, index=index)
    return pd.concat(available, axis=1).mean(axis=1).fillna(0.0)


def _trend_score(prices: pd.DataFrame, ticker: str, lookback: int) -> pd.Series:
    if ticker not in prices:
        return pd.Series(dtype=float)
    moving_average = prices[ticker].rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    return ((prices[ticker] / moving_average - 1.0) / 0.08).clip(lower=-1.0, upper=1.0)


def _relative_score(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    lookback: int,
    *,
    scale: float,
) -> pd.Series:
    if numerator not in prices or denominator not in prices:
        return pd.Series(dtype=float)
    relative = prices[numerator] / prices[denominator]
    return (relative.pct_change(lookback, fill_method=None) / scale).clip(lower=-1.0, upper=1.0)


def _momentum_score(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int,
    *,
    scale: float,
) -> pd.Series:
    if ticker not in prices:
        return pd.Series(dtype=float)
    return (prices[ticker].pct_change(lookback, fill_method=None) / scale).clip(
        lower=-1.0,
        upper=1.0,
    )


def _drawdown_pressure(prices: pd.DataFrame, ticker: str, lookback: int) -> pd.Series:
    if ticker not in prices:
        return pd.Series(0.0, index=prices.index)
    rolling_high = prices[ticker].rolling(lookback, min_periods=max(2, lookback // 2)).max()
    drawdown = prices[ticker] / rolling_high - 1.0
    return (-drawdown / 0.20).clip(lower=0.0, upper=1.0).fillna(0.0)


def _adverse(signal: pd.Series) -> pd.Series:
    return ((-signal + 1.0) / 2.0).clip(lower=0.0, upper=1.0).fillna(0.5)


def _candidate_tickers(candidates: tuple[ExperimentCandidate, ...]) -> set[str]:
    tickers: set[str] = set()
    for candidate in candidates:
        tickers.update(candidate.strategy.tickers)
        tickers.update(candidate.strategy.satellite_tickers)
        if candidate.strategy.defensive_ticker:
            tickers.add(candidate.strategy.defensive_ticker)
    return tickers


def _strategy_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    defensive_ticker: str | None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*tickers, *([defensive_ticker] if defensive_ticker else [])]))
    return prices[columns].dropna(how="all")


def _window_stat(window_summary: pd.DataFrame, window: str, column: str) -> pd.Series:
    if window_summary.empty:
        return pd.Series(dtype=float)
    frame = window_summary.reset_index()
    strategy_column = "strategy" if "strategy" in frame.columns else "name"
    selected = frame[frame["window"] == window].set_index(strategy_column)
    return selected[column]


def _benchmark_metrics(prices: pd.DataFrame, execution: ExecutionConfig) -> pd.DataFrame:
    benchmark_results: dict[str, BacktestResult] = {}
    calculated_metrics: list[PerformanceMetrics] = []
    for ticker in ["SPY", "QQQ"]:
        if ticker not in prices.columns:
            continue
        benchmark_prices = prices[[ticker]].dropna(how="all")
        if benchmark_prices.empty:
            continue
        target_weights = pd.DataFrame(1.0, index=benchmark_prices.index, columns=[ticker])
        result = run_backtest(
            f"benchmark_{ticker.lower()}",
            benchmark_prices,
            target_weights,
            execution,
        )
        benchmark_results[result.name] = result
        calculated_metrics.append(
            calculate_metrics(
                name=result.name,
                returns=result.returns,
                equity=result.equity,
                turnover=result.turnover,
                transaction_costs=result.transaction_costs,
            )
        )
    if not benchmark_results:
        return pd.DataFrame()
    return metrics_frame(calculated_metrics)


def _operability_metrics_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        turnover = result.turnover.dropna().astype(float).clip(lower=0.0)
        if turnover.empty:
            continue
        turnover.iloc[0] = 0.0
        gross_exposure = result.weights.abs().sum(axis=1).reindex(turnover.index).fillna(0.0)
        startup_trade = (gross_exposure.shift(1).fillna(0.0) <= 1e-9) & (gross_exposure > 1e-9)
        turnover.loc[startup_trade] = 0.0
        years = max((turnover.index[-1] - turnover.index[0]).days / 365.25, 1 / 365.25)
        material = turnover[turnover >= 0.05]
        material_days_per_year = len(material) / years
        mean_gap = _mean_event_gap_days(turnover.index, material.index)
        max_turnover = float(turnover.max())
        median_material = float(material.median()) if not material.empty else 0.0
        average_turnover = float(turnover.mean())
        operability_score = _operability_score(
            material_days_per_year=material_days_per_year,
            mean_days_between_material_trades=mean_gap,
            max_single_day_turnover=max_turnover,
            average_turnover=average_turnover,
        )
        rows.append(
            {
                "strategy": name,
                "material_trade_days_per_year": material_days_per_year,
                "mean_days_between_material_trades": mean_gap,
                "median_material_turnover": median_material,
                "max_single_day_turnover": max_turnover,
                "operability_score": operability_score,
                "operability_label": _operability_label(
                    material_days_per_year=material_days_per_year,
                    max_single_day_turnover=max_turnover,
                    average_turnover=average_turnover,
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("strategy")


def _transition_metrics_frame(
    candidates: tuple[ExperimentCandidate, ...],
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    rows = []
    for name, result in results.items():
        weights = result.weights.sort_index().fillna(0.0)
        if weights.empty:
            continue
        candidate = candidate_by_name.get(name)
        defensive_ticker = candidate.strategy.defensive_ticker if candidate else None
        risk_weight = _risk_weight_series(weights, defensive_ticker)
        reentry_days = _reentry_days(risk_weight)
        median_reentry_days = float(pd.Series(reentry_days).median()) if reentry_days else float("nan")
        low_risk_day_rate = float((risk_weight <= 0.35).mean())
        average_risk_weight = float(risk_weight.mean())
        min_risk_weight = float(risk_weight.min())
        latest_risk_weight = float(risk_weight.iloc[-1])
        reentry_score = _reentry_score(
            median_reentry_days=median_reentry_days,
            reentry_cycles=len(reentry_days),
            low_risk_day_rate=low_risk_day_rate,
            min_risk_weight=min_risk_weight,
        )
        rows.append(
            {
                "strategy": name,
                "average_risk_weight": average_risk_weight,
                "min_risk_weight": min_risk_weight,
                "latest_risk_weight": latest_risk_weight,
                "low_risk_day_rate": low_risk_day_rate,
                "median_reentry_days": median_reentry_days,
                "reentry_cycles": len(reentry_days),
                "reentry_score": reentry_score,
                "risk_cycle_label": _risk_cycle_label(
                    low_risk_day_rate=low_risk_day_rate,
                    reentry_cycles=len(reentry_days),
                    median_reentry_days=median_reentry_days,
                    min_risk_weight=min_risk_weight,
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("strategy")


def _risk_weight_series(weights: pd.DataFrame, defensive_ticker: str | None) -> pd.Series:
    if defensive_ticker and defensive_ticker in weights.columns:
        return (1.0 - weights[defensive_ticker].astype(float)).clip(lower=0.0, upper=1.0)
    return weights.sum(axis=1).astype(float).clip(lower=0.0, upper=1.0)


def _mean_event_gap_days(index: pd.Index, event_index: pd.Index) -> float:
    if len(event_index) <= 1:
        return float("nan")
    positions = pd.Series(range(len(index)), index=index).reindex(event_index).dropna().astype(int)
    if len(positions) <= 1:
        return float("nan")
    return float(positions.diff().dropna().mean())


def _reentry_days(risk_weight: pd.Series) -> list[int]:
    values = risk_weight.reset_index(drop=True).astype(float)
    days: list[int] = []
    low_start: int | None = None
    for position, value in enumerate(values):
        if low_start is None and value <= 0.35:
            low_start = position
            continue
        if low_start is not None and value >= 0.65:
            days.append(position - low_start)
            low_start = None
    return days


def _operability_score(
    *,
    material_days_per_year: float,
    mean_days_between_material_trades: float,
    max_single_day_turnover: float,
    average_turnover: float,
) -> float:
    cadence_score = 1.0 - _clip01((material_days_per_year - 8.0) / 44.0)
    gap_score = _clip01((mean_days_between_material_trades if mean_days_between_material_trades == mean_days_between_material_trades else 63.0) / 21.0)
    max_trade_score = 1.0 - _clip01((max_single_day_turnover - 0.35) / 0.65)
    average_turnover_score = 1.0 - _clip01((average_turnover - 0.04) / 0.14)
    return float(
        0.35 * cadence_score
        + 0.25 * gap_score
        + 0.25 * max_trade_score
        + 0.15 * average_turnover_score
    )


def _operability_label(
    *,
    material_days_per_year: float,
    max_single_day_turnover: float,
    average_turnover: float,
) -> str:
    if material_days_per_year <= 18.0 and max_single_day_turnover <= 0.65 and average_turnover <= 0.08:
        return "paper_operable"
    if material_days_per_year <= 40.0 and max_single_day_turnover <= 0.90:
        return "review_churn"
    return "too_twitchy"


def _reentry_score(
    *,
    median_reentry_days: float,
    reentry_cycles: int,
    low_risk_day_rate: float,
    min_risk_weight: float,
) -> float:
    if reentry_cycles == 0:
        return 0.45 if min_risk_weight > 0.55 else 0.25
    speed_score = 1.0 - _clip01((median_reentry_days - 10.0) / 53.0)
    cycle_score = _clip01(reentry_cycles / 4.0)
    sticky_penalty = _clip01((low_risk_day_rate - 0.35) / 0.35)
    return float((0.65 * speed_score + 0.35 * cycle_score) * (1.0 - 0.35 * sticky_penalty))


def _risk_cycle_label(
    *,
    low_risk_day_rate: float,
    reentry_cycles: int,
    median_reentry_days: float,
    min_risk_weight: float,
) -> str:
    if reentry_cycles and median_reentry_days == median_reentry_days and median_reentry_days <= 42:
        return "risk_off_then_reenters"
    if low_risk_day_rate >= 0.35 and not reentry_cycles:
        return "risk_off_sticky"
    if min_risk_weight > 0.55:
        return "mostly_risk_on"
    return "mixed_cycle"


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _add_benchmark_context(
    summary: pd.DataFrame,
    benchmark_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    enriched = summary.copy()
    for ticker in ["spy", "qqq"]:
        benchmark_name = f"benchmark_{ticker}"
        if benchmark_metrics is None or benchmark_name not in benchmark_metrics.index:
            enriched[f"excess_cagr_vs_{ticker}"] = float("nan")
            enriched[f"drawdown_improvement_vs_{ticker}"] = float("nan")
            enriched[f"calmar_excess_vs_{ticker}"] = float("nan")
            continue

        benchmark = benchmark_metrics.loc[benchmark_name]
        enriched[f"excess_cagr_vs_{ticker}"] = enriched["cagr"] - float(benchmark["cagr"])
        enriched[f"drawdown_improvement_vs_{ticker}"] = enriched["max_drawdown"] - float(
            benchmark["max_drawdown"]
        )
        enriched[f"calmar_excess_vs_{ticker}"] = enriched["calmar"] - float(benchmark["calmar"])
    return enriched


def _add_regime_context(
    summary: pd.DataFrame,
    regime_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "worst_regime_return",
        "median_regime_return",
        "worst_regime_cagr",
        "median_regime_cagr",
        "worst_regime_drawdown",
        "regime_positive_rate",
        "left_tail_regime_cagr",
        "left_tail_regime_return",
        "transition_regime_hit_rate",
        "transition_regime_return",
    ]
    return _join_optional_context(summary, regime_summary, columns)


def _add_walk_forward_context(
    summary: pd.DataFrame,
    walk_forward_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "holdout_folds",
        "walk_forward_median_cagr",
        "walk_forward_worst_cagr",
        "walk_forward_positive_rate",
        "walk_forward_median_calmar",
        "walk_forward_worst_drawdown",
    ]
    return _join_optional_context(summary, walk_forward_summary, columns)


def _add_operability_context(
    summary: pd.DataFrame,
    operability_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "material_trade_days_per_year",
        "mean_days_between_material_trades",
        "median_material_turnover",
        "max_single_day_turnover",
        "operability_score",
        "operability_label",
    ]
    return _join_optional_context(summary, operability_metrics, columns)


def _add_transition_context(
    summary: pd.DataFrame,
    transition_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "average_risk_weight",
        "min_risk_weight",
        "latest_risk_weight",
        "low_risk_day_rate",
        "median_reentry_days",
        "reentry_cycles",
        "reentry_score",
        "risk_cycle_label",
    ]
    return _join_optional_context(summary, transition_metrics, columns)


def _join_optional_context(
    summary: pd.DataFrame,
    context: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    enriched = summary.copy()
    if context is None or context.empty:
        for column in columns:
            enriched[column] = float("nan")
        return enriched
    available = [column for column in columns if column in context.columns]
    enriched = enriched.join(context[available], how="left")
    for column in columns:
        if column not in enriched:
            enriched[column] = float("nan")
    return enriched


def _robustness_score(summary: pd.DataFrame) -> pd.Series:
    components = [
        _rank_column(summary, "positive_1y_window_rate") * 0.20,
        _rank_column(summary, "worst_3y_cagr") * 0.15,
        _rank_column(summary, "walk_forward_positive_rate") * 0.20,
        _rank_column(summary, "walk_forward_worst_cagr") * 0.15,
        _rank_column(summary, "worst_regime_return") * 0.12,
        _rank_column(summary, "left_tail_regime_return") * 0.10,
        _rank_column(summary, "transition_regime_hit_rate") * 0.08,
    ]
    return sum(components)


def _promotion_score(summary: pd.DataFrame) -> pd.Series:
    return (
        _rank_column(summary, "calmar") * 0.18
        + _rank_column(summary, "sharpe") * 0.14
        + _rank_column(summary, "cagr") * 0.10
        + _rank_column(summary, "max_drawdown") * 0.12
        + _rank_column(summary, "worst_3y_cagr") * 0.10
        + _rank_column(summary, "positive_1y_window_rate") * 0.08
        + _rank_column(summary, "walk_forward_positive_rate") * 0.10
        + _rank_column(summary, "walk_forward_worst_cagr") * 0.08
        + _rank_column(summary, "worst_regime_return") * 0.06
        + _rank_column(summary, "left_tail_regime_return") * 0.04
    )


def _monitoring_readiness_score(summary: pd.DataFrame) -> pd.Series:
    return (
        _rank_column(summary, "promotion_score") * 0.30
        + _rank_column(summary, "robustness_score") * 0.22
        + _rank_column(summary, "operability_score") * 0.18
        + _rank_column(summary, "reentry_score") * 0.12
        + _rank_column(summary, "walk_forward_positive_rate") * 0.10
        + _rank_column(summary, "left_tail_regime_return") * 0.08
    )


def _monitoring_readiness_label(row: pd.Series) -> str:
    score = _numeric_value(row.get("monitoring_readiness_score"))
    operability = str(row.get("operability_label", ""))
    risk_cycle = str(row.get("risk_cycle_label", ""))
    if operability == "too_twitchy" or risk_cycle == "risk_off_sticky":
        return "inspect_before_paper"
    if score >= 0.78 and row.get("promotion_decision") == "promote_candidate":
        return "paper_ready"
    if score >= 0.60:
        return "paper_candidate"
    return "research_archive"


def _benchmark_knockout_score(summary: pd.DataFrame) -> pd.Series:
    tests = [
        _positive_test(summary, "excess_cagr_vs_spy"),
        _positive_test(summary, "drawdown_improvement_vs_spy"),
        _positive_test(summary, "calmar_excess_vs_spy"),
        _positive_test(summary, "excess_cagr_vs_qqq"),
        _positive_test(summary, "drawdown_improvement_vs_qqq"),
        _positive_test(summary, "calmar_excess_vs_qqq"),
    ]
    return sum(tests) / len(tests)


def _positive_test(summary: pd.DataFrame, column: str) -> pd.Series:
    if column not in summary:
        return pd.Series(0.5, index=summary.index)
    values = pd.to_numeric(summary[column], errors="coerce")
    return (values > 0.0).astype(float).where(values.notna(), 0.5)


def _benchmark_knockout_label(row: pd.Series) -> str:
    spy_score = _mean_present(
        [
            _indicator(row.get("excess_cagr_vs_spy")),
            _indicator(row.get("drawdown_improvement_vs_spy")),
            _indicator(row.get("calmar_excess_vs_spy")),
        ]
    )
    qqq_score = _mean_present(
        [
            _indicator(row.get("excess_cagr_vs_qqq")),
            _indicator(row.get("drawdown_improvement_vs_qqq")),
            _indicator(row.get("calmar_excess_vs_qqq")),
        ]
    )
    if spy_score >= 1.0 and qqq_score >= 1.0:
        return "beats_spy_and_qqq"
    if spy_score >= 2.0 / 3.0 and qqq_score >= 1.0 / 3.0:
        return "beats_spy_mixed_qqq"
    if spy_score >= 2.0 / 3.0:
        return "beats_spy_only"
    if max(spy_score, qqq_score) >= 1.0 / 3.0:
        return "mixed_benchmark"
    return "fails_index_bar"


def _confidence_score(summary: pd.DataFrame) -> pd.Series:
    return (
        _rank_column(summary, "promotion_score") * 0.22
        + _rank_column(summary, "robustness_score") * 0.18
        + _rank_column(summary, "monitoring_readiness_score") * 0.16
        + _rank_column(summary, "benchmark_knockout_score") * 0.16
        + _rank_column(summary, "walk_forward_positive_rate") * 0.12
        + _rank_column(summary, "left_tail_regime_return") * 0.10
        + _rank_column(summary, "operability_score") * 0.06
    )


def _confidence_label(row: pd.Series) -> str:
    blockers = _deployment_blockers_list(row)
    score = _numeric_value(row.get("confidence_score"))
    if score >= 0.80 and not blockers:
        return "paper_eligible"
    if score >= 0.68 and len(blockers) <= 1:
        return "paper_watchlist"
    if score >= 0.55 and len(blockers) <= 2:
        return "needs_specific_fix"
    return "research_only"


def _deployment_blockers(row: pd.Series) -> str:
    blockers = _deployment_blockers_list(row)
    return "; ".join(blockers) if blockers else "none"


def _deployment_blockers_list(row: pd.Series) -> list[str]:
    blockers: list[str] = []
    if str(row.get("promotion_decision", "")) not in {"promote_candidate", "evolve_next_iteration"}:
        blockers.append("weak_promotion")
    if str(row.get("benchmark_knockout_label", "")) in {"fails_index_bar", "mixed_benchmark"}:
        blockers.append("benchmark_gap")
    if str(row.get("monitoring_readiness_label", "")) in {"inspect_before_paper", "research_archive"}:
        blockers.append("readiness_gap")
    if str(row.get("operability_label", "")) == "too_twitchy":
        blockers.append("too_twitchy")
    if str(row.get("risk_cycle_label", "")) == "risk_off_sticky":
        blockers.append("sticky_risk_off")
    if _numeric_value(row.get("walk_forward_positive_rate"), default=1.0) < 0.65:
        blockers.append("walk_forward_gap")
    if _numeric_value(row.get("left_tail_regime_return"), default=0.0) < -0.15:
        blockers.append("left_tail_gap")
    if _numeric_value(row.get("max_drawdown"), default=0.0) < -0.25:
        blockers.append("drawdown_gap")
    return blockers


def _indicator(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return 1.0 if numeric > 0.0 else 0.0


def _mean_present(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    if not present:
        return 0.5
    return float(sum(present) / len(present))


def _rank_column(summary: pd.DataFrame, column: str) -> pd.Series:
    if column not in summary:
        return pd.Series(0.5, index=summary.index)
    values = pd.to_numeric(summary[column], errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=summary.index)
    return values.rank(pct=True).fillna(0.5)


def _promotion_decision(row: pd.Series) -> str:
    if row["max_drawdown"] <= -0.35:
        return "reject_left_tail"
    if _finite(row.get("left_tail_regime_return")) and row["left_tail_regime_return"] < -0.20:
        return "reject_regime_fragility"
    if _finite(row.get("worst_regime_return")) and row["worst_regime_return"] < -0.25:
        return "reject_regime_fragility"
    if row["worst_3y_cagr"] < -0.05:
        return "reject_regime_fragility"
    if _finite(row.get("walk_forward_positive_rate")) and row["walk_forward_positive_rate"] < 0.45:
        return "reject_walk_forward_fragility"
    if row["promotion_score"] >= 0.75 and row["calmar"] >= 0.45 and row["robustness_score"] >= 0.55:
        return "promote_candidate"
    if row["promotion_score"] >= 0.55:
        return "evolve_next_iteration"
    return "reject_or_hold_for_reference"


def _finite(value: object) -> bool:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return False
    return numeric == numeric


def _write_experiment_outputs(
    iteration: int,
    candidates: tuple[ExperimentCandidate, ...],
    scorecard: pd.DataFrame,
    metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    walk_forward_folds: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    operability_metrics: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir) / f"iteration_{iteration:02d}"
    output.mkdir(parents=True, exist_ok=True)
    _write_candidate_manifest(candidates, output / "candidates.csv")
    scorecard.to_csv(output / "scorecard.csv")
    metrics.sort_values("calmar", ascending=False).to_csv(output / "metrics.csv")
    window_summary.to_csv(output / "window_summary.csv")
    regime_metrics.to_csv(output / "regime_metrics.csv", index=False)
    regime_summary.to_csv(output / "regime_summary.csv")
    walk_forward_folds.to_csv(output / "walk_forward_folds.csv", index=False)
    walk_forward_summary.to_csv(output / "walk_forward_summary.csv")
    operability_metrics.to_csv(output / "operability_metrics.csv")
    transition_metrics.to_csv(output / "transition_metrics.csv")
    sanity_impact = _decision_sanity_impact_frame(scorecard)
    if not sanity_impact.empty:
        sanity_impact.to_csv(output / "decision_sanity_impact.csv", index=False)
        _decision_sanity_assessment_frame(sanity_impact).to_csv(
            output / "decision_sanity_assessment.csv",
            index=False,
        )
    _write_markdown_summary(iteration, scorecard, output / "summary.md")


def _decision_sanity_impact_frame(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty or "decision_sanity" not in scorecard or "parent" not in scorecard:
        return pd.DataFrame()
    frame = scorecard.reset_index().rename(columns={"index": "strategy"})
    if "strategy" not in frame:
        return pd.DataFrame()
    indexed = frame.set_index("strategy", drop=False)
    capped_rows = frame[frame["decision_sanity"].fillna("").astype(str).str.len() > 0]
    rows = []
    metrics_to_compare = [
        "promotion_score",
        "robustness_score",
        "cagr",
        "sharpe",
        "max_drawdown",
        "calmar",
        "average_turnover",
        "worst_3y_cagr",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
    ]
    for _, capped in capped_rows.iterrows():
        parent = str(capped.get("parent", ""))
        if parent not in indexed.index:
            continue
        raw = indexed.loc[parent]
        row = {
            "raw_strategy": parent,
            "capped_strategy": capped["strategy"],
            "family": capped.get("family", ""),
            "decision_sanity": capped.get("decision_sanity", ""),
            "raw_promotion_decision": raw.get("promotion_decision", ""),
            "capped_promotion_decision": capped.get("promotion_decision", ""),
        }
        for metric in metrics_to_compare:
            if metric not in capped or metric not in raw:
                continue
            capped_value = pd.to_numeric(pd.Series([capped.get(metric)]), errors="coerce").iloc[0]
            raw_value = pd.to_numeric(pd.Series([raw.get(metric)]), errors="coerce").iloc[0]
            row[f"raw_{metric}"] = raw_value
            row[f"capped_{metric}"] = capped_value
            row[f"delta_{metric}"] = capped_value - raw_value
        rows.append(row)
    return pd.DataFrame(rows)


def _decision_sanity_assessment_frame(impact: pd.DataFrame) -> pd.DataFrame:
    if impact.empty:
        return pd.DataFrame()
    frame = impact.copy()
    for column in [
        "delta_promotion_score",
        "delta_cagr",
        "delta_max_drawdown",
        "delta_calmar",
        "delta_average_turnover",
        "delta_walk_forward_positive_rate",
        "delta_left_tail_regime_return",
    ]:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    assessment = (
        frame.groupby("decision_sanity", as_index=False, dropna=False)
        .agg(
            pairs=("capped_strategy", "count"),
            mean_delta_promotion_score=("delta_promotion_score", "mean"),
            mean_delta_cagr=("delta_cagr", "mean"),
            mean_delta_max_drawdown=("delta_max_drawdown", "mean"),
            mean_delta_calmar=("delta_calmar", "mean"),
            mean_delta_turnover=("delta_average_turnover", "mean"),
            mean_delta_walk_forward_positive_rate=("delta_walk_forward_positive_rate", "mean"),
            mean_delta_left_tail_regime_return=("delta_left_tail_regime_return", "mean"),
            promotion_win_rate=("delta_promotion_score", lambda values: float((values > 0).mean())),
            drawdown_win_rate=("delta_max_drawdown", lambda values: float((values > 0).mean())),
            calmar_win_rate=("delta_calmar", lambda values: float((values > 0).mean())),
        )
        .sort_values(
            ["mean_delta_promotion_score", "mean_delta_max_drawdown"],
            ascending=False,
        )
    )
    assessment["adoption_read"] = assessment.apply(_decision_sanity_adoption_read, axis=1)
    return assessment


def _decision_sanity_adoption_read(row: pd.Series) -> str:
    promotion_delta = _numeric_value(row.get("mean_delta_promotion_score"))
    drawdown_win_rate = _numeric_value(row.get("drawdown_win_rate"))
    calmar_delta = _numeric_value(row.get("mean_delta_calmar"))
    if promotion_delta > 0.0 and drawdown_win_rate >= 0.50 and calmar_delta >= 0.0:
        return "promote_for_monitoring"
    if promotion_delta > -0.03 and drawdown_win_rate >= 0.50:
        return "mixed_keep_testing"
    return "tune_or_reject"


def _numeric_value(value: object, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _write_candidate_manifest(
    candidates: tuple[ExperimentCandidate, ...],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "strategy": candidate.name,
                "display_name": strategy_display_name(
                    candidate.name,
                    family=candidate.family,
                    phase=candidate.phase,
                ),
                "phase": candidate.phase,
                "family": candidate.family,
                "role": candidate.role,
                "parent": candidate.parent or "",
                "hypothesis": candidate.hypothesis,
                "scenario_sizing": (
                    candidate.scenario_sizing.profile if candidate.scenario_sizing else ""
                ),
                "future_state_model": (
                    _future_state_label(candidate.future_state_model)
                    if candidate.future_state_model
                    else ""
                ),
                "strategy_drawdown_model": (
                    _strategy_drawdown_label(candidate.strategy_drawdown_model)
                    if candidate.strategy_drawdown_model
                    else ""
                ),
                "scenario_sizing_json": (
                    json.dumps(asdict(candidate.scenario_sizing), sort_keys=True)
                    if candidate.scenario_sizing
                    else ""
                ),
                "future_state_model_json": (
                    json.dumps(asdict(candidate.future_state_model), sort_keys=True)
                    if candidate.future_state_model
                    else ""
                ),
                "strategy_drawdown_model_json": (
                    json.dumps(asdict(candidate.strategy_drawdown_model), sort_keys=True)
                    if candidate.strategy_drawdown_model
                    else ""
                ),
                "decision_sanity": (
                    candidate.decision_sanity.profile if candidate.decision_sanity else ""
                ),
                "decision_sanity_json": (
                    json.dumps(asdict(candidate.decision_sanity), sort_keys=True)
                    if candidate.decision_sanity
                    else ""
                ),
                "strategy_json": json.dumps(
                    candidate.strategy.model_dump(mode="json"),
                    sort_keys=True,
                ),
            }
            for candidate in candidates
        ]
    )
    frame.to_csv(output_path, index=False)


def _load_previous_scorecards(output_dir: str | Path, iteration: int) -> pd.DataFrame:
    frames = []
    for scorecard_path in _previous_iteration_files(output_dir, iteration, "scorecard.csv"):
        frame = pd.read_csv(scorecard_path)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        frame.insert(0, "iteration", _iteration_from_path(scorecard_path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_previous_candidates(output_dir: str | Path, iteration: int) -> pd.DataFrame:
    frames = []
    for candidate_path in _previous_iteration_files(output_dir, iteration, "candidates.csv"):
        frame = pd.read_csv(candidate_path)
        frame.insert(0, "iteration", _iteration_from_path(candidate_path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _previous_iteration_files(
    output_dir: str | Path,
    iteration: int,
    filename: str,
) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    paths = []
    for path in sorted(root.glob(f"iteration_*/{filename}")):
        path_iteration = _iteration_from_path(path)
        if 0 < path_iteration < iteration:
            paths.append(path)
    return paths


def _iteration_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def _write_markdown_summary(iteration: int, scorecard: pd.DataFrame, output_path: Path) -> None:
    display = scorecard.copy()
    percent_columns = [
        "cagr",
        "max_drawdown",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "average_turnover",
        "worst_1y_cagr",
        "worst_3y_cagr",
        "worst_5y_cagr",
        "positive_1y_window_rate",
        "walk_forward_median_cagr",
        "walk_forward_worst_cagr",
        "walk_forward_positive_rate",
        "worst_regime_return",
        "worst_regime_cagr",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "transition_regime_hit_rate",
        "transition_regime_return",
        "regime_positive_rate",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in [
        "promotion_score",
        "robustness_score",
        "sharpe",
        "sortino",
        "calmar",
        "walk_forward_median_calmar",
    ]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    markdown = "\n".join(
        [
            f"# Experiment Iteration {iteration:02d}",
            "",
            "Promotion decisions are research triage, not live-trading approval.",
            "",
            _markdown_table(display.reset_index()),
            "",
        ]
    )
    output_path.write_text(markdown, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)
