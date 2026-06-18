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
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER,
    DEFAULT_SCENARIO_MAX_MULTIPLIER,
    DEFAULT_SCENARIO_MIN_MULTIPLIER,
    DEFAULT_SCENARIO_RISK_ON_MULTIPLIER,
    DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS,
    DEFAULT_SCENARIO_STRESS_MULTIPLIER,
    DEFAULT_SCENARIO_TRANSITION_MULTIPLIER,
)
from trade_bot.strategies.momentum import build_strategy_weights


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    hypothesis: str
    role: str
    strategy: StrategyConfig
    scenario_sizing: ScenarioSizingConfig | None = None
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
        target_weights = build_strategy_weights(candidate_prices, candidate.strategy)
        if candidate.scenario_sizing is not None:
            target_weights = apply_scenario_position_sizing(
                target_weights,
                candidate_prices,
                candidate.scenario_sizing,
                defensive_ticker=candidate.strategy.defensive_ticker,
            )
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
    benchmark_metrics = _benchmark_metrics(prices, config.execution)
    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        regime_summary=regime_summary,
        walk_forward_summary=walk_forward_summary,
        benchmark_metrics=benchmark_metrics,
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
    )


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
) -> pd.DataFrame:
    candidate_meta = pd.DataFrame(
        [
            {
                "strategy": candidate.name,
                "phase": candidate.phase,
                "family": candidate.family,
                "role": candidate.role,
                "parent": candidate.parent or "",
                "scenario_sizing": (
                    candidate.scenario_sizing.profile if candidate.scenario_sizing else ""
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
    summary["robustness_score"] = _robustness_score(summary)
    summary["promotion_score"] = _promotion_score(summary)
    summary["promotion_decision"] = summary.apply(_promotion_decision, axis=1)
    columns = [
        "phase",
        "family",
        "role",
        "parent",
        "scenario_sizing",
        "promotion_decision",
        "promotion_score",
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
    _write_markdown_summary(iteration, scorecard, output / "summary.md")


def _write_candidate_manifest(
    candidates: tuple[ExperimentCandidate, ...],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "strategy": candidate.name,
                "phase": candidate.phase,
                "family": candidate.family,
                "role": candidate.role,
                "parent": candidate.parent or "",
                "hypothesis": candidate.hypothesis,
                "scenario_sizing": (
                    candidate.scenario_sizing.profile if candidate.scenario_sizing else ""
                ),
                "scenario_sizing_json": (
                    json.dumps(asdict(candidate.scenario_sizing), sort_keys=True)
                    if candidate.scenario_sizing
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
