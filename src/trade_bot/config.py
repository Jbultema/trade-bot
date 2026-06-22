from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from trade_bot.DEFAULTS import (
    DEFAULT_CYCLE_MAX_STEP_CHANGE,
    DEFAULT_CYCLE_MIN_HOLD_DAYS,
    DEFAULT_CYCLE_MIN_REBALANCE_CHANGE,
    DEFAULT_CYCLE_RISK_OFF_OVERRIDE_CHANGE,
    DEFAULT_CYCLE_SATELLITE_MAX_WEIGHT,
    DEFAULT_CYCLE_SATELLITE_REENTRY_WEIGHT,
    DEFAULT_CYCLE_SATELLITE_RISK_ON_WEIGHT,
    DEFAULT_DATA_ADJUSTED,
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_DIP_BREADTH_CONFIRMATION,
    DEFAULT_DIP_CONFIRMATION_DAYS,
    DEFAULT_DIP_CREDIT_CONFIRMATION,
    DEFAULT_DIP_DEEP_DRAWDOWN,
    DEFAULT_DIP_LOOKBACK_DAYS,
    DEFAULT_DIP_MAX_RISK_WEIGHT,
    DEFAULT_DIP_MIN_RECOVERY_RETURN,
    DEFAULT_DIP_RECOVERY_DAYS,
    DEFAULT_DIP_STARTER_WEIGHT,
    DEFAULT_DIP_STEP_WEIGHT,
    DEFAULT_DIP_TRIGGER_DRAWDOWN,
    DEFAULT_DIP_VOLATILITY_CEILING,
    DEFAULT_DRAWDOWN_EQUITY_LOOKBACK_DAYS,
    DEFAULT_DRAWDOWN_MAX_DRAWDOWN,
    DEFAULT_DRAWDOWN_RISK_MULTIPLIER,
    DEFAULT_EXCLUDED_TICKERS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_ASSET_WEIGHT,
    DEFAULT_MIN_RETURN,
    DEFAULT_MOMENTUM_LOOKBACK_DAYS,
    DEFAULT_MOMENTUM_SKIP_DAYS,
    DEFAULT_MOVING_AVERAGE_DAYS,
    DEFAULT_RANKING_METRIC,
    DEFAULT_REBALANCE,
    DEFAULT_SIGNAL_LAG_DAYS,
    DEFAULT_TOP_N,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_TREND_FILTER_DAYS,
    DEFAULT_VOL_TARGET_ANNUALIZED_VOLATILITY,
    DEFAULT_VOL_TARGET_LOOKBACK_DAYS,
    DEFAULT_VOL_TARGET_MAX_LEVERAGE,
    DEFAULT_VOLATILITY_LOOKBACK_DAYS,
    DEFAULT_WEIGHTING,
)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str | None = None
    cache_dir: str = DEFAULT_DATA_CACHE_DIR
    adjusted: bool = DEFAULT_DATA_ADJUSTED


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    rebalance: str = DEFAULT_REBALANCE
    signal_lag_days: int = Field(default=DEFAULT_SIGNAL_LAG_DAYS, ge=1)


class VolatilityTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annualized_volatility: float = Field(default=DEFAULT_VOL_TARGET_ANNUALIZED_VOLATILITY, gt=0)
    lookback_days: int = Field(default=DEFAULT_VOL_TARGET_LOOKBACK_DAYS, gt=1)
    max_leverage: float = Field(default=DEFAULT_VOL_TARGET_MAX_LEVERAGE, gt=0)


class DrawdownControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equity_lookback_days: int = Field(default=DEFAULT_DRAWDOWN_EQUITY_LOOKBACK_DAYS, gt=1)
    max_drawdown: float = Field(default=DEFAULT_DRAWDOWN_MAX_DRAWDOWN, lt=0)
    risk_multiplier: float = Field(default=DEFAULT_DRAWDOWN_RISK_MULTIPLIER, ge=0, le=1)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "buy_hold",
        "fixed_allocation",
        "absolute_momentum",
        "relative_momentum",
        "dual_momentum",
        "dip_reentry",
        "dip_reentry_overlay",
        "ai_risk_cycle_overlay",
        "sector_regime_rotation",
    ]
    tickers: list[str]
    satellite_tickers: list[str] = Field(default_factory=list)
    allocation_weights: dict[str, float] | None = None
    moving_average_days: int = Field(default=DEFAULT_MOVING_AVERAGE_DAYS, gt=1)
    lookback_days: int = Field(default=DEFAULT_MOMENTUM_LOOKBACK_DAYS, gt=1)
    skip_days: int = Field(default=DEFAULT_MOMENTUM_SKIP_DAYS, ge=0)
    top_n: int = Field(default=DEFAULT_TOP_N, gt=0)
    defensive_ticker: str | None = None
    min_return: float = DEFAULT_MIN_RETURN
    ranking_metric: Literal["return", "risk_adjusted_return", "return_trend_quality"] = (
        DEFAULT_RANKING_METRIC
    )
    weighting: Literal["equal", "inverse_volatility", "momentum_score", "risk_adjusted_score"] = (
        DEFAULT_WEIGHTING
    )
    volatility_lookback_days: int = Field(default=DEFAULT_VOLATILITY_LOOKBACK_DAYS, gt=1)
    trend_filter_days: int | None = Field(default=DEFAULT_TREND_FILTER_DAYS, gt=1)
    max_asset_weight: float | None = Field(default=DEFAULT_MAX_ASSET_WEIGHT, gt=0, le=1)
    volatility_target: VolatilityTargetConfig | None = None
    drawdown_control: DrawdownControlConfig | None = None
    dip_lookback_days: int = Field(default=DEFAULT_DIP_LOOKBACK_DAYS, gt=20)
    dip_trigger_drawdown: float = Field(default=DEFAULT_DIP_TRIGGER_DRAWDOWN, lt=0)
    dip_deep_drawdown: float = Field(default=DEFAULT_DIP_DEEP_DRAWDOWN, lt=0)
    dip_recovery_days: int = Field(default=DEFAULT_DIP_RECOVERY_DAYS, gt=1)
    dip_confirmation_days: int = Field(default=DEFAULT_DIP_CONFIRMATION_DAYS, gt=0)
    dip_min_recovery_return: float = DEFAULT_DIP_MIN_RECOVERY_RETURN
    dip_starter_weight: float = Field(default=DEFAULT_DIP_STARTER_WEIGHT, ge=0, le=1)
    dip_step_weight: float = Field(default=DEFAULT_DIP_STEP_WEIGHT, ge=0, le=1)
    dip_max_risk_weight: float = Field(default=DEFAULT_DIP_MAX_RISK_WEIGHT, ge=0, le=1)
    dip_volatility_ceiling: float = Field(default=DEFAULT_DIP_VOLATILITY_CEILING, gt=0)
    dip_credit_confirmation: bool = DEFAULT_DIP_CREDIT_CONFIRMATION
    dip_breadth_confirmation: bool = DEFAULT_DIP_BREADTH_CONFIRMATION
    cycle_satellite_max_weight: float = Field(
        default=DEFAULT_CYCLE_SATELLITE_MAX_WEIGHT,
        ge=0,
        le=1,
    )
    cycle_satellite_risk_on_weight: float = Field(
        default=DEFAULT_CYCLE_SATELLITE_RISK_ON_WEIGHT,
        ge=0,
        le=1,
    )
    cycle_satellite_reentry_weight: float = Field(
        default=DEFAULT_CYCLE_SATELLITE_REENTRY_WEIGHT,
        ge=0,
        le=1,
    )
    cycle_min_rebalance_change: float = Field(
        default=DEFAULT_CYCLE_MIN_REBALANCE_CHANGE,
        ge=0,
        le=2,
    )
    cycle_max_step_change: float = Field(default=DEFAULT_CYCLE_MAX_STEP_CHANGE, gt=0, le=2)
    cycle_min_hold_days: int = Field(default=DEFAULT_CYCLE_MIN_HOLD_DAYS, ge=0)
    cycle_risk_off_override_change: float = Field(
        default=DEFAULT_CYCLE_RISK_OFF_OVERRIDE_CHANGE,
        ge=0,
        le=1,
    )


class BotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    execution: ExecutionConfig
    universe: dict[str, list[str]]
    strategies: dict[str, StrategyConfig]


def filter_excluded_tickers(
    tickers: Iterable[str],
    excluded_tickers: frozenset[str] = DEFAULT_EXCLUDED_TICKERS,
) -> list[str]:
    """Return tickers after applying owner-directed hard exclusions."""

    excluded = {ticker.upper() for ticker in excluded_tickers}
    return list(dict.fromkeys(ticker for ticker in tickers if ticker.upper() not in excluded))


def _filter_excluded_allocation_weights(
    allocation_weights: dict[str, float] | None,
    excluded_tickers: frozenset[str] = DEFAULT_EXCLUDED_TICKERS,
) -> dict[str, float] | None:
    if allocation_weights is None:
        return None
    allowed_weights = {
        ticker: weight
        for ticker, weight in allocation_weights.items()
        if ticker.upper() not in {excluded.upper() for excluded in excluded_tickers}
    }
    total_weight = sum(allowed_weights.values())
    if total_weight <= 0:
        return allowed_weights
    return {ticker: weight / total_weight for ticker, weight in allowed_weights.items()}


def apply_excluded_ticker_policy_to_strategy(strategy: StrategyConfig) -> StrategyConfig:
    """Strip excluded tickers from a strategy while preserving its shape."""

    excluded = {ticker.upper() for ticker in DEFAULT_EXCLUDED_TICKERS}
    defensive_ticker = strategy.defensive_ticker
    if defensive_ticker is not None and defensive_ticker.upper() in excluded:
        defensive_ticker = None
    return strategy.model_copy(
        update={
            "tickers": filter_excluded_tickers(strategy.tickers),
            "satellite_tickers": filter_excluded_tickers(strategy.satellite_tickers),
            "allocation_weights": _filter_excluded_allocation_weights(strategy.allocation_weights),
            "defensive_ticker": defensive_ticker,
        }
    )


def apply_excluded_ticker_policy(config: BotConfig) -> BotConfig:
    """Apply hard ticker exclusions to loaded config universes and strategies."""

    return config.model_copy(
        update={
            "universe": {
                group: filter_excluded_tickers(tickers)
                for group, tickers in config.universe.items()
            },
            "strategies": {
                name: apply_excluded_ticker_policy_to_strategy(strategy)
                for name, strategy in config.strategies.items()
            },
        }
    )


def load_config(path: str | Path) -> BotConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return apply_excluded_ticker_policy(BotConfig.model_validate(raw))


def configured_tickers(config: BotConfig) -> list[str]:
    tickers: set[str] = set()
    for group_tickers in config.universe.values():
        tickers.update(group_tickers)
    for strategy in config.strategies.values():
        tickers.update(strategy.tickers)
        tickers.update(strategy.satellite_tickers)
        if strategy.defensive_ticker:
            tickers.add(strategy.defensive_ticker)
    return sorted(filter_excluded_tickers(tickers))
