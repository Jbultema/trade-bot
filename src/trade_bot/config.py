from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from trade_bot.DEFAULT import (
    DEFAULT_DATA_ADJUSTED,
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_DRAWDOWN_EQUITY_LOOKBACK_DAYS,
    DEFAULT_DRAWDOWN_MAX_DRAWDOWN,
    DEFAULT_DRAWDOWN_RISK_MULTIPLIER,
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

    type: Literal["buy_hold", "absolute_momentum", "relative_momentum", "dual_momentum"]
    tickers: list[str]
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


class BotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    execution: ExecutionConfig
    universe: dict[str, list[str]]
    strategies: dict[str, StrategyConfig]


def load_config(path: str | Path) -> BotConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return BotConfig.model_validate(raw)


def configured_tickers(config: BotConfig) -> list[str]:
    tickers: set[str] = set()
    for group_tickers in config.universe.values():
        tickers.update(group_tickers)
    for strategy in config.strategies.values():
        tickers.update(strategy.tickers)
        if strategy.defensive_ticker:
            tickers.add(strategy.defensive_ticker)
    return sorted(tickers)
