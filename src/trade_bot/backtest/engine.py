from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_bot.config import DrawdownControlConfig, ExecutionConfig, VolatilityTargetConfig
from trade_bot.features.indicators import (
    bounded_forward_fill,
    daily_returns,
    realized_volatility,
    rolling_drawdown,
)


@dataclass(frozen=True)
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series


class StaleHeldPositionError(ValueError):
    """Raised when a backtest would mark an open position with an unusable price."""


def build_execution_causality_trace(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    """Trace scheduled signals into the close-to-close return intervals they affect.

    Daily close data cannot represent a next-session-open fill. A lag-one signal
    first affects the following close-to-close return, whose interval begins at
    the signal close. The trace labels that boundary approximation explicitly;
    lag two is the first convention whose modeled return interval begins strictly
    after the feature-availability close.
    """

    index = pd.DatetimeIndex(pd.to_datetime(prices.sort_index().index))
    if index.empty:
        return pd.DataFrame()
    aligned = target_weights.reindex(index).astype(float).fillna(0.0)
    scheduled = _scheduled_rebalance_dates(aligned, execution.rebalance)
    positions = {date: position for position, date in enumerate(index)}
    rows: list[dict[str, object]] = []
    for signal_date in scheduled:
        signal_position = positions[pd.Timestamp(signal_date)]
        holding_position = signal_position + execution.signal_lag_days
        if holding_position >= len(index):
            continue
        interval_start_position = max(holding_position - 1, 0)
        interval_start = index[interval_start_position]
        holding_date = index[holding_position]
        boundary_approximation = interval_start <= pd.Timestamp(signal_date)
        rows.append(
            {
                "feature_observation_date": pd.Timestamp(signal_date),
                "signal_calculation_date": pd.Timestamp(signal_date),
                "target_generation_date": pd.Timestamp(signal_date),
                "first_modeled_holding_date": holding_date,
                "modeled_return_interval_start": interval_start,
                "modeled_return_interval_end": holding_date,
                "signal_lag_sessions": execution.signal_lag_days,
                "fill_price_field": "daily_close",
                "boundary_fill_approximation": boundary_approximation,
                "causal_status": (
                    "close_boundary_approximation"
                    if boundary_approximation
                    else "strictly_after_feature_close"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    name: str,
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig,
    *,
    volatility_target: VolatilityTargetConfig | None = None,
    drawdown_control: DrawdownControlConfig | None = None,
) -> BacktestResult:
    prices = prices.sort_index()
    asset_returns = daily_returns(prices)
    target_weights = target_weights.reindex(prices.index).astype(float).fillna(0.0)
    price_available = (
        bounded_forward_fill(prices)
        .notna()
        .reindex(
            columns=target_weights.columns,
            fill_value=False,
        )
    )
    target_weights = target_weights.where(price_available, 0.0)
    target_weights = _rebalance_weights(target_weights, execution.rebalance)
    target_weights = target_weights.where(price_available, 0.0)
    target_weights = _normalize_long_only(target_weights)

    execution_weights = target_weights.shift(execution.signal_lag_days).fillna(0.0)

    if volatility_target:
        execution_weights = apply_volatility_target(
            execution_weights,
            asset_returns,
            volatility_target,
        )

    if drawdown_control:
        execution_weights = apply_drawdown_control(
            execution_weights,
            asset_returns,
            drawdown_control,
        )

    validate_held_price_availability(execution_weights, price_available)

    execution_weights = execution_weights.where(price_available, 0.0)
    execution_weights = _normalize_long_only(execution_weights)
    turnover = (
        execution_weights.diff().abs().sum(axis=1).fillna(execution_weights.abs().sum(axis=1))
    )
    transaction_costs = turnover * execution.transaction_cost_bps / 10000.0
    gross_returns = (execution_weights * asset_returns).sum(axis=1)
    net_returns = gross_returns - transaction_costs
    equity = execution.initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=net_returns.rename(name),
        gross_returns=gross_returns.rename(name),
        weights=execution_weights,
        target_weights=target_weights,
        turnover=turnover.rename(name),
        transaction_costs=transaction_costs.rename(name),
    )


def validate_held_price_availability(
    execution_weights: pd.DataFrame,
    price_available: pd.DataFrame,
) -> None:
    """Fail instead of inventing a zero-cost exit for an unpriced open holding."""

    availability = price_available.reindex(
        index=execution_weights.index,
        columns=execution_weights.columns,
        fill_value=False,
    )
    held_without_price = execution_weights.gt(1e-12) & ~availability
    if not held_without_price.any(axis=None):
        return
    first_date = held_without_price.any(axis=1).idxmax()
    held_tickers = held_without_price.columns[held_without_price.loc[first_date]].tolist()
    raise StaleHeldPositionError(
        "Cannot value held positions after the bounded price-staleness limit: "
        f"date={pd.Timestamp(first_date).date()}, tickers={held_tickers}. "
        "Supply a valid terminal price or explicitly model the loss/exit event."
    )


def apply_volatility_target(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: VolatilityTargetConfig,
) -> pd.DataFrame:
    portfolio_returns = (weights * asset_returns).sum(axis=1)
    portfolio_vol = realized_volatility(portfolio_returns, config.lookback_days)
    scale = (config.annualized_volatility / portfolio_vol).clip(upper=config.max_leverage)
    scale = scale.replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
    scale = scale.shift(1).fillna(0.0)
    return weights.mul(scale, axis=0)


def apply_drawdown_control(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: DrawdownControlConfig,
) -> pd.DataFrame:
    strategy_returns = (weights * asset_returns).sum(axis=1)
    shadow_equity = (1.0 + strategy_returns).cumprod()
    dd = rolling_drawdown(shadow_equity, config.equity_lookback_days)
    scale = pd.Series(1.0, index=weights.index)
    scale.loc[dd <= config.max_drawdown] = config.risk_multiplier
    scale = scale.shift(1).fillna(1.0)
    return weights.mul(scale, axis=0)


def _rebalance_weights(weights: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    if rebalance.lower() in {"daily", "d"}:
        return weights
    periods = weights.index.to_period(rebalance)
    last_dates = pd.Series(weights.index, index=weights.index).groupby(periods).transform("max")
    rebalanced = weights.loc[weights.index == last_dates]
    return rebalanced.reindex(weights.index).ffill().fillna(0.0)


def _scheduled_rebalance_dates(weights: pd.DataFrame, rebalance: str) -> pd.DatetimeIndex:
    if weights.empty:
        return pd.DatetimeIndex([])
    if rebalance.lower() in {"daily", "d"}:
        return pd.DatetimeIndex(weights.index)
    periods = weights.index.to_period(rebalance)
    last_dates = pd.Series(weights.index, index=weights.index).groupby(periods).transform("max")
    return pd.DatetimeIndex(weights.index[weights.index == last_dates])


def _normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    clipped = weights.clip(lower=0.0)
    row_sum = clipped.sum(axis=1)
    over_invested = row_sum > 1.0
    clipped.loc[over_invested] = clipped.loc[over_invested].div(row_sum.loc[over_invested], axis=0)
    return clipped.fillna(0.0)
