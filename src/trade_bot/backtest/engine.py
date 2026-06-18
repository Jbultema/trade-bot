from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_bot.config import DrawdownControlConfig, ExecutionConfig, VolatilityTargetConfig
from trade_bot.features.indicators import daily_returns, realized_volatility, rolling_drawdown


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
    target_weights = _rebalance_weights(target_weights, execution.rebalance)
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


def _normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    clipped = weights.clip(lower=0.0)
    row_sum = clipped.sum(axis=1)
    over_invested = row_sum > 1.0
    clipped.loc[over_invested] = clipped.loc[over_invested].div(row_sum.loc[over_invested], axis=0)
    return clipped.fillna(0.0)
