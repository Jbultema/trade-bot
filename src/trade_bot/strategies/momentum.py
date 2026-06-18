from __future__ import annotations

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.DEFAULT import (
    DEFAULT_MAX_ASSET_WEIGHT,
    DEFAULT_RANKING_METRIC,
    DEFAULT_TREND_FILTER_DAYS,
    DEFAULT_VOLATILITY_LOOKBACK_DAYS,
    DEFAULT_WEIGHTING,
)
from trade_bot.features.indicators import (
    daily_returns,
    lookback_returns,
    moving_average,
    realized_volatility,
)


def build_strategy_weights(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    if strategy.type == "buy_hold":
        return buy_hold_weights(prices, strategy.tickers)
    if strategy.type == "fixed_allocation":
        if strategy.allocation_weights is None:
            raise ValueError("fixed_allocation strategies require allocation_weights.")
        return fixed_allocation_weights(prices, strategy.allocation_weights)
    if strategy.type == "absolute_momentum":
        return absolute_momentum_weights(
            prices,
            strategy.tickers,
            moving_average_days=strategy.moving_average_days,
            defensive_ticker=strategy.defensive_ticker,
        )
    if strategy.type == "relative_momentum":
        return relative_momentum_weights(
            prices,
            strategy.tickers,
            lookback_days=strategy.lookback_days,
            skip_days=strategy.skip_days,
            top_n=strategy.top_n,
            defensive_ticker=strategy.defensive_ticker,
            ranking_metric=strategy.ranking_metric,
            weighting=strategy.weighting,
            volatility_lookback_days=strategy.volatility_lookback_days,
            trend_filter_days=strategy.trend_filter_days,
            max_asset_weight=strategy.max_asset_weight,
        )
    if strategy.type == "dual_momentum":
        return dual_momentum_weights(
            prices,
            strategy.tickers,
            lookback_days=strategy.lookback_days,
            skip_days=strategy.skip_days,
            top_n=strategy.top_n,
            defensive_ticker=strategy.defensive_ticker,
            min_return=strategy.min_return,
            ranking_metric=strategy.ranking_metric,
            weighting=strategy.weighting,
            volatility_lookback_days=strategy.volatility_lookback_days,
            trend_filter_days=strategy.trend_filter_days,
            max_asset_weight=strategy.max_asset_weight,
        )
    raise ValueError(f"Unsupported strategy type: {strategy.type}")


def buy_hold_weights(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    _validate_tickers(prices, tickers)
    weights = _empty_weights(prices)
    weight = 1.0 / len(tickers)
    weights.loc[:, tickers] = weight
    return weights


def fixed_allocation_weights(
    prices: pd.DataFrame,
    allocation_weights: dict[str, float],
) -> pd.DataFrame:
    if not allocation_weights:
        raise ValueError("fixed_allocation strategies require at least one asset weight.")
    tickers = list(allocation_weights)
    _validate_tickers(prices, tickers)

    cleaned_weights = {ticker: float(weight) for ticker, weight in allocation_weights.items()}
    negative_weights = [ticker for ticker, weight in cleaned_weights.items() if weight < 0.0]
    if negative_weights:
        raise ValueError(f"fixed_allocation weights must be long-only: {negative_weights}")

    total_weight = sum(cleaned_weights.values())
    if total_weight <= 0.0:
        raise ValueError("fixed_allocation total weight must be positive.")
    if total_weight > 1.000001:
        raise ValueError(f"fixed_allocation total weight must be <= 1.0, got {total_weight:.4f}")

    weights = _empty_weights(prices)
    for ticker, weight in cleaned_weights.items():
        weights.loc[:, ticker] = weight
    return weights


def absolute_momentum_weights(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    moving_average_days: int,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    _validate_tickers(prices, tickers)
    if defensive_ticker:
        _validate_tickers(prices, [defensive_ticker])

    weights = _empty_weights(prices)
    ma = moving_average(prices[tickers], moving_average_days)
    risk_on = prices[tickers] > ma
    active_count = risk_on.sum(axis=1)

    for ticker in tickers:
        weights.loc[risk_on[ticker], ticker] = 1.0
    weights.loc[:, tickers] = (
        weights[tickers]
        .div(active_count.where(active_count != 0), axis=0)
        .fillna(0.0)
        .astype(float)
    )

    if defensive_ticker:
        risk_off = active_count == 0
        weights.loc[risk_off, defensive_ticker] = 1.0
    return weights


def relative_momentum_weights(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    lookback_days: int,
    skip_days: int,
    top_n: int,
    defensive_ticker: str | None,
    ranking_metric: str = DEFAULT_RANKING_METRIC,
    weighting: str = DEFAULT_WEIGHTING,
    volatility_lookback_days: int = DEFAULT_VOLATILITY_LOOKBACK_DAYS,
    trend_filter_days: int | None = DEFAULT_TREND_FILTER_DAYS,
    max_asset_weight: float | None = DEFAULT_MAX_ASSET_WEIGHT,
) -> pd.DataFrame:
    _validate_tickers(prices, tickers)
    if defensive_ticker:
        _validate_tickers(prices, [defensive_ticker])

    momentum = lookback_returns(prices[tickers], lookback_days, skip_days)
    volatility = realized_volatility(daily_returns(prices[tickers]), volatility_lookback_days)
    ranking_values = _ranking_values(
        prices[tickers],
        momentum,
        volatility,
        ranking_metric=ranking_metric,
        trend_filter_days=trend_filter_days,
    )
    ranks = ranking_values.rank(axis=1, ascending=False, method="first")
    selected = ranks <= min(top_n, len(tickers))
    selected = _apply_trend_filter(prices[tickers], selected, trend_filter_days)
    weights = _selected_to_weights(
        prices,
        selected,
        weighting=weighting,
        momentum=momentum,
        ranking_values=ranking_values,
        volatility=volatility,
        defensive_ticker=defensive_ticker,
        max_asset_weight=max_asset_weight,
    )

    if defensive_ticker:
        no_signal = selected.sum(axis=1) == 0
        weights.loc[no_signal, defensive_ticker] = 1.0
    return weights


def dual_momentum_weights(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    lookback_days: int,
    skip_days: int,
    top_n: int,
    defensive_ticker: str | None,
    min_return: float,
    ranking_metric: str = DEFAULT_RANKING_METRIC,
    weighting: str = DEFAULT_WEIGHTING,
    volatility_lookback_days: int = DEFAULT_VOLATILITY_LOOKBACK_DAYS,
    trend_filter_days: int | None = DEFAULT_TREND_FILTER_DAYS,
    max_asset_weight: float | None = DEFAULT_MAX_ASSET_WEIGHT,
) -> pd.DataFrame:
    _validate_tickers(prices, tickers)
    if defensive_ticker:
        _validate_tickers(prices, [defensive_ticker])

    momentum = lookback_returns(prices[tickers], lookback_days, skip_days)
    volatility = realized_volatility(daily_returns(prices[tickers]), volatility_lookback_days)
    ranking_values = _ranking_values(
        prices[tickers],
        momentum,
        volatility,
        ranking_metric=ranking_metric,
        trend_filter_days=trend_filter_days,
    )
    ranks = ranking_values.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= min(top_n, len(tickers))) & (momentum > min_return)
    selected = _apply_trend_filter(prices[tickers], selected, trend_filter_days)
    weights = _selected_to_weights(
        prices,
        selected,
        weighting=weighting,
        momentum=momentum,
        ranking_values=ranking_values,
        volatility=volatility,
        defensive_ticker=defensive_ticker,
        max_asset_weight=max_asset_weight,
    )

    if defensive_ticker:
        no_signal = selected.sum(axis=1) == 0
        weights.loc[no_signal, defensive_ticker] = 1.0
    return weights


def _ranking_values(
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    ranking_metric: str,
    trend_filter_days: int | None,
) -> pd.DataFrame:
    if ranking_metric == "return":
        return momentum
    if ranking_metric == "risk_adjusted_return":
        return momentum.div(volatility.where(volatility > 0.0))
    if ranking_metric == "return_trend_quality":
        trend_quality = _trend_quality(prices, trend_filter_days)
        return momentum + (0.25 * trend_quality)
    raise ValueError(f"Unsupported ranking metric: {ranking_metric}")


def _trend_quality(prices: pd.DataFrame, trend_filter_days: int | None) -> pd.DataFrame:
    window = trend_filter_days or 200
    ma = moving_average(prices, window)
    return prices / ma - 1.0


def _apply_trend_filter(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    trend_filter_days: int | None,
) -> pd.DataFrame:
    if trend_filter_days is None:
        return selected
    return selected & (prices > moving_average(prices, trend_filter_days))


def _selected_to_weights(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    weighting: str,
    momentum: pd.DataFrame,
    ranking_values: pd.DataFrame,
    volatility: pd.DataFrame,
    defensive_ticker: str | None,
    max_asset_weight: float | None,
) -> pd.DataFrame:
    weights = _empty_weights(prices)
    selected_weights = _raw_selected_weights(
        selected,
        weighting=weighting,
        momentum=momentum,
        ranking_values=ranking_values,
        volatility=volatility,
    )
    weights.loc[:, selected_weights.columns] = selected_weights
    if max_asset_weight is not None:
        weights = _cap_risk_weights(
            weights,
            risk_tickers=list(selected.columns),
            defensive_ticker=defensive_ticker,
            max_asset_weight=max_asset_weight,
        )
    return weights


def _raw_selected_weights(
    selected: pd.DataFrame,
    *,
    weighting: str,
    momentum: pd.DataFrame,
    ranking_values: pd.DataFrame,
    volatility: pd.DataFrame,
) -> pd.DataFrame:
    if weighting == "equal":
        scores = selected.astype(float)
    elif weighting == "inverse_volatility":
        scores = selected.astype(float).mul((1.0 / volatility.where(volatility > 0.0)), axis=0)
    elif weighting == "momentum_score":
        scores = selected.astype(float).mul(momentum.clip(lower=0.0), axis=0)
    elif weighting == "risk_adjusted_score":
        scores = selected.astype(float).mul(ranking_values.clip(lower=0.0), axis=0)
    else:
        raise ValueError(f"Unsupported weighting: {weighting}")

    score_sum = scores.sum(axis=1)
    equal_weights = selected.astype(float).div(
        selected.sum(axis=1).where(selected.sum(axis=1) != 0),
        axis=0,
    )
    score_weights = scores.div(score_sum.where(score_sum != 0), axis=0)
    return score_weights.fillna(equal_weights).fillna(0.0)


def _cap_risk_weights(
    weights: pd.DataFrame,
    *,
    risk_tickers: list[str],
    defensive_ticker: str | None,
    max_asset_weight: float,
) -> pd.DataFrame:
    capped = weights.copy()
    capped.loc[:, risk_tickers] = capped[risk_tickers].clip(upper=max_asset_weight)
    if defensive_ticker and defensive_ticker in capped.columns:
        residual = (1.0 - capped.sum(axis=1)).clip(lower=0.0)
        capped.loc[:, defensive_ticker] = capped[defensive_ticker] + residual
    return capped


def _empty_weights(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


def _validate_tickers(prices: pd.DataFrame, tickers: list[str]) -> None:
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing price columns: {missing}")
