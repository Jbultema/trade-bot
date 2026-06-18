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
from trade_bot.features.valuation import (
    normalized_discount,
    relative_repair_score,
    rolling_peak_discount,
    trend_discount,
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
    if strategy.type == "dip_reentry":
        return dip_reentry_weights(prices, strategy)
    if strategy.type == "dip_reentry_overlay":
        return dip_reentry_overlay_weights(prices, strategy)
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


def dip_reentry_weights(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    """Meter risk back in after large discounts only when repair signals confirm."""
    risk_tickers = [ticker for ticker in strategy.tickers if ticker != strategy.defensive_ticker]
    _validate_tickers(prices, risk_tickers)
    if strategy.defensive_ticker:
        _validate_tickers(prices, [strategy.defensive_ticker])
    if not risk_tickers:
        raise ValueError("dip_reentry strategies require at least one risk ticker.")

    filled = prices.ffill().sort_index()
    risk_prices = filled[risk_tickers]
    asset_returns = daily_returns(risk_prices)
    basket_returns = asset_returns.mean(axis=1).fillna(0.0)
    basket_equity = (1.0 + basket_returns).cumprod()
    basket_frame = pd.DataFrame({"basket": basket_equity}, index=filled.index)
    basket_discount = rolling_peak_discount(basket_frame, strategy.dip_lookback_days)["basket"]
    discount_score = normalized_discount(
        basket_discount,
        trigger_drawdown=strategy.dip_trigger_drawdown,
        deep_drawdown=strategy.dip_deep_drawdown,
    )
    deep_score = normalized_discount(
        basket_discount,
        trigger_drawdown=strategy.dip_deep_drawdown,
        deep_drawdown=min(strategy.dip_deep_drawdown * 1.60, -0.35),
    )

    recovery_return = basket_equity.pct_change(strategy.dip_recovery_days, fill_method=None)
    short_return = basket_equity.pct_change(strategy.dip_confirmation_days, fill_method=None)
    recovery_score = (recovery_return / max(strategy.dip_min_recovery_return, 1e-6)).clip(
        lower=0.0,
        upper=1.0,
    )
    short_repair_score = (short_return / 0.015).clip(lower=0.0, upper=1.0)
    basket_vol = realized_volatility(basket_returns, strategy.volatility_lookback_days)
    volatility_score = (1.0 - basket_vol / strategy.dip_volatility_ceiling).clip(
        lower=0.0,
        upper=1.0,
    )
    credit_score = (
        relative_repair_score(filled, "HYG", "LQD", strategy.dip_recovery_days)
        if strategy.dip_credit_confirmation
        else pd.Series(1.0, index=filled.index)
    )
    breadth_score = (
        relative_repair_score(filled, "RSP", "SPY", strategy.dip_recovery_days)
        if strategy.dip_breadth_confirmation
        else pd.Series(1.0, index=filled.index)
    )
    confirmation = pd.concat(
        [recovery_score, short_repair_score, volatility_score, credit_score, breadth_score],
        axis=1,
    ).mean(axis=1).fillna(0.0)

    risk_budget = (
        strategy.dip_starter_weight * discount_score
        + strategy.dip_step_weight * discount_score * confirmation
        + strategy.dip_step_weight * deep_score * confirmation
    ).clip(lower=0.0, upper=strategy.dip_max_risk_weight)
    falling_knife = (
        (short_return < -0.020)
        | (recovery_return < -abs(strategy.dip_min_recovery_return))
        | (basket_vol > strategy.dip_volatility_ceiling * 1.25)
    ).fillna(False)
    if strategy.dip_credit_confirmation:
        falling_knife = falling_knife | (credit_score < 0.20)
    risk_budget = risk_budget.mask(falling_knife, risk_budget * 0.15).fillna(0.0)

    asset_discount = rolling_peak_discount(risk_prices, strategy.dip_lookback_days)
    asset_discount_score = normalized_discount(
        asset_discount,
        trigger_drawdown=strategy.dip_trigger_drawdown,
        deep_drawdown=strategy.dip_deep_drawdown,
    )
    asset_recovery = risk_prices.pct_change(strategy.dip_recovery_days, fill_method=None)
    asset_short = risk_prices.pct_change(strategy.dip_confirmation_days, fill_method=None)
    asset_vol = realized_volatility(asset_returns, strategy.volatility_lookback_days)
    trend_window = strategy.trend_filter_days or max(63, strategy.dip_recovery_days * 3)
    trend_repair = ((trend_discount(risk_prices, trend_window) + 0.05) / 0.10).clip(
        lower=0.0,
        upper=1.0,
    )
    recovery_rank = (asset_recovery / max(strategy.dip_min_recovery_return, 1e-6)).clip(
        lower=0.0,
        upper=1.0,
    )
    short_rank = (asset_short / 0.015).clip(lower=0.0, upper=1.0)
    volatility_penalty = (asset_vol / strategy.dip_volatility_ceiling).clip(lower=0.0, upper=1.0)
    ranking_values = (
        0.42 * asset_discount_score
        + 0.28 * recovery_rank
        + 0.18 * short_rank
        + 0.12 * trend_repair
        - 0.18 * volatility_penalty
    ).fillna(0.0)
    selected = (
        (asset_discount_score > 0.0)
        & (asset_recovery > strategy.dip_min_recovery_return * 0.25)
        & (asset_short > -0.015)
        & (asset_vol < strategy.dip_volatility_ceiling * 1.50)
    )
    ranks = ranking_values.where(selected).rank(axis=1, ascending=False, method="first")
    selected = ranks <= min(strategy.top_n, len(risk_tickers))
    selected_weights = _raw_selected_weights(
        selected.fillna(False),
        weighting=strategy.weighting,
        momentum=asset_recovery.fillna(0.0),
        ranking_values=ranking_values.clip(lower=0.0),
        volatility=asset_vol,
    )

    weights = _empty_weights(filled)
    weights.loc[:, risk_tickers] = selected_weights.mul(risk_budget, axis=0).fillna(0.0)
    if strategy.max_asset_weight is not None:
        weights = _cap_risk_weights(
            weights,
            risk_tickers=risk_tickers,
            defensive_ticker=strategy.defensive_ticker,
            max_asset_weight=strategy.max_asset_weight,
        )
    if strategy.defensive_ticker:
        residual = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
        weights.loc[:, strategy.defensive_ticker] = weights[strategy.defensive_ticker] + residual
    return weights.clip(lower=0.0).fillna(0.0)


def dip_reentry_overlay_weights(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    """Use dip reentry rules to replace defensive cash inside a momentum/off-ramp system."""
    if strategy.defensive_ticker is None:
        raise ValueError("dip_reentry_overlay strategies require a defensive_ticker.")

    risk_tickers = [ticker for ticker in strategy.tickers if ticker != strategy.defensive_ticker]
    base_weights = dual_momentum_weights(
        prices,
        risk_tickers,
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
    reentry_weights = dip_reentry_weights(prices, strategy)

    base_risk_weight = base_weights[risk_tickers].sum(axis=1)
    reentry_risk_weight = reentry_weights[risk_tickers].sum(axis=1)
    add_budget = (reentry_risk_weight - base_risk_weight).clip(lower=0.0)
    reentry_risk_mix = reentry_weights[risk_tickers].div(
        reentry_risk_weight.where(reentry_risk_weight > 0.0),
        axis=0,
    )

    weights = base_weights.copy()
    weights.loc[:, risk_tickers] = weights[risk_tickers].add(
        reentry_risk_mix.mul(add_budget, axis=0),
        fill_value=0.0,
    )
    if strategy.max_asset_weight is not None:
        weights = _cap_risk_weights(
            weights,
            risk_tickers=risk_tickers,
            defensive_ticker=strategy.defensive_ticker,
            max_asset_weight=strategy.max_asset_weight,
        )
    residual = (1.0 - weights[risk_tickers].sum(axis=1)).clip(lower=0.0)
    weights.loc[:, strategy.defensive_ticker] = residual
    return weights.clip(lower=0.0).fillna(0.0)


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
