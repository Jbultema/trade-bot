from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_BETA_ADJUSTED_DELTA_BENCHMARK,
    DEFAULT_BETA_ADJUSTED_DELTA_LOOKBACK_DAYS,
    DEFAULT_OPERATING_SLEEVE_MAX_EXPOSURES,
    DEFAULT_OPERATING_SLEEVE_TICKERS,
    DEFAULT_TACTICAL_MATRIX_LOOKBACK_DAYS,
    DEFAULT_TACTICAL_MATRIX_TICKERS,
    DEFAULT_TACTICAL_MATRIX_TREND_DAYS,
)
from trade_bot.features.indicators import daily_returns

_MIN_WEIGHT = 1e-8


def weights_from_position_plan(
    position_plan: pd.DataFrame,
    *,
    weight_column: str = "scenario_adjusted_weight",
) -> pd.Series:
    """Return latest target weights from a trade-decision position plan."""

    if position_plan.empty or "ticker" not in position_plan or weight_column not in position_plan:
        return pd.Series(dtype=float)
    weights = position_plan.set_index("ticker")[weight_column].astype(float)
    return _normalize_weights(weights)


def build_sleeve_exposure_table(
    weights: pd.Series | dict[str, float],
    prices: pd.DataFrame | None = None,
    *,
    benchmark: str = DEFAULT_BETA_ADJUSTED_DELTA_BENCHMARK,
    lookback_days: int = DEFAULT_BETA_ADJUSTED_DELTA_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Summarize current exposure by operating sleeve and percent of max sleeve."""

    clean_weights = _normalize_weights(pd.Series(weights, dtype=float))
    if clean_weights.empty:
        return pd.DataFrame()

    beta_delta = pd.DataFrame()
    if prices is not None and not prices.empty:
        beta_delta = build_beta_adjusted_delta_table(
            prices,
            clean_weights,
            benchmark=benchmark,
            lookback_days=lookback_days,
        )

    rows = []
    for sleeve, max_exposure in DEFAULT_OPERATING_SLEEVE_MAX_EXPOSURES.items():
        tickers = [ticker for ticker in clean_weights.index if sleeve_for_ticker(str(ticker)) == sleeve]
        sleeve_weight = float(clean_weights.reindex(tickers).fillna(0.0).sum()) if tickers else 0.0
        sleeve_delta = 0.0
        if not beta_delta.empty:
            sleeve_delta = float(
                beta_delta.loc[
                    beta_delta["sleeve"].astype(str) == sleeve,
                    "beta_adjusted_spy_delta",
                ].sum()
            )
        rows.append(
            {
                "sleeve": sleeve,
                "current_weight": sleeve_weight,
                "max_sleeve_exposure": float(max_exposure),
                "percent_of_max_sleeve": (
                    sleeve_weight / float(max_exposure) if float(max_exposure) > 0 else np.nan
                ),
                "beta_adjusted_spy_delta": sleeve_delta,
                "tickers": ", ".join(tickers),
            }
        )
    return pd.DataFrame(rows)


def build_beta_adjusted_delta_table(
    prices: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    *,
    benchmark: str = DEFAULT_BETA_ADJUSTED_DELTA_BENCHMARK,
    lookback_days: int = DEFAULT_BETA_ADJUSTED_DELTA_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Calculate each holding's contribution to beta-adjusted S&P exposure."""

    clean_weights = _normalize_weights(pd.Series(weights, dtype=float))
    if clean_weights.empty or benchmark not in prices:
        return pd.DataFrame()

    tickers = [ticker for ticker in clean_weights.index if ticker in prices]
    if not tickers:
        return pd.DataFrame()
    columns = list(dict.fromkeys([benchmark, *tickers]))
    returns = daily_returns(prices[columns].sort_index()).tail(lookback_days).dropna(how="all")
    if returns.empty or benchmark not in returns:
        return pd.DataFrame()
    benchmark_returns = returns[benchmark].fillna(0.0)
    benchmark_variance = float(benchmark_returns.var())
    rows = []
    for ticker in tickers:
        asset_returns = returns[ticker].fillna(0.0)
        if ticker == benchmark:
            beta = 1.0
        elif benchmark_variance <= 0.0 or not np.isfinite(benchmark_variance):
            beta = np.nan
        else:
            beta = float(asset_returns.cov(benchmark_returns) / benchmark_variance)
        weight = float(clean_weights[ticker])
        rows.append(
            {
                "ticker": ticker,
                "sleeve": sleeve_for_ticker(ticker),
                "weight": weight,
                "spy_beta": beta,
                "beta_adjusted_spy_delta": weight * beta if np.isfinite(beta) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("beta_adjusted_spy_delta", ascending=False)


def aggregate_beta_adjusted_spy_delta(
    prices: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    *,
    benchmark: str = DEFAULT_BETA_ADJUSTED_DELTA_BENCHMARK,
    lookback_days: int = DEFAULT_BETA_ADJUSTED_DELTA_LOOKBACK_DAYS,
) -> float:
    table = build_beta_adjusted_delta_table(
        prices,
        weights,
        benchmark=benchmark,
        lookback_days=lookback_days,
    )
    if table.empty:
        return float("nan")
    return float(pd.to_numeric(table["beta_adjusted_spy_delta"], errors="coerce").sum())


def build_tactical_matrix(
    prices: pd.DataFrame,
    *,
    current_weights: pd.Series | dict[str, float] | None = None,
    tickers: tuple[str, ...] = DEFAULT_TACTICAL_MATRIX_TICKERS,
    lookback_days: int = DEFAULT_TACTICAL_MATRIX_LOOKBACK_DAYS,
    trend_days: int = DEFAULT_TACTICAL_MATRIX_TREND_DAYS,
    risk_status: str = "",
    regime: str = "",
) -> pd.DataFrame:
    """Create a sleeve-aware tactical condition matrix for human review."""

    clean_weights = (
        _normalize_weights(pd.Series(current_weights, dtype=float))
        if current_weights is not None
        else pd.Series(dtype=float)
    )
    rows = []
    for ticker in tickers:
        if ticker not in prices:
            continue
        series = prices[ticker].ffill().dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        lookback_return = _window_return(series, lookback_days)
        short_return = _window_return(series, max(21, lookback_days // 3))
        trend_average = series.rolling(trend_days, min_periods=max(20, trend_days // 2)).mean()
        latest_trend = float(trend_average.iloc[-1]) if trend_average.notna().any() else np.nan
        above_trend = bool(np.isfinite(latest_trend) and latest >= latest_trend)
        condition = _condition_label(lookback_return, above_trend)
        sleeve = sleeve_for_ticker(ticker)
        current_weight = float(clean_weights.get(ticker, 0.0))
        rows.append(
            {
                "ticker": ticker,
                "sleeve": sleeve,
                "latest_price": latest,
                "condition": condition,
                "suggested_position_size": _suggested_position_size(
                    sleeve,
                    condition,
                    current_weight=current_weight,
                    risk_status=risk_status,
                ),
                "current_target_weight": current_weight,
                "regime": regime,
                "lookback_return": lookback_return,
                "short_return": short_return,
                "above_trend": above_trend,
                "evidence": _tactical_evidence(lookback_return, short_return, above_trend, trend_days),
            }
        )
    if not rows:
        return pd.DataFrame()
    condition_rank = {"bullish": 0, "neutral": 1, "bearish": 2}
    output = pd.DataFrame(rows)
    output["_condition_rank"] = output["condition"].map(condition_rank).fillna(3)
    return output.sort_values(["_condition_rank", "sleeve", "ticker"]).drop(columns="_condition_rank")


def sleeve_for_ticker(ticker: str) -> str:
    upper = ticker.upper()
    for sleeve in ("defensive", "gold", "crypto", "credit", "stocks"):
        members = {member.upper() for member in DEFAULT_OPERATING_SLEEVE_TICKERS[sleeve]}
        if upper in members:
            return sleeve
    return "other"


def _normalize_weights(weights: pd.Series) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)
    output = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    output = output[output.abs() > _MIN_WEIGHT]
    if output.empty:
        return pd.Series(dtype=float)
    total = float(output.sum())
    if total > 1.000001:
        output = output / total
    return output.sort_index()


def _window_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return float(series.iloc[-1] / series.iloc[0] - 1.0) if len(series) > 1 else 0.0
    return float(series.iloc[-1] / series.iloc[-days - 1] - 1.0)


def _condition_label(lookback_return: float, above_trend: bool) -> str:
    if lookback_return > 0.0 and above_trend:
        return "bullish"
    if lookback_return < -0.03 or not above_trend:
        return "bearish"
    return "neutral"


def _suggested_position_size(
    sleeve: str,
    condition: str,
    *,
    current_weight: float,
    risk_status: str,
) -> str:
    risk = risk_status.lower()
    if sleeve == "defensive":
        return "Reserve | Active" if current_weight > 0.0 or risk in {"yellow", "orange", "red"} else "Reserve | Available"
    if condition == "bearish":
        return "No New Position"
    if condition == "neutral":
        return "Long | Half Position"
    if sleeve in {"crypto", "credit"}:
        return "Long | Half Position"
    if risk in {"orange", "red"}:
        return "Long | Review Size"
    return "Long | Max Position"


def _tactical_evidence(
    lookback_return: float,
    short_return: float,
    above_trend: bool,
    trend_days: int,
) -> str:
    trend_label = "above" if above_trend else "below"
    return (
        f"{lookback_return:.1%} lookback return; {short_return:.1%} short return; "
        f"price is {trend_label} {trend_days}d trend."
    )
