from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.features.indicators import daily_returns, lookback_returns, realized_volatility


@dataclass(frozen=True)
class SignalStateReport:
    latest: pd.Series
    assets: pd.DataFrame
    backtest: pd.DataFrame
    gated_result: BacktestResult | None


def build_signal_state_report(
    *,
    result: BacktestResult,
    prices: pd.DataFrame,
    strategy: StrategyConfig | None,
    execution: ExecutionConfig,
) -> SignalStateReport:
    """Build a transparent top-down/bottom-up confirmation read for a candidate.

    This is an explanatory overlay, not a replacement for the candidate's native
    strategy logic. It asks whether current risk posture and each asset's
    volatility-adjusted momentum are aligned with the candidate's latest sizing.
    """

    if strategy is None or result is None or prices.empty or result.target_weights.empty:
        return SignalStateReport(pd.Series(dtype=object), pd.DataFrame(), pd.DataFrame(), None)

    signal_frame = build_asset_signal_states(prices=prices, strategy=strategy)
    latest_assets = latest_asset_signal_states(
        signal_frame,
        result=result,
        defensive_ticker=strategy.defensive_ticker,
    )
    gated_weights = confirmation_gated_weights(
        result.target_weights,
        signal_frame,
        defensive_ticker=strategy.defensive_ticker,
    )
    gated_result = run_backtest(
        f"{result.name}_confirmation_gated",
        prices,
        gated_weights,
        execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )
    backtest = signal_state_backtest_context(result, gated_result)
    latest = latest_signal_state_readout(latest_assets, backtest)
    return SignalStateReport(latest, latest_assets, backtest, gated_result)


def build_asset_signal_states(
    *,
    prices: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    risk_tickers = [ticker for ticker in strategy.tickers if ticker in prices.columns]
    defensive_ticker = strategy.defensive_ticker
    if not risk_tickers:
        return pd.DataFrame()

    top_down = top_down_regime_signal(prices)
    momentum = lookback_returns(
        prices[risk_tickers],
        strategy.lookback_days,
        strategy.skip_days,
    )
    volatility = realized_volatility(
        daily_returns(prices[risk_tickers]),
        strategy.volatility_lookback_days,
    )
    vol_adjusted_momentum = momentum.div(volatility.replace(0.0, pd.NA))
    ranking = _ranking_values(momentum, vol_adjusted_momentum, strategy.ranking_metric)
    ranks = ranking.rank(axis=1, ascending=False, method="first")
    trend_pass = _trend_pass(prices[risk_tickers], strategy.trend_filter_days)
    selected_cutoff = min(max(int(strategy.top_n), 1), len(risk_tickers))
    min_return = float(strategy.min_return) if strategy.type == "dual_momentum" else 0.0

    frames: list[pd.DataFrame] = []
    for ticker in risk_tickers:
        ticker_frame = pd.DataFrame(
            {
                "date": prices.index,
                "ticker": ticker,
                "top_down_score": top_down["top_down_score"].reindex(prices.index).to_numpy(),
                "top_down_signal": top_down["top_down_signal"].reindex(prices.index).to_numpy(),
                "momentum": momentum[ticker].to_numpy(),
                "realized_volatility": volatility[ticker].to_numpy(),
                "vol_adjusted_momentum": vol_adjusted_momentum[ticker].to_numpy(),
                "rank": ranks[ticker].to_numpy(),
                "trend_pass": trend_pass[ticker].to_numpy(),
            },
            index=prices.index,
        )
        frames.append(ticker_frame)
    states = pd.concat(frames, ignore_index=True, sort=False)
    states["bottom_up_signal"] = states.apply(
        lambda row: _bottom_up_signal(
            momentum=row.get("momentum"),
            rank=row.get("rank"),
            trend_pass=row.get("trend_pass"),
            selected_cutoff=selected_cutoff,
            min_return=min_return,
        ),
        axis=1,
    )
    states["confirmation_state"] = states.apply(_confirmation_state, axis=1)
    states["confirmation_scale"] = states["confirmation_state"].map(
        {
            "long_max": 1.0,
            "long_half": 0.5,
            "watch_only": 0.0,
        }
    )
    if defensive_ticker:
        states["defensive_ticker"] = defensive_ticker
    return states


def top_down_regime_signal(prices: pd.DataFrame) -> pd.DataFrame:
    components: dict[str, pd.Series] = {}
    if "SPY" in prices:
        components["spy_trend"] = (prices["SPY"].ffill().pct_change(63) > 0.0).astype(float)
    if "QQQ" in prices:
        components["qqq_trend"] = (prices["QQQ"].ffill().pct_change(63) > 0.0).astype(float)
    if {"RSP", "SPY"}.issubset(prices.columns):
        relative = prices["RSP"].ffill() / prices["SPY"].ffill()
        components["breadth_relative"] = (relative.pct_change(63) > 0.0).astype(float)
    if {"HYG", "LQD"}.issubset(prices.columns):
        credit = prices["HYG"].ffill() / prices["LQD"].ffill()
        components["credit_relative"] = (credit.pct_change(63) > 0.0).astype(float)
    if "VIXY" in prices:
        components["vol_pressure"] = (prices["VIXY"].ffill().pct_change(21) < 0.0).astype(float)

    if not components:
        score = pd.Series(0.5, index=prices.index, name="top_down_score")
    else:
        score = pd.DataFrame(components, index=prices.index).mean(axis=1).fillna(0.5)
        score = score.rename("top_down_score")
    signal = pd.Series("neutral", index=prices.index, name="top_down_signal")
    signal.loc[score >= 0.60] = "bullish"
    signal.loc[score < 0.40] = "bearish"
    return pd.concat([score, signal], axis=1)


def latest_asset_signal_states(
    signal_frame: pd.DataFrame,
    *,
    result: BacktestResult,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    if signal_frame.empty:
        return pd.DataFrame()
    latest_date = signal_frame["date"].max()
    latest = signal_frame[signal_frame["date"].eq(latest_date)].copy()
    latest_weights = result.weights.reindex(result.weights.index.sort_values()).ffill().iloc[-1]
    latest_targets = result.target_weights.reindex(result.target_weights.index.sort_values()).ffill().iloc[-1]
    latest["current_weight"] = latest["ticker"].map(latest_weights.to_dict()).fillna(0.0)
    latest["target_weight"] = latest["ticker"].map(latest_targets.to_dict()).fillna(0.0)
    latest["state_read"] = latest.apply(_asset_state_read, axis=1)
    if defensive_ticker:
        defensive_weight = float(latest_weights.get(defensive_ticker, 0.0))
        target_defensive = float(latest_targets.get(defensive_ticker, 0.0))
        latest = pd.concat(
            [
                latest,
                pd.DataFrame(
                    [
                        {
                            "date": latest_date,
                            "ticker": defensive_ticker,
                            "top_down_score": latest["top_down_score"].iloc[0],
                            "top_down_signal": latest["top_down_signal"].iloc[0],
                            "momentum": float("nan"),
                            "realized_volatility": float("nan"),
                            "vol_adjusted_momentum": float("nan"),
                            "rank": float("nan"),
                            "trend_pass": False,
                            "bottom_up_signal": "defensive",
                            "confirmation_state": "defensive_reserve",
                            "confirmation_scale": 1.0,
                            "current_weight": defensive_weight,
                            "target_weight": target_defensive,
                            "state_read": _defensive_state_read(defensive_weight),
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    return latest.sort_values(["target_weight", "current_weight"], ascending=False).reset_index(
        drop=True
    )


def confirmation_gated_weights(
    base_target_weights: pd.DataFrame,
    signal_frame: pd.DataFrame,
    *,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    if base_target_weights.empty or signal_frame.empty:
        return base_target_weights.copy()
    scale = signal_frame.pivot(index="date", columns="ticker", values="confirmation_scale")
    scale = scale.reindex(base_target_weights.index).ffill().fillna(0.0)
    gated = base_target_weights.copy().astype(float).fillna(0.0)
    risk_columns = [column for column in gated.columns if column != defensive_ticker]
    for column in risk_columns:
        if column in scale:
            gated[column] = gated[column] * scale[column].clip(lower=0.0, upper=1.0)
        else:
            gated[column] = 0.0
    if defensive_ticker:
        if defensive_ticker not in gated:
            gated[defensive_ticker] = 0.0
        residual = (1.0 - gated[risk_columns].sum(axis=1)).clip(lower=0.0)
        gated[defensive_ticker] = gated[defensive_ticker].clip(lower=0.0).combine(residual, max)
    return gated.clip(lower=0.0)


def signal_state_backtest_context(
    base_result: BacktestResult,
    gated_result: BacktestResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, result in [
        ("Native strategy", base_result),
        ("Confirmation-gated overlay", gated_result),
    ]:
        metric = calculate_metrics(
            result.name,
            result.returns,
            result.equity,
            result.turnover,
            result.transaction_costs,
        )
        defensive_weight = 1.0 - result.weights.sum(axis=1).clip(upper=1.0)
        if "BIL" in result.weights:
            defensive_weight = (defensive_weight + result.weights["BIL"]).clip(upper=1.0)
        rows.append(
            {
                "variant": label,
                "cagr": metric.cagr,
                "max_drawdown": metric.max_drawdown,
                "sharpe": metric.sharpe,
                "calmar": metric.calmar,
                "average_turnover": metric.average_turnover,
                "current_defensive_weight": float(defensive_weight.iloc[-1]),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) == 2:
        output["delta_vs_native_cagr"] = output["cagr"] - float(output.loc[0, "cagr"])
        output["delta_vs_native_drawdown"] = output["max_drawdown"] - float(
            output.loc[0, "max_drawdown"]
        )
    return output


def latest_signal_state_readout(
    assets: pd.DataFrame,
    backtest: pd.DataFrame,
) -> pd.Series:
    if assets.empty:
        return pd.Series(dtype=object)
    risk_assets = assets[assets["confirmation_state"].ne("defensive_reserve")]
    top_down_signal = str(assets["top_down_signal"].iloc[0])
    top_down_score = float(assets["top_down_score"].iloc[0])
    confirmed = int(risk_assets["confirmation_state"].eq("long_max").sum())
    partial = int(risk_assets["confirmation_state"].eq("long_half").sum())
    watch_only = int(risk_assets["confirmation_state"].eq("watch_only").sum())
    backtest_label = "not tested"
    if not backtest.empty and len(backtest) >= 2:
        cagr_delta = float(backtest.iloc[1]["delta_vs_native_cagr"])
        dd_delta = float(backtest.iloc[1]["delta_vs_native_drawdown"])
        if cagr_delta >= 0.0 and dd_delta >= 0.0:
            backtest_label = "overlay improved return and drawdown"
        elif cagr_delta >= 0.0:
            backtest_label = "overlay improved return but not drawdown"
        elif dd_delta >= 0.0:
            backtest_label = "overlay reduced drawdown but cost return"
        else:
            backtest_label = "overlay hurt return and drawdown"
    return pd.Series(
        {
            "top_down_signal": top_down_signal,
            "top_down_score": top_down_score,
            "confirmed_assets": confirmed,
            "partial_assets": partial,
            "watch_only_assets": watch_only,
            "backtest_label": backtest_label,
        }
    )


def _ranking_values(
    momentum: pd.DataFrame,
    vol_adjusted_momentum: pd.DataFrame,
    ranking_metric: str,
) -> pd.DataFrame:
    if ranking_metric in {"risk_adjusted_return", "return_trend_quality"}:
        return vol_adjusted_momentum
    return momentum


def _trend_pass(prices: pd.DataFrame, trend_filter_days: int | None) -> pd.DataFrame:
    if trend_filter_days is None:
        return pd.DataFrame(True, index=prices.index, columns=prices.columns)
    moving_average = prices.ffill().rolling(window=trend_filter_days, min_periods=trend_filter_days).mean()
    return (prices.ffill() >= moving_average).fillna(False)


def _bottom_up_signal(
    *,
    momentum: object,
    rank: object,
    trend_pass: object,
    selected_cutoff: int,
    min_return: float,
) -> str:
    try:
        momentum_value = float(momentum)  # type: ignore[arg-type]
        rank_value = float(rank)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "bearish"
    if momentum_value != momentum_value or rank_value != rank_value:
        return "bearish"
    if bool(trend_pass) and rank_value <= selected_cutoff and momentum_value >= min_return:
        return "bullish"
    if bool(trend_pass) and momentum_value > 0.0:
        return "neutral"
    return "bearish"


def _confirmation_state(row: pd.Series) -> str:
    top_down = str(row.get("top_down_signal"))
    bottom_up = str(row.get("bottom_up_signal"))
    if top_down == "bullish" and bottom_up == "bullish":
        return "long_max"
    if top_down != "bearish" and bottom_up in {"bullish", "neutral"}:
        return "long_half"
    return "watch_only"


def _asset_state_read(row: pd.Series) -> str:
    state = str(row.get("confirmation_state"))
    top_down = str(row.get("top_down_signal"))
    bottom_up = str(row.get("bottom_up_signal"))
    if state == "long_max":
        return "Top-down regime and asset momentum agree."
    if state == "long_half":
        return f"Partial confirmation: top-down is {top_down}, asset momentum is {bottom_up}."
    return f"Not confirmed: top-down is {top_down}, asset momentum is {bottom_up}."


def _defensive_state_read(defensive_weight: float) -> str:
    if defensive_weight >= 0.65:
        return "Defensive reserve is the dominant allocation."
    if defensive_weight >= 0.35:
        return "Defensive reserve is material but not dominant."
    return "Defensive reserve is light."
