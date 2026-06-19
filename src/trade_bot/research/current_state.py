from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.data.fred_data import FredSeries
from trade_bot.DEFAULT import (
    DEFAULT_MOMENTUM_STATE_LOOKBACK_DAYS,
    DEFAULT_MOMENTUM_STATE_SKIP_DAYS,
    DEFAULT_MOMENTUM_STATE_VOL_DAYS,
)
from trade_bot.features.indicators import (
    daily_returns,
    drawdown,
    lookback_returns,
    realized_volatility,
)
from trade_bot.portfolio.risk import next_trade_weights
from trade_bot.research.future_scenarios import (
    build_scenario_lattice,
    build_scenario_rollup,
)
from trade_bot.research.macro_state import (
    build_macro_category_summary,
    build_macro_signal_table,
    build_signal_coverage_table,
)
from trade_bot.research.positioning_crowding import (
    build_positioning_crowding_table,
    build_positioning_summary,
)
from trade_bot.research.regime_pulse import (
    build_growth_inflation_map,
    build_regime_pulse_asset_table,
    build_regime_pulse_cycles,
)

MomentumState = Literal["bullish", "neutral", "bearish", "insufficient_data"]


@dataclass(frozen=True)
class CurrentStateRun:
    market_date: str
    risk_score: float
    risk_status: str
    risk_summary: str
    market_health: pd.DataFrame
    momentum_state: pd.DataFrame
    confirmation_matrix: pd.DataFrame
    strategy_alerts: pd.DataFrame
    scenario_outlook: pd.DataFrame
    scenario_lattice: pd.DataFrame
    scenario_drivers: pd.DataFrame
    macro_signals: pd.DataFrame
    macro_category_summary: pd.DataFrame
    signal_coverage: pd.DataFrame
    data_quality: pd.DataFrame
    regime_pulse_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_pulse_assets: pd.DataFrame = field(default_factory=pd.DataFrame)
    growth_inflation_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    positioning_crowding: pd.DataFrame = field(default_factory=pd.DataFrame)
    positioning_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class ConfirmationSignal:
    name: str
    ticker: str
    comparison_ticker: str | None
    theme: str
    status: str
    score: float
    latest_value: float
    explanation: str


def build_current_state(
    prices: pd.DataFrame,
    results: dict[str, BacktestResult],
    *,
    macro_data: pd.DataFrame | None = None,
    macro_catalog: tuple[FredSeries, ...] = (),
    preferred_strategy: str = "drawdown_managed_dual_momentum",
) -> CurrentStateRun:
    clean_prices = prices.dropna(how="all").sort_index()
    macro_frame = macro_data if macro_data is not None else pd.DataFrame()
    market_date = str(clean_prices.index.max().date())
    momentum_state = momentum_state_table(clean_prices)
    data_quality = data_quality_table(clean_prices)
    macro_signals = build_macro_signal_table(macro_frame, macro_catalog)
    macro_category_summary = build_macro_category_summary(macro_signals)
    positioning_crowding = build_positioning_crowding_table(clean_prices)
    positioning_summary = build_positioning_summary(positioning_crowding)
    regime_pulse_cycles = build_regime_pulse_cycles(macro_signals, positioning_summary)
    regime_pulse_assets = build_regime_pulse_asset_table(regime_pulse_cycles)
    growth_inflation_map = build_growth_inflation_map(regime_pulse_cycles)
    signal_coverage = build_signal_coverage_table(
        yahoo_prices=clean_prices,
        macro_data=macro_frame,
        macro_catalog=macro_catalog,
    )
    confirmation_matrix = build_confirmation_matrix(clean_prices, momentum_state)
    market_health = build_market_health(clean_prices, momentum_state)
    risk_score = _risk_score(confirmation_matrix, market_health)
    risk_status = _risk_status(risk_score)
    risk_summary = _risk_summary(risk_status, risk_score, confirmation_matrix)
    strategy_alerts = build_strategy_alerts(results, preferred_strategy=preferred_strategy)
    scenario_lattice, scenario_drivers = build_scenario_lattice(
        confirmation_matrix,
        market_health,
        momentum_state,
        risk_score,
        risk_status,
    )
    scenario_outlook = build_scenario_outlook(scenario_lattice, risk_status)

    return CurrentStateRun(
        market_date=market_date,
        risk_score=risk_score,
        risk_status=risk_status,
        risk_summary=risk_summary,
        market_health=market_health,
        momentum_state=momentum_state,
        confirmation_matrix=confirmation_matrix,
        strategy_alerts=strategy_alerts,
        scenario_outlook=scenario_outlook,
        scenario_lattice=scenario_lattice,
        scenario_drivers=scenario_drivers,
        macro_signals=macro_signals,
        macro_category_summary=macro_category_summary,
        regime_pulse_cycles=regime_pulse_cycles,
        regime_pulse_assets=regime_pulse_assets,
        growth_inflation_map=growth_inflation_map,
        positioning_crowding=positioning_crowding,
        positioning_summary=positioning_summary,
        signal_coverage=signal_coverage,
        data_quality=data_quality,
    )


def momentum_state_table(
    prices: pd.DataFrame,
    *,
    lookback_days: int = DEFAULT_MOMENTUM_STATE_LOOKBACK_DAYS,
    vol_days: int = DEFAULT_MOMENTUM_STATE_VOL_DAYS,
) -> pd.DataFrame:
    returns = daily_returns(prices)
    momentum = lookback_returns(
        prices,
        lookback_days=lookback_days,
        skip_days=DEFAULT_MOMENTUM_STATE_SKIP_DAYS,
    )
    vol = realized_volatility(returns, vol_days)
    score = (momentum / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    latest_prices = prices.ffill().iloc[-1]
    latest_momentum = momentum.iloc[-1]
    latest_vol = vol.iloc[-1]
    latest_score = score.iloc[-1]
    frame = pd.DataFrame(
        {
            "price": latest_prices,
            "momentum_6m_skip_1w": latest_momentum,
            "realized_vol_3m": latest_vol,
            "momentum_state_score": latest_score,
        }
    )
    frame["momentum_state_label"] = frame["momentum_state_score"].map(_momentum_state_label)
    return frame.sort_values("momentum_state_score", ascending=False, na_position="last")


def build_confirmation_matrix(prices: pd.DataFrame, momentum_state: pd.DataFrame) -> pd.DataFrame:
    signals = [
        _relative_signal(prices, momentum_state, "High Beta vs Low Vol", "SPHB", "SPLV", "market_risk"),
        _relative_signal(prices, momentum_state, "Cyclicals vs Defensives", "XLY", "XLP", "market_risk"),
        _relative_signal(prices, momentum_state, "Small Caps vs Mega Caps", "IWM", "MGC", "market_risk"),
        _relative_signal(prices, momentum_state, "Value vs Growth", "VTV", "VUG", "style_rotation"),
        _relative_signal(prices, momentum_state, "Equal Weight vs Cap Weight", "RSP", "SPY", "breadth"),
        _relative_signal(prices, momentum_state, "Nasdaq vs Equal Weight", "QQQ", "RSP", "concentration"),
        _relative_signal(prices, momentum_state, "High Yield vs IG Credit", "HYG", "LQD", "credit"),
        _relative_signal(prices, momentum_state, "Copper vs Gold", "CPER", "GLD", "growth_inflation"),
        _relative_signal(prices, momentum_state, "Semis vs Broad Market", "SMH", "SPY", "ai_beta"),
        _absolute_signal(momentum_state, "SPY Trend", "SPY", "broad_market"),
        _absolute_signal(momentum_state, "QQQ Trend", "QQQ", "ai_beta"),
        _absolute_signal(momentum_state, "Gold Trend", "GLD", "defensive"),
        _absolute_signal(momentum_state, "Long Duration Trend", "TLT", "defensive"),
        _inverse_signal(momentum_state, "Volatility ETF Pressure", "VIXY", "volatility"),
        _inverse_signal(momentum_state, "Dollar Pressure", "UUP", "liquidity"),
    ]
    return pd.DataFrame([asdict(signal) for signal in signals if signal is not None])


def build_market_health(prices: pd.DataFrame, momentum_state: pd.DataFrame) -> pd.DataFrame:
    focus = ["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "TLT", "GLD", "SMH", "VIXY", "UUP"]
    available = [ticker for ticker in focus if ticker in prices.columns]
    returns = daily_returns(prices[available])
    latest = pd.DataFrame(index=available)
    latest["momentum_state_label"] = momentum_state.reindex(available)["momentum_state_label"]
    latest["return_1d"] = returns.iloc[-1]
    latest["return_1w"] = prices[available].ffill().pct_change(5, fill_method=None).iloc[-1]
    latest["return_1m"] = prices[available].ffill().pct_change(21, fill_method=None).iloc[-1]
    latest["return_3m"] = prices[available].ffill().pct_change(63, fill_method=None).iloc[-1]
    latest["drawdown"] = prices[available].ffill().apply(lambda series: drawdown(series).iloc[-1])
    return latest


def build_strategy_alerts(
    results: dict[str, BacktestResult],
    *,
    preferred_strategy: str,
    min_trade_weight: float = 0.02,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        latest_weights = result.weights.iloc[-1].sort_values(ascending=False)
        previous_weights = (
            result.weights.iloc[-2].sort_values(ascending=False)
            if len(result.weights) > 1
            else latest_weights * 0
        )
        trade_weights = next_trade_weights(result.weights)
        material_trades = trade_weights[trade_weights.abs() >= min_trade_weight].sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
        action = "HOLD"
        if not material_trades.empty:
            buys = material_trades[material_trades > 0]
            sells = material_trades[material_trades < 0]
            if not buys.empty and not sells.empty:
                action = "ROTATE"
            elif not buys.empty:
                action = "ADD"
            else:
                action = "REDUCE"

        rows.append(
            {
                "strategy": name,
                "priority": "primary" if name == preferred_strategy else "watch",
                "action": action,
                "latest_position": _format_weight_vector(latest_weights[latest_weights > 0]),
                "previous_position": _format_weight_vector(previous_weights[previous_weights > 0]),
                "trade_alert": _format_trade_vector(material_trades),
                "latest_return_1m": result.returns.tail(21).add(1.0).prod() - 1.0,
                "latest_drawdown": drawdown(result.equity).iloc[-1],
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "strategy"])


def build_scenario_outlook(scenario_lattice: pd.DataFrame, risk_status: str) -> pd.DataFrame:
    return build_scenario_rollup(scenario_lattice, risk_status)


def data_quality_table(prices: pd.DataFrame) -> pd.DataFrame:
    latest_date = prices.index.max()
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "first_date": None,
                    "last_date": None,
                    "observations": 0,
                    "coverage": 0.0,
                    "stale_days": None,
                }
            )
            continue
        rows.append(
            {
                "ticker": ticker,
                "first_date": str(series.index.min().date()),
                "last_date": str(series.index.max().date()),
                "observations": int(series.shape[0]),
                "coverage": float(series.shape[0] / prices.shape[0]),
                "stale_days": int((latest_date - series.index.max()).days),
            }
        )
    return pd.DataFrame(rows).sort_values(["stale_days", "coverage"], ascending=[True, False])


def _absolute_signal(
    momentum_state: pd.DataFrame, name: str, ticker: str, theme: str
) -> ConfirmationSignal | None:
    if ticker not in momentum_state.index:
        return None
    row = momentum_state.loc[ticker]
    state = str(row["momentum_state_label"])
    score = _state_score(state)
    return ConfirmationSignal(
        name=name,
        ticker=ticker,
        comparison_ticker=None,
        theme=theme,
        status=state,
        score=score,
        latest_value=float(row["momentum_state_score"]) if pd.notna(row["momentum_state_score"]) else np.nan,
        explanation=f"{ticker} is {state} on volatility-adjusted momentum.",
    )


def _inverse_signal(
    momentum_state: pd.DataFrame, name: str, ticker: str, theme: str
) -> ConfirmationSignal | None:
    signal = _absolute_signal(momentum_state, name, ticker, theme)
    if signal is None:
        return None
    score = -signal.score
    status = "risk_on" if score > 0 else "risk_off" if score < 0 else "neutral"
    return ConfirmationSignal(
        name=signal.name,
        ticker=signal.ticker,
        comparison_ticker=None,
        theme=signal.theme,
        status=status,
        score=score,
        latest_value=signal.latest_value,
        explanation=f"{ticker} strength is treated as a risk-pressure signal.",
    )


def _relative_signal(
    prices: pd.DataFrame,
    momentum_state: pd.DataFrame,
    name: str,
    numerator: str,
    denominator: str,
    theme: str,
) -> ConfirmationSignal | None:
    if numerator not in prices.columns or denominator not in prices.columns:
        return None
    ratio = prices[numerator].ffill() / prices[denominator].ffill()
    ratio_momentum_state = momentum_state_table(pd.DataFrame({name: ratio})).loc[name]
    state = str(ratio_momentum_state["momentum_state_label"])
    score = _state_score(state)
    return ConfirmationSignal(
        name=name,
        ticker=numerator,
        comparison_ticker=denominator,
        theme=theme,
        status=state,
        score=score,
        latest_value=float(ratio.iloc[-1]),
        explanation=f"{numerator}/{denominator} is {state}; positive momentum confirms {theme}.",
    )


def _risk_score(confirmation_matrix: pd.DataFrame, market_health: pd.DataFrame) -> float:
    if confirmation_matrix.empty:
        return 0.5
    risk_on_score = confirmation_matrix["score"].mean()
    raw_risk = 0.5 - risk_on_score / 2.0

    if "SPY" in market_health.index and market_health.loc["SPY", "drawdown"] < -0.08:
        raw_risk += 0.10
    if "QQQ" in market_health.index and market_health.loc["QQQ", "drawdown"] < -0.10:
        raw_risk += 0.10
    if "HYG" in market_health.index and market_health.loc["HYG", "momentum_state_label"] == "bearish":
        raw_risk += 0.10
    if "VIXY" in market_health.index and market_health.loc["VIXY", "momentum_state_label"] == "bullish":
        raw_risk += 0.15
    return float(max(0.0, min(1.0, raw_risk)))


def _risk_status(risk_score: float) -> str:
    if risk_score < 0.25:
        return "green"
    if risk_score < 0.45:
        return "yellow"
    if risk_score < 0.65:
        return "orange"
    return "red"


def _risk_summary(risk_status: str, risk_score: float, confirmation_matrix: pd.DataFrame) -> str:
    bearish = confirmation_matrix[confirmation_matrix["score"] < 0].shape[0]
    bullish = confirmation_matrix[confirmation_matrix["score"] > 0].shape[0]
    neutral = confirmation_matrix[confirmation_matrix["score"] == 0].shape[0]
    return (
        f"Risk status is {risk_status.upper()} with score {risk_score:.2f}. "
        f"Confirmation matrix has {bullish} bullish, {neutral} neutral, and {bearish} bearish signals."
    )


def _momentum_state_label(score: float) -> MomentumState:
    if pd.isna(score):
        return "insufficient_data"
    if score >= 0.60:
        return "bullish"
    if score <= -0.40:
        return "bearish"
    return "neutral"


def _state_score(state: str) -> float:
    if state == "bullish":
        return 1.0
    if state == "bearish":
        return -1.0
    return 0.0


def _format_weight_vector(weights: pd.Series) -> str:
    if weights.empty:
        return "cash/no position"
    return ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in weights.items())


def _format_trade_vector(trades: pd.Series) -> str:
    if trades.empty:
        return "No material trade; hold current posture."
    parts = []
    for ticker, weight in trades.items():
        verb = "Buy/Add" if weight > 0 else "Sell/Reduce"
        parts.append(f"{verb} {ticker} {abs(weight):.0%}")
    return "; ".join(parts)
