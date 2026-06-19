from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.DEFAULT import (
    DEFAULT_RISK_AI_BETA_TICKERS,
    DEFAULT_RISK_BROAD_EQUITY_TICKERS,
    DEFAULT_RISK_COMMODITY_TICKERS,
    DEFAULT_RISK_CREDIT_TICKERS,
    DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS,
    DEFAULT_RISK_DOLLAR_TICKERS,
    DEFAULT_RISK_DURATION_TICKERS,
    DEFAULT_RISK_ENERGY_TICKERS,
    DEFAULT_RISK_INTERNATIONAL_TICKERS,
    DEFAULT_RISK_SECTOR_TICKERS,
)
from trade_bot.features.indicators import daily_returns


def build_positioning_crowding_table(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 63,
    rsi_days: int = 14,
    z_history_days: int = 504,
) -> pd.DataFrame:
    """Build a 42-style crowding proxy from price behavior.

    This is intentionally labeled as a proxy: commercial positioning models often use ETF flows, surveys, and
    positioning inputs that are not all in our local store yet. The same output
    schema can later accept true flow, survey, and CFTC series.
    """
    clean = prices.dropna(how="all").sort_index().ffill()
    if clean.empty:
        return pd.DataFrame()

    returns = daily_returns(clean)
    trailing_return = clean.pct_change(lookback_days, fill_method=None)
    return_z = _rolling_z_score(trailing_return, z_history_days)
    rsi = _rsi(clean, rsi_days)
    vol = returns.rolling(lookback_days).std() * np.sqrt(252)

    latest_rows: list[dict[str, object]] = []
    latest_date = clean.index.max()
    for ticker in clean.columns:
        price_series = clean[ticker].dropna()
        if price_series.empty:
            continue
        row = {
            "ticker": ticker,
            "asset_group": _asset_group(ticker),
            "latest_date": str(latest_date.date()),
            "price": float(price_series.iloc[-1]),
            "return_3m": _latest_value(trailing_return[ticker]),
            "return_3m_z": _latest_value(return_z[ticker]),
            "rsi_14d": _latest_value(rsi[ticker]),
            "realized_vol_3m": _latest_value(vol[ticker]),
        }
        row["crowding_score"] = _crowding_score(row["return_3m_z"], row["rsi_14d"])
        row["crowding_state"] = _crowding_state(row["return_3m_z"], row["rsi_14d"])
        row["positioning_read"] = _positioning_read(row["crowding_state"])
        latest_rows.append(row)

    if not latest_rows:
        return pd.DataFrame()
    return pd.DataFrame(latest_rows).sort_values(
        ["crowding_score", "ticker"],
        ascending=[False, True],
        na_position="last",
    )


def build_positioning_summary(crowding: pd.DataFrame) -> pd.DataFrame:
    if crowding.empty:
        return pd.DataFrame()

    summary = (
        crowding.groupby("asset_group", as_index=False)
        .agg(
            tickers=("ticker", "count"),
            mean_crowding_score=("crowding_score", "mean"),
            crowded_count=(
                "crowding_state",
                lambda values: int(
                    values.isin(["bearish_crowding", "crowded_risk"]).sum()
                ),
            ),
            washout_count=(
                "crowding_state",
                lambda values: int(
                    values.isin(["bullish_washout", "washed_out_opportunity"]).sum()
                ),
            ),
            strongest_crowding=("ticker", lambda values: _top_ticker(crowding, values, True)),
            strongest_washout=("ticker", lambda values: _top_ticker(crowding, values, False)),
        )
        .sort_values("mean_crowding_score", ascending=False)
    )
    summary["asset_group_read"] = summary["mean_crowding_score"].map(_group_read)
    summary["crowded_share"] = summary["crowded_count"] / summary["tickers"]
    summary["washout_share"] = summary["washout_count"] / summary["tickers"]
    return summary


def _rolling_z_score(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(20, window // 4)).mean()
    std = frame.rolling(window, min_periods=max(20, window // 4)).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def _rsi(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    delta = prices.diff()
    gains = delta.clip(lower=0.0).rolling(days, min_periods=days).mean()
    losses = (-delta.clip(upper=0.0)).rolling(days, min_periods=days).mean()
    rs = gains / losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _latest_value(series: pd.Series) -> float:
    value = series.dropna().iloc[-1] if not series.dropna().empty else np.nan
    return float(value) if pd.notna(value) else np.nan


def _crowding_score(return_z: object, rsi: object) -> float:
    z_value = 0.0 if pd.isna(return_z) else float(return_z)
    rsi_value = 50.0 if pd.isna(rsi) else float(rsi)
    rsi_component = (rsi_value - 50.0) / 25.0
    return float(np.clip(0.65 * np.tanh(z_value / 2.0) + 0.35 * rsi_component, -1.5, 1.5))


def _crowding_state(return_z: object, rsi: object) -> str:
    z_value = 0.0 if pd.isna(return_z) else float(return_z)
    rsi_value = 50.0 if pd.isna(rsi) else float(rsi)
    if z_value >= 2.0 and rsi_value >= 70.0:
        return "bearish_crowding"
    if z_value <= -2.0 and rsi_value <= 30.0:
        return "bullish_washout"
    if z_value >= 1.25 or rsi_value >= 65.0:
        return "crowded_risk"
    if z_value <= -1.25 or rsi_value <= 35.0:
        return "washed_out_opportunity"
    return "neutral"


def _positioning_read(state: str) -> str:
    return {
        "bearish_crowding": "risk headwind: crowded upside",
        "crowded_risk": "watch: elevated crowding",
        "bullish_washout": "possible contrarian re-risk setup",
        "washed_out_opportunity": "watch: washed-out re-entry candidate",
        "neutral": "no clear crowding signal",
    }[state]


def _group_read(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.45:
        return "crowding headwind"
    if score <= -0.45:
        return "washout opportunity"
    return "mixed or neutral"


def _top_ticker(crowding: pd.DataFrame, values: pd.Series, highest: bool) -> str:
    group = crowding[crowding["ticker"].isin(values)]
    if group.empty:
        return ""
    ordered = group.sort_values("crowding_score", ascending=not highest)
    row = ordered.iloc[0]
    return f"{row['ticker']} ({float(row['crowding_score']):.2f})"


def _asset_group(ticker: str) -> str:
    symbol = ticker.upper()
    if symbol in set(DEFAULT_RISK_AI_BETA_TICKERS):
        return "ai_beta"
    if symbol in set(DEFAULT_RISK_BROAD_EQUITY_TICKERS):
        return "broad_us_equity"
    if symbol in set(DEFAULT_RISK_SECTOR_TICKERS):
        return "us_equity_sectors"
    if symbol in set(DEFAULT_RISK_INTERNATIONAL_TICKERS):
        return "global_equity"
    if symbol in set(DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS):
        return "defensive_equity_factor"
    if symbol in set(DEFAULT_RISK_DURATION_TICKERS):
        return "duration"
    if symbol in set(DEFAULT_RISK_CREDIT_TICKERS):
        return "credit"
    if symbol in set(DEFAULT_RISK_COMMODITY_TICKERS) | set(DEFAULT_RISK_ENERGY_TICKERS):
        return "commodities_energy"
    if symbol in set(DEFAULT_RISK_DOLLAR_TICKERS):
        return "dollar_fx"
    return "other"
