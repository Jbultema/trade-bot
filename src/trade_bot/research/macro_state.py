from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.data.fred_data import FredSeries
from trade_bot.DEFAULT import DEFAULT_MACRO_SIGNAL_LOOKBACK_DAYS


def build_macro_signal_table(
    macro_data: pd.DataFrame,
    catalog: tuple[FredSeries, ...],
    *,
    lookback_days: int = DEFAULT_MACRO_SIGNAL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    if macro_data.empty or not catalog:
        return pd.DataFrame()

    latest_date = macro_data.index.max()
    rows: list[dict[str, object]] = []
    catalog_lookup = {series.series_id: series for series in catalog}
    for series_id, metadata in catalog_lookup.items():
        if series_id not in macro_data.columns:
            rows.append(_missing_row(metadata, latest_date))
            continue

        series = macro_data[series_id].dropna().sort_index()
        if series.empty:
            rows.append(_missing_row(metadata, latest_date))
            continue

        latest_value = float(series.iloc[-1])
        history = series.tail(lookback_days)
        percentile = float((history <= latest_value).mean())
        z_score = _z_score(latest_value, history)
        latest_value_1w = _value_at_or_before(series, series.index[-1] - pd.DateOffset(days=7))
        latest_value_1m = _value_at_or_before(series, series.index[-1] - pd.DateOffset(months=1))
        change_3m = _level_change(series, 3)
        change_12m = _level_change(series, 12)
        change_1w = _day_level_change(series, 7)
        change_2w = _day_level_change(series, 14)
        change_1m = _level_change(series, 1)
        pct_change_3m = _pct_change(series, 3)
        pct_change_12m = _pct_change(series, 12)
        pct_change_1w = _day_pct_change(series, 7)
        pct_change_2w = _day_pct_change(series, 14)
        pct_change_1m = _pct_change(series, 1)
        change_acceleration = _change_acceleration(change_1m, change_3m)
        short_move_z = _short_move_z(series, 21, lookback_days)
        range_position_1y = _range_position(series, 12)
        slope_1m = _slope(series, 1)
        slope_3m = _slope(series, 3)
        realized_vol_1m = _realized_volatility(series, 1)
        realized_vol_3m = _realized_volatility(series, 3)
        reversal_pressure = _reversal_pressure(
            latest_value,
            latest_value_1w,
            latest_value_1m,
            metadata.risk_polarity,
        )
        risk_score = _risk_score(
            metadata.risk_polarity,
            z_score,
            change_1m,
            pct_change_1m,
            short_move_z,
            change_acceleration,
        )

        rows.append(
            {
                "series_id": metadata.series_id,
                "name": metadata.name,
                "category": metadata.category,
                "risk_polarity": metadata.risk_polarity,
                "latest_date": str(series.index[-1].date()),
                "latest_value": latest_value,
                "z_score_5y": z_score,
                "percentile_5y": percentile,
                "change_1w": change_1w,
                "change_2w": change_2w,
                "change_1m": change_1m,
                "change_3m": change_3m,
                "change_12m": change_12m,
                "pct_change_1w": pct_change_1w,
                "pct_change_2w": pct_change_2w,
                "pct_change_1m": pct_change_1m,
                "pct_change_3m": pct_change_3m,
                "pct_change_12m": pct_change_12m,
                "short_move_z_1m": short_move_z,
                "change_acceleration_1m_vs_3m": change_acceleration,
                "range_position_1y": range_position_1y,
                "slope_1m": slope_1m,
                "slope_3m": slope_3m,
                "realized_vol_1m": realized_vol_1m,
                "realized_vol_3m": realized_vol_3m,
                "reversal_pressure": reversal_pressure,
                "risk_score": risk_score,
                "risk_state": _risk_state(risk_score),
                "near_term_state": _near_term_state(
                    metadata.risk_polarity,
                    short_move_z,
                    change_acceleration,
                    reversal_pressure,
                ),
                "stale_days": int((latest_date - series.index[-1]).days),
                "observations": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "risk_score"], ascending=[True, False])


def build_macro_category_summary(macro_signals: pd.DataFrame) -> pd.DataFrame:
    if macro_signals.empty:
        return pd.DataFrame()

    summary = (
        macro_signals.groupby("category", as_index=False)
        .agg(
            series=("series_id", "count"),
            usable_series=("latest_value", lambda values: values.notna().sum()),
            mean_risk_score=("risk_score", "mean"),
            max_stale_days=("stale_days", "max"),
            latest_date=("latest_date", _latest_text_date),
        )
        .sort_values("mean_risk_score", ascending=False)
    )
    summary["risk_state"] = summary["mean_risk_score"].map(_risk_state)
    summary["usable_share"] = summary["usable_series"] / summary["series"]
    return summary


def build_signal_coverage_table(
    *,
    yahoo_prices: pd.DataFrame,
    macro_data: pd.DataFrame,
    macro_catalog: tuple[FredSeries, ...],
) -> pd.DataFrame:
    market_categories = {
        "market_price_proxy": yahoo_prices.shape[1],
        "macro_fred_series": len(macro_catalog),
        "macro_fred_loaded": macro_data.shape[1],
    }
    macro_by_category: dict[str, int] = {}
    for series in macro_catalog:
        macro_by_category[series.category] = macro_by_category.get(series.category, 0) + 1

    rows = [
        {
            "coverage_area": "Yahoo market proxies",
            "series_count": market_categories["market_price_proxy"],
            "status": "implemented",
            "institutional_stack_gap": "Needs depth in constituents, global assets, options/vol surfaces, and fund flows.",
        },
        {
            "coverage_area": "FRED macro catalog",
            "series_count": market_categories["macro_fred_series"],
            "status": "implemented",
            "institutional_stack_gap": "Still missing release-lag discipline, revisions, global macro, and private Bloomberg datasets.",
        },
        {
            "coverage_area": "FRED macro loaded",
            "series_count": market_categories["macro_fred_loaded"],
            "status": "implemented",
            "institutional_stack_gap": "Loaded series depend on FRED availability and cache freshness.",
        },
    ]
    for category, count in sorted(macro_by_category.items()):
        rows.append(
            {
                "coverage_area": f"macro:{category}",
                "series_count": count,
                "status": "implemented",
                "institutional_stack_gap": "Empirical inclusion test available; see Signal Inclusion Tests before granting allocation authority.",
            }
        )

    rows.extend(
        [
            {
                "coverage_area": "regime pulse cycles",
                "series_count": 6,
                "status": "implemented",
                "institutional_stack_gap": "Cycle reads now exist for growth, inflation, monetary policy, fiscal policy, liquidity, and positioning; still needs consensus forecast and vintage discipline.",
            },
            {
                "coverage_area": "positioning and crowding",
                "series_count": yahoo_prices.shape[1],
                "status": "partial",
                "institutional_stack_gap": "Price/RSI crowding proxies are implemented; true CFTC COT, AAII/NAAIM, ETF/mutual-fund flows, short interest, and dealer/CTA estimates remain missing.",
            },
            {
                "coverage_area": "earnings and fundamentals",
                "series_count": 0,
                "status": "gap",
                "institutional_stack_gap": "Forward EPS, revisions, margins, valuations, buybacks, sector-level estimates.",
            },
            {
                "coverage_area": "global macro breadth",
                "series_count": 0,
                "status": "gap",
                "institutional_stack_gap": "Country-level PMIs, inflation, policy rates, FX reserves, trade, and balance-of-payments.",
            },
            {
                "coverage_area": "options and volatility surface",
                "series_count": 0,
                "status": "gap",
                "institutional_stack_gap": "Skew, term structure, implied correlation, MOVE detail, single-name vol, realized/implied spreads.",
            },
            {
                "coverage_area": "news and narrative features",
                "series_count": 0,
                "status": "partial",
                "institutional_stack_gap": "Event-risk scaffolding exists; needs source ingestion, tagging, and backtested narrative indices.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _missing_row(metadata: FredSeries, latest_date: pd.Timestamp) -> dict[str, object]:
    return {
        "series_id": metadata.series_id,
        "name": metadata.name,
        "category": metadata.category,
        "risk_polarity": metadata.risk_polarity,
        "latest_date": None,
        "latest_value": np.nan,
        "z_score_5y": np.nan,
        "percentile_5y": np.nan,
        "change_1w": np.nan,
        "change_2w": np.nan,
        "change_1m": np.nan,
        "change_3m": np.nan,
        "change_12m": np.nan,
        "pct_change_1w": np.nan,
        "pct_change_2w": np.nan,
        "pct_change_1m": np.nan,
        "pct_change_3m": np.nan,
        "pct_change_12m": np.nan,
        "short_move_z_1m": np.nan,
        "change_acceleration_1m_vs_3m": np.nan,
        "range_position_1y": np.nan,
        "slope_1m": np.nan,
        "slope_3m": np.nan,
        "realized_vol_1m": np.nan,
        "realized_vol_3m": np.nan,
        "reversal_pressure": np.nan,
        "risk_score": 0.0,
        "risk_state": "missing",
        "near_term_state": "missing",
        "stale_days": np.nan,
        "observations": 0,
    }


def _latest_text_date(values: pd.Series) -> str | None:
    clean = values.dropna().astype(str)
    if clean.empty:
        return None
    return str(clean.max())


def _level_change(series: pd.Series, months: int) -> float:
    prior = _value_at_or_before(series, series.index[-1] - pd.DateOffset(months=months))
    if pd.isna(prior):
        return np.nan
    return float(series.iloc[-1] - prior)


def _day_level_change(series: pd.Series, days: int) -> float:
    prior = _value_at_or_before(series, series.index[-1] - pd.DateOffset(days=days))
    if pd.isna(prior):
        return np.nan
    return float(series.iloc[-1] - prior)


def _pct_change(series: pd.Series, months: int) -> float:
    prior = _value_at_or_before(series, series.index[-1] - pd.DateOffset(months=months))
    if pd.isna(prior):
        return np.nan
    latest = float(series.iloc[-1])
    if prior == 0 or np.sign(prior) != np.sign(latest):
        return np.nan
    return float(latest / prior - 1.0)


def _day_pct_change(series: pd.Series, days: int) -> float:
    prior = _value_at_or_before(series, series.index[-1] - pd.DateOffset(days=days))
    if pd.isna(prior):
        return np.nan
    latest = float(series.iloc[-1])
    if prior == 0 or np.sign(prior) != np.sign(latest):
        return np.nan
    return float(latest / prior - 1.0)


def _value_at_or_before(series: pd.Series, target_date: pd.Timestamp) -> float:
    values = series.loc[:target_date]
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def _z_score(latest_value: float, history: pd.Series) -> float:
    std = float(history.std())
    if std == 0 or np.isnan(std):
        return 0.0
    return float((latest_value - float(history.mean())) / std)


def _change_acceleration(change_1m: float, change_3m: float) -> float:
    if pd.isna(change_1m) or pd.isna(change_3m):
        return np.nan
    return float(change_1m - change_3m / 3.0)


def _short_move_z(series: pd.Series, days: int, lookback_days: int) -> float:
    moves = _day_level_changes(series, days).dropna().tail(lookback_days)
    if moves.empty:
        return np.nan
    latest_move = _day_level_change(series, days)
    std = float(moves.std())
    if std == 0 or np.isnan(std):
        return 0.0
    return float((latest_move - float(moves.mean())) / std)


def _day_level_changes(series: pd.Series, days: int) -> pd.Series:
    indexed = series.sort_index()
    rows = []
    for date, value in indexed.items():
        prior = _value_at_or_before(indexed.loc[:date], date - pd.DateOffset(days=days))
        if pd.isna(prior):
            rows.append(np.nan)
        else:
            rows.append(float(value - prior))
    return pd.Series(rows, index=indexed.index)


def _range_position(series: pd.Series, months: int) -> float:
    start_date = series.index[-1] - pd.DateOffset(months=months)
    window = series.loc[start_date:]
    if window.empty:
        return np.nan
    low = float(window.min())
    high = float(window.max())
    if high == low:
        return 0.5
    return float((series.iloc[-1] - low) / (high - low))


def _slope(series: pd.Series, months: int) -> float:
    start_date = series.index[-1] - pd.DateOffset(months=months)
    window = series.loc[start_date:]
    if window.shape[0] < 2:
        return np.nan
    x = np.arange(window.shape[0], dtype=float)
    y = window.to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _realized_volatility(series: pd.Series, months: int) -> float:
    start_date = series.index[-1] - pd.DateOffset(months=months)
    window = series.loc[start_date:]
    if window.shape[0] < 3:
        return np.nan
    diffs = window.diff().dropna()
    if diffs.empty:
        return np.nan
    return float(diffs.std())


def _reversal_pressure(
    latest_value: float,
    latest_value_1w: float,
    latest_value_1m: float,
    risk_polarity: str,
) -> float:
    if pd.isna(latest_value_1w) or pd.isna(latest_value_1m):
        return np.nan
    recent_move = latest_value - latest_value_1w
    prior_move = latest_value_1w - latest_value_1m
    if recent_move == 0 or prior_move == 0 or np.sign(recent_move) == np.sign(prior_move):
        return 0.0
    direction = 1.0 if risk_polarity == "risk_off_when_rising" else -1.0
    if risk_polarity == "neutral":
        direction = 0.0
    return float(direction * np.sign(recent_move) * min(1.0, abs(recent_move / prior_move)))


def _risk_score(
    risk_polarity: str,
    z_score: float,
    change_1m: float,
    pct_change_1m: float,
    short_move_z: float,
    change_acceleration: float,
) -> float:
    if risk_polarity == "neutral":
        return 0.0
    direction = 1.0 if risk_polarity == "risk_off_when_rising" else -1.0
    change_component = pct_change_1m if pd.notna(pct_change_1m) else change_1m
    if pd.isna(change_component):
        change_component = 0.0
    if pd.isna(short_move_z):
        short_move_z = 0.0
    if pd.isna(change_acceleration):
        change_acceleration = 0.0
    scaled_change = float(np.tanh(change_component))
    scaled_z = float(np.tanh(z_score / 2.0))
    scaled_short_move = float(np.tanh(short_move_z / 2.0))
    scaled_acceleration = float(np.tanh(change_acceleration))
    return float(
        np.clip(
            direction
            * (
                0.45 * scaled_z
                + 0.25 * scaled_change
                + 0.20 * scaled_short_move
                + 0.10 * scaled_acceleration
            ),
            -1.0,
            1.0,
        )
    )


def _risk_state(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.35:
        return "risk_pressure"
    if score <= -0.35:
        return "risk_supportive"
    return "mixed"


def _near_term_state(
    risk_polarity: str,
    short_move_z: float,
    change_acceleration: float,
    reversal_pressure: float,
) -> str:
    if risk_polarity == "neutral":
        return "neutral"
    if pd.isna(short_move_z):
        short_move_z = 0.0
    if pd.isna(change_acceleration):
        change_acceleration = 0.0
    if pd.isna(reversal_pressure):
        reversal_pressure = 0.0
    direction = 1.0 if risk_polarity == "risk_off_when_rising" else -1.0
    pressure = (
        direction * (0.65 * short_move_z + 0.25 * change_acceleration) + 0.10 * reversal_pressure
    )
    if pressure >= 0.75:
        return "near_term_risk_pressure"
    if pressure <= -0.75:
        return "near_term_risk_supportive"
    if abs(reversal_pressure) >= 0.50:
        return "near_term_reversal"
    return "near_term_mixed"
