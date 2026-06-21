from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_RISK_AI_BETA_TICKERS,
    DEFAULT_RISK_BROAD_EQUITY_TICKERS,
    DEFAULT_RISK_CREDIT_TICKERS,
    DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS,
    DEFAULT_RISK_SECTOR_TICKERS,
    TRADING_DAYS_PER_YEAR,
)
from trade_bot.features.indicators import daily_returns


def build_regime_instability_index(
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a watch-only transition-risk diagnostic from price-observable signals.

    This intentionally does not set the trade risk score. It is a research signal meant to
    track whether the market is becoming statistically unstable even when headline index
    trend remains constructive.
    """
    clean = prices.dropna(how="all").sort_index().ffill()
    if clean.empty or benchmark not in clean:
        return pd.DataFrame(), pd.DataFrame()

    returns = daily_returns(clean)
    benchmark_returns = returns[benchmark]
    component_rows = [
        _large_move_component(benchmark_returns, window=21),
        _large_move_component(benchmark_returns, window=63),
        _realized_vol_component(benchmark_returns, window=21),
        _realized_vol_component(benchmark_returns, window=63),
        _cross_section_dispersion_component(returns),
        _vol_proxy_component(clean, returns),
        _correlation_shift_component(returns),
        _breadth_concentration_component(clean),
        _credit_stress_component(clean),
    ]
    components = pd.DataFrame([row for row in component_rows if row])
    if components.empty:
        return pd.DataFrame(), components

    components["component_score"] = pd.to_numeric(
        components["component_score"], errors="coerce"
    ).clip(0.0, 1.0)
    components["weight"] = pd.to_numeric(components["weight"], errors="coerce").fillna(0.0)
    usable = components.dropna(subset=["component_score"])
    if usable.empty or float(usable["weight"].sum()) <= 0:
        return pd.DataFrame(), components

    score = float(np.average(usable["component_score"], weights=usable["weight"]))
    ytd_share, ytd_count, ytd_days = _ytd_large_move_share(benchmark_returns)
    top_components = (
        usable.sort_values("component_score", ascending=False)
        .head(3)["component"]
        .astype(str)
        .tolist()
    )
    summary = pd.DataFrame(
        [
            {
                "market_date": str(clean.index.max().date()),
                "regime_instability_score": score,
                "regime_instability_state": _instability_state(score),
                "regime_instability_read": _instability_read(score),
                "spy_ytd_large_move_days": ytd_count,
                "spy_ytd_trading_days": ytd_days,
                "spy_ytd_large_move_share": ytd_share,
                "top_instability_components": "; ".join(top_components),
                "trading_use": (
                    "watch_only: does not alter sizing until backtested as an overlay"
                ),
            }
        ]
    )
    return summary, components.sort_values("component_score", ascending=False).reset_index(drop=True)


def _large_move_component(returns: pd.Series, *, window: int) -> dict[str, object]:
    share = returns.abs().ge(0.01).rolling(window, min_periods=max(5, window // 3)).mean()
    latest = _latest(share)
    percentile = _percentile_rank(share)
    absolute_score = _threshold_score(latest, calm=0.08, stressed=0.45)
    score = _blend_score(percentile, absolute_score)
    return {
        "component": f"large_move_share_{window}d",
        "component_score": score,
        "latest_value": latest,
        "latest_percentile": percentile,
        "state": _component_state(score),
        "weight": 0.14 if window == 21 else 0.10,
        "interpretation": f"Share of {window}d sessions where SPY moved at least +/-1%.",
    }


def _realized_vol_component(returns: pd.Series, *, window: int) -> dict[str, object]:
    vol = returns.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    latest = _latest(vol)
    percentile = _percentile_rank(vol)
    absolute_score = _threshold_score(
        latest,
        calm=0.12 if window == 21 else 0.10,
        stressed=0.35 if window == 21 else 0.30,
    )
    score = _blend_score(percentile, absolute_score)
    return {
        "component": f"realized_vol_{window}d",
        "component_score": score,
        "latest_value": latest,
        "latest_percentile": percentile,
        "state": _component_state(score),
        "weight": 0.12 if window == 21 else 0.10,
        "interpretation": f"Annualized {window}d realized volatility for SPY.",
    }


def _cross_section_dispersion_component(returns: pd.DataFrame) -> dict[str, object]:
    tickers = _available_equity_tickers(returns.columns)
    if len(tickers) < 5:
        return {}
    dispersion = returns[tickers].std(axis=1).rolling(21, min_periods=8).mean()
    latest = _latest(dispersion)
    percentile = _percentile_rank(dispersion)
    absolute_score = _threshold_score(latest, calm=0.006, stressed=0.020)
    score = _blend_score(percentile, absolute_score)
    return {
        "component": "cross_section_dispersion_21d",
        "component_score": score,
        "latest_value": latest,
        "latest_percentile": percentile,
        "state": _component_state(score),
        "weight": 0.18,
        "interpretation": (
            "Average 21d cross-sectional daily-return dispersion across liquid equity proxies."
        ),
    }


def _vol_proxy_component(clean: pd.DataFrame, returns: pd.DataFrame) -> dict[str, object]:
    if "VIXY" not in clean:
        return {}
    vixy_21d = clean["VIXY"].pct_change(21, fill_method=None)
    latest = _latest(vixy_21d)
    percentile = _percentile_rank(vixy_21d)
    absolute_score = _threshold_score(latest, calm=0.0, stressed=0.35)
    score = _blend_score(percentile, absolute_score)
    return {
        "component": "volatility_proxy_pressure",
        "component_score": score,
        "latest_value": latest,
        "latest_percentile": percentile,
        "state": _component_state(score),
        "weight": 0.10,
        "interpretation": "21d VIXY move as a tradable proxy for rising volatility pressure.",
    }


def _correlation_shift_component(returns: pd.DataFrame) -> dict[str, object]:
    tickers = _available_core_tickers(returns.columns)
    if len(tickers) < 4:
        return {}
    short_corr = _average_pairwise_rolling_correlation(returns[tickers], 21)
    long_corr = _average_pairwise_rolling_correlation(returns[tickers], 126)
    shift = short_corr - long_corr
    latest_shift = _latest(shift)
    score = float(np.clip((latest_shift + 0.15) / 0.45, 0.0, 1.0)) if pd.notna(latest_shift) else np.nan
    return {
        "component": "correlation_shift_21d_vs_126d",
        "component_score": score,
        "latest_value": latest_shift,
        "latest_percentile": _percentile_rank(shift),
        "state": _component_state(score),
        "weight": 0.08,
        "interpretation": "Short-run cross-asset correlation minus longer-run correlation.",
    }


def _breadth_concentration_component(clean: pd.DataFrame) -> dict[str, object]:
    scores: list[float] = []
    values: list[str] = []
    if {"RSP", "SPY"}.issubset(clean.columns):
        breadth = (clean["RSP"] / clean["SPY"]).pct_change(63, fill_method=None)
        breadth_score = 1.0 - _percentile_rank(breadth)
        scores.append(breadth_score)
        values.append(f"RSP/SPY 63d {_format_number(_latest(breadth))}")
    if {"QQQ", "RSP"}.issubset(clean.columns):
        concentration = (clean["QQQ"] / clean["RSP"]).pct_change(63, fill_method=None)
        scores.append(_percentile_rank(concentration))
        values.append(f"QQQ/RSP 63d {_format_number(_latest(concentration))}")
    if {"SMH", "SPY"}.issubset(clean.columns):
        ai = (clean["SMH"] / clean["SPY"]).pct_change(63, fill_method=None)
        scores.append(_percentile_rank(ai))
        values.append(f"SMH/SPY 63d {_format_number(_latest(ai))}")
    if not scores:
        return {}
    score = float(np.nanmean(scores))
    return {
        "component": "breadth_concentration_pressure",
        "component_score": score,
        "latest_value": score,
        "latest_percentile": score,
        "state": _component_state(score),
        "weight": 0.13,
        "interpretation": "; ".join(values),
    }


def _credit_stress_component(clean: pd.DataFrame) -> dict[str, object]:
    if not {"HYG", "LQD"}.issubset(clean.columns):
        return {}
    credit = (clean["HYG"] / clean["LQD"]).pct_change(21, fill_method=None)
    score = 1.0 - _percentile_rank(credit)
    return {
        "component": "credit_stress_pressure",
        "component_score": score,
        "latest_value": _latest(credit),
        "latest_percentile": score,
        "state": _component_state(score),
        "weight": 0.05,
        "interpretation": "Low 21d HYG/LQD relative return signals credit stress.",
    }


def _average_pairwise_rolling_correlation(frame: pd.DataFrame, window: int) -> pd.Series:
    pairs = []
    for left, right in combinations(frame.columns, 2):
        pairs.append(frame[left].rolling(window, min_periods=max(8, window // 3)).corr(frame[right]))
    if not pairs:
        return pd.Series(index=frame.index, dtype=float)
    return pd.concat(pairs, axis=1).mean(axis=1)


def _available_equity_tickers(columns: pd.Index) -> list[str]:
    candidates = (
        set(DEFAULT_RISK_BROAD_EQUITY_TICKERS)
        | set(DEFAULT_RISK_AI_BETA_TICKERS)
        | set(DEFAULT_RISK_SECTOR_TICKERS)
        | set(DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS)
    )
    return [ticker for ticker in columns if str(ticker).upper() in candidates]


def _available_core_tickers(columns: pd.Index) -> list[str]:
    candidates = [
        "SPY",
        "QQQ",
        "RSP",
        "IWM",
        "SMH",
        "XLK",
        "XLF",
        "XLE",
        "HYG",
        "LQD",
        "TLT",
        "GLD",
        *DEFAULT_RISK_CREDIT_TICKERS,
    ]
    return list(dict.fromkeys([ticker for ticker in candidates if ticker in columns]))


def _ytd_large_move_share(returns: pd.Series) -> tuple[float, int, int]:
    if returns.empty:
        return np.nan, 0, 0
    latest = returns.index.max()
    ytd = returns[returns.index.year == latest.year]
    if ytd.empty:
        return np.nan, 0, 0
    count = int(ytd.abs().ge(0.01).sum())
    days = int(ytd.shape[0])
    return float(count / days) if days else np.nan, count, days


def _percentile_rank(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    latest = clean.iloc[-1]
    return float((clean <= latest).mean())


def _threshold_score(value: float, *, calm: float, stressed: float) -> float:
    if pd.isna(value) or stressed <= calm:
        return np.nan
    return float(np.clip((value - calm) / (stressed - calm), 0.0, 1.0))


def _blend_score(percentile_score: float, absolute_score: float) -> float:
    scores = [score for score in [percentile_score, absolute_score] if pd.notna(score)]
    if not scores:
        return np.nan
    if len(scores) == 1:
        return float(scores[0])
    return float(0.55 * percentile_score + 0.45 * absolute_score)


def _latest(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.iloc[-1])


def _component_state(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.80:
        return "stressed"
    if score >= 0.65:
        return "elevated"
    if score >= 0.45:
        return "mixed"
    return "calm"


def _instability_state(score: float) -> str:
    if score >= 0.70:
        return "stressed"
    if score >= 0.55:
        return "unstable"
    if score >= 0.35:
        return "elevated"
    return "calm"


def _instability_read(score: float) -> str:
    state = _instability_state(score)
    if state == "stressed":
        return "Transition risk is high; demand confirmation before adding risk."
    if state == "unstable":
        return "Market internals are unstable even if trend is not fully broken."
    if state == "elevated":
        return "Instability is above normal but not a standalone de-risk command."
    return "Instability is contained."


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"
