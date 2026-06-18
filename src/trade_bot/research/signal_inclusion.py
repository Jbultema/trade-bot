from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics, metrics_frame
from trade_bot.backtest.windows import rolling_window_metrics, summarize_windows
from trade_bot.config import ExecutionConfig
from trade_bot.data.fred_data import FredSeries
from trade_bot.DEFAULT import (
    DEFAULT_SIGNAL_INCLUSION_DEFENSIVE_TICKER,
    DEFAULT_SIGNAL_INCLUSION_LOOKBACK_DAYS,
    DEFAULT_SIGNAL_INCLUSION_MIN_OBSERVATIONS,
    DEFAULT_SIGNAL_INCLUSION_PRESSURE_THRESHOLD,
    DEFAULT_SIGNAL_INCLUSION_PUBLICATION_LAG_DAYS,
    DEFAULT_SIGNAL_INCLUSION_RISK_MULTIPLIER,
)


@dataclass(frozen=True)
class SignalInclusionRun:
    summary: pd.DataFrame
    pressure: pd.DataFrame
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    window_summary: pd.DataFrame


def run_signal_inclusion_tests(
    prices: pd.DataFrame,
    macro_data: pd.DataFrame,
    macro_catalog: tuple[FredSeries, ...],
    base_result: BacktestResult,
    execution: ExecutionConfig,
    *,
    base_strategy_name: str,
    defensive_ticker: str = DEFAULT_SIGNAL_INCLUSION_DEFENSIVE_TICKER,
    publication_lag_days: int = DEFAULT_SIGNAL_INCLUSION_PUBLICATION_LAG_DAYS,
    lookback_days: int = DEFAULT_SIGNAL_INCLUSION_LOOKBACK_DAYS,
    min_observations: int = DEFAULT_SIGNAL_INCLUSION_MIN_OBSERVATIONS,
    pressure_threshold: float = DEFAULT_SIGNAL_INCLUSION_PRESSURE_THRESHOLD,
    risk_multiplier: float = DEFAULT_SIGNAL_INCLUSION_RISK_MULTIPLIER,
) -> SignalInclusionRun:
    if macro_data.empty or not macro_catalog:
        return _empty_inclusion_run(base_result)

    pressure = build_macro_category_pressure(
        macro_data,
        macro_catalog,
        base_result.target_weights.index,
        publication_lag_days=publication_lag_days,
        lookback_days=lookback_days,
        min_observations=min_observations,
    )
    if pressure.empty:
        return _empty_inclusion_run(base_result)

    strategy_prices = prices.reindex(index=base_result.target_weights.index)
    target_columns = list(base_result.target_weights.columns)
    strategy_prices = strategy_prices[target_columns].dropna(how="all")
    target_weights = base_result.target_weights.reindex(strategy_prices.index).fillna(0.0)
    pressure = pressure.reindex(strategy_prices.index).ffill()

    results: dict[str, BacktestResult] = {base_strategy_name: base_result}
    calculated_metrics: list[PerformanceMetrics] = [
        calculate_metrics(
            name=base_result.name,
            returns=base_result.returns,
            equity=base_result.equity,
            turnover=base_result.turnover,
            transaction_costs=base_result.transaction_costs,
        )
    ]
    category_rows: list[dict[str, object]] = []

    for category in pressure.columns:
        category_pressure = pressure[category]
        active = category_pressure > pressure_threshold
        usable_days = int(category_pressure.notna().sum())
        active_days = int(active.sum())
        if usable_days < min_observations or active_days == 0:
            category_rows.append(
                _insufficient_row(
                    category,
                    macro_catalog,
                    category_pressure,
                    usable_days=usable_days,
                    active_days=active_days,
                    publication_lag_days=publication_lag_days,
                    pressure_threshold=pressure_threshold,
                    risk_multiplier=risk_multiplier,
                )
            )
            continue

        overlay_name = f"macro_filter_{category}"
        overlay_weights = apply_macro_pressure_overlay(
            target_weights,
            category_pressure,
            defensive_ticker=defensive_ticker,
            pressure_threshold=pressure_threshold,
            risk_multiplier=risk_multiplier,
        )
        overlay_result = run_backtest(
            overlay_name,
            strategy_prices,
            overlay_weights,
            execution,
        )
        results[overlay_name] = overlay_result
        calculated_metrics.append(
            calculate_metrics(
                name=overlay_result.name,
                returns=overlay_result.returns,
                equity=overlay_result.equity,
                turnover=overlay_result.turnover,
                transaction_costs=overlay_result.transaction_costs,
            )
        )

    metrics = metrics_frame(calculated_metrics)
    window_summary = summarize_windows(rolling_window_metrics(results))
    for category in pressure.columns:
        overlay_name = f"macro_filter_{category}"
        if overlay_name not in results:
            continue
        category_rows.append(
            _comparison_row(
                category,
                macro_catalog,
                pressure[category],
                base_name=base_strategy_name,
                overlay_name=overlay_name,
                metrics=metrics,
                window_summary=window_summary,
                publication_lag_days=publication_lag_days,
                pressure_threshold=pressure_threshold,
                risk_multiplier=risk_multiplier,
            )
        )

    summary = pd.DataFrame(category_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["decision_rank", "delta_calmar", "max_drawdown_improvement"],
            ascending=[True, False, False],
        ).drop(columns=["decision_rank"])

    return SignalInclusionRun(
        summary=summary,
        pressure=pressure,
        results=results,
        metrics=metrics.sort_values("calmar", ascending=False),
        window_summary=window_summary,
    )


def build_macro_category_pressure(
    macro_data: pd.DataFrame,
    macro_catalog: tuple[FredSeries, ...],
    market_index: pd.DatetimeIndex,
    *,
    publication_lag_days: int,
    lookback_days: int,
    min_observations: int,
) -> pd.DataFrame:
    catalog_by_category: dict[str, list[FredSeries]] = {}
    for series in macro_catalog:
        catalog_by_category.setdefault(series.category, []).append(series)

    pressure_by_category: dict[str, pd.Series] = {}
    for category, series_group in catalog_by_category.items():
        series_pressures = []
        for metadata in series_group:
            if metadata.series_id not in macro_data.columns:
                continue
            raw_series = macro_data[metadata.series_id].dropna().sort_index()
            if raw_series.empty:
                continue
            pressure = _series_pressure(
                raw_series,
                market_index,
                metadata.risk_polarity,
                publication_lag_days=publication_lag_days,
                lookback_days=lookback_days,
                min_observations=min_observations,
            )
            if pressure.notna().any():
                series_pressures.append(pressure)
        if series_pressures:
            pressure_by_category[category] = pd.concat(series_pressures, axis=1).mean(axis=1)

    if not pressure_by_category:
        return pd.DataFrame(index=market_index)
    return pd.DataFrame(pressure_by_category, index=market_index).sort_index()


def apply_macro_pressure_overlay(
    target_weights: pd.DataFrame,
    pressure: pd.Series,
    *,
    defensive_ticker: str,
    pressure_threshold: float,
    risk_multiplier: float,
) -> pd.DataFrame:
    overlay = target_weights.copy().fillna(0.0)
    if defensive_ticker not in overlay.columns:
        overlay[defensive_ticker] = 0.0

    aligned_pressure = pressure.reindex(overlay.index).ffill()
    active = aligned_pressure > pressure_threshold
    risk_columns = [column for column in overlay.columns if column != defensive_ticker]

    original_risk = overlay.loc[active, risk_columns].sum(axis=1)
    overlay.loc[active, risk_columns] = overlay.loc[active, risk_columns] * risk_multiplier
    new_risk = overlay.loc[active, risk_columns].sum(axis=1)
    freed_weight = (original_risk - new_risk).clip(lower=0.0)
    overlay.loc[active, defensive_ticker] = overlay.loc[active, defensive_ticker] + freed_weight
    return overlay.reindex(columns=target_weights.columns).fillna(0.0)


def _series_pressure(
    raw_series: pd.Series,
    market_index: pd.DatetimeIndex,
    risk_polarity: str,
    *,
    publication_lag_days: int,
    lookback_days: int,
    min_observations: int,
) -> pd.Series:
    available = raw_series.copy()
    available.index = pd.to_datetime(available.index) + pd.Timedelta(days=publication_lag_days)
    available = available.sort_index()
    aligned = available.reindex(market_index).ffill()
    rolling_mean = aligned.rolling(lookback_days, min_periods=min_observations).mean()
    rolling_std = aligned.rolling(lookback_days, min_periods=min_observations).std()
    z_score = (aligned - rolling_mean) / rolling_std.replace(0.0, np.nan)
    if risk_polarity == "risk_on_when_rising":
        z_score = -z_score
    elif risk_polarity not in {"risk_off_when_rising", "risk_on_when_rising"}:
        z_score = z_score * 0.0
    return (z_score.clip(lower=-2.0, upper=2.0) / 2.0).rename(raw_series.name)


def _comparison_row(
    category: str,
    catalog: tuple[FredSeries, ...],
    pressure: pd.Series,
    *,
    base_name: str,
    overlay_name: str,
    metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
    publication_lag_days: int,
    pressure_threshold: float,
    risk_multiplier: float,
) -> dict[str, object]:
    base = metrics.loc[base_name]
    overlay = metrics.loc[overlay_name]
    row = {
        **_common_row_fields(
            category,
            catalog,
            pressure,
            publication_lag_days=publication_lag_days,
            pressure_threshold=pressure_threshold,
            risk_multiplier=risk_multiplier,
        ),
        "test_status": "tested",
        "overlay_name": overlay_name,
        "base_cagr": float(base["cagr"]),
        "overlay_cagr": float(overlay["cagr"]),
        "delta_cagr": float(overlay["cagr"] - base["cagr"]),
        "base_sharpe": float(base["sharpe"]),
        "overlay_sharpe": float(overlay["sharpe"]),
        "delta_sharpe": float(overlay["sharpe"] - base["sharpe"]),
        "base_max_drawdown": float(base["max_drawdown"]),
        "overlay_max_drawdown": float(overlay["max_drawdown"]),
        "max_drawdown_improvement": float(overlay["max_drawdown"] - base["max_drawdown"]),
        "base_calmar": float(base["calmar"]),
        "overlay_calmar": float(overlay["calmar"]),
        "delta_calmar": float(overlay["calmar"] - base["calmar"]),
        "delta_average_turnover": float(overlay["average_turnover"] - base["average_turnover"]),
        "delta_worst_1y_cagr": _window_delta(
            window_summary, base_name, overlay_name, "1y", "worst_cagr"
        ),
        "delta_worst_3y_cagr": _window_delta(
            window_summary, base_name, overlay_name, "3y", "worst_cagr"
        ),
        "delta_worst_5y_cagr": _window_delta(
            window_summary, base_name, overlay_name, "5y", "worst_cagr"
        ),
        "delta_positive_1y_window_rate": _window_delta(
            window_summary,
            base_name,
            overlay_name,
            "1y",
            "positive_window_rate",
        ),
    }
    decision, decision_rank, rationale = _decision(row)
    row["decision"] = decision
    row["decision_rank"] = decision_rank
    row["rationale"] = rationale
    return row


def _insufficient_row(
    category: str,
    catalog: tuple[FredSeries, ...],
    pressure: pd.Series,
    *,
    usable_days: int,
    active_days: int,
    publication_lag_days: int,
    pressure_threshold: float,
    risk_multiplier: float,
) -> dict[str, object]:
    row = {
        **_common_row_fields(
            category,
            catalog,
            pressure,
            publication_lag_days=publication_lag_days,
            pressure_threshold=pressure_threshold,
            risk_multiplier=risk_multiplier,
        ),
        "test_status": "insufficient_signal",
        "overlay_name": "",
        "usable_days": usable_days,
        "active_days": active_days,
        "base_cagr": np.nan,
        "overlay_cagr": np.nan,
        "delta_cagr": np.nan,
        "base_sharpe": np.nan,
        "overlay_sharpe": np.nan,
        "delta_sharpe": np.nan,
        "base_max_drawdown": np.nan,
        "overlay_max_drawdown": np.nan,
        "max_drawdown_improvement": np.nan,
        "base_calmar": np.nan,
        "overlay_calmar": np.nan,
        "delta_calmar": np.nan,
        "delta_average_turnover": np.nan,
        "delta_worst_1y_cagr": np.nan,
        "delta_worst_3y_cagr": np.nan,
        "delta_worst_5y_cagr": np.nan,
        "delta_positive_1y_window_rate": np.nan,
        "decision": "insufficient_history",
        "decision_rank": 4,
        "rationale": "Signal did not produce enough lagged active history to evaluate.",
    }
    return row


def _common_row_fields(
    category: str,
    catalog: tuple[FredSeries, ...],
    pressure: pd.Series,
    *,
    publication_lag_days: int,
    pressure_threshold: float,
    risk_multiplier: float,
) -> dict[str, object]:
    active = pressure > pressure_threshold
    latest_pressure = float(pressure.dropna().iloc[-1]) if pressure.notna().any() else np.nan
    return {
        "signal_group": f"macro:{category}",
        "category": category,
        "series_count": sum(1 for series in catalog if series.category == category),
        "tested_policy": "risk_reduction_only",
        "macro_publication_lag_days": publication_lag_days,
        "revision_safe": False,
        "pressure_threshold": pressure_threshold,
        "risk_multiplier": risk_multiplier,
        "latest_pressure": latest_pressure,
        "latest_pressure_state": _pressure_state(latest_pressure),
        "usable_days": int(pressure.notna().sum()),
        "active_days": int(active.sum()),
        "active_day_rate": float(active.mean()) if len(active) else np.nan,
    }


def _decision(row: dict[str, object]) -> tuple[str, int, str]:
    delta_calmar = _as_float(row["delta_calmar"])
    delta_cagr = _as_float(row["delta_cagr"])
    delta_sharpe = _as_float(row["delta_sharpe"])
    drawdown_improvement = _as_float(row["max_drawdown_improvement"])
    delta_worst_3y = _as_float(row["delta_worst_3y_cagr"])

    if (
        delta_calmar >= 0.05
        and delta_sharpe >= -0.05
        and delta_cagr >= -0.01
        and (drawdown_improvement >= 0.02 or delta_worst_3y >= 0.01)
    ):
        return (
            "paper_candidate",
            1,
            "Improved risk-adjusted performance with limited CAGR drag; test in paper mode before allocation authority.",
        )
    if drawdown_improvement >= 0.02 and delta_cagr >= -0.02:
        return (
            "watch_only",
            2,
            "Reduced drawdown but did not clearly improve enough metrics for allocation authority.",
        )
    if delta_calmar < -0.05 or delta_cagr < -0.02 or drawdown_improvement < -0.01:
        return (
            "reject_for_now",
            3,
            "Hurt CAGR, drawdown, or Calmar enough that it should not affect allocation.",
        )
    return (
        "watch_only",
        2,
        "Mixed incremental value; keep monitoring but do not grant allocation authority.",
    )


def _window_delta(
    window_summary: pd.DataFrame,
    base_name: str,
    overlay_name: str,
    window: str,
    column: str,
) -> float:
    if window_summary.empty:
        return np.nan
    try:
        return float(
            window_summary.loc[(overlay_name, window), column]
            - window_summary.loc[(base_name, window), column]
        )
    except KeyError:
        return np.nan


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return np.nan


def _pressure_state(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value >= 0.65:
        return "risk_pressure"
    if value <= -0.65:
        return "risk_support"
    return "neutral"


def _empty_inclusion_run(base_result: BacktestResult) -> SignalInclusionRun:
    return SignalInclusionRun(
        summary=pd.DataFrame(),
        pressure=pd.DataFrame(),
        results={base_result.name: base_result},
        metrics=pd.DataFrame(),
        window_summary=pd.DataFrame(),
    )
