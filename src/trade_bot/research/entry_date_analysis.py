from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.DEFAULTS import DEFAULT_ENTRY_HORIZONS


@dataclass(frozen=True)
class EntryDateAnalysis:
    windows: pd.DataFrame
    summary: pd.DataFrame


def build_entry_date_analysis(
    results: Mapping[str, BacktestResult],
    *,
    benchmarks: tuple[str, ...] = ("buy_hold_spy", "buy_hold_qqq"),
    horizons: Mapping[str, int] = DEFAULT_ENTRY_HORIZONS,
    start_frequency: str = "ME",
) -> EntryDateAnalysis:
    if not results:
        return EntryDateAnalysis(windows=pd.DataFrame(), summary=pd.DataFrame())

    rows: list[dict[str, object]] = []
    common_index = _common_index(results)
    if common_index.empty:
        return EntryDateAnalysis(windows=pd.DataFrame(), summary=pd.DataFrame())

    start_dates = _sample_start_dates(common_index, start_frequency)
    for strategy_name, result in results.items():
        strategy_equity = result.equity.reindex(common_index).dropna()
        for horizon_name, horizon_days in horizons.items():
            for start_date in start_dates:
                if start_date not in strategy_equity.index:
                    continue
                start_position = strategy_equity.index.get_loc(start_date)
                if not isinstance(start_position, int):
                    continue
                end_position = start_position + int(horizon_days)
                if end_position >= len(strategy_equity):
                    continue
                end_date = strategy_equity.index[end_position]
                strategy_slice = strategy_equity.iloc[start_position : end_position + 1]
                strategy_return = _total_return(strategy_slice)
                strategy_cagr = _window_cagr(
                    strategy_return,
                    strategy_slice.index[0],
                    strategy_slice.index[-1],
                )
                strategy_drawdown = _window_drawdown(strategy_slice)
                for benchmark_name in benchmarks:
                    if benchmark_name not in results:
                        continue
                    benchmark_equity = results[benchmark_name].equity.reindex(strategy_slice.index)
                    if benchmark_equity.isna().any():
                        continue
                    benchmark_return = _total_return(benchmark_equity)
                    rows.append(
                        {
                            "strategy": strategy_name,
                            "benchmark": benchmark_name,
                            "horizon": horizon_name,
                            "horizon_trading_days": int(horizon_days),
                            "start_date": start_date.date().isoformat(),
                            "end_date": end_date.date().isoformat(),
                            "total_return": strategy_return,
                            "cagr": strategy_cagr,
                            "max_drawdown": strategy_drawdown,
                            "benchmark_return": benchmark_return,
                            "excess_return": strategy_return - benchmark_return,
                            "beats_benchmark": strategy_return > benchmark_return,
                            "positive_return": strategy_return > 0,
                        }
                    )

    windows = pd.DataFrame(rows)
    return EntryDateAnalysis(windows=windows, summary=summarize_entry_date_windows(windows))


def summarize_entry_date_windows(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    grouped = windows.groupby(["strategy", "benchmark", "horizon"], as_index=False)
    summary = grouped.agg(
        windows=("start_date", "count"),
        beat_rate=("beats_benchmark", "mean"),
        positive_return_rate=("positive_return", "mean"),
        median_return=("total_return", "median"),
        worst_return=("total_return", "min"),
        median_cagr=("cagr", "median"),
        median_excess_return=("excess_return", "median"),
        worst_excess_return=("excess_return", "min"),
        median_max_drawdown=("max_drawdown", "median"),
        worst_max_drawdown=("max_drawdown", "min"),
    )
    return summary.sort_values(
        ["benchmark", "horizon", "beat_rate", "median_excess_return"],
        ascending=[True, True, False, False],
    )


def _common_index(results: Mapping[str, BacktestResult]) -> pd.DatetimeIndex:
    indexes = [
        result.equity.dropna().index for result in results.values() if not result.equity.empty
    ]
    if not indexes:
        return pd.DatetimeIndex([])
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    return pd.DatetimeIndex(common).sort_values()


def _sample_start_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if index.empty:
        return index
    periods = index.to_period(frequency)
    first_dates = pd.Series(index, index=index).groupby(periods).first()
    return pd.DatetimeIndex(first_dates.to_list())


def _total_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start = float(equity.iloc[0])
    if abs(start) < 1e-12:
        return 0.0
    return float(equity.iloc[-1] / start - 1.0)


def _window_cagr(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = max((end - start).days / 365.25, 1 / 365.25)
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _window_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    normalized = equity / float(equity.iloc[0])
    return float((normalized / normalized.cummax() - 1.0).min())
