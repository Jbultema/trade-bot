from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.DEFAULTS import (
    DEFAULT_CALENDAR_YEAR_MIN_OBSERVATIONS,
    DEFAULT_REGIME_MIN_OBSERVATIONS,
    DEFAULT_REGIMES,
    DEFAULT_ROLLING_STEP_MONTHS,
    DEFAULT_ROLLING_WINDOW_YEARS,
    DEFAULT_WALK_FORWARD_STEP_MONTHS,
    DEFAULT_WALK_FORWARD_TEST_YEARS,
    DEFAULT_WALK_FORWARD_TRAIN_YEARS,
    DEFAULT_WINDOW_MIN_OBSERVATION_RATIO,
    RegimeDefinition,
)


@dataclass(frozen=True)
class WindowMetrics:
    name: str
    window: str
    start: str
    end: str
    observations: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    average_turnover: float


def rolling_window_metrics(
    results: dict[str, BacktestResult],
    *,
    window_years: list[int] | None = None,
    step_months: int = DEFAULT_ROLLING_STEP_MONTHS,
    min_observation_ratio: float = DEFAULT_WINDOW_MIN_OBSERVATION_RATIO,
) -> pd.DataFrame:
    windows = window_years or list(DEFAULT_ROLLING_WINDOW_YEARS)
    metrics: list[WindowMetrics] = []
    for result in results.values():
        for years in windows:
            metrics.extend(
                _rolling_result_windows(
                    result,
                    window_years=years,
                    step_months=step_months,
                    min_observation_ratio=min_observation_ratio,
                )
            )
    return _window_frame(metrics)


def calendar_year_metrics(results: dict[str, BacktestResult]) -> pd.DataFrame:
    metrics: list[WindowMetrics] = []
    for result in results.values():
        for year, returns in result.returns.groupby(result.returns.index.year):
            if len(returns) < DEFAULT_CALENDAR_YEAR_MIN_OBSERVATIONS:
                continue
            metrics.append(_window_metrics(result, returns.index, window=str(year)))
    return _window_frame(metrics)


def regime_window_metrics(
    results: dict[str, BacktestResult],
    *,
    regimes: tuple[RegimeDefinition, ...] = DEFAULT_REGIMES,
    min_observations: int = DEFAULT_REGIME_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results.values():
        returns = result.returns.dropna()
        if returns.empty:
            continue
        for regime in regimes:
            start = pd.Timestamp(regime.start)
            end = min(pd.Timestamp(regime.end), returns.index.max())
            window_index = returns.loc[(returns.index >= start) & (returns.index <= end)].index
            if len(window_index) < min_observations:
                continue
            metrics = _window_metrics(result, window_index, window=regime.name)
            row = asdict(metrics)
            row["regime"] = regime.name
            row["regime_type"] = regime.regime_type
            row["description"] = regime.description
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["regime", "name"])


def walk_forward_holdout_metrics(
    results: dict[str, BacktestResult],
    *,
    train_years: int = DEFAULT_WALK_FORWARD_TRAIN_YEARS,
    test_years: int = DEFAULT_WALK_FORWARD_TEST_YEARS,
    step_months: int = DEFAULT_WALK_FORWARD_STEP_MONTHS,
    min_observation_ratio: float = DEFAULT_WINDOW_MIN_OBSERVATION_RATIO,
) -> pd.DataFrame:
    """Evaluate fixed strategy results on sequential later windows.

    This compatibility name is retained for existing artifacts. The function
    does not train, tune, or select a strategy in the preceding segment; callers
    must not present it as nested walk-forward optimization.
    """
    rows: list[dict[str, object]] = []
    for result in results.values():
        rows.extend(
            _walk_forward_result_folds(
                result,
                train_years=train_years,
                test_years=test_years,
                step_months=step_months,
                min_observation_ratio=min_observation_ratio,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fold", "name"])


def summarize_walk_forward(holdout_metrics: pd.DataFrame) -> pd.DataFrame:
    if holdout_metrics.empty:
        return pd.DataFrame()

    grouped = holdout_metrics.groupby("name", observed=True)
    return grouped.agg(
        holdout_folds=("fold", "count"),
        walk_forward_median_cagr=("cagr", "median"),
        walk_forward_worst_cagr=("cagr", "min"),
        walk_forward_positive_rate=("total_return", lambda values: float((values > 0).mean())),
        walk_forward_median_calmar=("calmar", "median"),
        walk_forward_worst_drawdown=("max_drawdown", "min"),
    ).sort_values("walk_forward_median_calmar", ascending=False)


def summarize_regimes(regime_metrics: pd.DataFrame) -> pd.DataFrame:
    if regime_metrics.empty:
        return pd.DataFrame()

    grouped = regime_metrics.groupby("name", observed=True)
    summary = grouped.agg(
        tested_regimes=("regime", "count"),
        worst_regime_return=("total_return", "min"),
        median_regime_return=("total_return", "median"),
        worst_regime_cagr=("cagr", "min"),
        median_regime_cagr=("cagr", "median"),
        worst_regime_drawdown=("max_drawdown", "min"),
        regime_positive_rate=("total_return", lambda values: float((values > 0).mean())),
    )

    left_tail = regime_metrics[regime_metrics["regime_type"] == "left_tail"]
    transition = regime_metrics[regime_metrics["regime_type"].isin(["transition", "left_tail"])]
    if not left_tail.empty:
        summary["left_tail_regime_cagr"] = left_tail.groupby("name")["cagr"].min()
        summary["left_tail_regime_return"] = left_tail.groupby("name")["total_return"].min()
    else:
        summary["left_tail_regime_cagr"] = float("nan")
        summary["left_tail_regime_return"] = float("nan")
    if not transition.empty:
        summary["transition_regime_hit_rate"] = transition.groupby("name")["total_return"].apply(
            lambda values: float((values > 0).mean())
        )
        summary["transition_regime_return"] = transition.groupby("name")["total_return"].min()
    else:
        summary["transition_regime_hit_rate"] = float("nan")
        summary["transition_regime_return"] = float("nan")
    return summary.sort_values("worst_regime_cagr", ascending=False)


def summarize_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    if window_metrics.empty:
        return pd.DataFrame()

    grouped = window_metrics.groupby(["name", "window"], observed=True)
    summary = grouped.agg(
        windows=("cagr", "count"),
        median_cagr=("cagr", "median"),
        worst_cagr=("cagr", "min"),
        best_cagr=("cagr", "max"),
        positive_window_rate=("total_return", lambda values: float((values > 0).mean())),
        median_sharpe=("sharpe", "median"),
        worst_drawdown=("max_drawdown", "min"),
        median_calmar=("calmar", "median"),
        average_turnover=("average_turnover", "median"),
    )
    return summary.sort_values(["window", "median_calmar"], ascending=[True, False])


def calendar_return_pivot(calendar_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    if calendar_metrics_frame.empty:
        return pd.DataFrame()
    return calendar_metrics_frame.pivot(
        index="window", columns="name", values="total_return"
    ).sort_index()


def _rolling_result_windows(
    result: BacktestResult,
    *,
    window_years: int,
    step_months: int,
    min_observation_ratio: float,
) -> list[WindowMetrics]:
    returns = result.returns.dropna()
    if returns.empty:
        return []

    min_observations = int(252 * window_years * min_observation_ratio)
    month_end_labels = returns.resample(f"{step_months}ME").last().dropna().index
    metrics: list[WindowMetrics] = []
    for end_label in month_end_labels:
        end_position = returns.index.searchsorted(end_label, side="right") - 1
        if end_position < 0:
            continue
        end_date = returns.index[end_position]
        start_boundary = end_date - pd.DateOffset(years=window_years)
        start_position = returns.index.searchsorted(start_boundary, side="left")
        window_index = returns.index[start_position : end_position + 1]
        if len(window_index) < min_observations:
            continue
        metrics.append(_window_metrics(result, window_index, window=f"{window_years}y"))
    return metrics


def _walk_forward_result_folds(
    result: BacktestResult,
    *,
    train_years: int,
    test_years: int,
    step_months: int,
    min_observation_ratio: float,
) -> list[dict[str, object]]:
    returns = result.returns.dropna()
    if returns.empty:
        return []

    min_observations = int(252 * test_years * min_observation_ratio)
    first_test_end = returns.index.min() + pd.DateOffset(years=train_years + test_years)
    fold_end_labels = (
        returns.loc[returns.index >= first_test_end].resample(f"{step_months}ME").last()
    )
    fold_end_index = fold_end_labels.dropna().index
    rows: list[dict[str, object]] = []
    for fold_number, end_label in enumerate(fold_end_index, start=1):
        end_position = returns.index.searchsorted(end_label, side="right") - 1
        if end_position < 0:
            continue
        test_end = returns.index[end_position]
        test_start_boundary = test_end - pd.DateOffset(years=test_years)
        train_start_boundary = test_start_boundary - pd.DateOffset(years=train_years)
        test_start_position = returns.index.searchsorted(test_start_boundary, side="left")
        train_start_position = returns.index.searchsorted(train_start_boundary, side="left")
        train_end_position = max(test_start_position - 1, train_start_position)
        test_index = returns.index[test_start_position : end_position + 1]
        if len(test_index) < min_observations:
            continue
        metrics = _window_metrics(result, test_index, window=f"{test_years}y_holdout")
        row = asdict(metrics)
        row["fold"] = fold_number
        row["train_start"] = str(returns.index[train_start_position].date())
        row["train_end"] = str(returns.index[train_end_position].date())
        row["test_start"] = metrics.start
        row["test_end"] = metrics.end
        row["evaluation_method"] = "sequential_fixed_strategy_holdout"
        row["selection_performed"] = False
        rows.append(row)
    return rows


def _window_metrics(
    result: BacktestResult,
    window_index: pd.DatetimeIndex,
    *,
    window: str,
) -> WindowMetrics:
    returns = result.returns.loc[window_index]
    equity = 100.0 * (1.0 + returns).cumprod()
    metrics = calculate_metrics(
        result.name,
        returns,
        equity,
        result.turnover.loc[window_index],
        result.transaction_costs.loc[window_index],
    )
    return WindowMetrics(
        name=result.name,
        window=window,
        start=metrics.start,
        end=metrics.end,
        observations=len(returns),
        total_return=float((1.0 + returns).prod() - 1.0),
        cagr=metrics.cagr,
        sharpe=metrics.sharpe,
        sortino=metrics.sortino,
        max_drawdown=metrics.max_drawdown,
        calmar=metrics.calmar,
        average_turnover=metrics.average_turnover,
    )


def _window_frame(metrics: list[WindowMetrics]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame([asdict(metric) for metric in metrics])
