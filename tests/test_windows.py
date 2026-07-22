from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.windows import (
    calendar_year_metrics,
    rolling_window_metrics,
    summarize_windows,
    walk_forward_holdout_metrics,
)


def test_calendar_year_metrics_reports_shorter_periods() -> None:
    index = pd.bdate_range("2020-01-01", "2021-12-31")
    returns = pd.Series(0.001, index=index, name="demo")
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    turnover = pd.Series(0.0, index=index)
    costs = pd.Series(0.0, index=index)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=turnover,
        transaction_costs=costs,
    )

    metrics = calendar_year_metrics({"demo": result})

    assert metrics["window"].tolist() == ["2020", "2021"]
    assert metrics["name"].eq("demo").all()
    assert metrics["total_return"].gt(0.0).all()


def test_rolling_window_summary_identifies_positive_windows() -> None:
    index = pd.bdate_range("2020-01-01", "2022-12-31")
    returns = pd.Series(0.001, index=index, name="demo")
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    turnover = pd.Series(0.0, index=index)
    costs = pd.Series(0.0, index=index)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=turnover,
        transaction_costs=costs,
    )

    windows = rolling_window_metrics({"demo": result}, window_years=[1], step_months=3)
    summary = summarize_windows(windows)

    assert not windows.empty
    assert ("demo", "1y") in summary.index
    assert summary.loc[("demo", "1y"), "positive_window_rate"] == 1.0


def test_legacy_walk_forward_name_labels_fixed_strategy_holdout_semantics() -> None:
    index = pd.bdate_range("2015-01-01", "2024-12-31")
    returns = pd.Series(0.0005, index=index, name="demo")
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    result = BacktestResult(
        name="demo",
        equity=100.0 * (1.0 + returns).cumprod(),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )

    holdouts = walk_forward_holdout_metrics({"demo": result})

    assert not holdouts.empty
    assert holdouts["evaluation_method"].eq("sequential_fixed_strategy_holdout").all()
    assert holdouts["selection_performed"].eq(False).all()  # noqa: E712
