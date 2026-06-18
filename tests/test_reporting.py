from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.reporting.report import make_equity_drawdown_figure, window_performance_frame


def test_windowed_equity_figure_rebases_growth_of_one() -> None:
    index = pd.bdate_range("2026-01-01", periods=5)
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 132.0], index=index)
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    result = BacktestResult(
        name="strategy_a",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=pd.DataFrame({"SPY": 1.0}, index=index),
        target_weights=pd.DataFrame({"SPY": 1.0}, index=index),
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )

    figure = make_equity_drawdown_figure(
        {"strategy_a": result},
        start=index[2],
        end=index[-1],
        rebase=True,
    )
    stats = window_performance_frame(
        {"strategy_a": result},
        start=index[2],
        end=index[-1],
    )

    assert float(figure.data[0].y[0]) == 1.0
    assert round(float(figure.data[0].y[-1]), 6) == round(132.0 / 105.0, 6)
    assert round(float(stats.loc[0, "total_return"]), 6) == round(132.0 / 105.0 - 1.0, 6)
    assert float(stats.loc[0, "max_drawdown"]) == 0.0


def test_window_performance_stats_exclude_pre_window_boundary_return() -> None:
    index = pd.bdate_range("2026-01-01", periods=4)
    equity = pd.Series([100.0, 50.0, 100.0, 110.0], index=index)
    stored_returns = equity.pct_change(fill_method=None).fillna(0.0)
    result = BacktestResult(
        name="strategy_a",
        equity=equity,
        returns=stored_returns,
        gross_returns=stored_returns,
        weights=pd.DataFrame({"SPY": 1.0}, index=index),
        target_weights=pd.DataFrame({"SPY": 1.0}, index=index),
        turnover=pd.Series([0.0, 0.5, 0.5, 0.1], index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )

    stats = window_performance_frame(
        {"strategy_a": result},
        start=index[2],
        end=index[-1],
    )

    assert round(float(stats.loc[0, "total_return"]), 6) == round(110.0 / 100.0 - 1.0, 6)
    assert stats.loc[0, "best_day"] == pytest.approx(0.10)
    assert stats.loc[0, "worst_day"] == pytest.approx(0.0)
    assert stats.loc[0, "average_turnover"] == pytest.approx(0.05)
