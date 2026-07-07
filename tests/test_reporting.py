from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.features.indicators import TRADING_DAYS_PER_YEAR
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

    window_equity = equity.loc[index[2] : index[-1]] / equity.loc[index[2]]
    window_returns = window_equity.pct_change(fill_method=None).fillna(0.0)
    expected_years = (index[-1] - index[2]).days / 365.25
    expected_cagr = float(window_equity.iloc[-1] ** (1.0 / expected_years) - 1.0)
    expected_volatility = float(window_returns.std() * (TRADING_DAYS_PER_YEAR**0.5))
    expected_sharpe = float(window_returns.mean() * TRADING_DAYS_PER_YEAR / expected_volatility)

    assert float(figure.data[0].y[0]) == 1.0
    assert round(float(figure.data[0].y[-1]), 6) == round(132.0 / 105.0, 6)
    assert round(float(stats.loc[0, "total_return"]), 6) == round(132.0 / 105.0 - 1.0, 6)
    assert stats.loc[0, "years"] == pytest.approx(expected_years)
    assert stats.loc[0, "cagr"] == pytest.approx(expected_cagr)
    assert stats.loc[0, "annualized_volatility"] == pytest.approx(expected_volatility)
    assert stats.loc[0, "sharpe"] == pytest.approx(expected_sharpe)
    assert float(stats.loc[0, "max_drawdown"]) == 0.0
    assert stats.loc[0, "calmar"] == 0.0


def test_equity_drawdown_figure_keeps_series_colors_aligned() -> None:
    index = pd.bdate_range("2026-01-01", periods=5)
    equity_a = pd.Series([100.0, 104.0, 102.0, 108.0, 112.0], index=index)
    equity_b = pd.Series([100.0, 101.0, 99.0, 103.0, 106.0], index=index)

    def _result(name: str, equity: pd.Series) -> BacktestResult:
        returns = equity.pct_change(fill_method=None).fillna(0.0)
        return BacktestResult(
            name=name,
            equity=equity,
            returns=returns,
            gross_returns=returns,
            weights=pd.DataFrame({"SPY": 1.0}, index=index),
            target_weights=pd.DataFrame({"SPY": 1.0}, index=index),
            turnover=pd.Series(0.0, index=index),
            transaction_costs=pd.Series(0.0, index=index),
        )

    figure = make_equity_drawdown_figure(
        {
            "buy_hold_spy": _result("buy_hold_spy", equity_a),
            "custom_strategy": _result("custom_strategy", equity_b),
        }
    )

    assert figure.data[0].name == "buy_hold_spy"
    assert figure.data[1].name == "buy_hold_spy drawdown"
    assert figure.data[0].line.color == figure.data[1].line.color
    assert figure.data[0].legendgroup == figure.data[1].legendgroup == "buy_hold_spy"
    assert figure.data[2].name == "custom_strategy"
    assert figure.data[3].name == "custom_strategy drawdown"
    assert figure.data[2].line.color == figure.data[3].line.color
    assert figure.data[2].legendgroup == figure.data[3].legendgroup == "custom_strategy"
    assert figure.data[0].line.color != figure.data[2].line.color


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


def test_window_performance_stats_are_finite_for_single_observation_window() -> None:
    index = pd.bdate_range("2026-01-01", periods=3)
    equity = pd.Series([100.0, 105.0, 110.0], index=index)
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

    stats = window_performance_frame(
        {"strategy_a": result},
        start=index[-1],
        end=index[-1],
    )

    assert stats.loc[0, "total_return"] == 0.0
    assert stats.loc[0, "cagr"] == 0.0
    assert stats.loc[0, "annualized_volatility"] == 0.0
    assert stats.loc[0, "sharpe"] == 0.0
    assert stats.loc[0, "calmar"] == 0.0
