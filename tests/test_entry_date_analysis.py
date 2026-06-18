from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.entry_date_analysis import build_entry_date_analysis


def test_entry_date_analysis_summarizes_start_date_sensitivity() -> None:
    index = pd.bdate_range("2020-01-01", periods=320)
    benchmark_returns = pd.Series(0.001, index=index)
    strategy_returns = pd.Series(0.002, index=index)
    results = {
        "buy_hold_spy": _result("buy_hold_spy", benchmark_returns),
        "better_strategy": _result("better_strategy", strategy_returns),
    }

    analysis = build_entry_date_analysis(
        results,
        benchmarks=("buy_hold_spy",),
        horizons={"3m": 63},
        start_frequency="M",
    )

    assert not analysis.windows.empty
    strategy_summary = analysis.summary[
        (analysis.summary["strategy"] == "better_strategy")
        & (analysis.summary["benchmark"] == "buy_hold_spy")
        & (analysis.summary["horizon"] == "3m")
    ].iloc[0]
    assert strategy_summary["windows"] > 0
    assert strategy_summary["beat_rate"] == 1.0
    assert strategy_summary["median_excess_return"] > 0


def _result(name: str, returns: pd.Series) -> BacktestResult:
    equity = 100.0 * (1.0 + returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=returns.rename(name),
        gross_returns=returns.rename(name),
        weights=pd.DataFrame({"SPY": 1.0}, index=returns.index),
        target_weights=pd.DataFrame({"SPY": 1.0}, index=returns.index),
        turnover=pd.Series(0.0, index=returns.index, name=name),
        transaction_costs=pd.Series(0.0, index=returns.index, name=name),
    )
