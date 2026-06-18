from __future__ import annotations

import pandas as pd

from trade_bot.backtest.metrics import calculate_metrics


def test_calculate_metrics_reports_drawdown_and_turnover() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.Series([0.0, 0.10, -0.05], index=index)
    equity = 100.0 * (1.0 + returns).cumprod()
    turnover = pd.Series([0.0, 0.25, 0.50], index=index)
    costs = pd.Series([0.0, 0.001, 0.002], index=index)

    metrics = calculate_metrics("demo", returns, equity, turnover, costs)

    assert metrics.name == "demo"
    assert round(metrics.final_equity, 2) == 104.50
    assert round(metrics.max_drawdown, 4) == -0.05
    assert round(metrics.average_turnover, 4) == 0.25
    assert round(metrics.total_transaction_cost, 4) == 0.003
