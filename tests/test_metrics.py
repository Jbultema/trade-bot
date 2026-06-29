from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.features.indicators import ulcer_index


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


def test_ulcer_index_weights_deep_persistent_drawdowns() -> None:
    equity = pd.Series([100.0, 110.0, 104.5, 99.0])
    # Drawdowns are 0%, 0%, -5%, -10%; ulcer index is RMS drawdown.
    expected = ((0.0**2 + 0.0**2 + 0.05**2 + 0.10**2) / 4) ** 0.5

    assert ulcer_index(equity) == pytest.approx(expected)
