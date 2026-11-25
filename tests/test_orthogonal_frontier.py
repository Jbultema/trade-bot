from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.research.orthogonal_frontier import (
    _event_holding_mask,
    _metric_row,
    _threshold_rebalance_weights,
    point_in_time_market_states,
)


def test_point_in_time_market_states_detects_damage_without_future_data() -> None:
    index = pd.bdate_range("2018-01-02", periods=1_100)
    calm = np.linspace(100.0, 160.0, 850)
    break_path = np.linspace(160.0, 112.0, 80)
    recovery = np.linspace(112.0, 145.0, 170)
    spy = pd.Series(np.concatenate([calm, break_path, recovery]), index=index)

    states = point_in_time_market_states(spy)

    assert states.iloc[:700].eq("normal").mean() > 0.80
    assert states.iloc[850:950].eq("break").any()


def test_event_holding_mask_does_not_activate_before_event() -> None:
    index = pd.bdate_range("2026-01-02", periods=10)
    events = pd.Series(False, index=index)
    events.iloc[4] = True

    active = _event_holding_mask(events, 3)

    assert not active.iloc[:4].any()
    assert active.iloc[4:7].all()
    assert not active.iloc[7:].any()


def test_threshold_rebalance_weights_remain_long_only_and_fully_invested() -> None:
    index = pd.bdate_range("2020-01-02", periods=100)
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 140.0, len(index)),
            "IEF": np.linspace(100.0, 95.0, len(index)),
            "GLD": np.linspace(100.0, 110.0, len(index)),
            "BIL": np.linspace(100.0, 101.0, len(index)),
        },
        index=index,
    )
    allocation = {"SPY": 0.55, "IEF": 0.25, "GLD": 0.15, "BIL": 0.05}

    weights = _threshold_rebalance_weights(prices, allocation, 0.03)

    assert weights.ge(0.0).all(axis=None)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_metric_row_reports_drawdown_and_positive_cagr() -> None:
    index = pd.bdate_range("2020-01-02", periods=600)
    returns = pd.Series(0.0004, index=index)
    returns.iloc[300] = -0.20

    metrics = _metric_row(returns)

    assert float(metrics["cagr"]) > 0.0
    assert float(metrics["max_drawdown"]) <= -0.1999
