"""Native staged-risk causality, initialization and missing-evidence checks."""

import numpy as np
import pandas as pd
import pytest

from trade_bot.config import StrategyConfig, required_strategy_tickers
from trade_bot.strategies.momentum import build_strategy_weights
from trade_bot.strategies.staged_risk import size_native_weights, staged_budget


def observations(n=400):
    index = pd.bdate_range("2023-01-02", periods=n)
    market = np.exp(np.linspace(4.6, 5.1, n))
    return pd.DataFrame({name: market.copy() for name in ("SPY", "QQQ", "RSP", "HYG", "LQD")}, index=index)


@pytest.mark.parametrize("mode", ["gentle", "confirmed", "repair"])
@pytest.mark.parametrize("lookback", [126, 252])
def test_causal_prefix_and_funded_budget(mode, lookback):
    prices = observations()
    first = staged_budget(prices, mode=mode, lookback=lookback)
    prices.iloc[350:] *= .6
    second = staged_budget(prices, mode=mode, lookback=lookback)
    pd.testing.assert_series_equal(first.iloc[:350], second.iloc[:350])
    assert second.between(0, 1).all()
    assert second.iloc[:125].eq(0).all()


def test_missing_evidence_cannot_be_read_as_benign_credit():
    prices = observations()
    prices.iloc[300, prices.columns.get_loc("HYG")] = np.nan
    with pytest.raises(ValueError, match="Missing staged-risk evidence"):
        staged_budget(prices)
    with pytest.raises(ValueError, match="requires"):
        staged_budget(prices.drop(columns="RSP"))


def test_multiple_late_confirmations_required_for_hard_defense():
    prices = observations()
    prices.iloc[-1, prices.columns.get_loc("SPY")] *= .88
    # Market damage alone does not prove a credit/participation unwind.
    assert staged_budget(prices).iloc[-1] == .65
    prices.iloc[-1, prices.columns.get_loc("QQQ")] *= .8
    prices.iloc[-1, prices.columns.get_loc("HYG")] *= .8
    assert staged_budget(prices).iloc[-1] == .35


def test_native_budget_preserves_cash_and_releases_only_risky_allocation():
    prices = observations()
    prices["BIL"] = 100.
    weights = pd.DataFrame({"QQQ": .6, "BIL": .4}, index=prices.index)
    result = size_native_weights(prices, weights, "BIL")
    assert result.sum(axis=1).eq(1).all()
    assert result.iloc[0].tolist() == [0., 1.]
    assert result.iloc[-1].tolist() == pytest.approx([.6, .4])


def test_native_strategy_is_registered_and_requests_its_actual_signal_inputs():
    prices = observations()
    prices["BIL"] = 100.
    strategy = StrategyConfig(type="staged_dual_momentum", tickers=["QQQ"], defensive_ticker="BIL", trend_filter_days=None)
    assert set(required_strategy_tickers(strategy)) >= {"QQQ", "BIL", "SPY", "HYG", "LQD", "RSP"}
    result = build_strategy_weights(prices, strategy)
    assert result.sum(axis=1).le(1.00000001).all()
    assert result.QQQ.iloc[:125].eq(0).all()
