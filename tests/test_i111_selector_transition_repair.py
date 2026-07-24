from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import StrategyConfig
from trade_bot.research.drawdown_attribution import build_drawdown_attribution
from trade_bot.research.i111_selector_transition_repair import (
    SelectorTransitionSpec,
    _buffer_selection,
    build_repaired_target_weights,
    build_selector_features,
)


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["A", "B", "C"],
        lookback_days=63,
        skip_days=5,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.0,
        volatility_lookback_days=63,
    )


def _features(
    index: pd.DatetimeIndex,
    *,
    recovered: bool = True,
    market_confirmed: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    rank = pd.DataFrame(
        [[1.0, 0.8, 0.2]] * len(index),
        index=index,
        columns=["A", "B", "C"],
    )
    momentum = pd.DataFrame(
        [[0.10, 0.08, 0.02]] * len(index),
        index=index,
        columns=["A", "B", "C"],
    )
    return {
        "short_momentum": momentum,
        "short_percentile": rank,
        "blended_percentile": rank,
        "recovery_confirmed": pd.DataFrame(
            recovered,
            index=index,
            columns=["A", "B", "C"],
        ),
        "market_confirmed": pd.Series(market_confirmed, index=index),
    }


def test_selector_features_do_not_change_before_mutated_future_price() -> None:
    index = pd.bdate_range("2020-01-01", periods=220)
    base = np.linspace(100.0, 180.0, len(index))
    prices = pd.DataFrame(
        {
            "A": base,
            "B": base * 0.9 + np.sin(np.arange(len(index))),
            "C": base * 1.1 - np.cos(np.arange(len(index))),
            "SPY": base * 0.8,
        },
        index=index,
    )
    changed = prices.copy()
    changed.loc[index[-1], "A"] *= 2.0

    original_features = build_selector_features(prices, _strategy())
    changed_features = build_selector_features(changed, _strategy())

    for key in original_features:
        left = original_features[key]
        right = changed_features[key]
        if isinstance(left, pd.DataFrame):
            pdt.assert_frame_equal(left.iloc[:-1], right.iloc[:-1])
        else:
            pdt.assert_series_equal(left.iloc[:-1], right.iloc[:-1])


def test_incumbent_buffer_keeps_near_winner_but_not_clear_loser() -> None:
    desired = ["A", "B"]
    previous = {"A", "C"}
    momentum = pd.Series({"A": 0.10, "B": 0.08, "C": 0.05})

    kept, kept_count, blocked = _buffer_selection(
        desired,
        previous,
        pd.Series({"A": 1.0, "B": 0.80, "C": 0.70}),
        momentum,
        top_n=2,
        min_return=0.0,
    )
    assert set(kept) == {"A", "C"}
    assert kept_count == blocked == 1

    replaced, kept_count, blocked = _buffer_selection(
        desired,
        previous,
        pd.Series({"A": 1.0, "B": 0.90, "C": 0.60}),
        momentum,
        top_n=2,
        min_return=0.0,
    )
    assert set(replaced) == {"A", "B"}
    assert kept_count == blocked == 0


def test_native_selector_repair_preserves_existing_diversifier_sleeve() -> None:
    index = pd.bdate_range("2024-01-02", periods=2)
    raw = pd.DataFrame(
        {
            "A": [0.4, 0.4],
            "B": [0.2, 0.2],
            "C": [0.0, 0.0],
            "SPY": [0.1, 0.1],
            "RSP": [0.0, 0.0],
            "GLD": [0.0, 0.0],
            "TLT": [0.0, 0.0],
            "BIL": [0.3, 0.3],
        },
        index=index,
    )
    repaired, _ = build_repaired_target_weights(
        raw,
        _features(index),
        _strategy(),
        SelectorTransitionSpec(name="identity", description="identity"),
        rebalance="D",
    )
    pdt.assert_frame_equal(
        repaired[raw.columns],
        raw.astype(float),
        check_freq=False,
    )


def test_core_and_recovery_meter_keep_risk_budget_and_bridge_entry() -> None:
    index = pd.bdate_range("2024-01-02", periods=2)
    raw = pd.DataFrame(
        {
            "A": [0.4, 0.1],
            "B": [0.2, 0.5],
            "C": [0.0, 0.0],
            "SPY": [0.1, 0.1],
            "RSP": [0.0, 0.0],
            "BIL": [0.3, 0.3],
        },
        index=index,
    )
    repaired, diagnostics = build_repaired_target_weights(
        raw,
        _features(index, recovered=False, market_confirmed=False),
        _strategy(),
        SelectorTransitionSpec(
            name="meter",
            description="meter",
            recovery_meter=True,
        ),
        rebalance="D",
    )

    assert repaired.loc[index[0], "B"] == pytest.approx(0.2)
    assert repaired.loc[index[1], "B"] == pytest.approx(0.35)
    assert repaired.loc[index[1], "SPY"] == pytest.approx(0.175)
    assert repaired.loc[index[1], "RSP"] == pytest.approx(0.075)
    assert repaired.loc[index[1], "BIL"] == pytest.approx(0.3)
    assert repaired.loc[index[1]].sum() == pytest.approx(1.0)
    assert diagnostics.loc[1, "deferred_entry_weight"] == pytest.approx(0.15)


def test_drawdown_attribution_reports_peak_contributors_and_recovery_gap() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    equity = pd.Series([100.0, 110.0, 88.0, 99.0], index=index)
    returns = equity.pct_change().fillna(0.0)
    weights = pd.DataFrame(
        {"QQQ": 1.0, "BIL": 0.0},
        index=index,
    )
    result = BacktestResult(
        name="test",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series([1.0, 0.0, 0.0, 0.0], index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 110.0, 88.0, 99.0],
            "BIL": [100.0, 100.0, 100.0, 100.0],
            "SPY": [100.0, 105.0, 100.0, 110.0],
        },
        index=index,
    )

    attribution = build_drawdown_attribution(
        result,
        prices,
        benchmarks=("SPY",),
    )
    summary = attribution.summary.iloc[0]
    assert summary["peak_date"] == index[1]
    assert summary["trough_date"] == index[2]
    assert summary["max_drawdown"] == pytest.approx(-0.20)
    assert attribution.contributors.iloc[0]["asset"] == "QQQ"
    assert attribution.contributors.iloc[0]["gross_return_contribution"] == pytest.approx(-0.20)
    assert summary["missed_spy_recovery"] == pytest.approx(0.10 - 0.125)
