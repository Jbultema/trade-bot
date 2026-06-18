from __future__ import annotations

import pandas as pd

from trade_bot.strategies.momentum import dual_momentum_weights


def test_dual_momentum_uses_defensive_asset_when_no_positive_momentum() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            "QQQ": [100.0, 99.5, 99.0, 98.5, 98.0, 97.5],
            "BIL": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05],
        },
        index=index,
    )

    weights = dual_momentum_weights(
        prices,
        ["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
        min_return=0.0,
    )

    assert weights["BIL"].iloc[-1] == 1.0
    assert weights[["SPY", "QQQ"]].iloc[-1].sum() == 0.0


def test_dual_momentum_can_cap_single_asset_and_hold_residual_defensive() -> None:
    index = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "QQQ": [100.0, 101.5, 103.0, 104.5, 106.0, 107.5, 109.0, 110.5],
            "BIL": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07],
        },
        index=index,
    )

    weights = dual_momentum_weights(
        prices,
        ["SPY", "QQQ"],
        lookback_days=2,
        skip_days=0,
        top_n=1,
        defensive_ticker="BIL",
        min_return=0.0,
        max_asset_weight=0.6,
    )

    assert weights["QQQ"].iloc[-1] == 0.6
    assert round(float(weights["BIL"].iloc[-1]), 6) == 0.4
    assert weights.iloc[-1].sum() == 1.0


def test_dual_momentum_supports_risk_adjusted_weighting() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    prices = pd.DataFrame(
        {
            "SMOOTH": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
            ],
            "CHOPPY": [
                100.0,
                104.0,
                99.0,
                106.0,
                101.0,
                108.0,
                103.0,
                110.0,
                105.0,
                112.0,
                107.0,
                114.0,
            ],
            "BIL": [100.0 + i * 0.01 for i in range(12)],
        },
        index=index,
    )

    weights = dual_momentum_weights(
        prices,
        ["SMOOTH", "CHOPPY"],
        lookback_days=4,
        skip_days=0,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.0,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        volatility_lookback_days=4,
    )

    assert round(float(weights[["SMOOTH", "CHOPPY"]].iloc[-1].sum()), 6) == 1.0
    assert weights["SMOOTH"].iloc[-1] > 0.0
