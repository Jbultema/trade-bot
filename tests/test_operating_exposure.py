from __future__ import annotations

import pandas as pd

from trade_bot.research.operating_exposure import (
    aggregate_beta_adjusted_spy_delta,
    build_sleeve_exposure_table,
    build_tactical_matrix,
)


def test_beta_adjusted_delta_uses_weighted_spy_beta() -> None:
    index = pd.bdate_range("2025-01-01", periods=80)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + index_position for index_position in range(len(index))],
            "BIL": [100.0] * len(index),
        },
        index=index,
    )
    delta = aggregate_beta_adjusted_spy_delta(prices, {"SPY": 0.60, "BIL": 0.40})

    assert delta == 0.60


def test_sleeve_exposure_maps_global_risk_sleeves_to_percent_of_max() -> None:
    exposure = build_sleeve_exposure_table(
        {"VT": 0.60, "USFR": 0.40, "GLDM": 0.0, "FBTC": 0.0}
    ).set_index("sleeve")

    assert exposure.loc["stocks", "current_weight"] == 0.60
    assert exposure.loc["stocks", "percent_of_max_sleeve"] == 1.0
    assert exposure.loc["defensive", "current_weight"] == 0.40
    assert exposure.loc["defensive", "percent_of_max_sleeve"] == 0.40
    assert exposure.loc["gold", "current_weight"] == 0.0
    assert exposure.loc["crypto", "current_weight"] == 0.0


def test_tactical_matrix_labels_rising_assets_bullish_and_breaking_assets_bearish() -> None:
    index = pd.bdate_range("2025-01-01", periods=140)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + index_position for index_position in range(len(index))],
            "AGG": [120.0 - index_position * 0.2 for index_position in range(len(index))],
        },
        index=index,
    )

    matrix = build_tactical_matrix(
        prices,
        tickers=("SPY", "AGG"),
        lookback_days=21,
        trend_days=63,
        risk_status="green",
        regime="test",
    ).set_index("ticker")

    assert matrix.loc["SPY", "condition"] == "bullish"
    assert matrix.loc["SPY", "suggested_position_size"] == "Long | Max Position"
    assert matrix.loc["AGG", "condition"] == "bearish"
    assert matrix.loc["AGG", "suggested_position_size"] == "No New Position"
