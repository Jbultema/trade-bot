from __future__ import annotations

import pandas as pd

from trade_bot.data.fred_data import FredSeries
from trade_bot.research.macro_state import (
    build_macro_category_summary,
    build_macro_signal_table,
    build_signal_coverage_table,
)


def test_macro_signal_table_scores_risk_polarity() -> None:
    index = pd.bdate_range("2024-01-01", periods=300)
    macro_data = pd.DataFrame(
        {
            "STRESS": [float(value) for value in range(300)],
            "GROWTH": [300.0 - float(value) for value in range(300)],
        },
        index=index,
    )
    catalog = (
        FredSeries("STRESS", "Stress", "financial_conditions", "risk_off_when_rising"),
        FredSeries("GROWTH", "Growth", "growth", "risk_on_when_rising"),
    )

    signals = build_macro_signal_table(macro_data, catalog)
    summary = build_macro_category_summary(signals)

    assert signals.loc[signals["series_id"] == "STRESS", "risk_state"].iloc[0] == "risk_pressure"
    assert signals.loc[signals["series_id"] == "GROWTH", "risk_state"].iloc[0] == "risk_pressure"
    assert "change_1w" in signals.columns
    assert "change_acceleration_1m_vs_3m" in signals.columns
    assert "short_move_z_1m" in signals.columns
    assert "near_term_state" in signals.columns
    assert signals.loc[signals["series_id"] == "STRESS", "change_1w"].iloc[0] > 0
    assert set(summary["category"]) == {"financial_conditions", "growth"}


def test_macro_signal_table_detects_near_term_pressure() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    values = [100.0 for _ in range(60)] + [100.0 + float(value) * 2.0 for value in range(20)]
    macro_data = pd.DataFrame({"STRESS": values}, index=index)
    catalog = (FredSeries("STRESS", "Stress", "financial_conditions", "risk_off_when_rising"),)

    signals = build_macro_signal_table(macro_data, catalog, lookback_days=80)
    row = signals.iloc[0]

    assert row["change_1m"] > 0
    assert row["slope_1m"] > 0
    assert row["range_position_1y"] == 1.0
    assert row["near_term_state"] == "near_term_risk_pressure"


def test_signal_coverage_table_includes_known_gaps() -> None:
    prices = pd.DataFrame({"SPY": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    macro = pd.DataFrame({"DGS10": [4.0, 4.1]}, index=prices.index)
    catalog = (FredSeries("DGS10", "10-year", "rates_curve", "risk_off_when_rising"),)

    coverage = build_signal_coverage_table(
        yahoo_prices=prices,
        macro_data=macro,
        macro_catalog=catalog,
    )

    assert "positioning and crowding" in set(coverage["coverage_area"])
    assert (
        coverage.loc[coverage["coverage_area"] == "Yahoo market proxies", "series_count"].iloc[0]
        == 1
    )
