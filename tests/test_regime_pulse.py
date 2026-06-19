from __future__ import annotations

import pandas as pd

from trade_bot.research.regime_pulse import (
    build_growth_inflation_map,
    build_regime_pulse_asset_table,
    build_regime_pulse_cycles,
)


def test_regime_pulse_translates_macro_pressure_to_asset_reads() -> None:
    macro_signals = pd.DataFrame(
        {
            "category": ["growth", "inflation_realized", "liquidity"],
            "latest_value": [1.0, 2.0, 3.0],
            "risk_score": [-0.60, 0.50, -0.40],
        }
    )

    cycles = build_regime_pulse_cycles(macro_signals)
    assets = build_regime_pulse_asset_table(cycles)
    grid = build_growth_inflation_map(cycles)

    growth = cycles[cycles["cycle"] == "growth"].iloc[0]
    inflation = cycles[cycles["cycle"] == "inflation"].iloc[0]
    stocks = assets[assets["asset_class"] == "stocks"].iloc[0]

    assert growth["cycle_state"] == "meaningful_tailwind"
    assert inflation["cycle_state"] == "meaningful_headwind"
    assert stocks["regime_pulse_read"] in {
        "macro supports buying or holding",
        "mixed or neutral",
    }
    assert round(float(grid["probability"].sum()), 6) == 1.0
    assert set(grid["regime"]) == {"Growth-disinflation", "Reflation", "Inflation", "Deflation"}


def test_regime_pulse_blends_positioning_proxy_into_positioning_cycle() -> None:
    macro_signals = pd.DataFrame(
        {
            "category": ["volatility"],
            "latest_value": [1.0],
            "risk_score": [0.0],
        }
    )
    positioning = pd.DataFrame(
        {
            "asset_group": ["broad_us_equity", "ai_beta"],
            "tickers": [3, 2],
            "mean_crowding_score": [0.80, 0.60],
        }
    )

    cycles = build_regime_pulse_cycles(macro_signals, positioning)
    positioning_cycle = cycles[cycles["cycle"] == "positioning"].iloc[0]

    assert positioning_cycle["cycle_state"] == "meaningful_headwind"
    assert positioning_cycle["series_count"] == 6
