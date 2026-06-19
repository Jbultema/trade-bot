from __future__ import annotations

import pandas as pd

from trade_bot.research.positioning_crowding import (
    build_positioning_crowding_table,
    build_positioning_summary,
)


def test_positioning_crowding_detects_crowded_and_washed_out_assets() -> None:
    index = pd.bdate_range("2022-01-03", periods=620)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0 + value * 0.20 for value in range(620)],
            "SPY": [100.0 + value * 0.05 for value in range(620)],
            "TLT": [220.0 - value * 0.18 for value in range(620)],
        },
        index=index,
    )
    prices.loc[index[-25]:, "QQQ"] = [
        prices.loc[index[-26], "QQQ"] + value * 2.0 for value in range(1, 26)
    ]
    prices.loc[index[-25]:, "TLT"] = [
        prices.loc[index[-26], "TLT"] - value * 2.0 for value in range(1, 26)
    ]

    crowding = build_positioning_crowding_table(prices)
    summary = build_positioning_summary(crowding)

    qqq = crowding[crowding["ticker"] == "QQQ"].iloc[0]
    tlt = crowding[crowding["ticker"] == "TLT"].iloc[0]

    assert qqq["crowding_state"] in {"bearish_crowding", "crowded_risk"}
    assert tlt["crowding_state"] in {"bullish_washout", "washed_out_opportunity"}
    assert "broad_us_equity" in set(summary["asset_group"])
    assert "duration" in set(summary["asset_group"])
