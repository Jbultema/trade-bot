from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.upside_capture import _apply_constructive_overlay


def test_constructive_overlay_adds_risk_floor_when_confirmed() -> None:
    index = pd.bdate_range("2026-01-01", periods=150)
    prices = pd.DataFrame(
        {
            "QQQ": [100 + i * 0.5 for i in range(150)],
            "SPY": [100 + i * 0.35 for i in range(150)],
            "RSP": [100 + i * 0.32 for i in range(150)],
            "SMH": [100 + i * 0.55 for i in range(150)],
            "HYG": [100 + i * 0.08 for i in range(150)],
            "LQD": [100 + i * 0.03 for i in range(150)],
            "BIL": [100 + i * 0.01 for i in range(150)],
        },
        index=index,
    )
    weights = pd.DataFrame(0.0, index=index, columns=["QQQ", "SPY", "SMH", "BIL"])
    weights["BIL"] = 1.0

    overlaid = _apply_constructive_overlay(
        prices,
        weights,
        {"mode": "risk_on_floor", "floor": 0.60, "signal": "balanced", "lookback": 21, "top_n": 2},
    )

    risk_weight = overlaid[["QQQ", "SPY", "SMH"]].sum(axis=1)
    assert risk_weight.iloc[-1] == pytest.approx(0.60)
    assert overlaid["BIL"].iloc[-1] == pytest.approx(0.40)
    assert risk_weight.iloc[:99].max() == 0.0
