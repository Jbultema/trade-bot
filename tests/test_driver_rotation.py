from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.research.driver_rotation import (
    build_driver_rotation_table,
    summarize_driver_rotation,
)


def test_driver_rotation_separates_allocation_drivers_from_explainers() -> None:
    index = pd.bdate_range("2026-01-01", periods=120)
    base = pd.Series(range(120), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 100.0 + base * 0.20,
            "HYG": 100.0 + base * 0.18,
            "LQD": 100.0 + base * 0.03,
            "SMH": 100.0 + base * 0.50,
            "QQQ": 100.0 + base * 0.30,
            "RSP": 100.0 + base * 0.05,
        },
        index=index,
    )
    current_state = SimpleNamespace(
        confirmation_matrix=pd.DataFrame(
            [
                {
                    "name": "High Yield vs IG Credit",
                    "theme": "credit",
                    "status": "bullish",
                    "score": 1.0,
                }
            ]
        ),
        macro_category_summary=pd.DataFrame(),
        regime_instability=pd.DataFrame(),
    )
    narrative = pd.DataFrame(
        [
            {
                "signal_id": "ipo_equity_supply_pressure",
                "signal_name": "IPO / equity supply pressure",
                "data_support": "thin_proxy",
                "score": 0.82,
                "evidence": "Large issuance calendar is active.",
            },
            {
                "signal_id": "paid_or_unavailable_data_watchlist",
                "signal_name": "Paid data watchlist",
                "data_support": "unsupported_watchlist",
                "score": 0.0,
                "evidence": "Missing paid flow data.",
            },
        ]
    )

    rotation = build_driver_rotation_table(
        prices,
        current_state,
        narrative_signals=narrative,
        family_importance_path=None,
    ).set_index("driver")

    assert rotation.loc["credit", "model_role"] == "allocation_driver"
    assert bool(rotation.loc["credit", "normally_important"])
    assert bool(rotation.loc["credit", "currently_active"])
    assert rotation.loc["equity_supply", "model_role"] == "explainer_only"
    assert bool(rotation.loc["equity_supply", "currently_active"])
    assert rotation.loc["unsupported_watchlist", "model_role"] == "unsupported"
    assert not bool(rotation.loc["unsupported_watchlist", "currently_active"])


def test_driver_rotation_detects_fading_recent_activation() -> None:
    index = pd.bdate_range("2026-01-01", periods=160)
    vixy = pd.Series(100.0, index=index)
    vixy.iloc[75:100] = [100.0 + value * 3.0 for value in range(25)]
    vixy.iloc[100:] = 174.0
    prices = pd.DataFrame({"VIXY": vixy}, index=index)
    current_state = SimpleNamespace(
        confirmation_matrix=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        regime_instability=pd.DataFrame(),
    )

    rotation = build_driver_rotation_table(
        prices,
        current_state,
        family_importance_path=None,
    ).set_index("driver")

    assert bool(rotation.loc["volatility", "normally_important"])
    assert not bool(rotation.loc["volatility", "currently_active"])
    assert bool(rotation.loc["volatility", "fading_importance"])
    assert rotation.loc["volatility", "primary_rotation_state"] == "fading_importance"


def test_driver_rotation_summary_names_active_and_emerging_counts() -> None:
    rotation = pd.DataFrame(
        [
            {
                "driver_label": "Credit conditions",
                "currently_active": True,
                "emerging_importance": False,
                "fading_importance": False,
                "normally_important": True,
                "current_activation": 0.9,
                "proven_relevance": 0.8,
            },
            {
                "driver_label": "AI capex pressure",
                "currently_active": True,
                "emerging_importance": True,
                "fading_importance": False,
                "normally_important": False,
                "current_activation": 0.7,
                "proven_relevance": 0.2,
            },
        ]
    )

    summary = summarize_driver_rotation(rotation)

    assert summary["answer"] == "2 active driver(s)"
    assert "Credit conditions" in summary["detail"]
    assert "AI capex pressure" in summary["detail"]
    assert "Emerging: 1" in summary["detail"]
