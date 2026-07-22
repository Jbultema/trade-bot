from __future__ import annotations

import pandas as pd

from trade_bot.research.risk_timing import assess_risk_timing


def test_fragility_without_market_breaks_only_uses_watch_budget() -> None:
    assessment = assess_risk_timing(0.72, _confirmation(), _health())

    assert assessment.state == "fragile_intact"
    assert assessment.multiplier == 1.00
    assert assessment.confirmation_breaks == ()


def test_two_independent_breaks_confirm_progressive_derisking() -> None:
    confirmation = _confirmation(credit="bearish", breadth="bearish")
    health = _health(hyg_1m=-0.03, rsp_1m=-0.08, spy_1m=-0.02)

    assessment = assess_risk_timing(0.56, confirmation, health)

    assert assessment.state == "confirmed_break"
    assert assessment.multiplier == 0.65
    assert assessment.confirmation_breaks == ("credit", "breadth")


def test_broad_recovery_relaxes_a_stale_slow_risk_score() -> None:
    health = _health(
        spy_1m=0.04,
        qqq_1m=0.05,
        rsp_1m=0.06,
        hyg_1m=0.02,
        vixy_1m=-0.10,
    )

    assessment = assess_risk_timing(0.68, _confirmation(), health)

    assert assessment.state == "stabilizing"
    assert assessment.multiplier == 1.00
    assert len(assessment.recovery_confirmations) == 4


def test_severe_state_requires_breadth_of_breaks_and_drawdown() -> None:
    confirmation = _confirmation(
        credit="bearish",
        volatility="risk_off",
        breadth="bearish",
    )
    health = _health(
        hyg_1m=-0.03,
        vixy_1m=0.20,
        rsp_1m=-0.10,
        spy_1m=-0.04,
        spy_3m=-0.05,
        spy_dd=-0.12,
    )

    assessment = assess_risk_timing(0.80, confirmation, health)

    assert assessment.state == "severe_break"
    assert assessment.multiplier == 0.40
    assert len(assessment.confirmation_breaks) == 4


def _confirmation(
    *,
    credit: str = "neutral",
    volatility: str = "neutral",
    breadth: str = "neutral",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "High Yield vs IG Credit", "status": credit},
            {"name": "Volatility ETF Pressure", "status": volatility},
            {"name": "Equal Weight vs Cap Weight", "status": breadth},
            {"name": "SPY Trend", "status": "neutral"},
            {"name": "QQQ Trend", "status": "neutral"},
        ]
    )


def _health(**overrides: float) -> pd.DataFrame:
    values = {
        "spy_1m": 0.01,
        "spy_3m": 0.02,
        "spy_dd": -0.02,
        "qqq_1m": 0.01,
        "qqq_3m": 0.02,
        "qqq_dd": -0.03,
        "rsp_1m": 0.01,
        "hyg_1m": 0.00,
        "vixy_1m": 0.00,
    }
    values.update(overrides)
    return pd.DataFrame(
        {
            "return_1m": {
                "SPY": values["spy_1m"],
                "QQQ": values["qqq_1m"],
                "RSP": values["rsp_1m"],
                "HYG": values["hyg_1m"],
                "VIXY": values["vixy_1m"],
            },
            "return_3m": {
                "SPY": values["spy_3m"],
                "QQQ": values["qqq_3m"],
            },
            "drawdown": {
                "SPY": values["spy_dd"],
                "QQQ": values["qqq_dd"],
            },
        }
    )
