from __future__ import annotations

import pandas as pd

from trade_bot.research.action_headline import build_action_headline
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.trade_decision import TradeDecisionRun


def test_action_headline_marks_quiet_day_as_do_nothing() -> None:
    headline = build_action_headline(
        current_state=_current_state("green", 0.1),
        trade_decision=_trade_decision("HOLD", 0.0, risk_off=0.05, event_pressure=0.0),
        news_monitor=_news_monitor(),
    )

    assert headline.level == "do_nothing"
    assert headline.label == "Do Nothing Day"
    assert "No new trade action" in headline.next_action


def test_action_headline_marks_review_trade_as_small_actions() -> None:
    headline = build_action_headline(
        current_state=_current_state("yellow", 0.43),
        trade_decision=_trade_decision(
            "REVIEW_REDUCE_RISK",
            -0.08,
            risk_off=0.18,
            event_pressure=0.08,
        ),
        news_monitor=_news_monitor(active=True),
    )

    assert headline.level == "small_actions"
    assert headline.label == "Small Actions"
    assert headline.drivers["driver"].str.contains("Trade decision").any()


def test_action_headline_marks_large_red_risk_as_critical() -> None:
    headline = build_action_headline(
        current_state=_current_state("red", 0.88),
        trade_decision=_trade_decision("REDUCE_RISK", -0.25, risk_off=0.42, event_pressure=0.18),
        news_monitor=_news_monitor(active=True, high_urgency=True),
        open_ticket_count=2,
    )

    assert headline.level == "critical_actions"
    assert headline.label == "Critical Actions"
    assert headline.severity >= 10
    assert "open tickets" in headline.explanation


def test_action_headline_uses_book_aware_position_plan_for_action_size() -> None:
    book_position_plan = pd.DataFrame(
        [
            {
                "ticker": "BIL",
                "current_weight": 0.28,
                "scenario_adjusted_weight": 0.25,
                "delta_weight": -0.03,
                "action": "REDUCE",
            }
        ]
    )

    headline = build_action_headline(
        current_state=_current_state("yellow", 0.40),
        trade_decision=_trade_decision(
            "REVIEW_REDUCE_RISK",
            -0.25,
            risk_off=0.27,
            event_pressure=0.20,
        ),
        news_monitor=_news_monitor(active=True, high_urgency=True),
        position_plan=book_position_plan,
    )

    assert headline.level == "small_actions"
    assert headline.metrics.iloc[0]["max_position_change"] == 0.03
    assert "3.0%" in headline.headline


def _current_state(risk_status: str, risk_score: float) -> CurrentStateRun:
    return CurrentStateRun(
        market_date="2026-06-17",
        risk_score=risk_score,
        risk_status=risk_status,
        risk_summary=f"Risk status is {risk_status.upper()} with score {risk_score:.2f}.",
        market_health=pd.DataFrame(),
        momentum_state=pd.DataFrame(),
        confirmation_matrix=pd.DataFrame(),
        strategy_alerts=pd.DataFrame({"strategy": ["demo"], "action": ["HOLD"]}),
        scenario_outlook=pd.DataFrame(),
        scenario_lattice=pd.DataFrame(),
        scenario_drivers=pd.DataFrame(),
        macro_signals=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        signal_coverage=pd.DataFrame(),
        data_quality=pd.DataFrame(),
    )


def _trade_decision(
    action: str,
    delta_weight: float,
    *,
    risk_off: float,
    event_pressure: float,
) -> TradeDecisionRun:
    return TradeDecisionRun(
        summary=pd.DataFrame(
            [
                {
                    "recommended_action": action,
                    "one_month_risk_off_probability": risk_off,
                    "one_month_transition_probability": 0.1,
                    "event_pressure": event_pressure,
                    "macro_pressure": 0.0,
                }
            ]
        ),
        position_plan=pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "current_weight": 0.5,
                    "scenario_adjusted_weight": 0.5 + delta_weight,
                    "delta_weight": delta_weight,
                    "action": "REDUCE" if delta_weight < 0 else "HOLD",
                }
            ]
        ),
        evidence=pd.DataFrame(),
        scenario_links=pd.DataFrame(),
    )


def _news_monitor(*, active: bool = False, high_urgency: bool = False) -> NewsMonitorRun:
    triage = pd.DataFrame()
    if active or high_urgency:
        triage = pd.DataFrame(
            [
                {
                    "activation_status": "event_risk_generated" if active else "triage_only",
                    "urgency_score": 0.95 if high_urgency else 0.82,
                }
            ]
        )
    return NewsMonitorRun(
        items=(),
        triage=triage,
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.8,
        lookback_days=7,
    )
