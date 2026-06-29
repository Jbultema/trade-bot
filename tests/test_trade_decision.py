from __future__ import annotations

from dataclasses import replace

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun, MarketEvent
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import _event_context, build_trade_decision


def test_trade_decision_reduces_risk_when_scenarios_and_events_are_adverse() -> None:
    index = pd.bdate_range("2026-06-01", periods=5)
    weights = pd.DataFrame({"QQQ": 0.5, "IWM": 0.5}, index=index)
    result = BacktestResult(
        name="primary",
        equity=pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index),
        returns=pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=index),
        gross_returns=pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=index),
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    decision = build_trade_decision(
        primary_result=result,
        current_state=_current_state(),
        event_risk=_event_risk(),
        news_monitor=_news_monitor(),
        signal_inclusion=_signal_inclusion(),
    )

    summary = decision.summary.iloc[0]
    bil_row = decision.position_plan[decision.position_plan["ticker"] == "BIL"].iloc[0]

    assert summary["recommended_action"] == "REVIEW_REDUCE_RISK"
    assert summary["risk_budget_multiplier"] < 1.0
    assert summary["posture_calibration_status"] == "defense_justified"
    assert bil_row["scenario_adjusted_weight"] > 0.0
    assert "Because risk status is YELLOW" in summary["human_explanation"]
    assert "Credit-led risk-off" in summary["human_explanation"]
    assert "Posture calibration says" in summary["human_explanation"]
    assert "posture_calibration" in set(decision.evidence["evidence_type"])


def test_trade_decision_uses_portfolio_risk_engine_when_prices_are_available() -> None:
    index = pd.bdate_range("2025-01-01", periods=180)
    weights = pd.DataFrame({"QQQ": 0.8, "SPY": 0.2}, index=index)
    prices = _risk_prices(index)
    result = BacktestResult(
        name="primary",
        equity=pd.Series(range(100, 280), index=index, dtype=float),
        returns=pd.Series(0.001, index=index),
        gross_returns=pd.Series(0.001, index=index),
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )

    decision = build_trade_decision(
        primary_result=result,
        current_state=_current_state(),
        event_risk=_event_risk(),
        news_monitor=_news_monitor(),
        signal_inclusion=_signal_inclusion(),
        prices=prices,
    )

    summary = decision.summary.iloc[0]
    bil_row = decision.position_plan[decision.position_plan["ticker"] == "BIL"].iloc[0]

    assert decision.portfolio_risk is not None
    assert summary["portfolio_risk_level"] != "not_available"
    assert summary["decision_authority"] == "scenario_event_risk_engine_review"
    assert bil_row["scenario_adjusted_weight"] > 0.0
    assert "risk_engine_delta" in decision.position_plan
    assert "Portfolio risk engine says" in summary["human_explanation"]


def test_trade_decision_flags_possible_opportunity_cost_when_constructive() -> None:
    index = pd.bdate_range("2026-06-01", periods=5)
    weights = pd.DataFrame({"QQQ": 0.5, "IWM": 0.5}, index=index)
    result = BacktestResult(
        name="primary",
        equity=pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index),
        returns=pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=index),
        gross_returns=pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=index),
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "rank": 1,
                "scenario": "Broad risk-on broadening",
                "probability": 0.55,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Maintain risk.",
                "preferred_exposure": "SPY/RSP",
                "avoid_exposure": "Concentration",
                "confirmation": "Breadth improves.",
                "off_ramp": "Cut if credit weakens.",
            },
            {
                "horizon": "1m",
                "rank": 2,
                "scenario": "AI upside squeeze",
                "probability": 0.20,
                "risk_bucket": "risk_on_fragile",
                "expected_bot_posture": "Participate smaller.",
                "preferred_exposure": "QQQ",
                "avoid_exposure": "Crowded beta",
                "confirmation": "Credit holds.",
                "off_ramp": "Cut if breadth rolls over.",
            },
            {
                "horizon": "1m",
                "rank": 3,
                "scenario": "Choppy transition",
                "probability": 0.15,
                "risk_bucket": "transition",
                "expected_bot_posture": "Use guardrails.",
                "preferred_exposure": "Quality",
                "avoid_exposure": "Over-trading",
                "confirmation": "Mixed leadership.",
                "off_ramp": "Defensive if credit breaks.",
            },
            {
                "horizon": "1m",
                "rank": 4,
                "scenario": "Risk-off false break",
                "probability": 0.10,
                "risk_bucket": "risk_off",
                "expected_bot_posture": "Respect stops.",
                "preferred_exposure": "BIL",
                "avoid_exposure": "High beta",
                "confirmation": "Credit weakens.",
                "off_ramp": "Stay defensive until recovery.",
            },
        ]
    )
    current_state = replace(_current_state(), scenario_lattice=scenario_lattice)

    decision = build_trade_decision(
        primary_result=result,
        current_state=current_state,
        event_risk=_empty_event_risk(),
        news_monitor=_news_monitor(),
        signal_inclusion=_empty_signal_inclusion(),
    )

    summary = decision.summary.iloc[0]

    assert summary["recommended_action"] == "REVIEW_REDUCE_RISK"
    assert summary["posture_calibration_status"] == "opportunity_cost_watch"
    assert summary["opportunity_pressure"] > 0.35
    assert summary["constructive_scenario_probability"] > summary["one_month_risk_off_probability"]
    assert "Opportunity-cost watch" in set(decision.evidence["signal"])


def test_trade_decision_caps_event_only_derisk_without_market_confirmation() -> None:
    index = pd.bdate_range("2025-01-01", periods=180)
    weights = pd.DataFrame({"QQQ": 0.8, "SPY": 0.2}, index=index)
    result = _backtest_result("primary", index, weights)
    current_state = replace(
        _current_state(),
        scenario_lattice=_constructive_scenario_lattice(),
        confirmation_matrix=_confirmation_matrix(negative_themes=()),
    )

    decision = build_trade_decision(
        primary_result=result,
        current_state=current_state,
        event_risk=_event_risk(),
        news_monitor=_news_monitor(),
        signal_inclusion=_empty_signal_inclusion(),
        prices=_risk_prices(index),
    )

    summary = decision.summary.iloc[0]
    bil_row = decision.position_plan[decision.position_plan["ticker"] == "BIL"].iloc[0]

    assert summary["decision_sanity_status"] == "event_only_cap_applied"
    assert bool(summary["decision_sanity_cap_applied"])
    assert summary["market_confirmation_break_count"] == 0
    assert summary["pre_sanity_risk_budget_multiplier"] < summary["risk_budget_multiplier"]
    assert bil_row["scenario_adjusted_weight"] <= 0.25 + 1e-9
    assert "Decision sanity says" in summary["human_explanation"]
    assert "decision_sanity" in set(decision.evidence["evidence_type"])


def test_trade_decision_allows_larger_derisk_when_market_confirmation_breaks() -> None:
    index = pd.bdate_range("2025-01-01", periods=180)
    weights = pd.DataFrame({"QQQ": 0.8, "SPY": 0.2}, index=index)
    result = _backtest_result("primary", index, weights)
    current_state = replace(
        _current_state(),
        scenario_lattice=_constructive_scenario_lattice(),
        confirmation_matrix=_confirmation_matrix(negative_themes=("credit", "volatility")),
    )

    decision = build_trade_decision(
        primary_result=result,
        current_state=current_state,
        event_risk=_event_risk(),
        news_monitor=_news_monitor(),
        signal_inclusion=_empty_signal_inclusion(),
        prices=_risk_prices(index),
    )

    summary = decision.summary.iloc[0]
    bil_row = decision.position_plan[decision.position_plan["ticker"] == "BIL"].iloc[0]

    assert summary["decision_sanity_status"] == "market_confirmation_allows_derisk"
    assert not bool(summary["decision_sanity_cap_applied"])
    assert summary["market_confirmation_break_count"] == 2
    assert summary["market_confirmation_breaks"] == "credit, volatility"
    assert summary["pre_sanity_risk_budget_multiplier"] == summary["risk_budget_multiplier"]
    assert bil_row["scenario_adjusted_weight"] > 0.25



def _backtest_result(name: str, index: pd.DatetimeIndex, weights: pd.DataFrame) -> BacktestResult:
    return BacktestResult(
        name=name,
        equity=pd.Series(range(100, 100 + len(index)), index=index, dtype=float),
        returns=pd.Series(0.001, index=index),
        gross_returns=pd.Series(0.001, index=index),
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )


def _constructive_scenario_lattice() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": "1m",
                "rank": 1,
                "scenario": "Broad risk-on broadening",
                "probability": 0.45,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Maintain risk.",
                "preferred_exposure": "SPY/RSP",
                "avoid_exposure": "Concentration",
                "confirmation": "Breadth improves.",
                "off_ramp": "Cut if credit weakens.",
            },
            {
                "horizon": "1m",
                "rank": 2,
                "scenario": "AI upside squeeze",
                "probability": 0.25,
                "risk_bucket": "risk_on_fragile",
                "expected_bot_posture": "Participate smaller.",
                "preferred_exposure": "QQQ",
                "avoid_exposure": "Crowded beta",
                "confirmation": "Credit holds.",
                "off_ramp": "Cut if breadth rolls over.",
            },
            {
                "horizon": "1m",
                "rank": 3,
                "scenario": "Choppy transition",
                "probability": 0.20,
                "risk_bucket": "transition",
                "expected_bot_posture": "Use guardrails.",
                "preferred_exposure": "Quality",
                "avoid_exposure": "Over-trading",
                "confirmation": "Mixed leadership.",
                "off_ramp": "Defensive if credit breaks.",
            },
            {
                "horizon": "1m",
                "rank": 4,
                "scenario": "Risk-off false break",
                "probability": 0.10,
                "risk_bucket": "risk_off",
                "expected_bot_posture": "Respect stops.",
                "preferred_exposure": "BIL",
                "avoid_exposure": "High beta",
                "confirmation": "Credit weakens.",
                "off_ramp": "Stay defensive until recovery.",
            },
        ]
    )


def _confirmation_matrix(negative_themes: tuple[str, ...]) -> pd.DataFrame:
    rows = [
        {"name": "High Yield vs IG Credit", "theme": "credit"},
        {"name": "Volatility ETF Pressure", "theme": "volatility"},
        {"name": "Equal Weight vs Cap Weight", "theme": "breadth"},
        {"name": "SPY Trend", "theme": "broad_market"},
        {"name": "QQQ Trend", "theme": "ai_beta"},
    ]
    for row in rows:
        theme = str(row["theme"])
        canonical = "trend" if theme in {"broad_market", "ai_beta"} else theme
        negative = canonical in negative_themes
        row["status"] = "bearish" if negative else "bullish"
        row["score"] = -1 if negative else 1
    return pd.DataFrame(rows)

def _risk_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    trend = pd.Series(range(len(index)), index=index, dtype=float)
    return pd.DataFrame(
        {
            "SPY": 100.0 + trend,
            "QQQ": 100.0 + trend * 1.5,
            "SMH": 100.0 + trend * 1.8,
            "RSP": 100.0 + trend * 0.8,
            "IWM": 100.0 + trend * 0.9,
            "HYG": 100.0 + trend * 0.4,
            "TLT": 100.0 - trend * 0.1,
            "GLD": 100.0 + trend * 0.1,
            "BIL": 100.0 + trend * 0.01,
        },
        index=index,
    )


def _current_state() -> CurrentStateRun:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "rank": 1,
                "scenario": "Credit-led risk-off",
                "probability": 0.45,
                "risk_bucket": "risk_off",
                "expected_bot_posture": "Prioritize drawdown control.",
                "preferred_exposure": "BIL/SGOV",
                "avoid_exposure": "High beta",
                "confirmation": "HYG/LQD weakens.",
                "off_ramp": "Stay defensive until credit recovers.",
            },
            {
                "horizon": "1m",
                "rank": 2,
                "scenario": "Choppy factor rotation",
                "probability": 0.35,
                "risk_bucket": "transition",
                "expected_bot_posture": "Keep allocations smaller.",
                "preferred_exposure": "Quality",
                "avoid_exposure": "Over-trading",
                "confirmation": "Leadership changes.",
                "off_ramp": "Move defensive if volatility expands.",
            },
            {
                "horizon": "1m",
                "rank": 3,
                "scenario": "Broad risk-on broadening",
                "probability": 0.20,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Maintain risk.",
                "preferred_exposure": "SPY/RSP",
                "avoid_exposure": "Concentration",
                "confirmation": "Breadth improves.",
                "off_ramp": "Cut if credit weakens.",
            },
        ]
    )
    return CurrentStateRun(
        market_date="2026-06-17",
        risk_score=0.43,
        risk_status="yellow",
        risk_summary="Risk status is YELLOW with score 0.43.",
        market_health=pd.DataFrame(),
        momentum_state=pd.DataFrame(),
        confirmation_matrix=pd.DataFrame(),
        strategy_alerts=pd.DataFrame(),
        scenario_outlook=pd.DataFrame(),
        scenario_lattice=scenario_lattice,
        scenario_drivers=pd.DataFrame(),
        macro_signals=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        signal_coverage=pd.DataFrame(),
        data_quality=pd.DataFrame(),
    )


def _event_risk() -> EventRiskRun:
    event = MarketEvent(
        event_id="openai_financials",
        name="OpenAI financials",
        date=pd.Timestamp("2026-06-15"),
        category="ai_unit_economics",
        direction="escalation",
        description="test",
        current=True,
        phase="leading_warning",
    )
    return EventRiskRun(
        events=(event,),
        asset_event_returns=pd.DataFrame(),
        strategy_event_returns=pd.DataFrame(),
        event_summary=pd.DataFrame(),
        scenario_playbook=pd.DataFrame(),
        current_event_scenarios=pd.DataFrame(),
    )


def test_watch_only_events_are_context_but_not_sizing_pressure() -> None:
    watch_event = MarketEvent(
        event_id="bis_ai_financing_warning",
        name="BIS AI financing warning",
        date=pd.Timestamp("2026-06-29"),
        category="ai_unit_economics",
        direction="escalation",
        description="watch context",
        current=True,
        phase="leading_warning",
        sizing_authority=False,
    )
    sizing_event = replace(watch_event, event_id="sizing_event", sizing_authority=True)

    watch_context = _event_context((watch_event,))
    sizing_context = _event_context((sizing_event,))

    assert watch_context["current_event_count"] == 1
    assert watch_context["watch_only_event_count"] == 1
    assert watch_context["event_pressure"] == 0.0
    assert watch_context["risk_multiplier"] == 1.0
    assert "watch-only" in str(watch_context["evidence"])
    assert sizing_context["event_pressure"] == 0.07


def _news_monitor() -> NewsMonitorRun:
    return NewsMonitorRun(
        items=(),
        triage=pd.DataFrame({"activation_status": ["covered_by_curated_event"]}),
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.8,
        lookback_days=7,
    )


def _empty_event_risk() -> EventRiskRun:
    return EventRiskRun(
        events=(),
        asset_event_returns=pd.DataFrame(),
        strategy_event_returns=pd.DataFrame(),
        event_summary=pd.DataFrame(),
        scenario_playbook=pd.DataFrame(),
        current_event_scenarios=pd.DataFrame(),
    )


def _empty_signal_inclusion() -> SignalInclusionRun:
    return SignalInclusionRun(
        summary=pd.DataFrame(),
        pressure=pd.DataFrame(),
        results={},
        metrics=pd.DataFrame(),
        window_summary=pd.DataFrame(),
    )


def _signal_inclusion() -> SignalInclusionRun:
    return SignalInclusionRun(
        summary=pd.DataFrame(
            {
                "decision": ["reject_for_now"],
                "latest_pressure_state": ["risk_pressure"],
                "signal_group": ["macro:inflation_realized"],
            }
        ),
        pressure=pd.DataFrame(),
        results={},
        metrics=pd.DataFrame(),
        window_summary=pd.DataFrame(),
    )
