from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.trade_decision import TradeDecisionRun


@dataclass(frozen=True)
class ActionHeadline:
    level: str
    label: str
    severity: int
    headline: str
    next_action: str
    explanation: str
    metrics: pd.DataFrame
    drivers: pd.DataFrame


def build_action_headline(
    *,
    current_state: CurrentStateRun,
    trade_decision: TradeDecisionRun,
    news_monitor: NewsMonitorRun,
    open_ticket_count: int = 0,
) -> ActionHeadline:
    summary = _summary_row(trade_decision.summary)
    position_plan = trade_decision.position_plan
    max_abs_delta = _max_abs_delta(position_plan)
    material_trade_count = _material_trade_count(position_plan)
    recommended_action = str(summary.get("recommended_action", "HOLD"))
    risk_off_probability = _as_float(summary.get("one_month_risk_off_probability"))
    transition_probability = _as_float(summary.get("one_month_transition_probability"))
    event_pressure = _as_float(summary.get("event_pressure"))
    macro_pressure = _as_float(summary.get("macro_pressure"))
    high_urgency_news = _high_urgency_news(news_monitor.triage)
    active_news = _active_news(news_monitor.triage)
    non_hold_alerts = _non_hold_strategy_alerts(current_state.strategy_alerts)

    drivers = _drivers(
        current_state=current_state,
        recommended_action=recommended_action,
        max_abs_delta=max_abs_delta,
        material_trade_count=material_trade_count,
        risk_off_probability=risk_off_probability,
        transition_probability=transition_probability,
        event_pressure=event_pressure,
        macro_pressure=macro_pressure,
        high_urgency_news=high_urgency_news,
        active_news=active_news,
        non_hold_alerts=non_hold_alerts,
        open_ticket_count=open_ticket_count,
    )
    severity = int(drivers["severity_points"].sum()) if not drivers.empty else 0
    level = _level(
        severity=severity,
        risk_status=current_state.risk_status,
        recommended_action=recommended_action,
        max_abs_delta=max_abs_delta,
        risk_off_probability=risk_off_probability,
    )
    label = {
        "do_nothing": "Do Nothing Day",
        "small_actions": "Small Actions",
        "critical_actions": "Critical Actions",
    }[level]
    next_action = _next_action(
        level=level,
        recommended_action=recommended_action,
        material_trade_count=material_trade_count,
        open_ticket_count=open_ticket_count,
    )
    headline = _headline(
        label=label,
        recommended_action=recommended_action,
        current_state=current_state,
        max_abs_delta=max_abs_delta,
    )
    explanation = _explanation(
        level=level,
        current_state=current_state,
        max_abs_delta=max_abs_delta,
        risk_off_probability=risk_off_probability,
        active_news=active_news,
        open_ticket_count=open_ticket_count,
    )
    metrics = pd.DataFrame(
        [
            {
                "risk_status": current_state.risk_status.upper(),
                "risk_score": current_state.risk_score,
                "recommended_action": recommended_action,
                "max_position_change": max_abs_delta,
                "material_trade_count": material_trade_count,
                "one_month_risk_off_probability": risk_off_probability,
                "one_month_transition_probability": transition_probability,
                "event_pressure": event_pressure,
                "macro_pressure": macro_pressure,
                "active_news_items": active_news,
                "high_urgency_news_items": high_urgency_news,
                "open_ticket_count": open_ticket_count,
            }
        ]
    )
    return ActionHeadline(
        level=level,
        label=label,
        severity=severity,
        headline=headline,
        next_action=next_action,
        explanation=explanation,
        metrics=metrics,
        drivers=drivers.sort_values("severity_points", ascending=False).reset_index(drop=True),
    )


def _summary_row(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {}
    return summary.iloc[0].to_dict()


def _max_abs_delta(position_plan: pd.DataFrame) -> float:
    if position_plan.empty or "delta_weight" not in position_plan:
        return 0.0
    return float(position_plan["delta_weight"].abs().max())


def _material_trade_count(position_plan: pd.DataFrame) -> int:
    if position_plan.empty or "action" not in position_plan:
        return 0
    return int(position_plan["action"].isin(["ADD", "REDUCE"]).sum())


def _high_urgency_news(triage: pd.DataFrame) -> int:
    if triage.empty or "urgency_score" not in triage:
        return 0
    return int((triage["urgency_score"].astype(float) >= 0.90).sum())


def _active_news(triage: pd.DataFrame) -> int:
    if triage.empty or "activation_status" not in triage:
        return 0
    active_statuses = {"event_risk_generated", "covered_by_curated_event"}
    return int(triage["activation_status"].isin(active_statuses).sum())


def _non_hold_strategy_alerts(strategy_alerts: pd.DataFrame) -> int:
    if strategy_alerts.empty or "action" not in strategy_alerts:
        return 0
    return int((strategy_alerts["action"].astype(str) != "HOLD").sum())


def _drivers(
    *,
    current_state: CurrentStateRun,
    recommended_action: str,
    max_abs_delta: float,
    material_trade_count: int,
    risk_off_probability: float,
    transition_probability: float,
    event_pressure: float,
    macro_pressure: float,
    high_urgency_news: int,
    active_news: int,
    non_hold_alerts: int,
    open_ticket_count: int,
) -> pd.DataFrame:
    rows = []
    rows.append(
        _driver(
            "Risk state",
            current_state.risk_status.upper(),
            _risk_status_points(current_state.risk_status),
            current_state.risk_summary,
        )
    )
    if recommended_action != "HOLD":
        rows.append(
            _driver(
                "Trade decision",
                recommended_action,
                2 if recommended_action == "REVIEW_REDUCE_RISK" else 4,
                f"{material_trade_count} material trade lines; largest target change {max_abs_delta:.1%}.",
            )
        )
    elif max_abs_delta >= 0.02:
        rows.append(
            _driver(
                "Trade decision",
                "Review small drift",
                1,
                f"Largest target change is {max_abs_delta:.1%}.",
            )
        )
    if risk_off_probability >= 0.25:
        rows.append(
            _driver(
                "Scenario risk",
                f"{risk_off_probability:.0%} 1M risk-off",
                3,
                "Scenario lattice is assigning meaningful probability to risk-off outcomes.",
            )
        )
    elif transition_probability >= 0.25:
        rows.append(
            _driver(
                "Scenario transition",
                f"{transition_probability:.0%} 1M transition",
                1,
                "Scenario lattice is tilted toward choppy or transitional outcomes.",
            )
        )
    if event_pressure >= 0.12:
        rows.append(
            _driver(
                "Event pressure",
                f"{event_pressure:.0%}",
                3,
                "Current event-risk layer is materially reducing the risk budget.",
            )
        )
    elif event_pressure > 0:
        rows.append(
            _driver(
                "Event pressure",
                f"{event_pressure:.0%}",
                1,
                "Current events are present but not at critical pressure.",
            )
        )
    if macro_pressure > 0:
        rows.append(
            _driver(
                "Macro pressure",
                f"{macro_pressure:.0%}",
                1,
                "Paper-candidate macro signals are currently pressuring risk.",
            )
        )
    if high_urgency_news:
        rows.append(
            _driver(
                "High-urgency news",
                str(high_urgency_news),
                min(3, high_urgency_news),
                "Recent news intake has high-urgency items.",
            )
        )
    if active_news:
        rows.append(
            _driver(
                "Active news events",
                str(active_news),
                min(2, active_news),
                "News items have been converted into current event-risk context.",
            )
        )
    if non_hold_alerts:
        rows.append(
            _driver(
                "Strategy alerts",
                str(non_hold_alerts),
                min(2, non_hold_alerts),
                "Configured strategies have non-hold alerts.",
            )
        )
    if open_ticket_count:
        rows.append(
            _driver(
                "Open tickets",
                str(open_ticket_count),
                2,
                "There are locked paper/live recommendation tickets that need disposition.",
            )
        )
    return pd.DataFrame(rows)


def _driver(driver: str, signal: str, severity_points: int, detail: str) -> dict[str, object]:
    return {
        "driver": driver,
        "signal": signal,
        "severity_points": severity_points,
        "detail": detail,
    }


def _risk_status_points(risk_status: str) -> int:
    return {
        "green": 0,
        "yellow": 1,
        "orange": 3,
        "red": 5,
    }.get(risk_status, 2)


def _level(
    *,
    severity: int,
    risk_status: str,
    recommended_action: str,
    max_abs_delta: float,
    risk_off_probability: float,
) -> str:
    if (
        risk_status == "red"
        or recommended_action == "REDUCE_RISK"
        or max_abs_delta >= 0.20
        or risk_off_probability >= 0.35
        or severity >= 10
    ):
        return "critical_actions"
    if (
        risk_status in {"yellow", "orange"}
        or recommended_action != "HOLD"
        or max_abs_delta >= 0.02
        or severity >= 3
    ):
        return "small_actions"
    return "do_nothing"


def _next_action(
    *,
    level: str,
    recommended_action: str,
    material_trade_count: int,
    open_ticket_count: int,
) -> str:
    if open_ticket_count:
        return "Review open locked tickets before creating new ones."
    if level == "critical_actions":
        return "Review risk-reduction tickets before the next execution window."
    if level == "small_actions" and material_trade_count:
        return (
            "Review the suggested ticket preview and decide whether to lock a paper recommendation."
        )
    if recommended_action != "HOLD":
        return "Review the trade decision, but do not execute without locking a ticket."
    return "No new trade action; keep monitoring headline drivers."


def _headline(
    *,
    label: str,
    recommended_action: str,
    current_state: CurrentStateRun,
    max_abs_delta: float,
) -> str:
    return (
        f"{label}: {recommended_action.replace('_', ' ').title()} "
        f"with {current_state.risk_status.upper()} risk and max target change {max_abs_delta:.1%}."
    )


def _explanation(
    *,
    level: str,
    current_state: CurrentStateRun,
    max_abs_delta: float,
    risk_off_probability: float,
    active_news: int,
    open_ticket_count: int,
) -> str:
    if level == "do_nothing":
        return "No material target-weight changes or urgent risk drivers are currently above action thresholds."
    if level == "critical_actions":
        return (
            f"Risk status is {current_state.risk_status.upper()}, largest target change is "
            f"{max_abs_delta:.1%}, 1-month risk-off probability is {risk_off_probability:.0%}, "
            f"active news events are {active_news}, and open tickets are {open_ticket_count}."
        )
    return (
        f"There are review-worthy signals, but not a forced emergency: risk is "
        f"{current_state.risk_status.upper()}, largest target change is {max_abs_delta:.1%}, "
        f"active news events are {active_news}, and open tickets are {open_ticket_count}."
    )


def _as_float(value: object) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric
