from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics, _format_decimal, _format_percent
from trade_bot.research.action_headline import ActionHeadline
from trade_bot.research.baselines import BaselineRun


@dataclass(frozen=True)
class MacroMinuteCard:
    label: str
    answer: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True)
class MacroMinuteReport:
    tone: str
    title: str
    summary: str
    paragraphs: tuple[str, ...]
    next_step: str
    daily_delta_cards: tuple[MacroMinuteCard, ...]
    cards: tuple[MacroMinuteCard, ...]
    detail_rows: pd.DataFrame


def _render_macro_minute(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    previous_run: BaselineRun | None = None,
) -> None:
    report = build_macro_minute_report(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=open_ticket_count,
        previous_run=previous_run,
    )
    paragraph_html = "".join(
        f'<p class="macro-minute-copy">{html.escape(paragraph)}</p>'
        for paragraph in report.paragraphs
    )
    daily_delta_html = (
        '<div class="macro-minute-delta-grid">'
        + "".join(_macro_minute_delta_html(card) for card in report.daily_delta_cards)
        + "</div>"
    )
    st.markdown(
        f"""
        <div class="macro-minute macro-minute-{html.escape(report.tone)}">
            <p class="macro-minute-label">Macro Minute</p>
            <div class="macro-minute-title">{html.escape(report.title)}</div>
            {daily_delta_html}
            <div class="macro-minute-body">{paragraph_html}</div>
            <p class="macro-minute-next">Next: {html.escape(report.next_step)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="macro-minute-readouts">'
        + "".join(_macro_minute_readout_html(card) for card in report.cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Macro Minute detail", expanded=False):
        _render_metric_dataframe(_display_metrics(report.detail_rows), hide_index=True)


def build_macro_minute_report(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    previous_run: BaselineRun | None = None,
) -> MacroMinuteReport:
    current_state = baseline_run.current_state
    trade_summary = _first_row(baseline_run.trade_decision.summary)
    scenario_links = baseline_run.trade_decision.scenario_links
    risk_summary = _portfolio_risk_summary(baseline_run)
    news_summary = _news_summary(baseline_run)
    scenario_summary = _scenario_summary(
        trade_summary, scenario_links, current_state.scenario_lattice
    )
    macro_summary = _macro_summary(current_state.macro_category_summary, baseline_run.macro_data)
    confirmation_summary = _confirmation_summary(current_state.confirmation_matrix)
    change_summary = _change_summary(baseline_run, previous_run)

    action = str(trade_summary.get("recommended_action", headline.label)).replace("_", " ").title()
    risk_status = current_state.risk_status.upper()
    risk_budget = _format_decimal(trade_summary.get("risk_budget_multiplier", "n/a"))
    tone = _macro_minute_tone(headline.level, current_state.risk_status)
    title = f"{current_state.market_date}: {risk_status} risk, {action}"
    posture = _posture_sentence(trade_summary)
    driver_summary = _driver_summary(current_state.scenario_drivers)
    trade_change = _trade_change_summary(baseline_run.trade_decision.position_plan)
    watch_summary = _watch_summary(scenario_links, current_state.scenario_lattice)
    summary = (
        f"{risk_status} risk with {action}. {scenario_summary['sentence']} "
        f"{news_summary['sentence']} {macro_summary['sentence']} {change_summary['sentence']}"
    )
    daily_delta_cards = _daily_delta_cards(
        change_summary=change_summary,
        scenario_summary=scenario_summary,
        news_summary=news_summary,
        action=action,
        risk_budget=risk_budget,
    )
    paragraphs = (
        _market_read_paragraph(
            market_date=str(current_state.market_date),
            risk_status=risk_status,
            risk_score=float(current_state.risk_score),
            action=action,
            confirmation_summary=confirmation_summary,
            scenario_summary=scenario_summary,
            posture=posture,
        ),
        _change_paragraph(change_summary),
        _driver_paragraph(
            news_summary=news_summary,
            macro_summary=macro_summary,
            driver_summary=driver_summary,
        ),
        _action_paragraph(
            action=action,
            risk_budget=risk_budget,
            risk_summary=risk_summary,
            trade_change=trade_change,
            watch_summary=watch_summary,
        ),
    )
    next_step = headline.next_action
    if open_ticket_count:
        next_step = f"Review {open_ticket_count} open ticket(s) before changing exposure."

    cards = (
        MacroMinuteCard(
            label="Market State",
            answer=f"{risk_status} risk ({current_state.risk_score:.2f})",
            detail=f"{confirmation_summary}. {current_state.risk_summary}",
            tone=_status_tone(current_state.risk_status),
        ),
        MacroMinuteCard(
            label="Change Since Prior",
            answer=str(change_summary["answer"]),
            detail=str(change_summary["detail"]),
            tone=str(change_summary["tone"]),
        ),
        MacroMinuteCard(
            label="News / Events",
            answer=news_summary["answer"],
            detail=news_summary["detail"],
            tone=news_summary["tone"],
        ),
        MacroMinuteCard(
            label="Scenario Map",
            answer=scenario_summary["answer"],
            detail=scenario_summary["detail"],
            tone=scenario_summary["tone"],
        ),
        MacroMinuteCard(
            label="Risk Budget / Action",
            answer=f"{risk_budget} budget; {action}",
            detail=_risk_budget_detail(trade_summary, risk_summary),
            tone=_brief_tone(headline.level),
        ),
    )
    return MacroMinuteReport(
        tone=tone,
        title=title,
        summary=summary,
        paragraphs=paragraphs,
        next_step=next_step,
        daily_delta_cards=daily_delta_cards,
        cards=cards,
        detail_rows=_macro_minute_details(
            current_state=current_state,
            trade_summary=trade_summary,
            news_summary=news_summary,
            scenario_summary=scenario_summary,
            macro_summary=macro_summary,
            risk_summary=risk_summary,
            change_summary=change_summary,
            open_ticket_count=open_ticket_count,
        ),
    )


def _macro_minute_card_html(card: MacroMinuteCard) -> str:
    return (
        f'<div class="macro-minute-card macro-minute-card-{html.escape(card.tone)}">'
        f'<p class="macro-card-label">{html.escape(card.label)}</p>'
        f'<p class="macro-card-answer">{html.escape(card.answer)}</p>'
        f'<p class="macro-card-detail">{html.escape(card.detail)}</p>'
        "</div>"
    )


def _macro_minute_readout_html(card: MacroMinuteCard) -> str:
    return (
        f'<div class="macro-minute-readout macro-minute-readout-{html.escape(card.tone)}">'
        f'<span class="macro-readout-label">{html.escape(card.label)}</span>'
        f'<span class="macro-readout-answer">{html.escape(card.answer)}</span>'
        "</div>"
    )


def _macro_minute_delta_html(card: MacroMinuteCard) -> str:
    return (
        f'<div class="macro-delta-card macro-delta-card-{html.escape(card.tone)}">'
        f'<p class="macro-delta-label">{html.escape(card.label)}</p>'
        f'<p class="macro-delta-answer">{html.escape(card.answer)}</p>'
        f'<p class="macro-delta-detail">{html.escape(card.detail)}</p>'
        "</div>"
    )


def _daily_delta_cards(
    *,
    change_summary: dict[str, object],
    scenario_summary: dict[str, str],
    news_summary: dict[str, str],
    action: str,
    risk_budget: str,
) -> tuple[MacroMinuteCard, ...]:
    changed_answer = str(change_summary.get("changed_answer", change_summary["answer"]))
    changed_detail = str(change_summary.get("changed_detail", change_summary["detail"]))
    still_answer = str(change_summary.get("still_answer", f"{action}; {risk_budget} risk budget"))
    still_detail = str(
        change_summary.get(
            "still_detail",
            (
                f"Current basis remains {scenario_summary['answer']}; "
                f"{news_summary['answer'].lower()}; {risk_budget} risk budget."
            ),
        )
    )
    return (
        MacroMinuteCard(
            label="What Changed Today",
            answer=changed_answer,
            detail=changed_detail,
            tone=str(change_summary.get("tone", "neutral")),
        ),
        MacroMinuteCard(
            label="Still True",
            answer=still_answer,
            detail=still_detail,
            tone="neutral",
        ),
    )


def _market_read_paragraph(
    *,
    market_date: str,
    risk_status: str,
    risk_score: float,
    action: str,
    confirmation_summary: str,
    scenario_summary: dict[str, str],
    posture: str,
) -> str:
    return (
        f"Current posture: as of {market_date}, the system is reading {risk_status} risk ({risk_score:.2f}) "
        f"and the operating action is {action}. {confirmation_summary}. "
        f"The one-month map is not cleanly risk-on: {scenario_summary['answer']}, with "
        f"{scenario_summary['top_scenario']} as the lead scenario. {posture}"
    )


def _driver_paragraph(
    *,
    news_summary: dict[str, str],
    macro_summary: dict[str, str],
    driver_summary: str,
) -> str:
    return (
        f"Still true: the driver stack is being pulled most by {news_summary['plain']}. "
        f"Macro is {macro_summary['plain']}. Scenario drivers point to {driver_summary}. "
        "That combination matters because the bot should not treat price trend alone as enough "
        "when news, macro, credit, liquidity, or breadth are arguing for smaller sizing."
    )


def _action_paragraph(
    *,
    action: str,
    risk_budget: str,
    risk_summary: dict[str, object],
    trade_change: str,
    watch_summary: str,
) -> str:
    constraints = str(risk_summary.get("applied_constraints", "none")) if risk_summary else "none"
    return (
        f"Action read-through: {action} with risk budget {risk_budget}. {trade_change} "
        f"The risk engine is applying {constraints}. What would change this: {watch_summary}"
    )


def _change_paragraph(change_summary: dict[str, object]) -> str:
    return str(change_summary.get("paragraph", ""))


def _change_summary(
    baseline_run: BaselineRun, previous_run: BaselineRun | None
) -> dict[str, object]:
    current = _run_change_snapshot(baseline_run)
    if previous_run is None:
        return {
            "answer": "No prior snapshot",
            "detail": "No stored prior run is available for comparison yet.",
            "tone": "neutral",
            "sentence": "No prior snapshot is available, so today's Macro Minute is the comparison baseline.",
            "paragraph": (
                "Daily delta: no prior snapshot is available yet. Treat today's posture as the baseline; "
                "future refreshes will separate persistent messaging from genuinely new information."
            ),
            "changed_answer": "No prior comparison",
            "changed_detail": "Today's snapshot is the baseline for future daily-change checks.",
            "still_answer": "Baseline posture set",
            "still_detail": f"Current baseline: {_snapshot_read(current)}.",
            "previous_read": "n/a",
            "current_read": _snapshot_read(current),
        }

    previous = _run_change_snapshot(previous_run)
    changes = _material_changes(previous, current)
    previous_read = _snapshot_read(previous)
    current_read = _snapshot_read(current)
    if not changes:
        return {
            "answer": "Mostly unchanged",
            "detail": f"Previous: {previous_read}. Now: {current_read}. No monitored change crossed a material threshold.",
            "tone": "neutral",
            "sentence": "The current posture is largely unchanged from the prior stored run.",
            "paragraph": (
                f"Daily delta: no monitored change crossed a material threshold. Prior stored posture was {previous_read}; "
                f"now it is {current_read}. The dashboard language is repeating because the system still sees the same broad setup, "
                "not because a fresh escalation was detected."
            ),
            "changed_answer": "No material change",
            "changed_detail": "No monitored variable crossed a material threshold from the prior stored snapshot.",
            "still_answer": "Same broad setup",
            "still_detail": f"Prior posture and current posture are aligned: {current_read}.",
            "previous_read": previous_read,
            "current_read": current_read,
        }

    tone = _change_tone(previous, current)
    change_text = "; ".join(changes[:6])
    return {
        "answer": _change_answer(previous, current, changes),
        "detail": f"Previous: {previous_read}. Now: {current_read}. Material changes: {change_text}.",
        "tone": tone,
        "sentence": f"Since the prior stored run: {change_text}.",
        "paragraph": (
            f"Daily delta: prior stored posture was {previous_read}; now it is {current_read}. "
            f"What changed: {change_text}. Read this as the reason today's recommendation is different, "
            "or as confirmation that the same risk posture is persisting if the action did not move."
        ),
        "changed_answer": _change_answer(previous, current, changes),
        "changed_detail": f"Material changes: {change_text}.",
        "still_answer": "Updated posture now active",
        "still_detail": f"Current posture after the change check: {current_read}.",
        "previous_read": previous_read,
        "current_read": current_read,
    }


def _run_change_snapshot(run: BaselineRun) -> dict[str, object]:
    trade_summary = _first_row(run.trade_decision.summary)
    news_counts = _news_counts(run)
    macro_pressure_count = _macro_pressure_count(run.current_state.macro_category_summary)
    return {
        "market_date": str(run.current_state.market_date),
        "risk_status": str(run.current_state.risk_status).upper(),
        "risk_score": float(run.current_state.risk_score),
        "action": str(trade_summary.get("recommended_action", "n/a")).replace("_", " ").title(),
        "risk_budget": _as_float(trade_summary.get("risk_budget_multiplier")),
        "risk_off": _as_float(trade_summary.get("one_month_risk_off_probability")),
        "transition": _as_float(trade_summary.get("one_month_transition_probability")),
        "risk_on": _as_float(trade_summary.get("one_month_risk_on_probability")),
        "event_pressure": _as_float(trade_summary.get("event_pressure")),
        "macro_pressure": _as_float(trade_summary.get("macro_pressure")),
        "active_events": int(news_counts["active_events"]),
        "activated_news": int(news_counts["activated_news"]),
        "high_urgency_news": int(news_counts["high_urgency_news"]),
        "macro_pressure_groups": macro_pressure_count,
        "top_scenario": _top_scenario(
            run.trade_decision.scenario_links,
            run.current_state.scenario_lattice,
        ),
        "target_position": str(trade_summary.get("scenario_adjusted_position", "")).strip(),
    }


def _snapshot_read(snapshot: dict[str, object]) -> str:
    return (
        f"{snapshot['market_date']} {snapshot['risk_status']} risk, "
        f"{snapshot['action']}, budget {_format_decimal(snapshot['risk_budget'])}, "
        f"target {snapshot.get('target_position') or 'n/a'}"
    )


def _material_changes(previous: dict[str, object], current: dict[str, object]) -> list[str]:
    changes: list[str] = []
    if previous["risk_status"] != current["risk_status"]:
        changes.append(f"risk status {previous['risk_status']} -> {current['risk_status']}")
    if previous["action"] != current["action"]:
        changes.append(f"action {previous['action']} -> {current['action']}")
    _append_numeric_change(
        changes,
        label="risk score",
        previous=_as_float(previous["risk_score"]),
        current=_as_float(current["risk_score"]),
        threshold=0.03,
        percent=False,
    )
    _append_numeric_change(
        changes,
        label="risk budget",
        previous=_as_float(previous["risk_budget"]),
        current=_as_float(current["risk_budget"]),
        threshold=0.05,
        percent=False,
    )
    _append_numeric_change(
        changes,
        label="1M risk-off probability",
        previous=_as_float(previous["risk_off"]),
        current=_as_float(current["risk_off"]),
        threshold=0.03,
        percent=True,
    )
    _append_numeric_change(
        changes,
        label="1M transition probability",
        previous=_as_float(previous["transition"]),
        current=_as_float(current["transition"]),
        threshold=0.05,
        percent=True,
    )
    _append_numeric_change(
        changes,
        label="active events",
        previous=float(previous["active_events"]),
        current=float(current["active_events"]),
        threshold=0.5,
        percent=False,
        integer=True,
    )
    _append_numeric_change(
        changes,
        label="activated news",
        previous=float(previous["activated_news"]),
        current=float(current["activated_news"]),
        threshold=0.5,
        percent=False,
        integer=True,
    )
    _append_numeric_change(
        changes,
        label="macro pressure groups",
        previous=float(previous["macro_pressure_groups"]),
        current=float(current["macro_pressure_groups"]),
        threshold=0.5,
        percent=False,
        integer=True,
    )
    if previous["top_scenario"] != current["top_scenario"]:
        changes.append(f"top 1M scenario {previous['top_scenario']} -> {current['top_scenario']}")
    if previous.get("target_position") != current.get("target_position"):
        changes.append("target posture changed")
    return changes


def _append_numeric_change(
    changes: list[str],
    *,
    label: str,
    previous: float,
    current: float,
    threshold: float,
    percent: bool,
    integer: bool = False,
) -> None:
    delta = current - previous
    if abs(delta) < threshold:
        return
    if integer:
        changes.append(f"{label} {int(previous)} -> {int(current)}")
    elif percent:
        changes.append(f"{label} {previous:.0%} -> {current:.0%} ({delta * 100:+.0f}pp)")
    else:
        changes.append(f"{label} {previous:.2f} -> {current:.2f} ({delta:+.2f})")


def _change_tone(previous: dict[str, object], current: dict[str, object]) -> str:
    deterioration = 0
    improvement = 0
    if _as_float(current["risk_score"]) - _as_float(previous["risk_score"]) >= 0.05:
        deterioration += 1
    if _as_float(previous["risk_score"]) - _as_float(current["risk_score"]) >= 0.05:
        improvement += 1
    if _as_float(current["risk_off"]) - _as_float(previous["risk_off"]) >= 0.05:
        deterioration += 1
    if _as_float(previous["risk_off"]) - _as_float(current["risk_off"]) >= 0.05:
        improvement += 1
    if _as_float(previous["risk_budget"]) - _as_float(current["risk_budget"]) >= 0.05:
        deterioration += 1
    if _as_float(current["risk_budget"]) - _as_float(previous["risk_budget"]) >= 0.05:
        improvement += 1
    if int(current["active_events"]) > int(previous["active_events"]):
        deterioration += 1
    if int(current["active_events"]) < int(previous["active_events"]):
        improvement += 1
    if deterioration > improvement:
        return "critical" if deterioration >= 2 else "warning"
    if improvement > deterioration:
        return "success"
    return "warning"


def _change_answer(
    previous: dict[str, object], current: dict[str, object], changes: list[str]
) -> str:
    if previous["action"] != current["action"]:
        return f"Action changed to {current['action']}"
    if previous["risk_status"] != current["risk_status"]:
        return f"Risk changed to {current['risk_status']}"
    if changes:
        return f"{len(changes)} material change(s)"
    return "Mostly unchanged"


def _news_counts(run: BaselineRun) -> dict[str, int]:
    triage = run.news_monitor.triage
    active_events = [event for event in run.event_risk.events if event.current]
    high_urgency = 0
    if not triage.empty and "urgency_score" in triage:
        high_urgency = int((pd.to_numeric(triage["urgency_score"], errors="coerce") >= 0.80).sum())
    activated = 0
    if not triage.empty and "activation_status" in triage:
        activated = int(triage["activation_status"].astype(str).str.contains("event_risk").sum())
    return {
        "active_events": len(active_events),
        "activated_news": activated,
        "high_urgency_news": high_urgency,
    }


def _macro_pressure_count(macro_category_summary: pd.DataFrame) -> int:
    if macro_category_summary.empty:
        return 0
    pressure_columns = [
        column
        for column in (
            "risk_state",
            "near_term_state",
            "latest_pressure_state",
            "state",
            "signal_state",
        )
        if column in macro_category_summary
    ]
    if not pressure_columns:
        return 0
    return int(
        macro_category_summary[pressure_columns[0]]
        .astype(str)
        .str.contains("risk_pressure")
        .sum()
    )


def _posture_sentence(trade_summary: dict[str, object]) -> str:
    base_position = str(trade_summary.get("base_position", "")).strip()
    adjusted_position = str(trade_summary.get("scenario_adjusted_position", "")).strip()
    if base_position and adjusted_position and base_position != adjusted_position:
        return (
            f"The scenario-adjusted posture moves from {base_position} toward {adjusted_position}."
        )
    if adjusted_position:
        return f"The scenario-adjusted posture is {adjusted_position}."
    return "The scenario layer is informing sizing before final risk constraints are applied."


def _trade_change_summary(position_plan: pd.DataFrame) -> str:
    if position_plan.empty or "delta_weight" not in position_plan:
        return "No material position change is visible in the current position plan."
    frame = position_plan.copy()
    frame["abs_delta"] = pd.to_numeric(frame["delta_weight"], errors="coerce").abs()
    frame = frame.dropna(subset=["abs_delta"]).sort_values("abs_delta", ascending=False)
    frame = frame[frame["abs_delta"] > 0.005]
    if frame.empty:
        return "No material position change is visible in the current position plan."
    parts = []
    for _, row in frame.head(3).iterrows():
        ticker = str(row.get("ticker", "n/a"))
        action = str(row.get("action", "adjust")).replace("_", " ").lower()
        delta_value = abs(_as_float(row.get("delta_weight")))
        delta = _format_percent(delta_value)
        parts.append(f"{action} {ticker} by {delta}")
    return "Position sizing translation: " + "; ".join(parts) + "."


def _watch_summary(scenario_links: pd.DataFrame, scenario_lattice: pd.DataFrame) -> str:
    row = _top_scenario_row(scenario_links, scenario_lattice)
    if not row:
        return "wait for breadth, credit, volatility, and trend confirmation before overriding the current action."
    confirmation = str(row.get("confirmation", "")).strip()
    off_ramp = str(row.get("off_ramp", "")).strip()
    if confirmation and off_ramp:
        confirmation = confirmation.rstrip(".;")
        off_ramp = off_ramp.rstrip(".;")
        return f"confirmation would be {confirmation}; invalidation/off-ramp is {off_ramp}."
    if confirmation:
        return f"confirmation would be {confirmation}"
    if off_ramp:
        return f"invalidation/off-ramp is {off_ramp}"
    return "wait for breadth, credit, volatility, and trend confirmation before overriding the current action."


def _driver_summary(scenario_drivers: pd.DataFrame) -> str:
    if (
        scenario_drivers.empty
        or "driver" not in scenario_drivers
        or "score" not in scenario_drivers
    ):
        return "mixed confirmation, with no dominant scenario-driver table available"
    frame = scenario_drivers.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score"])
    if frame.empty:
        return "mixed confirmation, with no usable scenario-driver scores"
    weakest = frame.sort_values("score").head(2)
    strongest = frame.sort_values("score", ascending=False).head(2)
    weak_text = ", ".join(f"{row['driver']} ({row['score']:.2f})" for _, row in weakest.iterrows())
    strong_text = ", ".join(
        f"{row['driver']} ({row['score']:.2f})" for _, row in strongest.iterrows()
    )
    return f"weakest drivers {weak_text}; strongest offsets {strong_text}"


def _scenario_summary(
    trade_summary: dict[str, object],
    scenario_links: pd.DataFrame,
    scenario_lattice: pd.DataFrame,
) -> dict[str, str]:
    risk_off = _format_percent(trade_summary.get("one_month_risk_off_probability"))
    transition = _format_percent(trade_summary.get("one_month_transition_probability"))
    risk_on = _format_percent(trade_summary.get("one_month_risk_on_probability"))
    fragile = _format_percent(trade_summary.get("one_month_fragile_upside_probability"))
    top_scenario = _top_scenario(scenario_links, scenario_lattice)
    answer = f"{risk_off} risk-off; {transition} transition"
    detail = f"Top 1M scenario: {top_scenario}. Risk-on is {risk_on}; fragile upside is {fragile}."
    tone = (
        "critical"
        if _as_float(trade_summary.get("one_month_risk_off_probability")) >= 0.35
        else "warning"
    )
    if _as_float(trade_summary.get("one_month_risk_on_probability")) >= 0.45:
        tone = "success" if tone != "critical" else tone
    return {
        "answer": answer,
        "detail": detail,
        "tone": tone,
        "top_scenario": top_scenario,
        "sentence": f"The 1-month scenario map shows {risk_off} risk-off and {transition} transition pressure.",
    }


def _news_summary(baseline_run: BaselineRun) -> dict[str, str]:
    triage = baseline_run.news_monitor.triage
    events = baseline_run.event_risk.events
    active_events = [event for event in events if event.current]
    event_pressure = _as_float(
        _first_row(baseline_run.trade_decision.summary).get("event_pressure")
    )
    if triage.empty and not active_events:
        return {
            "answer": "No active items",
            "detail": "No current news/event pressure is active in the latest run.",
            "tone": "success",
            "sentence": "News and event pressure are quiet.",
            "plain": "quiet news and event pressure",
        }
    high_urgency = 0
    if "urgency_score" in triage:
        high_urgency = int((pd.to_numeric(triage["urgency_score"], errors="coerce") >= 0.80).sum())
    activated = 0
    if "activation_status" in triage:
        activated = int(triage["activation_status"].astype(str).str.contains("event_risk").sum())
    categories = _top_values(triage, "category", limit=3, exclude=("unclassified",))
    top_items = _top_news_items(triage)
    active_event_text = _active_event_text(active_events)
    answer = f"{len(active_events)} active events; {activated} activated news"
    detail = (
        f"High-urgency news items: {high_urgency}. Event pressure: {_format_percent(event_pressure)}. "
        f"Top categories: {categories or 'n/a'}. Top items: {top_items or 'n/a'}."
    )
    tone = "critical" if event_pressure >= 0.12 or high_urgency >= 2 else "warning"
    sentence = (
        f"News/event pressure is active with {len(active_events)} curated event(s) and "
        f"{activated} activated news item(s)."
    )
    plain = (
        f"{len(active_events)} active event(s), {activated} activated news item(s), "
        f"{high_urgency} high-urgency item(s), categories {categories or 'n/a'}"
    )
    if top_items:
        plain = f"{plain}; top headlines include {top_items}"
    elif active_event_text:
        plain = f"{plain}; active events include {active_event_text}"
    return {
        "answer": answer,
        "detail": detail,
        "tone": tone,
        "sentence": sentence,
        "plain": plain,
    }


def _macro_summary(
    macro_category_summary: pd.DataFrame, macro_data: pd.DataFrame
) -> dict[str, str]:
    loaded = int(macro_data.shape[1]) if macro_data is not None else 0
    if macro_category_summary.empty:
        return {
            "answer": f"{loaded} macro series loaded",
            "detail": "No macro category pressure table is available in the latest run.",
            "tone": "warning" if loaded == 0 else "success",
            "sentence": f"Macro coverage has {loaded} loaded series.",
            "plain": f"limited in this run: {loaded} loaded macro series and no category pressure table",
        }
    pressure_columns = [
        column
        for column in (
            "risk_state",
            "near_term_state",
            "latest_pressure_state",
            "state",
            "signal_state",
        )
        if column in macro_category_summary
    ]
    pressure_count = 0
    if pressure_columns:
        state_column = pressure_columns[0]
        pressure_count = int(
            macro_category_summary[state_column].astype(str).str.contains("risk_pressure").sum()
        )
    top_groups = _top_macro_groups(macro_category_summary, limit=3)
    pressure_text = f"{pressure_count} pressure group(s) across {loaded} loaded series"
    if top_groups:
        pressure_text = f"{pressure_text}; most relevant groups: {top_groups}"
    return {
        "answer": f"{loaded} series; {pressure_count} pressure groups",
        "detail": f"Macro pressure groups: {pressure_count}. Top monitored groups: {top_groups or 'n/a'}.",
        "tone": "warning" if pressure_count else "success",
        "sentence": f"Macro coverage has {loaded} loaded series and {pressure_count} pressure group(s).",
        "plain": pressure_text,
    }


def _confirmation_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No confirmation matrix is available"
    state_column = "state" if "state" in frame else "status" if "status" in frame else ""
    if not state_column:
        return "No confirmation matrix is available"
    counts = frame[state_column].astype(str).str.lower().value_counts().to_dict()
    bullish = int(counts.get("bullish", 0))
    neutral = int(counts.get("neutral", 0))
    bearish = int(counts.get("bearish", 0))
    return f"Confirmation matrix: {bullish} bullish, {neutral} neutral, {bearish} bearish"


def _risk_budget_detail(trade_summary: dict[str, object], risk_summary: dict[str, object]) -> str:
    constraints = str(risk_summary.get("applied_constraints", "none")) if risk_summary else "none"
    return (
        f"Scenario/event/macro multiplier: {_format_decimal(trade_summary.get('scenario_event_macro_multiplier'))}; "
        f"portfolio-risk multiplier: {_format_decimal(trade_summary.get('portfolio_risk_multiplier'))}; "
        f"constraints: {constraints}."
    )


def _macro_minute_details(
    *,
    current_state: object,
    trade_summary: dict[str, object],
    news_summary: dict[str, str],
    scenario_summary: dict[str, str],
    macro_summary: dict[str, str],
    risk_summary: dict[str, object],
    change_summary: dict[str, object],
    open_ticket_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topic": "market_state",
                "current_read": f"{current_state.risk_status.upper()} ({current_state.risk_score:.2f})",
                "why_it_matters": current_state.risk_summary,
            },
            {
                "topic": "scenario_map",
                "current_read": scenario_summary["answer"],
                "why_it_matters": scenario_summary["detail"],
            },
            {
                "topic": "change_since_prior",
                "previous_read": change_summary.get("previous_read", "n/a"),
                "current_read": change_summary.get("current_read", change_summary["answer"]),
                "why_it_matters": change_summary["detail"],
            },
            {
                "topic": "news_events",
                "current_read": news_summary["answer"],
                "why_it_matters": news_summary["detail"],
            },
            {
                "topic": "macro_stack",
                "current_read": macro_summary["answer"],
                "why_it_matters": macro_summary["detail"],
            },
            {
                "topic": "risk_budget",
                "current_read": _format_decimal(trade_summary.get("risk_budget_multiplier")),
                "why_it_matters": _risk_budget_detail(trade_summary, risk_summary),
            },
            {
                "topic": "open_tickets",
                "current_read": f"{open_ticket_count}",
                "why_it_matters": "Open locked recommendations need review before the next execution window.",
            },
        ]
    )


def _top_news_items(triage: pd.DataFrame) -> str:
    if triage.empty or "title" not in triage:
        return ""
    frame = triage.copy()
    if "urgency_score" in frame:
        frame["urgency_score"] = pd.to_numeric(frame["urgency_score"], errors="coerce")
        frame = frame.sort_values("urgency_score", ascending=False)
    titles = []
    for title in frame["title"].dropna().astype(str):
        clean = title.strip()
        if clean and clean not in titles:
            titles.append(clean)
        if len(titles) >= 2:
            break
    return "; ".join(titles)


def _active_event_text(active_events: list[object]) -> str:
    parts = []
    for event in active_events[:2]:
        name = str(getattr(event, "name", "")).strip()
        category = str(getattr(event, "category", "")).strip()
        if name and category:
            parts.append(f"{name} ({category})")
        elif name:
            parts.append(name)
    return "; ".join(parts)


def _top_scenario_row(
    scenario_links: pd.DataFrame, scenario_lattice: pd.DataFrame
) -> dict[str, object]:
    if not scenario_links.empty:
        return scenario_links.iloc[0].to_dict()
    if scenario_lattice.empty:
        return {}
    frame = scenario_lattice.copy()
    if "horizon" in frame:
        one_month = frame[frame["horizon"] == "1m"]
        if not one_month.empty:
            frame = one_month
    if "rank" in frame:
        frame = frame.sort_values("rank")
    elif "probability" in frame:
        frame = frame.sort_values("probability", ascending=False)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _top_scenario(scenario_links: pd.DataFrame, scenario_lattice: pd.DataFrame) -> str:
    row = _top_scenario_row(scenario_links, scenario_lattice)
    return str(row.get("scenario", "n/a")) if row else "n/a"


def _portfolio_risk_summary(baseline_run: BaselineRun) -> dict[str, object]:
    risk = baseline_run.portfolio_risk or baseline_run.trade_decision.portfolio_risk
    if risk is None or risk.summary.empty:
        return {}
    return risk.summary.iloc[0].to_dict()


def _first_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _top_macro_groups(frame: pd.DataFrame, *, limit: int) -> str:
    if frame.empty:
        return ""
    group_column = (
        "signal_group" if "signal_group" in frame else "category" if "category" in frame else ""
    )
    if not group_column:
        return ""
    working = frame.copy()
    if "mean_risk_score" in working:
        working["mean_risk_score"] = pd.to_numeric(working["mean_risk_score"], errors="coerce")
        working = working.sort_values("mean_risk_score", ascending=False)
    values = []
    for value in working[group_column].dropna().astype(str):
        if value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return ", ".join(values)


def _top_values(
    frame: pd.DataFrame,
    column: str,
    *,
    limit: int,
    exclude: tuple[str, ...] = (),
) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    if exclude:
        excluded = {value.lower() for value in exclude}
        values = values[~values.str.lower().isin(excluded)]
    if values.empty:
        return ""
    return ", ".join(values.value_counts().head(limit).index)


def _as_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric


def _macro_minute_tone(headline_level: str, risk_status: str) -> str:
    if headline_level == "critical_actions" or risk_status.lower() in {"red", "orange"}:
        return "critical"
    if headline_level == "small_actions" or risk_status.lower() == "yellow":
        return "warning"
    return "success"


def _status_tone(risk_status: str) -> str:
    if risk_status.lower() in {"red", "orange"}:
        return "critical"
    if risk_status.lower() == "yellow":
        return "warning"
    return "success"


def _brief_tone(level: str) -> str:
    if level == "critical_actions":
        return "critical"
    if level == "small_actions":
        return "warning"
    return "success"
