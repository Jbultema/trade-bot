from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.portfolio.risk import (
    PortfolioRiskConfig,
    PortfolioRiskRun,
    build_portfolio_risk,
)
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun, MarketEvent
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun


@dataclass(frozen=True)
class TradeDecisionRun:
    summary: pd.DataFrame
    position_plan: pd.DataFrame
    evidence: pd.DataFrame
    scenario_links: pd.DataFrame
    portfolio_risk: PortfolioRiskRun | None = None


def build_trade_decision(
    *,
    primary_result: BacktestResult,
    current_state: CurrentStateRun,
    event_risk: EventRiskRun,
    news_monitor: NewsMonitorRun,
    signal_inclusion: SignalInclusionRun,
    prices: pd.DataFrame | None = None,
    defensive_ticker: str = "BIL",
    min_trade_weight: float = 0.02,
) -> TradeDecisionRun:
    base_weights = primary_result.weights.iloc[-1].astype(float)
    scenario_context = _scenario_context(current_state.scenario_lattice)
    event_context = _event_context(event_risk.events)
    macro_context = _macro_context(signal_inclusion.summary)

    risk_multiplier = min(
        _risk_status_multiplier(current_state.risk_status),
        _as_float(scenario_context["risk_multiplier"]),
        _as_float(event_context["risk_multiplier"]),
        _as_float(macro_context["risk_multiplier"]),
    )
    adjusted_weights = _scenario_adjusted_weights(
        base_weights,
        risk_multiplier=risk_multiplier,
        defensive_ticker=defensive_ticker,
    )
    portfolio_risk = _build_portfolio_risk_if_available(
        prices=prices,
        base_weights=base_weights,
        adjusted_weights=adjusted_weights,
        current_state=current_state,
        defensive_ticker=defensive_ticker,
    )
    portfolio_risk_context = _portfolio_risk_context(portfolio_risk)
    if portfolio_risk is not None and not portfolio_risk.risk_adjusted_weights.empty:
        final_weights = portfolio_risk.risk_adjusted_weights
    else:
        final_weights = adjusted_weights
    portfolio_risk_multiplier = _as_float(portfolio_risk_context["portfolio_risk_multiplier"])
    total_risk_budget_multiplier = float(np.clip(risk_multiplier * portfolio_risk_multiplier, 0, 1))
    position_plan = _position_plan(base_weights, final_weights, min_trade_weight)
    position_plan = _add_portfolio_risk_sizing_columns(position_plan, portfolio_risk)
    action = _recommended_action(position_plan, current_state.risk_status)
    explanation = _human_explanation(
        action=action,
        base_weights=base_weights,
        adjusted_weights=final_weights,
        current_state=current_state,
        scenario_context=scenario_context,
        event_context=event_context,
        macro_context=macro_context,
        portfolio_risk_context=portfolio_risk_context,
    )
    authority = _decision_authority(
        scenario_context,
        event_context,
        macro_context,
        portfolio_risk_context,
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": primary_result.name,
                "recommended_action": action,
                "decision_authority": authority,
                "base_position": _format_weight_vector(base_weights),
                "pre_risk_target_position": _format_weight_vector(adjusted_weights),
                "scenario_adjusted_position": _format_weight_vector(final_weights),
                "risk_budget_multiplier": total_risk_budget_multiplier,
                "scenario_event_macro_multiplier": risk_multiplier,
                "portfolio_risk_multiplier": portfolio_risk_multiplier,
                "risk_status": current_state.risk_status,
                "risk_score": current_state.risk_score,
                "one_month_risk_off_probability": scenario_context["risk_off_probability"],
                "one_month_transition_probability": scenario_context["transition_probability"],
                "one_month_fragile_upside_probability": scenario_context[
                    "fragile_upside_probability"
                ],
                "event_pressure": event_context["event_pressure"],
                "macro_pressure": macro_context["macro_pressure"],
                "portfolio_risk_level": portfolio_risk_context["portfolio_risk_level"],
                "portfolio_constraints": portfolio_risk_context["applied_constraints"],
                "portfolio_expected_shortfall_95": portfolio_risk_context[
                    "post_expected_shortfall_95"
                ],
                "portfolio_max_stress_loss": portfolio_risk_context["post_max_stress_loss"],
                "portfolio_equity_beta": portfolio_risk_context["post_equity_beta"],
                "portfolio_ai_beta": portfolio_risk_context["post_ai_beta"],
                "human_explanation": explanation,
            }
        ]
    )
    evidence = _evidence_table(
        current_state=current_state,
        scenario_context=scenario_context,
        event_context=event_context,
        macro_context=macro_context,
        portfolio_risk_context=portfolio_risk_context,
        news_monitor=news_monitor,
    )
    scenario_links = _scenario_links(current_state.scenario_lattice)
    return TradeDecisionRun(
        summary=summary,
        position_plan=position_plan,
        evidence=evidence,
        scenario_links=scenario_links,
        portfolio_risk=portfolio_risk,
    )


def _scenario_context(scenario_lattice: pd.DataFrame) -> dict[str, object]:
    if scenario_lattice.empty:
        return {
            "risk_multiplier": 1.0,
            "risk_off_probability": 0.0,
            "transition_probability": 0.0,
            "fragile_upside_probability": 0.0,
            "top_scenarios": (),
            "evidence": "No scenario lattice available.",
        }
    one_month = scenario_lattice[scenario_lattice["horizon"] == "1m"].copy()
    if one_month.empty:
        one_month = scenario_lattice.copy()

    risk_bucket = one_month["risk_bucket"].astype(str)
    risk_off_probability = float(
        one_month.loc[risk_bucket.str.contains("risk_off"), "probability"].sum()
    )
    transition_probability = float(
        (one_month.loc[risk_bucket == "transition", "probability"]).sum()
    )
    fragile_probability = float(
        (one_month.loc[risk_bucket == "risk_on_fragile", "probability"]).sum()
    )
    risk_multiplier = 1.0 - 0.55 * risk_off_probability
    risk_multiplier -= 0.20 * transition_probability
    risk_multiplier -= 0.15 * fragile_probability
    risk_multiplier = float(np.clip(risk_multiplier, 0.40, 1.0))
    top_scenarios = tuple(
        one_month.sort_values("rank")
        .head(3)[["scenario", "probability", "risk_bucket", "expected_bot_posture"]]
        .to_dict("records")
    )
    evidence = "; ".join(
        f"{row['scenario']} ({float(row['probability']):.0%}, {row['risk_bucket']})"
        for row in top_scenarios
    )
    return {
        "risk_multiplier": risk_multiplier,
        "risk_off_probability": risk_off_probability,
        "transition_probability": transition_probability,
        "fragile_upside_probability": fragile_probability,
        "top_scenarios": top_scenarios,
        "evidence": evidence,
    }


def _event_context(events: tuple[MarketEvent, ...]) -> dict[str, object]:
    current_events = [event for event in events if event.current]
    escalation_events = [event for event in current_events if event.direction == "escalation"]
    uncertain_events = [event for event in current_events if event.direction == "uncertain"]
    leading_events = [event for event in escalation_events if event.phase == "leading_warning"]
    event_pressure = min(
        0.25,
        0.07 * len(leading_events)
        + 0.04 * (len(escalation_events) - len(leading_events))
        + 0.02 * len(uncertain_events),
    )
    risk_multiplier = float(np.clip(1.0 - event_pressure, 0.75, 1.0))
    material_events = tuple(
        sorted(
            current_events,
            key=lambda event: (
                event.direction != "escalation",
                event.phase != "leading_warning",
                event.date,
            ),
        )[:5]
    )
    evidence = "; ".join(
        f"{event.name} ({event.category}, {event.direction}, {event.phase})"
        for event in material_events
    )
    return {
        "risk_multiplier": risk_multiplier,
        "event_pressure": event_pressure,
        "current_event_count": len(current_events),
        "escalation_event_count": len(escalation_events),
        "leading_event_count": len(leading_events),
        "material_events": material_events,
        "evidence": evidence or "No current event pressure.",
    }


def _macro_context(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {
            "risk_multiplier": 1.0,
            "macro_pressure": 0.0,
            "paper_candidate_count": 0,
            "evidence": "No signal-inclusion results available.",
        }
    candidates = summary[summary["decision"] == "paper_candidate"].copy()
    if candidates.empty:
        active_rejected = summary[
            (summary["latest_pressure_state"] == "risk_pressure")
            & (summary["decision"].isin(["reject_for_now", "watch_only"]))
        ]
        return {
            "risk_multiplier": 1.0,
            "macro_pressure": 0.0,
            "paper_candidate_count": 0,
            "evidence": (
                "No macro category has allocation authority; "
                f"{len(active_rejected)} rejected/watch categories show current pressure only as context."
            ),
        }
    active_candidates = candidates[candidates["latest_pressure_state"] == "risk_pressure"]
    macro_pressure = min(0.15, 0.05 * len(active_candidates))
    evidence = "; ".join(active_candidates["signal_group"].astype(str).head(3))
    return {
        "risk_multiplier": float(np.clip(1.0 - macro_pressure, 0.85, 1.0)),
        "macro_pressure": macro_pressure,
        "paper_candidate_count": int(len(candidates)),
        "evidence": evidence or "Paper-candidate macro signals are not currently pressuring risk.",
    }


def _scenario_adjusted_weights(
    base_weights: pd.Series,
    *,
    risk_multiplier: float,
    defensive_ticker: str,
) -> pd.Series:
    adjusted = base_weights.copy().astype(float)
    if defensive_ticker not in adjusted.index:
        adjusted.loc[defensive_ticker] = 0.0
    risk_assets = [ticker for ticker in adjusted.index if ticker != defensive_ticker]
    original_risk_weight = float(adjusted.loc[risk_assets].clip(lower=0.0).sum())
    adjusted.loc[risk_assets] = adjusted.loc[risk_assets] * risk_multiplier
    new_risk_weight = float(adjusted.loc[risk_assets].clip(lower=0.0).sum())
    adjusted.loc[defensive_ticker] = adjusted.loc[defensive_ticker] + max(
        0.0,
        original_risk_weight - new_risk_weight,
    )
    total = float(adjusted.clip(lower=0.0).sum())
    if total > 1.0:
        adjusted = adjusted / total
    return adjusted.sort_values(ascending=False)


def _position_plan(
    base_weights: pd.Series,
    adjusted_weights: pd.Series,
    min_trade_weight: float,
) -> pd.DataFrame:
    tickers = sorted(set(base_weights.index) | set(adjusted_weights.index))
    rows = []
    for ticker in tickers:
        current = float(base_weights.get(ticker, 0.0))
        adjusted = float(adjusted_weights.get(ticker, 0.0))
        delta = adjusted - current
        if abs(delta) < min_trade_weight:
            action = "HOLD"
        elif delta > 0:
            action = "ADD"
        else:
            action = "REDUCE"
        rows.append(
            {
                "ticker": ticker,
                "current_weight": current,
                "scenario_adjusted_weight": adjusted,
                "delta_weight": delta,
                "action": action,
            }
        )
    frame = pd.DataFrame(rows)
    material = (
        frame[["current_weight", "scenario_adjusted_weight", "delta_weight"]].abs().max(axis=1)
        >= 0.005
    )
    return frame[material].sort_values("delta_weight")


def _build_portfolio_risk_if_available(
    *,
    prices: pd.DataFrame | None,
    base_weights: pd.Series,
    adjusted_weights: pd.Series,
    current_state: CurrentStateRun,
    defensive_ticker: str,
) -> PortfolioRiskRun | None:
    if prices is None or prices.empty:
        return None
    return build_portfolio_risk(
        prices,
        adjusted_weights,
        current_state.scenario_lattice,
        current_weights=base_weights,
        config=PortfolioRiskConfig(defensive_ticker=defensive_ticker),
    )


def _portfolio_risk_context(portfolio_risk: PortfolioRiskRun | None) -> dict[str, object]:
    if portfolio_risk is None or portfolio_risk.summary.empty:
        return {
            "portfolio_risk_multiplier": 1.0,
            "portfolio_risk_level": "not_available",
            "applied_constraints": "none",
            "post_expected_shortfall_95": np.nan,
            "post_max_stress_loss": np.nan,
            "post_equity_beta": np.nan,
            "post_ai_beta": np.nan,
            "evidence": "Portfolio risk engine was not available for this decision.",
        }
    row = portfolio_risk.summary.iloc[0]
    constraints = str(row.get("applied_constraints", "none"))
    risk_level = str(row.get("portfolio_risk_level", "unknown"))
    return {
        "portfolio_risk_multiplier": _as_float(row.get("portfolio_risk_multiplier", 1.0)),
        "portfolio_risk_level": risk_level,
        "applied_constraints": constraints,
        "post_expected_shortfall_95": _as_float(row.get("post_expected_shortfall_95")),
        "post_max_stress_loss": _as_float(row.get("post_max_stress_loss")),
        "post_equity_beta": _as_float(row.get("post_equity_beta")),
        "post_ai_beta": _as_float(row.get("post_ai_beta")),
        "evidence": (
            f"{risk_level}; applied constraints: {constraints}; "
            f"ES95 {float(row.get('post_expected_shortfall_95', np.nan)):.2%}, "
            f"max stress loss {float(row.get('post_max_stress_loss', np.nan)):.2%}, "
            f"equity beta {float(row.get('post_equity_beta', np.nan)):.2f}, "
            f"AI beta {float(row.get('post_ai_beta', np.nan)):.2f}."
        ),
    }


def _add_portfolio_risk_sizing_columns(
    position_plan: pd.DataFrame,
    portfolio_risk: PortfolioRiskRun | None,
) -> pd.DataFrame:
    if portfolio_risk is None or portfolio_risk.sizing_adjustments.empty:
        return position_plan
    sizing_columns = [
        "ticker",
        "group",
        "pre_risk_target_weight",
        "risk_adjusted_weight",
        "risk_engine_delta",
        "risk_adjustment_reason",
    ]
    available_columns = [
        column for column in sizing_columns if column in portfolio_risk.sizing_adjustments
    ]
    return position_plan.merge(
        portfolio_risk.sizing_adjustments[available_columns],
        on="ticker",
        how="left",
    )


def _recommended_action(position_plan: pd.DataFrame, risk_status: str) -> str:
    material_reductions = position_plan[position_plan["delta_weight"] <= -0.05]
    material_adds = position_plan[position_plan["delta_weight"] >= 0.05]
    if risk_status in {"orange", "red"} and not material_reductions.empty:
        return "REDUCE_RISK"
    if not material_reductions.empty:
        return "REVIEW_REDUCE_RISK"
    if not material_adds.empty:
        return "REVIEW_ADD_RISK"
    return "HOLD"


def _human_explanation(
    *,
    action: str,
    base_weights: pd.Series,
    adjusted_weights: pd.Series,
    current_state: CurrentStateRun,
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
    portfolio_risk_context: dict[str, object],
) -> str:
    base_position = _format_weight_vector(base_weights)
    adjusted_position = _format_weight_vector(adjusted_weights)
    top_scenarios = str(scenario_context["evidence"])
    event_evidence = str(event_context["evidence"])
    macro_evidence = str(macro_context["evidence"])
    portfolio_risk_evidence = str(portfolio_risk_context["evidence"])
    if action in {"REDUCE_RISK", "REVIEW_REDUCE_RISK"}:
        verb = "review reducing risk toward"
    elif action == "REVIEW_ADD_RISK":
        verb = "review adding risk toward"
    else:
        verb = "hold the current posture near"
    return (
        f"Because risk status is {current_state.risk_status.upper()} ({current_state.risk_score:.2f}), "
        f"the one-month scenario mix is {top_scenarios}, and current event pressure is {event_evidence}, "
        f"{verb} {adjusted_position}. Base systematic position is {base_position}. "
        f"Portfolio risk engine says: {portfolio_risk_evidence} "
        f"Macro inclusion tests say: {macro_evidence}"
    )


def _decision_authority(
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
    portfolio_risk_context: dict[str, object],
) -> str:
    if str(portfolio_risk_context["portfolio_risk_level"]) in {
        "constraint_breach",
        "risk_reduced",
        "watch_correlation_shift",
    }:
        return "scenario_event_risk_engine_review"
    if _as_float(macro_context["paper_candidate_count"]) > 0:
        return "scenario_plus_validated_macro_review"
    if (
        _as_float(event_context["event_pressure"]) > 0
        or _as_float(scenario_context["risk_off_probability"]) > 0.20
    ):
        return "scenario_event_review"
    return "systematic_hold"


def _evidence_table(
    *,
    current_state: CurrentStateRun,
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
    portfolio_risk_context: dict[str, object],
    news_monitor: NewsMonitorRun,
) -> pd.DataFrame:
    rows = [
        {
            "evidence_type": "risk_state",
            "signal": current_state.risk_status.upper(),
            "impact": "sets base risk budget",
            "detail": current_state.risk_summary,
        },
        {
            "evidence_type": "scenario_mix",
            "signal": "1m scenario probabilities",
            "impact": "sizes scenario-adjusted risk budget",
            "detail": str(scenario_context["evidence"]),
        },
        {
            "evidence_type": "event_pressure",
            "signal": f"{event_context['current_event_count']} current events",
            "impact": "can cap risk budget before price confirmation",
            "detail": str(event_context["evidence"]),
        },
        {
            "evidence_type": "macro_inclusion",
            "signal": f"{macro_context['paper_candidate_count']} paper candidates",
            "impact": "only validated paper candidates can affect sizing",
            "detail": str(macro_context["evidence"]),
        },
        {
            "evidence_type": "portfolio_risk_engine",
            "signal": str(portfolio_risk_context["portfolio_risk_level"]),
            "impact": "checks factor, beta, tail, stress, correlation, and constraint risk",
            "detail": str(portfolio_risk_context["evidence"]),
        },
    ]
    if not news_monitor.triage.empty:
        activation_counts = news_monitor.triage["activation_status"].value_counts().to_dict()
        rows.append(
            {
                "evidence_type": "news_intake",
                "signal": "latest triage",
                "impact": "feeds current event-risk scenarios",
                "detail": ", ".join(f"{key}: {value}" for key, value in activation_counts.items()),
            }
        )
    return pd.DataFrame(rows)


def _scenario_links(scenario_lattice: pd.DataFrame) -> pd.DataFrame:
    if scenario_lattice.empty:
        return pd.DataFrame()
    one_month = scenario_lattice[scenario_lattice["horizon"] == "1m"].copy()
    if one_month.empty:
        one_month = scenario_lattice.copy()
    columns = [
        "rank",
        "scenario",
        "probability",
        "risk_bucket",
        "expected_bot_posture",
        "preferred_exposure",
        "avoid_exposure",
        "confirmation",
        "off_ramp",
    ]
    return one_month.sort_values("rank")[columns].head(5).reset_index(drop=True)


def _risk_status_multiplier(risk_status: str) -> float:
    return {
        "green": 1.0,
        "yellow": 0.90,
        "orange": 0.65,
        "red": 0.40,
    }.get(risk_status, 0.85)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return np.nan


def _format_weight_vector(weights: pd.Series) -> str:
    positive = weights[weights > 0.005].sort_values(ascending=False)
    if positive.empty:
        return "BIL/cash 100%"
    return ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in positive.items())
