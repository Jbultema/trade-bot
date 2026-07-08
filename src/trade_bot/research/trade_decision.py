from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.DEFAULTS import (
    DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS,
    DEFAULT_EVENT_CONFIRMATION_THEMES,
    DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD,
)
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
    base_weights = _weights_with_defensive_residual(
        primary_result.weights.iloc[-1].astype(float),
        defensive_ticker=defensive_ticker,
    )
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
        pre_sanity_final_weights = portfolio_risk.risk_adjusted_weights
    else:
        pre_sanity_final_weights = adjusted_weights
    decision_sanity_context = _decision_sanity_context(
        current_state=current_state,
        scenario_context=scenario_context,
        event_context=event_context,
        macro_context=macro_context,
    )
    final_weights, decision_sanity_context = _apply_decision_sanity_cap(
        base_weights=base_weights,
        candidate_weights=pre_sanity_final_weights,
        defensive_ticker=defensive_ticker,
        sanity_context=decision_sanity_context,
    )
    portfolio_risk_multiplier = _as_float(portfolio_risk_context["portfolio_risk_multiplier"])
    pre_sanity_risk_budget_multiplier = float(
        np.clip(risk_multiplier * portfolio_risk_multiplier, 0, 1)
    )
    total_risk_budget_multiplier = _risk_budget_multiplier_from_weights(
        base_weights,
        final_weights,
        defensive_ticker=defensive_ticker,
    )
    position_plan = _position_plan(base_weights, final_weights, min_trade_weight)
    position_plan = _add_portfolio_risk_sizing_columns(position_plan, portfolio_risk)
    position_plan = _add_decision_sanity_sizing_columns(position_plan)
    posture_context = _posture_calibration_context(
        current_state=current_state,
        scenario_context=scenario_context,
        event_context=event_context,
        macro_context=macro_context,
        position_plan=position_plan,
        defensive_ticker=defensive_ticker,
        risk_budget_multiplier=total_risk_budget_multiplier,
    )
    action = _recommended_action(
        position_plan,
        current_state.risk_status,
        defensive_ticker=defensive_ticker,
    )
    explanation = _human_explanation(
        action=action,
        base_weights=base_weights,
        adjusted_weights=final_weights,
        current_state=current_state,
        scenario_context=scenario_context,
        event_context=event_context,
        macro_context=macro_context,
        portfolio_risk_context=portfolio_risk_context,
        posture_context=posture_context,
        decision_sanity_context=decision_sanity_context,
    )
    authority = _decision_authority(
        scenario_context,
        event_context,
        macro_context,
        portfolio_risk_context,
        decision_sanity_context,
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": primary_result.name,
                "recommended_action": action,
                "decision_authority": authority,
                "base_position": _format_weight_vector(base_weights),
                "pre_risk_target_position": _format_weight_vector(adjusted_weights),
                "pre_sanity_target_position": _format_weight_vector(pre_sanity_final_weights),
                "scenario_adjusted_position": _format_weight_vector(final_weights),
                "risk_budget_multiplier": total_risk_budget_multiplier,
                "pre_sanity_risk_budget_multiplier": pre_sanity_risk_budget_multiplier,
                "scenario_event_macro_multiplier": risk_multiplier,
                "portfolio_risk_multiplier": portfolio_risk_multiplier,
                "decision_sanity_status": decision_sanity_context["status"],
                "decision_sanity_signal": decision_sanity_context["signal"],
                "decision_sanity_note": decision_sanity_context["detail"],
                "decision_sanity_cap_applied": decision_sanity_context["cap_applied"],
                "market_confirmation_break_count": decision_sanity_context[
                    "confirmation_break_count"
                ],
                "market_confirmation_breaks": decision_sanity_context["confirmation_breaks_text"],
                "event_only_max_defensive_add": decision_sanity_context[
                    "event_only_max_defensive_add"
                ],
                "risk_status": current_state.risk_status,
                "risk_score": current_state.risk_score,
                "one_month_risk_off_probability": scenario_context["risk_off_probability"],
                "one_month_transition_probability": scenario_context["transition_probability"],
                "one_month_fragile_upside_probability": scenario_context[
                    "fragile_upside_probability"
                ],
                "one_month_risk_on_probability": scenario_context["risk_on_probability"],
                "constructive_scenario_probability": scenario_context["constructive_probability"],
                "event_pressure": event_context["event_pressure"],
                "macro_pressure": macro_context["macro_pressure"],
                "posture_calibration_status": posture_context["status"],
                "posture_calibration_signal": posture_context["signal"],
                "posture_calibration_note": posture_context["detail"],
                "current_risk_asset_weight": posture_context["current_risk_asset_weight"],
                "target_risk_asset_weight": posture_context["target_risk_asset_weight"],
                "target_defensive_weight": posture_context["target_defensive_weight"],
                "opportunity_pressure": posture_context["opportunity_pressure"],
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
        posture_context=posture_context,
        decision_sanity_context=decision_sanity_context,
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


def _weights_with_defensive_residual(
    weights: pd.Series,
    *,
    defensive_ticker: str,
) -> pd.Series:
    clean = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if defensive_ticker not in clean.index:
        clean.loc[defensive_ticker] = 0.0
    total = float(clean.sum())
    if total < 1.0:
        clean.loc[defensive_ticker] = clean.loc[defensive_ticker] + (1.0 - total)
    elif total > 1.0:
        clean = clean / total
    return clean.sort_values(ascending=False)


def _scenario_context(scenario_lattice: pd.DataFrame) -> dict[str, object]:
    if scenario_lattice.empty:
        return {
            "risk_multiplier": 1.0,
            "risk_off_probability": 0.0,
            "transition_probability": 0.0,
            "fragile_upside_probability": 0.0,
            "risk_on_probability": 0.0,
            "constructive_probability": 0.0,
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
    risk_on_probability = float((one_month.loc[risk_bucket == "risk_on", "probability"]).sum())
    constructive_probability = float(
        np.clip(risk_on_probability + 0.50 * fragile_probability, 0.0, 1.0)
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
        "risk_on_probability": risk_on_probability,
        "constructive_probability": constructive_probability,
        "top_scenarios": top_scenarios,
        "evidence": evidence,
    }


def _event_context(events: tuple[MarketEvent, ...]) -> dict[str, object]:
    current_events = [event for event in events if event.current]
    sizing_events = [event for event in current_events if event.sizing_authority]
    escalation_events = [event for event in sizing_events if event.direction == "escalation"]
    uncertain_events = [event for event in sizing_events if event.direction == "uncertain"]
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
                not event.sizing_authority,
                event.direction != "escalation",
                event.phase != "leading_warning",
                event.date,
            ),
        )[:5]
    )
    evidence = "; ".join(
        (
            f"{event.name} ({event.category}, {event.direction}, {event.phase}, "
            f"{'sizing' if event.sizing_authority else 'watch-only'})"
        )
        for event in material_events
    )
    return {
        "risk_multiplier": risk_multiplier,
        "event_pressure": event_pressure,
        "current_event_count": len(current_events),
        "sizing_event_count": len(sizing_events),
        "watch_only_event_count": len(current_events) - len(sizing_events),
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


def _add_decision_sanity_sizing_columns(position_plan: pd.DataFrame) -> pd.DataFrame:
    if (
        position_plan.empty
        or "scenario_adjusted_weight" not in position_plan
        or "risk_adjusted_weight" not in position_plan
    ):
        return position_plan
    frame = position_plan.copy()
    frame["decision_sanity_delta"] = (
        pd.to_numeric(frame["scenario_adjusted_weight"], errors="coerce")
        - pd.to_numeric(frame["risk_adjusted_weight"], errors="coerce")
    )
    return frame


def _decision_sanity_context(
    *,
    current_state: CurrentStateRun,
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
) -> dict[str, object]:
    confirmation_breaks = _market_confirmation_breaks(current_state)
    break_count = len(confirmation_breaks)
    risk_status = str(current_state.risk_status).lower()
    risk_off_probability = _as_float(scenario_context["risk_off_probability"])
    event_pressure = _as_float(event_context["event_pressure"])
    left_tail_confirmed = risk_status in {"orange", "red"} or risk_off_probability >= 0.35
    market_confirmed = break_count >= DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS
    cap_eligible = (
        event_pressure > 0
        and not left_tail_confirmed
        and not market_confirmed
    )

    if market_confirmed:
        status = "market_confirmation_allows_derisk"
        signal = "Market confirmation broke"
        detail = (
            f"Bigger de-risking is allowed because {break_count}/"
            f"{DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS} confirmation gates are broken: "
            f"{_confirmation_breaks_text(confirmation_breaks)}."
        )
    elif left_tail_confirmed:
        status = "left_tail_allows_derisk"
        signal = "Left-tail pressure confirmed"
        detail = (
            "Bigger de-risking is allowed because risk status or one-month scenario risk-off "
            "probability is already in the severe zone."
        )
    elif cap_eligible:
        status = "event_only_cap_eligible"
        signal = "Event-only de-risk cap active"
        detail = (
            "News/events are pressuring risk, but fewer than two of credit, volatility, "
            "breadth, or trend have broken. Large cash moves should wait for market "
            "confirmation."
        )
    elif event_pressure > 0:
        status = "event_watch"
        signal = "Event pressure watched"
        detail = (
            "Event pressure is active, but the cap is not governing because left-tail scenario "
            "or market-confirmation evidence has separate sizing authority."
        )
    else:
        status = "not_needed"
        signal = "No sanity cap needed"
        detail = "No event/news-only de-risk cap is needed for this decision."

    return {
        "status": status,
        "signal": signal,
        "detail": detail,
        "cap_eligible": cap_eligible,
        "cap_applied": False,
        "confirmation_break_count": break_count,
        "confirmation_breaks": confirmation_breaks,
        "confirmation_breaks_text": _confirmation_breaks_text(confirmation_breaks),
        "event_only_max_defensive_add": DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD,
        "left_tail_confirmed": left_tail_confirmed,
        "market_confirmed": market_confirmed,
    }


def _apply_decision_sanity_cap(
    *,
    base_weights: pd.Series,
    candidate_weights: pd.Series,
    defensive_ticker: str,
    sanity_context: dict[str, object],
) -> tuple[pd.Series, dict[str, object]]:
    context = dict(sanity_context)
    if not bool(context.get("cap_eligible", False)):
        return candidate_weights.sort_values(ascending=False), context

    base, candidate = _aligned_weights(
        base_weights,
        candidate_weights,
        defensive_ticker=defensive_ticker,
    )
    base_defensive = _weight_for_ticker(base, defensive_ticker)
    candidate_defensive = _weight_for_ticker(candidate, defensive_ticker)
    max_defensive = float(
        np.clip(base_defensive + DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD, 0.0, 1.0)
    )
    context["max_defensive_weight"] = max_defensive
    context["pre_sanity_defensive_weight"] = candidate_defensive
    if candidate_defensive <= max_defensive + 1e-9:
        context.update(
            {
                "status": "event_only_cap_not_needed",
                "signal": "Event-only cap not binding",
                "cap_applied": False,
                "detail": (
                    "News/events are pressuring risk without enough market confirmation, but "
                    f"the requested defensive weight of {candidate_defensive:.0%} is already "
                    f"within the event-only cap of {max_defensive:.0%}."
                ),
            }
        )
        return candidate.sort_values(ascending=False), context

    capped = candidate.copy()
    freed_weight = candidate_defensive - max_defensive
    capped.loc[defensive_ticker] = max_defensive
    risk_assets = [ticker for ticker in capped.index if ticker.upper() != defensive_ticker.upper()]
    candidate_risk = capped.loc[risk_assets].clip(lower=0.0)
    if float(candidate_risk.sum()) > 0:
        allocation_basis = candidate_risk / float(candidate_risk.sum())
    else:
        base_risk = base.loc[risk_assets].clip(lower=0.0)
        if float(base_risk.sum()) > 0:
            allocation_basis = base_risk / float(base_risk.sum())
        else:
            allocation_basis = pd.Series(1.0 / max(len(risk_assets), 1), index=risk_assets)
    capped.loc[risk_assets] = capped.loc[risk_assets] + freed_weight * allocation_basis
    capped = capped.clip(lower=0.0)
    total = float(capped.sum())
    if total > 1.0:
        capped = capped / total
    context.update(
        {
            "status": "event_only_cap_applied",
            "signal": "Event-only cap applied",
            "cap_applied": True,
            "detail": (
                f"Decision sanity capped the defensive target at {max_defensive:.0%} because "
                "news/events are active but fewer than two of credit, volatility, breadth, "
                "or trend have broken. The uncapped defensive target was "
                f"{candidate_defensive:.0%}; bigger cash moves require market confirmation."
            ),
        }
    )
    return capped.sort_values(ascending=False), context


def _aligned_weights(
    base_weights: pd.Series,
    candidate_weights: pd.Series,
    *,
    defensive_ticker: str,
) -> tuple[pd.Series, pd.Series]:
    tickers = sorted(set(base_weights.index) | set(candidate_weights.index) | {defensive_ticker})
    base = base_weights.reindex(tickers).fillna(0.0).astype(float).clip(lower=0.0)
    candidate = candidate_weights.reindex(tickers).fillna(0.0).astype(float).clip(lower=0.0)
    return base, candidate


def _risk_budget_multiplier_from_weights(
    base_weights: pd.Series,
    final_weights: pd.Series,
    *,
    defensive_ticker: str,
) -> float:
    base_risk = _risk_asset_weight_from_weights(base_weights, defensive_ticker=defensive_ticker)
    final_risk = _risk_asset_weight_from_weights(final_weights, defensive_ticker=defensive_ticker)
    if base_risk <= 1e-9:
        return 1.0 if final_risk <= 1e-9 else 0.0
    return float(np.clip(final_risk / base_risk, 0.0, 1.0))


def _risk_asset_weight_from_weights(weights: pd.Series, *, defensive_ticker: str) -> float:
    if weights.empty:
        return 0.0
    clean = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk_assets = [ticker for ticker in clean.index if str(ticker).upper() != defensive_ticker.upper()]
    return float(clean.loc[risk_assets].clip(lower=0.0).sum())


def _weight_for_ticker(weights: pd.Series, ticker: str) -> float:
    matches = [index for index in weights.index if str(index).upper() == ticker.upper()]
    if not matches:
        return 0.0
    return float(weights.loc[matches[0]])


def _market_confirmation_breaks(current_state: CurrentStateRun) -> tuple[str, ...]:
    breaks: set[str] = set()
    frame = current_state.confirmation_matrix
    if not frame.empty:
        for _, row in frame.iterrows():
            if not _confirmation_row_is_negative(row):
                continue
            key = " ".join(
                str(row.get(column, ""))
                for column in ("theme", "name", "signal", "driver", "detail")
            ).lower()
            breaks.update(_confirmation_themes_from_key(key))
    health = current_state.market_health
    if not health.empty:
        for _, row in health.iterrows():
            if not _confirmation_row_is_negative(row):
                continue
            key = " ".join(str(value) for value in row.to_dict().values()).lower()
            breaks.update(_confirmation_themes_from_key(key))
    ordered = [theme for theme in DEFAULT_EVENT_CONFIRMATION_THEMES if theme in breaks]
    return tuple(ordered)


def _confirmation_row_is_negative(row: pd.Series) -> bool:
    state_text = " ".join(
        str(row.get(column, "")) for column in ("status", "state", "risk_state", "signal_state")
    ).lower()
    if any(token in state_text for token in ("bearish", "risk_off", "risk-pressure", "risk_pressure")):
        return True
    score = _as_float(row.get("score", np.nan))
    return bool(np.isfinite(score) and score < 0)


def _confirmation_themes_from_key(key: str) -> set[str]:
    themes: set[str] = set()
    if any(token in key for token in ("credit", "hyg", "lqd", "spread")):
        themes.add("credit")
    if any(token in key for token in ("volatility", "vix", "vol ")):
        themes.add("volatility")
    if any(token in key for token in ("breadth", "equal", "rsp", "cap weight")):
        themes.add("breadth")
    if any(
        token in key
        for token in (
            "trend",
            "momentum",
            "broad_market",
            "ai_beta",
            "spy",
            "qqq",
            "smh",
        )
    ):
        themes.add("trend")
    return themes


def _confirmation_breaks_text(confirmation_breaks: tuple[str, ...]) -> str:
    if not confirmation_breaks:
        return "none"
    return ", ".join(confirmation_breaks)


def _recommended_action(
    position_plan: pd.DataFrame,
    risk_status: str,
    *,
    defensive_ticker: str,
) -> str:
    current_risk = _risk_asset_weight(
        position_plan,
        weight_column="current_weight",
        defensive_ticker=defensive_ticker,
    )
    target_risk = _risk_asset_weight(
        position_plan,
        weight_column="scenario_adjusted_weight",
        defensive_ticker=defensive_ticker,
    )
    risk_delta = target_risk - current_risk
    if risk_delta <= -0.05 and risk_status in {"orange", "red"}:
        return "REDUCE_RISK"
    if risk_delta <= -0.05:
        return "REVIEW_REDUCE_RISK"
    if risk_delta >= 0.05:
        return "REVIEW_ADD_RISK"

    risk_assets = position_plan[
        position_plan["ticker"].astype(str).str.upper() != defensive_ticker.upper()
    ]
    material_reductions = risk_assets[risk_assets["delta_weight"] <= -0.05]
    material_adds = risk_assets[risk_assets["delta_weight"] >= 0.05]
    if not material_reductions.empty:
        return "REVIEW_REDUCE_RISK"
    if not material_adds.empty:
        return "REVIEW_ADD_RISK"
    return "HOLD"


def _posture_calibration_context(
    *,
    current_state: CurrentStateRun,
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
    position_plan: pd.DataFrame,
    defensive_ticker: str,
    risk_budget_multiplier: float,
) -> dict[str, object]:
    current_risk_asset_weight = _risk_asset_weight(
        position_plan,
        weight_column="current_weight",
        defensive_ticker=defensive_ticker,
    )
    target_risk_asset_weight = _risk_asset_weight(
        position_plan,
        weight_column="scenario_adjusted_weight",
        defensive_ticker=defensive_ticker,
    )
    target_defensive_weight = _ticker_weight(
        position_plan,
        weight_column="scenario_adjusted_weight",
        ticker=defensive_ticker,
    )
    risk_reduction = max(0.0, current_risk_asset_weight - target_risk_asset_weight)
    risk_off_probability = _as_float(scenario_context["risk_off_probability"])
    transition_probability = _as_float(scenario_context["transition_probability"])
    fragile_probability = _as_float(scenario_context["fragile_upside_probability"])
    risk_on_probability = _as_float(scenario_context["risk_on_probability"])
    constructive_probability = _as_float(scenario_context["constructive_probability"])
    event_pressure = _as_float(event_context["event_pressure"])
    macro_pressure = _as_float(macro_context["macro_pressure"])
    opportunity_pressure = float(
        np.clip(
            risk_on_probability
            + fragile_probability
            + 0.50 * transition_probability
            - risk_off_probability
            - event_pressure
            - macro_pressure,
            0.0,
            1.0,
        )
    )

    status = "balanced"
    signal = "No bearish-bias warning"
    detail = (
        "Risk sizing is not materially defensive relative to current scenario, event, "
        "macro, and price-risk evidence."
    )
    if current_state.risk_status in {"red", "orange"} or risk_off_probability >= 0.35:
        status = "defense_justified"
        signal = "Defensive posture supported"
        detail = (
            "Risk-off, orange/red market state, or left-tail scenario pressure is large enough "
            "that defensive sizing is not treated as psychological bearishness."
        )
    elif event_pressure >= 0.12:
        status = "event_defense_review"
        signal = "Event-driven defense"
        detail = (
            "Current event pressure is the main reason for smaller sizing. Re-risking should wait "
            "for tradable confirmation rather than narrative comfort."
        )
    elif (
        risk_budget_multiplier <= 0.75
        and opportunity_pressure >= 0.45
        and constructive_probability >= risk_off_probability + 0.15
    ):
        status = "under_risk_review"
        signal = "Possible under-risking"
        detail = (
            "Constructive or fragile-upside scenario weight is high relative to risk-off pressure. "
            "Do not reduce risk further without checking whether price, breadth, and credit are "
            "actually confirming deterioration."
        )
    elif (
        risk_reduction >= 0.10
        and opportunity_pressure >= 0.35
        and current_state.risk_status in {"green", "yellow"}
    ):
        status = "opportunity_cost_watch"
        signal = "Opportunity-cost watch"
        detail = (
            "The target posture cuts risk meaningfully while medium-term upside evidence remains "
            "plausible. Treat the trade as a review item, not a reflexive de-risk."
        )
    elif constructive_probability >= 0.45 and target_risk_asset_weight >= (
        current_risk_asset_weight - 0.05
    ):
        status = "upside_participation_ok"
        signal = "Upside participation intact"
        detail = (
            "Constructive scenario evidence is being allowed to participate; the system is not "
            "currently over-suppressing risk."
        )

    return {
        "status": status,
        "signal": signal,
        "detail": detail,
        "current_risk_asset_weight": current_risk_asset_weight,
        "target_risk_asset_weight": target_risk_asset_weight,
        "target_defensive_weight": target_defensive_weight,
        "risk_reduction": risk_reduction,
        "opportunity_pressure": opportunity_pressure,
    }


def _risk_asset_weight(
    position_plan: pd.DataFrame,
    *,
    weight_column: str,
    defensive_ticker: str,
) -> float:
    if position_plan.empty or weight_column not in position_plan or "ticker" not in position_plan:
        return 0.0
    risk_rows = position_plan[
        position_plan["ticker"].astype(str).str.upper() != defensive_ticker.upper()
    ]
    return float(risk_rows[weight_column].clip(lower=0.0).sum())


def _ticker_weight(position_plan: pd.DataFrame, *, weight_column: str, ticker: str) -> float:
    if position_plan.empty or weight_column not in position_plan or "ticker" not in position_plan:
        return 0.0
    rows = position_plan[position_plan["ticker"].astype(str).str.upper() == ticker.upper()]
    if rows.empty:
        return 0.0
    return float(rows[weight_column].clip(lower=0.0).sum())


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
    posture_context: dict[str, object],
    decision_sanity_context: dict[str, object],
) -> str:
    base_position = _format_weight_vector(base_weights)
    adjusted_position = _format_weight_vector(adjusted_weights)
    top_scenarios = str(scenario_context["evidence"])
    event_evidence = str(event_context["evidence"])
    macro_evidence = str(macro_context["evidence"])
    portfolio_risk_evidence = str(portfolio_risk_context["evidence"])
    posture_evidence = str(posture_context["detail"])
    decision_sanity_evidence = str(decision_sanity_context["detail"])
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
        f"Macro inclusion tests say: {macro_evidence} "
        f"Decision sanity says: {decision_sanity_evidence} "
        f"Posture calibration says: {posture_evidence}"
    )


def _decision_authority(
    scenario_context: dict[str, object],
    event_context: dict[str, object],
    macro_context: dict[str, object],
    portfolio_risk_context: dict[str, object],
    decision_sanity_context: dict[str, object],
) -> str:
    if bool(decision_sanity_context.get("cap_applied", False)):
        return "decision_sanity_capped_review"
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
    posture_context: dict[str, object],
    decision_sanity_context: dict[str, object],
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
            "evidence_type": "decision_sanity",
            "signal": str(decision_sanity_context["signal"]),
            "impact": "caps event/news-only de-risking unless market confirmation breaks",
            "detail": str(decision_sanity_context["detail"]),
        },
        {
            "evidence_type": "portfolio_risk_engine",
            "signal": str(portfolio_risk_context["portfolio_risk_level"]),
            "impact": "checks factor, beta, tail, stress, correlation, and constraint risk",
            "detail": str(portfolio_risk_context["evidence"]),
        },
        {
            "evidence_type": "posture_calibration",
            "signal": str(posture_context["signal"]),
            "impact": "checks whether defensive sizing may be over-bearish",
            "detail": str(posture_context["detail"]),
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
