from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _format_decimal,
    _format_percent,
    _optional_float,
)
from trade_bot.DEFAULTS import DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT
from trade_bot.reporting.report import window_performance_frame
from trade_bot.research.action_headline import ActionHeadline
from trade_bot.research.baselines import BaselineRun
from trade_bot.trading.book_alignment import BookAlignmentRun


def _render_operating_brief(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    book_alignment: BookAlignmentRun | None = None,
) -> None:
    st.subheader("Operating Brief")
    st.caption(
        "Execution checklist for today's recommendation: sizing translation, scenario constraints, and bias checks."
    )
    position_plan = _execution_position_plan(baseline_run, book_alignment)
    cards = _operating_brief_cards(
        baseline_run=baseline_run,
        headline=headline,
        position_plan=position_plan,
    )
    st.markdown(
        '<div class="operating-grid">'
        + "".join(_operating_card_html(card) for card in cards)
        + "</div>",
        unsafe_allow_html=True,
    )

    sizing_steps = _recommended_sizing_steps(position_plan)
    scenario_bridge = _scenario_bridge_table(baseline_run)
    evidence = _operating_evidence_table(baseline_run, headline)

    with st.expander("Execution details", expanded=False):
        detail_tab, sizing_tab, scenario_tab, evidence_tab = st.tabs(
            ["Read This First", "Sizing Steps", "Scenario Bridge", "Why"]
        )
        with detail_tab:
            _render_metric_dataframe(
                _operating_instruction_table(baseline_run, headline, book_alignment),
                hide_index=True,
            )
        with sizing_tab:
            st.caption(
                "Sizing is shown in portfolio-weight percentage points. Use the Forward Test section to convert these into ticket dollar/share ranges."
            )
            _render_metric_dataframe(_display_metrics(sizing_steps), hide_index=True)
        with scenario_tab:
            st.caption(
                "Future scenarios do not directly predict one exact future. They adjust the allowed risk budget, defensive minimums, beta limits, and stress-loss guardrails."
            )
            _render_metric_dataframe(_display_metrics(scenario_bridge), hide_index=True)
        with evidence_tab:
            _render_metric_dataframe(evidence, hide_index=True)


def _operating_brief_cards(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    position_plan: pd.DataFrame,
) -> list[dict[str, str]]:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    action = str(trade_summary.get("recommended_action", headline.label))
    scenario_effect = _scenario_effect_sentence(trade_summary, risk_summary)
    primary_sizing = _primary_sizing_sentence(position_plan)
    posture_check = _posture_calibration_sentence(trade_summary)
    sanity_check = _decision_sanity_sentence(trade_summary)
    return [
        {
            "tone": "warning" if "REDUCE" in action else "success",
            "label": "Sizing Translation",
            "answer": primary_sizing["answer"],
            "detail": primary_sizing["detail"],
        },
        {
            "tone": "warning",
            "label": "Risk Constraints",
            "answer": scenario_effect["answer"],
            "detail": scenario_effect["detail"],
        },
        {
            "tone": sanity_check["tone"],
            "label": "Decision Sanity",
            "answer": sanity_check["answer"],
            "detail": sanity_check["detail"],
        },
        {
            "tone": posture_check["tone"],
            "label": "Bias Check",
            "answer": posture_check["answer"],
            "detail": posture_check["detail"],
        },
    ]


def _operating_card_html(card: dict[str, str]) -> str:
    return (
        f'<div class="operating-card operating-card-{html.escape(card["tone"])}">'
        f'<p class="operating-label">{html.escape(card["label"])}</p>'
        f'<p class="operating-answer">{html.escape(card["answer"])}</p>'
        f'<p class="operating-detail">{html.escape(card["detail"])}</p>'
        "</div>"
    )


def _operating_instruction_table(
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    book_alignment: BookAlignmentRun | None = None,
) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    action = str(trade_summary.get("recommended_action", headline.label))
    target_position = _target_position_text(trade_summary, book_alignment)
    risk_status = baseline_run.current_state.risk_status.upper()
    risk_budget = _format_decimal(trade_summary.get("risk_budget_multiplier", "n/a"))
    return pd.DataFrame(
        [
            {
                "question": "What is the conclusion?",
                "answer": headline.headline,
                "what_to_do": headline.next_action,
            },
            {
                "question": "What action is recommended?",
                "answer": action.replace("_", " "),
                "what_to_do": f"Review the sizing steps and move toward {target_position} only if the recommendation still fits your execution window.",
            },
            {
                "question": "How aggressive should sizing be?",
                "answer": f"Risk status is {risk_status}; combined risk budget is {risk_budget}.",
                "what_to_do": "Treat the target weights as capped by scenario/event/macro and portfolio-risk constraints, not as a full-risk forecast.",
            },
            {
                "question": "Where do I execute or paper-test it?",
                "answer": "Use Forward Test after reviewing the Command Center and Risk & Scenarios sections.",
                "what_to_do": "Lock the recommendation set, then log paper or live executions with exact time, price, quantity, and notes.",
            },
        ]
    )


def _execution_position_plan(
    baseline_run: BaselineRun,
    book_alignment: BookAlignmentRun | None,
) -> pd.DataFrame:
    if book_alignment is not None and not book_alignment.position_plan.empty:
        return book_alignment.position_plan
    return baseline_run.trade_decision.position_plan


def _target_position_text(
    trade_summary: dict[str, object],
    book_alignment: BookAlignmentRun | None,
) -> str:
    if book_alignment is not None and not book_alignment.summary.empty:
        target_position = str(book_alignment.summary.iloc[0].get("target_position", "")).strip()
        if target_position:
            return target_position
    return str(
        trade_summary.get("scenario_adjusted_position")
        or trade_summary.get("base_position")
        or "No target position available."
    )


def _recommended_sizing_steps(position_plan: pd.DataFrame) -> pd.DataFrame:
    if position_plan.empty:
        return pd.DataFrame(
            [
                {
                    "step": 1,
                    "ticker": "Portfolio",
                    "action": "NO_DATA",
                    "current_weight": 0.0,
                    "target_weight": 0.0,
                    "delta_weight": 0.0,
                    "instruction": "No position-plan rows are available.",
                }
            ]
        )

    target_column = _first_existing_column(
        position_plan,
        (
            "scenario_adjusted_weight",
            "risk_adjusted_weight",
            "target_weight",
            "pre_risk_target_weight",
        ),
    )
    current_column = _first_existing_column(
        position_plan,
        ("current_weight", "base_weight", "weight"),
    )
    if target_column is None or current_column is None:
        return pd.DataFrame(
            [
                {
                    "step": 1,
                    "ticker": "Portfolio",
                    "action": "REVIEW",
                    "current_weight": 0.0,
                    "target_weight": 0.0,
                    "delta_weight": 0.0,
                    "instruction": "Position-plan weights are incomplete; review the raw Command Center table.",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in position_plan.copy().iterrows():
        ticker = str(row.get("ticker", "Portfolio"))
        action = str(row.get("action", "HOLD"))
        current_weight = _as_float(row.get(current_column))
        target_weight = _as_float(row.get(target_column))
        delta_weight = _as_float(row.get("delta_weight", target_weight - current_weight))
        if action == "HOLD" and abs(delta_weight) < DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT:
            continue
        rows.append(
            {
                "step": len(rows) + 1,
                "ticker": ticker,
                "action": action,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "instruction": _sizing_instruction(
                    ticker,
                    action,
                    current_weight,
                    target_weight,
                    delta_weight,
                ),
            }
        )

    if not rows:
        rows.append(
            {
                "step": 1,
                "ticker": "Portfolio",
                "action": "HOLD",
                "current_weight": 0.0,
                "target_weight": 0.0,
                "delta_weight": 0.0,
                "instruction": "No material sizing change is currently recommended.",
            }
        )
    return pd.DataFrame(rows)


def _scenario_bridge_table(baseline_run: BaselineRun) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    rows = [
        {
            "bridge_step": "Scenario probabilities",
            "current_read": (
                f"1M risk-off {_format_percent(trade_summary.get('one_month_risk_off_probability'))}; "
                f"transition {_format_percent(trade_summary.get('one_month_transition_probability'))}; "
                f"fragile upside {_format_percent(trade_summary.get('one_month_fragile_upside_probability'))}"
            ),
            "how_it_changes_action": "Higher risk-off, transition, or fragile-upside probabilities shrink the allowed risk budget before tickets are created.",
        },
        {
            "bridge_step": "Scenario/event/macro multiplier",
            "current_read": _format_decimal(trade_summary.get("scenario_event_macro_multiplier")),
            "how_it_changes_action": "This scales down the base strategy target when scenarios, current events, or tested macro pressure argue for caution.",
        },
        {
            "bridge_step": "Portfolio risk multiplier",
            "current_read": _format_decimal(trade_summary.get("portfolio_risk_multiplier")),
            "how_it_changes_action": "This applies factor, beta, expected-shortfall, stress-loss, and concentration limits after the scenario overlay.",
        },
        {
            "bridge_step": "Final risk budget",
            "current_read": _format_decimal(trade_summary.get("risk_budget_multiplier")),
            "how_it_changes_action": "This is the final exposure throttle used to move from the base position to the scenario-adjusted target.",
        },
        {
            "bridge_step": "Posture calibration",
            "current_read": (
                f"{trade_summary.get('posture_calibration_signal', 'n/a')}; "
                f"opportunity {_format_percent(trade_summary.get('opportunity_pressure'))}"
            ),
            "how_it_changes_action": str(
                trade_summary.get(
                    "posture_calibration_note",
                    "Checks whether defensive sizing may be over-bearish.",
                )
            ),
        },
    ]
    if risk_summary:
        rows.append(
            {
                "bridge_step": "Risk constraints",
                "current_read": str(risk_summary.get("applied_constraints", "none")),
                "how_it_changes_action": (
                    f"Post-risk ES95 is {_format_percent(risk_summary.get('post_expected_shortfall_95'))}; "
                    f"max stress loss is {_format_percent(risk_summary.get('post_max_stress_loss'))}; "
                    f"AI beta is {_format_decimal(risk_summary.get('post_ai_beta'))}."
                ),
            }
        )

    scenario_links = baseline_run.trade_decision.scenario_links
    if not scenario_links.empty:
        for _, row in scenario_links.head(3).iterrows():
            rows.append(
                {
                    "bridge_step": f"Top scenario: {row.get('scenario', 'scenario')}",
                    "current_read": (
                        f"{_format_percent(row.get('probability'))}; "
                        f"{row.get('risk_bucket', 'unknown bucket')}"
                    ),
                    "how_it_changes_action": str(
                        row.get("expected_bot_posture")
                        or row.get("off_ramp")
                        or row.get("confirmation")
                        or ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def _operating_evidence_table(
    baseline_run: BaselineRun,
    headline: ActionHeadline,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in headline.drivers.head(6).iterrows():
        rows.append(
            {
                "evidence_type": "headline_driver",
                "evidence": f"{row.get('driver')}: {row.get('signal')}",
                "interpretation": str(row.get("detail", "")),
            }
        )

    decision_evidence = baseline_run.trade_decision.evidence
    if not decision_evidence.empty:
        for _, row in decision_evidence.head(6).iterrows():
            signal = row.get("signal", row.get("evidence", "decision evidence"))
            impact = row.get("impact", row.get("interpretation", ""))
            rows.append(
                {
                    "evidence_type": str(row.get("evidence_type", "trade_decision")),
                    "evidence": str(signal),
                    "interpretation": str(impact),
                }
            )

    if not rows:
        rows.append(
            {
                "evidence_type": "none",
                "evidence": "No evidence rows available.",
                "interpretation": "Review raw diagnostics in the section tabs.",
            }
        )
    return pd.DataFrame(rows)


def _scenario_effect_sentence(
    trade_summary: dict[str, object],
    risk_summary: dict[str, object],
) -> dict[str, str]:
    risk_off = _format_percent(trade_summary.get("one_month_risk_off_probability"))
    transition = _format_percent(trade_summary.get("one_month_transition_probability"))
    final_budget = _format_decimal(trade_summary.get("risk_budget_multiplier"))
    portfolio_multiplier = _format_decimal(trade_summary.get("portfolio_risk_multiplier"))
    constraints = str(risk_summary.get("applied_constraints", "none")) if risk_summary else "none"
    return {
        "answer": f"1M risk-off {risk_off}, transition {transition}",
        "detail": (
            f"Scenarios feed the risk budget first, then the risk engine applies constraints. "
            f"Final risk budget is {final_budget}; portfolio-risk multiplier is {portfolio_multiplier}; "
            f"constraints: {constraints}."
        ),
    }


def _decision_sanity_sentence(trade_summary: dict[str, object]) -> dict[str, str]:
    signal = str(trade_summary.get("decision_sanity_signal", "No sanity cap needed"))
    note = str(
        trade_summary.get(
            "decision_sanity_note",
            "No event/news-only de-risk cap is active for this recommendation.",
        )
    )
    cap_applied = bool(trade_summary.get("decision_sanity_cap_applied", False))
    breaks = str(trade_summary.get("market_confirmation_breaks", "none"))
    break_count = _format_decimal(trade_summary.get("market_confirmation_break_count"))
    cap = _format_percent(trade_summary.get("event_only_max_defensive_add"))
    return {
        "tone": "warning" if cap_applied else "success",
        "answer": signal,
        "detail": f"{note} Confirmation breaks: {break_count} ({breaks}); event-only defensive-add cap: {cap}.",
    }


def _posture_calibration_sentence(trade_summary: dict[str, object]) -> dict[str, str]:
    status = str(trade_summary.get("posture_calibration_status", "not_available"))
    signal = str(trade_summary.get("posture_calibration_signal", "No calibration signal"))
    note = str(
        trade_summary.get(
            "posture_calibration_note",
            "No posture calibration is available for this run.",
        )
    )
    current_risk = _format_percent(trade_summary.get("current_risk_asset_weight"))
    target_risk = _format_percent(trade_summary.get("target_risk_asset_weight"))
    risk_on = _format_percent(trade_summary.get("one_month_risk_on_probability"))
    constructive = _format_percent(trade_summary.get("constructive_scenario_probability"))
    opportunity = _format_percent(trade_summary.get("opportunity_pressure"))
    tone = {
        "defense_justified": "success",
        "event_defense_review": "warning",
        "under_risk_review": "warning",
        "opportunity_cost_watch": "warning",
        "upside_participation_ok": "success",
        "balanced": "success",
    }.get(status, "warning")
    return {
        "tone": tone,
        "answer": signal,
        "detail": (
            f"{note} Current risk assets {current_risk}; target risk assets {target_risk}; "
            f"risk-on {risk_on}; constructive {constructive}; opportunity pressure {opportunity}."
        ),
    }


def _primary_sizing_sentence(position_plan: pd.DataFrame) -> dict[str, str]:
    sizing_steps = _recommended_sizing_steps(position_plan)
    material = sizing_steps[sizing_steps["ticker"] != "Portfolio"]
    if material.empty:
        return {
            "answer": "No Material Change",
            "detail": "The current plan does not require a meaningful target-weight move.",
        }
    largest = material.copy()
    largest["abs_delta"] = largest["delta_weight"].astype(float).abs()
    row = largest.sort_values("abs_delta", ascending=False).iloc[0]
    return {
        "answer": f"{str(row['action']).replace('_', ' ').title()} {row['ticker']}",
        "detail": (
            f"Largest proposed move is {_format_percent(row['delta_weight'])}: "
            f"{_format_percent(row['current_weight'])} to {_format_percent(row['target_weight'])}. "
            "Translate each percentage-point change into dollars using the account value in Forward Test."
        ),
    }


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _as_float(value: object, default: float = 0.0) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _sizing_instruction(
    ticker: str,
    action: str,
    current_weight: float,
    target_weight: float,
    delta_weight: float,
) -> str:
    if abs(delta_weight) < 0.005:
        return f"Hold {ticker} near {_format_percent(target_weight)}."
    verb = "Add to" if delta_weight > 0 else "Reduce"
    if action in {"SELL", "EXIT"}:
        verb = "Exit or reduce"
    return (
        f"{verb} {ticker} by {_format_percent(abs(delta_weight))} of the account, "
        f"moving from {_format_percent(current_weight)} to {_format_percent(target_weight)}."
    )


def _render_decision_brief(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    experiment_scorecards: pd.DataFrame,
) -> None:
    st.subheader("Decision Brief")
    st.caption(
        "Research and performance context. Use the action headline and operating brief for the operating summary."
    )
    with st.expander("Decision context", expanded=False):
        cards = _decision_brief_cards(
            baseline_run=baseline_run,
            headline=headline,
            open_ticket_count=open_ticket_count,
            experiment_scorecards=experiment_scorecards,
        )
        st.markdown(
            '<div class="brief-grid">'
            + "".join(_brief_card_html(card) for card in cards)
            + "</div>",
            unsafe_allow_html=True,
        )

        conclusions = _decision_conclusions_table(
            baseline_run=baseline_run,
            headline=headline,
            experiment_scorecards=experiment_scorecards,
        )
        st.caption("Interpretation layer: read this before scanning detailed tables.")
        st.dataframe(conclusions, use_container_width=True, hide_index=True)

        watch_items = _decision_watch_items(baseline_run)
        if not watch_items.empty:
            st.caption("What would change the decision")
            st.dataframe(_display_metrics(watch_items), use_container_width=True, hide_index=True)


def _decision_brief_cards(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    experiment_scorecards: pd.DataFrame,
) -> list[dict[str, str]]:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    strongest_experiment = _strongest_experiment_summary(experiment_scorecards)
    recent_tension = _recent_performance_tension(baseline_run)
    target_position = str(
        trade_summary.get("scenario_adjusted_position")
        or trade_summary.get("base_position")
        or "No target position available."
    )
    risk_level = str(risk_summary.get("portfolio_risk_level", "not available"))
    constraints = str(risk_summary.get("applied_constraints", "none"))
    return [
        {
            "tone": "warning" if risk_level != "within_limits" else "success",
            "label": "Risk rationale",
            "answer": risk_level.replace("_", " ").title(),
            "detail": (
                f"Portfolio risk constraints: {constraints}. "
                f"Current market state is {baseline_run.current_state.risk_status.upper()} "
                f"with score {baseline_run.current_state.risk_score:.2f}."
            ),
        },
        {
            "tone": "warning",
            "label": "Main tension",
            "answer": recent_tension["answer"],
            "detail": (
                f"{recent_tension['detail']} Current target posture is {target_position}; "
                f"open recommendation tickets: {open_ticket_count}."
            ),
        },
        {
            "tone": (
                "success" if strongest_experiment["decision"] == "promote_candidate" else "warning"
            ),
            "label": "Research takeaway",
            "answer": strongest_experiment["answer"],
            "detail": strongest_experiment["detail"],
        },
    ]


def _brief_card_html(card: dict[str, str]) -> str:
    return (
        f'<div class="brief-card brief-card-{html.escape(card["tone"])}">'
        f'<p class="brief-label">{html.escape(card["label"])}</p>'
        f'<p class="brief-answer">{html.escape(card["answer"])}</p>'
        f'<p class="brief-detail">{html.escape(card["detail"])}</p>'
        "</div>"
    )


def _decision_conclusions_table(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    experiment_scorecards: pd.DataFrame,
) -> pd.DataFrame:
    trade_summary = _first_display_row(baseline_run.trade_decision.summary)
    risk_summary = _portfolio_risk_summary(baseline_run)
    strongest_experiment = _strongest_experiment_summary(experiment_scorecards)
    recent_tension = _recent_performance_tension(baseline_run)
    rows = [
        {
            "question": "What is the system asking me to do?",
            "conclusion": str(trade_summary.get("recommended_action", headline.label)).replace(
                "_", " "
            ),
            "evidence": str(trade_summary.get("human_explanation", headline.explanation)),
            "drill_down": "Trade Plan",
        },
        {
            "question": "Is this mostly a signal or a risk-control decision?",
            "conclusion": str(risk_summary.get("portfolio_risk_level", "not available")).replace(
                "_", " "
            ),
            "evidence": (
                f"Constraints: {risk_summary.get('applied_constraints', 'none')}; "
                f"post ES95 { _format_percent(risk_summary.get('post_expected_shortfall_95'))}; "
                f"max stress loss { _format_percent(risk_summary.get('post_max_stress_loss'))}."
            ),
            "drill_down": "Risk Engine",
        },
        {
            "question": "What is the strongest tested approach right now?",
            "conclusion": strongest_experiment["answer"],
            "evidence": strongest_experiment["detail"],
            "drill_down": "Research",
        },
        {
            "question": "What is the main tension?",
            "conclusion": recent_tension["answer"],
            "evidence": recent_tension["detail"],
            "drill_down": "Performance",
        },
    ]
    return pd.DataFrame(rows)


def _decision_watch_items(baseline_run: BaselineRun) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_links = baseline_run.trade_decision.scenario_links
    if not scenario_links.empty:
        for _, row in scenario_links.head(3).iterrows():
            rows.append(
                {
                    "watch_item": str(row.get("scenario", "scenario")),
                    "current_read": _format_percent(row.get("probability")),
                    "why_it_matters": str(row.get("expected_bot_posture", "")),
                    "off_ramp_or_confirmation": str(
                        row.get("off_ramp") or row.get("confirmation") or ""
                    ),
                }
            )
    if baseline_run.portfolio_risk is not None and not baseline_run.portfolio_risk.summary.empty:
        risk_row = baseline_run.portfolio_risk.summary.iloc[0]
        rows.append(
            {
                "watch_item": "Portfolio constraints",
                "current_read": str(risk_row.get("applied_constraints", "none")),
                "why_it_matters": "These constraints directly change target position sizing.",
                "off_ramp_or_confirmation": "Relax only if scenario and stress losses normalize.",
            }
        )
    if not baseline_run.news_monitor.triage.empty:
        active = baseline_run.news_monitor.triage[
            baseline_run.news_monitor.triage["activation_status"]
            .astype(str)
            .str.contains("event_risk", na=False)
        ]
        if not active.empty:
            rows.append(
                {
                    "watch_item": "Active news pressure",
                    "current_read": f"{len(active):,} active items",
                    "why_it_matters": "News has been converted into event-risk context.",
                    "off_ramp_or_confirmation": "Watch whether price, credit, or scenario drivers confirm it.",
                }
            )
    return pd.DataFrame(rows)


def _portfolio_risk_summary(baseline_run: BaselineRun) -> dict[str, object]:
    risk = baseline_run.portfolio_risk or baseline_run.trade_decision.portfolio_risk
    if risk is None or risk.summary.empty:
        return {}
    return risk.summary.iloc[0].to_dict()


def _strongest_experiment_summary(scorecards: pd.DataFrame) -> dict[str, str]:
    if scorecards.empty:
        return {
            "answer": "No experiment result loaded",
            "detail": "Run experiment iterations to populate the research monitor.",
            "decision": "",
        }
    frame = scorecards.copy()
    if "robustness_score" in frame:
        robust = frame[frame["robustness_score"].notna()]
        if not robust.empty:
            frame = robust
    for column in ["promotion_score", "robustness_score", "calmar"]:
        if column not in frame:
            frame[column] = 0.0
    top = frame.sort_values(
        ["promotion_score", "robustness_score", "calmar"],
        ascending=False,
    ).iloc[0]
    strategy = str(top.get("strategy", "unknown"))
    decision = str(top.get("promotion_decision", "unknown"))
    detail = (
        f"{decision.replace('_', ' ')}; CAGR {_format_percent(top.get('cagr'))}; "
        f"max drawdown {_format_percent(top.get('max_drawdown'))}; "
        f"walk-forward positive rate {_format_percent(top.get('walk_forward_positive_rate'))}."
    )
    return {"answer": strategy, "detail": detail, "decision": decision}


def _recent_performance_tension(baseline_run: BaselineRun) -> dict[str, str]:
    latest = baseline_run.prices.index.max()
    start = pd.Timestamp(latest) - pd.DateOffset(days=90)
    window = window_performance_frame(
        baseline_run.results,
        start=start,
        end=latest,
    )
    if window.empty:
        return {
            "answer": "No recent window available",
            "detail": "Recent performance diagnostics are unavailable.",
        }
    qqq = _window_return(window, "buy_hold_qqq")
    primary = _window_return(window, "drawdown_managed_dual_momentum")
    if qqq is not None and primary is not None and qqq > primary + 0.05:
        return {
            "answer": "Momentum is fighting risk control",
            "detail": (
                f"QQQ is up {_format_percent(qqq)} over ~90 days versus "
                f"{_format_percent(primary)} for the primary strategy, but the risk engine is still "
                "throttling exposure."
            ),
        }
    leader = window.sort_values("total_return", ascending=False).iloc[0]
    return {
        "answer": f"Recent leader: {leader['strategy']}",
        "detail": (
            f"Best ~90 day return is {_format_percent(leader['total_return'])}; "
            "compare this against risk status before adding exposure."
        ),
    }


def _window_return(window: pd.DataFrame, strategy: str) -> float | None:
    rows = window[window["strategy"] == strategy]
    if rows.empty:
        return None
    value = rows.iloc[0].get("total_return")
    numeric = _optional_float(value)
    return numeric


def _first_display_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _brief_tone(level: str) -> str:
    if level == "critical_actions":
        return "critical"
    if level == "small_actions":
        return "warning"
    return "success"
