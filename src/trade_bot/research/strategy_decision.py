from __future__ import annotations

import pandas as pd

from trade_bot.DEFAULTS import DEFAULT_FORWARD_TEST_STRATEGY
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.trade_decision import TradeDecisionRun, build_trade_decision


def resolve_trade_decision_for_strategy(
    baseline_run: BaselineRun,
    strategy_name: str,
) -> TradeDecisionRun:
    """Return the trade decision that should drive a named operating book."""

    selected_strategy = str(strategy_name or "").strip()
    current_strategy = trade_decision_strategy_name(baseline_run.trade_decision)
    if selected_strategy in {"", DEFAULT_FORWARD_TEST_STRATEGY, current_strategy}:
        return baseline_run.trade_decision

    selected_result = baseline_run.results.get(selected_strategy)
    if selected_result is None:
        return baseline_run.trade_decision

    return build_trade_decision(
        primary_result=selected_result,
        current_state=baseline_run.current_state,
        event_risk=baseline_run.event_risk,
        news_monitor=baseline_run.news_monitor,
        signal_inclusion=baseline_run.signal_inclusion,
        prices=baseline_run.prices,
    )


def trade_decision_strategy_name(trade_decision: TradeDecisionRun) -> str:
    summary = getattr(trade_decision, "summary", pd.DataFrame())
    if summary.empty or "strategy" not in summary:
        return ""
    return str(summary.iloc[0].get("strategy", "") or "").strip()
