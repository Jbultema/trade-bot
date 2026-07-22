from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import AllocationPolicyConfig, StrategyConfig
from trade_bot.DEFAULTS import DEFAULT_FORWARD_TEST_STRATEGY
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun, build_trade_decision


class StrategyDecisionUnavailableError(LookupError):
    """Raised when a named operating strategy cannot be resolved exactly and safely."""


def operating_strategy_config(
    bot_config: object,
    strategy_name: str,
    *,
    current_strategy: str,
) -> StrategyConfig | None:
    """Resolve the configuration behind an operating alias or exact strategy name."""

    strategies = getattr(bot_config, "strategies", {})
    if not isinstance(strategies, dict):
        return None
    requested = str(strategy_name or "").strip()
    if requested in {"", DEFAULT_FORWARD_TEST_STRATEGY, current_strategy}:
        requested = str(getattr(bot_config, "primary_strategy", "") or current_strategy)
    strategy = strategies.get(requested)
    return strategy if isinstance(strategy, StrategyConfig) else None


def resolve_trade_decision_for_strategy(
    baseline_run: BaselineRun,
    strategy_name: str,
    *,
    strategy_config: StrategyConfig | None = None,
    allocation_policy: AllocationPolicyConfig | None = None,
) -> TradeDecisionRun:
    """Return the trade decision that should drive a named operating book."""

    selected_strategy = str(strategy_name or "").strip()
    current_strategy = trade_decision_strategy_name(baseline_run.trade_decision)
    if selected_strategy in {"", DEFAULT_FORWARD_TEST_STRATEGY, current_strategy}:
        if strategy_config is None:
            raise StrategyDecisionUnavailableError(
                "The configured primary strategy cannot be matched to an operating policy."
            )
        if not str(strategy_config.defensive_ticker or "").strip():
            raise StrategyDecisionUnavailableError(
                f"Operating strategy {current_strategy!r} has no explicit defensive ticker. "
                "Scenario-adjusted operating decisions are unavailable for this strategy."
            )
        return baseline_run.trade_decision

    selected_result = baseline_run.results.get(selected_strategy)
    if selected_result is None:
        available = ", ".join(sorted(str(name) for name in baseline_run.results)) or "none"
        raise StrategyDecisionUnavailableError(
            f"Operating strategy {selected_strategy!r} is not present in this run. "
            f"Available strategies: {available}."
        )
    if strategy_config is None:
        raise StrategyDecisionUnavailableError(
            f"Operating strategy {selected_strategy!r} has no matching configuration; "
            "its defensive policy cannot be resolved safely."
        )
    defensive_ticker = str(strategy_config.defensive_ticker or "").strip()
    if not defensive_ticker:
        raise StrategyDecisionUnavailableError(
            f"Operating strategy {selected_strategy!r} has no explicit defensive ticker. "
            "Scenario-adjusted operating decisions are unavailable for this strategy."
        )

    return build_trade_decision(
        primary_result=selected_result,
        current_state=baseline_run.current_state,
        event_risk=baseline_run.event_risk,
        news_monitor=baseline_run.news_monitor,
        signal_inclusion=_neutral_signal_inclusion(selected_result, selected_strategy),
        prices=baseline_run.prices,
        defensive_ticker=defensive_ticker,
        allocation_policy=allocation_policy,
    )


def trade_decision_strategy_name(trade_decision: TradeDecisionRun) -> str:
    summary = getattr(trade_decision, "summary", pd.DataFrame())
    if summary.empty or "strategy" not in summary:
        return ""
    return str(summary.iloc[0].get("strategy", "") or "").strip()


def _neutral_signal_inclusion(
    selected_result: BacktestResult,
    strategy_name: str,
) -> SignalInclusionRun:
    """Remove primary-strategy macro authority from an alternate operating decision."""

    return SignalInclusionRun(
        summary=pd.DataFrame(
            [
                {
                    "signal_group": "all_macro_categories",
                    "decision": "not_evaluated_for_strategy",
                    "latest_pressure_state": "not_evaluated",
                    "rationale": (
                        f"Macro inclusion was evaluated for the configured primary, not "
                        f"{strategy_name}; it has no allocation authority here."
                    ),
                }
            ]
        ),
        pressure=pd.DataFrame(),
        results={selected_result.name: selected_result},
        metrics=pd.DataFrame(),
        window_summary=pd.DataFrame(),
    )
