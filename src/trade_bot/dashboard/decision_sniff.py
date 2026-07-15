from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics, _format_decimal, _format_percent
from trade_bot.research.approach_explorer import (
    build_approach_catalog,
    execution_for_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_judgement import (
    build_defensive_judgement_audit,
    current_defensive_setup_context,
    defensive_false_alarm_bayes_update,
    effective_defensive_weight,
    load_scenario_context,
)
from trade_bot.research.signal_state import SignalStateReport, build_signal_state_report


@dataclass(frozen=True)
class OperationalSniffRead:
    strategy_name: str
    benchmark_ticker: str
    report: SignalStateReport
    false_alarm_update: dict[str, float | int | str | None]
    current_defensive_weight: float | None


def build_operational_sniff_read(
    *,
    baseline_run: BaselineRun,
    bot_config: Any,
    strategy_name: str,
    result: BacktestResult | None = None,
    benchmark_ticker: str | None = None,
    threshold: float = 0.65,
    horizon: str = "1m",
) -> OperationalSniffRead | None:
    if result is None:
        result = baseline_run.results.get(strategy_name)
    if result is None or baseline_run.prices.empty:
        return None
    resolved = _resolve_strategy_and_execution(
        bot_config=bot_config,
        strategy_name=strategy_name,
    )
    if resolved is None:
        return None
    strategy, execution = resolved
    benchmark = _default_benchmark(baseline_run.prices, benchmark_ticker)
    report = build_signal_state_report(
        result=result,
        prices=baseline_run.prices,
        strategy=strategy,
        execution=execution,
        include_overlay_backtest=False,
    )
    defensive = effective_defensive_weight(result)
    current_defensive = float(defensive.iloc[-1]) if not defensive.empty else None
    scenario_context = load_scenario_context()
    audit = build_defensive_judgement_audit(
        result,
        baseline_run.prices,
        thresholds=(threshold,),
        benchmark_ticker=benchmark,
        scenario_context=scenario_context,
    )
    current_setup = current_defensive_setup_context(
        result,
        baseline_run.prices,
        benchmark_ticker=benchmark,
        scenario_context=scenario_context,
    )
    false_alarm = defensive_false_alarm_bayes_update(
        audit["events"],
        threshold=threshold,
        horizon=horizon,
        current_defensive_weight=current_defensive,
        current_setup=current_setup,
    )
    return OperationalSniffRead(
        strategy_name=strategy_name,
        benchmark_ticker=benchmark,
        report=report,
        false_alarm_update=false_alarm,
        current_defensive_weight=current_defensive,
    )


def render_operational_sniff_read(
    read: OperationalSniffRead | None,
    *,
    title: str = "Decision Sniff Test",
    include_summary: bool = True,
    include_details: bool = True,
    expanded_details: bool = False,
) -> None:
    st.markdown(f"**{title}**")
    st.caption(
        "Current decision context: broad market state, asset confirmation, and recent "
        "false-alarm evidence for defensive sizing."
    )
    if read is None or read.report.latest.empty:
        st.info("Decision sniff-test context is not available for this strategy yet.")
        return

    latest = read.report.latest
    false_alarm = read.false_alarm_update
    cols = st.columns(5)
    _helped_metric(
        cols[0],
        "Top-Down State",
        str(latest.get("top_down_signal", "n/a")).replace("_", " ").title(),
    )
    _helped_metric(cols[1], "Top-Down Score", _format_decimal(latest.get("top_down_score")))
    _helped_metric(cols[2], "Confirmed Risk Assets", _count_value(latest.get("confirmed_assets")))
    _helped_metric(cols[3], "Current Defensive", _format_percent(read.current_defensive_weight))
    _helped_metric(
        cols[4],
        "Similar False Alarm",
        _format_percent(false_alarm.get("similar_false_alarm_rate")),
    )

    if include_summary:
        st.info(_operational_sniff_sentence(read))
    if not include_details:
        return
    with st.expander("Sniff-test detail", expanded=expanded_details):
        if not read.report.assets.empty:
            columns = [
                "ticker",
                "target_weight",
                "current_weight",
                "confirmation_state",
                "top_down_signal",
                "bottom_up_signal",
                "state_read",
            ]
            available = [column for column in columns if column in read.report.assets]
            _render_metric_dataframe(
                _display_metrics(read.report.assets[available]),
                hide_index=True,
            )


def _resolve_strategy_and_execution(
    *,
    bot_config: Any,
    strategy_name: str,
) -> tuple[StrategyConfig, ExecutionConfig] | None:
    configured = getattr(bot_config, "strategies", {}) or {}
    if strategy_name in configured:
        return configured[strategy_name], bot_config.execution
    catalog = build_approach_catalog(bot_config)
    if catalog.empty or "strategy" not in catalog:
        return None
    matches = catalog[catalog["strategy"].astype(str).eq(str(strategy_name))]
    if matches.empty:
        return None
    row = matches.iloc[0]
    try:
        return strategy_from_catalog_row(row), execution_for_catalog_row(row, bot_config.execution)
    except (TypeError, ValueError, AttributeError):
        return None


def _default_benchmark(prices: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in prices.columns:
        return requested
    for ticker in ("QQQ", "SPY"):
        if ticker in prices.columns:
            return ticker
    return str(prices.columns[0])


def _count_value(value: object) -> str:
    try:
        return str(int(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def _operational_sniff_sentence(read: OperationalSniffRead) -> str:
    latest = read.report.latest
    false_alarm = read.false_alarm_update
    top_down = str(latest.get("top_down_signal", "n/a")).replace("_", " ")
    confirmed = _count_value(latest.get("confirmed_assets"))
    partial = _count_value(latest.get("partial_assets"))
    posterior = false_alarm.get("posterior_false_alarm_rate")
    similar_rate = false_alarm.get("similar_false_alarm_rate")
    false_alarm_read = _format_percent(similar_rate if similar_rate is not None else posterior)
    similar_count = _count_value(false_alarm.get("similar_episode_starts"))
    label = str(false_alarm.get("sniff_test_label", "mixed_context")).replace("_", " ")
    return (
        f"For {read.strategy_name}, the broad state is {top_down}; {confirmed} assets are "
        f"fully confirmed and {partial} are partial. Recent/contextual evidence puts the "
        f"{read.benchmark_ticker} false-alarm read near {false_alarm_read} across "
        f"{similar_count} similar historical setups ({label}). "
        "This is an explanatory challenge read, not an allocation override."
    )
