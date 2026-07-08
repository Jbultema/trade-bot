from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.portfolio.risk import PortfolioRiskConfig, PortfolioRiskRun, build_portfolio_risk
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import (
    CurrentStateRun,
    _risk_score,
    _risk_status,
    _risk_summary,
    build_confirmation_matrix,
    build_market_health,
    build_scenario_outlook,
    momentum_state_table,
)
from trade_bot.research.driver_rotation import build_driver_rotation_table
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.future_scenarios import build_scenario_lattice
from trade_bot.research.narrative_signals import build_narrative_signal_table
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.regime_instability import build_regime_instability_index
from trade_bot.research.trade_decision import TradeDecisionRun

DEFAULT_OPERATING_HISTORY_PRIMARY_STRATEGY = "drawdown_managed_dual_momentum"
DEFAULT_OPERATING_HISTORY_SOURCE = "reconstructed_price_fast_point_in_time"
DEFAULT_OPERATING_HISTORY_NOTE = (
    "Reconstructed by truncating local prices and strategy outputs at the history date. "
    "Current event/news overlays are intentionally excluded; macro history comes from live "
    "snapshots, not this fast backfill."
)


@dataclass(frozen=True)
class OperatingHistoryFrames:
    metrics: pd.DataFrame
    components: pd.DataFrame
    scenario_drivers: pd.DataFrame
    driver_rotation: pd.DataFrame


def reconstruct_operating_history(
    baseline_run: BaselineRun,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    frequency: str = "W-WED",
    max_points: int = 260,
    daily_tail_market_days: int = 30,
    min_history_days: int = 252,
    primary_strategy: str = DEFAULT_OPERATING_HISTORY_PRIMARY_STRATEGY,
) -> OperatingHistoryFrames:
    """Reconstruct lightweight operating metric history without fabricating snapshots."""

    dates = _sample_history_dates(
        baseline_run.prices.index,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        max_points=max_points,
        daily_tail_market_days=daily_tail_market_days,
        min_history_days=min_history_days,
    )
    if not dates:
        return _empty_history_frames()
    if primary_strategy not in baseline_run.results:
        msg = f"Primary strategy {primary_strategy!r} is not available in baseline results."
        raise ValueError(msg)

    metric_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    scenario_driver_rows: list[dict[str, object]] = []
    driver_rotation_rows: list[dict[str, object]] = []
    event_risk = _empty_event_risk()
    news_monitor = _empty_news_monitor()

    for history_date in dates:
        date_label = str(history_date.date())
        prices = baseline_run.prices.loc[:history_date].copy()
        if prices.empty:
            continue
        results = _slice_results(baseline_run.results, history_date)
        primary_result = results.get(primary_strategy)
        if primary_result is None or primary_result.weights.empty:
            continue
        current_state = _build_fast_current_state(prices)
        trade_decision = _build_fast_trade_decision(
            primary_result=primary_result,
            current_state=current_state,
            prices=prices,
        )
        portfolio_risk = trade_decision.portfolio_risk
        base = _base_fields(date_label)
        metric_rows.append(
            {
                **base,
                **_metric_fields(
                    current_state=current_state,
                    trade_decision=trade_decision,
                    portfolio_risk=portfolio_risk,
                ),
            }
        )
        component_rows.extend(_component_rows(current_state.regime_instability_components, base))
        scenario_driver_rows.extend(_scenario_driver_rows(current_state.scenario_drivers, base))
        driver_rotation_rows.extend(
            _driver_rotation_rows(
                prices=prices,
                current_state=current_state,
                news_monitor=news_monitor,
                event_risk=event_risk,
                base=base,
            )
        )

    return OperatingHistoryFrames(
        metrics=pd.DataFrame(metric_rows),
        components=pd.DataFrame(component_rows),
        scenario_drivers=pd.DataFrame(scenario_driver_rows),
        driver_rotation=pd.DataFrame(driver_rotation_rows),
    )


def _sample_history_dates(
    index: pd.Index,
    *,
    start_date: str | None,
    end_date: str | None,
    frequency: str,
    max_points: int,
    daily_tail_market_days: int,
    min_history_days: int,
) -> list[pd.Timestamp]:
    dates = pd.to_datetime(index, errors="coerce").dropna().sort_values().unique()
    if len(dates) <= min_history_days:
        return []
    available = pd.DatetimeIndex(dates)
    start = pd.Timestamp(start_date) if start_date else available[min_history_days]
    end = pd.Timestamp(end_date) if end_date else available[-1]
    scoped = available[(available >= start) & (available <= end)]
    if scoped.empty:
        return []
    if daily_tail_market_days > 0:
        daily_tail = scoped[-daily_tail_market_days:]
        tail_start = daily_tail.min() if not daily_tail.empty else end
        historical_scoped = scoped[scoped < tail_start]
    else:
        historical_scoped = scoped
        daily_tail = pd.DatetimeIndex([])

    if frequency.lower() in {"d", "daily", "b", "business"}:
        sampled = historical_scoped
        daily_tail = pd.DatetimeIndex([])
    else:
        sample_series = pd.Series(historical_scoped, index=historical_scoped)
        sampled = (
            pd.DatetimeIndex(sample_series.resample(frequency).last().dropna().values)
            if not sample_series.empty
            else pd.DatetimeIndex([])
        )
    sampled = sampled.drop_duplicates().sort_values()
    if max_points > 0 and len(sampled) > max_points:
        sampled = sampled[-max_points:]
    sampled = sampled.union(daily_tail).drop_duplicates().sort_values()
    return [pd.Timestamp(value) for value in sampled]


def _slice_results(results: dict[str, BacktestResult], history_date: pd.Timestamp) -> dict[str, BacktestResult]:
    output = {}
    for name, result in results.items():
        sliced = BacktestResult(
            name=result.name,
            equity=result.equity.loc[:history_date],
            returns=result.returns.loc[:history_date],
            gross_returns=result.gross_returns.loc[:history_date],
            weights=result.weights.loc[:history_date],
            target_weights=result.target_weights.loc[:history_date],
            turnover=result.turnover.loc[:history_date],
            transaction_costs=result.transaction_costs.loc[:history_date],
        )
        if not sliced.returns.empty and not sliced.weights.empty:
            output[name] = sliced
    return output


def _build_fast_current_state(prices: pd.DataFrame) -> CurrentStateRun:
    clean_prices = prices.dropna(how="all").sort_index()
    market_date = str(clean_prices.index.max().date())
    momentum_state = momentum_state_table(clean_prices)
    confirmation_matrix = build_confirmation_matrix(clean_prices, momentum_state)
    market_health = build_market_health(clean_prices, momentum_state)
    risk_score = _risk_score(confirmation_matrix, market_health)
    risk_status = _risk_status(risk_score)
    risk_summary = _risk_summary(risk_status, risk_score, confirmation_matrix)
    scenario_lattice, scenario_drivers = build_scenario_lattice(
        confirmation_matrix,
        market_health,
        momentum_state,
        risk_score,
        risk_status,
    )
    scenario_outlook = build_scenario_outlook(scenario_lattice, risk_status)
    regime_instability, regime_instability_components = build_regime_instability_index(
        clean_prices
    )
    return CurrentStateRun(
        market_date=market_date,
        risk_score=risk_score,
        risk_status=risk_status,
        risk_summary=risk_summary,
        market_health=market_health,
        momentum_state=momentum_state,
        confirmation_matrix=confirmation_matrix,
        strategy_alerts=pd.DataFrame(),
        scenario_outlook=scenario_outlook,
        scenario_lattice=scenario_lattice,
        scenario_drivers=scenario_drivers,
        macro_signals=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        signal_coverage=pd.DataFrame(),
        data_quality=pd.DataFrame(),
        regime_instability=regime_instability,
        regime_instability_components=regime_instability_components,
    )


def _build_fast_trade_decision(
    *,
    primary_result: BacktestResult,
    current_state: CurrentStateRun,
    prices: pd.DataFrame,
    defensive_ticker: str = "BIL",
) -> TradeDecisionRun:
    base_weights = primary_result.weights.iloc[-1].astype(float)
    scenario_context = _scenario_context(current_state.scenario_lattice)
    risk_multiplier = _risk_status_multiplier(current_state.risk_status) * float(
        scenario_context["risk_multiplier"]
    )
    risk_multiplier = float(np.clip(risk_multiplier, 0.0, 1.0))
    adjusted_weights = _scenario_adjusted_weights(
        base_weights,
        risk_multiplier=risk_multiplier,
        defensive_ticker=defensive_ticker,
    )
    portfolio_risk = build_portfolio_risk(
        prices,
        adjusted_weights,
        current_state.scenario_lattice,
        current_weights=base_weights,
        config=PortfolioRiskConfig(defensive_ticker=defensive_ticker),
    )
    portfolio_summary = _first_row(portfolio_risk.summary)
    correlation = _first_row(portfolio_risk.correlation_regime)
    final_weights = (
        portfolio_risk.risk_adjusted_weights
        if not portfolio_risk.risk_adjusted_weights.empty
        else adjusted_weights
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": primary_result.name,
                "risk_status": current_state.risk_status,
                "risk_score": current_state.risk_score,
                "risk_budget_multiplier": _risk_budget_multiplier_from_weights(
                    base_weights,
                    final_weights,
                    defensive_ticker=defensive_ticker,
                ),
                "scenario_event_macro_multiplier": risk_multiplier,
                "portfolio_risk_multiplier": _safe_float(
                    portfolio_summary.get("portfolio_risk_multiplier")
                ),
                "one_month_risk_off_probability": scenario_context[
                    "risk_off_probability"
                ],
                "portfolio_expected_shortfall_95": _safe_float(
                    portfolio_summary.get("post_expected_shortfall_95")
                ),
                "portfolio_max_stress_loss": _safe_float(
                    portfolio_summary.get("post_max_stress_loss")
                ),
                "portfolio_equity_beta": _safe_float(
                    portfolio_summary.get("post_equity_beta")
                ),
                "portfolio_ai_beta": _safe_float(portfolio_summary.get("post_ai_beta")),
                "correlation_regime_shift": _safe_float(correlation.get("correlation_shift")),
            }
        ]
    )
    return TradeDecisionRun(
        summary=summary,
        position_plan=pd.DataFrame(),
        evidence=pd.DataFrame(),
        scenario_links=pd.DataFrame(),
        portfolio_risk=portfolio_risk,
    )


def _scenario_context(scenario_lattice: pd.DataFrame) -> dict[str, float]:
    if scenario_lattice.empty:
        return {
            "risk_multiplier": 1.0,
            "risk_off_probability": 0.0,
        }
    one_month = scenario_lattice[scenario_lattice["horizon"] == "1m"].copy()
    if one_month.empty:
        one_month = scenario_lattice.copy()
    risk_bucket = one_month["risk_bucket"].astype(str)
    risk_off_probability = float(
        one_month.loc[risk_bucket.str.contains("risk_off"), "probability"].sum()
    )
    transition_probability = float(
        one_month.loc[risk_bucket == "transition", "probability"].sum()
    )
    fragile_probability = float(
        one_month.loc[risk_bucket == "risk_on_fragile", "probability"].sum()
    )
    risk_multiplier = 1.0 - 0.55 * risk_off_probability
    risk_multiplier -= 0.20 * transition_probability
    risk_multiplier -= 0.15 * fragile_probability
    return {
        "risk_multiplier": float(np.clip(risk_multiplier, 0.40, 1.0)),
        "risk_off_probability": risk_off_probability,
    }


def _risk_status_multiplier(risk_status: str) -> float:
    return {
        "green": 1.0,
        "yellow": 0.90,
        "orange": 0.65,
        "red": 0.40,
    }.get(risk_status, 0.85)


def _scenario_adjusted_weights(
    base_weights: pd.Series,
    *,
    risk_multiplier: float,
    defensive_ticker: str,
) -> pd.Series:
    adjusted = base_weights.copy().astype(float)
    if defensive_ticker not in adjusted.index:
        adjusted.loc[defensive_ticker] = 0.0
    risk_assets = [ticker for ticker in adjusted.index if str(ticker) != defensive_ticker]
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


def _risk_budget_multiplier_from_weights(
    base_weights: pd.Series,
    final_weights: pd.Series,
    *,
    defensive_ticker: str,
) -> float:
    base_risk = _risk_asset_weight(base_weights, defensive_ticker=defensive_ticker)
    final_risk = _risk_asset_weight(final_weights, defensive_ticker=defensive_ticker)
    if base_risk <= 1e-9:
        return 1.0 if final_risk <= 1e-9 else 0.0
    return float(np.clip(final_risk / base_risk, 0.0, 1.0))


def _risk_asset_weight(weights: pd.Series, *, defensive_ticker: str) -> float:
    if weights.empty:
        return 0.0
    clean = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk_assets = [ticker for ticker in clean.index if str(ticker).upper() != defensive_ticker.upper()]
    return float(clean.loc[risk_assets].clip(lower=0.0).sum())


def _base_fields(history_date: str) -> dict[str, object]:
    return {
        "history_id": f"{DEFAULT_OPERATING_HISTORY_SOURCE}:{history_date}",
        "history_time": history_date,
        "snapshot_time": history_date,
        "market_date": history_date,
        "run_id": f"reconstructed:{history_date}",
        "source": DEFAULT_OPERATING_HISTORY_SOURCE,
        "reconstruction_note": DEFAULT_OPERATING_HISTORY_NOTE,
    }


def _metric_fields(
    *,
    current_state: object,
    trade_decision: TradeDecisionRun,
    portfolio_risk: PortfolioRiskRun | None,
) -> dict[str, object]:
    decision = _first_row(trade_decision.summary)
    portfolio_summary = _first_row(getattr(portfolio_risk, "summary", pd.DataFrame()))
    correlation = _first_row(getattr(portfolio_risk, "correlation_regime", pd.DataFrame()))
    instability = _first_row(getattr(current_state, "regime_instability", pd.DataFrame()))
    return {
        "risk_score": _safe_float(getattr(current_state, "risk_score", None)),
        "one_month_risk_off_probability": _safe_float(
            decision.get("one_month_risk_off_probability")
        ),
        "risk_budget_multiplier": _safe_float(decision.get("risk_budget_multiplier")),
        "portfolio_risk_multiplier": _coalesce_float(
            portfolio_summary.get("portfolio_risk_multiplier"),
            decision.get("portfolio_risk_multiplier"),
        ),
        "post_expected_shortfall_95": _coalesce_float(
            portfolio_summary.get("post_expected_shortfall_95"),
            decision.get("portfolio_expected_shortfall_95"),
        ),
        "post_max_stress_loss": _coalesce_float(
            portfolio_summary.get("post_max_stress_loss"),
            decision.get("portfolio_max_stress_loss"),
        ),
        "post_equity_beta": _coalesce_float(
            portfolio_summary.get("post_equity_beta"),
            decision.get("portfolio_equity_beta"),
        ),
        "post_ai_beta": _coalesce_float(
            portfolio_summary.get("post_ai_beta"),
            decision.get("portfolio_ai_beta"),
        ),
        "correlation_shift": _safe_float(correlation.get("correlation_shift")),
        "regime_instability_score": _safe_float(instability.get("regime_instability_score")),
        "spy_ytd_large_move_share": _safe_float(instability.get("spy_ytd_large_move_share")),
    }


def _component_rows(components: pd.DataFrame, base: dict[str, object]) -> list[dict[str, object]]:
    if components.empty or "component" not in components:
        return []
    rows = []
    for _, row in components.iterrows():
        rows.append(
            {
                **base,
                "history_id": f"{base['history_id']}:component:{row.get('component')}",
                "component": str(row.get("component", "")),
                "component_score": _safe_float(row.get("component_score")),
                "latest_value": _safe_float(row.get("latest_value")),
                "state": str(row.get("state", "")),
            }
        )
    return rows


def _scenario_driver_rows(drivers: pd.DataFrame, base: dict[str, object]) -> list[dict[str, object]]:
    if drivers.empty or "driver" not in drivers:
        return []
    rows = []
    for _, row in drivers.iterrows():
        rows.append(
            {
                **base,
                "history_id": f"{base['history_id']}:scenario_driver:{row.get('driver')}",
                "driver": str(row.get("driver", "")),
                "score": _safe_float(row.get("score")),
                "state": str(row.get("state", "")),
            }
        )
    return rows


def _driver_rotation_rows(
    *,
    prices: pd.DataFrame,
    current_state: object,
    news_monitor: NewsMonitorRun,
    event_risk: EventRiskRun,
    base: dict[str, object],
) -> list[dict[str, object]]:
    narrative_signals = build_narrative_signal_table(
        prices,
        news_triage=news_monitor.triage,
        events=event_risk.events,
    )
    rotation = build_driver_rotation_table(
        prices,
        current_state,
        narrative_signals=narrative_signals,
        news_triage=news_monitor.triage,
    )
    if rotation.empty or "driver" not in rotation:
        return []
    rows = []
    for _, row in rotation.iterrows():
        rows.append(
            {
                **base,
                "history_id": f"{base['history_id']}:driver_rotation:{row.get('driver')}",
                "driver": str(row.get("driver", "")),
                "driver_label": str(row.get("driver_label", row.get("driver", ""))),
                "current_activation": _safe_float(row.get("current_activation")),
                "proven_relevance": _safe_float(row.get("proven_relevance")),
                "change_30d": _safe_float(row.get("change_30d")),
                "change_90d": _safe_float(row.get("change_90d")),
                "model_role": str(row.get("model_role", "")),
            }
        )
    return rows


def _empty_event_risk() -> EventRiskRun:
    return EventRiskRun(
        events=(),
        asset_event_returns=pd.DataFrame(),
        strategy_event_returns=pd.DataFrame(),
        event_summary=pd.DataFrame(),
        scenario_playbook=pd.DataFrame(),
        current_event_scenarios=pd.DataFrame(),
    )


def _empty_news_monitor() -> NewsMonitorRun:
    return NewsMonitorRun(
        items=(),
        triage=pd.DataFrame(),
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.0,
        lookback_days=0,
    )


def _empty_history_frames() -> OperatingHistoryFrames:
    return OperatingHistoryFrames(
        metrics=pd.DataFrame(),
        components=pd.DataFrame(),
        scenario_drivers=pd.DataFrame(),
        driver_rotation=pd.DataFrame(),
    )


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame.iloc[0]
    return pd.Series(dtype=object)


def _safe_float(value: object) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(output):
        return None
    return output


def _coalesce_float(*values: object) -> float | None:
    for value in values:
        output = _safe_float(value)
        if output is not None:
            return output
    return None
