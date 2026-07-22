from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import (
    BotConfig,
    ExecutionConfig,
    StrategyConfig,
    required_strategy_tickers,
)
from trade_bot.DEFAULTS import (
    DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS,
    DEFAULT_DECISION_TIMELINE_FORWARD_DAYS,
    DEFAULT_DECISION_TIMELINE_MAX_EVENTS,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_RESET_EXPERIMENTS_DIR,
)
from trade_bot.features.indicators import drawdown, unusable_required_price_columns
from trade_bot.portfolio.risk import current_positions
from trade_bot.research.curation import add_research_status
from trade_bot.research.experiments import (
    DecisionSanityConfig,
    ScenarioSizingConfig,
    apply_decision_sanity_overlay,
    apply_operability_hysteresis,
    apply_scenario_position_sizing,
)
from trade_bot.research.future_state_ml import (
    FutureStateModelConfig,
    StrategyDrawdownModelConfig,
    apply_future_state_position_sizing,
    apply_strategy_drawdown_position_sizing,
)
from trade_bot.research.strategy_naming import strategy_display_name
from trade_bot.strategies.momentum import build_strategy_weights


def build_approach_catalog(
    config: BotConfig,
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, strategy in config.strategies.items():
        rows.append(
            {
                "approach_id": f"baseline::{name}",
                "display_name": strategy_display_name(name, family="baseline", phase="baseline"),
                "label": f"baseline | {strategy_display_name(name, family='baseline', phase='baseline')}",
                "source": "baseline",
                "iteration": pd.NA,
                "strategy": name,
                "phase": "configured",
                "family": "baseline",
                "role": "configured_strategy",
                "parent": "",
                "promotion_decision": "configured",
                "promotion_score": pd.NA,
                "hypothesis": "Configured baseline strategy currently included in the main run.",
                "strategy_json": json.dumps(strategy.model_dump(mode="json"), sort_keys=True),
                "scenario_sizing": "",
                "scenario_sizing_json": "",
                "future_state_model": "",
                "future_state_model_json": "",
                "strategy_drawdown_model": "",
                "strategy_drawdown_model_json": "",
                "decision_sanity": "",
                "decision_sanity_json": "",
                "research_status": "reference",
                "prune_reason": "configured_baseline",
            }
        )

    experiment_candidates = load_experiment_candidates(_active_experiment_root(experiment_root))
    if not experiment_candidates.empty:
        rows.extend(experiment_candidates.to_dict(orient="records"))

    return pd.DataFrame(rows)


def _active_experiment_root(root: str | Path) -> Path:
    requested = Path(root)
    if requested == DEFAULT_EXPERIMENTS_DIR:
        return DEFAULT_RESET_EXPERIMENTS_DIR
    return requested


def load_experiment_candidates(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    experiment_root = _active_experiment_root(root)
    if not experiment_root.exists():
        return pd.DataFrame()

    candidate_frames = []
    for manifest_path in sorted(experiment_root.glob("iteration_*/candidates.csv")):
        frame = pd.read_csv(manifest_path)
        iteration = _iteration_from_path(manifest_path)
        frame.insert(0, "iteration", iteration)
        candidate_frames.append(frame)
    if not candidate_frames:
        return pd.DataFrame()

    candidates = pd.concat(candidate_frames, ignore_index=True)
    scorecards = _load_experiment_scorecards(experiment_root)
    if not scorecards.empty:
        scorecard_columns = [
            "iteration",
            "strategy",
            "display_name",
            "promotion_decision",
            "promotion_score",
            "cagr",
            "sharpe",
            "max_drawdown",
            "calmar",
            "excess_cagr_vs_spy",
            "excess_cagr_vs_qqq",
            "drawdown_improvement_vs_spy",
            "drawdown_improvement_vs_qqq",
            "worst_1y_cagr",
            "worst_3y_cagr",
            "positive_1y_window_rate",
            "future_state_model",
            "strategy_drawdown_model",
            "decision_sanity",
            "research_status",
            "prune_reason",
        ]
        available_columns = [column for column in scorecard_columns if column in scorecards]
        candidates = candidates.merge(
            scorecards[available_columns],
            on=["iteration", "strategy"],
            how="left",
            suffixes=("", "_scorecard"),
        )

    if "display_name" not in candidates:
        candidates["display_name"] = ""
    missing_names = candidates["display_name"].isna() | (
        candidates["display_name"].astype(str).str.len() == 0
    )
    if missing_names.any():
        candidates.loc[missing_names, "display_name"] = candidates.loc[missing_names].apply(
            lambda row: strategy_display_name(
                str(row.get("strategy", "")),
                family=str(row.get("family", "")),
                phase=str(row.get("phase", "")),
            ),
            axis=1,
        )

    candidates["source"] = "experiment"
    candidates["approach_id"] = candidates.apply(
        lambda row: f"experiment::{int(row['iteration']):02d}::{row['strategy']}",
        axis=1,
    )
    candidates["label"] = candidates.apply(
        lambda row: (
            f"experiment {int(row['iteration']):02d} | {row['display_name']} "
            f"| {row.get('promotion_decision', 'unscored')}"
        ),
        axis=1,
    )
    for column, default in {
        "phase": "unknown",
        "family": "unknown",
        "role": "unknown",
        "parent": "",
        "hypothesis": "",
        "promotion_decision": "unscored",
        "research_status": "research_archive",
        "prune_reason": "unclassified",
    }.items():
        if column not in candidates:
            candidates[column] = default
    candidates = add_research_status(candidates)
    return candidates


def strategy_from_catalog_row(row: pd.Series) -> StrategyConfig:
    raw = row.get("strategy_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Approach row does not contain a strategy_json payload.")
    return StrategyConfig.model_validate(json.loads(raw))


def decision_sanity_from_catalog_row(row: pd.Series) -> DecisionSanityConfig | None:
    raw = row.get("decision_sanity_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return DecisionSanityConfig(**values)
    except TypeError:
        return None


def scenario_sizing_from_catalog_row(row: pd.Series) -> ScenarioSizingConfig | None:
    raw = row.get("scenario_sizing_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return ScenarioSizingConfig(**values)
    except TypeError:
        return None


def future_state_model_from_catalog_row(row: pd.Series) -> FutureStateModelConfig | None:
    raw = row.get("future_state_model_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return FutureStateModelConfig(**values)
    except TypeError:
        return None


def strategy_drawdown_model_from_catalog_row(
    row: pd.Series,
) -> StrategyDrawdownModelConfig | None:
    raw = row.get("strategy_drawdown_model_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return StrategyDrawdownModelConfig(**values)
    except TypeError:
        return None


def execution_for_catalog_row(
    row: pd.Series, default_execution: ExecutionConfig
) -> ExecutionConfig:
    if str(row.get("phase", "")) == "active_trading":
        return ExecutionConfig(
            initial_capital=default_execution.initial_capital,
            transaction_cost_bps=10.0,
            rebalance="D",
            signal_lag_days=default_execution.signal_lag_days,
        )
    return default_execution


def build_approach_explanation(
    strategy: StrategyConfig,
    row: pd.Series,
    config: BotConfig,
    *,
    execution: ExecutionConfig | None = None,
    scenario_sizing: ScenarioSizingConfig | None = None,
    future_state_model: FutureStateModelConfig | None = None,
    strategy_drawdown_model: StrategyDrawdownModelConfig | None = None,
    decision_sanity: DecisionSanityConfig | None = None,
) -> list[str]:
    execution = execution or config.execution
    family = str(row.get("family", "unknown")).replace("_", " ")
    role = str(row.get("role", "unknown")).replace("_", " ")
    decision = str(row.get("promotion_decision", "unscored")).replace("_", " ")
    defensive = strategy.defensive_ticker or "cash/no explicit defensive asset"
    paragraphs = [
        (
            f"This is a {strategy.type.replace('_', ' ')} approach in the {family} category. "
            f"Its research role is {role}, and the current research decision is {decision}. "
            f"The display below uses {execution.rebalance} rebalance checks, "
            f"a {execution.signal_lag_days}-session execution lag, and "
            f"{execution.transaction_cost_bps:.1f} bps turnover cost assumptions."
        )
    ]

    if strategy.type == "buy_hold":
        paragraphs.append(
            "It does not rank or time assets. It simply holds the configured assets and rebalances "
            "back to equal weights on the execution cadence."
        )
    elif strategy.type == "fixed_allocation":
        allocations = strategy.allocation_weights or {}
        allocation_text = ", ".join(
            f"{ticker} {weight:.0%}" for ticker, weight in sorted(allocations.items())
        )
        paragraphs.append(
            f"It is a static allocation policy: {allocation_text}. Position changes should mostly "
            "come from rebalancing drift, not signal changes."
        )
    elif strategy.type == "absolute_momentum":
        paragraphs.append(
            f"It checks whether each asset is above its own {strategy.moving_average_days}-day "
            f"moving average. Assets above trend are held; if none qualify, capital moves to {defensive}."
        )
    elif strategy.type == "ai_risk_cycle_overlay":
        satellite_text = ", ".join(strategy.satellite_tickers)
        paragraphs.append(
            f"It runs a diversified momentum/off-ramp core, then layers an AI satellite ({satellite_text}) "
            f"with a maximum budget of {strategy.cycle_satellite_max_weight:.0%}. AI exposure can be "
            "earned two ways: normal risk-on momentum, or post-drawdown reentry when discount, "
            "repair, volatility, credit, and breadth gates confirm."
        )
        paragraphs.append(
            f"To avoid twitchy trading, target changes below {strategy.cycle_min_rebalance_change:.0%} "
            f"are ignored and any one-step target move is capped at {strategy.cycle_max_step_change:.0%}."
        )
    else:
        paragraphs.append(
            f"Each rebalance, it computes {strategy.lookback_days}-day momentum after skipping the "
            f"most recent {strategy.skip_days} trading day(s), ranks the universe by "
            f"{strategy.ranking_metric.replace('_', ' ')}, and keeps the top {strategy.top_n}. "
            f"Survivors are sized with {strategy.weighting.replace('_', ' ')} weighting."
        )
        filters: list[str] = []
        if strategy.type == "dual_momentum":
            filters.append(f"a {strategy.min_return:.2%} absolute-return hurdle")
        if strategy.trend_filter_days:
            filters.append(f"a {strategy.trend_filter_days}-day trend confirmation filter")
        if strategy.max_asset_weight:
            filters.append(f"a {strategy.max_asset_weight:.0%} single-asset cap")
        if filters:
            paragraphs.append(
                "It then applies "
                + ", ".join(filters)
                + f". Rejected or residual capital goes to {defensive}."
            )
        elif strategy.defensive_ticker:
            paragraphs.append(f"If no asset qualifies, the strategy moves to {defensive}.")

    if scenario_sizing is not None:
        paragraphs.append(
            "Scenario sizing is active. After the base strategy chooses holdings, the scenario layer "
            f"scales risk exposure using the {scenario_sizing.profile} profile, with risk multipliers "
            f"bounded from {scenario_sizing.min_multiplier:.0%} to {scenario_sizing.max_multiplier:.0%}. "
            f"The removed risk budget is routed to {defensive}."
        )
    if decision_sanity is not None:
        paragraphs.append(
            "Decision-sanity sizing is active. It caps additional defensive allocation unless at least "
            f"{decision_sanity.required_confirmation_breaks} of credit, volatility/liquidity, breadth, "
            "or trend confirm deterioration, or left-tail pressure is already severe. The current "
            f"event/news-only defensive-add cap is {decision_sanity.max_defensive_add:.0%}."
        )
    if strategy.volatility_target:
        paragraphs.append(
            f"A volatility throttle targets {strategy.volatility_target.annualized_volatility:.0%} "
            f"annualized volatility using a lagged {strategy.volatility_target.lookback_days}-day realized-volatility estimate."
        )
    if strategy.drawdown_control:
        paragraphs.append(
            f"A drawdown control cuts exposure to {strategy.drawdown_control.risk_multiplier:.0%} "
            f"after a {strategy.drawdown_control.max_drawdown:.0%} rolling strategy drawdown trigger."
        )
    if future_state_model is not None:
        if future_state_model.model.startswith("bayesian"):
            model_detail = (
                "It adds a Bayesian future-state probability overlay. The model estimates posterior "
                "regime probabilities with explicit priors, recency-weighted evidence, and shrinkage "
                "so sparse risk-off or transition samples do not dominate sizing."
            )
        else:
            model_detail = "It adds a supervised future-state probability overlay."
        paragraphs.append(
            f"{model_detail} The model predicts regime buckets over "
            f"{future_state_model.horizon_days} trading days using the "
            f"{future_state_model.feature_set} feature set and `{future_state_model.model}` method; "
            "those probabilities resize risk exposure and route unused budget to the defensive sleeve."
        )
    if strategy_drawdown_model is not None:
        paragraphs.append(
            "It also has a strategy-specific ML drawdown guard. This model labels the strategy's "
            f"own forward {strategy_drawdown_model.horizon_days}-trading-day drawdown risk, not an "
            "index forecast. If the predicted drawdown probability clears the activation threshold, "
            f"risk exposure is multiplied toward {strategy_drawdown_model.stress_multiplier:.0%} "
            f"with a floor of {strategy_drawdown_model.min_multiplier:.0%}; unused budget is routed "
            f"to {defensive}."
        )
    parent = str(row.get("parent", "") or "")
    if parent and parent != "nan":
        paragraphs.append(
            f"This candidate is an evolution of {parent}; the parent link tells us what prior idea it modified."
        )
    return paragraphs


def build_approach_backtest_result(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
    execution: ExecutionConfig,
    *,
    scenario_sizing: ScenarioSizingConfig | None = None,
    future_state_model: FutureStateModelConfig | None = None,
    strategy_drawdown_model: StrategyDrawdownModelConfig | None = None,
    decision_sanity: DecisionSanityConfig | None = None,
    name: str = "approach",
) -> tuple[BacktestResult | None, list[str]]:
    strategy_prices, strategy_for_prices, missing_columns = _prepare_strategy_prices(
        prices, strategy
    )
    if strategy_prices.empty or not strategy_for_prices.tickers:
        return None, missing_columns

    base_target_weights = build_strategy_weights(strategy_prices, strategy_for_prices)
    target_weights = base_target_weights
    if future_state_model is not None:
        target_weights = apply_future_state_position_sizing(
            target_weights,
            prices,
            future_state_model,
            defensive_ticker=strategy_for_prices.defensive_ticker,
        )
    if scenario_sizing is not None:
        target_weights = apply_scenario_position_sizing(
            target_weights,
            strategy_prices,
            scenario_sizing,
            defensive_ticker=strategy_for_prices.defensive_ticker,
        )
    if strategy_drawdown_model is not None:
        target_weights = apply_strategy_drawdown_position_sizing(
            target_weights,
            prices,
            strategy_drawdown_model,
            defensive_ticker=strategy_for_prices.defensive_ticker,
        )
    if decision_sanity is not None:
        target_weights = apply_decision_sanity_overlay(
            base_target_weights,
            target_weights,
            strategy_prices,
            decision_sanity,
            defensive_ticker=strategy_for_prices.defensive_ticker,
        )
    target_weights = apply_operability_hysteresis(target_weights, strategy_for_prices)
    return (
        run_backtest(
            name,
            strategy_prices,
            target_weights,
            execution,
            volatility_target=strategy_for_prices.volatility_target,
            drawdown_control=strategy_for_prices.drawdown_control,
        ),
        missing_columns,
    )


def build_latest_weight_frame(
    weights: pd.DataFrame,
    *,
    missing_columns: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    if weights.empty:
        frame = pd.DataFrame(
            [{"ticker": "none", "weight": 0.0, "note": "No current active position."}]
        )
    else:
        positions = current_positions(weights, top_n=top_n)
        frame = pd.DataFrame(
            [
                {"ticker": ticker, "weight": float(weight), "note": ""}
                for ticker, weight in positions.items()
            ]
        )
        if frame.empty:
            frame = pd.DataFrame(
                [{"ticker": "none", "weight": 0.0, "note": "No current active position."}]
            )
    for missing_column in missing_columns or []:
        frame.loc[len(frame)] = {
            "ticker": missing_column,
            "weight": 0.0,
            "note": "Ticker not available in loaded prices.",
        }
    return frame


def build_approach_weight_history(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str | None = None,
    lookback_days: int | None = 252,
    max_assets: int = 8,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = _window_weights(weights, lookback_days=lookback_days, start=start, end=end)
    if history.empty:
        return pd.DataFrame()
    selected_columns = _important_weight_columns(
        history,
        defensive_ticker=defensive_ticker,
        max_assets=max_assets,
    )
    visible = (
        history[selected_columns].copy() if selected_columns else pd.DataFrame(index=history.index)
    )
    hidden_columns = [column for column in history.columns if column not in selected_columns]
    other_weight = pd.Series(0.0, index=history.index)
    if hidden_columns:
        other_weight = other_weight.add(history[hidden_columns].sum(axis=1), fill_value=0.0)
    cash_residual = (1.0 - history.sum(axis=1)).clip(lower=0.0)
    other_weight = other_weight.add(cash_residual, fill_value=0.0)
    if float(other_weight.max()) > 0.005:
        visible["other_or_cash"] = other_weight
    visible.index.name = "date"
    return visible


def build_approach_exposure_history(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str | None = None,
    lookback_days: int | None = 252,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = _window_weights(weights, lookback_days=lookback_days, start=start, end=end)
    if history.empty:
        return pd.DataFrame()
    defensive_weight = (
        history[defensive_ticker] if defensive_ticker and defensive_ticker in history else 0.0
    )
    defensive_series = pd.Series(defensive_weight, index=history.index, dtype=float)
    invested_weight = history.sum(axis=1)
    risk_weight = (invested_weight - defensive_series).clip(lower=0.0)
    frame = pd.DataFrame(
        {
            "risk_assets": risk_weight,
            "defensive": defensive_series,
            "cash_or_unallocated": (1.0 - invested_weight).clip(lower=0.0),
        },
        index=history.index,
    )
    frame.index.name = "date"
    return frame


def build_approach_position_summary(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str | None = None,
    lookback_days: int | None = 252,
    material_change: float = 0.05,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = _window_weights(weights, lookback_days=lookback_days, start=start, end=end)
    if history.empty:
        return pd.DataFrame()
    turnover = history.diff().abs().sum(axis=1).fillna(history.abs().sum(axis=1))
    material_turnover = turnover[turnover >= material_change]
    defensive_weight = (
        history[defensive_ticker] if defensive_ticker and defensive_ticker in history else 0.0
    )
    defensive_series = pd.Series(defensive_weight, index=history.index, dtype=float)
    risk_weight = (history.sum(axis=1) - defensive_series).clip(lower=0.0)
    latest = history.iloc[-1]
    active_positions = int((latest > 0.005).sum())
    median_days = _median_days_between(material_turnover.index)
    return pd.DataFrame(
        [
            {
                "metric": "Current risk exposure",
                "value": f"{risk_weight.iloc[-1]:.1%}",
                "interpretation": "Weight currently assigned to non-defensive holdings.",
            },
            {
                "metric": "Current defensive/cash exposure",
                "value": f"{(1.0 - risk_weight.iloc[-1]):.1%}",
                "interpretation": "Weight currently parked in the defensive asset or unallocated cash.",
            },
            {
                "metric": "Average risk exposure",
                "value": f"{risk_weight.mean():.1%}",
                "interpretation": f"Average non-defensive exposure over the selected {len(history):,} sessions.",
            },
            {
                "metric": "Material change days",
                "value": f"{len(material_turnover):,}",
                "interpretation": f"Days with at least {material_change:.0%} one-way allocation change.",
            },
            {
                "metric": "Median days between material changes",
                "value": "n/a" if median_days is None else f"{median_days:.0f}",
                "interpretation": "Lower values imply more frequent human review or trading.",
            },
            {
                "metric": "Current active positions",
                "value": f"{active_positions:,}",
                "interpretation": "Number of tickers with more than 0.5% current weight.",
            },
        ]
    )


def build_approach_change_log(
    weights: pd.DataFrame,
    *,
    defensive_ticker: str | None = None,
    lookback_days: int | None = 252,
    material_change: float = 0.05,
    max_rows: int = 30,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = _window_weights(weights, lookback_days=lookback_days, start=start, end=end)
    if history.empty:
        return pd.DataFrame()
    previous = history.shift(1).fillna(0.0)
    deltas = history - previous
    turnover = deltas.abs().sum(axis=1)
    rows = []
    for date, total_change in turnover[turnover >= material_change].tail(max_rows).items():
        delta = deltas.loc[date].sort_values(ascending=False)
        current = history.loc[date]
        defensive_weight = float(current.get(defensive_ticker, 0.0)) if defensive_ticker else 0.0
        rows.append(
            {
                "date": date.date().isoformat() if hasattr(date, "date") else str(date),
                "total_change": float(total_change),
                "risk_weight": max(float(current.sum() - defensive_weight), 0.0),
                "defensive_weight": defensive_weight,
                "top_adds": _format_delta_vector(delta[delta > 0.005]),
                "top_reductions": _format_delta_vector(
                    (-delta[delta < -0.005]).sort_values(ascending=False)
                ),
                "position_after": _format_weight_vector(current),
            }
        )
    return pd.DataFrame(rows)


def build_approach_decision_events(
    result: BacktestResult,
    *,
    defensive_ticker: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    context_days: int = DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS,
    forward_days: int = DEFAULT_DECISION_TIMELINE_FORWARD_DAYS,
    material_change: float = 0.05,
    max_events: int = DEFAULT_DECISION_TIMELINE_MAX_EVENTS,
) -> pd.DataFrame:
    """Return repeated material allocation decisions for chart/table inspection.

    This intentionally differs from ``build_approach_allocation_transition_events``:
    transition events are single landmarks, while decision events are the largest
    recurring allocation moves that a human reviewer would have had to act on.
    Historical rows usually do not persist the exact internal signal that moved a
    target weight, so the driver text is inferred from the reconstructed weights.
    """
    weights = result.weights.sort_index().copy().fillna(0.0)
    equity = result.equity.sort_index().dropna()
    if weights.empty or equity.empty:
        return pd.DataFrame()

    common_index = weights.index.intersection(equity.index)
    if len(common_index) < 3:
        return pd.DataFrame()
    weights = weights.reindex(common_index).fillna(0.0)
    equity = equity.reindex(common_index).dropna()
    weights = weights.reindex(equity.index).fillna(0.0)

    exposure_history = build_approach_exposure_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
    )
    risk_weight = exposure_history["risk_assets"]
    defensive_weight = exposure_history["defensive"]
    normalized = equity / equity.iloc[0]
    strategy_drawdown = drawdown(normalized)

    window_weights = _window_weights(weights, lookback_days=None, start=start, end=end)
    if window_weights.empty:
        return pd.DataFrame()
    window_index = window_weights.index
    deltas = weights.diff().reindex(window_index).fillna(0.0)
    turnover = deltas.abs().sum(axis=1)
    risk_delta = risk_weight.diff().reindex(window_index).fillna(0.0)
    defensive_delta = defensive_weight.diff().reindex(window_index).fillna(0.0)
    material_mask = (
        (turnover >= material_change)
        | (risk_delta.abs() >= material_change)
        | (defensive_delta.abs() >= material_change)
    )
    candidate_dates = turnover[material_mask].sort_values(ascending=False).head(max_events).index
    if len(candidate_dates) == 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for event_date in sorted(candidate_dates):
        current = weights.loc[event_date]
        delta = deltas.loc[event_date]
        risk_change = float(risk_delta.loc[event_date])
        defensive_change = float(defensive_delta.loc[event_date])
        total_change = float(turnover.loc[event_date])
        event = _decision_event_type(
            risk_change=risk_change,
            defensive_change=defensive_change,
            material_change=material_change,
        )
        risk_position = risk_weight.index.get_loc(event_date)
        if not isinstance(risk_position, int):
            risk_position = int(
                risk_position.start if isinstance(risk_position, slice) else risk_position[0]
            )
        before = risk_weight.iloc[max(0, risk_position - context_days) : risk_position]
        after = risk_weight.iloc[risk_position + 1 : risk_position + 1 + context_days]
        forward = equity.iloc[risk_position : risk_position + 1 + forward_days]
        forward_return = (
            float(forward.iloc[-1] / forward.iloc[0] - 1.0) if len(forward) >= 2 else float("nan")
        )
        short_forward = equity.iloc[risk_position : risk_position + 1 + context_days]
        short_forward_return = (
            float(short_forward.iloc[-1] / short_forward.iloc[0] - 1.0)
            if len(short_forward) >= 2
            else float("nan")
        )
        top_adds = _format_delta_vector(delta[delta > 0.005])
        top_reductions = _format_reduction_vector((-delta[delta < -0.005]).sort_values())
        rows.append(
            {
                "event": event,
                "date": (
                    event_date.date().isoformat()
                    if hasattr(event_date, "date")
                    else str(event_date)
                ),
                "signal": (
                    f"{event}: risk {risk_change:+.1%}, defensive {defensive_change:+.1%}, "
                    f"total move {total_change:.1%}."
                ),
                "inferred_driver": _decision_event_driver(
                    event=event,
                    risk_change=risk_change,
                    defensive_change=defensive_change,
                    top_adds=top_adds,
                    top_reductions=top_reductions,
                ),
                "total_change": total_change,
                "risk_weight_change": risk_change,
                "defensive_weight_change": defensive_change,
                "risk_weight_before_1m": (
                    float(before.mean())
                    if not before.empty
                    else float(risk_weight.iloc[risk_position])
                ),
                "risk_weight_at_event": float(risk_weight.iloc[risk_position]),
                "risk_weight_after_1m": (
                    float(after.mean())
                    if not after.empty
                    else float(risk_weight.iloc[risk_position])
                ),
                "defensive_weight_at_event": float(defensive_weight.iloc[risk_position]),
                "drawdown_at_event": float(strategy_drawdown.iloc[risk_position]),
                "forward_return_1m": short_forward_return,
                "forward_return_3m": forward_return,
                "top_adds": top_adds,
                "top_reductions": top_reductions,
                "position_after": _format_weight_vector(current),
            }
        )
    return pd.DataFrame(rows)


def _decision_event_type(
    *,
    risk_change: float,
    defensive_change: float,
    material_change: float,
) -> str:
    if risk_change <= -material_change:
        return "De-risking move"
    if risk_change >= material_change:
        return "Re-risking move"
    if defensive_change >= material_change:
        return "Defensive add"
    if defensive_change <= -material_change:
        return "Defensive reduce"
    return "Risk rotation"


def _decision_event_driver(
    *,
    event: str,
    risk_change: float,
    defensive_change: float,
    top_adds: str,
    top_reductions: str,
) -> str:
    if event == "De-risking move":
        return (
            "Inferred off-ramp: non-defensive exposure fell and capital moved toward "
            f"defensive/cash sleeves. Adds: {top_adds}; reductions: {top_reductions}."
        )
    if event == "Re-risking move":
        return (
            "Inferred re-entry: non-defensive exposure rose, usually after repair, trend, "
            f"or risk-budget confirmation. Adds: {top_adds}; reductions: {top_reductions}."
        )
    if event == "Risk rotation":
        return (
            "Inferred rotation: total risk stayed broadly similar, but leadership changed "
            f"inside the risk sleeve. Adds: {top_adds}; reductions: {top_reductions}."
        )
    if defensive_change > 0:
        return (
            "Defensive sleeve increased without a large aggregate risk-weight change. "
            f"Adds: {top_adds}; reductions: {top_reductions}."
        )
    return (
        f"Defensive sleeve decreased by {abs(defensive_change):.1%} while risk changed "
        f"{risk_change:+.1%}. Adds: {top_adds}; reductions: {top_reductions}."
    )


def build_approach_holding_stats(
    weights: pd.DataFrame,
    *,
    lookback_days: int | None = 252,
    max_assets: int = 20,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = _window_weights(weights, lookback_days=lookback_days, start=start, end=end)
    if history.empty:
        return pd.DataFrame()
    rows = []
    for ticker in history.columns:
        series = history[ticker]
        if float(series.max()) <= 0.005:
            continue
        rows.append(
            {
                "ticker": ticker,
                "current_weight": float(series.iloc[-1]),
                "average_weight": float(series.mean()),
                "max_weight": float(series.max()),
                "active_day_rate": float((series > 0.005).mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["current_weight", "average_weight", "max_weight"], ascending=False)
        .head(max_assets)
    )


def build_approach_allocation_transition_events(
    result: BacktestResult,
    *,
    defensive_ticker: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    context_days: int = 21,
    forward_days: int = 63,
    material_change: float = 0.05,
) -> pd.DataFrame:
    weights = _window_weights(result.weights, lookback_days=None, start=start, end=end)
    equity = _window_series(result.equity, start=start, end=end)
    if weights.empty or equity.empty:
        return pd.DataFrame()

    common_index = weights.index.intersection(equity.index)
    if len(common_index) < 3:
        return pd.DataFrame()
    weights = weights.reindex(common_index).fillna(0.0)
    equity = equity.reindex(common_index).dropna()
    weights = weights.reindex(equity.index).fillna(0.0)
    risk_weight = build_approach_exposure_history(
        weights,
        defensive_ticker=defensive_ticker,
        lookback_days=None,
    )["risk_assets"]
    normalized = equity / equity.iloc[0]
    window_drawdown = drawdown(normalized)
    risk_delta = risk_weight.diff().fillna(0.0)

    rows = []
    rows.append(
        _allocation_event_row(
            event="Worst drawdown point",
            signal="How exposed was the strategy at the deepest selected-window drawdown?",
            date=window_drawdown.idxmin(),
            event_value=float(window_drawdown.min()),
            event_value_label="drawdown",
            equity=normalized,
            window_drawdown=window_drawdown,
            risk_weight=risk_weight,
            context_days=context_days,
            forward_days=forward_days,
        )
    )

    biggest_derisk = risk_delta.idxmin()
    if float(risk_delta.loc[biggest_derisk]) <= -material_change:
        rows.append(
            _allocation_event_row(
                event="Largest de-risking move",
                signal="The biggest one-day reduction in non-defensive exposure in the window.",
                date=biggest_derisk,
                event_value=float(risk_delta.loc[biggest_derisk]),
                event_value_label="risk_weight_change",
                equity=normalized,
                window_drawdown=window_drawdown,
                risk_weight=risk_weight,
                context_days=context_days,
                forward_days=forward_days,
            )
        )

    biggest_rerisk = risk_delta.idxmax()
    if float(risk_delta.loc[biggest_rerisk]) >= material_change:
        rows.append(
            _allocation_event_row(
                event="Largest re-risking move",
                signal="The biggest one-day increase in non-defensive exposure in the window.",
                date=biggest_rerisk,
                event_value=float(risk_delta.loc[biggest_rerisk]),
                event_value_label="risk_weight_change",
                equity=normalized,
                window_drawdown=window_drawdown,
                risk_weight=risk_weight,
                context_days=context_days,
                forward_days=forward_days,
            )
        )

    return pd.DataFrame(rows)


def build_approach_mechanics(
    strategy: StrategyConfig,
    config: BotConfig,
    *,
    execution: ExecutionConfig | None = None,
) -> pd.DataFrame:
    execution = execution or config.execution
    rows = [
        _mechanic("Strategy type", strategy.type, _strategy_type_explanation(strategy)),
        _mechanic(
            "Tradable universe",
            f"{len(strategy.tickers)} assets",
            ", ".join(strategy.tickers),
        ),
        _mechanic(
            "Decision cadence",
            execution.rebalance,
            "Signals are converted to target weights on this rebalance cadence.",
        ),
        _mechanic(
            "Execution lag",
            f"{execution.signal_lag_days} session(s)",
            "Backtests assume trades happen after signals are known, not at the same close.",
        ),
        _mechanic(
            "Transaction cost",
            f"{execution.transaction_cost_bps:.1f} bps turnover cost",
            "Every weight change pays this cost in the backtest.",
        ),
    ]
    if strategy.type == "absolute_momentum":
        rows.append(
            _mechanic(
                "Trend filter",
                f"{strategy.moving_average_days}-day moving average",
                "Risk assets are held only when price is above its moving average.",
            )
        )
    if strategy.type in {"relative_momentum", "dual_momentum", "ai_risk_cycle_overlay"}:
        rows.extend(
            [
                _mechanic(
                    "Momentum lookback",
                    f"{strategy.lookback_days} trading days",
                    "Historical return window used to compare candidate assets.",
                ),
                _mechanic(
                    "Skip window",
                    f"{strategy.skip_days} trading days",
                    "Recent days excluded from the momentum calculation to reduce short-term reversal noise.",
                ),
                _mechanic(
                    "Number selected",
                    f"Top {strategy.top_n}",
                    "Only the highest-ranked assets are eligible for risk exposure.",
                ),
                _mechanic(
                    "Ranking metric",
                    strategy.ranking_metric,
                    _ranking_explanation(strategy.ranking_metric),
                ),
                _mechanic(
                    "Weighting method",
                    strategy.weighting,
                    _weighting_explanation(strategy.weighting),
                ),
                _mechanic(
                    "Volatility lookback",
                    f"{strategy.volatility_lookback_days} trading days",
                    "Used when ranking or sizing depends on realized volatility.",
                ),
            ]
        )
    if strategy.type == "dual_momentum":
        rows.append(
            _mechanic(
                "Absolute return hurdle",
                f"{strategy.min_return:.2%}",
                "Selected assets must also clear this return threshold or capital can move to the defensive asset.",
            )
        )
    if strategy.trend_filter_days:
        rows.append(
            _mechanic(
                "Selection trend confirmation",
                f"{strategy.trend_filter_days}-day moving average",
                "Selected risk assets must remain above this trend filter.",
            )
        )
    if strategy.max_asset_weight:
        rows.append(
            _mechanic(
                "Single-asset cap",
                f"{strategy.max_asset_weight:.0%}",
                "Any excess weight is moved to the defensive asset when one is configured.",
            )
        )
    rows.append(
        _mechanic(
            "Defensive asset",
            strategy.defensive_ticker or "none",
            "Destination for capital when the strategy has no eligible risk signal or capped residual weight.",
        )
    )
    if strategy.volatility_target:
        rows.append(
            _mechanic(
                "Volatility target",
                f"{strategy.volatility_target.annualized_volatility:.0%} annualized",
                (
                    f"Exposure is scaled from lagged {strategy.volatility_target.lookback_days}-day "
                    f"realized volatility and capped at {strategy.volatility_target.max_leverage:.1f}x."
                ),
            )
        )
    if strategy.drawdown_control:
        rows.append(
            _mechanic(
                "Drawdown control",
                f"{strategy.drawdown_control.max_drawdown:.0%} trigger",
                (
                    f"Lagged strategy drawdown over {strategy.drawdown_control.equity_lookback_days} "
                    f"days scales risk to {strategy.drawdown_control.risk_multiplier:.0%}."
                ),
            )
        )
    return pd.DataFrame(rows)


def build_approach_steps(strategy: StrategyConfig) -> pd.DataFrame:
    if strategy.type == "buy_hold":
        steps = [
            "Assign equal target weights to the configured ticker list.",
            "Rebalance on the execution cadence.",
            "Do not use trend, macro, scenario, or defensive off-ramp logic inside this strategy.",
        ]
    elif strategy.type == "absolute_momentum":
        steps = [
            f"Compute each asset's {strategy.moving_average_days}-day moving average.",
            "Hold assets whose price is above trend and split capital equally across active assets.",
            f"If no asset is above trend, hold {strategy.defensive_ticker or 'cash/no position'}.",
        ]
    else:
        steps = [
            f"Compute {strategy.lookback_days}-day returns after skipping the most recent {strategy.skip_days} trading days.",
            f"Rank assets by {strategy.ranking_metric}.",
            f"Keep only the top {strategy.top_n} ranked assets.",
        ]
        if strategy.type == "dual_momentum":
            steps.append(
                f"Drop selected assets that fail the {strategy.min_return:.2%} absolute return hurdle."
            )
        if strategy.trend_filter_days:
            steps.append(
                f"Drop selected assets below their {strategy.trend_filter_days}-day moving average."
            )
        steps.extend(
            [
                f"Size surviving assets with {strategy.weighting} weights.",
                "Clip long-only weights so total risk exposure never exceeds 100%.",
            ]
        )
        if strategy.defensive_ticker:
            steps.append(f"Move unallocated or no-signal capital into {strategy.defensive_ticker}.")
    if strategy.volatility_target:
        steps.append("Apply lagged volatility-target scaling after target weights are formed.")
    if strategy.drawdown_control:
        steps.append("Apply lagged drawdown-control scaling after target weights are formed.")
    return pd.DataFrame(
        [{"step": index + 1, "detail": detail} for index, detail in enumerate(steps)]
    )


def build_approach_risk_notes(strategy: StrategyConfig, row: pd.Series) -> pd.DataFrame:
    notes = [
        {
            "topic": "Execution realism",
            "why_it_matters": "The approach is only tradable if signals can be reviewed and executed after they are known.",
            "watch_item": "Signal lag, weekly rebalance cadence, turnover, and stale-price checks.",
        },
        {
            "topic": "Regime transition risk",
            "why_it_matters": "Momentum and trend systems can be late when leadership changes quickly.",
            "watch_item": "Worst 1-year and 3-year rolling windows, scenario risk, breadth, credit, and news phase.",
        },
    ]
    if strategy.type in {"relative_momentum", "dual_momentum"}:
        notes.append(
            {
                "topic": "Whipsaw",
                "why_it_matters": "Repeated rotations can lose money when markets chop without persistent leadership.",
                "watch_item": "Turnover, false breakouts, and whether top holdings change every rebalance.",
            }
        )
    if "QQQ" in strategy.tickers or any(
        ticker in strategy.tickers for ticker in ["SMH", "SOXX", "NVDA"]
    ):
        notes.append(
            {
                "topic": "AI/concentration dependence",
                "why_it_matters": "Recent outperformance can reflect a narrow historical period that may not persist.",
                "watch_item": "AI-unit-economics events, semis versus broad market, and QQQ/RSP breadth.",
            }
        )
    if strategy.defensive_ticker:
        notes.append(
            {
                "topic": "Defensive destination",
                "why_it_matters": "The off-ramp only helps if the defensive asset behaves as expected when risk sells off.",
                "watch_item": f"{strategy.defensive_ticker} liquidity, yield behavior, and whether duration/credit exposure is intended.",
            }
        )
    if row.get("promotion_decision") in {"reject_left_tail", "reject_regime_fragility"}:
        notes.append(
            {
                "topic": "Experiment warning",
                "why_it_matters": "This approach was rejected by current research triage.",
                "watch_item": str(row.get("promotion_decision")),
            }
        )
    return pd.DataFrame(notes)


def build_latest_approach_weights(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    strategy_prices, strategy_for_prices, missing_columns = _prepare_strategy_prices(
        prices, strategy
    )
    if strategy_prices.empty or not strategy_for_prices.tickers:
        return pd.DataFrame(
            [
                {
                    "ticker": "n/a",
                    "weight": 0.0,
                    "note": (
                        "Missing required price inputs: " + ", ".join(missing_columns)
                        if missing_columns
                        else "No strategy tickers are available in loaded prices."
                    ),
                }
            ]
        )
    weights = build_strategy_weights(strategy_prices, strategy_for_prices)
    positions = current_positions(weights, top_n=20)
    frame = pd.DataFrame(
        [
            {"ticker": ticker, "weight": float(weight), "note": ""}
            for ticker, weight in positions.items()
        ]
    )
    if frame.empty:
        frame = pd.DataFrame(
            [{"ticker": "none", "weight": 0.0, "note": "No current active position."}]
        )
    if missing_columns:
        frame.loc[len(frame)] = {
            "ticker": "missing",
            "weight": 0.0,
            "note": ", ".join(missing_columns),
        }
    return frame


def approach_scorecard_row(row: pd.Series) -> pd.DataFrame:
    metric_columns = [
        "promotion_decision",
        "promotion_score",
        "cagr",
        "sharpe",
        "max_drawdown",
        "calmar",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "worst_1y_cagr",
        "worst_3y_cagr",
        "positive_1y_window_rate",
    ]
    return pd.DataFrame(
        [
            {
                column: row[column]
                for column in metric_columns
                if column in row and pd.notna(row[column])
            }
        ]
    )


def _prepare_strategy_prices(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
) -> tuple[pd.DataFrame, StrategyConfig, list[str]]:
    columns = required_strategy_tickers(strategy)
    missing_columns = unusable_required_price_columns(prices, columns)
    if missing_columns:
        return pd.DataFrame(), strategy, missing_columns
    available_columns = columns
    available_risk_tickers = [ticker for ticker in strategy.tickers if ticker in available_columns]
    if not available_risk_tickers:
        empty_strategy = StrategyConfig.model_validate(
            {
                **strategy.model_dump(mode="json"),
                "tickers": [],
                "defensive_ticker": None,
            }
        )
        return pd.DataFrame(), empty_strategy, missing_columns

    strategy_data = strategy.model_dump(mode="json")
    strategy_data["tickers"] = available_risk_tickers
    if strategy.defensive_ticker not in available_columns:
        strategy_data["defensive_ticker"] = None
    if strategy_data.get("allocation_weights"):
        strategy_data["allocation_weights"] = {
            ticker: weight
            for ticker, weight in strategy_data["allocation_weights"].items()
            if ticker in available_columns
        }
    strategy_for_prices = StrategyConfig.model_validate(strategy_data)
    strategy_prices = prices[available_columns].dropna(how="all")
    return strategy_prices, strategy_for_prices, missing_columns


def _important_weight_columns(
    history: pd.DataFrame,
    *,
    defensive_ticker: str | None,
    max_assets: int,
) -> list[str]:
    if history.empty:
        return []
    stats = pd.DataFrame(
        {
            "latest": history.iloc[-1].abs(),
            "average": history.abs().mean(),
            "maximum": history.abs().max(),
        }
    )
    stats["score"] = stats["latest"] * 3.0 + stats["average"] + stats["maximum"]
    selected = stats.sort_values("score", ascending=False).head(max_assets).index.tolist()
    if (
        defensive_ticker
        and defensive_ticker in history.columns
        and defensive_ticker not in selected
    ):
        selected.append(defensive_ticker)
    return selected


def _format_weight_vector(
    weights: pd.Series, *, min_weight: float = 0.005, max_items: int = 6
) -> str:
    positive = weights[weights > min_weight].sort_values(ascending=False).head(max_items)
    if positive.empty:
        return "none"
    return ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in positive.items())


def _format_delta_vector(delta: pd.Series, *, min_weight: float = 0.005, max_items: int = 4) -> str:
    meaningful = delta[delta > min_weight].sort_values(ascending=False).head(max_items)
    if meaningful.empty:
        return "none"
    return ", ".join(f"{ticker} +{weight:.0%}" for ticker, weight in meaningful.items())


def _format_reduction_vector(
    reductions: pd.Series, *, min_weight: float = 0.005, max_items: int = 4
) -> str:
    meaningful = reductions[reductions > min_weight].sort_values(ascending=False).head(max_items)
    if meaningful.empty:
        return "none"
    return ", ".join(f"{ticker} -{weight:.0%}" for ticker, weight in meaningful.items())


def _median_days_between(index: pd.Index) -> float | None:
    if len(index) < 2:
        return None
    dates = pd.Series(pd.to_datetime(index)).sort_values()
    gaps = dates.diff().dropna().dt.days
    if gaps.empty:
        return None
    return float(gaps.median())


def _load_experiment_scorecards(root: Path) -> pd.DataFrame:
    frames = []
    for scorecard_path in sorted(root.glob("iteration_*/scorecard.csv")):
        frame = pd.read_csv(scorecard_path)
        frame.insert(0, "iteration", _iteration_from_path(scorecard_path))
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _iteration_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def _mechanic(component: str, setting: str, interpretation: str) -> dict[str, str]:
    return {
        "component": component,
        "setting": setting,
        "interpretation": interpretation,
    }


def _strategy_type_explanation(strategy: StrategyConfig) -> str:
    explanations = {
        "buy_hold": "Static long-only exposure to the configured assets.",
        "absolute_momentum": "Trend-following system that exits risk assets when their own trend breaks.",
        "relative_momentum": "Cross-sectional rotation into the strongest assets, without an absolute return hurdle.",
        "dual_momentum": "Cross-sectional rotation plus an absolute momentum hurdle before taking risk.",
        "dip_reentry": "Metered reentry system that buys discounted assets only after repair signals confirm.",
        "dip_reentry_overlay": "Momentum/off-ramp system that lets confirmed dip-reentry signals replace defensive cash.",
        "ai_risk_cycle_overlay": "Diversified off-ramp core with an aggressive AI satellite that can reenter after confirmed repair.",
        "sector_regime_rotation": "Sector/theme rotation system that scores leadership by regime, then meters total risk through credit, breadth, volatility, and discount-repair signals.",
        "fixed_allocation": "Static long-only allocation with explicit target weights.",
    }
    return explanations.get(strategy.type, f"Strategy type {strategy.type}.")


def _ranking_explanation(metric: str) -> str:
    explanations = {
        "return": "Ranks by raw lookback return.",
        "risk_adjusted_return": "Ranks by lookback return divided by realized volatility.",
        "return_trend_quality": "Ranks by return with a trend-quality boost from price versus moving average.",
    }
    return explanations.get(metric, metric)


def _weighting_explanation(weighting: str) -> str:
    explanations = {
        "equal": "Splits capital equally across selected assets.",
        "inverse_volatility": "Allocates more to lower-volatility selected assets.",
        "momentum_score": "Allocates more to stronger positive momentum scores.",
        "risk_adjusted_score": "Allocates more to stronger risk-adjusted scores.",
    }
    return explanations.get(weighting, weighting)


def _window_weights(
    weights: pd.DataFrame,
    *,
    lookback_days: int | None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    history = weights.sort_index().copy().fillna(0.0)
    if start is not None:
        history = history.loc[history.index >= pd.Timestamp(start)]
    if end is not None:
        history = history.loc[history.index <= pd.Timestamp(end)]
    if lookback_days is not None:
        history = history.tail(lookback_days)
    return history


def _window_series(
    series: pd.Series,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    windowed = series.sort_index().dropna()
    if start is not None:
        windowed = windowed.loc[windowed.index >= pd.Timestamp(start)]
    if end is not None:
        windowed = windowed.loc[windowed.index <= pd.Timestamp(end)]
    return windowed


def _allocation_event_row(
    *,
    event: str,
    signal: str,
    date: pd.Timestamp,
    event_value: float,
    event_value_label: str,
    equity: pd.Series,
    window_drawdown: pd.Series,
    risk_weight: pd.Series,
    context_days: int,
    forward_days: int,
) -> dict[str, object]:
    position = risk_weight.index.get_loc(date)
    if not isinstance(position, int):
        position = int(position.start if isinstance(position, slice) else position[0])
    before = risk_weight.iloc[max(0, position - context_days) : position]
    after = risk_weight.iloc[position + 1 : position + 1 + context_days]
    forward = equity.iloc[position : position + 1 + forward_days]
    forward_return = float(forward.iloc[-1] / forward.iloc[0] - 1.0) if len(forward) >= 2 else 0.0
    return {
        "event": event,
        "date": date.date().isoformat() if hasattr(date, "date") else str(date),
        "signal": signal,
        event_value_label: event_value,
        "risk_weight_before_1m": (
            float(before.mean()) if not before.empty else float(risk_weight.iloc[position])
        ),
        "risk_weight_at_event": float(risk_weight.iloc[position]),
        "risk_weight_after_1m": (
            float(after.mean()) if not after.empty else float(risk_weight.iloc[position])
        ),
        "drawdown_at_event": float(window_drawdown.iloc[position]),
        "forward_return_3m": forward_return,
        "interpretation": _allocation_event_interpretation(event, event_value, forward_return),
    }


def _allocation_event_interpretation(event: str, event_value: float, forward_return: float) -> str:
    if event == "Worst drawdown point":
        return (
            "Use this to check whether the strategy was defensive before or only after the damage."
        )
    if event == "Largest de-risking move":
        return (
            f"Risk exposure dropped {abs(event_value):.1%}; next-window return was {forward_return:.1%}. "
            "Good off-ramps reduce exposure before follow-through losses, not after a completed move."
        )
    if event == "Largest re-risking move":
        return (
            f"Risk exposure rose {event_value:.1%}; next-window return was {forward_return:.1%}. "
            "Good re-entry adds risk when repair is tradable rather than waiting for all upside to pass."
        )
    return "Review this event alongside the allocation and drawdown chart."
