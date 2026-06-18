from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig
from trade_bot.DEFAULT import DEFAULT_EXPERIMENTS_DIR
from trade_bot.portfolio.risk import current_positions
from trade_bot.research.experiments import ScenarioSizingConfig, apply_scenario_position_sizing
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
                "label": f"baseline | {name}",
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
            }
        )

    experiment_candidates = load_experiment_candidates(experiment_root)
    if not experiment_candidates.empty:
        rows.extend(experiment_candidates.to_dict(orient="records"))

    return pd.DataFrame(rows)


def load_experiment_candidates(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    experiment_root = Path(root)
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
        available_columns = [column for column in scorecard_columns if column in scorecards]
        candidates = candidates.merge(
            scorecards[available_columns],
            on=["iteration", "strategy"],
            how="left",
            suffixes=("", "_scorecard"),
        )

    candidates["source"] = "experiment"
    candidates["approach_id"] = candidates.apply(
        lambda row: f"experiment::{int(row['iteration']):02d}::{row['strategy']}",
        axis=1,
    )
    candidates["label"] = candidates.apply(
        lambda row: (
            f"experiment {int(row['iteration']):02d} | {row['strategy']} "
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
    }.items():
        if column not in candidates:
            candidates[column] = default
    return candidates


def strategy_from_catalog_row(row: pd.Series) -> StrategyConfig:
    raw = row.get("strategy_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Approach row does not contain a strategy_json payload.")
    return StrategyConfig.model_validate(json.loads(raw))


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
    name: str = "approach",
) -> tuple[BacktestResult | None, list[str]]:
    strategy_prices, strategy_for_prices, missing_columns = _prepare_strategy_prices(
        prices, strategy
    )
    if strategy_prices.empty or not strategy_for_prices.tickers:
        return None, missing_columns

    target_weights = build_strategy_weights(strategy_prices, strategy_for_prices)
    if scenario_sizing is not None:
        target_weights = apply_scenario_position_sizing(
            target_weights,
            strategy_prices,
            scenario_sizing,
            defensive_ticker=strategy_for_prices.defensive_ticker,
        )
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
    lookback_days: int = 252,
    max_assets: int = 8,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = weights.tail(lookback_days).copy().fillna(0.0)
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
    lookback_days: int = 252,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = weights.tail(lookback_days).copy().fillna(0.0)
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
    lookback_days: int = 252,
    material_change: float = 0.05,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = weights.tail(lookback_days).copy().fillna(0.0)
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
                "interpretation": f"Average non-defensive exposure over the last {len(history):,} sessions.",
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
    lookback_days: int = 252,
    material_change: float = 0.05,
    max_rows: int = 30,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = weights.tail(lookback_days).copy().fillna(0.0)
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


def build_approach_holding_stats(
    weights: pd.DataFrame,
    *,
    lookback_days: int = 252,
    max_assets: int = 20,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    history = weights.tail(lookback_days).copy().fillna(0.0)
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
    columns = list(
        dict.fromkeys(
            [*strategy.tickers, *([strategy.defensive_ticker] if strategy.defensive_ticker else [])]
        )
    )
    available_columns = [column for column in columns if column in prices.columns]
    missing_columns = sorted(set(columns) - set(available_columns))
    available_risk_tickers = [ticker for ticker in strategy.tickers if ticker in available_columns]
    if not available_risk_tickers:
        return pd.DataFrame(
            [
                {
                    "ticker": "n/a",
                    "weight": 0.0,
                    "note": "No strategy tickers are available in loaded prices.",
                }
            ]
        )
    strategy_prices = prices[available_columns].dropna(how="all")
    strategy_for_prices = StrategyConfig.model_validate(
        {
            **strategy.model_dump(mode="json"),
            "tickers": available_risk_tickers,
            "defensive_ticker": (
                strategy.defensive_ticker
                if strategy.defensive_ticker in available_columns
                else None
            ),
        }
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
    columns = list(
        dict.fromkeys(
            [*strategy.tickers, *([strategy.defensive_ticker] if strategy.defensive_ticker else [])]
        )
    )
    available_columns = [column for column in columns if column in prices.columns]
    missing_columns = sorted(set(columns) - set(available_columns))
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
