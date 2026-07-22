from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from trade_bot.backtest.engine import (
    BacktestResult,
    apply_volatility_target,
    run_backtest,
    validate_held_price_availability,
)
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.backtest.windows import (
    calendar_year_metrics,
    rolling_window_metrics,
    summarize_walk_forward,
    summarize_windows,
    walk_forward_holdout_metrics,
)
from trade_bot.config import (
    BotConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    configured_tickers,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import (
    bounded_forward_fill,
    daily_returns,
    moving_average,
    realized_volatility,
    rolling_drawdown,
    unusable_required_price_columns,
)
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.i111_orthogonal_search import (
    CORE_AI_TICKERS,
    DEFAULT_I111_NATIVE_CHALLENGER,
    SIGNAL_TICKERS,
)
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_FRONTIER_SEARCH_OUTPUT_DIR = Path("reports/i111_frontier_search")
DEFAULT_FRONTIER_MAX_ITERATIONS = 250
DEFAULT_CHECKPOINT_SIZE = 20


@dataclass(frozen=True)
class SourceProfile:
    name: str
    updates: dict[str, Any]


@dataclass(frozen=True)
class ConcentrationProfile:
    name: str
    top_n: int
    max_asset_weight: float


@dataclass(frozen=True)
class DynamicGuardProfile:
    name: str
    default_max_drawdown: float
    default_multiplier: float
    healthy_max_drawdown: float
    healthy_multiplier: float
    stress_max_drawdown: float
    stress_multiplier: float
    health_threshold: float
    crash_threshold: float


@dataclass(frozen=True)
class CrashActionProfile:
    name: str
    threshold: float
    risk_weight: float
    use_gradient: bool


@dataclass(frozen=True)
class RouterProfile:
    name: str
    mode: Literal["gated_high", "blend_high", "defensive_select"]
    health_threshold: float
    crash_threshold: float
    defensive_crash_threshold: float
    defensive_risk_weight: float


@dataclass(frozen=True)
class FrontierPolicy:
    health_profile: str = "balanced"
    crash_profile: str = "balanced"
    concentration: ConcentrationProfile | None = None
    router: RouterProfile | None = None
    dynamic_guard: DynamicGuardProfile | None = None
    crash_action: CrashActionProfile | None = None


@dataclass(frozen=True)
class FrontierCandidate:
    iteration: int
    name: str
    primary_topic: str
    topic_stack: tuple[str, ...]
    source_profile: str
    hypothesis: str
    base_strategy: StrategyConfig
    high_strategy: StrategyConfig | None
    policy: FrontierPolicy


@dataclass(frozen=True)
class FrontierSearchResult:
    strategy_metrics: pd.DataFrame
    candidate_roster: pd.DataFrame
    checkpoint_summary: pd.DataFrame
    family_summary: pd.DataFrame
    rolling_windows: pd.DataFrame
    walk_forward: pd.DataFrame
    calendar_years: pd.DataFrame
    summary: str


def run_i111_frontier_search(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_FRONTIER_SEARCH_OUTPUT_DIR,
    max_iterations: int = DEFAULT_FRONTIER_MAX_ITERATIONS,
    checkpoint_size: int = DEFAULT_CHECKPOINT_SIZE,
    refresh_data: bool = False,
) -> FrontierSearchResult:
    baselines, candidates = build_i111_frontier_candidates(
        config,
        max_iterations=max_iterations,
    )
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(tuple(candidate.base_strategy for candidate in candidates))
        | _candidate_tickers(
            tuple(
                candidate.high_strategy
                for candidate in candidates
                if candidate.high_strategy is not None
            )
        )
        | _candidate_tickers(tuple(baselines.values()))
        | set(AI_GROWTH_TICKERS)
        | set(SIGNAL_TICKERS)
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    baseline_results = {
        name: _run_standard_strategy(config, name, strategy, prices)
        for name, strategy in baselines.items()
    }
    baseline_metrics = {name: _metrics(result) for name, result in baseline_results.items()}
    primary_metrics = baseline_metrics.get("baseline_primary")
    native_metrics = baseline_metrics.get("baseline_native_challenger")

    metric_rows: list[dict[str, object]] = []
    results: dict[str, BacktestResult] = dict(baseline_results)
    for result in baseline_results.values():
        metric_rows.append(
            _metric_row(
                result=result,
                iteration=0,
                role="baseline",
                primary_topic="baseline",
                topic_stack=("baseline",),
                source_profile="baseline",
                hypothesis="Reference strategy.",
                primary_metrics=primary_metrics,
                native_metrics=native_metrics,
            )
        )

    for candidate in candidates:
        result = _run_frontier_candidate(config, candidate, prices)
        results[candidate.name] = result
        metric_rows.append(
            _metric_row(
                result=result,
                iteration=candidate.iteration,
                role="frontier_candidate",
                primary_topic=candidate.primary_topic,
                topic_stack=candidate.topic_stack,
                source_profile=candidate.source_profile,
                hypothesis=candidate.hypothesis,
                primary_metrics=primary_metrics,
                native_metrics=native_metrics,
            )
        )

    strategy_metrics = pd.DataFrame(metric_rows)
    rolling = summarize_windows(rolling_window_metrics(results)).reset_index()
    walk_forward = summarize_walk_forward(walk_forward_holdout_metrics(results)).reset_index()
    calendar = calendar_year_metrics(results)
    strategy_metrics = _add_robustness(strategy_metrics, rolling, walk_forward, calendar)
    strategy_metrics = _score(strategy_metrics)
    candidate_roster = _candidate_roster(candidates)
    checkpoint_summary = _checkpoint_summary(
        strategy_metrics,
        checkpoint_size=checkpoint_size,
        native_metrics=native_metrics,
    )
    family_summary = _family_summary(strategy_metrics)
    summary = _summary_markdown(
        strategy_metrics,
        checkpoint_summary,
        family_summary,
        max_iterations=len(candidates),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    candidate_roster.to_csv(output_path / "candidate_roster.csv", index=False)
    checkpoint_summary.to_csv(output_path / "checkpoint_summary.csv", index=False)
    family_summary.to_csv(output_path / "family_summary.csv", index=False)
    rolling.to_csv(output_path / "rolling_windows.csv", index=False)
    walk_forward.to_csv(output_path / "walk_forward.csv", index=False)
    calendar.to_csv(output_path / "calendar_years.csv", index=False)
    (output_path / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output_path,
        study="i111_frontier_search",
        config=config,
        prices=prices,
        parameters={
            "max_iterations": max_iterations,
            "checkpoint_size": checkpoint_size,
            "refresh_data": refresh_data,
            "candidate_count": len(candidates),
        },
        artifacts=[
            "strategy_metrics.csv",
            "candidate_roster.csv",
            "checkpoint_summary.csv",
            "family_summary.csv",
            "rolling_windows.csv",
            "walk_forward.csv",
            "calendar_years.csv",
            "summary.md",
        ],
    )
    return FrontierSearchResult(
        strategy_metrics=strategy_metrics,
        candidate_roster=candidate_roster,
        checkpoint_summary=checkpoint_summary,
        family_summary=family_summary,
        rolling_windows=rolling,
        walk_forward=walk_forward,
        calendar_years=calendar,
        summary=summary,
    )


def build_i111_frontier_candidates(
    config: BotConfig,
    *,
    max_iterations: int = DEFAULT_FRONTIER_MAX_ITERATIONS,
) -> tuple[dict[str, StrategyConfig], tuple[FrontierCandidate, ...]]:
    primary = config.strategies[config.primary_strategy]
    baselines = {"baseline_primary": primary}
    if DEFAULT_I111_NATIVE_CHALLENGER in config.strategies:
        baselines["baseline_native_challenger"] = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]

    candidates: list[FrontierCandidate] = []

    def add(
        *,
        primary_topic: str,
        topic_stack: tuple[str, ...],
        source: SourceProfile,
        policy: FrontierPolicy,
        hypothesis: str,
        high_strategy: StrategyConfig | None = None,
    ) -> None:
        if len(candidates) >= max_iterations:
            return
        iteration = len(candidates) + 1
        name = f"f{iteration:03d}_{primary_topic}_{source.name}_{_policy_slug(policy)}"
        base_strategy = _strategy_from_updates(primary, source.updates)
        candidates.append(
            FrontierCandidate(
                iteration=iteration,
                name=name,
                primary_topic=primary_topic,
                topic_stack=topic_stack,
                source_profile=source.name,
                hypothesis=hypothesis,
                base_strategy=base_strategy,
                high_strategy=high_strategy or _high_strategy(base_strategy, policy.concentration),
                policy=policy,
            )
        )

    sources = _source_profiles()
    health_profiles = ("balanced", "strict", "fast", "breadth_credit", "ai_pure")
    crash_profiles = ("balanced", "credit_first", "tech_break", "breadth_break", "fast_drawdown")
    concentrations = _concentration_profiles()
    dynamic_guards = _dynamic_guard_profiles()
    crash_actions = _crash_action_profiles()
    routers = _router_profiles()

    for source in sources:
        for concentration in concentrations[:2]:
            for health_profile in health_profiles:
                add(
                    primary_topic="gated_concentration",
                    topic_stack=("confirmation_gated_high_concentration", "ai_leadership_health"),
                    source=source,
                    policy=FrontierPolicy(
                        health_profile=health_profile,
                        crash_profile="balanced",
                        concentration=concentration,
                        router=routers[0],
                    ),
                    hypothesis="Allow high concentration only when AI leadership health is strong and crash onset is quiet.",
                )

    for source in sources:
        for guard in dynamic_guards:
            for health_profile in health_profiles[:2]:
                add(
                    primary_topic="dynamic_guard",
                    topic_stack=(
                        "dynamic_guard_selection",
                        "ai_leadership_health",
                        "crash_onset_mesh",
                    ),
                    source=source,
                    policy=FrontierPolicy(
                        health_profile=health_profile,
                        crash_profile="balanced",
                        dynamic_guard=guard,
                    ),
                    hypothesis="Loosen the drawdown guard in healthy AI leadership and tighten it when crash-onset evidence clusters.",
                )

    for source in sources:
        for health_profile in health_profiles:
            for router in routers[:2]:
                add(
                    primary_topic="ai_health_score",
                    topic_stack=("ai_leadership_health", "confirmation_gated_high_concentration"),
                    source=source,
                    policy=FrontierPolicy(
                        health_profile=health_profile,
                        crash_profile="balanced",
                        concentration=concentrations[0],
                        router=router,
                    ),
                    hypothesis="Test whether different AI health definitions can decide when concentration is earned.",
                )

    for source in sources:
        for crash_profile in crash_profiles:
            for crash_action in crash_actions[:2]:
                add(
                    primary_topic="crash_onset_mesh",
                    topic_stack=("crash_onset_mesh", "dynamic_guard_selection"),
                    source=source,
                    policy=FrontierPolicy(
                        health_profile="balanced",
                        crash_profile=crash_profile,
                        dynamic_guard=dynamic_guards[1],
                        crash_action=crash_action,
                    ),
                    hypothesis="Use a late crash-onset mesh to cut risk only when price, credit, and breadth damage cluster.",
                )

    for source in sources:
        for router in routers:
            for crash_profile in crash_profiles[:2]:
                add(
                    primary_topic="two_model_router",
                    topic_stack=(
                        "two_model_router",
                        "confirmation_gated_high_concentration",
                        "ai_leadership_health",
                        "crash_onset_mesh",
                    ),
                    source=source,
                    policy=FrontierPolicy(
                        health_profile="balanced",
                        crash_profile=crash_profile,
                        concentration=concentrations[0],
                        router=router,
                        dynamic_guard=dynamic_guards[0],
                    ),
                    hypothesis="Route between native challenger behavior, high-concentration risk-on, and defensive scaling.",
                )

    return baselines, tuple(candidates[:max_iterations])


def _source_profiles() -> tuple[SourceProfile, ...]:
    return (
        SourceProfile("guard16_vol185_mult60", _source_updates(-0.16, 0.60, 0.185)),
        SourceProfile("guard17_vol185_mult65", _source_updates(-0.17, 0.65, 0.185)),
        SourceProfile("guard18_vol185_mult70", _source_updates(-0.18, 0.70, 0.185)),
        SourceProfile("guard15_vol19_mult60", _source_updates(-0.15, 0.60, 0.190)),
        SourceProfile("guard17_vol20_mult65", _source_updates(-0.17, 0.65, 0.200)),
    )


def _source_updates(max_drawdown: float, risk_multiplier: float, vol: float) -> dict[str, Any]:
    return {
        "type": "dual_momentum_risk_repair",
        "tickers": list(CORE_AI_TICKERS),
        "lookback_days": 63,
        "skip_days": 5,
        "top_n": 4,
        "defensive_ticker": "BIL",
        "min_return": 0.025,
        "ranking_metric": "risk_adjusted_return",
        "weighting": "risk_adjusted_score",
        "volatility_lookback_days": 63,
        "trend_filter_days": None,
        "max_asset_weight": 0.35,
        "risk_repair_signal": "balanced",
        "risk_repair_defensive_cap": 0.85,
        "risk_repair_defensive_release": 0.15,
        "risk_repair_ai_soft_cap": 0.85,
        "risk_repair_ai_soft_threshold": 0.90,
        "risk_repair_ai_excess_destination": "diversifier_mix",
        "risk_repair_ai_diversifier_tickers": ["SPY", "RSP", "GLD", "TLT"],
        "risk_repair_lookback_days": 42,
        "volatility_target": VolatilityTargetConfig(
            annualized_volatility=vol,
            lookback_days=21,
            max_leverage=1.0,
        ),
        "drawdown_control": DrawdownControlConfig(
            equity_lookback_days=84,
            max_drawdown=max_drawdown,
            risk_multiplier=risk_multiplier,
        ),
    }


def _concentration_profiles() -> tuple[ConcentrationProfile, ...]:
    return (
        ConcentrationProfile("top3_max40", 3, 0.40),
        ConcentrationProfile("top3_max45", 3, 0.45),
        ConcentrationProfile("top4_max40", 4, 0.40),
    )


def _dynamic_guard_profiles() -> tuple[DynamicGuardProfile, ...]:
    return (
        DynamicGuardProfile("balanced", -0.17, 0.65, -0.18, 0.75, -0.13, 0.45, 0.70, 0.58),
        DynamicGuardProfile("crash_fast", -0.17, 0.65, -0.18, 0.70, -0.10, 0.35, 0.65, 0.55),
        DynamicGuardProfile("late_only", -0.18, 0.70, -0.19, 0.80, -0.145, 0.55, 0.75, 0.62),
        DynamicGuardProfile("credit_tight", -0.16, 0.60, -0.18, 0.70, -0.12, 0.40, 0.70, 0.50),
        DynamicGuardProfile("soft_landing", -0.17, 0.70, -0.19, 0.85, -0.14, 0.55, 0.78, 0.65),
    )


def _crash_action_profiles() -> tuple[CrashActionProfile, ...]:
    return (
        CrashActionProfile("scale55", 0.58, 0.55, False),
        CrashActionProfile("scale40", 0.65, 0.40, False),
        CrashActionProfile("gradient75to35", 0.50, 0.35, True),
    )


def _router_profiles() -> tuple[RouterProfile, ...]:
    return (
        RouterProfile("gate_h70_c55", "gated_high", 0.70, 0.55, 0.72, 0.45),
        RouterProfile("gate_h75_c50", "gated_high", 0.75, 0.50, 0.70, 0.45),
        RouterProfile("blend_h65_c60", "blend_high", 0.65, 0.60, 0.72, 0.50),
        RouterProfile("def_select_h70_c62", "defensive_select", 0.70, 0.62, 0.68, 0.35),
        RouterProfile("def_select_h78_c58", "defensive_select", 0.78, 0.58, 0.65, 0.40),
    )


def _strategy_from_updates(strategy: StrategyConfig, updates: dict[str, Any]) -> StrategyConfig:
    return strategy.model_copy(update=updates)


def _high_strategy(
    base_strategy: StrategyConfig,
    concentration: ConcentrationProfile | None,
) -> StrategyConfig | None:
    if concentration is None:
        return None
    return base_strategy.model_copy(
        update={
            "top_n": concentration.top_n,
            "max_asset_weight": concentration.max_asset_weight,
        }
    )


def _run_standard_strategy(
    config: BotConfig,
    name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
) -> BacktestResult:
    strategy_prices = _strategy_prices(prices, strategy)
    weights = build_strategy_weights(strategy_prices, strategy)
    return run_backtest(
        name,
        strategy_prices,
        weights,
        config.execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )


def _run_frontier_candidate(
    config: BotConfig,
    candidate: FrontierCandidate,
    prices: pd.DataFrame,
) -> BacktestResult:
    strategy_prices = _strategy_prices(prices, candidate.base_strategy)
    health = _ai_health_score(strategy_prices, candidate.policy.health_profile)
    crash = _crash_onset_score(strategy_prices, candidate.policy.crash_profile)
    base_weights = build_strategy_weights(strategy_prices, candidate.base_strategy)
    high_weights = (
        build_strategy_weights(strategy_prices, candidate.high_strategy)
        if candidate.high_strategy is not None
        else None
    )
    target_weights = _route_weights(
        base_weights,
        high_weights,
        health=health,
        crash=crash,
        policy=candidate.policy,
        defensive_ticker=candidate.base_strategy.defensive_ticker,
    )
    if candidate.policy.crash_action is not None:
        target_weights = _apply_crash_action(
            target_weights,
            crash=crash,
            action=candidate.policy.crash_action,
            defensive_ticker=candidate.base_strategy.defensive_ticker,
        )
    drawdown_policy = candidate.policy.dynamic_guard or _fixed_guard_policy(
        candidate.base_strategy.drawdown_control
    )
    return _run_policy_backtest(
        candidate.name,
        strategy_prices,
        target_weights,
        config.execution,
        volatility_target=candidate.base_strategy.volatility_target,
        drawdown_policy=drawdown_policy,
        health=health,
        crash=crash,
    )


def _strategy_prices(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = list(dict.fromkeys([*required_strategy_tickers(strategy), *SIGNAL_TICKERS]))
    missing = unusable_required_price_columns(prices, columns)
    if missing:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing}")
    return prices[columns].dropna(how="all")


def _route_weights(
    base_weights: pd.DataFrame,
    high_weights: pd.DataFrame | None,
    *,
    health: pd.Series,
    crash: pd.Series,
    policy: FrontierPolicy,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    if high_weights is None or policy.router is None:
        return base_weights
    high = high_weights.reindex(base_weights.index).fillna(0.0)
    router = policy.router
    healthy = health.reindex(base_weights.index).fillna(0.0) >= router.health_threshold
    quiet = crash.reindex(base_weights.index).fillna(0.0) < router.crash_threshold
    if router.mode == "gated_high":
        use_high = healthy & quiet
        output = base_weights.copy()
        output.loc[use_high] = high.loc[use_high]
        return output
    if router.mode == "blend_high":
        blend = (
            (health - router.health_threshold) / max(1.0 - router.health_threshold, 0.01)
        ).clip(lower=0.0, upper=1.0)
        crash_discount = (1.0 - crash).clip(lower=0.0, upper=1.0)
        blend = (blend * crash_discount).reindex(base_weights.index).fillna(0.0)
        return base_weights.mul(1.0 - blend, axis=0).add(high.mul(blend, axis=0), fill_value=0.0)
    if router.mode == "defensive_select":
        output = _route_weights(
            base_weights,
            high_weights,
            health=health,
            crash=crash,
            policy=FrontierPolicy(
                health_profile=policy.health_profile,
                crash_profile=policy.crash_profile,
                router=RouterProfile(
                    router.name,
                    "gated_high",
                    router.health_threshold,
                    router.crash_threshold,
                    router.defensive_crash_threshold,
                    router.defensive_risk_weight,
                ),
            ),
            defensive_ticker=defensive_ticker,
        )
        crashy = crash.reindex(output.index).fillna(0.0) >= router.defensive_crash_threshold
        scaled = _scale_to_defensive(
            output,
            risk_weight=pd.Series(1.0, index=output.index).where(
                ~crashy, router.defensive_risk_weight
            ),
            defensive_ticker=defensive_ticker,
        )
        return scaled
    return base_weights


def _apply_crash_action(
    weights: pd.DataFrame,
    *,
    crash: pd.Series,
    action: CrashActionProfile,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    aligned = crash.reindex(weights.index).fillna(0.0)
    if action.use_gradient:
        severity = ((aligned - action.threshold) / max(1.0 - action.threshold, 0.01)).clip(
            lower=0.0,
            upper=1.0,
        )
        risk_weight = 1.0 - severity * (1.0 - action.risk_weight)
    else:
        risk_weight = pd.Series(1.0, index=weights.index).where(
            aligned < action.threshold,
            action.risk_weight,
        )
    return _scale_to_defensive(weights, risk_weight=risk_weight, defensive_ticker=defensive_ticker)


def _scale_to_defensive(
    weights: pd.DataFrame,
    *,
    risk_weight: pd.Series,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    if not defensive_ticker or defensive_ticker not in weights.columns:
        return weights.mul(risk_weight.reindex(weights.index).fillna(1.0), axis=0)
    output = weights.copy()
    risk_columns = [column for column in output.columns if column != defensive_ticker]
    scale = risk_weight.reindex(output.index).fillna(1.0).clip(lower=0.0, upper=1.0)
    output.loc[:, risk_columns] = output[risk_columns].mul(scale, axis=0)
    risk_sum = output[risk_columns].sum(axis=1).clip(lower=0.0, upper=1.0)
    output.loc[:, defensive_ticker] = (1.0 - risk_sum).clip(lower=0.0)
    return output.fillna(0.0)


def _run_policy_backtest(
    name: str,
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig,
    *,
    volatility_target: VolatilityTargetConfig | None,
    drawdown_policy: DynamicGuardProfile,
    health: pd.Series,
    crash: pd.Series,
) -> BacktestResult:
    prices = prices.sort_index()
    asset_returns = daily_returns(prices)
    target_weights = target_weights.reindex(prices.index).astype(float).fillna(0.0)
    price_available = (
        bounded_forward_fill(prices)
        .notna()
        .reindex(
            columns=target_weights.columns,
            fill_value=False,
        )
    )
    target_weights = target_weights.where(price_available, 0.0)
    target_weights = _rebalance_weights(target_weights, execution.rebalance)
    target_weights = target_weights.where(price_available, 0.0)
    target_weights = _normalize_long_only(target_weights)
    execution_weights = target_weights.shift(execution.signal_lag_days).fillna(0.0)
    if volatility_target is not None:
        execution_weights = apply_volatility_target(
            execution_weights, asset_returns, volatility_target
        )
    execution_weights = _apply_dynamic_drawdown_control(
        execution_weights,
        asset_returns,
        drawdown_policy,
        health=health,
        crash=crash,
    )
    validate_held_price_availability(execution_weights, price_available)
    execution_weights = execution_weights.where(price_available, 0.0)
    execution_weights = _normalize_long_only(execution_weights)
    turnover = (
        execution_weights.diff().abs().sum(axis=1).fillna(execution_weights.abs().sum(axis=1))
    )
    transaction_costs = turnover * execution.transaction_cost_bps / 10000.0
    gross_returns = (execution_weights * asset_returns).sum(axis=1)
    net_returns = gross_returns - transaction_costs
    equity = execution.initial_capital * (1.0 + net_returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=net_returns.rename(name),
        gross_returns=gross_returns.rename(name),
        weights=execution_weights,
        target_weights=target_weights,
        turnover=turnover.rename(name),
        transaction_costs=transaction_costs.rename(name),
    )


def _apply_dynamic_drawdown_control(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    policy: DynamicGuardProfile,
    *,
    health: pd.Series,
    crash: pd.Series,
) -> pd.DataFrame:
    strategy_returns = (weights * asset_returns).sum(axis=1)
    shadow_equity = (1.0 + strategy_returns).cumprod()
    dd = rolling_drawdown(shadow_equity, 84)
    threshold = pd.Series(policy.default_max_drawdown, index=weights.index)
    multiplier = pd.Series(policy.default_multiplier, index=weights.index)
    healthy = health.reindex(weights.index).fillna(0.0) >= policy.health_threshold
    crashy = crash.reindex(weights.index).fillna(0.0) >= policy.crash_threshold
    threshold.loc[healthy & ~crashy] = policy.healthy_max_drawdown
    multiplier.loc[healthy & ~crashy] = policy.healthy_multiplier
    threshold.loc[crashy] = policy.stress_max_drawdown
    multiplier.loc[crashy] = policy.stress_multiplier
    scale = pd.Series(1.0, index=weights.index)
    scale.loc[dd <= threshold] = multiplier.loc[dd <= threshold]
    scale = scale.shift(1).fillna(1.0)
    return weights.mul(scale, axis=0)


def _fixed_guard_policy(control: DrawdownControlConfig | None) -> DynamicGuardProfile:
    if control is None:
        return DynamicGuardProfile("none", -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0)
    return DynamicGuardProfile(
        "fixed",
        control.max_drawdown,
        control.risk_multiplier,
        control.max_drawdown,
        control.risk_multiplier,
        control.max_drawdown,
        control.risk_multiplier,
        1.0,
        1.0,
    )


def _ai_health_score(prices: pd.DataFrame, profile: str) -> pd.Series:
    filled = bounded_forward_fill(prices)
    components = {
        "qqq_trend": _trend_component(filled, "QQQ", 100),
        "smh_trend": _trend_component(filled, "SMH", 100),
        "qqq_rsp": _relative_component(filled, "QQQ", "RSP", 42, -0.01),
        "smh_spy": _relative_component(filled, "SMH", "SPY", 42, -0.015),
        "credit": _relative_component(filled, "HYG", "LQD", 42, -0.015),
        "breadth": _relative_component(filled, "RSP", "SPY", 42, -0.020),
        "short_ai": _relative_component(filled, "SMH", "SPY", 21, -0.010),
    }
    if profile == "strict":
        selected = ["qqq_trend", "smh_trend", "qqq_rsp", "smh_spy", "credit"]
    elif profile == "fast":
        selected = ["qqq_trend", "short_ai", "qqq_rsp", "credit"]
    elif profile == "breadth_credit":
        selected = ["qqq_trend", "smh_trend", "breadth", "credit"]
    elif profile == "ai_pure":
        selected = ["qqq_trend", "smh_trend", "qqq_rsp", "smh_spy", "short_ai"]
    else:
        selected = list(components)
    return _mean_components([components[name] for name in selected], filled.index)


def _crash_onset_score(prices: pd.DataFrame, profile: str) -> pd.Series:
    filled = bounded_forward_fill(prices)
    components = {
        "qqq_dd": _drawdown_component(filled, "QQQ", 126, -0.08),
        "smh_dd": _drawdown_component(filled, "SMH", 84, -0.10),
        "qqq_trend_break": 1.0 - _trend_component(filled, "QQQ", 100),
        "smh_trend_break": 1.0 - _trend_component(filled, "SMH", 100),
        "credit_break": 1.0 - _relative_component(filled, "HYG", "LQD", 42, -0.015),
        "breadth_break": 1.0 - _relative_component(filled, "RSP", "SPY", 42, -0.020),
        "fast_drawdown": _drawdown_component(filled, "QQQ", 42, -0.05),
        "vol_break": _vol_component(filled, "QQQ", 21),
    }
    if profile == "credit_first":
        selected = ["credit_break", "breadth_break", "qqq_trend_break", "fast_drawdown"]
    elif profile == "tech_break":
        selected = ["qqq_dd", "smh_dd", "qqq_trend_break", "smh_trend_break"]
    elif profile == "breadth_break":
        selected = ["breadth_break", "credit_break", "qqq_dd", "vol_break"]
    elif profile == "fast_drawdown":
        selected = ["fast_drawdown", "vol_break", "qqq_trend_break", "smh_trend_break"]
    else:
        selected = list(components)
    return _mean_components([components[name] for name in selected], filled.index)


def _trend_component(prices: pd.DataFrame, ticker: str, days: int) -> pd.Series:
    if ticker not in prices.columns:
        return pd.Series(0.5, index=prices.index)
    ma = moving_average(prices[[ticker]], days)[ticker]
    valid = prices[ticker].notna() & ma.notna()
    return prices[ticker].gt(ma).astype(float).where(valid, 0.5)


def _relative_component(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    days: int,
    threshold: float,
) -> pd.Series:
    if numerator not in prices.columns or denominator not in prices.columns:
        return pd.Series(0.5, index=prices.index)
    relative = prices[numerator] / prices[denominator]
    relative_return = relative.pct_change(days, fill_method=None)
    return relative_return.gt(threshold).astype(float).where(relative_return.notna(), 0.5)


def _drawdown_component(
    prices: pd.DataFrame,
    ticker: str,
    days: int,
    threshold: float,
) -> pd.Series:
    if ticker not in prices.columns:
        return pd.Series(0.5, index=prices.index)
    drawdown = (
        prices[ticker] / prices[ticker].rolling(days, min_periods=max(5, days // 5)).max()
    ) - 1.0
    return drawdown.le(threshold).astype(float).where(drawdown.notna(), 0.5)


def _vol_component(prices: pd.DataFrame, ticker: str, days: int) -> pd.Series:
    if ticker not in prices.columns:
        return pd.Series(0.5, index=prices.index)
    vol = realized_volatility(daily_returns(prices[ticker]), days)
    threshold = vol.rolling(252, min_periods=63).quantile(0.80)
    valid = prices[ticker].notna() & vol.notna() & threshold.notna()
    return vol.gt(threshold).astype(float).where(valid, 0.5)


def _mean_components(components: list[pd.Series], index: pd.Index) -> pd.Series:
    if not components:
        return pd.Series(0.5, index=index)
    return pd.concat(
        [component.reindex(index).fillna(0.5) for component in components], axis=1
    ).mean(axis=1)


def _rebalance_weights(weights: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    if rebalance.lower() in {"daily", "d"}:
        return weights
    periods = weights.index.to_period(rebalance)
    last_dates = pd.Series(weights.index, index=weights.index).groupby(periods).transform("max")
    rebalanced = weights.loc[weights.index == last_dates]
    return rebalanced.reindex(weights.index).ffill().fillna(0.0)


def _normalize_long_only(weights: pd.DataFrame) -> pd.DataFrame:
    clipped = weights.clip(lower=0.0).fillna(0.0)
    row_sum = clipped.sum(axis=1)
    over_invested = row_sum > 1.0
    clipped.loc[over_invested] = clipped.loc[over_invested].div(row_sum.loc[over_invested], axis=0)
    return clipped.fillna(0.0)


def _candidate_tickers(strategies: tuple[StrategyConfig, ...]) -> set[str]:
    tickers: set[str] = set(SIGNAL_TICKERS)
    for strategy in strategies:
        tickers.update(required_strategy_tickers(strategy))
    return tickers


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _metric_row(
    *,
    result: BacktestResult,
    iteration: int,
    role: str,
    primary_topic: str,
    topic_stack: tuple[str, ...],
    source_profile: str,
    hypothesis: str,
    primary_metrics: PerformanceMetrics | None,
    native_metrics: PerformanceMetrics | None,
) -> dict[str, object]:
    metrics = _metrics(result)
    defensive = (
        result.weights["BIL"]
        if "BIL" in result.weights.columns
        else pd.Series(0.0, index=result.weights.index)
    )
    row: dict[str, object] = {
        "result_name": result.name,
        "iteration": iteration,
        "role": role,
        "primary_topic": primary_topic,
        "topic_stack": "+".join(topic_stack),
        "source_profile": source_profile,
        "hypothesis": hypothesis,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "average_ai_growth_weight": _average_ai_weight(result.weights),
        "average_defensive_weight": float(defensive.mean()),
        "hard_defensive_day_rate": float((defensive >= 0.50).mean()),
    }
    if primary_metrics is not None:
        row["delta_vs_primary_cagr"] = metrics.cagr - primary_metrics.cagr
        row["delta_vs_primary_max_drawdown"] = metrics.max_drawdown - primary_metrics.max_drawdown
        row["delta_vs_primary_calmar"] = metrics.calmar - primary_metrics.calmar
    if native_metrics is not None:
        row["delta_vs_native_cagr"] = metrics.cagr - native_metrics.cagr
        row["delta_vs_native_max_drawdown"] = metrics.max_drawdown - native_metrics.max_drawdown
        row["delta_vs_native_calmar"] = metrics.calmar - native_metrics.calmar
    return row


def _average_ai_weight(weights: pd.DataFrame) -> float:
    columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    if not columns:
        return 0.0
    return float(weights[columns].sum(axis=1).mean())


def _add_robustness(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    walk_forward: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    output = metrics.copy()
    if not rolling.empty:
        three_year = rolling[rolling["window"].eq("3y")][
            ["name", "worst_cagr", "positive_window_rate", "worst_drawdown"]
        ].rename(
            columns={
                "name": "result_name",
                "worst_cagr": "worst_3y_cagr",
                "positive_window_rate": "positive_3y_window_rate",
                "worst_drawdown": "worst_3y_drawdown",
            }
        )
        output = output.merge(three_year, on="result_name", how="left")
    if not walk_forward.empty:
        output = output.merge(
            walk_forward[
                [
                    "name",
                    "walk_forward_median_cagr",
                    "walk_forward_worst_cagr",
                    "walk_forward_positive_rate",
                    "walk_forward_worst_drawdown",
                ]
            ].rename(columns={"name": "result_name"}),
            on="result_name",
            how="left",
        )
    if not calendar.empty:
        year_summary = (
            calendar.groupby("name", observed=True)
            .agg(
                calendar_years=("window", "count"),
                negative_calendar_years=(
                    "total_return",
                    lambda values: int((values < 0).sum()),
                ),
            )
            .reset_index()
            .rename(columns={"name": "result_name"})
        )
        output = output.merge(year_summary, on="result_name", how="left")
    return output


def _score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    native_cagr_delta = _numeric_column(output, "delta_vs_native_cagr")
    native_dd_delta = _numeric_column(output, "delta_vs_native_max_drawdown")
    native_calmar_delta = _numeric_column(output, "delta_vs_native_calmar")
    output["drawdown_penalty"] = (output["max_drawdown"].abs() - 0.205).clip(lower=0.0) * 2.8
    output["native_cagr_shortfall_penalty"] = native_cagr_delta.clip(upper=0.0).abs() * 1.7
    output["low_ai_penalty"] = (0.60 - output["average_ai_growth_weight"]).clip(lower=0.0) * 0.25
    output["stale_defense_penalty"] = output["hard_defensive_day_rate"].clip(lower=0.08) * 0.10
    output["walk_forward_penalty"] = (
        0.12 - _numeric_column(output, "walk_forward_worst_cagr")
    ).clip(lower=0.0) * 0.25
    output["frontier_score"] = (
        output["cagr"]
        + 0.42 * output["calmar"]
        + 0.20 * native_dd_delta
        + 0.08 * native_calmar_delta
        - output["drawdown_penalty"]
        - output["native_cagr_shortfall_penalty"]
        - output["low_ai_penalty"]
        - output["stale_defense_penalty"]
        - output["walk_forward_penalty"]
    )
    output["promotion_candidate"] = (
        output["role"].eq("frontier_candidate")
        & (output["cagr"] >= 0.220)
        & (output["max_drawdown"] >= -0.205)
        & (output["average_ai_growth_weight"] >= 0.60)
    )
    output["big_improvement"] = (
        output["role"].eq("frontier_candidate")
        & (output["cagr"] >= 0.225)
        & (output["max_drawdown"] >= -0.205)
        & (native_cagr_delta >= 0.004)
    ) | (
        output["role"].eq("frontier_candidate")
        & (output["cagr"] >= 0.219)
        & (native_dd_delta >= 0.008)
        & (output["max_drawdown"] >= -0.190)
    )
    return output.sort_values("frontier_score", ascending=False)


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _candidate_roster(candidates: tuple[FrontierCandidate, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_name": candidate.name,
                "iteration": candidate.iteration,
                "primary_topic": candidate.primary_topic,
                "topic_stack": "+".join(candidate.topic_stack),
                "source_profile": candidate.source_profile,
                "health_profile": candidate.policy.health_profile,
                "crash_profile": candidate.policy.crash_profile,
                "concentration": (
                    candidate.policy.concentration.name
                    if candidate.policy.concentration is not None
                    else ""
                ),
                "router": (
                    candidate.policy.router.name if candidate.policy.router is not None else ""
                ),
                "dynamic_guard": (
                    candidate.policy.dynamic_guard.name
                    if candidate.policy.dynamic_guard is not None
                    else ""
                ),
                "crash_action": (
                    candidate.policy.crash_action.name
                    if candidate.policy.crash_action is not None
                    else ""
                ),
                "hypothesis": candidate.hypothesis,
                "tickers": ",".join(candidate.base_strategy.tickers),
            }
            for candidate in candidates
        ]
    )


def _checkpoint_summary(
    metrics: pd.DataFrame,
    *,
    checkpoint_size: int,
    native_metrics: PerformanceMetrics | None,
) -> pd.DataFrame:
    candidates = metrics[metrics["role"].eq("frontier_candidate")].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["checkpoint"] = ((candidates["iteration"].astype(int) - 1) // checkpoint_size) + 1
    rows: list[dict[str, object]] = []
    native_cagr = native_metrics.cagr if native_metrics is not None else float("nan")
    native_dd = native_metrics.max_drawdown if native_metrics is not None else float("nan")
    for checkpoint, group in candidates.groupby("checkpoint", observed=True):
        best_score = group.sort_values("frontier_score", ascending=False).iloc[0]
        best_cagr = group.sort_values("cagr", ascending=False).iloc[0]
        best_drawdown = group.sort_values("max_drawdown", ascending=False).iloc[0]
        topic_counts = group.sort_values("frontier_score", ascending=False)["primary_topic"].head(5)
        pursue = group[group["promotion_candidate"] | group["big_improvement"]]
        signal = (
            str(pursue.sort_values("frontier_score", ascending=False).iloc[0]["primary_topic"])
            if not pursue.empty
            else str(topic_counts.mode().iloc[0])
        )
        rows.append(
            {
                "checkpoint": int(checkpoint),
                "iteration_start": int(group["iteration"].min()),
                "iteration_end": int(group["iteration"].max()),
                "candidates": int(len(group)),
                "best_score_result": best_score["result_name"],
                "best_score_topic": best_score["primary_topic"],
                "best_frontier_score": best_score["frontier_score"],
                "best_score_cagr": best_score["cagr"],
                "best_score_max_drawdown": best_score["max_drawdown"],
                "best_cagr_result": best_cagr["result_name"],
                "best_cagr": best_cagr["cagr"],
                "best_cagr_max_drawdown": best_cagr["max_drawdown"],
                "best_drawdown_result": best_drawdown["result_name"],
                "best_max_drawdown": best_drawdown["max_drawdown"],
                "promotion_candidates": int(group["promotion_candidate"].sum()),
                "big_improvements": int(group["big_improvement"].sum()),
                "signal_to_pursue": signal,
                "native_cagr_reference": native_cagr,
                "native_max_drawdown_reference": native_dd,
            }
        )
    return pd.DataFrame(rows)


def _family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[metrics["role"].eq("frontier_candidate")]
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for topic, group in candidates.groupby("primary_topic", observed=True):
        best_score = group.sort_values("frontier_score", ascending=False).iloc[0]
        best_cagr = group.sort_values("cagr", ascending=False).iloc[0]
        best_drawdown = group.sort_values("max_drawdown", ascending=False).iloc[0]
        rows.append(
            {
                "primary_topic": topic,
                "candidates": int(len(group)),
                "best_score_result": best_score["result_name"],
                "best_frontier_score": best_score["frontier_score"],
                "best_score_cagr": best_score["cagr"],
                "best_score_max_drawdown": best_score["max_drawdown"],
                "best_cagr_result": best_cagr["result_name"],
                "best_cagr": best_cagr["cagr"],
                "best_cagr_max_drawdown": best_cagr["max_drawdown"],
                "median_cagr": group["cagr"].median(),
                "best_drawdown_result": best_drawdown["result_name"],
                "best_max_drawdown": best_drawdown["max_drawdown"],
                "median_max_drawdown": group["max_drawdown"].median(),
                "best_calmar": group["calmar"].max(),
                "median_ai_growth_weight": group["average_ai_growth_weight"].median(),
                "median_hard_defensive_day_rate": group["hard_defensive_day_rate"].median(),
                "promotion_rate": group["promotion_candidate"].mean(),
                "big_improvement_count": int(group["big_improvement"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["big_improvement_count", "best_frontier_score"],
        ascending=False,
    )


def _summary_markdown(
    metrics: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    *,
    max_iterations: int,
) -> str:
    baselines = metrics[metrics["role"].eq("baseline")]
    candidates = metrics[metrics["role"].eq("frontier_candidate")]
    top = candidates.sort_values("frontier_score", ascending=False).head(15)
    big = candidates[candidates["big_improvement"]].sort_values(
        "frontier_score",
        ascending=False,
    )
    lines = [
        "# I111 Frontier Search",
        "",
        "## Goal",
        "",
        (
            "Test five larger research mechanisms and combinations: confirmation-gated "
            "high concentration, dynamic guard selection, AI leadership health, crash "
            "onset mesh, and a two-model router."
        ),
        "",
        f"Iterations tested: {max_iterations}. Checkpoint cadence: every 20 iterations.",
        "",
        "## Baselines",
        "",
        "| baseline | CAGR | max DD | Calmar | AI/growth wt | hard defensive days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in baselines.iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['calmar']:.2f} | "
            f"{row['average_ai_growth_weight']:.2%} | "
            f"{row['hard_defensive_day_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Best Candidates",
            "",
            "| result | topic | CAGR | max DD | native CAGR delta | native DD delta | AI/growth wt | big? |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['primary_topic']} | "
            f"{row['cagr']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row.get('delta_vs_native_cagr', 0.0):.2%} | "
            f"{row.get('delta_vs_native_max_drawdown', 0.0):.2%} | "
            f"{row['average_ai_growth_weight']:.2%} | {bool(row['big_improvement'])} |"
        )
    lines.extend(["", "## Checkpoints", ""])
    if checkpoint_summary.empty:
        lines.append("No checkpoint rows were produced.")
    else:
        lines.append(
            "| checkpoint | iterations | best topic | best CAGR | best max DD | promotions | big | pursue |"
        )
        lines.append("|---:|---:|---|---:|---:|---:|---:|---|")
        for _, row in checkpoint_summary.iterrows():
            lines.append(
                f"| {int(row['checkpoint'])} | "
                f"{int(row['iteration_start'])}-{int(row['iteration_end'])} | "
                f"{row['best_score_topic']} | {row['best_score_cagr']:.2%} | "
                f"{row['best_score_max_drawdown']:.2%} | "
                f"{int(row['promotion_candidates'])} | {int(row['big_improvements'])} | "
                f"{row['signal_to_pursue']} |"
            )
    lines.extend(["", "## Topic Summary", ""])
    if family_summary.empty:
        lines.append("No topic summary rows were produced.")
    else:
        lines.append(
            "| topic | candidates | best score result | score CAGR | score max DD | best CAGR | best CAGR max DD | promotion rate | big |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for _, row in family_summary.iterrows():
            lines.append(
                f"| {row['primary_topic']} | {int(row['candidates'])} | "
                f"`{row['best_score_result']}` | {row['best_score_cagr']:.2%} | "
                f"{row['best_score_max_drawdown']:.2%} | {row['best_cagr']:.2%} | "
                f"{row['best_cagr_max_drawdown']:.2%} | {row['promotion_rate']:.0%} | "
                f"{int(row['big_improvement_count'])} |"
            )
    lines.extend(["", "## Readout", ""])
    if big.empty:
        lines.append(
            "No candidate cleared the big-improvement gate. Treat the frontier search as "
            "landscape evidence unless a manual review finds a narrower robustness reason "
            "to promote one of the top candidates."
        )
    else:
        winner = big.iloc[0]
        lines.append(
            f"Big improvement found: `{winner['result_name']}` with CAGR "
            f"{winner['cagr']:.2%}, max DD {winner['max_drawdown']:.2%}, Calmar "
            f"{winner['calmar']:.2f}, and AI/growth exposure "
            f"{winner['average_ai_growth_weight']:.2%}."
        )
    return "\n".join(lines) + "\n"


def _policy_slug(policy: FrontierPolicy) -> str:
    parts = [f"h{policy.health_profile}", f"c{policy.crash_profile}"]
    if policy.concentration is not None:
        parts.append(policy.concentration.name)
    if policy.router is not None:
        parts.append(policy.router.name)
    if policy.dynamic_guard is not None:
        parts.append(policy.dynamic_guard.name)
    if policy.crash_action is not None:
        parts.append(policy.crash_action.name)
    return "_".join(parts)
