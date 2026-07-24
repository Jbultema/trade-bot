from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import (
    BacktestResult,
    _scheduled_rebalance_dates,
    run_backtest,
)
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.backtest.windows import calendar_year_metrics, rolling_window_metrics
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig
from trade_bot.features.indicators import (
    daily_returns,
    lookback_returns,
    moving_average,
    realized_volatility,
)
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.backtest_pbo import (
    candidate_return_matrix,
    estimate_probability_of_backtest_overfitting,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_bias_calibration import CRISIS_WINDOWS
from trade_bot.research.drawdown_attribution import build_drawdown_attribution
from trade_bot.research.i111_orthogonal_search import (
    DEFAULT_I111_NATIVE_CHALLENGER,
)
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_SELECTOR_TRANSITION_OUTPUT_DIR = Path("reports/i111_selector_transition_repair")
CORE_TICKERS = ("SPY", "RSP")
CONFIGURED_PROFILE = "configured_wed_lag2"
BLOCK_SESSIONS = 63
BOOTSTRAP_SIMULATIONS = 1_000


@dataclass(frozen=True)
class SelectorTransitionSpec:
    name: str
    description: str
    research_phase: str = "pre_registered"
    incumbent_buffer: bool = False
    blended_rank: bool = False
    core_weight: float = 0.0
    recovery_meter: bool = False


@dataclass(frozen=True)
class I111SelectorTransitionRun:
    candidate_metrics: pd.DataFrame
    schedule_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    rolling_windows: pd.DataFrame
    calendar_years: pd.DataFrame
    crisis_metrics: pd.DataFrame
    transition_diagnostics: pd.DataFrame
    block_bootstrap: pd.DataFrame
    pbo_summary: pd.DataFrame
    promotion_gates: pd.DataFrame
    drawdown_summary: pd.DataFrame
    drawdown_contributors: pd.DataFrame
    drawdown_exposure_path: pd.DataFrame
    output_paths: dict[str, Path]


def fixed_selector_transition_specs() -> tuple[SelectorTransitionSpec, ...]:
    return (
        SelectorTransitionSpec(
            name="native_reference",
            description="Unmodified configured native i111 selector and transitions.",
        ),
        SelectorTransitionSpec(
            name="incumbent_buffer",
            description="Retain near-winning incumbents unless a challenger has a clear rank edge.",
            incumbent_buffer=True,
        ),
        SelectorTransitionSpec(
            name="blended_rank_63_126",
            description="Blend fast and slow risk-adjusted percentile ranks before selection.",
            blended_rank=True,
        ),
        SelectorTransitionSpec(
            name="diversified_core15",
            description="Reserve 15% of active risk for an equal SPY/RSP core.",
            core_weight=0.15,
        ),
        SelectorTransitionSpec(
            name="recovery_meter_core",
            description="Meter unconfirmed entries and route deferred risk through SPY/RSP.",
            recovery_meter=True,
        ),
        SelectorTransitionSpec(
            name="integrated_selector_transition",
            description="Combine blended rank, incumbent buffer, core, and recovery entry.",
            incumbent_buffer=True,
            blended_rank=True,
            core_weight=0.15,
            recovery_meter=True,
        ),
    )


def secondary_diagnostic_specs() -> tuple[SelectorTransitionSpec, ...]:
    """Ablate the initial integrated failure without granting promotion eligibility."""
    phase = "post_initial_mechanism_ablation"
    return (
        SelectorTransitionSpec(
            name="blended_rank_incumbent",
            description="Combine the slower blended rank with the incumbent buffer only.",
            research_phase=phase,
            incumbent_buffer=True,
            blended_rank=True,
        ),
        SelectorTransitionSpec(
            name="incumbent_buffer_core15",
            description="Combine the incumbent buffer with the 15% diversified core.",
            research_phase=phase,
            incumbent_buffer=True,
            core_weight=0.15,
        ),
        SelectorTransitionSpec(
            name="incumbent_buffer_recovery",
            description="Combine the incumbent buffer with recovery-metered entries.",
            research_phase=phase,
            incumbent_buffer=True,
            recovery_meter=True,
        ),
        SelectorTransitionSpec(
            name="core15_recovery",
            description="Combine the 15% diversified core with recovery-metered entries.",
            research_phase=phase,
            core_weight=0.15,
            recovery_meter=True,
        ),
        SelectorTransitionSpec(
            name="incumbent_core15_recovery",
            description="Combine buffer, core, and recovery mechanics without blended ranks.",
            research_phase=phase,
            incumbent_buffer=True,
            core_weight=0.15,
            recovery_meter=True,
        ),
    )


def fixed_execution_profiles(
    execution: ExecutionConfig,
) -> tuple[tuple[str, ExecutionConfig], ...]:
    return (
        (
            CONFIGURED_PROFILE,
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 2}),
        ),
        *tuple(
            (
                f"{weekday.lower()}_lag1",
                execution.model_copy(
                    update={
                        "rebalance": f"W-{weekday}",
                        "signal_lag_days": 1,
                    }
                ),
            )
            for weekday in ("MON", "TUE", "WED", "THU", "FRI")
        ),
        (
            "wed_lag5",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 5}),
        ),
        (
            "daily_lag2",
            execution.model_copy(update={"rebalance": "D", "signal_lag_days": 2}),
        ),
    )


def run_i111_selector_transition_repair(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_SELECTOR_TRANSITION_OUTPUT_DIR,
) -> I111SelectorTransitionRun:
    if DEFAULT_I111_NATIVE_CHALLENGER not in config.strategies:
        raise KeyError(f"Configured strategy {DEFAULT_I111_NATIVE_CHALLENGER!r} is required.")
    strategy = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]
    prices = baseline_run.prices.sort_index()
    required = set(strategy.tickers) | {
        strategy.defensive_ticker or "BIL",
        *CORE_TICKERS,
    }
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise KeyError("Missing selector-repair inputs: " + ", ".join(missing))

    raw_weights = build_strategy_weights(prices, strategy).reindex(prices.index).fillna(0.0)
    features = build_selector_features(prices, strategy)
    specs = (
        *fixed_selector_transition_specs(),
        *secondary_diagnostic_specs(),
    )
    profiles = fixed_execution_profiles(config.execution)
    cost_levels = tuple(dict.fromkeys([float(config.execution.transaction_cost_bps), 20.0]))

    result_lookup: dict[tuple[str, str, float], BacktestResult] = {}
    transition_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    target_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for spec, (profile_name, profile) in product(specs, profiles):
        if spec.name == "native_reference":
            target = raw_weights
            diagnostics = pd.DataFrame()
        else:
            target, diagnostics = build_repaired_target_weights(
                raw_weights,
                features,
                strategy,
                spec,
                rebalance=profile.rebalance,
            )
        target_cache[(spec.name, profile_name)] = target
        if not diagnostics.empty:
            diagnostics.insert(0, "candidate", spec.name)
            diagnostics.insert(1, "execution_profile", profile_name)
            transition_frames.append(diagnostics)
        for cost_bps in cost_levels:
            execution = profile.model_copy(update={"transaction_cost_bps": cost_bps})
            result_name = (
                spec.name
                if profile_name == CONFIGURED_PROFILE
                and cost_bps == float(config.execution.transaction_cost_bps)
                else f"{spec.name}__{profile_name}__{cost_bps:g}bps"
            )
            result = run_backtest(
                result_name,
                prices,
                target,
                execution,
                volatility_target=strategy.volatility_target,
                drawdown_control=strategy.drawdown_control,
            )
            result_lookup[(spec.name, profile_name, cost_bps)] = result
            metric_rows.append(
                _metric_row(
                    result,
                    candidate=spec.name,
                    description=spec.description,
                    execution_profile=profile_name,
                    cost_bps=cost_bps,
                    strategy_tickers=strategy.tickers,
                    research_phase=spec.research_phase,
                )
            )

    candidate_metrics = pd.DataFrame(metric_rows)
    transition_diagnostics = (
        pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    )
    schedule_summary = build_schedule_summary(
        candidate_metrics,
        base_cost_bps=float(config.execution.transaction_cost_bps),
    )
    configured_results = {
        spec.name: result_lookup[
            (
                spec.name,
                CONFIGURED_PROFILE,
                float(config.execution.transaction_cost_bps),
            )
        ]
        for spec in specs
    }
    rolling = rolling_window_metrics(
        configured_results,
        window_years=[1, 3, 5],
    )
    calendar = calendar_year_metrics(configured_results)
    folds = build_fold_metrics(configured_results)
    crises = build_crisis_metrics(configured_results)
    bootstrap = build_block_bootstrap(configured_results)
    pre_registered_names = {spec.name for spec in specs if spec.research_phase == "pre_registered"}
    return_matrix = candidate_return_matrix(
        {
            name: result
            for name, result in configured_results.items()
            if name in pre_registered_names
        },
        min_observations=252,
    )
    pbo = estimate_probability_of_backtest_overfitting(
        return_matrix,
        partitions=8,
        metric="sharpe",
    )
    gates = build_promotion_gates(
        candidate_metrics,
        schedule_summary,
        rolling,
        calendar,
        crises,
        bootstrap,
        pbo.summary,
        eligible_candidates=pre_registered_names,
        base_cost_bps=float(config.execution.transaction_cost_bps),
    )
    (
        drawdown_summary,
        drawdown_contributors,
        drawdown_exposure_path,
    ) = build_candidate_drawdown_artifacts(
        configured_results,
        prices,
        defensive_ticker=strategy.defensive_ticker,
    )

    frames = {
        "candidate_metrics": candidate_metrics,
        "schedule_summary": schedule_summary,
        "fold_metrics": folds,
        "rolling_windows": rolling,
        "calendar_years": calendar,
        "crisis_metrics": crises,
        "transition_diagnostics": transition_diagnostics,
        "block_bootstrap": bootstrap,
        "pbo_summary": pbo.summary,
        "pbo_splits": pbo.splits,
        "pbo_strategy_selection": pbo.strategy_selection,
        "promotion_gates": gates,
        "drawdown_summary": drawdown_summary,
        "drawdown_contributors": drawdown_contributors,
        "drawdown_exposure_path": drawdown_exposure_path,
    }
    paths = write_selector_transition_outputs(
        output_dir,
        config=config,
        prices=prices,
        specs=specs,
        profiles=profiles,
        frames=frames,
    )
    return I111SelectorTransitionRun(
        candidate_metrics=candidate_metrics,
        schedule_summary=schedule_summary,
        fold_metrics=folds,
        rolling_windows=rolling,
        calendar_years=calendar,
        crisis_metrics=crises,
        transition_diagnostics=transition_diagnostics,
        block_bootstrap=bootstrap,
        pbo_summary=pbo.summary,
        promotion_gates=gates,
        drawdown_summary=drawdown_summary,
        drawdown_contributors=drawdown_contributors,
        drawdown_exposure_path=drawdown_exposure_path,
        output_paths=paths,
    )


def build_selector_features(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
) -> dict[str, pd.DataFrame | pd.Series]:
    risk_prices = prices[strategy.tickers]
    returns = daily_returns(risk_prices)
    short_momentum = lookback_returns(
        risk_prices,
        strategy.lookback_days,
        strategy.skip_days,
    )
    short_volatility = realized_volatility(
        returns,
        strategy.volatility_lookback_days,
    )
    short_score = short_momentum.div(short_volatility.where(short_volatility.gt(0.0)))
    slow_momentum = lookback_returns(risk_prices, 126, 10)
    slow_volatility = realized_volatility(returns, 126)
    slow_score = slow_momentum.div(slow_volatility.where(slow_volatility.gt(0.0)))
    short_percentile = short_score.rank(
        axis=1,
        pct=True,
        method="average",
    )
    slow_percentile = slow_score.rank(
        axis=1,
        pct=True,
        method="average",
    )
    blended_percentile = 0.65 * short_percentile + 0.35 * slow_percentile
    recovery_confirmed = lookback_returns(risk_prices, 21, 0).ge(0.02) & risk_prices.gt(
        moving_average(risk_prices, 50)
    )
    market_confirmed = prices["SPY"].gt(moving_average(prices[["SPY"]], 100)["SPY"])
    return {
        "short_momentum": short_momentum,
        "short_percentile": short_percentile,
        "blended_percentile": blended_percentile,
        "recovery_confirmed": recovery_confirmed,
        "market_confirmed": market_confirmed,
    }


def build_repaired_target_weights(
    raw_weights: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    strategy: StrategyConfig,
    spec: SelectorTransitionSpec,
    *,
    rebalance: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_columns = list(
        dict.fromkeys(
            [
                *raw_weights.columns,
                *strategy.tickers,
                *CORE_TICKERS,
                strategy.defensive_ticker or "BIL",
            ]
        )
    )
    raw = raw_weights.reindex(columns=output_columns, fill_value=0.0)
    output = raw.copy()
    decision_dates = _scheduled_rebalance_dates(raw, rebalance)
    decision_set = set(decision_dates)
    defensive_ticker = strategy.defensive_ticker or "BIL"
    short_momentum = features["short_momentum"]
    rank_frame = (
        features["blended_percentile"] if spec.blended_rank else features["short_percentile"]
    )
    recovery_confirmed = features["recovery_confirmed"]
    market_confirmed = features["market_confirmed"]
    previous_target = pd.Series(0.0, index=output_columns)
    previous_selected: set[str] = set()
    diagnostic_rows: list[dict[str, object]] = []

    for date in raw.index:
        if date not in decision_set:
            continue
        raw_row = raw.loc[date].clip(lower=0.0)
        native_risk = raw_row.drop(
            labels=[defensive_ticker],
            errors="ignore",
        )
        risk_budget = float(native_risk.sum())
        selector_budget = float(raw_row.reindex(strategy.tickers).fillna(0.0).sum())
        residual_risk = native_risk.drop(
            labels=strategy.tickers,
            errors="ignore",
        )
        rank_row = rank_frame.loc[date].reindex(strategy.tickers)
        momentum_row = short_momentum.loc[date].reindex(strategy.tickers)
        eligible = momentum_row.gt(strategy.min_return) & rank_row.notna()
        ranked = rank_row.where(eligible).dropna().sort_values(ascending=False)
        selected = list(ranked.head(strategy.top_n).index)
        raw_selected = {
            ticker for ticker in strategy.tickers if float(raw_row.get(ticker, 0.0)) > 1e-8
        }
        buffer_kept = 0
        replacements_blocked = 0
        if not spec.blended_rank:
            selected = list(raw_selected)
        if spec.incumbent_buffer and previous_selected:
            selected, buffer_kept, replacements_blocked = _buffer_selection(
                selected,
                previous_selected,
                rank_row,
                momentum_row,
                top_n=strategy.top_n,
                min_return=strategy.min_return,
            )
        selector_mix = _selection_mix(
            selected,
            rank_row,
            output_columns=output_columns,
            raw_row=raw_row,
            previous_target=previous_target,
            use_rank=spec.blended_rank,
        )
        if selector_mix.sum() <= 0.0 and selector_budget > 0.0:
            raw_selector = raw_row.reindex(strategy.tickers).fillna(0.0)
            selector_mix = raw_selector.div(
                raw_selector.sum() if raw_selector.sum() > 0.0 else 1.0
            ).reindex(output_columns, fill_value=0.0)

        target = selector_mix * selector_budget
        target = target.add(
            residual_risk.reindex(output_columns, fill_value=0.0),
            fill_value=0.0,
        )
        core_budget = risk_budget * spec.core_weight
        if core_budget > 0.0:
            target *= 1.0 - spec.core_weight
            for ticker in CORE_TICKERS:
                target.loc[ticker] += core_budget / len(CORE_TICKERS)

        deferred_weight = 0.0
        deferred_names = 0
        if spec.recovery_meter and previous_target.sum() > 0.0:
            for ticker in strategy.tickers:
                desired = float(target.get(ticker, 0.0))
                prior = float(previous_target.get(ticker, 0.0))
                increase = desired - prior
                asset_confirmation = recovery_confirmed.loc[date, ticker]
                market_confirmation = market_confirmed.loc[date]
                confirmed = bool(
                    pd.notna(asset_confirmation)
                    and pd.notna(market_confirmation)
                    and asset_confirmation
                    and market_confirmation
                )
                if increase > 1e-12 and not confirmed:
                    allowed = prior + 0.50 * increase
                    deferred_weight += desired - allowed
                    deferred_names += 1
                    target.loc[ticker] = allowed
            for ticker in CORE_TICKERS:
                target.loc[ticker] += deferred_weight / len(CORE_TICKERS)

        target.loc[defensive_ticker] = max(0.0, 1.0 - risk_budget)
        total = float(target.sum())
        if total > 1.0 + 1e-12:
            target /= total
        output.loc[date] = target.reindex(output_columns, fill_value=0.0)
        entries = sum(
            float(target.get(ticker, 0.0)) > 1e-8
            and float(previous_target.get(ticker, 0.0)) <= 1e-8
            for ticker in strategy.tickers
        )
        exits = sum(
            float(target.get(ticker, 0.0)) <= 1e-8
            and float(previous_target.get(ticker, 0.0)) > 1e-8
            for ticker in strategy.tickers
        )
        diagnostic_rows.append(
            {
                "market_date": date,
                "risk_budget": risk_budget,
                "selected_names": len(selected),
                "entries": entries,
                "exits": exits,
                "buffer_kept": buffer_kept,
                "replacements_blocked": replacements_blocked,
                "deferred_entry_names": deferred_names,
                "deferred_entry_weight": deferred_weight,
                "core_weight": float(target.reindex(CORE_TICKERS).fillna(0.0).sum()),
            }
        )
        previous_target = target.reindex(output_columns, fill_value=0.0)
        previous_selected = {
            ticker for ticker in strategy.tickers if float(previous_target.get(ticker, 0.0)) > 1e-8
        }
    output = output.loc[decision_dates].reindex(raw.index).ffill().fillna(0.0)
    return output, pd.DataFrame(diagnostic_rows)


def _buffer_selection(
    desired: list[str],
    previous: set[str],
    rank: pd.Series,
    momentum: pd.Series,
    *,
    top_n: int,
    min_return: float,
) -> tuple[list[str], int, int]:
    selected = list(dict.fromkeys(desired))
    newcomers = [ticker for ticker in selected if ticker not in previous]
    incumbents = [
        ticker
        for ticker in previous
        if ticker not in selected
        and momentum.get(ticker, np.nan) > min_return
        and rank.rank(ascending=False, method="min").get(ticker, np.inf) <= top_n + 2
    ]
    kept = 0
    blocked = 0
    for incumbent in sorted(
        incumbents,
        key=lambda ticker: float(rank.get(ticker, -np.inf)),
        reverse=True,
    )[:2]:
        if not newcomers:
            break
        weakest = min(
            newcomers,
            key=lambda ticker: float(rank.get(ticker, -np.inf)),
        )
        advantage = float(rank.get(weakest, 0.0)) - float(rank.get(incumbent, 0.0))
        if advantage < 0.15:
            selected.remove(weakest)
            selected.append(incumbent)
            newcomers.remove(weakest)
            kept += 1
            blocked += 1
    selected = sorted(
        selected,
        key=lambda ticker: float(rank.get(ticker, -np.inf)),
        reverse=True,
    )
    return selected[:top_n], kept, blocked


def _selection_mix(
    selected: list[str],
    rank: pd.Series,
    *,
    output_columns: list[str],
    raw_row: pd.Series | None = None,
    previous_target: pd.Series | None = None,
    use_rank: bool = True,
) -> pd.Series:
    mix = pd.Series(0.0, index=output_columns)
    if not selected:
        return mix
    if use_rank:
        scores = rank.reindex(selected).clip(lower=0.0).fillna(0.0)
    else:
        scores = (
            raw_row.reindex(selected).clip(lower=0.0).fillna(0.0)
            if raw_row is not None
            else pd.Series(0.0, index=selected)
        )
        if previous_target is not None:
            prior = previous_target.reindex(selected).clip(lower=0.0).fillna(0.0)
            scores = scores.where(scores.gt(0.0), prior)
    if scores.sum() <= 0.0:
        scores = rank.reindex(selected).clip(lower=0.0).fillna(0.0)
    if scores.sum() <= 0.0:
        scores = pd.Series(1.0, index=selected, dtype=float)
    mix.loc[selected] = scores / scores.sum()
    return mix


def _metric_row(
    result: BacktestResult,
    *,
    candidate: str,
    description: str,
    execution_profile: str,
    cost_bps: float,
    strategy_tickers: list[str],
    research_phase: str,
) -> dict[str, object]:
    metrics = calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )
    risk = result.weights.reindex(
        columns=strategy_tickers,
        fill_value=0.0,
    )
    risk_total = risk.sum(axis=1).replace(0.0, np.nan)
    normalized = risk.div(risk_total, axis=0)
    return {
        "candidate": candidate,
        "description": description,
        "research_phase": research_phase,
        "execution_profile": execution_profile,
        "transaction_cost_bps": cost_bps,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "return_2022": _period_return(
            result.returns,
            "2022-01-01",
            "2022-12-31",
        ),
        "return_aug2023_jan2024": _period_return(
            result.returns,
            "2023-08-01",
            "2024-01-31",
        ),
        "average_risk_concentration": float(normalized.pow(2).sum(axis=1).fillna(0.0).mean()),
        "average_active_names": float(risk.gt(1e-8).sum(axis=1).mean()),
        "execution_failure": bool(metrics.max_drawdown < -0.30 or metrics.cagr < 0.18),
    }


def build_schedule_summary(
    metrics: pd.DataFrame,
    *,
    base_cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, cost), group in metrics.groupby(
        ["candidate", "transaction_cost_bps"],
        sort=False,
    ):
        configured = group[group["execution_profile"].eq(CONFIGURED_PROFILE)].iloc[0]
        rows.append(
            {
                "candidate": candidate,
                "research_phase": group["research_phase"].iloc[0],
                "transaction_cost_bps": cost,
                "is_base_cost": float(cost) == base_cost_bps,
                "configured_cagr": configured["cagr"],
                "configured_max_drawdown": configured["max_drawdown"],
                "configured_calmar": configured["calmar"],
                "median_execution_cagr": group["cagr"].median(),
                "worst_execution_cagr": group["cagr"].min(),
                "worst_execution_drawdown": group["max_drawdown"].min(),
                "execution_failure_count": int(group["execution_failure"].sum()),
                "median_turnover": group["average_turnover"].median(),
                "worst_2022_return": group["return_2022"].min(),
                "worst_aug2023_jan2024_return": group["return_aug2023_jan2024"].min(),
            }
        )
    return pd.DataFrame(rows)


def build_fold_metrics(
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    folds = (
        ("2008_2012", "2008-01-01", "2012-12-31"),
        ("2013_2017", "2013-01-01", "2017-12-31"),
        ("2018_2022", "2018-01-01", "2022-12-31"),
        ("2023_2026", "2023-01-01", "2026-12-31"),
    )
    reference = results["native_reference"]
    rows: list[dict[str, object]] = []
    for candidate, result in results.items():
        for fold, start, end in folds:
            base = _slice_metrics(reference.returns, start, end)
            test = _slice_metrics(result.returns, start, end)
            if not base:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "cagr": test["cagr"],
                    "max_drawdown": test["max_drawdown"],
                    "cagr_delta": test["cagr"] - base["cagr"],
                    "max_drawdown_delta": (test["max_drawdown"] - base["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_crisis_metrics(
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    reference = results["native_reference"]
    rows: list[dict[str, object]] = []
    for candidate, result in results.items():
        for crisis, start, end in CRISIS_WINDOWS:
            base = _slice_metrics(reference.returns, start, end)
            test = _slice_metrics(result.returns, start, end)
            if not base:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "crisis": crisis,
                    "return_delta": (test["cumulative_return"] - base["cumulative_return"]),
                    "max_drawdown_delta": (test["max_drawdown"] - base["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_block_bootstrap(
    results: dict[str, BacktestResult],
    *,
    simulations: int = BOOTSTRAP_SIMULATIONS,
    seed: int = 11_117,
) -> pd.DataFrame:
    reference = results["native_reference"].returns.fillna(0.0)
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for candidate, result in results.items():
        paired = pd.concat(
            [reference.rename("base"), result.returns.rename("candidate")],
            axis=1,
        ).dropna()
        values = paired.to_numpy(dtype=float)
        length = len(values)
        starts = np.arange(max(1, length - BLOCK_SESSIONS + 1))
        blocks_needed = int(np.ceil(length / BLOCK_SESSIONS))
        cagr_delta: list[float] = []
        drawdown_delta: list[float] = []
        for _ in range(simulations):
            chosen = generator.choice(starts, blocks_needed, replace=True)
            sampled = np.concatenate(
                [values[start : start + BLOCK_SESSIONS] for start in chosen],
                axis=0,
            )[:length]
            base_equity = np.cumprod(1.0 + sampled[:, 0])
            test_equity = np.cumprod(1.0 + sampled[:, 1])
            cagr_delta.append(
                float(test_equity[-1] ** (252.0 / length) - base_equity[-1] ** (252.0 / length))
            )
            base_dd = np.min(base_equity / np.maximum.accumulate(base_equity) - 1.0)
            test_dd = np.min(test_equity / np.maximum.accumulate(test_equity) - 1.0)
            drawdown_delta.append(float(test_dd - base_dd))
        cagr = np.asarray(cagr_delta)
        dd = np.asarray(drawdown_delta)
        rows.append(
            {
                "candidate": candidate,
                "simulations": simulations,
                "block_sessions": BLOCK_SESSIONS,
                "cagr_delta_p05": np.quantile(cagr, 0.05),
                "cagr_delta_p50": np.quantile(cagr, 0.50),
                "cagr_delta_p95": np.quantile(cagr, 0.95),
                "probability_cagr_delta_positive": np.mean(cagr > 0.0),
                "max_drawdown_delta_p05": np.quantile(dd, 0.05),
                "max_drawdown_delta_p50": np.quantile(dd, 0.50),
                "probability_drawdown_damage_over_1pp": np.mean(dd < -0.01),
            }
        )
    return pd.DataFrame(rows)


def build_promotion_gates(
    metrics: pd.DataFrame,
    schedule: pd.DataFrame,
    rolling: pd.DataFrame,
    calendar: pd.DataFrame,
    crises: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    *,
    eligible_candidates: set[str],
    base_cost_bps: float,
) -> pd.DataFrame:
    base_schedule = schedule[
        schedule["is_base_cost"] & schedule["candidate"].eq("native_reference")
    ].iloc[0]
    pbo_probability = (
        float(pbo_summary.iloc[0]["pbo_probability"]) if not pbo_summary.empty else np.nan
    )
    rows: list[dict[str, object]] = []
    for candidate in schedule["candidate"].unique():
        base_cost = schedule[schedule["candidate"].eq(candidate) & schedule["is_base_cost"]].iloc[0]
        stress = schedule[
            schedule["candidate"].eq(candidate) & schedule["transaction_cost_bps"].eq(20.0)
        ].iloc[0]
        stress_reference = schedule[
            schedule["candidate"].eq("native_reference") & schedule["transaction_cost_bps"].eq(20.0)
        ].iloc[0]
        rolling_candidate = rolling[rolling["name"].eq(candidate) & rolling["window"].eq("3y")]
        rolling_reference = rolling[
            rolling["name"].eq("native_reference") & rolling["window"].eq("3y")
        ]
        rolling_delta = rolling_candidate["cagr"].reset_index(drop=True) - rolling_reference[
            "cagr"
        ].reset_index(drop=True)
        candidate_calendar = calendar[calendar["name"].eq(candidate)]
        reference_calendar = calendar[calendar["name"].eq("native_reference")]
        year_delta = candidate_calendar.set_index("window")["total_return"].sub(
            reference_calendar.set_index("window")["total_return"],
            fill_value=np.nan,
        )
        crisis = crises[crises["candidate"].eq(candidate)]
        boot = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        gates = {
            "configured_cagr_noninferiority": float(base_cost["configured_cagr"])
            >= float(base_schedule["configured_cagr"]) - 0.005,
            "configured_drawdown_noninferiority": float(base_cost["configured_max_drawdown"])
            >= float(base_schedule["configured_max_drawdown"]) - 0.01,
            "median_execution_cagr_noninferiority": float(base_cost["median_execution_cagr"])
            >= float(base_schedule["median_execution_cagr"]) - 0.005,
            "worst_execution_drawdown_improvement": float(base_cost["worst_execution_drawdown"])
            >= float(base_schedule["worst_execution_drawdown"]) + 0.02,
            "execution_failures_fall": int(base_cost["execution_failure_count"])
            < int(base_schedule["execution_failure_count"]),
            "turnover_budget": float(base_cost["median_turnover"])
            <= 1.25 * float(base_schedule["median_turnover"]),
            "rolling_three_year_positive": (
                not rolling_delta.empty and float(rolling_delta.median()) > 0.0
            ),
            "calendar_year_breadth": float(year_delta.ge(0.0).mean()) >= 0.60,
            "crisis_drawdown_budget": (
                not crisis.empty and float(crisis["max_drawdown_delta"].ge(-0.015).mean()) >= 0.75
            ),
            "cost_stress_positive": float(stress["configured_cagr"])
            >= float(stress_reference["configured_cagr"]),
            "family_pbo": pd.notna(pbo_probability) and pbo_probability <= 0.25,
            "bootstrap_tradeoff": float(boot["probability_cagr_delta_positive"]) >= 0.70
            and float(boot["probability_drawdown_damage_over_1pp"]) <= 0.35,
        }
        passed = sum(bool(value) for value in gates.values())
        is_reference = candidate == "native_reference"
        eligible = candidate in eligible_candidates and not is_reference
        rows.append(
            {
                "candidate": candidate,
                **gates,
                "gates_passed": passed,
                "gates_total": len(gates),
                "failed_gates": ", ".join(gate for gate, value in gates.items() if not value),
                "retrospective_gate_passed": (eligible and all(gates.values())),
                "research_status": (
                    "reference"
                    if is_reference
                    else (
                        "post_initial_diagnostic"
                        if not eligible
                        else (
                            "prospective_shadow_candidate"
                            if all(gates.values())
                            else "research_only"
                        )
                    )
                ),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["retrospective_gate_passed", "gates_passed"],
        ascending=[False, False],
    )


def build_candidate_drawdown_artifacts(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    *,
    defensive_ticker: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    contributors: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    episodes = (
        ("worst_full_history", None, None),
        ("aug2023_jan2024", "2023-08-01", "2024-01-31"),
    )
    for candidate, result in results.items():
        for episode, start, end in episodes:
            attribution = build_drawdown_attribution(
                result,
                prices,
                defensive_ticker=defensive_ticker,
                start=start,
                end=end,
            )
            for frame, destination in (
                (attribution.summary, summaries),
                (attribution.contributors, contributors),
                (attribution.exposure_path, exposures),
            ):
                if frame.empty:
                    continue
                tagged = frame.copy()
                tagged.insert(0, "candidate", candidate)
                tagged.insert(1, "episode", episode)
                destination.append(tagged)
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(contributors, ignore_index=True),
        pd.concat(exposures, ignore_index=True),
    )


def write_selector_transition_outputs(
    output_dir: str | Path,
    *,
    config: BotConfig,
    prices: pd.DataFrame,
    specs: tuple[SelectorTransitionSpec, ...],
    profiles: tuple[tuple[str, ExecutionConfig], ...],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    summary_path = root / "summary.md"
    summary_path.write_text(
        build_summary(frames) + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    manifest = write_research_manifest(
        root,
        study="i111_selector_transition_repair",
        config=config,
        prices=prices,
        parameters={
            "strategy": DEFAULT_I111_NATIVE_CHALLENGER,
            "candidate_names": [spec.name for spec in specs],
            "candidate_phases": {spec.name: spec.research_phase for spec in specs},
            "execution_profiles": [name for name, _ in profiles],
            "pre_registered_candidates_fixed_before_initial_execution": True,
            "post_initial_ablations_promotion_eligible": False,
            "automatic_promotion_allowed": False,
            "allocation_authority": 0.0,
            "result": "no_candidate_passed",
            "decision": "retain_native_reference",
            "decision_reason": (
                "No pre-registered selector or transition candidate cleared "
                "the fixed retrospective promotion-like screen."
            ),
            "closed_hypotheses": [
                "static_15pct_spy_rsp_core_as_native_repair",
                "recovery_metered_entry_as_native_repair",
                "integrated_buffer_blend_core_recovery_replacement",
            ],
        },
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest
    return paths


def build_summary(frames: dict[str, pd.DataFrame]) -> str:
    gates = frames["promotion_gates"]
    schedule = frames["schedule_summary"]
    base = schedule[schedule["is_base_cost"]]
    candidates = gates[
        gates["candidate"].ne("native_reference")
        & gates["research_status"].ne("post_initial_diagnostic")
    ]
    top = candidates.iloc[0]
    top_metrics = base[base["candidate"].eq(top["candidate"])].iloc[0]
    reference = base[base["candidate"].eq("native_reference")].iloc[0]
    return "\n".join(
        [
            "# I111 Selector And Transition Repair",
            "",
            "Status: fixed-slate retrospective research; allocation authority is 0%.",
            "Decision: retain `native_reference`; no candidate passed.",
            "",
            f"- Candidates: {gates['candidate'].nunique()}.",
            f"- Execution profiles: {frames['candidate_metrics']['execution_profile'].nunique()}.",
            f"- Retrospective passes: {int(gates['retrospective_gate_passed'].sum())}.",
            f"- Closest promotion-eligible candidate: `{top['candidate']}`.",
            f"- Gates: {int(top['gates_passed'])}/{int(top['gates_total'])}.",
            f"- Failed gates: {top['failed_gates'] or 'none'}.",
            (
                "- Configured-path CAGR delta: "
                f"{float(top_metrics['configured_cagr'] - reference['configured_cagr']):.2%}."
            ),
            (
                "- Configured-path max-drawdown delta: "
                f"{float(top_metrics['configured_max_drawdown'] - reference['configured_max_drawdown']):.2%}."
            ),
            (
                "- Worst execution drawdown delta: "
                f"{float(top_metrics['worst_execution_drawdown'] - reference['worst_execution_drawdown']):.2%}."
            ),
        ]
    )


def _slice_metrics(
    returns: pd.Series,
    start: str,
    end: str,
) -> dict[str, float]:
    selected = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if selected.empty:
        return {}
    equity = (1.0 + selected).cumprod()
    return {
        "cagr": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }


def _period_return(
    returns: pd.Series,
    start: str,
    end: str,
) -> float:
    selected = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    return float((1.0 + selected).prod() - 1.0) if not selected.empty else np.nan
