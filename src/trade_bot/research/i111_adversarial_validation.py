from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.backtest.windows import (
    calendar_year_metrics,
    regime_window_metrics,
    rolling_window_metrics,
    summarize_regimes,
    summarize_walk_forward,
    summarize_windows,
    walk_forward_holdout_metrics,
)
from trade_bot.config import (
    BotConfig,
    ExecutionConfig,
    StrategyConfig,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.backtest_pbo import PBOResult, estimate_probability_of_backtest_overfitting
from trade_bot.research.i111_candidates import build_i111_candidates
from trade_bot.research.i111_frontier_search import _ai_health_score, _crash_onset_score
from trade_bot.research.i111_orthogonal_search import (
    DEFAULT_I111_NATIVE_CHALLENGER,
    SIGNAL_TICKERS,
)
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_ADVERSARIAL_VALIDATION_OUTPUT_DIR = Path("reports/i111_adversarial_validation")
DEFAULT_ADVERSARIAL_START_DATES = (
    "2007-01-03",
    "2010-01-04",
    "2013-01-02",
    "2016-01-04",
    "2019-01-02",
    "2022-01-03",
)
DEFAULT_MONITOR_HORIZONS = (21, 63, 126)
DEFAULT_AI_BENCHMARK = "QQQ"
DEFAULT_BOOTSTRAP_PATHS = 2500
DEFAULT_BOOTSTRAP_HORIZON_DAYS = 252 * 5
DEFAULT_BOOTSTRAP_BLOCK_DAYS = 21
DEFAULT_BOOTSTRAP_RANDOM_SEED = 20260720


@dataclass(frozen=True)
class AdversarialStrategySpec:
    name: str
    source_group: str
    strategy: StrategyConfig


@dataclass(frozen=True)
class I111AdversarialValidationResult:
    output_dir: Path
    artifacts: dict[str, Path]
    strategy_metrics: pd.DataFrame
    robustness_summary: pd.DataFrame
    start_date_sensitivity: pd.DataFrame
    execution_sensitivity: pd.DataFrame
    ai_monitor_audit: pd.DataFrame
    overlay_metrics: pd.DataFrame
    candidate_pbo_summary: pd.DataFrame
    candidate_pbo_splits: pd.DataFrame
    candidate_pbo_selection: pd.DataFrame
    sequence_bootstrap: pd.DataFrame
    synthetic_ai_crash: pd.DataFrame
    research_artifact_audit: pd.DataFrame
    gap_audit: pd.DataFrame
    summary: str


def run_i111_adversarial_validation(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_ADVERSARIAL_VALIDATION_OUTPUT_DIR,
    refresh_data: bool = False,
    max_candidates: int | None = None,
) -> I111AdversarialValidationResult:
    """Run adversarial validation around the i111 / native risk-repair family."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = build_i111_adversarial_strategy_specs(config, max_candidates=max_candidates)
    tickers = _strategy_tickers(specs) | set(SIGNAL_TICKERS) | set(AI_GROWTH_TICKERS)
    tickers.update({"SPY", "QQQ", "RSP", "SMH", "BIL", "GLD", "TLT", "HYG", "LQD"})
    prices = load_or_fetch_yahoo_prices(
        sorted(tickers),
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    results = _run_strategy_results(config, specs, prices)
    strategy_metrics = _strategy_metrics(results)
    rolling = summarize_windows(rolling_window_metrics(results)).reset_index()
    walk_forward = summarize_walk_forward(walk_forward_holdout_metrics(results)).reset_index()
    calendar = calendar_year_metrics(results)
    regimes = regime_window_metrics(results)
    regime_summary = summarize_regimes(regimes).reset_index()
    start_date_sensitivity = _start_date_sensitivity(config, specs, prices, results=results)
    execution_sensitivity = _execution_sensitivity(config, specs, prices, strategy_metrics)
    robustness_summary = _robustness_summary(
        strategy_metrics, rolling, walk_forward, regime_summary
    )
    robustness_summary = _add_adversarial_context(
        robustness_summary,
        start_date_sensitivity,
        execution_sensitivity,
    )
    ai_monitor_audit = _ai_monitor_audit(prices)
    overlay_metrics = _overlay_metrics(config, results, prices)
    candidate_pbo = _candidate_pbo(results)
    sequence_bootstrap = _sequence_bootstrap(results)
    synthetic_ai_crash = _synthetic_ai_crash(results)
    research_artifact_audit = _research_artifact_audit(Path("reports"))
    gap_audit = _gap_audit(
        strategy_metrics=strategy_metrics,
        robustness_summary=robustness_summary,
        start_date_sensitivity=start_date_sensitivity,
        execution_sensitivity=execution_sensitivity,
        ai_monitor_audit=ai_monitor_audit,
        overlay_metrics=overlay_metrics,
        candidate_pbo_summary=candidate_pbo.summary,
        sequence_bootstrap=sequence_bootstrap,
        synthetic_ai_crash=synthetic_ai_crash,
        artifact_audit=research_artifact_audit,
    )
    summary = _summary_markdown(
        strategy_metrics=strategy_metrics,
        robustness_summary=robustness_summary,
        start_date_sensitivity=start_date_sensitivity,
        execution_sensitivity=execution_sensitivity,
        ai_monitor_audit=ai_monitor_audit,
        overlay_metrics=overlay_metrics,
        candidate_pbo_summary=candidate_pbo.summary,
        sequence_bootstrap=sequence_bootstrap,
        synthetic_ai_crash=synthetic_ai_crash,
        artifact_audit=research_artifact_audit,
        gap_audit=gap_audit,
    )

    frames = {
        "strategy_metrics": strategy_metrics,
        "robustness_summary": robustness_summary,
        "rolling_windows": rolling,
        "walk_forward": walk_forward,
        "calendar_years": calendar,
        "regime_windows": regimes,
        "regime_summary": regime_summary,
        "start_date_sensitivity": start_date_sensitivity,
        "execution_sensitivity": execution_sensitivity,
        "ai_monitor_audit": ai_monitor_audit,
        "overlay_metrics": overlay_metrics,
        "candidate_pbo_summary": candidate_pbo.summary,
        "candidate_pbo_splits": candidate_pbo.splits,
        "candidate_pbo_selection": candidate_pbo.strategy_selection,
        "candidate_pbo_stats": candidate_pbo.strategy_stats,
        "sequence_bootstrap": sequence_bootstrap,
        "synthetic_ai_crash": synthetic_ai_crash,
        "research_artifact_audit": research_artifact_audit,
        "gap_audit": gap_audit,
    }
    artifacts: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[name] = path
    summary_path = output / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    artifacts["summary"] = summary_path
    manifest_path = write_research_manifest(
        output,
        study="i111_adversarial_validation",
        config=config,
        prices=prices,
        parameters={
            "refresh_data": refresh_data,
            "max_candidates": max_candidates,
            "candidate_names": [spec.name for spec in specs],
            "start_dates": list(DEFAULT_ADVERSARIAL_START_DATES),
            "bootstrap_paths": DEFAULT_BOOTSTRAP_PATHS,
            "bootstrap_horizon_days": DEFAULT_BOOTSTRAP_HORIZON_DAYS,
            "bootstrap_block_days": DEFAULT_BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_random_seed": DEFAULT_BOOTSTRAP_RANDOM_SEED,
        },
        artifacts=[path.name for path in artifacts.values()],
    )
    artifacts["manifest"] = manifest_path

    return I111AdversarialValidationResult(
        output_dir=output,
        artifacts=artifacts,
        strategy_metrics=strategy_metrics,
        robustness_summary=robustness_summary,
        start_date_sensitivity=start_date_sensitivity,
        execution_sensitivity=execution_sensitivity,
        ai_monitor_audit=ai_monitor_audit,
        overlay_metrics=overlay_metrics,
        candidate_pbo_summary=candidate_pbo.summary,
        candidate_pbo_splits=candidate_pbo.splits,
        candidate_pbo_selection=candidate_pbo.strategy_selection,
        sequence_bootstrap=sequence_bootstrap,
        synthetic_ai_crash=synthetic_ai_crash,
        research_artifact_audit=research_artifact_audit,
        gap_audit=gap_audit,
        summary=summary,
    )


def build_i111_adversarial_strategy_specs(
    config: BotConfig,
    *,
    max_candidates: int | None = None,
) -> tuple[AdversarialStrategySpec, ...]:
    candidates = [
        AdversarialStrategySpec(candidate.name, candidate.source_group, candidate.strategy)
        for candidate in build_i111_candidates(config, include_upside_research=True)
    ]
    if DEFAULT_I111_NATIVE_CHALLENGER in config.strategies:
        native = AdversarialStrategySpec(
            DEFAULT_I111_NATIVE_CHALLENGER,
            "native_risk_repair_challenger",
            config.strategies[DEFAULT_I111_NATIVE_CHALLENGER],
        )
        candidates.append(native)
    deduped = {candidate.name: candidate for candidate in candidates}
    ordered = sorted(
        deduped.values(),
        key=lambda spec: (
            spec.name != DEFAULT_I111_NATIVE_CHALLENGER,
            spec.name != config.primary_strategy,
            spec.name,
        ),
    )
    if max_candidates is not None:
        ordered = ordered[:max_candidates]
    return tuple(ordered)


def _run_strategy_results(
    config: BotConfig,
    specs: tuple[AdversarialStrategySpec, ...],
    prices: pd.DataFrame,
) -> dict[str, BacktestResult]:
    return {
        spec.name: _run_strategy(config.execution, spec.name, spec.strategy, prices)
        for spec in specs
    }


def _run_strategy(
    execution: ExecutionConfig,
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
        execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )


def _strategy_prices(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = required_strategy_tickers(strategy)
    missing = unusable_required_price_columns(prices, columns)
    if missing:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing}")
    return prices[columns].sort_index().dropna(how="all")


def _strategy_metrics(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        metrics = _metrics(result)
        defensive = (
            result.weights["BIL"]
            if "BIL" in result.weights.columns
            else pd.Series(0.0, index=result.weights.index)
        )
        rows.append(
            {
                "result_name": name,
                "start": metrics.start,
                "end": metrics.end,
                "years": metrics.years,
                "cagr": metrics.cagr,
                "max_drawdown": metrics.max_drawdown,
                "calmar": metrics.calmar,
                "sharpe": metrics.sharpe,
                "annualized_volatility": metrics.annualized_volatility,
                "average_turnover": metrics.average_turnover,
                "total_transaction_cost": metrics.total_transaction_cost,
                "average_ai_growth_weight": _average_ai_weight(result.weights),
                "average_defensive_weight": float(defensive.mean()),
                "hard_defensive_day_rate": float((defensive >= 0.50).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("calmar", ascending=False)


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        result.name,
        result.returns,
        result.equity,
        result.turnover,
        result.transaction_costs,
    )


def _robustness_summary(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    walk_forward: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = metrics.copy()
    if not rolling.empty:
        for window in ("1y", "3y", "5y"):
            subset = rolling[rolling["window"].eq(window)][
                ["name", "worst_cagr", "positive_window_rate", "worst_drawdown"]
            ].rename(
                columns={
                    "name": "result_name",
                    "worst_cagr": f"worst_{window}_cagr",
                    "positive_window_rate": f"positive_{window}_rate",
                    "worst_drawdown": f"worst_{window}_drawdown",
                }
            )
            output = output.merge(subset, on="result_name", how="left")
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
    if not regime_summary.empty:
        output = output.merge(
            regime_summary[
                [
                    "name",
                    "tested_regimes",
                    "worst_regime_cagr",
                    "worst_regime_drawdown",
                    "left_tail_regime_cagr",
                    "left_tail_regime_return",
                    "transition_regime_hit_rate",
                ]
            ].rename(columns={"name": "result_name"}),
            on="result_name",
            how="left",
        )
    output["base_failure_count"] = _failure_count(output)
    output["adversarial_failure_count"] = output["base_failure_count"]
    output["adversarial_score"] = (
        output["cagr"]
        + 0.35 * output["calmar"]
        - (output["max_drawdown"].abs() - 0.205).clip(lower=0.0) * 2.0
        - output["adversarial_failure_count"] * 0.03
    )
    output["review_status"] = "base_only_pending_stress_context"
    return output.sort_values("adversarial_score", ascending=False)


def _add_adversarial_context(
    robustness: pd.DataFrame,
    start_date_sensitivity: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    output = robustness.copy()
    if not start_date_sensitivity.empty:
        start_summary = (
            start_date_sensitivity.groupby("result_name", observed=True)
            .agg(
                min_start_cagr=("cagr", "min"),
                max_start_cagr=("cagr", "max"),
                start_cagr_range=("cagr", lambda values: float(values.max() - values.min())),
                worst_start_drawdown=("max_drawdown", "min"),
            )
            .reset_index()
        )
        output = output.merge(start_summary, on="result_name", how="left")
    if not execution_sensitivity.empty:
        execution_summary = (
            execution_sensitivity.groupby("result_name", observed=True)
            .agg(
                execution_stress_rows=("stress", "count"),
                execution_failure_count=("failure", "sum"),
                worst_execution_cagr=("cagr", "min"),
                worst_execution_drawdown=("max_drawdown", "min"),
                worst_execution_delta_cagr=("delta_cagr", "min"),
                worst_execution_delta_drawdown=("delta_max_drawdown", "min"),
            )
            .reset_index()
        )
        execution_summary["execution_failure_rate"] = (
            execution_summary["execution_failure_count"]
            / execution_summary["execution_stress_rows"].replace(0, pd.NA)
        ).fillna(0.0)
        output = output.merge(execution_summary, on="result_name", how="left")
    output["adversarial_failure_count"] = _adversarial_failure_count(output)
    output["adversarial_score"] = (
        output["cagr"]
        + 0.35 * output["calmar"]
        - (output["max_drawdown"].abs() - 0.205).clip(lower=0.0) * 2.0
        - output["adversarial_failure_count"] * 0.035
    )
    output["review_status"] = output["adversarial_failure_count"].map(_review_status)
    return output.sort_values("adversarial_score", ascending=False)


def _adversarial_failure_count(frame: pd.DataFrame) -> pd.Series:
    base = _num(frame, "base_failure_count")
    execution_rate = _num(frame, "execution_failure_rate")
    checks = [
        execution_rate >= 0.75,
        (execution_rate > 0.0) & (execution_rate < 0.75),
        _num(frame, "worst_execution_drawdown") < -0.25,
        _num(frame, "worst_execution_cagr") < 0.18,
        _num(frame, "min_start_cagr") < 0.18,
        _num(frame, "start_cagr_range") > 0.12,
    ]
    return base + sum(check.astype(int) for check in checks)


def _review_status(failure_count: int | float) -> str:
    if failure_count <= 1:
        return "promotable_evidence"
    if failure_count <= 3:
        return "promising_but_fragile"
    return "research_only_until_execution_review"


def _failure_count(frame: pd.DataFrame) -> pd.Series:
    checks = [
        frame["max_drawdown"] < -0.205,
        _num(frame, "worst_3y_cagr") < 0.0,
        _num(frame, "worst_5y_cagr") < 0.0,
        _num(frame, "walk_forward_worst_cagr") < -0.08,
        _num(frame, "walk_forward_positive_rate") < 0.60,
        _num(frame, "worst_regime_drawdown") < -0.22,
        frame["average_ai_growth_weight"] > 0.70,
        frame["hard_defensive_day_rate"] > 0.22,
    ]
    return sum(check.astype(int) for check in checks)


def _start_date_sensitivity(
    config: BotConfig,
    specs: tuple[AdversarialStrategySpec, ...],
    prices: pd.DataFrame,
    *,
    results: dict[str, BacktestResult] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        full_result = results.get(spec.name) if results is not None else None
        if full_result is None:
            try:
                full_result = _run_strategy(config.execution, spec.name, spec.strategy, prices)
            except (KeyError, ValueError):
                continue
        for start in DEFAULT_ADVERSARIAL_START_DATES:
            sliced_result = _slice_and_rebase_result(
                full_result,
                start=pd.Timestamp(start),
                initial_capital=config.execution.initial_capital,
            )
            if len(sliced_result.returns) < 252:
                continue
            metrics = _metrics(sliced_result)
            rows.append(
                {
                    "result_name": spec.name,
                    "start_date": start,
                    "state_mode": "carried_state",
                    "cagr": metrics.cagr,
                    "max_drawdown": metrics.max_drawdown,
                    "calmar": metrics.calmar,
                    "average_turnover": metrics.average_turnover,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby("result_name", observed=True)
        .agg(
            tested_start_dates=("start_date", "count"),
            min_start_cagr=("cagr", "min"),
            median_start_cagr=("cagr", "median"),
            max_start_cagr=("cagr", "max"),
            worst_start_drawdown=("max_drawdown", "min"),
            median_start_drawdown=("max_drawdown", "median"),
        )
        .reset_index()
    )
    return frame.merge(summary, on="result_name", how="left")


def _slice_and_rebase_result(
    result: BacktestResult,
    *,
    start: pd.Timestamp,
    initial_capital: float,
) -> BacktestResult:
    """Slice performance while retaining strategy state computed before ``start``."""

    index = result.returns.index[result.returns.index >= start]
    returns = result.returns.reindex(index)
    equity = initial_capital * (1.0 + returns).cumprod()
    return BacktestResult(
        name=result.name,
        equity=equity.rename(result.name),
        returns=returns.rename(result.name),
        gross_returns=result.gross_returns.reindex(index).rename(result.name),
        weights=result.weights.reindex(index),
        target_weights=result.target_weights.reindex(index),
        turnover=result.turnover.reindex(index).rename(result.name),
        transaction_costs=result.transaction_costs.reindex(index).rename(result.name),
    )


def _execution_sensitivity(
    config: BotConfig,
    specs: tuple[AdversarialStrategySpec, ...],
    prices: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_metrics.set_index("result_name")
    variants = _execution_variants(config.execution)
    rows: list[dict[str, object]] = []
    for spec in specs:
        for label, execution in variants:
            try:
                result = _run_strategy(execution, spec.name, spec.strategy, prices)
            except (KeyError, ValueError):
                continue
            metrics = _metrics(result)
            base = baseline.loc[spec.name] if spec.name in baseline.index else None
            rows.append(
                {
                    "result_name": spec.name,
                    "stress": label,
                    "cagr": metrics.cagr,
                    "max_drawdown": metrics.max_drawdown,
                    "calmar": metrics.calmar,
                    "average_turnover": metrics.average_turnover,
                    "delta_cagr": metrics.cagr - float(base["cagr"]) if base is not None else 0.0,
                    "delta_max_drawdown": (
                        metrics.max_drawdown - float(base["max_drawdown"])
                        if base is not None
                        else 0.0
                    ),
                    "failure": bool(metrics.max_drawdown < -0.22 or metrics.cagr < 0.18),
                }
            )
    return pd.DataFrame(rows)


def _execution_variants(execution: ExecutionConfig) -> tuple[tuple[str, ExecutionConfig], ...]:
    base_cost = execution.transaction_cost_bps
    return (
        ("lag_2_days", execution.model_copy(update={"signal_lag_days": 2})),
        ("lag_5_days", execution.model_copy(update={"signal_lag_days": 5})),
        ("cost_3x", execution.model_copy(update={"transaction_cost_bps": base_cost * 3})),
        ("cost_5x", execution.model_copy(update={"transaction_cost_bps": base_cost * 5})),
        ("daily_rebalance", execution.model_copy(update={"rebalance": "D"})),
        ("friday_rebalance", execution.model_copy(update={"rebalance": "W-FRI"})),
        ("monday_rebalance", execution.model_copy(update={"rebalance": "W-MON"})),
    )


def _ai_monitor_audit(prices: pd.DataFrame) -> pd.DataFrame:
    benchmark = prices[DEFAULT_AI_BENCHMARK].dropna()
    if benchmark.empty:
        return pd.DataFrame()
    monitor_frame = pd.DataFrame(index=benchmark.index)
    monitor_frame["health_balanced"] = _ai_health_score(prices, "balanced").reindex(benchmark.index)
    monitor_frame["health_breadth_credit"] = _ai_health_score(prices, "breadth_credit").reindex(
        benchmark.index
    )
    monitor_frame["health_ai_pure"] = _ai_health_score(prices, "ai_pure").reindex(benchmark.index)
    monitor_frame["crash_balanced"] = _crash_onset_score(prices, "balanced").reindex(
        benchmark.index
    )
    monitor_frame["crash_breadth"] = _crash_onset_score(prices, "breadth_break").reindex(
        benchmark.index
    )
    monitors = {
        "ai_health_weak_balanced": monitor_frame["health_balanced"] < 0.55,
        "ai_health_weak_breadth_credit": monitor_frame["health_breadth_credit"] < 0.55,
        "ai_health_weak_pure": monitor_frame["health_ai_pure"] < 0.55,
        "crash_mesh_balanced": monitor_frame["crash_balanced"] >= 0.58,
        "crash_mesh_breadth": monitor_frame["crash_breadth"] >= 0.58,
        "confirmed_ai_break": (
            (monitor_frame["health_breadth_credit"] < 0.55)
            & (monitor_frame["crash_breadth"] >= 0.58)
        ),
    }
    rows: list[dict[str, object]] = []
    for horizon in DEFAULT_MONITOR_HORIZONS:
        forward_return = benchmark.shift(-horizon) / benchmark - 1.0
        forward_dd = _forward_drawdown(benchmark, horizon)
        eligible = forward_return.notna() & forward_dd.notna()
        eligible_days = int(eligible.sum())
        censored_days = int((~eligible).sum())
        severe = forward_dd <= -0.10
        benign = (forward_return > 0.0) & (forward_dd > -0.05)
        base_severe_rate = float(severe.loc[eligible].mean())
        for monitor, active in monitors.items():
            active = active.reindex(benchmark.index).fillna(False).astype(bool) & eligible
            count = int(active.sum())
            if count == 0:
                rows.append(
                    _empty_monitor_row(
                        monitor,
                        horizon,
                        base_severe_rate,
                        eligible_days=eligible_days,
                        censored_days=censored_days,
                    )
                )
                continue
            severe_rate = float(severe.loc[active].mean())
            false_positive_rate = float(benign.loc[active].mean())
            rows.append(
                {
                    "monitor": monitor,
                    "horizon_days": horizon,
                    "eligible_days": eligible_days,
                    "censored_days": censored_days,
                    "active_days": count,
                    "active_rate": count / eligible_days,
                    "avg_forward_return": float(forward_return.loc[active].mean()),
                    "avg_forward_drawdown": float(forward_dd.loc[active].mean()),
                    "severe_forward_drawdown_rate": severe_rate,
                    "base_severe_forward_drawdown_rate": base_severe_rate,
                    "severe_rate_lift": severe_rate - base_severe_rate,
                    "false_positive_rate": false_positive_rate,
                    "monitor_read": _monitor_read(
                        severe_rate, false_positive_rate, base_severe_rate
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["horizon_days", "severe_rate_lift"],
        ascending=[True, False],
    )


def _empty_monitor_row(
    monitor: str,
    horizon: int,
    base_severe_rate: float,
    *,
    eligible_days: int,
    censored_days: int,
) -> dict[str, object]:
    return {
        "monitor": monitor,
        "horizon_days": horizon,
        "eligible_days": eligible_days,
        "censored_days": censored_days,
        "active_days": 0,
        "active_rate": 0.0,
        "avg_forward_return": 0.0,
        "avg_forward_drawdown": 0.0,
        "severe_forward_drawdown_rate": 0.0,
        "base_severe_forward_drawdown_rate": base_severe_rate,
        "severe_rate_lift": -base_severe_rate,
        "false_positive_rate": 0.0,
        "monitor_read": "insufficient_events",
    }


def _monitor_read(severe_rate: float, false_positive_rate: float, base_rate: float) -> str:
    if severe_rate >= base_rate + 0.12 and false_positive_rate <= 0.45:
        return "useful_warning"
    if severe_rate >= base_rate + 0.05:
        return "noisy_warning"
    return "weak_or_noisy"


def _forward_drawdown(series: pd.Series, horizon: int) -> pd.Series:
    future_min = (
        series.shift(-1)
        .iloc[::-1]
        .rolling(
            horizon,
            min_periods=max(5, horizon // 4),
        )
        .min()
        .iloc[::-1]
    )
    return future_min / series - 1.0


def _overlay_metrics(
    config: BotConfig,
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    base_name = (
        DEFAULT_I111_NATIVE_CHALLENGER
        if DEFAULT_I111_NATIVE_CHALLENGER in results
        else next(iter(results), "")
    )
    if not base_name:
        return pd.DataFrame()
    base = results[base_name]
    signals = pd.DataFrame(index=base.weights.index)
    signals["health_breadth_credit"] = _ai_health_score(prices, "breadth_credit").reindex(
        signals.index
    )
    signals["crash_breadth"] = _crash_onset_score(prices, "breadth_break").reindex(signals.index)
    watch = (signals["health_breadth_credit"] < 0.55) | (signals["crash_breadth"] >= 0.58)
    confirmed = (signals["health_breadth_credit"] < 0.55) & (signals["crash_breadth"] >= 0.58)
    overlays = {
        "native_reference": base.weights,
        "static_bil_5": _shift_weight(base.weights, 0.05, {"BIL": 1.0}),
        "static_gld_tlt_10": _shift_weight(base.weights, 0.10, {"GLD": 0.5, "TLT": 0.5}),
        "ai_watch_bil_10": _conditional_shift(base.weights, watch, 0.10, {"BIL": 1.0}),
        "confirmed_ai_break_bil_20": _conditional_shift(
            base.weights, confirmed, 0.20, {"BIL": 1.0}
        ),
        "confirmed_ai_break_gld_tlt_20": _conditional_shift(
            base.weights,
            confirmed,
            0.20,
            {"GLD": 0.5, "TLT": 0.5},
        ),
        "crash_mesh_bil_20": _conditional_shift(
            base.weights,
            signals["crash_breadth"] >= 0.58,
            0.20,
            {"BIL": 1.0},
        ),
    }
    rows: list[dict[str, object]] = []
    for name, weights in overlays.items():
        result = _result_from_weights(
            name,
            prices,
            weights,
            config.execution,
        )
        metrics = _metrics(result)
        rows.append(
            {
                "overlay": name,
                "cagr": metrics.cagr,
                "max_drawdown": metrics.max_drawdown,
                "calmar": metrics.calmar,
                "average_turnover": metrics.average_turnover,
                "average_ai_growth_weight": _average_ai_weight(result.weights),
                "average_defensive_weight": (
                    float(result.weights["BIL"].mean()) if "BIL" in result.weights else 0.0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    base_row = frame[frame["overlay"].eq("native_reference")].iloc[0]
    frame["delta_cagr"] = frame["cagr"] - float(base_row["cagr"])
    frame["delta_max_drawdown"] = frame["max_drawdown"] - float(base_row["max_drawdown"])
    frame["overlay_status"] = frame.apply(_overlay_status, axis=1)
    return frame.sort_values("calmar", ascending=False)


def _shift_weight(
    weights: pd.DataFrame,
    shift: float,
    destinations: dict[str, float],
) -> pd.DataFrame:
    trigger = pd.Series(True, index=weights.index)
    return _conditional_shift(weights, trigger, shift, destinations)


def _conditional_shift(
    weights: pd.DataFrame,
    trigger: pd.Series,
    shift: float,
    destinations: dict[str, float],
) -> pd.DataFrame:
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    destination_mix = pd.Series(destinations, dtype=float)
    if destination_mix.empty or destination_mix.lt(0.0).any() or destination_mix.sum() <= 0.0:
        raise ValueError("destinations must contain positive, non-negative weights")
    destination_mix = destination_mix / destination_mix.sum()

    shifted = _normalize_weights(weights)
    trigger = trigger.reindex(shifted.index).shift(1).eq(True)
    for ticker in destination_mix.index:
        if ticker not in shifted.columns:
            shifted[ticker] = 0.0
    risk_columns = [column for column in shifted.columns if column not in destination_mix.index]
    if not risk_columns or shift == 0.0:
        return shifted

    risk_mass = shifted[risk_columns].sum(axis=1)
    transfer = risk_mass.clip(upper=shift).where(trigger, 0.0)
    scale = pd.Series(1.0, index=shifted.index)
    has_risk = risk_mass > 0.0
    scale.loc[has_risk] = (risk_mass.loc[has_risk] - transfer.loc[has_risk]).div(
        risk_mass.loc[has_risk]
    )
    shifted.loc[:, risk_columns] = shifted[risk_columns].mul(scale, axis=0)
    for ticker, weight in destination_mix.items():
        shifted.loc[trigger, ticker] = shifted.loc[trigger, ticker] + transfer.loc[trigger] * weight
    return _normalize_weights(shifted)


def _result_from_weights(
    name: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    execution: ExecutionConfig,
) -> BacktestResult:
    columns = [column for column in weights.columns if column in prices.columns]
    price_frame = prices[columns].reindex(weights.index).dropna(how="all")
    weight_frame = _normalize_weights(weights.reindex(price_frame.index).fillna(0.0))
    overlay_execution = execution.model_copy(update={"rebalance": "D", "signal_lag_days": 0})
    return run_backtest(
        name,
        price_frame,
        weight_frame,
        overlay_execution,
    )


def _overlay_status(row: pd.Series) -> str:
    if row["overlay"] == "native_reference":
        return "reference"
    if row["delta_max_drawdown"] > 0.005 and row["delta_cagr"] > -0.003:
        return "promising_insurance"
    if row["delta_max_drawdown"] > 0.0 and row["delta_cagr"] > -0.01:
        return "monitor_or_secondary"
    return "drag_or_no_help"


def _candidate_pbo(results: dict[str, BacktestResult]) -> PBOResult:
    returns = pd.DataFrame({name: result.returns for name, result in results.items()}).sort_index()
    returns = returns.dropna(axis=0, how="all").fillna(0.0)
    return estimate_probability_of_backtest_overfitting(
        returns,
        partitions=8,
        metric="sharpe",
    )


def _sequence_bootstrap(
    results: dict[str, BacktestResult],
    *,
    paths: int = DEFAULT_BOOTSTRAP_PATHS,
    horizon_days: int = DEFAULT_BOOTSTRAP_HORIZON_DAYS,
    block_days: int = DEFAULT_BOOTSTRAP_BLOCK_DAYS,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, (name, result) in enumerate(results.items()):
        returns = pd.to_numeric(result.returns, errors="coerce").dropna().to_numpy(dtype=float)
        if len(returns) < block_days:
            continue
        path_returns = _bootstrap_return_paths(
            returns,
            paths=paths,
            horizon_days=horizon_days,
            block_days=block_days,
            random_seed=random_seed + offset,
        )
        terminal = np.prod(1.0 + path_returns, axis=1) - 1.0
        annualized = (1.0 + terminal).clip(min=0.0) ** (252.0 / float(horizon_days)) - 1.0
        equity = np.cumprod(1.0 + path_returns, axis=1)
        running_max = np.maximum.accumulate(equity, axis=1)
        drawdowns = equity / running_max - 1.0
        max_drawdown = drawdowns.min(axis=1)
        rows.append(
            {
                "result_name": name,
                "paths": paths,
                "horizon_days": horizon_days,
                "block_days": block_days,
                "terminal_return_p05": float(np.quantile(terminal, 0.05)),
                "terminal_return_p50": float(np.quantile(terminal, 0.50)),
                "terminal_return_p95": float(np.quantile(terminal, 0.95)),
                "annualized_return_p05": float(np.quantile(annualized, 0.05)),
                "annualized_return_p50": float(np.quantile(annualized, 0.50)),
                "annualized_return_p95": float(np.quantile(annualized, 0.95)),
                "max_drawdown_p05": float(np.quantile(max_drawdown, 0.05)),
                "max_drawdown_p50": float(np.quantile(max_drawdown, 0.50)),
                "max_drawdown_p95": float(np.quantile(max_drawdown, 0.95)),
                "probability_negative_terminal": float((terminal < 0.0).mean()),
                "probability_breach_25dd": float((max_drawdown <= -0.25).mean()),
                "probability_breach_35dd": float((max_drawdown <= -0.35).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["probability_breach_25dd", "annualized_return_p05"],
        ascending=[True, False],
    )


def _bootstrap_return_paths(
    returns: np.ndarray,
    *,
    paths: int,
    horizon_days: int,
    block_days: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    starts = np.arange(0, max(len(returns) - block_days + 1, 1))
    output = np.empty((paths, horizon_days), dtype=float)
    for path in range(paths):
        sampled: list[np.ndarray] = []
        while sum(len(block) for block in sampled) < horizon_days:
            start = int(rng.choice(starts))
            sampled.append(returns[start : start + block_days])
        output[path] = np.concatenate(sampled)[:horizon_days]
    return output


def _synthetic_ai_crash(results: dict[str, BacktestResult]) -> pd.DataFrame:
    shock = _ai_crash_shock_vector()
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        weights = result.weights.copy()
        stress_returns = pd.Series(0.0, index=weights.columns)
        for ticker in weights.columns:
            stress_returns.loc[ticker] = shock.get(ticker, _default_shock(ticker))
        stress_loss = weights.mul(stress_returns, axis=1).sum(axis=1)
        current_weights = weights.iloc[-1]
        current_loss = float((current_weights * stress_returns.reindex(weights.columns)).sum())
        rows.append(
            {
                "result_name": name,
                "scenario": "ai_leadership_crash",
                "current_stress_return": current_loss,
                "median_historical_weight_stress_return": float(stress_loss.median()),
                "p05_historical_weight_stress_return": float(stress_loss.quantile(0.05)),
                "worst_historical_weight_stress_return": float(stress_loss.min()),
                "average_ai_growth_weight": _average_ai_weight(weights),
                "current_ai_growth_weight": _average_ai_weight(current_weights.to_frame().T),
                "average_defensive_weight": (
                    float(weights["BIL"].mean()) if "BIL" in weights.columns else 0.0
                ),
                "current_defensive_weight": (
                    float(current_weights.get("BIL", 0.0)) if not current_weights.empty else 0.0
                ),
                "stress_status": _stress_status(current_loss, float(stress_loss.quantile(0.05))),
            }
        )
    return pd.DataFrame(rows).sort_values("p05_historical_weight_stress_return", ascending=False)


def _ai_crash_shock_vector() -> dict[str, float]:
    return {
        "NVDA": -0.38,
        "AVGO": -0.34,
        "PLTR": -0.40,
        "SMH": -0.32,
        "SOXX": -0.32,
        "IGV": -0.26,
        "QQQ": -0.24,
        "MSFT": -0.22,
        "META": -0.26,
        "AMZN": -0.24,
        "SPY": -0.15,
        "RSP": -0.12,
        "IWM": -0.18,
        "VTI": -0.15,
        "GLD": 0.04,
        "TLT": 0.07,
        "BIL": 0.0,
    }


def _default_shock(ticker: str) -> float:
    if ticker in AI_GROWTH_TICKERS:
        return -0.28
    if ticker in {"GLD", "TLT", "BIL"}:
        return _ai_crash_shock_vector().get(ticker, 0.0)
    return -0.15


def _stress_status(current_loss: float, p05_loss: float) -> str:
    if current_loss <= -0.24 or p05_loss <= -0.24:
        return "high_ai_crash_exposure"
    if current_loss <= -0.18 or p05_loss <= -0.18:
        return "material_ai_crash_exposure"
    return "moderate_or_lower_exposure"


def _research_artifact_audit(report_root: Path) -> pd.DataFrame:
    expected = {
        "native_i111_risk_repair": (
            "summary.md",
            "strategy_metrics.csv",
            "walk_forward.csv",
            "manifest.json",
        ),
        "i111_orthogonal_search": (
            "summary.md",
            "strategy_metrics.csv",
            "walk_forward.csv",
            "manifest.json",
        ),
        "i111_frontier_search": (
            "summary.md",
            "strategy_metrics.csv",
            "checkpoint_summary.csv",
            "family_summary.csv",
            "manifest.json",
        ),
        "backtest_qc": (
            "summary.md",
            "headline.csv",
            "parameter_neighborhood.csv",
            "manifest.json",
        ),
        "pbo_diagnostics": ("summary.md", "pbo_summary.csv", "pbo_splits.csv"),
        "prebreak_hindsight": ("summary.md",),
        "defensive_signal_audit": ("summary.md",),
    }
    rows: list[dict[str, object]] = []
    for directory, files in expected.items():
        base = report_root / directory
        for file_name in files:
            path = base / file_name
            rows.append(
                {
                    "report": directory,
                    "artifact": file_name,
                    "exists": path.exists(),
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            )
    return pd.DataFrame(rows)


def _gap_audit(
    *,
    strategy_metrics: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    start_date_sensitivity: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
    ai_monitor_audit: pd.DataFrame,
    overlay_metrics: pd.DataFrame,
    candidate_pbo_summary: pd.DataFrame,
    sequence_bootstrap: pd.DataFrame,
    synthetic_ai_crash: pd.DataFrame,
    artifact_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    top = robustness_summary.iloc[0] if not robustness_summary.empty else pd.Series(dtype=object)
    ai_weight = float(top.get("average_ai_growth_weight", 0.0))
    rows.append(
        _gap(
            "single_engine_ai_dependency",
            "medium" if ai_weight < 0.68 else "high",
            f"Top candidate average AI/growth exposure is {ai_weight:.2%}.",
            "Keep AI concentration because it is the return engine, but monitor leadership health.",
        )
    )
    if not ai_monitor_audit.empty:
        useful = ai_monitor_audit[ai_monitor_audit["monitor_read"].eq("useful_warning")]
        rows.append(
            _gap(
                "ai_drawdown_monitor_quality",
                "medium" if useful.empty else "low",
                f"{len(useful)} monitor/horizon rows cleared the useful-warning read.",
                "Use monitors as warnings until event counts and false positives are reviewed.",
            )
        )
    fragile_starts = (
        start_date_sensitivity.groupby("result_name", observed=True)["min_start_cagr"].min() < 0.0
        if not start_date_sensitivity.empty
        else pd.Series(dtype=bool)
    )
    rows.append(
        _gap(
            "start_date_fragility",
            "high" if bool(fragile_starts.any()) else "low",
            (
                f"{int(fragile_starts.sum())} candidates had a negative minimum "
                "carried-state start-date CAGR."
            ),
            (
                "Prefer clusters that survive carried-state start-date shifts; do not promote "
                "single lucky starts."
            ),
        )
    )
    exec_failures = (
        int(execution_sensitivity["failure"].sum()) if not execution_sensitivity.empty else 0
    )
    rows.append(
        _gap(
            "execution_assumption_fragility",
            "high" if exec_failures else "low",
            f"{exec_failures} execution-stress rows breached the failure threshold.",
            "Review lag/cost/rebalance rows before moving any candidate toward live use.",
        )
    )
    insurance = (
        overlay_metrics[overlay_metrics["overlay_status"].eq("promising_insurance")]
        if not overlay_metrics.empty
        else pd.DataFrame()
    )
    rows.append(
        _gap(
            "hedge_overlay_evidence",
            "medium" if insurance.empty else "low",
            f"{len(insurance)} overlay rows qualified as promising insurance.",
            "Treat overlays as research-only unless implemented as native, human-executable rules.",
        )
    )
    pbo = (
        float(candidate_pbo_summary.iloc[0].get("pbo_probability", float("nan")))
        if not candidate_pbo_summary.empty
        else float("nan")
    )
    pbo_severity = "high" if pbo >= 0.50 else "medium" if pbo >= 0.25 else "low"
    rows.append(
        _gap(
            "candidate_selection_overfit_risk",
            pbo_severity,
            f"Candidate-family PBO probability is {pbo:.2%}.",
            "Prefer robust clusters over single best rows; rerun after adding new candidate families.",
        )
    )
    if not sequence_bootstrap.empty:
        worst_bootstrap = float(sequence_bootstrap["probability_breach_25dd"].max())
        rows.append(
            _gap(
                "sequence_risk",
                (
                    "high"
                    if worst_bootstrap >= 0.25
                    else "medium" if worst_bootstrap >= 0.10 else "low"
                ),
                f"Worst 5-year block-bootstrap probability of breaching -25% DD is {worst_bootstrap:.2%}.",
                "Use sequence-risk rows to judge path risk, not only realized historical max drawdown.",
            )
        )
    if not synthetic_ai_crash.empty:
        top_crash = synthetic_ai_crash.iloc[0]
        rows.append(
            _gap(
                "synthetic_ai_crash_exposure",
                (
                    "high"
                    if str(top_crash.get("stress_status")) == "high_ai_crash_exposure"
                    else "medium"
                ),
                (
                    "Best synthetic AI-crash p05 stress return is "
                    f"{float(top_crash.get('p05_historical_weight_stress_return')):.2%}."
                ),
                "Use synthetic AI-crash stress as a portfolio-risk monitor before live promotion.",
            )
        )
    else:
        rows.append(
            _gap(
                "synthetic_ai_crash_exposure",
                "high",
                "No synthetic AI-crash stress rows were produced.",
                "Add AI-crash stress rows before judging forward-looking AI risk.",
            )
        )
    missing_artifacts = int((~artifact_audit["exists"]).sum()) if not artifact_audit.empty else 0
    rows.append(
        _gap(
            "research_artifact_completeness",
            "medium" if missing_artifacts else "low",
            f"{missing_artifacts} expected prior-research artifacts are missing.",
            "Run missing QC/PBO/prebreak jobs before claiming full adversarial closure.",
        )
    )
    high_drawdown = int((strategy_metrics["max_drawdown"] < -0.205).sum())
    rows.append(
        _gap(
            "performance_frontier_tradeoff",
            "medium" if high_drawdown else "low",
            f"{high_drawdown} tested candidates breached the -20.5% max-DD review band.",
            "Higher-CAGR rows should be treated as risk-budget expansion, not improvement.",
        )
    )
    rows.append(
        _gap(
            "ai_event_scarcity",
            "high",
            "Recent history has limited clean AI-led bubble-break examples.",
            "Use QQQ/SMH/breadth/credit proxies, but label this as forward-looking uncertainty.",
        )
    )
    return pd.DataFrame(rows)


def _gap(name: str, severity: str, evidence: str, action: str) -> dict[str, object]:
    return {"gap": name, "severity": severity, "evidence": evidence, "recommended_action": action}


def _summary_markdown(
    *,
    strategy_metrics: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    start_date_sensitivity: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
    ai_monitor_audit: pd.DataFrame,
    overlay_metrics: pd.DataFrame,
    candidate_pbo_summary: pd.DataFrame,
    sequence_bootstrap: pd.DataFrame,
    synthetic_ai_crash: pd.DataFrame,
    artifact_audit: pd.DataFrame,
    gap_audit: pd.DataFrame,
) -> str:
    lines = [
        "# I111 Adversarial Validation",
        "",
        "## Purpose",
        "",
        (
            "Evaluate the current i111 / native risk-repair work adversarially: "
            "robustness, AI-drawdown monitors, portfolio/hedge variants, execution "
            "assumptions, candidate-selection overfit risk, sequence risk, synthetic "
            "AI-crash exposure, and missing evidence from prior research artifacts."
        ),
        "",
        "## Strategy Ranking",
        "",
        "| strategy | CAGR | max DD | Calmar | AI/growth wt | failures | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in robustness_summary.head(8).iterrows():
        lines.append(
            f"| `{row['result_name']}` | {row['cagr']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['calmar']:.2f} | {row['average_ai_growth_weight']:.2%} | "
            f"{int(row['adversarial_failure_count'])} | {row['review_status']} |"
        )
    lines.extend(["", "## Execution Stress Read", ""])
    if execution_sensitivity.empty:
        lines.append("No execution-stress rows were produced.")
    else:
        execution_summary = (
            execution_sensitivity.groupby("stress", observed=True)
            .agg(
                rows=("result_name", "count"),
                failures=("failure", "sum"),
                median_cagr=("cagr", "median"),
                worst_drawdown=("max_drawdown", "min"),
            )
            .reset_index()
            .sort_values(["failures", "worst_drawdown"], ascending=[False, True])
        )
        for _, row in execution_summary.iterrows():
            lines.append(
                f"- `{row['stress']}`: {int(row['failures'])}/{int(row['rows'])} failures, "
                f"median CAGR {row['median_cagr']:.2%}, worst max DD "
                f"{row['worst_drawdown']:.2%}."
            )
    lines.extend(["", "## AI Monitor Read", ""])
    if ai_monitor_audit.empty:
        lines.append("No AI monitor rows were produced.")
    else:
        useful = ai_monitor_audit[ai_monitor_audit["monitor_read"].eq("useful_warning")]
        lines.append(
            f"Useful warning rows: {len(useful)} of {len(ai_monitor_audit)} monitor/horizon rows."
        )
        monitor_display = ai_monitor_audit.copy()
        monitor_display["_read_rank"] = (
            monitor_display["monitor_read"]
            .map({"useful_warning": 0, "noisy_warning": 1, "weak_or_noisy": 2})
            .fillna(3)
        )
        monitor_display = monitor_display.sort_values(
            ["_read_rank", "severe_rate_lift"],
            ascending=[True, False],
        )
        for _, row in monitor_display.head(6).iterrows():
            lines.append(
                f"- `{row['monitor']}` {int(row['horizon_days'])}d: severe-rate lift "
                f"{row['severe_rate_lift']:.2%}, false-positive rate "
                f"{row['false_positive_rate']:.2%}, read `{row['monitor_read']}`."
            )
    lines.extend(["", "## Candidate PBO Read", ""])
    if candidate_pbo_summary.empty:
        lines.append("No candidate PBO summary was produced.")
    else:
        pbo = candidate_pbo_summary.iloc[0]
        lines.append(
            f"Candidate-family PBO probability: {float(pbo['pbo_probability']):.2%}; "
            f"OOS loss probability: {float(pbo['oos_loss_probability']):.2%}; "
            f"label `{pbo['pbo_label']}`."
        )
    lines.extend(["", "## Sequence Bootstrap Read", ""])
    if sequence_bootstrap.empty:
        lines.append("No sequence-bootstrap rows were produced.")
    else:
        for _, row in sequence_bootstrap.head(6).iterrows():
            lines.append(
                f"- `{row['result_name']}`: 5y annualized return p05 "
                f"{row['annualized_return_p05']:.2%}, median "
                f"{row['annualized_return_p50']:.2%}, -25% DD breach probability "
                f"{row['probability_breach_25dd']:.2%}."
            )
    lines.extend(["", "## Synthetic AI-Crash Stress", ""])
    if synthetic_ai_crash.empty:
        lines.append("No synthetic AI-crash stress rows were produced.")
    else:
        for _, row in synthetic_ai_crash.head(6).iterrows():
            lines.append(
                f"- `{row['result_name']}`: current stress return "
                f"{row['current_stress_return']:.2%}, p05 historical-weight stress "
                f"{row['p05_historical_weight_stress_return']:.2%}, status "
                f"`{row['stress_status']}`."
            )
    lines.extend(["", "## Overlay Read", ""])
    if overlay_metrics.empty:
        lines.append("No overlay rows were produced.")
    else:
        for _, row in overlay_metrics.head(7).iterrows():
            lines.append(
                f"- `{row['overlay']}`: CAGR {row['cagr']:.2%}, max DD "
                f"{row['max_drawdown']:.2%}, delta CAGR {row['delta_cagr']:.2%}, "
                f"delta max DD {row['delta_max_drawdown']:.2%}, status `{row['overlay_status']}`."
            )
    lines.extend(["", "## Adversarial Gaps", ""])
    for _, row in gap_audit.iterrows():
        lines.append(
            f"- **{row['severity']}** `{row['gap']}`: {row['evidence']} "
            f"Action: {row['recommended_action']}"
        )
    missing = int((~artifact_audit["exists"]).sum()) if not artifact_audit.empty else 0
    failed_exec = (
        int(execution_sensitivity["failure"].sum()) if not execution_sensitivity.empty else 0
    )
    start_rows = len(start_date_sensitivity)
    bootstrap_rows = len(sequence_bootstrap)
    crash_rows = len(synthetic_ai_crash)
    pbo_splits = (
        int(candidate_pbo_summary.iloc[0].get("valid_splits", 0))
        if not candidate_pbo_summary.empty
        else 0
    )
    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            (
                "This is an adversarial validation suite, not a promotion search. "
                f"It produced {start_rows} carried-state start-date rows, {failed_exec} "
                f"execution-stress failures, {pbo_splits} PBO splits, {bootstrap_rows} "
                f"sequence-bootstrap rows, {crash_rows} synthetic AI-crash rows, and "
                f"{missing} missing prior-artifact checks."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _strategy_tickers(specs: tuple[AdversarialStrategySpec, ...]) -> set[str]:
    tickers: set[str] = set()
    for spec in specs:
        tickers.update(required_strategy_tickers(spec.strategy))
    return tickers


def _average_ai_weight(weights: pd.DataFrame) -> float:
    columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    if not columns:
        return 0.0
    return float(weights[columns].sum(axis=1).mean())


def _normalize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    clipped = weights.clip(lower=0.0).fillna(0.0)
    row_sum = clipped.sum(axis=1)
    over = row_sum > 1.0
    clipped.loc[over] = clipped.loc[over].div(row_sum.loc[over], axis=0)
    return clipped.fillna(0.0)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)
