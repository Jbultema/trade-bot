from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.backtest.windows import calendar_year_metrics, rolling_window_metrics
from trade_bot.config import BotConfig, ExecutionConfig, required_strategy_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.backtest_pbo import (
    candidate_return_matrix,
    estimate_probability_of_backtest_overfitting,
)
from trade_bot.research.i111_orthogonal_search import DEFAULT_I111_NATIVE_CHALLENGER
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_EXECUTION_SMOOTHING_OUTPUT_DIR = Path("reports/i111_execution_smoothing")
SmoothingKind = Literal["raw", "ewm", "mean"]

MAX_SCHEDULE_CAGR_RANGE = 0.01
MIN_WORST_DRAWDOWN_IMPROVEMENT = 0.02
MAX_MEDIAN_CAGR_SACRIFICE = 0.005
MAX_WEDNESDAY_CAGR_SACRIFICE = 0.005
MAX_WEDNESDAY_DRAWDOWN_DEGRADATION = 0.01
MIN_COST_STRESS_CAGR_IMPROVEMENT = 0.01
MIN_ROLLING_THREE_YEAR_CAGR = 0.0
MAX_FAMILY_PBO = 0.25


@dataclass(frozen=True)
class ExecutionSmoothingSpec:
    name: str
    description: str
    kind: SmoothingKind
    span_or_window: int | None = None


@dataclass(frozen=True)
class I111ExecutionSmoothingResult:
    output_dir: Path
    candidate_metrics: pd.DataFrame
    schedule_summary: pd.DataFrame
    rolling_windows: pd.DataFrame
    calendar_years: pd.DataFrame
    promotion_gates: pd.DataFrame
    pbo_summary: pd.DataFrame
    pbo_splits: pd.DataFrame
    summary: str


def run_i111_execution_smoothing(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_EXECUTION_SMOOTHING_OUTPUT_DIR,
    refresh_data: bool = False,
) -> I111ExecutionSmoothingResult:
    """Run the fixed V2.3 causal smoothing slate without automatic promotion."""
    if DEFAULT_I111_NATIVE_CHALLENGER not in config.strategies:
        raise KeyError(
            f"Configured strategy {DEFAULT_I111_NATIVE_CHALLENGER!r} is required for this lab."
        )
    strategy = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]
    required_tickers = required_strategy_tickers(strategy)
    prices = load_or_fetch_yahoo_prices(
        required_tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()
    unusable = unusable_required_price_columns(prices, required_tickers)
    if unusable:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {unusable}")
    raw_weights = build_strategy_weights(prices, strategy)
    specs = default_execution_smoothing_specs()
    schedules = _weekday_executions(config.execution)
    cost_levels = tuple(dict.fromkeys([float(config.execution.transaction_cost_bps), 25.0]))

    results: dict[str, BacktestResult] = {}
    metric_rows: list[dict[str, object]] = []
    base_cost_results: dict[str, BacktestResult] = {}
    for spec in specs:
        target_weights = causal_smooth_weights(raw_weights, spec)
        for schedule_name, schedule_execution in schedules:
            for cost_bps in cost_levels:
                execution = schedule_execution.model_copy(update={"transaction_cost_bps": cost_bps})
                result_name = f"{spec.name}__{schedule_name}__{cost_bps:g}bps"
                result = run_backtest(
                    result_name,
                    prices,
                    target_weights,
                    execution,
                    volatility_target=strategy.volatility_target,
                    drawdown_control=strategy.drawdown_control,
                )
                results[result_name] = result
                metric_rows.append(
                    _metric_row(
                        result,
                        transform=spec.name,
                        description=spec.description,
                        schedule=schedule_name,
                        transaction_cost_bps=cost_bps,
                    )
                )
                if cost_bps == float(config.execution.transaction_cost_bps):
                    base_cost_results[f"{spec.name}__{schedule_name}"] = result

    candidate_metrics = pd.DataFrame(metric_rows)
    schedule_summary = _schedule_summary(
        candidate_metrics,
        base_transaction_cost_bps=float(config.execution.transaction_cost_bps),
    )
    rolling = rolling_window_metrics(
        base_cost_results,
        window_years=[1, 3, 5],
    )
    calendar = calendar_year_metrics(base_cost_results)
    return_matrix = candidate_return_matrix(base_cost_results, min_observations=252)
    pbo = estimate_probability_of_backtest_overfitting(
        return_matrix,
        partitions=8,
        metric="sharpe",
    )
    gates = _promotion_gates(schedule_summary, rolling, pbo.summary)
    summary = _summary_markdown(schedule_summary, gates, pbo.summary)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "candidate_metrics.csv": candidate_metrics,
        "schedule_summary.csv": schedule_summary,
        "rolling_windows.csv": rolling,
        "calendar_years.csv": calendar,
        "promotion_gates.csv": gates,
        "pbo_summary.csv": pbo.summary,
        "pbo_splits.csv": pbo.splits,
        "pbo_strategy_selection.csv": pbo.strategy_selection,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(output / filename, index=False)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output,
        study="i111_execution_smoothing_v2_3",
        config=config,
        prices=prices,
        parameters={
            "strategy": DEFAULT_I111_NATIVE_CHALLENGER,
            "candidate_set": [spec.name for spec in specs],
            "weekday_schedules": [name for name, _ in schedules],
            "cost_levels_bps": list(cost_levels),
            "refresh_data": refresh_data,
            "fixed_slate_evidence_gates": {
                "max_schedule_cagr_range": MAX_SCHEDULE_CAGR_RANGE,
                "min_worst_drawdown_improvement": MIN_WORST_DRAWDOWN_IMPROVEMENT,
                "max_median_cagr_sacrifice": MAX_MEDIAN_CAGR_SACRIFICE,
                "max_wednesday_cagr_sacrifice": MAX_WEDNESDAY_CAGR_SACRIFICE,
                "max_wednesday_drawdown_degradation": (MAX_WEDNESDAY_DRAWDOWN_DEGRADATION),
                "min_cost_stress_cagr_improvement": MIN_COST_STRESS_CAGR_IMPROVEMENT,
                "min_rolling_three_year_cagr": MIN_ROLLING_THREE_YEAR_CAGR,
                "max_family_pbo": MAX_FAMILY_PBO,
            },
        },
        artifacts=[*artifacts, "summary.md"],
    )
    return I111ExecutionSmoothingResult(
        output_dir=output,
        candidate_metrics=candidate_metrics,
        schedule_summary=schedule_summary,
        rolling_windows=rolling,
        calendar_years=calendar,
        promotion_gates=gates,
        pbo_summary=pbo.summary,
        pbo_splits=pbo.splits,
        summary=summary,
    )


def default_execution_smoothing_specs() -> tuple[ExecutionSmoothingSpec, ...]:
    """Return the fixed candidate slate; expanding it requires a new study version."""
    return (
        ExecutionSmoothingSpec(
            name="raw",
            description="Unmodified native target weights.",
            kind="raw",
        ),
        ExecutionSmoothingSpec(
            name="ewm5",
            description="Causal exponentially weighted target average with span five.",
            kind="ewm",
            span_or_window=5,
        ),
        ExecutionSmoothingSpec(
            name="mean10",
            description="Causal trailing ten-session target average.",
            kind="mean",
            span_or_window=10,
        ),
    )


def causal_smooth_weights(
    weights: pd.DataFrame,
    spec: ExecutionSmoothingSpec,
) -> pd.DataFrame:
    ordered = weights.sort_index().astype(float).clip(lower=0.0).fillna(0.0)
    if spec.kind == "raw":
        smoothed = ordered.copy()
    elif spec.kind == "ewm" and spec.span_or_window:
        smoothed = ordered.ewm(
            span=spec.span_or_window,
            adjust=False,
            min_periods=1,
        ).mean()
    elif spec.kind == "mean" and spec.span_or_window:
        smoothed = ordered.rolling(
            spec.span_or_window,
            min_periods=1,
        ).mean()
    else:
        raise ValueError(f"Invalid smoothing specification: {spec!r}")
    row_sum = smoothed.sum(axis=1)
    overinvested = row_sum > 1.0
    smoothed.loc[overinvested] = smoothed.loc[overinvested].div(
        row_sum.loc[overinvested],
        axis=0,
    )
    return smoothed.clip(lower=0.0).fillna(0.0)


def _weekday_executions(
    execution: ExecutionConfig,
) -> tuple[tuple[str, ExecutionConfig], ...]:
    return tuple(
        (
            weekday.lower(),
            execution.model_copy(update={"rebalance": f"W-{weekday}", "signal_lag_days": 1}),
        )
        for weekday in ("MON", "TUE", "WED", "THU", "FRI")
    )


def _metric_row(
    result: BacktestResult,
    *,
    transform: str,
    description: str,
    schedule: str,
    transaction_cost_bps: float,
) -> dict[str, object]:
    metrics = calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )
    return {
        "candidate": result.name,
        "transform": transform,
        "description": description,
        "schedule": schedule,
        "transaction_cost_bps": transaction_cost_bps,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "return_2022": _period_return(result.returns, "2022-01-01", "2022-12-31"),
    }


def _schedule_summary(
    metrics: pd.DataFrame,
    *,
    base_transaction_cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (transform, cost_bps), group in metrics.groupby(
        ["transform", "transaction_cost_bps"],
        sort=False,
        observed=True,
    ):
        wednesday = group[group["schedule"].eq("wed")]
        rows.append(
            {
                "transform": transform,
                "transaction_cost_bps": float(cost_bps),
                "is_base_cost": float(cost_bps) == base_transaction_cost_bps,
                "wednesday_cagr": _first_or_nan(wednesday, "cagr"),
                "wednesday_max_drawdown": _first_or_nan(wednesday, "max_drawdown"),
                "median_schedule_cagr": float(group["cagr"].median()),
                "worst_schedule_cagr": float(group["cagr"].min()),
                "schedule_cagr_range": float(group["cagr"].max() - group["cagr"].min()),
                "worst_schedule_drawdown": float(group["max_drawdown"].min()),
                "median_turnover": float(group["average_turnover"].median()),
                "worst_2022_return": float(group["return_2022"].min()),
            }
        )
    return pd.DataFrame(rows)


def _promotion_gates(
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    pbo_summary: pd.DataFrame,
) -> pd.DataFrame:
    base_rows = summary[summary["is_base_cost"]].set_index("transform")
    stress_rows = summary[~summary["is_base_cost"]].set_index("transform")
    raw_base = base_rows.loc["raw"]
    raw_stress = stress_rows.loc["raw"] if "raw" in stress_rows.index else raw_base
    pbo_value = (
        float(pbo_summary.iloc[0].get("pbo_probability", float("nan")))
        if not pbo_summary.empty
        else float("nan")
    )
    pbo_strategy_count = _pbo_dimension(pbo_summary, "strategy_count")
    pbo_split_count = _pbo_dimension(pbo_summary, "valid_splits")
    pbo_scope = f"family_{pbo_strategy_count}_strategies_{pbo_split_count}_splits"
    rows: list[dict[str, object]] = []
    for transform, candidate in base_rows.iterrows():
        candidate_stress = (
            stress_rows.loc[transform] if transform in stress_rows.index else candidate
        )
        three_year = rolling[
            rolling["name"].astype(str).str.startswith(f"{transform}__")
            & rolling["window"].eq("3y")
        ]
        worst_three_year_cagr = (
            float(three_year["cagr"].min()) if not three_year.empty else float("nan")
        )
        gates = {
            "edge_preservation": float(candidate["median_schedule_cagr"])
            >= float(raw_base["median_schedule_cagr"]) - MAX_MEDIAN_CAGR_SACRIFICE,
            "wednesday_edge_noninferiority": float(candidate["wednesday_cagr"])
            >= float(raw_base["wednesday_cagr"]) - MAX_WEDNESDAY_CAGR_SACRIFICE,
            "wednesday_tail_noninferiority": float(candidate["wednesday_max_drawdown"])
            >= float(raw_base["wednesday_max_drawdown"]) - MAX_WEDNESDAY_DRAWDOWN_DEGRADATION,
            "tail_improvement": transform == "raw"
            or float(candidate["worst_schedule_drawdown"])
            >= float(raw_base["worst_schedule_drawdown"]) + MIN_WORST_DRAWDOWN_IMPROVEMENT,
            "schedule_stability": float(candidate["schedule_cagr_range"])
            <= MAX_SCHEDULE_CAGR_RANGE,
            "cost_stress_improvement": transform == "raw"
            or float(candidate_stress["median_schedule_cagr"])
            >= float(raw_stress["median_schedule_cagr"]) + MIN_COST_STRESS_CAGR_IMPROVEMENT,
            "rolling_three_year_floor": pd.notna(worst_three_year_cagr)
            and worst_three_year_cagr >= MIN_ROLLING_THREE_YEAR_CAGR,
            "family_pbo_gate": pd.notna(pbo_value) and pbo_value <= MAX_FAMILY_PBO,
        }
        passed = sum(bool(value) for value in gates.values())
        retrospective_pass = all(gates.values())
        rows.append(
            {
                "transform": transform,
                **gates,
                "gates_passed": passed,
                "gates_total": len(gates),
                "worst_rolling_three_year_cagr": worst_three_year_cagr,
                "pbo_probability": pbo_value,
                "pbo_scope": pbo_scope,
                "retrospective_gate_pass": retrospective_pass,
                "promotion_eligible": False,
                "research_status": (
                    "prospective_challenger"
                    if transform != "raw" and retrospective_pass
                    else "reference" if transform == "raw" else "research_only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(
    schedule_summary: pd.DataFrame,
    gates: pd.DataFrame,
    pbo_summary: pd.DataFrame,
) -> str:
    base = schedule_summary[schedule_summary["is_base_cost"]].set_index("transform")
    stress = schedule_summary[~schedule_summary["is_base_cost"]].set_index("transform")
    pbo = (
        float(pbo_summary.iloc[0].get("pbo_probability", float("nan")))
        if not pbo_summary.empty
        else float("nan")
    )
    pbo_strategy_count = _pbo_dimension(pbo_summary, "strategy_count")
    pbo_split_count = _pbo_dimension(pbo_summary, "valid_splits")
    lines = [
        "# I111 V2.3 Fixed-Slate Execution Smoothing",
        "",
        "## Policy",
        "",
        (
            "This fixed three-candidate study is retrospective research only. It cannot promote a "
            "strategy automatically; a passing transform must enter prospective paper monitoring."
        ),
        "",
        "## Results",
        "",
    ]
    for transform, row in base.iterrows():
        stress_row = stress.loc[transform] if transform in stress.index else row
        gate_row = gates[gates["transform"].eq(transform)].iloc[0]
        lines.append(
            f"- `{transform}`: Wednesday {_pct(row['wednesday_cagr'])} CAGR / "
            f"{_pct(row['wednesday_max_drawdown'])} max drawdown; median weekday "
            f"{_pct(row['median_schedule_cagr'])}; worst weekday drawdown "
            f"{_pct(row['worst_schedule_drawdown'])}; CAGR range "
            f"{_pct(row['schedule_cagr_range'])}; stressed-cost median "
            f"{_pct(stress_row['median_schedule_cagr'])}; "
            f"status `{gate_row['research_status']}`."
        )
    lines.extend(
        [
            "",
            "## Evidence Gates",
            "",
            f"- Family-level CSCV PBO across {pbo_strategy_count} strategies and "
            f"{pbo_split_count} splits: {_pct(pbo)}.",
            (
                "- The eight retrospective gates require: preserve median CAGR within 0.5 points; "
                "keep configured-Wednesday CAGR within 0.5 points and max drawdown within one point "
                "of raw; improve worst-schedule drawdown by two points; keep weekday CAGR range "
                "within one point; improve 25 bps median CAGR by one point; avoid a negative worst "
                "rolling three-year CAGR; and keep family-level PBO at or below 25%. The same PBO "
                "gate applies to every row because it evaluates the full 15-strategy family, not an "
                "individual smoothing transform."
            ),
            (
                "- Passing all eight gates changes a non-raw row only to "
                "`prospective_challenger`; automatic promotion remains false until prospective "
                "monitoring is complete."
            ),
            "",
            "## Interpretation Boundary",
            "",
            (
                "This experiment tests execution robustness inside the current fixed universe. It does "
                "not address point-in-time universe selection, delistings, or the full historical trial "
                "count, so a clean result is evidence for paper monitoring rather than expected return."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _first_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame.iloc[0][column]) if not frame.empty else float("nan")


def _pbo_dimension(summary: pd.DataFrame, column: str) -> int:
    if summary.empty or column not in summary:
        return 0
    value = pd.to_numeric(pd.Series([summary.iloc[0].get(column)]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else 0


def _period_return(returns: pd.Series, start: str, end: str) -> float:
    index = pd.to_datetime(returns.index)
    sliced = returns.loc[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return float((1.0 + sliced).prod() - 1.0) if not sliced.empty else float("nan")


def _pct(value: object) -> str:
    return f"{float(value):.2%}" if pd.notna(value) else "n/a"  # type: ignore[arg-type]
