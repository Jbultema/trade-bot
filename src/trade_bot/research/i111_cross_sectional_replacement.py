from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig, required_strategy_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.i111_orthogonal_search import DEFAULT_I111_NATIVE_CHALLENGER
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import (
    _risk_repair_ai_stress_score,
    build_strategy_weights,
)

DEFAULT_I111_CROSS_SECTIONAL_REPLACEMENT_OUTPUT_DIR = Path(
    "reports/i111_cross_sectional_replacement"
)
CLUSTERED_AI_STRESS_THRESHOLD = 0.75


@dataclass(frozen=True)
class CrossSectionalReplacementSpec:
    name: str
    destination: Literal["none", "BIL", "RSP"]
    description: str


@dataclass(frozen=True)
class I111CrossSectionalReplacementResult:
    output_dir: Path
    execution_variant_metrics: pd.DataFrame
    policy_summary: pd.DataFrame
    replacement_path: pd.DataFrame
    calendar_year_comparison: pd.DataFrame
    summary: str


def fixed_cross_sectional_replacement_specs() -> tuple[CrossSectionalReplacementSpec, ...]:
    """Return the predeclared, non-swept replacement policies."""

    return (
        CrossSectionalReplacementSpec(
            name="native_reference",
            destination="none",
            description="Configured native challenger with no replacement transform.",
        ),
        CrossSectionalReplacementSpec(
            name="defer_ai_increases_to_bil",
            destination="BIL",
            description=(
                "During clustered AI stress, allow AI exits but redirect new or increased "
                "AI positions to BIL."
            ),
        ),
        CrossSectionalReplacementSpec(
            name="defer_ai_increases_to_rsp",
            destination="RSP",
            description=(
                "During clustered AI stress, allow AI exits but redirect new or increased "
                "AI positions to equal-weight US equities."
            ),
        ),
    )


def fixed_execution_profiles(
    execution: ExecutionConfig,
) -> tuple[tuple[str, ExecutionConfig], ...]:
    """Use the already-established execution stress slate without tuning it."""

    return (
        (
            "wednesday_lag1",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 1}),
        ),
        (
            "monday_lag1",
            execution.model_copy(update={"rebalance": "W-MON", "signal_lag_days": 1}),
        ),
        (
            "tuesday_lag1",
            execution.model_copy(update={"rebalance": "W-TUE", "signal_lag_days": 1}),
        ),
        (
            "thursday_lag1",
            execution.model_copy(update={"rebalance": "W-THU", "signal_lag_days": 1}),
        ),
        (
            "friday_lag1",
            execution.model_copy(update={"rebalance": "W-FRI", "signal_lag_days": 1}),
        ),
        (
            "daily_lag1",
            execution.model_copy(update={"rebalance": "D", "signal_lag_days": 1}),
        ),
        (
            "wednesday_lag2",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 2}),
        ),
        (
            "wednesday_lag5",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 5}),
        ),
    )


def apply_clustered_stress_replacement(
    raw_weights: pd.DataFrame,
    stress_score: pd.Series,
    *,
    destination: Literal["BIL", "RSP"],
    threshold: float = CLUSTERED_AI_STRESS_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Defer AI increases during clustered stress while preserving every AI reduction.

    The transform is sequential and causal. On a stress date, an AI name may not exceed
    its prior transformed target. Any blocked increase is redirected to the fixed
    destination. When stress is inactive, the raw strategy target is used unchanged.
    """

    if destination not in {"BIL", "RSP"}:
        raise ValueError("destination must be BIL or RSP")
    output = raw_weights.astype(float).clip(lower=0.0).fillna(0.0).copy()
    if destination not in output:
        output[destination] = 0.0
    ai_columns = [column for column in output if column in AI_GROWTH_TICKERS]
    aligned_stress = pd.to_numeric(stress_score, errors="coerce").reindex(output.index).fillna(0.0)
    diagnostic_rows: list[dict[str, object]] = []

    for position, market_date in enumerate(output.index):
        raw_target = output.loc[market_date].copy()
        transformed = raw_target.copy()
        is_clustered = bool(aligned_stress.loc[market_date] >= threshold)
        blocked_increase = 0.0
        if position > 0 and is_clustered and ai_columns:
            previous = output.iloc[position - 1]
            desired_ai = raw_target.loc[ai_columns]
            previous_ai = previous.loc[ai_columns]
            blocked = (desired_ai - previous_ai).clip(lower=0.0)
            blocked_increase = float(blocked.sum())
            transformed.loc[ai_columns] = desired_ai.where(blocked.eq(0.0), previous_ai)
            transformed.loc[destination] += blocked_increase
        row_sum = float(transformed.sum())
        if row_sum > 1.0 + 1e-12:
            transformed = transformed / row_sum
        output.loc[market_date] = transformed
        diagnostic_rows.append(
            {
                "market_date": pd.Timestamp(market_date).date().isoformat(),
                "stress_score": float(aligned_stress.loc[market_date]),
                "clustered_stress": is_clustered,
                "raw_ai_weight": float(raw_target.loc[ai_columns].sum()),
                "transformed_ai_weight": float(transformed.loc[ai_columns].sum()),
                "blocked_ai_increase": blocked_increase,
                "destination": destination,
                "destination_weight": float(transformed.loc[destination]),
            }
        )
    return output, pd.DataFrame(diagnostic_rows)


def run_i111_cross_sectional_replacement(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_CROSS_SECTIONAL_REPLACEMENT_OUTPUT_DIR,
    refresh_data: bool = False,
) -> I111CrossSectionalReplacementResult:
    """Run one fixed replacement experiment against the native i111 challenger."""

    if DEFAULT_I111_NATIVE_CHALLENGER not in config.strategies:
        raise KeyError(
            f"Configured strategy {DEFAULT_I111_NATIVE_CHALLENGER!r} is required for this lab."
        )
    strategy = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]
    prices = _load_prices(config, strategy, refresh_data=refresh_data)
    raw_weights = build_strategy_weights(prices, strategy)
    stress_score = _risk_repair_ai_stress_score(prices, strategy)
    specs = fixed_cross_sectional_replacement_specs()
    executions = fixed_execution_profiles(config.execution)

    target_weights: dict[str, pd.DataFrame] = {"native_reference": raw_weights}
    path_frames: list[pd.DataFrame] = []
    for spec in specs:
        if spec.destination == "none":
            continue
        transformed, path = apply_clustered_stress_replacement(
            raw_weights,
            stress_score,
            destination=spec.destination,
        )
        target_weights[spec.name] = transformed
        path.insert(0, "policy", spec.name)
        path_frames.append(path)

    results: dict[tuple[str, str], BacktestResult] = {}
    metric_rows: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    for spec in specs:
        for execution_name, execution in executions:
            result = run_backtest(
                f"{spec.name}__{execution_name}",
                prices,
                target_weights[spec.name],
                execution,
                volatility_target=strategy.volatility_target,
                drawdown_control=strategy.drawdown_control,
            )
            results[(spec.name, execution_name)] = result
            metric_rows.append(
                _metric_row(
                    result,
                    policy=spec.name,
                    description=spec.description,
                    execution_name=execution_name,
                )
            )
            calendar_rows.extend(_calendar_rows(result, spec.name, execution_name))

    execution_variant_metrics = pd.DataFrame(metric_rows)
    replacement_path = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    policy_summary = _policy_summary(execution_variant_metrics, replacement_path)
    calendar_year_comparison = pd.DataFrame(calendar_rows)
    summary = _summary_markdown(policy_summary, execution_variant_metrics, replacement_path)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "execution_variant_metrics": execution_variant_metrics,
        "policy_summary": policy_summary,
        "replacement_path": replacement_path,
        "calendar_year_comparison": calendar_year_comparison,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output,
        study="i111_cross_sectional_replacement",
        config=config,
        prices=prices,
        parameters={
            "strategy": DEFAULT_I111_NATIVE_CHALLENGER,
            "refresh_data": refresh_data,
            "candidate_names": [spec.name for spec in specs],
            "clustered_ai_stress_threshold": CLUSTERED_AI_STRESS_THRESHOLD,
            "stress_components_required": "6_of_8",
            "execution_profiles": [name for name, _ in executions],
            "policy_frozen_before_execution": True,
        },
        artifacts=[*[f"{name}.csv" for name in frames], "summary.md"],
    )
    return I111CrossSectionalReplacementResult(
        output_dir=output,
        execution_variant_metrics=execution_variant_metrics,
        policy_summary=policy_summary,
        replacement_path=replacement_path,
        calendar_year_comparison=calendar_year_comparison,
        summary=summary,
    )


def _load_prices(
    config: BotConfig,
    strategy: StrategyConfig,
    *,
    refresh_data: bool,
) -> pd.DataFrame:
    tickers = set(required_strategy_tickers(strategy))
    tickers.update({"SPY", "QQQ", "RSP", "SMH", "BIL", "GLD", "TLT", "HYG", "LQD"})
    prices = load_or_fetch_yahoo_prices(
        sorted(tickers),
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()
    unusable = unusable_required_price_columns(prices, tickers)
    if unusable:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {unusable}")
    return prices


def _metric_row(
    result: BacktestResult,
    *,
    policy: str,
    description: str,
    execution_name: str,
) -> dict[str, object]:
    metrics = calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )
    ai_columns = [column for column in result.weights if column in AI_GROWTH_TICKERS]
    ai_weight = result.weights[ai_columns].sum(axis=1)
    return {
        "policy": policy,
        "description": description,
        "execution": execution_name,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "average_ai_growth_weight": float(ai_weight.mean()),
        "return_2022": _period_return(result.returns, "2022-01-01", "2022-12-31"),
        "failure": bool(metrics.max_drawdown < -0.22 or metrics.cagr < 0.18),
    }


def _policy_summary(metrics: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[metrics["policy"].eq("native_reference")]
    reference_wed_cagr = float(
        reference.loc[reference["execution"].eq("wednesday_lag1"), "cagr"].iloc[0]
    )
    reference_worst_dd = float(reference["max_drawdown"].min())
    reference_failure_count = int(reference["failure"].sum())
    rows: list[dict[str, object]] = []
    for policy, group in metrics.groupby("policy", sort=False, observed=True):
        wed = group[group["execution"].eq("wednesday_lag1")].iloc[0]
        policy_path = path[path["policy"].eq(policy)] if not path.empty else pd.DataFrame()
        blocked_dates = (
            int(policy_path["blocked_ai_increase"].gt(1e-12).sum()) if not policy_path.empty else 0
        )
        blocked_weight = (
            float(policy_path["blocked_ai_increase"].sum()) if not policy_path.empty else 0.0
        )
        worst_dd = float(group["max_drawdown"].min())
        failure_count = int(group["failure"].sum())
        preserves_edge = float(wed["cagr"]) >= reference_wed_cagr - 0.005
        improves_tail = worst_dd >= reference_worst_dd + 0.02
        reduces_failures = failure_count < reference_failure_count
        qualifies = (
            policy != "native_reference"
            and blocked_dates > 0
            and preserves_edge
            and improves_tail
            and reduces_failures
        )
        if policy == "native_reference":
            status = "reference_only"
        elif qualifies:
            status = "candidate_for_prospective_test"
        elif improves_tail or reduces_failures:
            status = "tradeoff_only"
        else:
            status = "no_robust_improvement"
        rows.append(
            {
                "policy": policy,
                "wednesday_cagr": float(wed["cagr"]),
                "wednesday_max_drawdown": float(wed["max_drawdown"]),
                "median_execution_cagr": float(group["cagr"].median()),
                "worst_execution_drawdown": worst_dd,
                "execution_failure_count": failure_count,
                "average_turnover": float(group["average_turnover"].mean()),
                "blocked_replacement_dates": blocked_dates,
                "cumulative_blocked_target_weight": blocked_weight,
                "preserves_reference_edge": preserves_edge,
                "improves_worst_execution_tail": improves_tail,
                "reduces_execution_failures": reduces_failures,
                "research_status": status,
            }
        )
    return pd.DataFrame(rows)


def _calendar_rows(
    result: BacktestResult,
    policy: str,
    execution_name: str,
) -> list[dict[str, object]]:
    return [
        {
            "policy": policy,
            "execution": execution_name,
            "year": int(year),
            "total_return": float((1.0 + returns).prod() - 1.0),
        }
        for year, returns in result.returns.groupby(result.returns.index.year)
    ]


def _period_return(returns: pd.Series, start: str, end: str) -> float:
    index = pd.to_datetime(returns.index)
    sliced = returns.loc[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return float((1.0 + sliced).prod() - 1.0) if not sliced.empty else 0.0


def _summary_markdown(
    policies: pd.DataFrame,
    metrics: pd.DataFrame,
    path: pd.DataFrame,
) -> str:
    reference = policies[policies["policy"].eq("native_reference")].iloc[0]
    candidates = policies[policies["research_status"].eq("candidate_for_prospective_test")]
    clustered_dates = (
        int(path.loc[path["policy"].eq("defer_ai_increases_to_bil"), "clustered_stress"].sum())
        if not path.empty
        else 0
    )
    if candidates.empty:
        read = (
            "Neither fixed replacement policy cleared the predeclared preservation, tail, and "
            "execution-failure gates. The result does not justify changing the operating strategy."
        )
    else:
        read = (
            f"{len(candidates)} fixed replacement policy or policies cleared the retrospective "
            "screen. They remain research-only and require a newly frozen prospective test."
        )
    lines = [
        "# I111 Fixed Cross-Sectional Replacement Study",
        "",
        "## Result",
        "",
        read,
        "",
        "## Fixed Design",
        "",
        (
            "- The cutoff was fixed at 0.75 before execution: six of the native strategy's "
            "eight AI-stress components must agree."
        ),
        (
            "- AI reductions and exits remain immediate. Only a new or increased AI target is "
            "deferred, with the blocked weight sent to either BIL or RSP."
        ),
        "- No threshold, cap, hedge, or smoothing grid was run.",
        f"- Clustered-stress target dates observed: {clustered_dates:,}.",
        "",
        "## Reference",
        "",
        (
            f"- Wednesday CAGR {reference['wednesday_cagr']:.2%}; Wednesday max drawdown "
            f"{reference['wednesday_max_drawdown']:.2%}."
        ),
        (
            f"- Worst execution-profile drawdown {reference['worst_execution_drawdown']:.2%}; "
            f"execution failures {int(reference['execution_failure_count'])}."
        ),
        "",
        "## Policy Readout",
        "",
    ]
    for _, row in policies[~policies["policy"].eq("native_reference")].iterrows():
        lines.append(
            f"- `{row['policy']}`: Wednesday {row['wednesday_cagr']:.2%} CAGR / "
            f"{row['wednesday_max_drawdown']:.2%} drawdown; worst execution drawdown "
            f"{row['worst_execution_drawdown']:.2%}; {int(row['execution_failure_count'])} "
            f"failures; status `{row['research_status']}`."
        )
    lines.extend(
        [
            "",
            "## Sniff Test",
            "",
            (
                "This is retrospective fixed-policy evidence, not prospective proof. Its universe "
                "and delisting evidence remain promotion-blocking unless the manifest audit says "
                "otherwise. A favorable result can nominate a monitored challenger; it cannot "
                "replace the primary automatically."
            ),
            "",
            f"Execution rows evaluated: {len(metrics):,}.",
            "",
        ]
    )
    return "\n".join(lines)
