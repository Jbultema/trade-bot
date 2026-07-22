from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import (
    BotConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import daily_returns, unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.i111_orthogonal_search import DEFAULT_I111_NATIVE_CHALLENGER
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_I111_EXECUTION_HARDENING_OUTPUT_DIR = Path("reports/i111_execution_hardening")


@dataclass(frozen=True)
class ExecutionHardeningSpec:
    name: str
    description: str
    updates: dict[str, Any]


@dataclass(frozen=True)
class I111ExecutionHardeningResult:
    output_dir: Path
    execution_variant_metrics: pd.DataFrame
    mechanism_summary: pd.DataFrame
    component_decomposition: pd.DataFrame
    action_path_diagnostics: pd.DataFrame
    calendar_year_comparison: pd.DataFrame
    summary: str


def run_i111_execution_hardening(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_I111_EXECUTION_HARDENING_OUTPUT_DIR,
    refresh_data: bool = False,
) -> I111ExecutionHardeningResult:
    """Explain i111 execution fragility and test bounded native hardening mechanisms."""
    if DEFAULT_I111_NATIVE_CHALLENGER not in config.strategies:
        raise KeyError(
            f"Configured strategy {DEFAULT_I111_NATIVE_CHALLENGER!r} is required for this lab."
        )
    base = config.strategies[DEFAULT_I111_NATIVE_CHALLENGER]
    prices = _load_prices(config, base, refresh_data=refresh_data)
    executions = _execution_profiles(config.execution)
    specs = default_execution_hardening_specs()

    results: dict[tuple[str, str], BacktestResult] = {}
    metric_rows: list[dict[str, object]] = []
    for spec in specs:
        strategy = base.model_copy(update=spec.updates)
        target_weights = build_strategy_weights(prices, strategy)
        for execution_name, execution in executions:
            result = run_backtest(
                f"{spec.name}__{execution_name}",
                prices,
                target_weights,
                execution,
                volatility_target=strategy.volatility_target,
                drawdown_control=strategy.drawdown_control,
            )
            results[(spec.name, execution_name)] = result
            metric_rows.append(
                _metric_row(
                    result,
                    mechanism=spec.name,
                    description=spec.description,
                    execution_name=execution_name,
                )
            )
    execution_variant_metrics = pd.DataFrame(metric_rows)
    mechanism_summary = _mechanism_summary(execution_variant_metrics)

    component_decomposition = _component_decomposition(
        prices,
        base,
        config.execution,
        executions,
    )
    base_results = {
        execution_name: results[("native_reference", execution_name)]
        for execution_name, _ in executions
    }
    action_path_diagnostics = _action_path_diagnostics(prices, base_results)
    calendar_year_comparison = _calendar_year_comparison(base_results)
    summary = _summary_markdown(
        execution_variant_metrics,
        mechanism_summary,
        component_decomposition,
        action_path_diagnostics,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "execution_variant_metrics": execution_variant_metrics,
        "mechanism_summary": mechanism_summary,
        "component_decomposition": component_decomposition,
        "action_path_diagnostics": action_path_diagnostics,
        "calendar_year_comparison": calendar_year_comparison,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_research_manifest(
        output,
        study="i111_execution_hardening",
        config=config,
        prices=prices,
        parameters={
            "strategy": DEFAULT_I111_NATIVE_CHALLENGER,
            "refresh_data": refresh_data,
            "mechanisms": [spec.name for spec in specs],
            "execution_profiles": [name for name, _ in executions],
        },
        artifacts=[*[f"{name}.csv" for name in frames], "summary.md"],
    )
    return I111ExecutionHardeningResult(
        output_dir=output,
        execution_variant_metrics=execution_variant_metrics,
        mechanism_summary=mechanism_summary,
        component_decomposition=component_decomposition,
        action_path_diagnostics=action_path_diagnostics,
        calendar_year_comparison=calendar_year_comparison,
        summary=summary,
    )


def default_execution_hardening_specs() -> tuple[ExecutionHardeningSpec, ...]:
    return (
        ExecutionHardeningSpec(
            "native_reference",
            "Configured native challenger without V2.2 changes.",
            {},
        ),
        ExecutionHardeningSpec(
            "risk_sleeve_ai50_extreme",
            "Cap AI at 50% of the pre-repair active sleeve only at the existing extreme threshold.",
            {
                "risk_repair_ai_cap_basis": "risk_sleeve",
                "risk_repair_ai_soft_cap": 0.50,
                "risk_repair_ai_soft_threshold": 0.90,
                "risk_repair_ai_excess_destination": "defensive",
            },
        ),
        ExecutionHardeningSpec(
            "risk_sleeve_ai70_clustered",
            "Cap AI at 70% of active risk when at least seven eighths of stress evidence agrees.",
            {
                "risk_repair_ai_cap_basis": "risk_sleeve",
                "risk_repair_ai_soft_cap": 0.70,
                "risk_repair_ai_soft_threshold": 0.875,
                "risk_repair_ai_excess_destination": "defensive",
            },
        ),
        ExecutionHardeningSpec(
            "risk_sleeve_ai60_early",
            "Diagnostic earlier cap at 60% of active risk; expected to expose protection drag.",
            {
                "risk_repair_ai_cap_basis": "risk_sleeve",
                "risk_repair_ai_soft_cap": 0.60,
                "risk_repair_ai_soft_threshold": 0.75,
                "risk_repair_ai_excess_destination": "defensive",
            },
        ),
        ExecutionHardeningSpec(
            "weight_buffer08",
            "Ignore native target changes below 8% gross portfolio turnover.",
            {"risk_repair_min_rebalance_change": 0.08},
        ),
        ExecutionHardeningSpec(
            "hold5_step50",
            "Five-session hold with 50% maximum gross step and fast risk-off override.",
            {
                "risk_repair_min_rebalance_change": 0.04,
                "risk_repair_max_step_change": 0.50,
                "risk_repair_min_hold_days": 5,
            },
        ),
        ExecutionHardeningSpec(
            "hold10_step30",
            "Ten-session hold with 30% maximum gross step and fast risk-off override.",
            {
                "risk_repair_min_rebalance_change": 0.04,
                "risk_repair_max_step_change": 0.30,
                "risk_repair_min_hold_days": 10,
            },
        ),
    )


def _load_prices(
    config: BotConfig,
    strategy: StrategyConfig,
    *,
    refresh_data: bool,
) -> pd.DataFrame:
    tickers = set(required_strategy_tickers(strategy))
    tickers.update({"SPY", "QQQ", "RSP", "SMH", "BIL", "GLD", "TLT", "HYG", "LQD"})
    if strategy.defensive_ticker:
        tickers.add(strategy.defensive_ticker)
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


def _execution_profiles(
    execution: ExecutionConfig,
) -> tuple[tuple[str, ExecutionConfig], ...]:
    return (
        (
            "wednesday_lag1",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 1}),
        ),
        ("monday_lag1", execution.model_copy(update={"rebalance": "W-MON", "signal_lag_days": 1})),
        ("tuesday_lag1", execution.model_copy(update={"rebalance": "W-TUE", "signal_lag_days": 1})),
        (
            "thursday_lag1",
            execution.model_copy(update={"rebalance": "W-THU", "signal_lag_days": 1}),
        ),
        ("friday_lag1", execution.model_copy(update={"rebalance": "W-FRI", "signal_lag_days": 1})),
        ("daily_lag1", execution.model_copy(update={"rebalance": "D", "signal_lag_days": 1})),
        (
            "wednesday_lag2",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 2}),
        ),
        (
            "wednesday_lag5",
            execution.model_copy(update={"rebalance": "W-WED", "signal_lag_days": 5}),
        ),
    )


def _metric_row(
    result: BacktestResult,
    *,
    mechanism: str,
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
    ai_weight = (
        result.weights[ai_columns].sum(axis=1)
        if ai_columns
        else pd.Series(0.0, index=result.weights.index)
    )
    defensive = result.weights.get("BIL", pd.Series(0.0, index=result.weights.index))
    return {
        "mechanism": mechanism,
        "description": description,
        "execution": execution_name,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
        "average_ai_growth_weight": float(ai_weight.mean()),
        "average_defensive_weight": float(defensive.mean()),
        "return_2022": _period_return(result.returns, "2022-01-01", "2022-12-31"),
        "failure": bool(metrics.max_drawdown < -0.22 or metrics.cagr < 0.18),
    }


def _mechanism_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[metrics["mechanism"].eq("native_reference")]
    reference_worst_dd = float(reference["max_drawdown"].min())
    reference_wed = float(
        reference.loc[reference["execution"].eq("wednesday_lag1"), "cagr"].iloc[0]
    )
    rows: list[dict[str, object]] = []
    for mechanism, group in metrics.groupby("mechanism", sort=False, observed=True):
        wed = group[group["execution"].eq("wednesday_lag1")].iloc[0]
        worst_dd = float(group["max_drawdown"].min())
        failure_count = int(group["failure"].sum())
        preserves_edge = float(wed["cagr"]) >= reference_wed - 0.005
        improves_tail = worst_dd >= reference_worst_dd + 0.02
        rows.append(
            {
                "mechanism": mechanism,
                "wednesday_cagr": float(wed["cagr"]),
                "wednesday_max_drawdown": float(wed["max_drawdown"]),
                "median_execution_cagr": float(group["cagr"].median()),
                "worst_execution_drawdown": worst_dd,
                "execution_failure_count": failure_count,
                "average_turnover": float(group["average_turnover"].mean()),
                "preserves_reference_edge": preserves_edge,
                "improves_worst_execution_tail": improves_tail,
                "research_status": (
                    "promotion_like"
                    if preserves_edge
                    and improves_tail
                    and failure_count < int(reference["failure"].sum())
                    else "tradeoff_only" if improves_tail else "no_robust_improvement"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["research_status", "median_execution_cagr"],
        ascending=[True, False],
    )


def _component_decomposition(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
    base_execution: ExecutionConfig,
    executions: tuple[tuple[str, ExecutionConfig], ...],
) -> pd.DataFrame:
    target_weights = build_strategy_weights(prices, strategy)
    components: tuple[
        tuple[
            str,
            VolatilityTargetConfig | None,
            DrawdownControlConfig | None,
            float | None,
        ],
        ...,
    ] = (
        ("full", strategy.volatility_target, strategy.drawdown_control, None),
        ("without_transaction_costs", strategy.volatility_target, strategy.drawdown_control, 0.0),
        ("without_drawdown_guard", strategy.volatility_target, None, None),
        ("without_volatility_target", None, strategy.drawdown_control, None),
        ("raw_weights", None, None, None),
    )
    del base_execution
    rows: list[dict[str, object]] = []
    for execution_name, execution in executions:
        for component, volatility_target, drawdown_control, transaction_cost_bps in components:
            component_execution = execution
            if transaction_cost_bps is not None:
                component_execution = execution.model_copy(
                    update={"transaction_cost_bps": transaction_cost_bps}
                )
            result = run_backtest(
                f"{component}__{execution_name}",
                prices,
                target_weights,
                component_execution,
                volatility_target=volatility_target,
                drawdown_control=drawdown_control,
            )
            row = _metric_row(
                result,
                mechanism=component,
                description="Execution component decomposition.",
                execution_name=execution_name,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _action_path_diagnostics(
    prices: pd.DataFrame,
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    reference = results["wednesday_lag1"]
    asset_returns = daily_returns(prices)
    rows: list[dict[str, object]] = []
    for execution_name, result in results.items():
        drawdown = result.equity / result.equity.cummax() - 1.0
        trough = pd.Timestamp(drawdown.idxmin())
        pre_trough = result.equity.loc[result.equity.index <= trough]
        peak = pd.Timestamp(pre_trough.idxmax())
        post_trough = result.equity.loc[result.equity.index >= trough]
        recovery_rows = post_trough[post_trough >= result.equity.loc[peak]]
        recovery = recovery_rows.index[0] if not recovery_rows.empty else pd.NaT
        aligned = result.weights.reindex(columns=reference.weights.columns, fill_value=0.0)
        ref_aligned = reference.weights.reindex(columns=aligned.columns, fill_value=0.0)
        l1_distance = aligned.sub(ref_aligned).abs().sum(axis=1)
        selection_disagreement = aligned.gt(0.01).ne(ref_aligned.gt(0.01)).mean(axis=1)
        index = pd.to_datetime(aligned.index)
        mask_2022 = (index >= pd.Timestamp("2022-01-01")) & (index <= pd.Timestamp("2022-12-31"))
        aligned_2022 = aligned.loc[mask_2022]
        contributions_2022 = (
            aligned.mul(asset_returns.reindex(columns=aligned.columns)).loc[mask_2022].sum()
        )
        ai_columns = [column for column in aligned if column in AI_GROWTH_TICKERS]
        worst_contributor = contributions_2022.idxmin() if not contributions_2022.empty else ""
        rows.append(
            {
                "execution": execution_name,
                "worst_drawdown": float(drawdown.min()),
                "drawdown_peak": peak.date().isoformat(),
                "drawdown_trough": trough.date().isoformat(),
                "drawdown_recovery": (
                    recovery.date().isoformat() if pd.notna(recovery) else "unrecovered"
                ),
                "return_2022": _period_return(result.returns, "2022-01-01", "2022-12-31"),
                "average_ai_weight_2022": float(aligned_2022[ai_columns].sum(axis=1).mean()),
                "average_bil_weight_2022": float(aligned_2022["BIL"].mean()),
                "mean_l1_weight_distance_vs_wednesday": float(l1_distance.mean()),
                "selection_disagreement_vs_wednesday": float(selection_disagreement.mean()),
                "worst_2022_contributor": worst_contributor,
                "worst_2022_contribution": float(contributions_2022.min()),
            }
        )
    return pd.DataFrame(rows)


def _calendar_year_comparison(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for execution_name, result in results.items():
        for year, returns in result.returns.groupby(result.returns.index.year):
            rows.append(
                {
                    "execution": execution_name,
                    "year": int(year),
                    "total_return": float((1.0 + returns).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _period_return(returns: pd.Series, start: str, end: str) -> float:
    index = pd.to_datetime(returns.index)
    sliced = returns.loc[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return float((1.0 + sliced).prod() - 1.0) if not sliced.empty else 0.0


def _summary_markdown(
    metrics: pd.DataFrame,
    mechanisms: pd.DataFrame,
    decomposition: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    reference = metrics[metrics["mechanism"].eq("native_reference")].set_index("execution")
    wed = reference.loc["wednesday_lag1"]
    daily = reference.loc["daily_lag1"]
    monday = reference.loc["monday_lag1"]
    full = decomposition[decomposition["mechanism"].eq("full")].set_index("execution")
    no_guard = decomposition[decomposition["mechanism"].eq("without_drawdown_guard")].set_index(
        "execution"
    )
    no_cost = decomposition[decomposition["mechanism"].eq("without_transaction_costs")].set_index(
        "execution"
    )
    promotion_like = mechanisms[mechanisms["research_status"].eq("promotion_like")]
    path_tradeoffs = mechanisms[mechanisms["research_status"].eq("tradeoff_only")]
    diagnostic_2022 = diagnostics.set_index("execution")
    result_read = (
        "No V2.2 mechanism met the retrospective promotion-like screen. The configured "
        "Wednesday result remains the strongest historical path, but the edge does not survive "
        "ordinary execution shifts."
        if promotion_like.empty
        else (
            f"{len(promotion_like)} V2.2 mechanism(s) met the retrospective promotion-like "
            "screen. This is not automatic promotion; each requires prospective paper monitoring."
        )
    )
    lines = [
        "# I111 V2.2 Execution Hardening",
        "",
        "## Result",
        "",
        result_read,
        "",
        "## Native Reference",
        "",
        f"- Wednesday: {_pct(wed['cagr'])} CAGR / {_pct(wed['max_drawdown'])} max drawdown.",
        f"- Daily: {_pct(daily['cagr'])} / {_pct(daily['max_drawdown'])}.",
        f"- Monday: {_pct(monday['cagr'])} / {_pct(monday['max_drawdown'])}.",
        (
            f"- In 2022, Wednesday returned {_pct(diagnostic_2022.loc['wednesday_lag1', 'return_2022'])}; "
            f"daily returned {_pct(diagnostic_2022.loc['daily_lag1', 'return_2022'])}."
        ),
        "",
        "## Mechanism Attribution",
        "",
        (
            "- Removing the drawdown guard barely changes the daily max drawdown "
            f"({_pct(full.loc['daily_lag1', 'max_drawdown'])} full versus "
            f"{_pct(no_guard.loc['daily_lag1', 'max_drawdown'])} without the guard)."
        ),
        (
            "- Removing transaction costs changes daily CAGR from "
            f"{_pct(full.loc['daily_lag1', 'cagr'])} to "
            f"{_pct(no_cost.loc['daily_lag1', 'cagr'])}, while max drawdown changes from "
            f"{_pct(full.loc['daily_lag1', 'max_drawdown'])} to "
            f"{_pct(no_cost.loc['daily_lag1', 'max_drawdown'])}. Daily turnover therefore explains "
            "much of the return drag, but not the tail gap."
        ),
        (
            "- The drawdown fragility is driven mainly by which momentum names are held and when, "
            "rather than guard activation. Volatility targeting reduces loss size but does not "
            "remove weekday/path dependence."
        ),
        (
            "- The whole-portfolio AI cap was structurally weak in defensive periods. V2.2 added a "
            "risk-sleeve basis, but earlier/stronger use sacrificed return without consistently "
            "repairing the worst execution path."
        ),
        "",
        "## Candidate Read",
        "",
        f"- Promotion-like mechanisms: {len(promotion_like)}.",
        f"- Tail-improving tradeoffs that failed edge preservation: {len(path_tradeoffs)}.",
    ]
    for _, row in (
        mechanisms.sort_values("median_execution_cagr", ascending=False).head(7).iterrows()
    ):
        lines.append(
            f"- `{row['mechanism']}`: Wednesday {_pct(row['wednesday_cagr'])}, "
            f"worst execution DD {_pct(row['worst_execution_drawdown'])}, "
            f"median execution CAGR {_pct(row['median_execution_cagr'])}, "
            f"status `{row['research_status']}`."
        )
    lines.extend(
        [
            "",
            "## Durable Conclusion",
            "",
            (
                "Treat roughly 22% CAGR / -20% drawdown as a Wednesday-path result, not yet a robust "
                "live expectation. The strongest next tests are fixed-slate causal target smoothing "
                "and cross-sectional exit/replacement behavior during clustered AI stress. Earlier AI "
                "caps only exchange the return engine for a smaller but still fragile path."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: object) -> str:
    return f"{float(value):.2%}"  # type: ignore[arg-type]
