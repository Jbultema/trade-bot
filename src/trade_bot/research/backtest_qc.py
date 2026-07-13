from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_QC_STRATEGY = "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145"
SEMICONDUCTOR_TICKERS = {"NVDA", "AVGO", "SMH", "SOXX", "TSM", "AMD", "ASML", "ARM"}
ETF_ONLY_TICKERS = {"QQQ", "SMH", "SOXX", "IGV", "SPY", "IWM", "VEA", "VWO", "ACWX", "BIL"}
POST_2012_NAMES = {"AVGO", "META", "PLTR", "ARM"}


@dataclass(frozen=True)
class BacktestQCGauntlet:
    strategy_name: str
    output_dir: Path
    artifacts: dict[str, Path]
    headline: pd.DataFrame
    readout: str


def run_backtest_qc_gauntlet(
    *,
    config: BotConfig,
    prices: pd.DataFrame,
    strategy_name: str = DEFAULT_QC_STRATEGY,
    output_dir: str | Path = "reports/backtest_qc",
    benchmark_tickers: tuple[str, ...] = ("SPY", "QQQ"),
) -> BacktestQCGauntlet:
    """Run structural and robustness checks around one configured strategy."""

    if strategy_name not in config.strategies:
        raise KeyError(f"Strategy not found in config: {strategy_name}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    strategy = config.strategies[strategy_name]
    strategy_prices = _strategy_prices(prices, strategy)

    base_result, base_metrics = _evaluate_strategy(
        strategy_name,
        strategy,
        strategy_prices,
        config.execution,
    )
    headline = pd.DataFrame([_metrics_row("base", base_metrics)])
    benchmark_metrics = _benchmark_metrics(prices, benchmark_tickers, config.execution)
    data_coverage = _data_coverage(prices, strategy, benchmark_tickers)
    causality = _future_perturbation_check(strategy_name, strategy, strategy_prices, config.execution)
    lag_stress = _lag_stress(strategy_name, strategy, strategy_prices, config.execution)
    cost_stress = _cost_stress(strategy_name, strategy, strategy_prices, config.execution)
    rebalance_stress = _rebalance_stress(strategy_name, strategy, strategy_prices, config.execution)
    universe_ablations = _universe_ablations(
        strategy_name,
        strategy,
        prices,
        config.execution,
        base_result,
    )
    parameter_neighborhood = _parameter_neighborhood(
        strategy_name,
        strategy,
        strategy_prices,
        config.execution,
    )
    subperiods = _subperiod_metrics(
        strategy_name,
        base_result,
        prices,
        benchmark_tickers,
        config.execution,
    )
    concentration = _contribution_concentration(base_result, strategy_prices)
    issues = _issue_flags(
        base=base_metrics,
        data_coverage=data_coverage,
        causality=causality,
        lag_stress=lag_stress,
        cost_stress=cost_stress,
        universe_ablations=universe_ablations,
        concentration=concentration,
    )

    frames = {
        "headline": headline,
        "benchmark_metrics": benchmark_metrics,
        "data_coverage": data_coverage,
        "causality": causality,
        "lag_stress": lag_stress,
        "cost_stress": cost_stress,
        "rebalance_stress": rebalance_stress,
        "universe_ablations": universe_ablations,
        "parameter_neighborhood": parameter_neighborhood,
        "subperiods": subperiods,
        "contribution_concentration": concentration,
        "issues": issues,
    }
    artifacts: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[name] = path

    readout = _markdown_readout(
        strategy_name=strategy_name,
        base=base_metrics,
        benchmark_metrics=benchmark_metrics,
        causality=causality,
        lag_stress=lag_stress,
        cost_stress=cost_stress,
        universe_ablations=universe_ablations,
        parameter_neighborhood=parameter_neighborhood,
        subperiods=subperiods,
        concentration=concentration,
        issues=issues,
    )
    readout_path = output / "summary.md"
    readout_path.write_text(readout, encoding="utf-8")
    artifacts["summary"] = readout_path
    return BacktestQCGauntlet(
        strategy_name=strategy_name,
        output_dir=output,
        artifacts=artifacts,
        headline=headline,
        readout=readout,
    )


def _evaluate_strategy(
    name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> tuple[BacktestResult, PerformanceMetrics]:
    target_weights = build_strategy_weights(prices, strategy)
    result = run_backtest(
        name,
        prices,
        target_weights,
        execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )
    metrics = calculate_metrics(
        name,
        result.returns,
        result.equity,
        result.turnover,
        result.transaction_costs,
    )
    return result, metrics


def _strategy_prices(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = list(dict.fromkeys([*strategy.tickers, *([strategy.defensive_ticker] if strategy.defensive_ticker else [])]))
    missing = [ticker for ticker in columns if ticker not in prices.columns]
    if missing:
        raise KeyError(f"Missing price columns for strategy: {missing}")
    frame = prices[columns].sort_index()
    valid_rows = frame.notna().any(axis=1)
    return frame.loc[valid_rows]


def _metrics_row(label: str, metrics: PerformanceMetrics, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "name": metrics.name,
        "start": metrics.start,
        "end": metrics.end,
        "years": metrics.years,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "calmar": metrics.calmar,
        "annualized_volatility": metrics.annualized_volatility,
        "average_turnover": metrics.average_turnover,
        "total_transaction_cost": metrics.total_transaction_cost,
        "final_equity": metrics.final_equity,
    }
    row.update(extra)
    return row


def _benchmark_metrics(
    prices: pd.DataFrame,
    benchmark_tickers: tuple[str, ...],
    execution: ExecutionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in benchmark_tickers:
        if ticker not in prices.columns:
            continue
        strategy = StrategyConfig(type="buy_hold", tickers=[ticker])
        result, metrics = _evaluate_strategy(
            ticker,
            strategy,
            prices[[ticker]].dropna(how="all"),
            execution,
        )
        rows.append(_metrics_row(ticker, metrics))
    return pd.DataFrame(rows)


def _data_coverage(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
    benchmark_tickers: tuple[str, ...],
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([*strategy.tickers, *([strategy.defensive_ticker] if strategy.defensive_ticker else []), *benchmark_tickers]))
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker not in prices:
            rows.append({"ticker": ticker, "available": False})
            continue
        series = prices[ticker].dropna()
        first = series.index.min() if not series.empty else None
        last = series.index.max() if not series.empty else None
        rows.append(
            {
                "ticker": ticker,
                "available": not series.empty,
                "first_valid_date": str(first.date()) if first is not None else "",
                "last_valid_date": str(last.date()) if last is not None else "",
                "observations": int(series.shape[0]),
                "starts_after_2007": bool(first is not None and first > pd.Timestamp("2007-12-31")),
                "starts_after_2012": bool(first is not None and first > pd.Timestamp("2012-12-31")),
            }
        )
    return pd.DataFrame(rows)


def _future_perturbation_check(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    cutoff = _causality_cutoff(prices.index)
    perturbed = prices.copy()
    post_mask = perturbed.index > cutoff
    if post_mask.any():
        for offset, column in enumerate(perturbed.columns, start=1):
            pre_values = perturbed.loc[~post_mask, column].dropna()
            anchor = float(pre_values.iloc[-1]) if not pre_values.empty else 100.0
            path = anchor * (1.0 + 0.0005 * offset) ** np.arange(1, int(post_mask.sum()) + 1)
            perturbed.loc[post_mask, column] = path
    original_result, _ = _evaluate_strategy(strategy_name, strategy, prices, execution)
    perturbed_result, _ = _evaluate_strategy(f"{strategy_name}_future_perturbed", strategy, perturbed, execution)
    pre = original_result.returns.index <= cutoff
    target_diff = _max_abs_frame_diff(
        original_result.target_weights.loc[pre],
        perturbed_result.target_weights.loc[pre],
    )
    execution_diff = _max_abs_frame_diff(
        original_result.weights.loc[pre],
        perturbed_result.weights.loc[pre],
    )
    return_diff = float((original_result.returns.loc[pre] - perturbed_result.returns.loc[pre]).abs().max())
    passed = target_diff <= 1e-12 and execution_diff <= 1e-12 and return_diff <= 1e-12
    return pd.DataFrame(
        [
            {
                "check": "future_price_perturbation",
                "cutoff": str(cutoff.date()),
                "target_weight_max_abs_diff_before_cutoff": target_diff,
                "execution_weight_max_abs_diff_before_cutoff": execution_diff,
                "return_max_abs_diff_before_cutoff": return_diff,
                "passed": passed,
            }
        ]
    )


def _causality_cutoff(index: pd.DatetimeIndex) -> pd.Timestamp:
    midpoint = index.min() + (index.max() - index.min()) / 2
    position = max(0, index.searchsorted(midpoint, side="right") - 1)
    return pd.Timestamp(index[position])


def _max_abs_frame_diff(left: pd.DataFrame, right: pd.DataFrame) -> float:
    columns = sorted(set(left.columns) | set(right.columns))
    diff = left.reindex(columns=columns, fill_value=0.0) - right.reindex(columns=columns, fill_value=0.0)
    if diff.empty:
        return 0.0
    return float(diff.abs().max().max())


def _lag_stress(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    rows = []
    for lag in [1, 2, 3, 5, 10]:
        _, metrics = _evaluate_strategy(
            f"{strategy_name}_lag_{lag}",
            strategy,
            prices,
            execution.model_copy(update={"signal_lag_days": lag}),
        )
        rows.append(_metrics_row(f"lag_{lag}", metrics, signal_lag_days=lag))
    return pd.DataFrame(rows)


def _cost_stress(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    rows = []
    for bps in [0, 1, 3, 5, 10, 25, 50]:
        _, metrics = _evaluate_strategy(
            f"{strategy_name}_cost_{bps}",
            strategy,
            prices,
            execution.model_copy(update={"transaction_cost_bps": float(bps)}),
        )
        rows.append(_metrics_row(f"cost_{bps}_bps", metrics, transaction_cost_bps=bps))
    return pd.DataFrame(rows)


def _rebalance_stress(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    rows = []
    for rebalance in ["D", "W-WED", "W-FRI", "M"]:
        _, metrics = _evaluate_strategy(
            f"{strategy_name}_rebalance_{rebalance}",
            strategy,
            prices,
            execution.model_copy(update={"rebalance": rebalance}),
        )
        rows.append(_metrics_row(f"rebalance_{rebalance}", metrics, rebalance=rebalance))
    return pd.DataFrame(rows)


def _universe_ablations(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
    base_result: BacktestResult,
) -> pd.DataFrame:
    contributions = _contribution_series(base_result, _strategy_prices(prices, strategy))
    top_contributor = str(contributions.index[0]) if not contributions.empty else ""
    ablations = {
        "base": strategy.tickers,
        f"no_top_contributor_{top_contributor}": [ticker for ticker in strategy.tickers if ticker != top_contributor],
        "no_nvda": [ticker for ticker in strategy.tickers if ticker != "NVDA"],
        "no_semiconductors": [ticker for ticker in strategy.tickers if ticker not in SEMICONDUCTOR_TICKERS],
        "etf_only": [ticker for ticker in strategy.tickers if ticker in ETF_ONLY_TICKERS],
        "no_post_2012_names": [ticker for ticker in strategy.tickers if ticker not in POST_2012_NAMES],
    }
    rows: list[dict[str, object]] = []
    for label, tickers in ablations.items():
        tickers = list(dict.fromkeys(tickers))
        if len(tickers) < 2:
            continue
        variant = strategy.model_copy(update={"tickers": tickers, "top_n": min(strategy.top_n, len(tickers))})
        variant_prices = _strategy_prices(prices, variant)
        _, metrics = _evaluate_strategy(f"{strategy_name}_{label}", variant, variant_prices, execution)
        rows.append(_metrics_row(label, metrics, ticker_count=len(tickers), tickers=",".join(tickers)))
    return pd.DataFrame(rows)


def _parameter_neighborhood(
    strategy_name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if strategy.volatility_target is not None:
        base_vol = float(strategy.volatility_target.annualized_volatility)
        for vol in sorted({0.16, 0.17, 0.18, base_vol, 0.19, 0.20, 0.22}):
            variant = strategy.model_copy(
                update={
                    "volatility_target": strategy.volatility_target.model_copy(
                        update={"annualized_volatility": vol}
                    )
                }
            )
            _, metrics = _evaluate_strategy(f"{strategy_name}_vol_{vol}", variant, prices, execution)
            rows.append(_metrics_row(f"vol_{vol:.3f}", metrics, parameter="volatility_target", value=vol))
    if strategy.drawdown_control is not None:
        guards = {-0.10, -0.12, float(strategy.drawdown_control.max_drawdown), -0.16, -0.18, -0.20}
        for guard in sorted(guards, reverse=True):
            drawdown_control = strategy.drawdown_control.model_copy(update={"max_drawdown": guard})
            variant = strategy.model_copy(update={"drawdown_control": drawdown_control})
            _, metrics = _evaluate_strategy(f"{strategy_name}_guard_{guard}", variant, prices, execution)
            rows.append(_metrics_row(f"guard_{guard:.3f}", metrics, parameter="drawdown_guard", value=guard))
    for lookback in [42, int(strategy.lookback_days), 84, 126]:
        for min_return in [0.015, float(strategy.min_return), 0.035]:
            variant = strategy.model_copy(
                update={"lookback_days": lookback, "min_return": min_return}
            )
            _, metrics = _evaluate_strategy(
                f"{strategy_name}_lookback_{lookback}_min_{min_return}",
                variant,
                prices,
                execution,
            )
            rows.append(
                _metrics_row(
                    f"lookback_{lookback}_min_{min_return:.3f}",
                    metrics,
                    parameter="lookback_min_return",
                    value=f"{lookback}:{min_return:.3f}",
                )
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["label"])


def _subperiod_metrics(
    strategy_name: str,
    base_result: BacktestResult,
    prices: pd.DataFrame,
    benchmark_tickers: tuple[str, ...],
    execution: ExecutionConfig,
) -> pd.DataFrame:
    periods = [
        ("2005_2009", "2005-01-01", "2009-12-31"),
        ("2010_2014", "2010-01-01", "2014-12-31"),
        ("2015_2019", "2015-01-01", "2019-12-31"),
        ("2020_2026", "2020-01-01", "2026-12-31"),
    ]
    benchmark_results: dict[str, BacktestResult] = {}
    for ticker in benchmark_tickers:
        if ticker in prices.columns:
            benchmark_results[ticker], _ = _evaluate_strategy(
                ticker,
                StrategyConfig(type="buy_hold", tickers=[ticker]),
                prices[[ticker]].dropna(how="all"),
                execution,
            )
    rows: list[dict[str, object]] = []
    for period, start, end in periods:
        for name, result in [(strategy_name, base_result), *benchmark_results.items()]:
            row = _period_metric_row(period, start, end, name, result)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def _period_metric_row(
    period: str,
    start: str,
    end: str,
    name: str,
    result: BacktestResult,
) -> dict[str, object] | None:
    mask = (result.returns.index >= pd.Timestamp(start)) & (result.returns.index <= pd.Timestamp(end))
    returns = result.returns.loc[mask]
    if returns.empty:
        return None
    equity = 100.0 * (1.0 + returns).cumprod()
    metrics = calculate_metrics(
        name,
        returns,
        equity,
        result.turnover.loc[returns.index],
        result.transaction_costs.loc[returns.index],
    )
    return _metrics_row(period, metrics, period=period, subject=name)


def _contribution_concentration(
    result: BacktestResult,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    contributions = _contribution_series(result, prices)
    average_weight = result.weights.mean().reindex(contributions.index).fillna(0.0)
    held_share = (result.weights > 0.01).mean().reindex(contributions.index).fillna(0.0)
    total_positive = float(contributions.clip(lower=0.0).sum())
    rows: list[dict[str, object]] = []
    for rank, (ticker, contribution) in enumerate(contributions.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "ticker": ticker,
                "gross_return_contribution_sum": float(contribution),
                "share_of_positive_contribution": (
                    float(contribution / total_positive) if total_positive > 0 and contribution > 0 else 0.0
                ),
                "average_weight": float(average_weight.get(ticker, 0.0)),
                "held_gt_1pct_share": float(held_share.get(ticker, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _contribution_series(result: BacktestResult, prices: pd.DataFrame) -> pd.Series:
    returns = prices.reindex(result.weights.index).ffill().pct_change(fill_method=None).fillna(0.0)
    columns = sorted(set(result.weights.columns) & set(returns.columns))
    if not columns:
        return pd.Series(dtype=float)
    return (result.weights[columns] * returns[columns]).sum(axis=0).sort_values(ascending=False)


def _issue_flags(
    *,
    base: PerformanceMetrics,
    data_coverage: pd.DataFrame,
    causality: pd.DataFrame,
    lag_stress: pd.DataFrame,
    cost_stress: pd.DataFrame,
    universe_ablations: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    _add_issue(
        rows,
        "causality_future_perturbation",
        "pass" if bool(causality.iloc[0]["passed"]) else "fail",
        "Future-price perturbation changed pre-cutoff targets/weights/returns." if not bool(causality.iloc[0]["passed"]) else "No future-price perturbation leakage detected.",
    )
    late_assets = data_coverage[data_coverage["starts_after_2012"].fillna(False)]
    _add_issue(
        rows,
        "late_asset_availability",
        "warning" if not late_assets.empty else "pass",
        f"{len(late_assets)} strategy/benchmark assets start after 2012; this raises universe-selection and live-availability questions.",
    )
    lag_2 = _row_by_label(lag_stress, "lag_2")
    _add_issue(
        rows,
        "extra_execution_lag",
        "warning" if lag_2 is not None and float(lag_2["cagr"]) < base.cagr - 0.02 else "pass",
        "CAGR decays by more than 2 percentage points under a two-day signal lag." if lag_2 is not None and float(lag_2["cagr"]) < base.cagr - 0.02 else "Two-day lag does not materially break the result.",
    )
    cost_25 = _row_by_label(cost_stress, "cost_25_bps")
    _add_issue(
        rows,
        "high_cost_sensitivity",
        "warning" if cost_25 is not None and float(cost_25["cagr"]) < base.cagr - 0.04 else "pass",
        "CAGR decays by more than 4 percentage points under 25 bps transaction costs." if cost_25 is not None and float(cost_25["cagr"]) < base.cagr - 0.04 else "25 bps transaction costs do not erase the result.",
    )
    no_semis = _row_by_label(universe_ablations, "no_semiconductors")
    _add_issue(
        rows,
        "semiconductor_universe_dependence",
        "warning" if no_semis is not None and float(no_semis["cagr"]) < base.cagr - 0.05 else "pass",
        "Removing semiconductor/AI leaders reduces CAGR by more than 5 percentage points." if no_semis is not None and float(no_semis["cagr"]) < base.cagr - 0.05 else "No-semis universe ablation remains close to base.",
    )
    top = concentration.iloc[0] if not concentration.empty else None
    _add_issue(
        rows,
        "top_contributor_concentration",
        "warning" if top is not None and float(top["share_of_positive_contribution"]) >= 0.25 else "pass",
        f"Top contributor {top['ticker']} accounts for {float(top['share_of_positive_contribution']):.1%} of positive gross contribution." if top is not None else "No concentration data.",
    )
    return pd.DataFrame(rows)


def _add_issue(rows: list[dict[str, object]], check: str, status: str, detail: str) -> None:
    rows.append({"check": check, "status": status, "detail": detail})


def _row_by_label(frame: pd.DataFrame, label: str) -> pd.Series | None:
    if frame.empty or "label" not in frame:
        return None
    match = frame[frame["label"] == label]
    if match.empty:
        return None
    return match.iloc[0]


def _markdown_readout(
    *,
    strategy_name: str,
    base: PerformanceMetrics,
    benchmark_metrics: pd.DataFrame,
    causality: pd.DataFrame,
    lag_stress: pd.DataFrame,
    cost_stress: pd.DataFrame,
    universe_ablations: pd.DataFrame,
    parameter_neighborhood: pd.DataFrame,
    subperiods: pd.DataFrame,
    concentration: pd.DataFrame,
    issues: pd.DataFrame,
) -> str:
    qqq = _metric_for_name(benchmark_metrics, "QQQ")
    spy = _metric_for_name(benchmark_metrics, "SPY")
    lag_2 = _row_by_label(lag_stress, "lag_2")
    cost_25 = _row_by_label(cost_stress, "cost_25_bps")
    no_semis = _row_by_label(universe_ablations, "no_semiconductors")
    etf_only = _row_by_label(universe_ablations, "etf_only")
    top = concentration.iloc[0] if not concentration.empty else None
    warning_count = int((issues["status"] == "warning").sum()) if not issues.empty else 0
    fail_count = int((issues["status"] == "fail").sum()) if not issues.empty else 0

    lines = [
        f"# Backtest QC Gauntlet: `{strategy_name}`",
        "",
        "## Headline",
        f"- Base result: CAGR {_pct(base.cagr)}, max drawdown {_pct(base.max_drawdown)}, Sharpe {base.sharpe:.2f}, average turnover {_pct(base.average_turnover)}.",
        f"- Causality perturbation: {'PASS' if bool(causality.iloc[0]['passed']) else 'FAIL'}.",
        f"- Issue flags: {fail_count} fail, {warning_count} warning.",
    ]
    if qqq is not None:
        lines.append(f"- QQQ benchmark: CAGR {_pct(float(qqq['cagr']))}, max drawdown {_pct(float(qqq['max_drawdown']))}.")
    if spy is not None:
        lines.append(f"- SPY benchmark: CAGR {_pct(float(spy['cagr']))}, max drawdown {_pct(float(spy['max_drawdown']))}.")
    lines.extend(
        [
            "",
            "## Stress Read",
            f"- Two-day signal lag: CAGR {_pct(float(lag_2['cagr']))}, max drawdown {_pct(float(lag_2['max_drawdown']))}." if lag_2 is not None else "- Two-day signal lag: unavailable.",
            f"- 25 bps cost stress: CAGR {_pct(float(cost_25['cagr']))}, max drawdown {_pct(float(cost_25['max_drawdown']))}." if cost_25 is not None else "- 25 bps cost stress: unavailable.",
            f"- No semiconductor/AI leader basket: CAGR {_pct(float(no_semis['cagr']))}, max drawdown {_pct(float(no_semis['max_drawdown']))}." if no_semis is not None else "- No semiconductor/AI leader basket: unavailable.",
            f"- ETF-only basket: CAGR {_pct(float(etf_only['cagr']))}, max drawdown {_pct(float(etf_only['max_drawdown']))}." if etf_only is not None else "- ETF-only basket: unavailable.",
        ]
    )
    if top is not None:
        lines.append(
            f"- Top contributor: {top['ticker']} contributed {float(top['gross_return_contribution_sum']):.1%} gross return-sum and {float(top['share_of_positive_contribution']):.1%} of positive contribution."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            _interpretation_sentence(fail_count=fail_count, warning_count=warning_count),
            "",
            "## Issue Flags",
        ]
    )
    for _, row in issues.iterrows():
        lines.append(f"- {row['status'].upper()} `{row['check']}`: {row['detail']}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `headline.csv`, `benchmark_metrics.csv`, `data_coverage.csv`, `causality.csv`, `lag_stress.csv`, `cost_stress.csv`, `rebalance_stress.csv`, `universe_ablations.csv`, `parameter_neighborhood.csv`, `subperiods.csv`, `contribution_concentration.csv`, `issues.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metric_for_name(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if frame.empty:
        return None
    match = frame[(frame["name"] == name) | (frame["label"] == name)]
    if match.empty:
        return None
    return match.iloc[0]


def _interpretation_sentence(*, fail_count: int, warning_count: int) -> str:
    if fail_count:
        return "Do not trust this backtest until failing structural checks are fixed."
    if warning_count >= 3:
        return "No direct leakage was detected, but the result needs heavy skepticism because multiple robustness warnings remain."
    if warning_count:
        return "No direct leakage was detected; treat the result as promising but not yet live-grade until warning items are understood."
    return "No direct leakage or major robustness warnings were detected in this gauntlet."


def _pct(value: float) -> str:
    return f"{value:.2%}"
