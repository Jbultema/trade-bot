from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import daily_returns, drawdown, moving_average, rolling_drawdown
from trade_bot.research.experiments import (
    _candidate_tickers,
    _load_previous_candidates,
    _load_previous_scorecards,
    _strategy_prices,
)
from trade_bot.research.prebreak_hindsight import _safe_float
from trade_bot.research.risk_landscape_survey import (
    AI_GROWTH_TICKERS,
    _credit_weak,
    _result_from_weights,
    _ticker_family,
)
from trade_bot.research.risk_policy_backtest import (
    _active_experiment_root,
    _run_candidate_backtest,
    _selected_candidates,
)

DEFAULT_AI_REPAIR_OUTPUT_DIR = Path("reports/ai_concentration_repair")
DEFAULT_REPAIR_TOP_N = 20
REPAIR_WINDOWS: dict[str, tuple[str, str]] = {
    "2011_2012_ai_growth_wound": ("2011-02-17", "2014-06-05"),
    "2021_2023_growth_rates_wound": ("2021-12-27", "2023-03-31"),
    "2024_2025_ai_liquidity_wound": ("2024-07-16", "2025-04-30"),
}
DESTINATION_WEIGHTS: dict[str, dict[str, float]] = {
    "bil": {"BIL": 1.0},
    "spy": {"SPY": 1.0},
    "spy_bil": {"SPY": 0.50, "BIL": 0.50},
    "bil_gld": {"BIL": 0.70, "GLD": 0.30},
    "bil_gld_tlt": {"BIL": 0.60, "GLD": 0.25, "TLT": 0.15},
}


@dataclass(frozen=True)
class AiRepairSpec:
    name: str
    stress_signal: str
    ai_cap: float
    destination: str


@dataclass(frozen=True)
class AiConcentrationRepairResult:
    strategy_metrics: pd.DataFrame
    variant_metrics: pd.DataFrame
    variant_summary: pd.DataFrame
    window_summary: pd.DataFrame
    summary: str


def run_ai_concentration_repair_lab(
    config: BotConfig,
    *,
    iteration: int = 164,
    top_n: int = DEFAULT_REPAIR_TOP_N,
    specs: tuple[AiRepairSpec, ...] | None = None,
    experiment_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_AI_REPAIR_OUTPUT_DIR,
    refresh_data: bool = False,
) -> AiConcentrationRepairResult:
    experiment_root = Path(experiment_root) if experiment_root else _active_experiment_root()
    candidates = _selected_candidates(
        iteration,
        scorecards=_load_previous_scorecards(experiment_root, iteration + 1),
        candidates_manifest=_load_previous_candidates(experiment_root, iteration + 1),
        top_n=top_n,
        experiment_root=experiment_root,
    )
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(candidates)
        | {"SPY", "QQQ", "SMH", "HYG", "LQD", "BIL", "GLD", "TLT"}
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )
    repair_specs = specs or _repair_specs()
    strategy_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_result = _run_candidate_backtest(config, candidate, prices, candidate_prices)
        base_metrics = _metrics(base_result)
        strategy_rows.append(
            {
                "strategy": candidate.name,
                "family": candidate.family,
                "phase": candidate.phase,
                "cagr": base_metrics.cagr,
                "max_drawdown": base_metrics.max_drawdown,
                "calmar": base_metrics.calmar,
                "average_ai_growth_weight": _average_ai_weight(base_result.weights),
            }
        )
        for window_name, (start, end) in REPAIR_WINDOWS.items():
            window_rows.append(
                _window_metric_row(
                    strategy=candidate.name,
                    variant_name="base",
                    window_name=window_name,
                    base_result=base_result,
                    variant_result=base_result,
                    start=start,
                    end=end,
                )
            )
        for spec in repair_specs:
            variant_result = apply_ai_repair_variant(
                base_result,
                prices,
                spec,
                transaction_cost_bps=config.execution.transaction_cost_bps,
            )
            variant_metrics = _metrics(variant_result)
            metric_rows.append(
                _variant_metric_row(
                    strategy=candidate.name,
                    family=candidate.family,
                    spec=spec,
                    base_result=base_result,
                    variant_result=variant_result,
                    base_metrics=base_metrics,
                    variant_metrics=variant_metrics,
                    prices=prices,
                )
            )
            for window_name, (start, end) in REPAIR_WINDOWS.items():
                window_rows.append(
                    _window_metric_row(
                        strategy=candidate.name,
                        variant_name=spec.name,
                        window_name=window_name,
                        base_result=base_result,
                        variant_result=variant_result,
                        start=start,
                        end=end,
                    )
                )
    strategy_metrics = pd.DataFrame(strategy_rows)
    variant_metrics = pd.DataFrame(metric_rows)
    window_summary = summarize_window_metrics(pd.DataFrame(window_rows))
    variant_summary = summarize_variant_metrics(variant_metrics, window_summary)
    summary = build_ai_repair_summary(strategy_metrics, variant_summary, window_summary)
    result = AiConcentrationRepairResult(
        strategy_metrics=strategy_metrics,
        variant_metrics=variant_metrics,
        variant_summary=variant_summary,
        window_summary=window_summary,
        summary=summary,
    )
    write_ai_repair_outputs(result, output_dir=output_dir)
    return result


def apply_ai_repair_variant(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    spec: AiRepairSpec,
    *,
    transaction_cost_bps: float,
) -> BacktestResult:
    aligned_prices = prices.reindex(base_result.weights.index).ffill()
    weights = base_result.weights.reindex(aligned_prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=aligned_prices.columns, fill_value=0.0)
    ai_columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    if not ai_columns:
        return base_result
    stress = _stress_signal(spec.stress_signal, aligned_prices, base_result)
    adjusted = weights.copy()
    ai_weight = adjusted[ai_columns].sum(axis=1)
    excess = (ai_weight - spec.ai_cap).clip(lower=0.0)
    reduction = pd.Series(0.0, index=adjusted.index)
    reduction.loc[stress] = excess.loc[stress]
    scale = ((ai_weight - reduction) / ai_weight.where(ai_weight.ne(0.0))).fillna(1.0).astype(float)
    adjusted.loc[:, ai_columns] = adjusted[ai_columns].mul(scale, axis=0)
    for ticker, weight in DESTINATION_WEIGHTS[spec.destination].items():
        if ticker in adjusted:
            adjusted[ticker] = adjusted[ticker] + reduction * weight
    return _result_from_weights(
        base_result,
        aligned_prices,
        adjusted,
        transaction_cost_bps,
        f"{base_result.name}__{spec.name}",
    )


def summarize_variant_metrics(
    variant_metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    if variant_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    wound_window = window_summary[
        window_summary["window_name"].eq("2011_2012_ai_growth_wound")
        & window_summary["variant_name"].ne("base")
    ]
    for variant_name, group in variant_metrics.groupby("variant_name", sort=False):
        wound_group = wound_window[wound_window["variant_name"].eq(variant_name)]
        rows.append(
            {
                "variant_name": variant_name,
                "stress_signal": str(group["stress_signal"].iloc[0]),
                "ai_cap": _safe_float(group["ai_cap"].iloc[0]),
                "destination": str(group["destination"].iloc[0]),
                "strategies": len(group),
                "median_cagr": pd.to_numeric(group["cagr"], errors="coerce").median(),
                "median_max_drawdown": pd.to_numeric(
                    group["max_drawdown"],
                    errors="coerce",
                ).median(),
                "median_calmar": pd.to_numeric(group["calmar"], errors="coerce").median(),
                "median_delta_cagr": pd.to_numeric(
                    group["delta_cagr_vs_base"],
                    errors="coerce",
                ).median(),
                "median_delta_max_drawdown": pd.to_numeric(
                    group["delta_max_drawdown_vs_base"],
                    errors="coerce",
                ).median(),
                "median_active_day_rate": pd.to_numeric(
                    group["active_day_rate"],
                    errors="coerce",
                ).median(),
                "median_2011_drawdown_delta": pd.to_numeric(
                    wound_group["median_delta_window_max_drawdown_vs_base"],
                    errors="coerce",
                ).median(),
                "cagr_win_rate": float((group["delta_cagr_vs_base"] > 0).mean()),
                "drawdown_win_rate": float((group["delta_max_drawdown_vs_base"] > 0).mean()),
                "research_read": _variant_read(group, wound_group),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["median_delta_cagr", "median_delta_max_drawdown", "median_2011_drawdown_delta"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_window_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame:
    if window_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (window_name, variant_name), group in window_metrics.groupby(
        ["window_name", "variant_name"],
        sort=False,
    ):
        rows.append(
            {
                "window_name": window_name,
                "variant_name": variant_name,
                "strategies": len(group),
                "median_window_return": pd.to_numeric(
                    group["window_return"],
                    errors="coerce",
                ).median(),
                "median_window_max_drawdown": pd.to_numeric(
                    group["window_max_drawdown"],
                    errors="coerce",
                ).median(),
                "median_delta_window_return_vs_base": pd.to_numeric(
                    group["delta_window_return_vs_base"],
                    errors="coerce",
                ).median(),
                "median_delta_window_max_drawdown_vs_base": pd.to_numeric(
                    group["delta_window_max_drawdown_vs_base"],
                    errors="coerce",
                ).median(),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_ai_repair_summary(
    strategy_metrics: pd.DataFrame,
    variant_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> str:
    lines = [
        "# AI Concentration Repair Lab",
        "",
        "This lab tests larger architecture changes for the actual max-drawdown wound:",
        "AI/growth concentration losses around the 2011-2012 NVDA drawdown.",
        "",
        "## Baseline",
        "",
    ]
    if strategy_metrics.empty:
        lines.append("- no strategy rows were available")
    else:
        lines.append(
            "- strategies tested: "
            f"{len(strategy_metrics)}; median CAGR "
            f"{pd.to_numeric(strategy_metrics['cagr'], errors='coerce').median():.2%}; "
            "median max DD "
            f"{pd.to_numeric(strategy_metrics['max_drawdown'], errors='coerce').median():.2%}; "
            "median AI/growth weight "
            f"{pd.to_numeric(strategy_metrics['average_ai_growth_weight'], errors='coerce').median():.1%}"
        )
    lines.extend(["", "## Best Variants", ""])
    if variant_summary.empty:
        lines.append("- no variants were available")
    else:
        for _, row in variant_summary.head(12).iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: median CAGR {_safe_float(row['median_cagr']):.2%}, "
                f"max DD {_safe_float(row['median_max_drawdown']):.2%}, "
                f"delta CAGR {_safe_float(row['median_delta_cagr']):+.2%}, "
                f"DD delta {_safe_float(row['median_delta_max_drawdown']):+.2%}, "
                f"2011 DD delta {_safe_float(row['median_2011_drawdown_delta']):+.2%}; "
                f"{row['research_read']}"
            )
    lines.extend(["", "## 2011-2012 Wound", ""])
    wound = window_summary[
        window_summary["window_name"].eq("2011_2012_ai_growth_wound")
        & window_summary["variant_name"].ne("base")
    ].copy()
    if wound.empty:
        lines.append("- no 2011 wound window rows were available")
    else:
        wound = wound.sort_values("median_delta_window_max_drawdown_vs_base", ascending=False)
        for _, row in wound.head(8).iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: window DD "
                f"{_safe_float(row['median_window_max_drawdown']):.2%}, "
                f"DD delta {_safe_float(row['median_delta_window_max_drawdown_vs_base']):+.2%}, "
                f"return delta {_safe_float(row['median_delta_window_return_vs_base']):+.2%}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Variants that improve the 2011 wound but lose too much full-history CAGR",
            "  remain diagnostics, not promotion candidates.",
            "- A durable winner should improve or preserve full-history CAGR, reduce max",
            "  drawdown, and also improve the 2011 wound across most top strategies.",
            "- If no cap/destination variant clears that bar, the next architecture should",
            "  be a sleeve router, not another cap.",
        ]
    )
    return "\n".join(lines)


def write_ai_repair_outputs(
    result: AiConcentrationRepairResult,
    *,
    output_dir: str | Path = DEFAULT_AI_REPAIR_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    result.variant_metrics.to_csv(output_path / "variant_metrics.csv", index=False)
    result.variant_summary.to_csv(output_path / "variant_summary.csv", index=False)
    result.window_summary.to_csv(output_path / "window_summary.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _repair_specs() -> tuple[AiRepairSpec, ...]:
    specs = []
    for stress_signal in (
        "ai_drawdown_smh_trend",
        "strategy_ai_drawdown",
        "qqq_spy_credit_trend",
        "nvda_drawdown",
        "ai_relative_break",
        "ai_basket_drawdown",
        "owned_ai_sleeve_drawdown",
        "owned_ai_leader_drawdown",
        "ai_breadth_break",
        "ai_breadth_or_relative_break",
        "ai_breadth_or_credit_trend",
        "ai_breadth_or_owned_sleeve",
        "ai_dual_confirm_break",
    ):
        for ai_cap in (0.65, 0.55, 0.45, 0.35):
            for destination in DESTINATION_WEIGHTS:
                specs.append(
                    AiRepairSpec(
                        name=f"{stress_signal}_cap{int(ai_cap * 100)}_{destination}",
                        stress_signal=stress_signal,
                        ai_cap=ai_cap,
                        destination=destination,
                    )
                )
    return tuple(specs)


def _stress_signal(signal: str, prices: pd.DataFrame, result: BacktestResult) -> pd.Series:
    if signal == "ai_drawdown_smh_trend":
        qqq_dd = rolling_drawdown(prices["QQQ"].ffill(), 126) <= -0.08
        smh_below = prices["SMH"] < moving_average(prices[["SMH"]], 100)["SMH"]
        return (qqq_dd & smh_below).shift(1, fill_value=False).astype(bool)
    if signal == "strategy_ai_drawdown":
        strategy_dd = rolling_drawdown(result.equity.reindex(prices.index).ffill(), 126) <= -0.08
        ai_proxy_dd = rolling_drawdown(prices["QQQ"].ffill(), 126) <= -0.08
        return (strategy_dd & ai_proxy_dd).shift(1, fill_value=False).astype(bool)
    if signal == "qqq_spy_credit_trend":
        qqq_below = prices["QQQ"] < moving_average(prices[["QQQ"]], 200)["QQQ"]
        spy_below = prices["SPY"] < moving_average(prices[["SPY"]], 200)["SPY"]
        credit_weak = _credit_weak(prices)
        return (qqq_below & (spy_below | credit_weak)).shift(1, fill_value=False).astype(bool)
    if signal == "nvda_drawdown":
        if "NVDA" not in prices:
            return pd.Series(False, index=prices.index)
        return (rolling_drawdown(prices["NVDA"].ffill(), 126) <= -0.20).shift(
            1,
            fill_value=False,
        ).astype(bool)
    if signal == "ai_relative_break":
        if "SMH" not in prices or "QQQ" not in prices:
            return pd.Series(False, index=prices.index)
        ratio = prices["SMH"].ffill() / prices["QQQ"].ffill()
        relative_break = ratio < moving_average(ratio.to_frame("ratio"), 100)["ratio"]
        qqq_dd = rolling_drawdown(prices["QQQ"].ffill(), 126) <= -0.05
        return (relative_break & qqq_dd).shift(1, fill_value=False).astype(bool)
    if signal == "ai_basket_drawdown":
        basket = _ai_price_basket(prices)
        if basket.empty:
            return pd.Series(False, index=prices.index)
        return (rolling_drawdown(basket, 126) <= -0.12).shift(1, fill_value=False).astype(bool)
    if signal == "owned_ai_sleeve_drawdown":
        sleeve_equity = _owned_ai_sleeve_equity(prices, result)
        if sleeve_equity.empty:
            return pd.Series(False, index=prices.index)
        return (rolling_drawdown(sleeve_equity, 126) <= -0.12).shift(
            1,
            fill_value=False,
        ).astype(bool)
    if signal == "owned_ai_leader_drawdown":
        leader_drawdown = _owned_ai_leader_drawdown(prices, result)
        if leader_drawdown.empty:
            return pd.Series(False, index=prices.index)
        return (leader_drawdown <= -0.20).shift(1, fill_value=False).astype(bool)
    if signal == "ai_breadth_break":
        ai_columns = [column for column in prices.columns if column in AI_GROWTH_TICKERS]
        if not ai_columns:
            return pd.Series(False, index=prices.index)
        ma = moving_average(prices[ai_columns], 100)
        breadth = (prices[ai_columns] > ma).mean(axis=1)
        qqq_dd = rolling_drawdown(prices["QQQ"].ffill(), 126) <= -0.05
        return ((breadth <= 0.35) & qqq_dd).shift(1, fill_value=False).astype(bool)
    if signal == "ai_breadth_or_relative_break":
        return (
            _stress_signal("ai_breadth_break", prices, result)
            | _stress_signal("ai_relative_break", prices, result)
        )
    if signal == "ai_breadth_or_credit_trend":
        return (
            _stress_signal("ai_breadth_break", prices, result)
            | _stress_signal("qqq_spy_credit_trend", prices, result)
        )
    if signal == "ai_breadth_or_owned_sleeve":
        return (
            _stress_signal("ai_breadth_break", prices, result)
            | _stress_signal("owned_ai_sleeve_drawdown", prices, result)
        )
    if signal == "ai_dual_confirm_break":
        return (
            _stress_signal("ai_breadth_break", prices, result)
            & _stress_signal("ai_relative_break", prices, result)
        )
    if signal == "ai_dual_confirm_break_persist3":
        base = _stress_signal("ai_dual_confirm_break", prices, result)
        return (base.astype(float).rolling(5, min_periods=1).sum() >= 3).astype(bool)
    if signal == "ai_dual_confirm_break_persist5":
        base = _stress_signal("ai_dual_confirm_break", prices, result)
        return (base.astype(float).rolling(10, min_periods=1).sum() >= 5).astype(bool)
    if signal == "ai_dual_confirm_break_sticky10":
        base = _stress_signal("ai_dual_confirm_break", prices, result)
        return (base.astype(float).rolling(10, min_periods=1).max() > 0).astype(bool)
    msg = f"Unknown AI repair stress signal: {signal}"
    raise ValueError(msg)


def _variant_metric_row(
    *,
    strategy: str,
    family: str,
    spec: AiRepairSpec,
    base_result: BacktestResult,
    variant_result: BacktestResult,
    base_metrics: PerformanceMetrics,
    variant_metrics: PerformanceMetrics,
    prices: pd.DataFrame,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "variant_name": spec.name,
        "stress_signal": spec.stress_signal,
        "ai_cap": spec.ai_cap,
        "destination": spec.destination,
        "cagr": variant_metrics.cagr,
        "max_drawdown": variant_metrics.max_drawdown,
        "calmar": variant_metrics.calmar,
        "sharpe": variant_metrics.sharpe,
        "average_turnover": variant_metrics.average_turnover,
        "delta_cagr_vs_base": variant_metrics.cagr - base_metrics.cagr,
        "delta_max_drawdown_vs_base": variant_metrics.max_drawdown - base_metrics.max_drawdown,
        "delta_calmar_vs_base": variant_metrics.calmar - base_metrics.calmar,
        "active_day_rate": _variant_active_day_rate(base_result.weights, variant_result.weights),
        "stress_day_rate": float(_stress_signal(spec.stress_signal, prices, base_result).mean()),
    }


def _window_metric_row(
    *,
    strategy: str,
    variant_name: str,
    window_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
    start: str,
    end: str,
) -> dict[str, object]:
    base_return, base_dd = _window_return_drawdown(base_result.equity, start, end)
    variant_return, variant_dd = _window_return_drawdown(variant_result.equity, start, end)
    return {
        "strategy": strategy,
        "variant_name": variant_name,
        "window_name": window_name,
        "window_return": variant_return,
        "window_max_drawdown": variant_dd,
        "base_window_return": base_return,
        "base_window_max_drawdown": base_dd,
        "delta_window_return_vs_base": variant_return - base_return,
        "delta_window_max_drawdown_vs_base": variant_dd - base_dd,
    }


def _window_return_drawdown(equity: pd.Series, start: str, end: str) -> tuple[float, float]:
    window = equity.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if len(window) < 2:
        return 0.0, 0.0
    return float(window.iloc[-1] / window.iloc[0] - 1.0), float(drawdown(window).min())


def _ai_price_basket(prices: pd.DataFrame) -> pd.Series:
    ai_columns = [column for column in prices.columns if column in AI_GROWTH_TICKERS]
    if not ai_columns:
        return pd.Series(dtype=float)
    normalized = prices[ai_columns].ffill().div(prices[ai_columns].ffill().iloc[0])
    return normalized.mean(axis=1).dropna()


def _owned_ai_sleeve_equity(prices: pd.DataFrame, result: BacktestResult) -> pd.Series:
    weights = result.weights.reindex(prices.index).ffill().fillna(0.0)
    ai_columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS and column in prices]
    if not ai_columns:
        return pd.Series(dtype=float)
    sleeve_returns = (
        weights[ai_columns]
        .mul(daily_returns(prices[ai_columns]).reindex(weights.index).fillna(0.0), axis=0)
        .sum(axis=1)
    )
    return (1.0 + sleeve_returns).cumprod()


def _owned_ai_leader_drawdown(prices: pd.DataFrame, result: BacktestResult) -> pd.Series:
    weights = result.weights.reindex(prices.index).ffill().fillna(0.0)
    ai_columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS and column in prices]
    if not ai_columns:
        return pd.Series(dtype=float)
    leader = weights[ai_columns].idxmax(axis=1)
    ticker_drawdowns = pd.DataFrame(
        {
            ticker: rolling_drawdown(prices[ticker].ffill(), 126)
            for ticker in ai_columns
        }
    )
    values = pd.Series(index=weights.index, dtype=float)
    for ticker in ai_columns:
        mask = leader.eq(ticker)
        values.loc[mask] = ticker_drawdowns.loc[mask, ticker]
    return values


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _average_ai_weight(weights: pd.DataFrame) -> float:
    columns = [column for column in weights.columns if _ticker_family(column) == "ai_growth"]
    if not columns:
        return 0.0
    return float(weights[columns].sum(axis=1).mean())


def _variant_active_day_rate(base_weights: pd.DataFrame, variant_weights: pd.DataFrame) -> float:
    columns = sorted(set(base_weights.columns) | set(variant_weights.columns))
    base = base_weights.reindex(variant_weights.index).reindex(columns=columns, fill_value=0.0)
    variant = variant_weights.reindex(columns=columns, fill_value=0.0)
    difference = variant.sub(base, fill_value=0.0).abs().sum(axis=1)
    return float((difference > 0.001).mean())


def _variant_read(group: pd.DataFrame, wound_group: pd.DataFrame) -> str:
    cagr_delta = pd.to_numeric(group["delta_cagr_vs_base"], errors="coerce").median()
    dd_delta = pd.to_numeric(group["delta_max_drawdown_vs_base"], errors="coerce").median()
    wound_delta = pd.to_numeric(
        wound_group["median_delta_window_max_drawdown_vs_base"],
        errors="coerce",
    ).median()
    if cagr_delta >= 0.0 and dd_delta >= 0.002 and wound_delta >= 0.002:
        return "promising"
    if cagr_delta >= -0.005 and dd_delta >= 0.005:
        return "survivability_watch"
    if cagr_delta < -0.015 and dd_delta < 0.010:
        return "reject_drag_without_enough_protection"
    return "diagnostic"
