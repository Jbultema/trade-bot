from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.research.ai_concentration_repair import (
    DEFAULT_REPAIR_TOP_N,
    REPAIR_WINDOWS,
    AiRepairSpec,
    _average_ai_weight,
    apply_ai_repair_variant,
    summarize_variant_metrics,
    summarize_window_metrics,
)
from trade_bot.research.experiments import (
    _candidate_tickers,
    _load_previous_candidates,
    _load_previous_scorecards,
    _strategy_prices,
)
from trade_bot.research.prebreak_hindsight import _safe_float
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.research.risk_policy_backtest import (
    _active_experiment_root,
    _run_candidate_backtest,
    _selected_candidates,
)

DEFAULT_AI_REPAIR_VALIDATION_OUTPUT_DIR = Path("reports/ai_repair_validation")
VALIDATION_ERAS: dict[str, tuple[str, str]] = {
    "pre_ai_growth_wound": ("2005-01-01", "2010-12-31"),
    "ai_growth_wound_and_repair": ("2011-01-01", "2014-12-31"),
    "cloud_growth_bull": ("2015-01-01", "2020-12-31"),
    "rates_growth_wound": ("2021-01-01", "2023-12-31"),
    "current_ai_cycle": ("2024-01-01", "2026-12-31"),
}
ROLLING_WINDOW_YEARS = (1, 3, 5)
ROLLING_STEP_MONTHS = 6
MIN_TRADING_DAYS_PER_YEAR = 180
MIN_ROLLING_OBSERVATION_RATIO = 0.70


@dataclass(frozen=True)
class AiRepairValidationResult:
    overall_metrics: pd.DataFrame
    yearly_deltas: pd.DataFrame
    rolling_deltas: pd.DataFrame
    era_deltas: pd.DataFrame
    window_summary: pd.DataFrame
    exposure_summary: pd.DataFrame
    validation_summary: pd.DataFrame
    summary: str


def run_ai_repair_validation_gauntlet(
    config: BotConfig,
    *,
    iteration: int = 164,
    top_n: int = DEFAULT_REPAIR_TOP_N,
    specs: tuple[AiRepairSpec, ...] | None = None,
    experiment_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_AI_REPAIR_VALIDATION_OUTPUT_DIR,
    refresh_data: bool = False,
) -> AiRepairValidationResult:
    experiment_root = Path(experiment_root) if experiment_root else _active_experiment_root()
    repair_specs = specs or default_validation_specs()
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
        | set(AI_GROWTH_TICKERS)
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

    overall_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    era_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    variant_metric_rows: list[dict[str, object]] = []

    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_result = _run_candidate_backtest(config, candidate, prices, candidate_prices)
        base_metrics = _metrics(base_result)
        average_ai_growth_weight = _average_ai_weight(base_result.weights)
        overall_rows.append(
            _overall_metric_row(
                strategy=candidate.name,
                family=candidate.family,
                variant_name="base",
                metrics=base_metrics,
                base_metrics=base_metrics,
                average_ai_growth_weight=average_ai_growth_weight,
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
            overall_rows.append(
                _overall_metric_row(
                    strategy=candidate.name,
                    family=candidate.family,
                    variant_name=spec.name,
                    metrics=variant_metrics,
                    base_metrics=base_metrics,
                    average_ai_growth_weight=average_ai_growth_weight,
                )
            )
            variant_metric_rows.append(
                {
                    "strategy": candidate.name,
                    "family": candidate.family,
                    "variant_name": spec.name,
                    "stress_signal": spec.stress_signal,
                    "ai_cap": spec.ai_cap,
                    "destination": spec.destination,
                    "average_ai_growth_weight": average_ai_growth_weight,
                    "cagr": variant_metrics.cagr,
                    "max_drawdown": variant_metrics.max_drawdown,
                    "calmar": variant_metrics.calmar,
                    "delta_cagr_vs_base": variant_metrics.cagr - base_metrics.cagr,
                    "delta_max_drawdown_vs_base": (
                        variant_metrics.max_drawdown - base_metrics.max_drawdown
                    ),
                    "active_day_rate": _active_day_rate(base_result, variant_result),
                }
            )
            yearly_rows.extend(_calendar_delta_rows(candidate.name, spec.name, base_result, variant_result))
            rolling_rows.extend(
                _rolling_delta_rows(candidate.name, spec.name, base_result, variant_result)
            )
            era_rows.extend(_fixed_window_delta_rows(candidate.name, spec.name, base_result, variant_result, VALIDATION_ERAS))
            window_rows.extend(_fixed_window_delta_rows(candidate.name, spec.name, base_result, variant_result, REPAIR_WINDOWS))

    variant_metrics = pd.DataFrame(variant_metric_rows)
    window_summary = summarize_window_metrics(_stress_window_frame(window_rows))
    validation_summary = summarize_validation(
        variant_metrics=variant_metrics,
        yearly_deltas=pd.DataFrame(yearly_rows),
        rolling_deltas=pd.DataFrame(rolling_rows),
        era_deltas=pd.DataFrame(era_rows),
        window_summary=window_summary,
    )
    exposure_summary = summarize_exposure_segments(variant_metrics)
    result = AiRepairValidationResult(
        overall_metrics=pd.DataFrame(overall_rows),
        yearly_deltas=pd.DataFrame(yearly_rows),
        rolling_deltas=pd.DataFrame(rolling_rows),
        era_deltas=pd.DataFrame(era_rows),
        window_summary=window_summary,
        exposure_summary=exposure_summary,
        validation_summary=validation_summary,
        summary=build_validation_summary(validation_summary, window_summary, exposure_summary),
    )
    write_ai_repair_validation_outputs(result, output_dir=output_dir)
    return result


def default_validation_specs() -> tuple[AiRepairSpec, ...]:
    return (
        AiRepairSpec(
            name="ai_dual_confirm_break_cap45_bil",
            stress_signal="ai_dual_confirm_break",
            ai_cap=0.45,
            destination="bil",
        ),
        AiRepairSpec(
            name="ai_dual_confirm_break_cap35_bil",
            stress_signal="ai_dual_confirm_break",
            ai_cap=0.35,
            destination="bil",
        ),
        AiRepairSpec(
            name="ai_breadth_break_cap45_bil",
            stress_signal="ai_breadth_break",
            ai_cap=0.45,
            destination="bil",
        ),
        AiRepairSpec(
            name="ai_relative_break_cap45_bil",
            stress_signal="ai_relative_break",
            ai_cap=0.45,
            destination="bil",
        ),
        AiRepairSpec(
            name="nvda_drawdown_cap35_bil_gld_tlt",
            stress_signal="nvda_drawdown",
            ai_cap=0.35,
            destination="bil_gld_tlt",
        ),
    )


def summarize_validation(
    *,
    variant_metrics: pd.DataFrame,
    yearly_deltas: pd.DataFrame,
    rolling_deltas: pd.DataFrame,
    era_deltas: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    if variant_metrics.empty:
        return pd.DataFrame()
    base_summary = summarize_variant_metrics(variant_metrics, window_summary)
    rows: list[dict[str, object]] = []
    for _, row in base_summary.iterrows():
        variant_name = str(row["variant_name"])
        yearly = yearly_deltas[yearly_deltas["variant_name"].eq(variant_name)]
        rolling = rolling_deltas[rolling_deltas["variant_name"].eq(variant_name)]
        era = era_deltas[era_deltas["variant_name"].eq(variant_name)]
        one_year = rolling[rolling["window_years"].eq(1)]
        three_year = rolling[rolling["window_years"].eq(3)]
        five_year = rolling[rolling["window_years"].eq(5)]
        current_era = era[era["window_name"].eq("current_ai_cycle")]
        rates_wound = era[era["window_name"].eq("rates_growth_wound")]
        validation_row = row.to_dict()
        validation_row.update(
            {
                "calendar_years": len(yearly),
                "calendar_positive_cagr_delta_rate": _positive_rate(yearly, "delta_cagr"),
                "calendar_large_drag_rate": _large_drag_rate(yearly, "delta_cagr", -0.0025),
                "rolling_1y_positive_cagr_delta_rate": _positive_rate(one_year, "delta_cagr"),
                "rolling_1y_large_drag_rate": _large_drag_rate(one_year, "delta_cagr", -0.0025),
                "rolling_3y_positive_cagr_delta_rate": _positive_rate(three_year, "delta_cagr"),
                "rolling_3y_median_cagr_delta": _median(three_year, "delta_cagr"),
                "rolling_3y_large_drag_rate": _large_drag_rate(three_year, "delta_cagr", -0.0025),
                "rolling_5y_positive_cagr_delta_rate": _positive_rate(five_year, "delta_cagr"),
                "rolling_5y_median_cagr_delta": _median(five_year, "delta_cagr"),
                "current_ai_cycle_median_return_delta": _median(current_era, "delta_total_return"),
                "rates_wound_median_return_delta": _median(rates_wound, "delta_total_return"),
            }
        )
        validation_row["promotion_gate"] = _promotion_gate(validation_row)
        validation_row["promotion_rank"] = _promotion_rank(str(validation_row["promotion_gate"]))
        rows.append(validation_row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "promotion_rank",
                "median_delta_cagr",
                "median_delta_max_drawdown",
                "rolling_3y_median_cagr_delta",
            ],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )


def summarize_exposure_segments(variant_metrics: pd.DataFrame) -> pd.DataFrame:
    if variant_metrics.empty or "average_ai_growth_weight" not in variant_metrics:
        return pd.DataFrame()
    frame = variant_metrics.copy()
    frame["ai_exposure_bucket"] = pd.cut(
        pd.to_numeric(frame["average_ai_growth_weight"], errors="coerce"),
        bins=[-0.001, 0.25, 0.50, 1.001],
        labels=["low_ai_exposure", "medium_ai_exposure", "high_ai_exposure"],
    )
    rows: list[dict[str, object]] = []
    for (variant_name, bucket), group in frame.groupby(
        ["variant_name", "ai_exposure_bucket"],
        observed=True,
        sort=False,
    ):
        rows.append(
            {
                "variant_name": variant_name,
                "ai_exposure_bucket": str(bucket),
                "strategies": len(group),
                "median_average_ai_growth_weight": _median(group, "average_ai_growth_weight"),
                "median_delta_cagr": _median(group, "delta_cagr_vs_base"),
                "median_delta_max_drawdown": _median(group, "delta_max_drawdown_vs_base"),
                "cagr_win_rate": _positive_rate(group, "delta_cagr_vs_base"),
                "drawdown_win_rate": _positive_rate(group, "delta_max_drawdown_vs_base"),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_validation_summary(
    validation_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# AI Repair Validation Gauntlet",
        "",
        "This validation pass promotes only variants that improve full-history CAGR,",
        "reduce drawdown, avoid ordinary-period drag, and help more than one stress window.",
        "",
        "## Promotion Read",
        "",
    ]
    if validation_summary.empty:
        lines.append("- no validation rows were available")
    else:
        for _, row in validation_summary.head(8).iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: gate {row['promotion_gate']}; "
                f"delta CAGR {_safe_float(row['median_delta_cagr']):+.2%}, "
                f"DD delta {_safe_float(row['median_delta_max_drawdown']):+.2%}, "
                f"3Y rolling delta {_safe_float(row['rolling_3y_median_cagr_delta']):+.2%}, "
                f"calendar drag rate {_safe_float(row['calendar_large_drag_rate']):.1%}, "
                f"current-cycle return delta {_safe_float(row['current_ai_cycle_median_return_delta']):+.2%}"
            )
    lines.extend(["", "## Exposure Segments", ""])
    if exposure_summary is None or exposure_summary.empty:
        lines.append("- no AI exposure segment rows were available")
    else:
        for _, row in exposure_summary.iterrows():
            lines.append(
                "- "
                f"{row['variant_name']} / {row['ai_exposure_bucket']}: "
                f"strategies {int(row['strategies'])}, "
                f"avg AI weight {_safe_float(row['median_average_ai_growth_weight']):.1%}, "
                f"delta CAGR {_safe_float(row['median_delta_cagr']):+.2%}, "
                f"DD delta {_safe_float(row['median_delta_max_drawdown']):+.2%}"
            )
    lines.extend(["", "## Stress Windows", ""])
    if window_summary.empty:
        lines.append("- no stress-window rows were available")
    else:
        for _, row in window_summary[window_summary["variant_name"].ne("base")].iterrows():
            lines.append(
                "- "
                f"{row['window_name']} / {row['variant_name']}: "
                f"return delta {_safe_float(row['median_delta_window_return_vs_base']):+.2%}, "
                f"DD delta {_safe_float(row['median_delta_window_max_drawdown_vs_base']):+.2%}"
            )
    lines.extend(
        [
            "",
            "## Gate Definition",
            "",
            "- promote: positive median CAGR and max-DD deltas, at least 85% strategy win",
            "  rates on both, less than 15% large yearly drag, positive 3Y rolling median",
            "  CAGR delta, and no material current-cycle drag.",
            "- watchlist: directionally useful, but misses at least one robustness bar.",
            "- reject: too narrow, too draggy, or insufficient full-history improvement.",
        ]
    )
    return "\n".join(lines)


def write_ai_repair_validation_outputs(
    result: AiRepairValidationResult,
    *,
    output_dir: str | Path = DEFAULT_AI_REPAIR_VALIDATION_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.overall_metrics.to_csv(output_path / "overall_metrics.csv", index=False)
    result.yearly_deltas.to_csv(output_path / "yearly_deltas.csv", index=False)
    result.rolling_deltas.to_csv(output_path / "rolling_deltas.csv", index=False)
    result.era_deltas.to_csv(output_path / "era_deltas.csv", index=False)
    result.window_summary.to_csv(output_path / "window_summary.csv", index=False)
    result.exposure_summary.to_csv(output_path / "exposure_summary.csv", index=False)
    result.validation_summary.to_csv(output_path / "validation_summary.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _overall_metric_row(
    *,
    strategy: str,
    family: str,
    variant_name: str,
    metrics: PerformanceMetrics,
    base_metrics: PerformanceMetrics,
    average_ai_growth_weight: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "variant_name": variant_name,
        "average_ai_growth_weight": average_ai_growth_weight,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "delta_cagr_vs_base": metrics.cagr - base_metrics.cagr,
        "delta_max_drawdown_vs_base": metrics.max_drawdown - base_metrics.max_drawdown,
        "delta_calmar_vs_base": metrics.calmar - base_metrics.calmar,
    }


def _stress_window_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).rename(
        columns={
            "total_return": "window_return",
            "max_drawdown": "window_max_drawdown",
            "base_total_return": "base_window_return",
            "base_max_drawdown": "base_window_max_drawdown",
            "delta_total_return": "delta_window_return_vs_base",
            "delta_max_drawdown": "delta_window_max_drawdown_vs_base",
        }
    )


def _calendar_delta_rows(
    strategy: str,
    variant_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
) -> list[dict[str, object]]:
    rows = []
    for year, returns in base_result.returns.groupby(base_result.returns.index.year):
        if len(returns) < MIN_TRADING_DAYS_PER_YEAR:
            continue
        window_name = str(year)
        rows.append(
            _period_delta_row(
                strategy,
                variant_name,
                window_name,
                base_result,
                variant_result,
                returns.index.min(),
                returns.index.max(),
            )
        )
    return rows


def _rolling_delta_rows(
    strategy: str,
    variant_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
) -> list[dict[str, object]]:
    rows = []
    index = base_result.returns.dropna().index
    if index.empty:
        return rows
    final_date = index.max()
    for years in ROLLING_WINDOW_YEARS:
        start = index.min()
        window_days = int(365.25 * years)
        min_observations = int(MIN_TRADING_DAYS_PER_YEAR * years * MIN_ROLLING_OBSERVATION_RATIO)
        while start + pd.DateOffset(days=window_days) <= final_date:
            end = start + pd.DateOffset(days=window_days)
            window_index = index[(index >= start) & (index <= end)]
            if len(window_index) >= min_observations:
                row = _period_delta_row(
                    strategy,
                    variant_name,
                    f"{years}y_{window_index.min().date()}_{window_index.max().date()}",
                    base_result,
                    variant_result,
                    window_index.min(),
                    window_index.max(),
                )
                row["window_years"] = years
                rows.append(row)
            start = start + pd.DateOffset(months=ROLLING_STEP_MONTHS)
    return rows


def _fixed_window_delta_rows(
    strategy: str,
    variant_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
    windows: dict[str, tuple[str, str]],
) -> list[dict[str, object]]:
    rows = []
    for window_name, (start, end) in windows.items():
        window_index = base_result.returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
        if len(window_index) < 2:
            continue
        rows.append(
            _period_delta_row(
                strategy,
                variant_name,
                window_name,
                base_result,
                variant_result,
                window_index.min(),
                window_index.max(),
            )
        )
    return rows


def _period_delta_row(
    strategy: str,
    variant_name: str,
    window_name: str,
    base_result: BacktestResult,
    variant_result: BacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    base_metrics = _window_metrics(base_result, start, end)
    variant_metrics = _window_metrics(variant_result, start, end)
    return {
        "strategy": strategy,
        "variant_name": variant_name,
        "window_name": window_name,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "total_return": variant_metrics["total_return"],
        "cagr": variant_metrics["cagr"],
        "max_drawdown": variant_metrics["max_drawdown"],
        "base_total_return": base_metrics["total_return"],
        "base_cagr": base_metrics["cagr"],
        "base_max_drawdown": base_metrics["max_drawdown"],
        "delta_total_return": variant_metrics["total_return"] - base_metrics["total_return"],
        "delta_cagr": variant_metrics["cagr"] - base_metrics["cagr"],
        "delta_max_drawdown": variant_metrics["max_drawdown"] - base_metrics["max_drawdown"],
    }


def _window_metrics(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    returns = result.returns.loc[start:end].dropna()
    if len(returns) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
    equity = 100.0 * (1.0 + returns).cumprod()
    metrics = calculate_metrics(
        name=result.name,
        returns=returns,
        equity=equity,
        turnover=result.turnover.reindex(returns.index).fillna(0.0),
        transaction_costs=result.transaction_costs.reindex(returns.index).fillna(0.0),
    )
    return {
        "total_return": float(equity.iloc[-1] / 100.0 - 1.0),
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
    }


def _metrics(result: BacktestResult) -> PerformanceMetrics:
    return calculate_metrics(
        name=result.name,
        returns=result.returns,
        equity=result.equity,
        turnover=result.turnover,
        transaction_costs=result.transaction_costs,
    )


def _active_day_rate(base_result: BacktestResult, variant_result: BacktestResult) -> float:
    columns = sorted(set(base_result.weights.columns) | set(variant_result.weights.columns))
    base = base_result.weights.reindex(variant_result.weights.index).reindex(
        columns=columns,
        fill_value=0.0,
    )
    variant = variant_result.weights.reindex(columns=columns, fill_value=0.0)
    difference = variant.sub(base, fill_value=0.0).abs().sum(axis=1)
    return float((difference > 0.001).mean())


def _promotion_gate(row: dict[str, object]) -> str:
    if str(row.get("stress_signal")) == "nvda_drawdown":
        return "diagnostic_only_single_name"
    strong = (
        _safe_float(row.get("median_delta_cagr")) >= 0.0005
        and _safe_float(row.get("median_delta_max_drawdown")) >= 0.005
        and _safe_float(row.get("cagr_win_rate")) >= 0.85
        and _safe_float(row.get("drawdown_win_rate")) >= 0.85
        and _safe_float(row.get("calendar_large_drag_rate")) <= 0.15
        and _safe_float(row.get("rolling_3y_median_cagr_delta")) >= 0.0
        and _safe_float(row.get("current_ai_cycle_median_return_delta")) >= -0.002
    )
    if strong:
        return "promote_candidate"
    watch = (
        _safe_float(row.get("median_delta_cagr")) >= 0.0
        and _safe_float(row.get("median_delta_max_drawdown")) >= 0.0025
        and _safe_float(row.get("drawdown_win_rate")) >= 0.75
        and _safe_float(row.get("current_ai_cycle_median_return_delta")) >= -0.005
    )
    if watch:
        return "watchlist"
    return "reject"


def _promotion_rank(gate: str) -> int:
    return {
        "promote_candidate": 0,
        "watchlist": 1,
        "diagnostic_only_single_name": 2,
        "reject": 3,
    }.get(gate, 9)


def _positive_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float((values > 0.0).mean())


def _large_drag_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float((values < threshold).mean())


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").median())
