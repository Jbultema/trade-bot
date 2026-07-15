from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import BotConfig
from trade_bot.DEFAULTS import DEFAULT_EXPERIMENTS_DIR
from trade_bot.research.approach_explorer import (
    build_approach_backtest_result,
    build_approach_catalog,
    decision_sanity_from_catalog_row,
    execution_for_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)

PBOMetric = Literal["sharpe", "mean_return", "total_return"]
DEFAULT_PBO_STRATEGY = "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145"


@dataclass(frozen=True)
class PBOResult:
    returns: pd.DataFrame
    summary: pd.DataFrame
    splits: pd.DataFrame
    strategy_selection: pd.DataFrame
    strategy_stats: pd.DataFrame


@dataclass(frozen=True)
class PBOGauntlet:
    output_dir: Path
    artifacts: dict[str, Path]
    result: PBOResult
    readout: str


def run_backtest_pbo_gauntlet(
    *,
    config: BotConfig,
    prices: pd.DataFrame,
    output_dir: str | Path = "reports/pbo_diagnostics",
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    strategies: tuple[str, ...] = (),
    top_n: int = 20,
    partitions: int = 8,
    metric: PBOMetric = "sharpe",
    min_observations: int = 252,
) -> PBOGauntlet:
    """Estimate Probability of Backtest Overfitting across candidate returns.

    This implements the Bailey/Lopez de Prado CSCV idea at the strategy-candidate
    level: split synchronized candidate returns into equal time blocks, use every
    half-block training combination to choose the in-sample winner, and measure
    where that winner ranks on the complementary out-of-sample blocks.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = select_pbo_candidate_rows(
        config,
        experiment_root=experiment_root,
        strategies=strategies,
        top_n=top_n,
    )
    results = build_pbo_candidate_results(config, prices, rows)
    returns = candidate_return_matrix(results, min_observations=min_observations)
    result = estimate_probability_of_backtest_overfitting(
        returns,
        partitions=partitions,
        metric=metric,
    )

    artifacts = {
        "summary": output / "pbo_summary.csv",
        "splits": output / "pbo_splits.csv",
        "strategy_selection": output / "pbo_strategy_selection.csv",
        "strategy_stats": output / "pbo_strategy_stats.csv",
        "returns": output / "pbo_candidate_returns.csv",
    }
    result.summary.to_csv(artifacts["summary"], index=False)
    result.splits.to_csv(artifacts["splits"], index=False)
    result.strategy_selection.to_csv(artifacts["strategy_selection"], index=False)
    result.strategy_stats.to_csv(artifacts["strategy_stats"], index=False)
    result.returns.to_csv(artifacts["returns"], index=True)

    readout = _markdown_readout(result)
    summary_path = output / "summary.md"
    summary_path.write_text(readout, encoding="utf-8")
    artifacts["readout"] = summary_path
    return PBOGauntlet(
        output_dir=output,
        artifacts=artifacts,
        result=result,
        readout=readout,
    )


def select_pbo_candidate_rows(
    config: BotConfig,
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    strategies: tuple[str, ...] = (),
    top_n: int = 20,
) -> pd.DataFrame:
    catalog = build_approach_catalog(config, experiment_root=experiment_root)
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    frame = catalog.copy()
    frame["strategy"] = frame["strategy"].astype(str)
    if strategies:
        requested = list(dict.fromkeys(str(strategy) for strategy in strategies if strategy))
        return frame[frame["strategy"].isin(requested)].drop_duplicates("strategy")

    priority = [config.primary_strategy] if config.primary_strategy else []
    if DEFAULT_PBO_STRATEGY in set(frame["strategy"]) and DEFAULT_PBO_STRATEGY not in priority:
        priority.append(DEFAULT_PBO_STRATEGY)

    sort_column = _first_available_column(
        frame,
        (
            "growth_constrained_utility_score",
            "selection_adjusted_promotion_score",
            "promotion_score",
            "cagr",
            "calmar",
        ),
    )
    scored = frame.copy()
    if sort_column:
        scored["_pbo_sort"] = pd.to_numeric(scored[sort_column], errors="coerce")
        scored = scored.sort_values("_pbo_sort", ascending=False, na_position="last")
    selected_names = list(dict.fromkeys([*priority, *scored["strategy"].tolist()]))[:top_n]
    return frame[frame["strategy"].isin(selected_names)].drop_duplicates("strategy")


def pbo_candidate_tickers(
    config: BotConfig,
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    strategies: tuple[str, ...] = (),
    top_n: int = 20,
) -> set[str]:
    rows = select_pbo_candidate_rows(
        config,
        experiment_root=experiment_root,
        strategies=strategies,
        top_n=top_n,
    )
    tickers: set[str] = set()
    for _, row in rows.iterrows():
        try:
            strategy = strategy_from_catalog_row(row)
        except (TypeError, ValueError):
            continue
        tickers.update(strategy.tickers)
        tickers.update(strategy.satellite_tickers)
        if strategy.defensive_ticker:
            tickers.add(strategy.defensive_ticker)
    return tickers


def build_pbo_candidate_results(
    config: BotConfig,
    prices: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, BacktestResult]:
    results: dict[str, BacktestResult] = {}
    for _, row in rows.iterrows():
        name = str(row.get("strategy", "") or "")
        if not name or name in results:
            continue
        try:
            strategy = strategy_from_catalog_row(row)
            execution = execution_for_catalog_row(row, config.execution)
            result, _missing = build_approach_backtest_result(
                prices,
                strategy,
                execution,
                scenario_sizing=scenario_sizing_from_catalog_row(row),
                future_state_model=future_state_model_from_catalog_row(row),
                strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
                decision_sanity=decision_sanity_from_catalog_row(row),
                name=name,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if result is not None:
            results[name] = result
    return results


def candidate_return_matrix(
    results: dict[str, BacktestResult],
    *,
    min_observations: int = 252,
) -> pd.DataFrame:
    returns = {
        name: pd.to_numeric(result.returns, errors="coerce")
        for name, result in results.items()
        if not result.returns.empty
    }
    if not returns:
        return pd.DataFrame()
    frame = pd.DataFrame(returns).sort_index()
    frame = frame.dropna(axis=0, how="all")
    enough_history = frame.notna().sum(axis=0) >= min_observations
    frame = frame.loc[:, enough_history].fillna(0.0)
    if frame.shape[1] < 2:
        return pd.DataFrame()
    return frame


def estimate_probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    *,
    partitions: int = 8,
    metric: PBOMetric = "sharpe",
) -> PBOResult:
    frame = returns.sort_index().copy()
    if frame.empty or frame.shape[1] < 2:
        empty_summary = _summary_frame(
            strategy_count=int(frame.shape[1]),
            observations=int(frame.shape[0]),
            partitions=partitions,
            metric=metric,
            splits=pd.DataFrame(),
        )
        return PBOResult(
            returns=frame,
            summary=empty_summary,
            splits=pd.DataFrame(),
            strategy_selection=pd.DataFrame(),
            strategy_stats=_strategy_stats(frame),
        )
    if partitions < 4 or partitions % 2:
        raise ValueError("partitions must be an even integer >= 4.")
    if len(frame) < partitions:
        raise ValueError("Not enough observations to create requested CSCV partitions.")

    blocks = _partition_positions(len(frame), partitions)
    split_rows: list[dict[str, object]] = []
    half = partitions // 2
    all_blocks = tuple(range(partitions))
    for split_id, train_blocks in enumerate(itertools.combinations(all_blocks, half), start=1):
        train_set = set(train_blocks)
        test_blocks = tuple(block for block in all_blocks if block not in train_set)
        train_positions = np.concatenate([blocks[block] for block in train_blocks])
        test_positions = np.concatenate([blocks[block] for block in test_blocks])
        train_stats = _performance_stat(frame.iloc[train_positions], metric)
        test_stats = _performance_stat(frame.iloc[test_positions], metric)
        test_total_returns = _performance_stat(frame.iloc[test_positions], "total_return")
        if train_stats.dropna().empty or test_stats.dropna().empty:
            continue
        selected = str(train_stats.idxmax())
        selected_train = float(train_stats[selected])
        selected_test = float(test_stats.get(selected, np.nan))
        selected_test_total_return = float(test_total_returns.get(selected, np.nan))
        if not math.isfinite(selected_test):
            continue
        rank = float(test_stats.rank(method="average", ascending=True).loc[selected])
        relative_rank = rank / (len(test_stats.dropna()) + 1.0)
        relative_rank = min(max(relative_rank, 1e-12), 1.0 - 1e-12)
        logit = math.log(relative_rank / (1.0 - relative_rank))
        test_best = str(test_stats.idxmax())
        split_rows.append(
            {
                "split_id": split_id,
                "train_blocks": ",".join(str(block) for block in train_blocks),
                "test_blocks": ",".join(str(block) for block in test_blocks),
                "selected_strategy": selected,
                "train_metric": selected_train,
                "test_metric": selected_test,
                "test_rank": rank,
                "relative_rank": relative_rank,
                "logit_relative_rank": logit,
                "overfit": bool(logit < 0.0),
                "oos_loss": bool(selected_test_total_return < 0.0),
                "test_total_return": selected_test_total_return,
                "test_best_strategy": test_best,
                "test_best_metric": float(test_stats[test_best]),
                "test_best_total_return": float(test_total_returns.get(test_best, np.nan)),
                "performance_degradation": selected_test - selected_train,
            }
        )

    splits = pd.DataFrame(split_rows)
    return PBOResult(
        returns=frame,
        summary=_summary_frame(
            strategy_count=int(frame.shape[1]),
            observations=int(frame.shape[0]),
            partitions=partitions,
            metric=metric,
            splits=splits,
        ),
        splits=splits,
        strategy_selection=_strategy_selection_frame(splits),
        strategy_stats=_strategy_stats(frame),
    )


def _partition_positions(length: int, partitions: int) -> list[np.ndarray]:
    positions = np.arange(length)
    return [block.astype(int) for block in np.array_split(positions, partitions)]


def _performance_stat(frame: pd.DataFrame, metric: PBOMetric) -> pd.Series:
    clean = frame.fillna(0.0)
    if metric == "mean_return":
        return clean.mean()
    if metric == "total_return":
        return (1.0 + clean).prod() - 1.0
    if metric != "sharpe":
        raise ValueError(f"Unsupported PBO metric: {metric}")
    mean = clean.mean()
    std = clean.std(ddof=1).replace(0.0, np.nan)
    return (mean / std) * math.sqrt(252.0)


def _summary_frame(
    *,
    strategy_count: int,
    observations: int,
    partitions: int,
    metric: PBOMetric,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    valid_splits = int(len(splits))
    pbo = float(splits["overfit"].mean()) if valid_splits and "overfit" in splits else float("nan")
    oos_loss = (
        float(splits["oos_loss"].mean())
        if valid_splits and "oos_loss" in splits
        else float("nan")
    )
    row = {
        "strategy_count": strategy_count,
        "observations": observations,
        "partitions": partitions,
        "metric": metric,
        "valid_splits": valid_splits,
        "pbo_probability": pbo,
        "oos_loss_probability": oos_loss,
        "median_relative_rank": _median_or_nan(splits.get("relative_rank")),
        "median_logit_relative_rank": _median_or_nan(splits.get("logit_relative_rank")),
        "median_train_metric": _median_or_nan(splits.get("train_metric")),
        "median_test_metric": _median_or_nan(splits.get("test_metric")),
        "median_performance_degradation": _median_or_nan(
            splits.get("performance_degradation")
        ),
        "pbo_label": _pbo_label(pbo),
    }
    return pd.DataFrame([row])


def _strategy_selection_frame(splits: pd.DataFrame) -> pd.DataFrame:
    if splits.empty or "selected_strategy" not in splits:
        return pd.DataFrame()
    grouped = (
        splits.groupby("selected_strategy")
        .agg(
            selected_count=("split_id", "count"),
            overfit_rate=("overfit", "mean"),
            oos_loss_rate=("oos_loss", "mean"),
            median_relative_rank=("relative_rank", "median"),
            median_train_metric=("train_metric", "median"),
            median_test_metric=("test_metric", "median"),
            median_degradation=("performance_degradation", "median"),
        )
        .reset_index()
        .rename(columns={"selected_strategy": "strategy"})
    )
    grouped["selection_rate"] = grouped["selected_count"] / max(float(len(splits)), 1.0)
    return grouped.sort_values(
        ["selected_count", "median_relative_rank"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _strategy_stats(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy in returns.columns:
        series = pd.to_numeric(returns[strategy], errors="coerce").fillna(0.0)
        equity = (1.0 + series).cumprod()
        years = max(len(series) / 252.0, 1e-12)
        total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if not equity.empty else float("nan")
        drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else float("nan")
        sharpe = _performance_stat(pd.DataFrame({strategy: series}), "sharpe").iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "observations": int(series.shape[0]),
                "total_return": total_return,
                "cagr": cagr,
                "max_drawdown": drawdown,
                "sharpe": float(sharpe) if pd.notna(sharpe) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False, na_position="last")


def _median_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def _pbo_label(value: float) -> str:
    if not math.isfinite(value):
        return "not_enough_data"
    if value >= 0.50:
        return "high_overfit_risk"
    if value >= 0.25:
        return "moderate_overfit_risk"
    return "low_overfit_risk"


def _first_available_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _markdown_readout(result: PBOResult) -> str:
    summary = result.summary.iloc[0] if not result.summary.empty else pd.Series(dtype=object)
    pbo = summary.get("pbo_probability", float("nan"))
    label = str(summary.get("pbo_label", "not_enough_data"))
    lines = [
        "# Backtest Overfit PBO Gauntlet",
        "",
        "Combinatorial symmetric cross-validation over synchronized candidate returns.",
        "",
        f"- Strategies: {int(summary.get('strategy_count', 0) or 0)}",
        f"- Observations: {int(summary.get('observations', 0) or 0)}",
        f"- Partitions: {int(summary.get('partitions', 0) or 0)}",
        f"- Selection metric: `{summary.get('metric', '')}`",
        f"- PBO probability: {_format_percent(pbo)}",
        f"- Label: `{label}`",
        f"- OOS loss probability: {_format_percent(summary.get('oos_loss_probability'))}",
        f"- Median OOS relative rank: {_format_percent(summary.get('median_relative_rank'))}",
        f"- Median performance degradation: {_format_decimal(summary.get('median_performance_degradation'))}",
        "",
        "Interpretation rule: low PBO supports the research process; high PBO means the best-looking candidate is often a cross-validation mirage.",
    ]
    if not result.strategy_selection.empty:
        top = result.strategy_selection.iloc[0]
        lines.extend(
            [
                "",
                "Most often selected in-sample:",
                f"- `{top['strategy']}` in {float(top['selection_rate']):.1%} of valid splits; "
                f"overfit rate {float(top['overfit_rate']):.1%}; "
                f"median OOS rank {float(top['median_relative_rank']):.1%}.",
            ]
        )
    return "\n".join(lines) + "\n"


def _format_percent(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2%}"


def _format_decimal(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"
