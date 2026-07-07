from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.DEFAULTS import (
    DEFAULT_ENTRY_HORIZONS,
    DEFAULT_LAUNCH_BAD_START_DRAWDOWN,
    DEFAULT_LAUNCH_CAPITAL,
    DEFAULT_LAUNCH_INITIAL_RAMP_FRACTION,
    DEFAULT_LAUNCH_MIN_WINDOWS,
    DEFAULT_LAUNCH_PRIMARY_HORIZON,
    DEFAULT_LAUNCH_PROTOCOL_MATERIAL_SPREAD,
    DEFAULT_LAUNCH_PROTOCOL_SMALL_SPREAD,
    DEFAULT_LAUNCH_RAMP_WEEKS,
    DEFAULT_LAUNCH_READY_SCORE,
    DEFAULT_LAUNCH_SET_SCORE,
    DEFAULT_LAUNCH_START_FREQUENCY,
    DEFAULT_LAUNCH_TARGET_FRACTION,
    DEFAULT_LAUNCH_WAIT_SCORE,
)


@dataclass(frozen=True)
class LaunchReadinessRun:
    """Historical entry-window evidence plus current-state launch guidance."""

    windows: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    ramp_plan: pd.DataFrame
    recommendation: dict[str, object]


@dataclass(frozen=True)
class AggregateLaunchReadinessRun:
    """Launch-readiness evidence summarized across multiple candidates."""

    strategy_horizon_summary: pd.DataFrame
    horizon_label_counts: pd.DataFrame
    horizon_transition_matrix: pd.DataFrame
    protocol_separation: pd.DataFrame
    protocol_separation_by_horizon: pd.DataFrame
    strategy_count: int


def build_launch_readiness(
    strategy_result: BacktestResult,
    *,
    benchmark_result: BacktestResult | None = None,
    current_state: Any | None = None,
    capital_to_launch: float = DEFAULT_LAUNCH_CAPITAL,
    target_fraction: float = DEFAULT_LAUNCH_TARGET_FRACTION,
    horizons: Mapping[str, int] = DEFAULT_ENTRY_HORIZONS,
    ramp_weeks: Sequence[int] = DEFAULT_LAUNCH_RAMP_WEEKS,
    primary_horizon: str = DEFAULT_LAUNCH_PRIMARY_HORIZON,
    start_frequency: str = DEFAULT_LAUNCH_START_FREQUENCY,
    bad_start_drawdown: float = DEFAULT_LAUNCH_BAD_START_DRAWDOWN,
    initial_ramp_fraction: float = DEFAULT_LAUNCH_INITIAL_RAMP_FRACTION,
) -> LaunchReadinessRun:
    """Score whether a strategy is favorable to launch now.

    This is not the daily operating-book target. It asks whether fresh paper or
    live capital should begin tracking a strategy immediately, gradually, or not
    yet.
    """

    strategy_returns = _daily_returns(strategy_result)
    benchmark_returns = _daily_returns(benchmark_result) if benchmark_result is not None else None
    common_index = _common_return_index(strategy_returns, benchmark_returns)
    strategy_returns = strategy_returns.reindex(common_index).dropna()
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(common_index).dropna()

    windows = _build_launch_windows(
        strategy_returns,
        benchmark_returns=benchmark_returns,
        horizons=horizons,
        ramp_weeks=ramp_weeks,
        primary_horizon=primary_horizon,
        start_frequency=start_frequency,
        bad_start_drawdown=bad_start_drawdown,
        initial_ramp_fraction=initial_ramp_fraction,
        target_fraction=target_fraction,
    )
    diagnostics = _current_launch_diagnostics(
        strategy_result,
        current_state=current_state,
    )
    current_score = _current_score_from_diagnostics(diagnostics)
    summary = summarize_launch_windows(windows, current_score=current_score)
    recommendation = _recommend_launch(summary, primary_horizon=primary_horizon)
    ramp_plan = build_launch_ramp_plan(
        capital_to_launch=capital_to_launch,
        target_fraction=target_fraction,
        ramp_weeks=int(recommendation.get("ramp_weeks", 0) or 0),
        initial_ramp_fraction=initial_ramp_fraction,
        launch_label=str(recommendation.get("launch_label", "wait")),
    )
    return LaunchReadinessRun(
        windows=windows,
        summary=summary,
        diagnostics=diagnostics,
        ramp_plan=ramp_plan,
        recommendation=recommendation,
    )


def build_aggregate_launch_readiness(
    strategy_results: Mapping[str, BacktestResult],
    *,
    benchmark_result: BacktestResult | None = None,
    current_state: Any | None = None,
    horizons: Mapping[str, int] = DEFAULT_ENTRY_HORIZONS,
    ramp_weeks: Sequence[int] = DEFAULT_LAUNCH_RAMP_WEEKS,
    start_frequency: str = DEFAULT_LAUNCH_START_FREQUENCY,
    target_fraction: float = DEFAULT_LAUNCH_TARGET_FRACTION,
    bad_start_drawdown: float = DEFAULT_LAUNCH_BAD_START_DRAWDOWN,
    initial_ramp_fraction: float = DEFAULT_LAUNCH_INITIAL_RAMP_FRACTION,
    protocol_small_spread: float = DEFAULT_LAUNCH_PROTOCOL_SMALL_SPREAD,
    protocol_material_spread: float = DEFAULT_LAUNCH_PROTOCOL_MATERIAL_SPREAD,
) -> AggregateLaunchReadinessRun:
    """Summarize launch-readiness behavior across a strategy shelf."""

    summaries: list[pd.DataFrame] = []
    windows: list[pd.DataFrame] = []
    for strategy_name, strategy_result in strategy_results.items():
        if strategy_result is None:
            continue
        run = build_launch_readiness(
            strategy_result,
            benchmark_result=benchmark_result,
            current_state=current_state,
            horizons=horizons,
            ramp_weeks=ramp_weeks,
            start_frequency=start_frequency,
            target_fraction=target_fraction,
            bad_start_drawdown=bad_start_drawdown,
            initial_ramp_fraction=initial_ramp_fraction,
        )
        if not run.summary.empty:
            summary = run.summary.copy()
            summary["strategy"] = str(strategy_name)
            summaries.append(summary)
        if not run.windows.empty:
            window = run.windows.copy()
            window["strategy"] = str(strategy_name)
            windows.append(window)

    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    windows_frame = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    strategy_horizon_summary = _best_launch_rows(summary_frame, horizons=horizons)
    protocol_separation = _protocol_separation_frame(
        windows_frame,
        small_spread=protocol_small_spread,
        material_spread=protocol_material_spread,
    )
    return AggregateLaunchReadinessRun(
        strategy_horizon_summary=strategy_horizon_summary,
        horizon_label_counts=_horizon_label_counts(strategy_horizon_summary, horizons=horizons),
        horizon_transition_matrix=_horizon_transition_matrix(
            strategy_horizon_summary,
            horizons=horizons,
        ),
        protocol_separation=protocol_separation,
        protocol_separation_by_horizon=_protocol_separation_by_horizon(
            protocol_separation,
            horizons=horizons,
        ),
        strategy_count=int(strategy_horizon_summary["strategy"].nunique())
        if "strategy" in strategy_horizon_summary
        else 0,
    )


def summarize_launch_windows(
    windows: pd.DataFrame,
    *,
    current_score: float,
) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()

    summary = (
        windows.groupby(["protocol", "ramp_weeks", "horizon"], as_index=False)
        .agg(
            windows=("start_date", "count"),
            positive_return_rate=("positive_return", "mean"),
            beat_rate=("beats_benchmark", "mean"),
            bad_start_rate=("bad_start", "mean"),
            median_return=("total_return", "median"),
            worst_return=("total_return", "min"),
            median_excess_return=("excess_return", "median"),
            worst_excess_return=("excess_return", "min"),
            median_max_drawdown=("max_drawdown", "median"),
            worst_max_drawdown=("max_drawdown", "min"),
            median_first_month_drawdown=("first_month_drawdown", "median"),
            worst_first_month_drawdown=("first_month_drawdown", "min"),
        )
        .reset_index(drop=True)
    )
    summary["historical_entry_score"] = summary.apply(_historical_entry_score, axis=1)
    summary["current_entry_score"] = float(current_score)
    summary["launch_score"] = (
        0.65 * summary["historical_entry_score"] + 0.35 * summary["current_entry_score"]
    ).clip(0.0, 1.0)
    summary.loc[summary["windows"] < DEFAULT_LAUNCH_MIN_WINDOWS, "launch_score"] *= 0.65
    summary["launch_label"] = summary["launch_score"].map(_launch_label)
    summary["launch_action"] = summary.apply(_launch_action, axis=1)
    return summary.sort_values(
        ["horizon", "launch_score", "bad_start_rate", "median_excess_return"],
        ascending=[True, False, True, False],
    )


def build_launch_ramp_plan(
    *,
    capital_to_launch: float,
    target_fraction: float,
    ramp_weeks: int,
    initial_ramp_fraction: float = DEFAULT_LAUNCH_INITIAL_RAMP_FRACTION,
    launch_label: str = "set",
) -> pd.DataFrame:
    target_capital = max(float(capital_to_launch), 0.0) * _clip01(target_fraction)
    if launch_label in {"wait", "no_go"}:
        return pd.DataFrame(
            [
                {
                    "week": 0,
                    "target_fraction_of_strategy": 0.0,
                    "account_fraction_deployed": 0.0,
                    "capital_deployed": 0.0,
                    "cash_reserved": target_capital,
                    "instruction": "Do not launch yet; keep the intended test capital reserved.",
                }
            ]
        )
    if launch_label == "set" and ramp_weeks <= 0:
        starter_fraction = _clip01(initial_ramp_fraction)
        deployed = target_capital * starter_fraction
        return pd.DataFrame(
            [
                {
                    "week": 0,
                    "target_fraction_of_strategy": starter_fraction,
                    "account_fraction_deployed": _clip01(target_fraction) * starter_fraction,
                    "capital_deployed": deployed,
                    "cash_reserved": target_capital - deployed,
                    "instruction": (
                        "Open only a starter sleeve; re-check the launch gate before scaling."
                    ),
                }
            ]
        )
    if ramp_weeks <= 0:
        return pd.DataFrame(
            [
                {
                    "week": 0,
                    "target_fraction_of_strategy": 1.0,
                    "account_fraction_deployed": _clip01(target_fraction),
                    "capital_deployed": target_capital,
                    "cash_reserved": 0.0,
                    "instruction": "Launch the intended sleeve immediately.",
                }
            ]
        )

    weeks = list(range(0, ramp_weeks + 1))
    rows: list[dict[str, object]] = []
    for week in weeks:
        progress = week / max(ramp_weeks, 1)
        strategy_fraction = _clip01(
            initial_ramp_fraction + (1.0 - initial_ramp_fraction) * progress
        )
        deployed = target_capital * strategy_fraction
        rows.append(
            {
                "week": week,
                "target_fraction_of_strategy": strategy_fraction,
                "account_fraction_deployed": _clip01(target_fraction) * strategy_fraction,
                "capital_deployed": deployed,
                "cash_reserved": target_capital - deployed,
                "instruction": _ramp_instruction(week, ramp_weeks),
            }
        )
    return pd.DataFrame(rows)


def _best_launch_rows(summary: pd.DataFrame, *, horizons: Mapping[str, int]) -> pd.DataFrame:
    columns = [
        "strategy",
        "horizon",
        "horizon_order",
        "protocol",
        "ramp_weeks",
        "launch_label",
        "launch_action",
        "launch_score",
        "positive_return_rate",
        "beat_rate",
        "bad_start_rate",
        "median_return",
        "median_excess_return",
        "median_max_drawdown",
        "windows",
    ]
    if summary.empty or not {"strategy", "horizon"}.issubset(summary.columns):
        return pd.DataFrame(columns=columns)
    frame = summary.copy()
    frame["horizon_order"] = frame["horizon"].astype(str).map(_horizon_rank(horizons))
    for column in [
        "launch_score",
        "bad_start_rate",
        "median_excess_return",
        "median_return",
        "median_max_drawdown",
        "windows",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    sorted_frame = frame.sort_values(
        ["strategy", "horizon_order", "launch_score", "bad_start_rate", "median_excess_return"],
        ascending=[True, True, False, True, False],
    )
    best = sorted_frame.groupby(["strategy", "horizon"], as_index=False).head(1)
    return best[[column for column in columns if column in best]].reset_index(drop=True)


def _horizon_label_counts(
    strategy_horizon_summary: pd.DataFrame,
    *,
    horizons: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = _launch_labels()
    for horizon in _horizon_names(horizons):
        frame = (
            strategy_horizon_summary[
                strategy_horizon_summary["horizon"].astype(str).eq(str(horizon))
            ]
            if not strategy_horizon_summary.empty and "horizon" in strategy_horizon_summary
            else pd.DataFrame()
        )
        denominator = max(int(frame["strategy"].nunique()), 1) if "strategy" in frame else 1
        counts = frame["launch_label"].astype(str).value_counts() if "launch_label" in frame else {}
        for label in labels:
            count = int(counts.get(label, 0)) if hasattr(counts, "get") else 0
            rows.append(
                {
                    "horizon": horizon,
                    "horizon_order": _horizon_rank(horizons).get(horizon, 999_999),
                    "launch_label": label,
                    "count": count,
                    "share": count / denominator if denominator else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _horizon_transition_matrix(
    strategy_horizon_summary: pd.DataFrame,
    *,
    horizons: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = _launch_labels()
    horizon_names = _horizon_names(horizons)
    if not strategy_horizon_summary.empty and {"strategy", "horizon", "launch_label"}.issubset(
        strategy_horizon_summary.columns
    ):
        pivot = strategy_horizon_summary.pivot_table(
            index="strategy",
            columns="horizon",
            values="launch_label",
            aggfunc="first",
        )
    else:
        pivot = pd.DataFrame()
    for from_horizon, to_horizon in zip(horizon_names, horizon_names[1:], strict=False):
        if pivot.empty or from_horizon not in pivot or to_horizon not in pivot:
            pair = pd.DataFrame(columns=["strategy", "from_label", "to_label"])
        else:
            pair = (
                pivot[[from_horizon, to_horizon]]
                .dropna()
                .rename(columns={from_horizon: "from_label", to_horizon: "to_label"})
                .reset_index()
            )
            pair["from_label"] = pair["from_label"].astype(str)
            pair["to_label"] = pair["to_label"].astype(str)
        for from_label in labels:
            for to_label in labels:
                matches = pair[
                    pair["from_label"].eq(from_label) & pair["to_label"].eq(to_label)
                ]
                examples = ", ".join(matches["strategy"].astype(str).head(5).tolist())
                rows.append(
                    {
                        "from_horizon": from_horizon,
                        "to_horizon": to_horizon,
                        "horizon_pair": f"{from_horizon} -> {to_horizon}",
                        "transition": f"{from_label} -> {to_label}",
                        "from_label": from_label,
                        "to_label": to_label,
                        "direction": _transition_direction(from_label, to_label),
                        "count": int(len(matches)),
                        "example_strategies": examples,
                    }
                )
    return pd.DataFrame(rows)


def _protocol_separation_frame(
    windows: pd.DataFrame,
    *,
    small_spread: float,
    material_spread: float,
) -> pd.DataFrame:
    columns = [
        "strategy",
        "horizon",
        "horizon_order",
        "best_protocol",
        "worst_protocol",
        "best_median_return",
        "worst_median_return",
        "protocol_spread",
        "separation_label",
    ]
    if windows.empty or not {"strategy", "horizon", "protocol", "total_return"}.issubset(
        windows.columns
    ):
        return pd.DataFrame(columns=columns)
    frame = windows.copy()
    frame["total_return"] = pd.to_numeric(frame["total_return"], errors="coerce")
    medians = (
        frame.dropna(subset=["total_return"])
        .groupby(["strategy", "horizon", "protocol"], as_index=False)["total_return"]
        .median()
    )
    rows: list[dict[str, object]] = []
    for (strategy, horizon), group in medians.groupby(["strategy", "horizon"]):
        if len(group) < 2:
            continue
        ranked = group.sort_values("total_return", ascending=False)
        best = ranked.iloc[0]
        worst = ranked.iloc[-1]
        spread = float(best["total_return"] - worst["total_return"])
        rows.append(
            {
                "strategy": str(strategy),
                "horizon": str(horizon),
                "horizon_order": _horizon_order_value(str(horizon)),
                "best_protocol": str(best["protocol"]),
                "worst_protocol": str(worst["protocol"]),
                "best_median_return": float(best["total_return"]),
                "worst_median_return": float(worst["total_return"]),
                "protocol_spread": spread,
                "separation_label": _separation_label(
                    spread,
                    small_spread=small_spread,
                    material_spread=material_spread,
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["horizon_order", "protocol_spread"],
        ascending=[True, False],
        ignore_index=True,
    )


def _protocol_separation_by_horizon(
    protocol_separation: pd.DataFrame,
    *,
    horizons: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in _horizon_names(horizons):
        frame = (
            protocol_separation[protocol_separation["horizon"].astype(str).eq(str(horizon))]
            if not protocol_separation.empty and "horizon" in protocol_separation
            else pd.DataFrame()
        )
        strategies = int(frame["strategy"].nunique()) if "strategy" in frame else 0
        labels = (
            frame["separation_label"].astype(str)
            if "separation_label" in frame
            else pd.Series(dtype=str)
        )
        spreads = (
            pd.to_numeric(frame["protocol_spread"], errors="coerce").dropna()
            if "protocol_spread" in frame
            else pd.Series(dtype=float)
        )
        denominator = max(strategies, 1)
        rows.append(
            {
                "horizon": horizon,
                "horizon_order": _horizon_rank(horizons).get(horizon, 999_999),
                "strategies": strategies,
                "median_protocol_spread": float(spreads.median()) if not spreads.empty else 0.0,
                "mean_protocol_spread": float(spreads.mean()) if not spreads.empty else 0.0,
                "material_separation_rate": float((labels == "material").sum() / denominator),
                "small_separation_rate": float((labels == "small").sum() / denominator),
                "effectively_identical_rate": float(
                    (labels == "effectively_identical").sum() / denominator
                ),
            }
        )
    return pd.DataFrame(rows)


def _horizon_names(horizons: Mapping[str, int]) -> list[str]:
    return sorted([str(name) for name in horizons], key=_horizon_order_value)


def _horizon_rank(horizons: Mapping[str, int]) -> dict[str, int]:
    return {name: index for index, name in enumerate(_horizon_names(horizons))}


def _horizon_order_value(horizon: str) -> int:
    text = str(horizon).strip().lower()
    if text.endswith("w"):
        return int(float(text[:-1]) * 5)
    if text.endswith("m"):
        return int(float(text[:-1]) * 21)
    if text.endswith("y"):
        return int(float(text[:-1]) * 252)
    try:
        return int(float(text))
    except ValueError:
        return 999_999


def _launch_labels() -> list[str]:
    return ["no_go", "wait", "set", "ready"]


def _launch_label_rank(label: str) -> int:
    ranks = {label: index for index, label in enumerate(_launch_labels())}
    return ranks.get(str(label), -1)


def _transition_direction(from_label: str, to_label: str) -> str:
    from_rank = _launch_label_rank(from_label)
    to_rank = _launch_label_rank(to_label)
    if to_rank > from_rank:
        return "upgrade"
    if to_rank < from_rank:
        return "downgrade"
    return "unchanged"


def _separation_label(
    spread: float,
    *,
    small_spread: float,
    material_spread: float,
) -> str:
    if spread < small_spread:
        return "effectively_identical"
    if spread < material_spread:
        return "small"
    return "material"


def _build_launch_windows(
    strategy_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None,
    horizons: Mapping[str, int],
    ramp_weeks: Sequence[int],
    primary_horizon: str,
    start_frequency: str,
    bad_start_drawdown: float,
    initial_ramp_fraction: float,
    target_fraction: float,
) -> pd.DataFrame:
    if strategy_returns.empty:
        return pd.DataFrame()
    start_dates = _sample_start_dates(pd.DatetimeIndex(strategy_returns.index), start_frequency)
    rows: list[dict[str, object]] = []
    for start_date in start_dates:
        start_position = strategy_returns.index.get_loc(start_date)
        if not isinstance(start_position, int):
            continue
        for horizon_name, horizon_days in horizons.items():
            end_position = start_position + int(horizon_days)
            if end_position >= len(strategy_returns):
                continue
            strategy_slice = _clean_return_slice(
                strategy_returns.iloc[start_position + 1 : end_position + 1]
            )
            if strategy_slice.empty:
                continue
            benchmark_slice = (
                _clean_return_slice(benchmark_returns.reindex(strategy_slice.index))
                if benchmark_returns is not None
                else pd.Series(0.0, index=strategy_slice.index)
            )
            common_window_index = pd.DatetimeIndex(strategy_slice.index).intersection(
                pd.DatetimeIndex(benchmark_slice.index)
            )
            strategy_slice = strategy_slice.reindex(common_window_index).dropna()
            benchmark_slice = benchmark_slice.reindex(common_window_index).dropna()
            if len(benchmark_slice) != len(strategy_slice) or strategy_slice.empty:
                continue
            for weeks in sorted({int(value) for value in ramp_weeks}):
                protocol_return = _protocol_returns(
                    strategy_slice,
                    ramp_weeks=weeks,
                    initial_ramp_fraction=initial_ramp_fraction,
                    target_fraction=target_fraction,
                )
                protocol_equity = (1.0 + protocol_return).cumprod()
                benchmark_equity = (1.0 + benchmark_slice).cumprod()
                total_return = _total_return(protocol_equity)
                benchmark_return = _total_return(benchmark_equity)
                max_drawdown = _window_drawdown(protocol_equity)
                first_month_drawdown = _window_drawdown(protocol_equity.head(min(21, len(protocol_equity))))
                cagr = _window_cagr(
                    total_return,
                    strategy_slice.index[0],
                    strategy_slice.index[-1],
                )
                if not all(
                    _is_finite(value)
                    for value in [
                        total_return,
                        cagr,
                        max_drawdown,
                        first_month_drawdown,
                        benchmark_return,
                    ]
                ):
                    continue
                rows.append(
                    {
                        "strategy": strategy_returns.name or "strategy",
                        "protocol": _protocol_name(weeks),
                        "ramp_weeks": weeks,
                        "horizon": horizon_name,
                        "is_primary_horizon": horizon_name == primary_horizon,
                        "horizon_trading_days": int(horizon_days),
                        "start_date": start_date.date().isoformat(),
                        "end_date": strategy_slice.index[-1].date().isoformat(),
                        "total_return": total_return,
                        "cagr": cagr,
                        "max_drawdown": max_drawdown,
                        "first_month_drawdown": first_month_drawdown,
                        "benchmark_return": benchmark_return,
                        "excess_return": total_return - benchmark_return,
                        "beats_benchmark": total_return > benchmark_return,
                        "positive_return": total_return > 0.0,
                        "bad_start": bool(
                            total_return < 0.0 or first_month_drawdown <= bad_start_drawdown
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _protocol_returns(
    returns: pd.Series,
    *,
    ramp_weeks: int,
    initial_ramp_fraction: float,
    target_fraction: float,
) -> pd.Series:
    exposure = _exposure_path(
        len(returns),
        ramp_weeks=ramp_weeks,
        initial_ramp_fraction=initial_ramp_fraction,
        target_fraction=target_fraction,
    )
    exposure.index = returns.index
    return returns.astype(float) * exposure


def _exposure_path(
    length: int,
    *,
    ramp_weeks: int,
    initial_ramp_fraction: float,
    target_fraction: float,
) -> pd.Series:
    target = _clip01(target_fraction)
    if length <= 0:
        return pd.Series(dtype=float)
    if ramp_weeks <= 0:
        return pd.Series(target, index=range(length), dtype=float)
    ramp_days = max(int(ramp_weeks) * 5, 1)
    initial = _clip01(initial_ramp_fraction) * target
    values = []
    for position in range(length):
        progress = min(position / ramp_days, 1.0)
        values.append(initial + (target - initial) * progress)
    return pd.Series(values, index=range(length), dtype=float)


def _current_launch_diagnostics(
    strategy_result: BacktestResult,
    *,
    current_state: Any | None,
) -> pd.DataFrame:
    strategy_equity = _finite_numeric_series(strategy_result.equity)
    drawdown = (
        strategy_equity / strategy_equity.cummax() - 1.0
        if not strategy_equity.empty
        else pd.Series(dtype=float)
    )
    diagnostics = [
        _diagnostic_row(
            "strategy_current_drawdown",
            _latest_value(drawdown),
            "Strategy drawdown now",
            "Launching near a fresh equity high is usually easier than launching during strategy stress.",
        ),
        _diagnostic_row(
            "strategy_return_1m",
            _window_return(strategy_equity, 21),
            "Strategy 1M return",
            "Positive near-term behavior supports launch only when risk controls also agree.",
        ),
        _diagnostic_row(
            "strategy_return_3m",
            _window_return(strategy_equity, 63),
            "Strategy 3M return",
            "Medium-term repair helps avoid starting into a still-breaking strategy.",
        ),
    ]
    if current_state is not None:
        probabilities = _scenario_probabilities(current_state)
        diagnostics.extend(
            [
                _diagnostic_row(
                    "risk_status",
                    str(getattr(current_state, "risk_status", "unknown")).upper(),
                    "Current operating risk status",
                    "Launch Lab treats yellow/orange/red as entry friction, not an automatic rebalance command.",
                ),
                _diagnostic_row(
                    "risk_score",
                    _safe_float(getattr(current_state, "risk_score", None)),
                    "Current risk score",
                    "Higher scores support launch; lower scores argue for staging or waiting.",
                ),
                _diagnostic_row(
                    "risk_off_1m_probability",
                    probabilities.get("risk_off", 0.0),
                    "1M risk-off probability",
                    "High risk-off probability favors staged launch or waiting.",
                ),
                _diagnostic_row(
                    "transition_1m_probability",
                    probabilities.get("transition", 0.0),
                    "1M transition probability",
                    "Transition regimes can still be investable, but full-size launch should be harder.",
                ),
                *_confirmation_diagnostics(current_state),
            ]
        )
    frame = pd.DataFrame(diagnostics)
    frame["score_impact"] = frame.apply(_diagnostic_score_impact, axis=1)
    return frame


def _scenario_probabilities(current_state: Any) -> dict[str, float]:
    lattice = getattr(current_state, "scenario_lattice", pd.DataFrame())
    if not isinstance(lattice, pd.DataFrame) or lattice.empty:
        return {}
    frame = lattice.copy()
    if "horizon" in frame:
        one_month = frame[frame["horizon"].astype(str).eq("1m")]
        if not one_month.empty:
            frame = one_month
    if not {"risk_bucket", "probability"}.issubset(frame.columns):
        return {}
    values = pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0)
    if values.sum() > 1.5:
        values = values / 100.0
    grouped = values.groupby(frame["risk_bucket"].astype(str)).sum()
    return {str(index): float(value) for index, value in grouped.items()}


def _confirmation_diagnostics(current_state: Any) -> list[dict[str, object]]:
    matrix = getattr(current_state, "confirmation_matrix", pd.DataFrame())
    if not isinstance(matrix, pd.DataFrame) or matrix.empty or "score" not in matrix:
        return []
    scores = pd.to_numeric(matrix["score"], errors="coerce").fillna(0.0)
    total = max(len(scores), 1)
    return [
        _diagnostic_row(
            "bullish_confirmation_share",
            float((scores > 0).sum() / total),
            "Bullish confirmation share",
            "Launch is easier when more market, credit, breadth, and trend checks are supportive.",
        ),
        _diagnostic_row(
            "bearish_confirmation_share",
            float((scores < 0).sum() / total),
            "Bearish confirmation share",
            "Bearish confirmation raises the bar for starting fresh risk capital.",
        ),
    ]


def _current_score_from_diagnostics(diagnostics: pd.DataFrame) -> float:
    if diagnostics.empty:
        return 0.50
    score = 0.72
    values = dict(zip(diagnostics["metric"], diagnostics["value"], strict=False))
    status = str(values.get("risk_status", "UNKNOWN")).lower()
    score += {
        "green": 0.10,
        "yellow": -0.06,
        "orange": -0.18,
        "red": -0.35,
    }.get(status, -0.02)
    risk_off = _safe_float(values.get("risk_off_1m_probability")) or 0.0
    transition = _safe_float(values.get("transition_1m_probability")) or 0.0
    bearish = _safe_float(values.get("bearish_confirmation_share")) or 0.0
    bullish = _safe_float(values.get("bullish_confirmation_share")) or 0.0
    current_drawdown = _safe_float(values.get("strategy_current_drawdown")) or 0.0
    one_month_return = _safe_float(values.get("strategy_return_1m")) or 0.0
    score -= max(risk_off - 0.20, 0.0) * 0.55
    score -= max(transition - 0.35, 0.0) * 0.25
    score -= max(bearish - 0.20, 0.0) * 0.35
    score += max(bullish - 0.45, 0.0) * 0.20
    if current_drawdown <= -0.10:
        score -= 0.08
    elif current_drawdown >= -0.03:
        score += 0.04
    if one_month_return > 0.0:
        score += 0.03
    return _clip01(score)


def _historical_entry_score(row: pd.Series) -> float:
    positive = _safe_float(row.get("positive_return_rate")) or 0.0
    beat = _safe_float(row.get("beat_rate")) or 0.0
    bad = _safe_float(row.get("bad_start_rate")) or 0.0
    median_excess = _safe_float(row.get("median_excess_return")) or 0.0
    median_drawdown = _safe_float(row.get("median_max_drawdown")) or 0.0
    excess_score = 0.5 + min(max(median_excess / 0.10, -0.5), 0.5)
    drawdown_score = 1.0 - min(abs(min(median_drawdown, 0.0)) / 0.20, 1.0)
    return _clip01(
        0.35 * positive
        + 0.25 * beat
        + 0.20 * (1.0 - bad)
        + 0.12 * excess_score
        + 0.08 * drawdown_score
    )


def _recommend_launch(summary: pd.DataFrame, *, primary_horizon: str) -> dict[str, object]:
    if summary.empty:
        return {
            "launch_label": "no_go",
            "launch_action": "No launch evidence is available.",
            "protocol": "none",
            "ramp_weeks": 0,
            "launch_score": 0.0,
        }
    candidates = summary[summary["horizon"].astype(str).eq(primary_horizon)].copy()
    if candidates.empty:
        candidates = summary.copy()
    best = candidates.sort_values(
        ["launch_score", "bad_start_rate", "median_excess_return"],
        ascending=[False, True, False],
    ).iloc[0]
    output = best.to_dict()
    output["launch_read"] = _launch_read(best)
    output["operating_boundary"] = (
        "Launch guidance applies to new or scale-up capital. Once the book is running, "
        "use Book Alignment and Forward Test for daily target drift."
    )
    return output


def _launch_label(score: float) -> str:
    if score >= DEFAULT_LAUNCH_READY_SCORE:
        return "ready"
    if score >= DEFAULT_LAUNCH_SET_SCORE:
        return "set"
    if score >= DEFAULT_LAUNCH_WAIT_SCORE:
        return "wait"
    return "no_go"


def _launch_action(row: pd.Series) -> str:
    label = str(row.get("launch_label") or _launch_label(float(row.get("launch_score", 0.0))))
    ramp = int(row.get("ramp_weeks", 0) or 0)
    if label == "ready" and ramp <= 0:
        return "Launch the intended sleeve now."
    if label == "set" and ramp <= 0:
        return "Open only a starter sleeve; do not fully launch yet."
    if label in {"ready", "set"} and ramp > 0:
        return f"Stage in over {ramp} weeks."
    if label == "wait":
        return "Keep the strategy on deck; wait for better entry confirmation."
    return "Do not launch from this setup."


def _launch_read(row: pd.Series) -> str:
    return (
        f"{str(row.get('launch_label', 'wait')).replace('_', ' ').title()}: "
        f"{row.get('launch_action', 'Review launch evidence.')} "
        f"Historical positive-start rate is {_format_percent(row.get('positive_return_rate'))}, "
        f"beat rate is {_format_percent(row.get('beat_rate'))}, and bad-start rate is "
        f"{_format_percent(row.get('bad_start_rate'))} over {int(row.get('windows', 0) or 0)} "
        f"{row.get('horizon', 'window')} start windows."
    )


def _diagnostic_score_impact(row: pd.Series) -> str:
    metric = str(row.get("metric", ""))
    value = row.get("value")
    numeric = _safe_float(value)
    if metric == "risk_status":
        status = str(value).lower()
        if status in {"orange", "red"}:
            return "launch_friction"
        if status == "yellow":
            return "stage_in_bias"
        return "supports_launch"
    if metric == "risk_off_1m_probability" and numeric is not None:
        return "launch_friction" if numeric >= 0.25 else "neutral"
    if metric == "transition_1m_probability" and numeric is not None:
        return "stage_in_bias" if numeric >= 0.35 else "neutral"
    if metric == "strategy_current_drawdown" and numeric is not None:
        return "launch_friction" if numeric <= -0.10 else "supports_launch"
    if metric == "bearish_confirmation_share" and numeric is not None:
        return "launch_friction" if numeric >= 0.25 else "neutral"
    if metric == "bullish_confirmation_share" and numeric is not None:
        return "supports_launch" if numeric >= 0.50 else "neutral"
    return "context"


def _daily_returns(result: BacktestResult | None) -> pd.Series:
    if result is None:
        return pd.Series(dtype=float)
    returns = _clean_return_slice(result.returns)
    returns.name = result.name
    return returns


def _clean_return_slice(returns: pd.Series) -> pd.Series:
    return _finite_numeric_series(returns)


def _finite_numeric_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[clean.map(_is_finite)]
    return clean.astype(float)


def _common_return_index(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> pd.DatetimeIndex:
    if strategy_returns.empty:
        return pd.DatetimeIndex([])
    index = pd.DatetimeIndex(strategy_returns.index)
    if benchmark_returns is not None and not benchmark_returns.empty:
        index = index.intersection(pd.DatetimeIndex(benchmark_returns.index))
    return pd.DatetimeIndex(index).sort_values()


def _sample_start_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if index.empty:
        return index
    periods = index.to_period(frequency)
    first_dates = pd.Series(index, index=index).groupby(periods).first()
    return pd.DatetimeIndex(first_dates.to_list())


def _total_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] - 1.0)


def _window_cagr(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = max((end - start).days / 365.25, 1 / 365.25)
    base = max(1.0 + total_return, 1e-9)
    return float(base ** (1.0 / years) - 1.0)


def _window_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _latest_value(series: pd.Series) -> float | None:
    clean = _finite_numeric_series(series)
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _window_return(equity: pd.Series, periods: int) -> float | None:
    clean = _finite_numeric_series(equity)
    if len(clean) <= periods:
        return None
    start = float(clean.iloc[-periods - 1])
    if abs(start) <= 1e-12:
        return None
    return float(clean.iloc[-1] / start - 1.0)


def _diagnostic_row(
    metric: str,
    value: object,
    read: str,
    interpretation: str,
) -> dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "read": read,
        "interpretation": interpretation,
    }


def _protocol_name(ramp_weeks: int) -> str:
    if ramp_weeks <= 0:
        return "Immediate full launch"
    return f"25% now / {ramp_weeks}w ramp"


def _ramp_instruction(week: int, ramp_weeks: int) -> str:
    if week == 0:
        return "Open a starter sleeve; keep the rest reserved."
    if week >= ramp_weeks:
        return "Complete the intended launch sleeve if conditions remain acceptable."
    return "Add the next staged tranche only if the launch gate has not deteriorated."


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not _is_finite(numeric):
        return None
    return numeric


def _is_finite(value: object) -> bool:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in {float("inf"), float("-inf")}


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _format_percent(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.1%}"
