from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.DEFAULTS import (
    DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS,
    DEFAULT_TREND_HISTORY_SNAPSHOT_LIMIT,
)
from trade_bot.research.driver_rotation import build_driver_rotation_table
from trade_bot.research.narrative_signals import build_narrative_signal_table
from trade_bot.storage.run_store import RunStore
from trade_bot.storage.warehouse import TradingWarehouse

SNAPSHOT_METRIC_COLUMNS = [
    "history_time",
    "snapshot_time",
    "market_date",
    "run_id",
    "risk_score",
    "one_month_risk_off_probability",
    "risk_budget_multiplier",
    "portfolio_risk_multiplier",
    "post_expected_shortfall_95",
    "post_max_stress_loss",
    "post_equity_beta",
    "post_ai_beta",
    "correlation_shift",
    "regime_instability_score",
    "spy_ytd_large_move_share",
]
SNAPSHOT_COMPONENT_COLUMNS = [
    "history_time",
    "snapshot_time",
    "market_date",
    "run_id",
    "component",
    "component_score",
    "latest_value",
    "state",
]
SNAPSHOT_DRIVER_COLUMNS = [
    "history_time",
    "snapshot_time",
    "market_date",
    "run_id",
    "driver",
    "score",
    "state",
]
DRIVER_ROTATION_COLUMNS = [
    "history_time",
    "snapshot_time",
    "market_date",
    "run_id",
    "driver",
    "driver_label",
    "current_activation",
    "proven_relevance",
    "change_30d",
    "change_90d",
    "model_role",
]


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_snapshot_trend_frames(
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
    limit: int = DEFAULT_TREND_HISTORY_SNAPSHOT_LIMIT,
    *,
    force_snapshot_reconstruction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_snapshot_trend_frames(
        store_path_string,
        artifact_dir_string,
        job_log_dir_string,
        limit=limit,
        force_snapshot_reconstruction=force_snapshot_reconstruction,
    )


def _load_snapshot_trend_frames(
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
    *,
    limit: int,
    force_snapshot_reconstruction: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stored_frames = _load_materialized_operating_history(store_path_string)
    if _has_materialized_operating_history(stored_frames) and not force_snapshot_reconstruction:
        return _clean_combined_snapshot_trend_frames(stored_frames)

    try:
        store = RunStore(
            store_path_string,
            artifact_dir=artifact_dir_string,
            job_log_dir=job_log_dir_string,
            read_only=True,
        )
        snapshots = store.list_snapshots(limit=limit)
    except (duckdb.Error, OSError):
        return _clean_combined_snapshot_trend_frames(stored_frames)
    if snapshots.empty or "run_id" not in snapshots:
        return _clean_combined_snapshot_trend_frames(stored_frames)

    metric_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    scenario_driver_rows: list[dict[str, object]] = []
    driver_rotation_rows: list[dict[str, object]] = []
    for _, snapshot_row in snapshots.iloc[::-1].iterrows():
        run_id = str(snapshot_row["run_id"])
        try:
            run, manifest = store.load_snapshot(run_id)
        except (
            duckdb.Error,
            FileNotFoundError,
            TypeError,
            OSError,
            AttributeError,
            ValueError,
        ):
            continue
        base = _snapshot_base_fields(
            market_date=str(getattr(manifest, "market_date", "")),
            created_at_utc=str(getattr(manifest, "created_at_utc", "")),
            run_id=str(getattr(manifest, "run_id", run_id)),
        )
        metric_rows.append({**base, **_snapshot_metric_fields(run)})
        component_rows.extend(_snapshot_component_rows(run, base))
        scenario_driver_rows.extend(_snapshot_scenario_driver_rows(run, base))
        driver_rotation_rows.extend(_snapshot_driver_rotation_rows(run, base))

    snapshot_frames = (
        pd.DataFrame(metric_rows),
        pd.DataFrame(component_rows),
        pd.DataFrame(scenario_driver_rows),
        pd.DataFrame(driver_rotation_rows),
    )
    return _clean_combined_snapshot_trend_frames((*stored_frames, *snapshot_frames))


def _load_materialized_operating_history(
    store_path_string: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        warehouse = TradingWarehouse(store_path_string, read_only=True)
        return warehouse.operating_history_frames()
    except (duckdb.Error, OSError):
        return _empty_snapshot_trend_frames()


def _has_materialized_operating_history(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> bool:
    metrics = frames[0]
    required_columns = {"history_time", "snapshot_time", "market_date", "run_id"}
    return not metrics.empty and required_columns.issubset(metrics.columns)


@st.cache_data(show_spinner=False, ttl=60)
def load_monitoring_trend_frame(warehouse_path: str) -> pd.DataFrame:
    warehouse = TradingWarehouse(warehouse_path, read_only=True)
    valuations = warehouse.read_table("strategy_daily_valuations")
    if valuations.empty:
        return pd.DataFrame()
    frame = valuations.copy()
    windows = warehouse.list_monitoring_windows(status=None)
    if not windows.empty and "window_id" in frame and "window_id" in windows:
        window_columns = [
            column
            for column in [
                "window_id",
                "window_role",
                "start_date",
                "status",
                "benchmark",
                "capital_base",
            ]
            if column in windows
        ]
        frame = frame.merge(
            windows[window_columns].drop_duplicates("window_id"),
            on="window_id",
            how="left",
        )
    snapshot_metrics = warehouse.read_table("snapshot_strategy_metrics")
    if not snapshot_metrics.empty and "strategy" in snapshot_metrics:
        metric_columns = [
            column
            for column in ["strategy", "max_drawdown", "calmar"]
            if column in snapshot_metrics
        ]
        frame = frame.merge(
            snapshot_metrics[metric_columns]
            .rename(
                columns={
                    "strategy": "strategy_name",
                    "max_drawdown": "snapshot_max_drawdown",
                    "calmar": "snapshot_calmar",
                }
            )
            .drop_duplicates("strategy_name"),
            on="strategy_name",
            how="left",
        )
    frame["valuation_date"] = pd.to_datetime(frame.get("valuation_date"), errors="coerce")
    frame["history_time"] = frame["valuation_date"]
    numeric_columns = [
        "cumulative_return",
        "benchmark_cumulative_return",
        "excess_return",
        "drawdown",
        "beta_adjusted_spy_delta",
        "snapshot_max_drawdown",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if {"drawdown", "snapshot_max_drawdown"}.issubset(frame.columns):
        denominator = frame["snapshot_max_drawdown"].abs().replace(0.0, pd.NA)
        frame["drawdown_envelope_used"] = frame["drawdown"].abs() / denominator
    frame["window_label"] = (
        frame.get("strategy_name", pd.Series("", index=frame.index)).astype(str)
        + " | "
        + frame.get("start_date", pd.Series("", index=frame.index)).astype(str)
    )
    return frame.sort_values(["history_time", "window_label"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60)
def load_simulation_validation_trend_frame(warehouse_path: str, limit: int = 2_000) -> pd.DataFrame:
    warehouse = TradingWarehouse(warehouse_path, read_only=True)
    metrics = warehouse.simulation_validation_metrics(limit=limit)
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], errors="coerce", utc=True)
    frame["history_time"] = frame["created_at_utc"]
    for column in [
        "coverage_error",
        "median_abs_error",
        "launch_action_score",
        "launch_overrisk_rate",
        "constructive_capture_rate",
        "launch_decision_accuracy",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("history_time").reset_index(drop=True)


def compact_metric_line_figure(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    labels: dict[str, str] | None = None,
    title: str,
    yaxis_title: str = "Value",
    percent: bool = False,
    height: int = 260,
) -> go.Figure:
    if frame.empty or "history_time" not in frame:
        return go.Figure()
    labels = labels or {}
    plot_frame = frame.copy()
    plot_frame["history_time"] = pd.to_datetime(plot_frame["history_time"], errors="coerce")
    plot_frame = (
        plot_frame.dropna(subset=["history_time"])
        .sort_values("history_time")
        .drop_duplicates("history_time", keep="last")
    )
    figure = go.Figure()
    for column in columns:
        if column not in plot_frame:
            continue
        values = pd.to_numeric(plot_frame[column], errors="coerce")
        if values.dropna().empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=plot_frame["history_time"],
                y=values,
                mode="lines+markers",
                name=labels.get(column, column.replace("_", " ").title()),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>"
                    if percent
                    else "%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.3f}<extra></extra>"
                ),
            )
        )
    if not figure.data:
        return go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin={"l": 20, "r": 20, "t": 54, "b": 56},
        yaxis={"title": yaxis_title, "tickformat": ".0%" if percent else None},
        xaxis={"title": ""},
        legend={"orientation": "h", "yanchor": "top", "y": -0.20, "xanchor": "left", "x": 0},
    )
    return figure


def long_metric_line_figure(
    frame: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    title: str,
    yaxis_title: str = "Value",
    percent: bool = False,
    top_n: int = 6,
    height: int = 300,
) -> go.Figure:
    if frame.empty or "history_time" not in frame or category_column not in frame:
        return go.Figure()
    if value_column not in frame:
        return go.Figure()
    plot_frame = frame.copy()
    plot_frame["history_time"] = pd.to_datetime(plot_frame["history_time"], errors="coerce")
    plot_frame[value_column] = pd.to_numeric(plot_frame[value_column], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["history_time", value_column, category_column])
    if plot_frame.empty:
        return go.Figure()
    latest_time = plot_frame["history_time"].max()
    latest = plot_frame[plot_frame["history_time"] == latest_time]
    categories = (
        latest.assign(_abs=latest[value_column].abs())
        .sort_values("_abs", ascending=False)[category_column]
        .astype(str)
        .drop_duplicates()
        .head(top_n)
        .tolist()
    )
    plot_frame = plot_frame[plot_frame[category_column].astype(str).isin(categories)]
    figure = go.Figure()
    for category, group in plot_frame.groupby(category_column, sort=False):
        ordered = group.sort_values("history_time")
        figure.add_trace(
            go.Scatter(
                x=ordered["history_time"],
                y=ordered[value_column],
                mode="lines+markers",
                name=str(category).replace("_", " ").title(),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>"
                    if percent
                    else "%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.3f}<extra></extra>"
                ),
            )
        )
    if not figure.data:
        return go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin={"l": 20, "r": 20, "t": 54, "b": 72},
        yaxis={"title": yaxis_title, "tickformat": ".0%" if percent else None},
        xaxis={"title": ""},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "xanchor": "left", "x": 0},
    )
    return figure


def filter_history_time_range(
    frame: pd.DataFrame,
    range_choice: str,
    *,
    custom_start: object | None = None,
    custom_end: object | None = None,
    time_column: str = "history_time",
) -> pd.DataFrame:
    if frame.empty or time_column not in frame:
        return frame
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working[working[time_column].notna()].sort_values(time_column)
    if working.empty or range_choice == "All":
        return working
    end = pd.to_datetime(custom_end) if custom_end is not None else working[time_column].max()
    if range_choice == "Custom":
        start = (
            pd.to_datetime(custom_start) if custom_start is not None else working[time_column].min()
        )
    elif range_choice == "YTD":
        start = pd.Timestamp(year=int(end.year), month=1, day=1)
    else:
        days = {
            "1M": 31,
            "3M": 92,
            "6M": 183,
            "1Y": 365,
            "3Y": 365 * 3,
            "5Y": 365 * 5,
        }.get(range_choice, 365)
        start = end - pd.Timedelta(days=days)
    return working[(working[time_column] >= start) & (working[time_column] <= end)]


def latest_per_market_date(frame: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    if frame.empty or "market_date" not in frame or "snapshot_time" not in frame:
        return frame.copy()
    keys = ["market_date", *(subset or [])]
    return (
        frame.sort_values("snapshot_time")
        .drop_duplicates(keys, keep="last")
        .sort_values(["market_date", *(subset or [])])
        .reset_index(drop=True)
    )


def _empty_snapshot_trend_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=SNAPSHOT_METRIC_COLUMNS),
        pd.DataFrame(columns=SNAPSHOT_COMPONENT_COLUMNS),
        pd.DataFrame(columns=SNAPSHOT_DRIVER_COLUMNS),
        pd.DataFrame(columns=DRIVER_ROTATION_COLUMNS),
    )


def _clean_combined_snapshot_trend_frames(
    frames: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frames) == 4:
        metric_frame, component_frame, scenario_driver_frame, driver_rotation_frame = frames
    else:
        metric_frame = _concat_frames(frames[0], frames[4])
        component_frame = _concat_frames(frames[1], frames[5])
        scenario_driver_frame = _concat_frames(frames[2], frames[6])
        driver_rotation_frame = _concat_frames(frames[3], frames[7])
    return (
        _clean_snapshot_metrics(metric_frame),
        _clean_long_frame(component_frame, SNAPSHOT_COMPONENT_COLUMNS),
        _clean_long_frame(scenario_driver_frame, SNAPSHOT_DRIVER_COLUMNS),
        _clean_long_frame(driver_rotation_frame, DRIVER_ROTATION_COLUMNS),
    )


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    populated = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not populated:
        return pd.DataFrame()
    return pd.concat(populated, ignore_index=True, sort=False)


def _snapshot_base_fields(
    *, market_date: str, created_at_utc: str, run_id: str
) -> dict[str, object]:
    snapshot_time = pd.to_datetime(created_at_utc, errors="coerce", utc=True)
    market_timestamp = pd.to_datetime(market_date, errors="coerce")
    history_time = market_timestamp if pd.notna(market_timestamp) else snapshot_time
    return {
        "history_time": history_time,
        "snapshot_time": snapshot_time,
        "market_date": market_timestamp.date() if pd.notna(market_timestamp) else pd.NaT,
        "run_id": run_id,
    }


def _snapshot_metric_fields(run: Any) -> dict[str, object]:
    current_state = getattr(run, "current_state", None)
    trade_decision = getattr(run, "trade_decision", None)
    portfolio_risk = getattr(run, "portfolio_risk", None)
    decision = _first_row(getattr(trade_decision, "summary", pd.DataFrame()))
    portfolio_summary = _first_row(getattr(portfolio_risk, "summary", pd.DataFrame()))
    correlation = _first_row(getattr(portfolio_risk, "correlation_regime", pd.DataFrame()))
    instability = _first_row(getattr(current_state, "regime_instability", pd.DataFrame()))
    return {
        "risk_score": _safe_float(getattr(current_state, "risk_score", None)),
        "one_month_risk_off_probability": _safe_float(
            decision.get("one_month_risk_off_probability")
        ),
        "risk_budget_multiplier": _safe_float(decision.get("risk_budget_multiplier")),
        "portfolio_risk_multiplier": _coalesce_float(
            portfolio_summary.get("portfolio_risk_multiplier"),
            decision.get("portfolio_risk_multiplier"),
        ),
        "post_expected_shortfall_95": _coalesce_float(
            portfolio_summary.get("post_expected_shortfall_95"),
            decision.get("post_expected_shortfall_95"),
        ),
        "post_max_stress_loss": _coalesce_float(
            portfolio_summary.get("post_max_stress_loss"),
            decision.get("post_max_stress_loss"),
        ),
        "post_equity_beta": _coalesce_float(
            portfolio_summary.get("post_equity_beta"),
            decision.get("post_equity_beta"),
        ),
        "post_ai_beta": _coalesce_float(
            portfolio_summary.get("post_ai_beta"),
            decision.get("post_ai_beta"),
        ),
        "correlation_shift": _coalesce_float(
            correlation.get("correlation_shift"),
            decision.get("correlation_regime_shift"),
        ),
        "regime_instability_score": _safe_float(instability.get("regime_instability_score")),
        "spy_ytd_large_move_share": _safe_float(instability.get("spy_ytd_large_move_share")),
    }


def _snapshot_component_rows(run: Any, base: dict[str, object]) -> list[dict[str, object]]:
    current_state = getattr(run, "current_state", None)
    components = getattr(current_state, "regime_instability_components", pd.DataFrame())
    if components.empty or "component" not in components:
        return []
    rows = []
    for _, row in components.iterrows():
        rows.append(
            {
                **base,
                "component": str(row.get("component", "")),
                "component_score": _safe_float(row.get("component_score")),
                "latest_value": _safe_float(row.get("latest_value")),
                "state": str(row.get("state", "")),
            }
        )
    return rows


def _snapshot_scenario_driver_rows(run: Any, base: dict[str, object]) -> list[dict[str, object]]:
    current_state = getattr(run, "current_state", None)
    drivers = getattr(current_state, "scenario_drivers", pd.DataFrame())
    if drivers.empty or "driver" not in drivers:
        return []
    rows = []
    for _, row in drivers.iterrows():
        rows.append(
            {
                **base,
                "driver": str(row.get("driver", "")),
                "score": _safe_float(row.get("score")),
                "state": str(row.get("state", "")),
            }
        )
    return rows


def _snapshot_driver_rotation_rows(run: Any, base: dict[str, object]) -> list[dict[str, object]]:
    current_state = getattr(run, "current_state", None)
    try:
        narrative_signals = build_narrative_signal_table(
            getattr(run, "prices", pd.DataFrame()),
            news_triage=getattr(getattr(run, "news_monitor", None), "triage", pd.DataFrame()),
            events=getattr(getattr(run, "event_risk", None), "events", ()),
        )
        rotation = build_driver_rotation_table(
            getattr(run, "prices", pd.DataFrame()),
            current_state,
            narrative_signals=narrative_signals,
            news_triage=getattr(getattr(run, "news_monitor", None), "triage", pd.DataFrame()),
        )
    except (TypeError, ValueError, AttributeError, KeyError):
        return []
    if rotation.empty or "driver" not in rotation:
        return []
    rows = []
    for _, row in rotation.iterrows():
        rows.append(
            {
                **base,
                "driver": str(row.get("driver", "")),
                "driver_label": str(row.get("driver_label", row.get("driver", ""))),
                "current_activation": _safe_float(row.get("current_activation")),
                "proven_relevance": _safe_float(row.get("proven_relevance")),
                "change_30d": _safe_float(row.get("change_30d")),
                "change_90d": _safe_float(row.get("change_90d")),
                "model_role": str(row.get("model_role", "")),
            }
        )
    return rows


def _clean_snapshot_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SNAPSHOT_METRIC_COLUMNS)
    output = frame.copy()
    output["history_time"] = pd.to_datetime(output["history_time"], errors="coerce")
    output["snapshot_time"] = pd.to_datetime(output["snapshot_time"], errors="coerce", utc=True)
    output["market_date"] = pd.to_datetime(output["market_date"], errors="coerce").dt.date
    for column in SNAPSHOT_METRIC_COLUMNS:
        if column in {"history_time", "snapshot_time", "market_date", "run_id"}:
            continue
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[[column for column in SNAPSHOT_METRIC_COLUMNS if column in output]].sort_values(
        ["history_time", "snapshot_time"]
    )


def _clean_long_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output = frame.copy()
    output["history_time"] = pd.to_datetime(output["history_time"], errors="coerce")
    output["snapshot_time"] = pd.to_datetime(output["snapshot_time"], errors="coerce", utc=True)
    output["market_date"] = pd.to_datetime(output["market_date"], errors="coerce").dt.date
    for column in [
        "component_score",
        "latest_value",
        "score",
        "current_activation",
        "proven_relevance",
        "change_30d",
        "change_90d",
    ]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[[column for column in columns if column in output]].sort_values(
        ["history_time", "snapshot_time"]
    )


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame.iloc[0]
    return pd.Series(dtype=object)


def _safe_float(value: object) -> float | None:
    try:
        output = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(output):
        return None
    return output


def _coalesce_float(*values: object) -> float | None:
    for value in values:
        output = _safe_float(value)
        if output is not None:
            return output
    return None
