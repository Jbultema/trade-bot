from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.launch_readiness import LaunchReadinessRun

CAPITAL_PRESETS = (1_000.0, 2_500.0, 5_000.0, 10_000.0)
EXPERIMENT_HORIZON_DAYS = {"1m": 21, "3m": 63, "6m": 126}
PRIMARY_EXPERIMENT_BENCHMARK = "buy_hold_qqq"
SECONDARY_EXPERIMENT_BENCHMARK = "buy_hold_spy"
CASH_EXPERIMENT_BENCHMARK = "buy_hold_bil"


@dataclass(frozen=True)
class ExperimentOperatorPlan:
    strategy_name: str
    mode: str
    account: str
    required_horizon: str
    required_trading_days: int
    launch_path: str
    launch_label: str
    confidence_score: float
    recommended_capital: float
    capital_rationale: str
    primary_benchmark: str
    cash_floor: str
    secondary_benchmark: str
    status_label: str
    status_read: str
    summary_cards: pd.DataFrame
    checkpoint_contract: pd.DataFrame
    success_contract: pd.DataFrame
    current_status: pd.DataFrame


def build_experiment_operator_plan(
    strategy_result: BacktestResult,
    *,
    launch_run: LaunchReadinessRun,
    mode: str = "paper",
    account: str = "default_paper_account",
    monitoring_windows: pd.DataFrame | None = None,
    valuations: pd.DataFrame | None = None,
    primary_benchmark: str = PRIMARY_EXPERIMENT_BENCHMARK,
    cash_floor: str = CASH_EXPERIMENT_BENCHMARK,
    secondary_benchmark: str = SECONDARY_EXPERIMENT_BENCHMARK,
) -> ExperimentOperatorPlan:
    """Build the paper/live trial contract for a strategy launch experiment."""

    mode = _normalize_mode(mode)
    account = str(account or "default_paper_account")
    recommendation = dict(launch_run.recommendation or {})
    launch_label = str(recommendation.get("launch_label", "no_go"))
    confidence_score = _safe_float(recommendation.get("launch_score"), 0.0)
    signal_cycle_days = estimate_signal_cycle_days(strategy_result)
    required_horizon = recommend_required_horizon(
        launch_label=launch_label,
        confidence_score=confidence_score,
        signal_cycle_days=signal_cycle_days,
    )
    launch_path = recommend_launch_path(launch_label=launch_label, mode=mode)
    recommended_capital, capital_rationale = recommend_trial_capital(
        confidence_score=confidence_score,
        launch_label=launch_label,
        mode=mode,
    )
    max_drawdown = _max_drawdown(strategy_result.returns)
    current_status = evaluate_current_experiment_status(
        strategy_result,
        monitoring_windows=monitoring_windows,
        valuations=valuations,
        mode=mode,
        account=account,
        required_horizon=required_horizon,
        max_drawdown=max_drawdown,
    )
    status_label, status_read = _status_read(current_status, required_horizon)
    summary_cards = _summary_cards(
        required_horizon=required_horizon,
        launch_path=launch_path,
        confidence_score=confidence_score,
        recommended_capital=recommended_capital,
        signal_cycle_days=signal_cycle_days,
        status_label=status_label,
    )
    return ExperimentOperatorPlan(
        strategy_name=strategy_result.name,
        mode=mode,
        account=account,
        required_horizon=required_horizon,
        required_trading_days=EXPERIMENT_HORIZON_DAYS[required_horizon],
        launch_path=launch_path,
        launch_label=launch_label,
        confidence_score=confidence_score,
        recommended_capital=recommended_capital,
        capital_rationale=capital_rationale,
        primary_benchmark=primary_benchmark,
        cash_floor=cash_floor,
        secondary_benchmark=secondary_benchmark,
        status_label=status_label,
        status_read=status_read,
        summary_cards=summary_cards,
        checkpoint_contract=build_checkpoint_contract(
            required_horizon=required_horizon,
            primary_benchmark=primary_benchmark,
            cash_floor=cash_floor,
        ),
        success_contract=build_success_contract(
            primary_benchmark=primary_benchmark,
            cash_floor=cash_floor,
            secondary_benchmark=secondary_benchmark,
        ),
        current_status=current_status,
    )


def estimate_signal_cycle_days(strategy_result: BacktestResult) -> int:
    """Estimate how long a forward trial needs before target changes are observable."""

    weights = getattr(strategy_result, "target_weights", pd.DataFrame())
    if isinstance(weights, pd.DataFrame) and not weights.empty and len(weights) >= 3:
        material_change = weights.diff().abs().sum(axis=1).fillna(0.0) >= 0.15
        change_dates = pd.DatetimeIndex(weights.index[material_change])
        if len(change_dates) >= 3:
            gaps = pd.Series(change_dates).diff().dt.days.dropna()
            if not gaps.empty:
                return int(max(5, min(round(float(gaps.median()) * 5.0 / 7.0), 126)))

    turnover = getattr(strategy_result, "turnover", pd.Series(dtype=float))
    if isinstance(turnover, pd.Series) and not turnover.empty:
        average_turnover = float(pd.to_numeric(turnover, errors="coerce").dropna().mean() or 0.0)
        if average_turnover >= 0.08:
            return 21
        if average_turnover >= 0.03:
            return 42
    return 63


def recommend_required_horizon(
    *,
    launch_label: str,
    confidence_score: float,
    signal_cycle_days: int,
) -> str:
    """Choose the upfront monitoring horizon that can support a real decision."""

    label = str(launch_label)
    score = float(confidence_score)
    cycle = int(signal_cycle_days)
    if label in {"no_go", "wait"} or score < 0.45:
        return "6m"
    if cycle > 45 or score < 0.70:
        return "3m"
    return "3m"


def recommend_launch_path(*, launch_label: str, mode: str) -> str:
    label = str(launch_label)
    normalized_mode = _normalize_mode(mode)
    if label == "ready":
        return "launch_now" if normalized_mode == "paper" else "live_staged_launch"
    if label == "set":
        return "staged_trial"
    if label == "wait":
        return "paper_watch" if normalized_mode == "paper" else "wait_for_live_entry"
    return "do_not_launch"


def recommend_trial_capital(
    *,
    confidence_score: float,
    launch_label: str,
    mode: str,
    presets: tuple[float, ...] = CAPITAL_PRESETS,
) -> tuple[float, str]:
    label = str(launch_label)
    score = float(confidence_score)
    if label in {"no_go", "wait"} or score < 0.45:
        index = 0
        reason = "Low launch confidence; use the smallest preset as a paper/watch trial."
    elif score < 0.65:
        index = 1
        reason = "Moderate evidence; use a small starter sleeve."
    elif score < 0.80:
        index = 2
        reason = "Good evidence; use a medium trial sleeve."
    else:
        index = 3
        reason = "High confidence; the largest trial preset is acceptable."

    if _normalize_mode(mode) == "live" and label != "ready":
        index = max(0, index - 1)
        reason = f"{reason} Live mode is capped one preset lower unless the launch gate is ready."
    return float(presets[index]), reason


def build_success_contract(
    *,
    primary_benchmark: str,
    cash_floor: str,
    secondary_benchmark: str,
) -> pd.DataFrame:
    rows = [
        {
            "outcome": "validate",
            "definition": (
                f"Beats {primary_benchmark} after costs, stays above the {cash_floor} floor, "
                "keeps drawdown inside the expected envelope, and follows target weights cleanly."
            ),
        },
        {
            "outcome": "continue",
            "definition": (
                "Evidence is promising but sample length is thin, or the strategy is behaving "
                "as designed without enough elapsed time for a validate call."
            ),
        },
        {
            "outcome": "fail",
            "definition": (
                f"Underperforms {primary_benchmark} or cash, breaches drawdown or behavior "
                "expectations, or no longer matches the historical strategy thesis."
            ),
        },
        {
            "outcome": "context",
            "definition": f"{secondary_benchmark} remains secondary market context, not the main hurdle.",
        },
    ]
    return pd.DataFrame(rows)


def build_checkpoint_contract(
    *,
    required_horizon: str,
    primary_benchmark: str,
    cash_floor: str,
) -> pd.DataFrame:
    rows = [
        {
            "checkpoint": "1m",
            "role": "early warning",
            "decision_weight": "low",
            "validate_if": "Do not validate here unless behavior is exceptionally clean.",
            "continue_if": "Target adherence is clean and drawdown is inside the early envelope.",
            "fail_if": f"Meaningfully trails both {primary_benchmark} and {cash_floor}, or breaches the early drawdown envelope.",
        },
        {
            "checkpoint": "3m",
            "role": "first real decision" if required_horizon == "3m" else "intermediate read",
            "decision_weight": "primary" if required_horizon == "3m" else "medium",
            "validate_if": f"Excess return versus {primary_benchmark} is positive after costs and drawdown stayed controlled.",
            "continue_if": "Evidence is mixed but behavior matches the historical thesis.",
            "fail_if": "Benchmark/cash lag is persistent or drawdown behavior is worse than the tested envelope.",
        },
        {
            "checkpoint": "6m",
            "role": "first real decision" if required_horizon == "6m" else "scale-up confirmation",
            "decision_weight": "primary" if required_horizon == "6m" else "confirmation",
            "validate_if": f"Beats {primary_benchmark}, clears cash, and shows no thesis break across multiple re-risk cycles.",
            "continue_if": "Keeps risk controlled but has not yet separated enough from benchmark/cash.",
            "fail_if": "The live/paper path has not earned its complexity after a full evaluation window.",
        },
    ]
    return pd.DataFrame(rows)


def evaluate_current_experiment_status(
    strategy_result: BacktestResult,
    *,
    monitoring_windows: pd.DataFrame | None,
    valuations: pd.DataFrame | None,
    mode: str,
    account: str,
    required_horizon: str,
    max_drawdown: float,
) -> pd.DataFrame:
    columns = [
        "status",
        "window_id",
        "start_date",
        "valuation_date",
        "elapsed_days",
        "required_days",
        "cumulative_return",
        "benchmark_cumulative_return",
        "excess_return",
        "drawdown",
        "drawdown_envelope_used",
        "read",
    ]
    if monitoring_windows is None or monitoring_windows.empty:
        return pd.DataFrame(columns=columns)
    windows = monitoring_windows.copy()
    matches = windows[
        windows.get("strategy_name", pd.Series(dtype=str)).astype(str).eq(strategy_result.name)
        & windows.get("mode", pd.Series(dtype=str)).astype(str).eq(_normalize_mode(mode))
        & windows.get("account", pd.Series(dtype=str)).astype(str).eq(str(account))
    ].copy()
    if matches.empty:
        return pd.DataFrame(columns=columns)
    matches["start_date_ts"] = pd.to_datetime(matches["start_date"], errors="coerce")
    window = matches.sort_values(["start_date_ts", "created_at_utc"], ascending=[False, False]).iloc[0]
    if valuations is None or valuations.empty:
        return pd.DataFrame(
            [
                {
                    "status": "not_started",
                    "window_id": str(window.get("window_id", "")),
                    "start_date": str(window.get("start_date", "")),
                    "valuation_date": "",
                    "elapsed_days": 0,
                    "required_days": EXPERIMENT_HORIZON_DAYS[required_horizon],
                    "read": "Monitoring window exists, but no daily valuation has been recorded yet.",
                }
            ],
            columns=columns,
        )
    value_frame = valuations[
        valuations.get("window_id", pd.Series(dtype=str)).astype(str).eq(str(window["window_id"]))
    ].copy()
    if value_frame.empty:
        return pd.DataFrame(
            [
                {
                    "status": "not_started",
                    "window_id": str(window.get("window_id", "")),
                    "start_date": str(window.get("start_date", "")),
                    "valuation_date": "",
                    "elapsed_days": 0,
                    "required_days": EXPERIMENT_HORIZON_DAYS[required_horizon],
                    "read": "Monitoring window exists, but no valuation rows match it yet.",
                }
            ],
            columns=columns,
        )

    value_frame["valuation_date_ts"] = pd.to_datetime(
        value_frame["valuation_date"],
        errors="coerce",
    )
    latest = value_frame.sort_values("valuation_date_ts").iloc[-1]
    start_date = pd.to_datetime(window.get("start_date"), errors="coerce")
    valuation_date = latest.get("valuation_date_ts")
    elapsed_days = (
        int((valuation_date - start_date).days * 5 / 7)
        if pd.notna(start_date) and pd.notna(valuation_date)
        else 0
    )
    required_days = EXPERIMENT_HORIZON_DAYS[required_horizon]
    excess_return = _safe_float(latest.get("excess_return"), None)
    cumulative_return = _safe_float(latest.get("cumulative_return"), None)
    benchmark_return = _safe_float(latest.get("benchmark_cumulative_return"), None)
    drawdown = _safe_float(latest.get("drawdown"), 0.0)
    envelope_used = abs(min(float(drawdown or 0.0), 0.0)) / max(abs(max_drawdown), 0.08)
    status = _classify_status(
        elapsed_days=elapsed_days,
        required_days=required_days,
        excess_return=excess_return,
        cumulative_return=cumulative_return,
        drawdown_envelope_used=envelope_used,
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "window_id": str(window.get("window_id", "")),
                "start_date": str(window.get("start_date", "")),
                "valuation_date": str(latest.get("valuation_date", "")),
                "elapsed_days": elapsed_days,
                "required_days": required_days,
                "cumulative_return": cumulative_return,
                "benchmark_cumulative_return": benchmark_return,
                "excess_return": excess_return,
                "drawdown": drawdown,
                "drawdown_envelope_used": envelope_used,
                "read": _status_sentence(status, elapsed_days, required_days),
            }
        ],
        columns=columns,
    )


def _classify_status(
    *,
    elapsed_days: int,
    required_days: int,
    excess_return: float | None,
    cumulative_return: float | None,
    drawdown_envelope_used: float,
) -> str:
    if drawdown_envelope_used >= 0.55:
        return "fail"
    if elapsed_days < required_days:
        return "continue"
    if excess_return is not None and excess_return > 0 and (cumulative_return or 0.0) >= 0:
        return "validate"
    if excess_return is not None and excess_return < -0.02:
        return "fail"
    return "continue"


def _summary_cards(
    *,
    required_horizon: str,
    launch_path: str,
    confidence_score: float,
    recommended_capital: float,
    signal_cycle_days: int,
    status_label: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "Required horizon", "value": required_horizon, "detail": "Upfront minimum evidence window before a validate/fail decision."},
            {"metric": "Launch path", "value": launch_path, "detail": "How fresh paper/live capital should be introduced from today."},
            {"metric": "Trial capital", "value": recommended_capital, "detail": "Module-suggested preset based on launch confidence and mode."},
            {"metric": "Launch confidence", "value": confidence_score, "detail": "Launch Lab score used to size the experiment."},
            {"metric": "Signal cycle", "value": signal_cycle_days, "detail": "Estimated trading days needed to observe one target-change cycle."},
            {"metric": "Current status", "value": status_label, "detail": "Existing monitoring read if a matching window already exists."},
        ]
    )


def _status_read(status: pd.DataFrame, required_horizon: str) -> tuple[str, str]:
    if status.empty:
        return (
            "not_started",
            f"No matching monitoring window exists yet. Start the experiment and judge it at the {required_horizon} checkpoint.",
        )
    row = status.iloc[0]
    label = str(row.get("status", "not_started"))
    return label, str(row.get("read") or _status_sentence(label, 0, EXPERIMENT_HORIZON_DAYS[required_horizon]))


def _status_sentence(status: str, elapsed_days: int, required_days: int) -> str:
    if status == "validate":
        return "The experiment has reached its required horizon and is ahead of the benchmark without a drawdown breach."
    if status == "fail":
        return "The experiment has breached a drawdown or benchmark/cash expectation; treat the launch thesis as failed until reviewed."
    if status == "continue":
        return f"Keep monitoring: {elapsed_days} of {required_days} estimated trading days are in the evaluation window."
    return "The experiment is configured but has not produced enough valuation evidence yet."


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return -0.01
    equity = (1.0 + pd.to_numeric(returns, errors="coerce").fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else -0.01


def _normalize_mode(mode: str) -> str:
    return "live" if str(mode).lower() == "live" else "paper"


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(output):
        return default
    return output
