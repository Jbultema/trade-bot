from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult


@dataclass(frozen=True)
class DefensiveJudgementHorizon:
    label: str
    trading_days: int
    drawdown_correct_threshold: float
    false_alarm_excess_threshold: float
    false_alarm_drawdown_floor: float


DEFAULT_DEFENSIVE_THRESHOLDS: tuple[float, ...] = (0.65, 0.75, 0.85, 0.90)
DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS: tuple[DefensiveJudgementHorizon, ...] = (
    DefensiveJudgementHorizon(
        label="1w",
        trading_days=5,
        drawdown_correct_threshold=-0.03,
        false_alarm_excess_threshold=0.01,
        false_alarm_drawdown_floor=-0.02,
    ),
    DefensiveJudgementHorizon(
        label="1m",
        trading_days=21,
        drawdown_correct_threshold=-0.05,
        false_alarm_excess_threshold=0.02,
        false_alarm_drawdown_floor=-0.03,
    ),
    DefensiveJudgementHorizon(
        label="3m",
        trading_days=63,
        drawdown_correct_threshold=-0.08,
        false_alarm_excess_threshold=0.05,
        false_alarm_drawdown_floor=-0.05,
    ),
)


def effective_defensive_weight(
    result: BacktestResult,
    *,
    defensive_ticker: str = "BIL",
) -> pd.Series:
    """Return explicit defensive allocation plus residual cash from risk scaling."""

    if result.weights.empty:
        return pd.Series(dtype=float, name="effective_defensive_weight")
    weights = result.weights.sort_index().astype(float).fillna(0.0)
    explicit_defensive = (
        weights[defensive_ticker]
        if defensive_ticker in weights
        else pd.Series(0.0, index=weights.index)
    )
    residual_cash = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    return (explicit_defensive + residual_cash).clip(lower=0.0, upper=1.0).rename(
        "effective_defensive_weight"
    )


def build_defensive_judgement_audit(
    result: BacktestResult,
    prices: pd.DataFrame,
    *,
    thresholds: Iterable[float] = DEFAULT_DEFENSIVE_THRESHOLDS,
    horizons: Iterable[DefensiveJudgementHorizon] = DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS,
    defensive_ticker: str = "BIL",
    benchmark_ticker: str = "SPY",
    cash_ticker: str = "BIL",
    scenario_context: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Audit whether high defensive sizing historically protected or overreacted.

    The audit is intentionally episode-based. Consecutive high-defensive days are counted
    once at the crossing date so a long defensive spell does not dominate the evidence.
    """

    if result.weights.empty or prices.empty:
        return _empty_audit()
    if benchmark_ticker not in prices or cash_ticker not in prices:
        return _empty_audit()

    prices = prices.sort_index()
    common_index = result.weights.index.intersection(prices.index)
    if common_index.empty:
        return _empty_audit()

    prices = prices.reindex(common_index)
    defensive_weight = effective_defensive_weight(
        result,
        defensive_ticker=defensive_ticker,
    ).reindex(common_index).ffill().fillna(0.0)
    risk_weight = (1.0 - defensive_weight).clip(lower=0.0, upper=1.0)
    returns = result.returns.reindex(common_index).fillna(0.0)

    day_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    threshold_values = tuple(float(threshold) for threshold in thresholds)
    horizon_values = tuple(horizons)

    for threshold in threshold_values:
        signal = defensive_weight >= threshold
        episode_start = signal & ~signal.shift(1, fill_value=False)
        for position, date in enumerate(common_index):
            if not bool(signal.iloc[position]):
                continue
            row = {
                "strategy": result.name,
                "date": date,
                "benchmark_ticker": benchmark_ticker,
                "cash_ticker": cash_ticker,
                "threshold": threshold,
                "defensive_weight": float(defensive_weight.iloc[position]),
                "risk_weight": float(risk_weight.iloc[position]),
                "episode_start": bool(episode_start.iloc[position]),
                "nonpanic_defensive": bool(defensive_weight.iloc[position] < 0.90),
                "panic_defensive": bool(defensive_weight.iloc[position] >= 0.90),
            }
            day_rows.append(row)
            if not bool(episode_start.iloc[position]):
                continue
            for horizon in horizon_values:
                benchmark_return = _forward_return(
                    prices[benchmark_ticker],
                    position,
                    horizon.trading_days,
                )
                cash_return = _forward_return(
                    prices[cash_ticker],
                    position,
                    horizon.trading_days,
                )
                benchmark_drawdown = _forward_max_drawdown(
                    prices[benchmark_ticker],
                    position,
                    horizon.trading_days,
                )
                strategy_return = _forward_strategy_return(
                    returns,
                    position,
                    horizon.trading_days,
                )
                event_rows.append(
                    {
                        **row,
                        "horizon": horizon.label,
                        "forward_days": horizon.trading_days,
                        "benchmark_forward_return": benchmark_return,
                        "cash_forward_return": cash_return,
                        "strategy_forward_return": strategy_return,
                        "benchmark_excess_vs_cash": (
                            benchmark_return - cash_return
                            if pd.notna(benchmark_return) and pd.notna(cash_return)
                            else pd.NA
                        ),
                        "benchmark_forward_max_drawdown": benchmark_drawdown,
                        "judgement": classify_defensive_judgement(
                            benchmark_forward_return=benchmark_return,
                            cash_forward_return=cash_return,
                            benchmark_forward_max_drawdown=benchmark_drawdown,
                            horizon=horizon,
                        ),
                    }
                )

    days = pd.DataFrame(day_rows)
    events = pd.DataFrame(event_rows)
    if not events.empty and scenario_context is not None and not scenario_context.empty:
        events = _attach_scenario_context(events, scenario_context)
    summary = summarize_defensive_judgement(events)
    return {"days": days, "events": events, "summary": summary}


def classify_defensive_judgement(
    *,
    benchmark_forward_return: float | pd.NA,
    cash_forward_return: float | pd.NA,
    benchmark_forward_max_drawdown: float | pd.NA,
    horizon: DefensiveJudgementHorizon,
) -> str:
    if (
        pd.isna(benchmark_forward_return)
        or pd.isna(cash_forward_return)
        or pd.isna(benchmark_forward_max_drawdown)
    ):
        return "insufficient_forward_data"
    benchmark_return = float(benchmark_forward_return)
    cash_return = float(cash_forward_return)
    benchmark_drawdown = float(benchmark_forward_max_drawdown)
    if (
        benchmark_return <= cash_return
        or benchmark_drawdown <= horizon.drawdown_correct_threshold
    ):
        return "correct_defense"
    if (
        benchmark_return - cash_return >= horizon.false_alarm_excess_threshold
        and benchmark_drawdown >= horizon.false_alarm_drawdown_floor
    ):
        return "false_alarm"
    return "mixed_or_early"


def summarize_defensive_judgement(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        column
        for column in ["strategy", "benchmark_ticker", "cash_ticker", "threshold", "horizon"]
        if column in events
    ]
    for keys, group in events.groupby(group_columns, dropna=False):
        key_values = dict(
            zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=False)
        )
        eligible = group[group["judgement"].astype(str).ne("insufficient_forward_data")]
        if eligible.empty:
            continue
        judgement = eligible["judgement"].astype(str)
        rows.append(
            {
                **key_values,
                "episode_starts": int(len(eligible)),
                "correct_defense": int(judgement.eq("correct_defense").sum()),
                "false_alarm": int(judgement.eq("false_alarm").sum()),
                "mixed_or_early": int(judgement.eq("mixed_or_early").sum()),
                "correct_defense_rate": float(judgement.eq("correct_defense").mean()),
                "false_alarm_rate": float(judgement.eq("false_alarm").mean()),
                "mixed_rate": float(judgement.eq("mixed_or_early").mean()),
                "avg_benchmark_forward_return": float(
                    pd.to_numeric(eligible["benchmark_forward_return"], errors="coerce").mean()
                ),
                "avg_cash_forward_return": float(
                    pd.to_numeric(eligible["cash_forward_return"], errors="coerce").mean()
                ),
                "avg_benchmark_excess_vs_cash": float(
                    pd.to_numeric(eligible["benchmark_excess_vs_cash"], errors="coerce").mean()
                ),
                "median_benchmark_forward_return": float(
                    pd.to_numeric(eligible["benchmark_forward_return"], errors="coerce").median()
                ),
                "median_benchmark_forward_max_drawdown": float(
                    pd.to_numeric(
                        eligible["benchmark_forward_max_drawdown"],
                        errors="coerce",
                    ).median()
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    order = {"1w": 0, "1m": 1, "3m": 2}
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [column for column in ["strategy", "benchmark_ticker", "threshold", "horizon"] if column in summary],
        key=lambda column: column.map(order) if column.name == "horizon" else column,
    ).reset_index(drop=True)


def defensive_judgement_scorecard(
    result: BacktestResult,
    prices: pd.DataFrame,
    *,
    threshold: float = 0.65,
    horizon: str = "1m",
    defensive_ticker: str = "BIL",
    benchmark_ticker: str = "SPY",
    cash_ticker: str = "BIL",
) -> dict[str, float | int | str | None]:
    audit = build_defensive_judgement_audit(
        result,
        prices,
        thresholds=(threshold,),
        horizons=DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS,
        defensive_ticker=defensive_ticker,
        benchmark_ticker=benchmark_ticker,
        cash_ticker=cash_ticker,
    )
    summary = audit["summary"]
    defensive_weight = effective_defensive_weight(
        result,
        defensive_ticker=defensive_ticker,
    )
    current_defensive_weight = (
        float(defensive_weight.iloc[-1]) if not defensive_weight.empty else None
    )
    output: dict[str, float | int | str | None] = {
        "defensive_threshold": threshold,
        "defensive_judgement_horizon": horizon,
        "defensive_benchmark_ticker": benchmark_ticker,
        "current_defensive_weight": current_defensive_weight,
        "current_risk_weight": (
            1.0 - current_defensive_weight
            if current_defensive_weight is not None
            else None
        ),
        "defensive_judgement_label": "not_enough_history",
        "defensive_episode_starts": 0,
        "defensive_correct_rate": None,
        "defensive_false_alarm_rate": None,
        "defensive_mixed_rate": None,
        "defensive_avg_benchmark_excess_vs_cash": None,
        "defensive_median_forward_drawdown": None,
    }
    if summary.empty or not {"threshold", "horizon"}.issubset(summary.columns):
        return output
    selected = summary[
        summary["threshold"].eq(threshold) & summary["horizon"].astype(str).eq(horizon)
    ]
    if selected.empty:
        return output
    row = selected.iloc[0]
    output.update(
        {
            "defensive_episode_starts": int(row.get("episode_starts", 0)),
            "defensive_correct_rate": float(row["correct_defense_rate"]),
            "defensive_false_alarm_rate": float(row["false_alarm_rate"]),
            "defensive_mixed_rate": float(row["mixed_rate"]),
            "defensive_avg_benchmark_excess_vs_cash": float(
                row["avg_benchmark_excess_vs_cash"]
            ),
            "defensive_median_forward_drawdown": float(
                row["median_benchmark_forward_max_drawdown"]
            ),
            "defensive_judgement_label": defensive_judgement_label(row),
        }
    )
    return output


def defensive_judgement_label(summary_row: pd.Series) -> str:
    correct_rate = _safe_float(summary_row.get("correct_defense_rate"))
    false_alarm_rate = _safe_float(summary_row.get("false_alarm_rate"))
    episode_starts = _safe_float(summary_row.get("episode_starts"))
    if episode_starts is None or episode_starts < 8:
        return "thin_history"
    if correct_rate is not None and correct_rate >= 0.60 and (
        false_alarm_rate is None or false_alarm_rate <= 0.25
    ):
        return "defensive_signal_useful"
    if false_alarm_rate is not None and false_alarm_rate >= 0.40:
        return "frequent_false_alarm"
    if correct_rate is not None and correct_rate >= 0.45:
        return "mixed_but_informative"
    return "weak_defensive_signal"


def defensive_false_alarm_bayes_update(
    events: pd.DataFrame,
    *,
    threshold: float,
    horizon: str = "1m",
    current_defensive_weight: float | None = None,
    recent_years: float = 3.0,
    similar_band: float = 0.10,
    prior_strength: float = 20.0,
) -> dict[str, float | int | str | None]:
    """Estimate whether recent/relevant evidence changes the false-alarm prior.

    The prior is the full historical false-alarm rate for the selected threshold and
    horizon. The posterior then updates that prior using recent episodes and, when
    available, episodes with a defensive weight near today's defensive weight.
    """

    output: dict[str, float | int | str | None] = {
        "historical_episode_starts": 0,
        "historical_false_alarm_rate": None,
        "recent_episode_starts": 0,
        "recent_false_alarm_rate": None,
        "similar_episode_starts": 0,
        "similar_false_alarm_rate": None,
        "posterior_false_alarm_rate": None,
        "posterior_false_alarm_low": None,
        "posterior_false_alarm_high": None,
        "sniff_test_label": "not_enough_evidence",
        "sniff_test_readout": "Not enough defensive episode history to update the false-alarm read.",
    }
    if events.empty:
        return output
    eligible = events[
        events["threshold"].eq(threshold)
        & events["horizon"].astype(str).eq(str(horizon))
        & events["judgement"].astype(str).ne("insufficient_forward_data")
    ].copy()
    if eligible.empty:
        return output
    eligible["date"] = pd.to_datetime(eligible["date"], errors="coerce")
    eligible = eligible.dropna(subset=["date"])
    if eligible.empty:
        return output

    historical_false_rate = _false_alarm_rate(eligible)
    historical_count = int(len(eligible))
    if historical_count < 5 or historical_false_rate is None:
        return output

    latest_date = eligible["date"].max()
    recent_cutoff = latest_date - pd.Timedelta(days=int(recent_years * 365.25))
    recent = eligible[eligible["date"] >= recent_cutoff]
    similar = pd.DataFrame()
    if current_defensive_weight is not None and "defensive_weight" in eligible:
        defensive = pd.to_numeric(eligible["defensive_weight"], errors="coerce")
        similar = eligible[
            defensive.between(
                max(0.0, current_defensive_weight - similar_band),
                min(1.0, current_defensive_weight + similar_band),
            )
        ]
    contextual = _concat_contextual_evidence(recent, similar)
    if contextual.empty:
        contextual = recent

    prior_alpha = 1.0 + historical_false_rate * prior_strength
    prior_beta = 1.0 + (1.0 - historical_false_rate) * prior_strength
    contextual_false_count = int(contextual["judgement"].astype(str).eq("false_alarm").sum())
    contextual_nonfalse_count = int(len(contextual) - contextual_false_count)
    posterior_alpha = prior_alpha + contextual_false_count
    posterior_beta = prior_beta + contextual_nonfalse_count
    posterior_rate = posterior_alpha / (posterior_alpha + posterior_beta)
    low, high = _beta_normal_interval(posterior_alpha, posterior_beta)
    recent_rate = _false_alarm_rate(recent)
    similar_rate = _false_alarm_rate(similar)
    label = _false_alarm_sniff_label(
        historical_rate=historical_false_rate,
        recent_rate=recent_rate,
        posterior_rate=posterior_rate,
        recent_count=len(recent),
    )
    output.update(
        {
            "historical_episode_starts": historical_count,
            "historical_false_alarm_rate": historical_false_rate,
            "recent_episode_starts": int(len(recent)),
            "recent_false_alarm_rate": recent_rate,
            "similar_episode_starts": int(len(similar)),
            "similar_false_alarm_rate": similar_rate,
            "posterior_false_alarm_rate": float(posterior_rate),
            "posterior_false_alarm_low": low,
            "posterior_false_alarm_high": high,
            "sniff_test_label": label,
            "sniff_test_readout": _false_alarm_sniff_readout(
                label=label,
                historical_rate=historical_false_rate,
                recent_rate=recent_rate,
                posterior_rate=posterior_rate,
                recent_count=len(recent),
            ),
        }
    )
    return output


def load_scenario_context(
    paths: Iterable[Path] = (
        Path("reports/simulation_validation/scenario_history.csv"),
        Path("reports/simulation_validation/reconstructed_scenario_history.csv"),
    ),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"market_date", "horizon", "risk_bucket", "probability"}
        if not required.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["market_date"], errors="coerce").dt.tz_localize(
            None
        )
        frame["source_file"] = path.name
        frames.append(frame.dropna(subset=["date"]))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["source_priority"] = combined["source_file"].eq("scenario_history.csv").astype(int)
    combined = combined.sort_values(["date", "horizon", "risk_bucket", "source_priority"])
    combined = combined.drop_duplicates(["date", "horizon", "risk_bucket"], keep="last")
    bucket = (
        combined[combined["horizon"].astype(str).eq("1m")]
        .groupby(["date", "risk_bucket"], as_index=False)["probability"]
        .sum()
        .pivot(index="date", columns="risk_bucket", values="probability")
        .fillna(0.0)
        .sort_index()
    )
    for column in ["risk_off", "transition", "risk_on_fragile", "risk_on"]:
        if column not in bucket:
            bucket[column] = 0.0
    bucket["defensive_or_transition_probability"] = bucket["risk_off"] + bucket["transition"]
    bucket["constructive_probability"] = bucket["risk_on"] + bucket["risk_on_fragile"]
    bucket["scenario_context"] = "mixed"
    bucket.loc[bucket["risk_off"] >= 0.25, "scenario_context"] = "risk_off_elevated"
    bucket.loc[
        bucket["defensive_or_transition_probability"] >= 0.45,
        "scenario_context",
    ] = "transition_defensive"
    bucket.loc[
        bucket["risk_on_fragile"] >= 0.20,
        "scenario_context",
    ] = "fragile_constructive"
    bucket.loc[
        bucket["constructive_probability"] >= 0.35,
        "scenario_context",
    ] = "constructive"
    return bucket.reset_index()


def write_defensive_judgement_report(
    *,
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    output_dir: Path,
    strategy_names: Iterable[str] | None = None,
    benchmark_tickers: Iterable[str] = ("SPY", "QQQ"),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = list(strategy_names) if strategy_names is not None else list(results)
    scenario_context = load_scenario_context()
    days_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    scorecard_rows: list[dict[str, object]] = []
    for strategy_name in selected_names:
        result = results.get(strategy_name)
        if result is None:
            continue
        for benchmark_ticker in benchmark_tickers:
            if benchmark_ticker not in prices:
                continue
            audit = build_defensive_judgement_audit(
                result,
                prices,
                benchmark_ticker=benchmark_ticker,
                scenario_context=scenario_context,
            )
            days_frames.append(audit["days"])
            event_frames.append(audit["events"])
            summary_frames.append(audit["summary"])
            scorecard = defensive_judgement_scorecard(
                result,
                prices,
                benchmark_ticker=benchmark_ticker,
            )
            scorecard["strategy"] = strategy_name
            scorecard_rows.append(scorecard)
    outputs = {
        "days": output_dir / "defensive_signal_days.csv",
        "events": output_dir / "defensive_episode_outcomes.csv",
        "summary": output_dir / "defensive_signal_summary.csv",
        "scorecards": output_dir / "defensive_signal_scorecards.csv",
    }
    _concat_or_empty(days_frames).to_csv(outputs["days"], index=False)
    _concat_or_empty(event_frames).to_csv(outputs["events"], index=False)
    _concat_or_empty(summary_frames).to_csv(outputs["summary"], index=False)
    pd.DataFrame(scorecard_rows).to_csv(outputs["scorecards"], index=False)
    return outputs


def _forward_return(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    start_value = series.iloc[position]
    end_value = series.iloc[end]
    if pd.isna(start_value) or pd.isna(end_value) or float(start_value) == 0.0:
        return pd.NA
    return float(end_value / start_value - 1.0)


def _forward_strategy_return(returns: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(returns):
        return pd.NA
    return float((1.0 + returns.iloc[position + 1 : end + 1]).prod() - 1.0)


def _forward_max_drawdown(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    path = series.iloc[position : end + 1].dropna()
    if path.empty:
        return pd.NA
    relative = path / path.iloc[0]
    drawdown = relative / relative.cummax() - 1.0
    return float(drawdown.min())


def _attach_scenario_context(events: pd.DataFrame, scenario_context: pd.DataFrame) -> pd.DataFrame:
    if "date" not in scenario_context:
        return events
    left = events.copy().sort_values("date")
    right = scenario_context.copy().sort_values("date")
    left["date"] = pd.to_datetime(left["date"]).dt.tz_localize(None)
    right["date"] = pd.to_datetime(right["date"]).dt.tz_localize(None)
    return pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=65),
    )


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _false_alarm_rate(events: pd.DataFrame) -> float | None:
    if events.empty:
        return None
    judgement = events["judgement"].astype(str)
    return float(judgement.eq("false_alarm").mean())


def _concat_contextual_evidence(*frames: pd.DataFrame) -> pd.DataFrame:
    clean = [frame for frame in frames if not frame.empty]
    if not clean:
        return pd.DataFrame()
    combined = pd.concat(clean, ignore_index=False, sort=False)
    if "date" in combined:
        combined = combined.sort_values("date")
    return combined[~combined.index.duplicated(keep="last")]


def _beta_normal_interval(alpha: float, beta: float) -> tuple[float, float]:
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / ((total**2) * (total + 1.0))
    margin = 1.96 * math.sqrt(max(variance, 0.0))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _false_alarm_sniff_label(
    *,
    historical_rate: float,
    recent_rate: float | None,
    posterior_rate: float,
    recent_count: int,
) -> str:
    if recent_count < 4:
        return "thin_recent_evidence"
    if recent_rate is not None and recent_rate >= historical_rate + 0.12 and posterior_rate >= 0.35:
        return "recent_false_alarms_elevated"
    if recent_rate is not None and recent_rate <= historical_rate - 0.12 and posterior_rate <= 0.30:
        return "recent_evidence_supports_defense"
    if posterior_rate >= 0.40:
        return "false_alarm_risk_high"
    if posterior_rate <= 0.25:
        return "false_alarm_risk_low"
    return "mixed_context"


def _false_alarm_sniff_readout(
    *,
    label: str,
    historical_rate: float,
    recent_rate: float | None,
    posterior_rate: float,
    recent_count: int,
) -> str:
    recent_text = "not enough recent episodes" if recent_rate is None else f"{recent_rate:.1%}"
    if label == "recent_false_alarms_elevated":
        prefix = "Recent defensive signals have false-alarmed more often than long history."
    elif label == "recent_evidence_supports_defense":
        prefix = "Recent defensive signals have been more useful than long history."
    elif label == "false_alarm_risk_high":
        prefix = "The updated false-alarm risk is high."
    elif label == "false_alarm_risk_low":
        prefix = "The updated false-alarm risk is low."
    elif label == "thin_recent_evidence":
        prefix = "Recent evidence is thin, so the long-run prior still dominates."
    else:
        prefix = "Recent evidence is mixed."
    return (
        f"{prefix} Historical false-alarm rate is {historical_rate:.1%}; recent rate is "
        f"{recent_text} across {recent_count} recent episodes; Bayesian-updated rate is "
        f"{posterior_rate:.1%}."
    )


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    clean = [frame for frame in frames if not frame.empty]
    if not clean:
        return pd.DataFrame()
    return pd.concat(clean, ignore_index=True, sort=False)


def _empty_audit() -> dict[str, pd.DataFrame]:
    return {
        "days": pd.DataFrame(),
        "events": pd.DataFrame(),
        "summary": pd.DataFrame(),
    }
