from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import calculate_metrics


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
                **_episode_setup_features(
                    prices[benchmark_ticker],
                    defensive_weight,
                    position,
                ),
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
                lifecycle = _episode_lifecycle_metrics(
                    signal,
                    prices[benchmark_ticker],
                    prices[cash_ticker],
                    returns,
                    defensive_weight,
                    common_index,
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
                        "strategy_excess_vs_benchmark": (
                            strategy_return - benchmark_return
                            if pd.notna(strategy_return) and pd.notna(benchmark_return)
                            else pd.NA
                        ),
                        **lifecycle,
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
                "avg_strategy_forward_return": float(
                    pd.to_numeric(eligible["strategy_forward_return"], errors="coerce").mean()
                ),
                "avg_strategy_excess_vs_benchmark": float(
                    pd.to_numeric(
                        eligible.get("strategy_excess_vs_benchmark"),
                        errors="coerce",
                    ).mean()
                ),
                "median_avoided_drawdown": _median_avoided_drawdown(eligible),
                "median_missed_upside": _median_missed_upside(eligible),
                "rerisk_within_horizon_rate": _bool_rate(eligible, "rerisked_within_horizon"),
                "median_days_to_rerisk": float(
                    pd.to_numeric(eligible.get("days_to_rerisk"), errors="coerce").median()
                ),
                "after_recovery_rerisk_rate": _rerisk_timing_rate(
                    eligible,
                    timing="after_recovery",
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
            "defensive_median_avoided_drawdown": _safe_float(
                row.get("median_avoided_drawdown")
            ),
            "defensive_median_missed_upside": _safe_float(row.get("median_missed_upside")),
            "defensive_rerisk_within_horizon_rate": _safe_float(
                row.get("rerisk_within_horizon_rate")
            ),
            "defensive_avg_strategy_excess_vs_benchmark": _safe_float(
                row.get("avg_strategy_excess_vs_benchmark")
            ),
            "defensive_judgement_label": defensive_judgement_label(row),
        }
    )
    return output


def current_defensive_setup_context(
    result: BacktestResult,
    prices: pd.DataFrame,
    *,
    defensive_ticker: str = "BIL",
    benchmark_ticker: str = "SPY",
    scenario_context: pd.DataFrame | None = None,
) -> dict[str, float | str | pd.Timestamp | None]:
    """Return point-in-time setup features for the latest strategy date."""

    if result.weights.empty or prices.empty or benchmark_ticker not in prices:
        return {}
    common_index = result.weights.index.intersection(prices.index)
    if common_index.empty:
        return {}
    prices = prices.reindex(common_index).sort_index()
    defensive = effective_defensive_weight(
        result,
        defensive_ticker=defensive_ticker,
    ).reindex(common_index).ffill().fillna(0.0)
    position = len(common_index) - 1
    context: dict[str, float | str | pd.Timestamp | None] = {
        "date": pd.to_datetime(common_index[position], errors="coerce"),
        "defensive_weight": float(defensive.iloc[position]),
        "risk_weight": max(0.0, min(1.0, float(1.0 - defensive.iloc[position]))),
        **_episode_setup_features(prices[benchmark_ticker], defensive, position),
    }
    scenario_row = _latest_scenario_context_row(
        pd.to_datetime(common_index[position], errors="coerce"),
        scenario_context,
    )
    context.update(scenario_row)
    return context


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
    current_setup: dict[str, object] | None = None,
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
        "similar_correct_defense_rate": None,
        "similar_median_avoided_drawdown": None,
        "similar_median_missed_upside": None,
        "similar_avg_strategy_excess_vs_benchmark": None,
        "similar_similarity_score": None,
        "similarity_basis": "defensive_weight_only",
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
    similar, similarity_basis = _similar_setup_events(
        eligible,
        current_setup=current_setup,
        current_defensive_weight=current_defensive_weight,
        similar_band=similar_band,
    )
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
    similar_judgement = similar["judgement"].astype(str) if not similar.empty else pd.Series(dtype=str)
    label = _false_alarm_sniff_label(
        historical_rate=historical_false_rate,
        recent_rate=recent_rate,
        similar_rate=similar_rate,
        posterior_rate=posterior_rate,
        recent_count=len(recent),
        similar_count=len(similar),
    )
    output.update(
        {
            "historical_episode_starts": historical_count,
            "historical_false_alarm_rate": historical_false_rate,
            "recent_episode_starts": int(len(recent)),
            "recent_false_alarm_rate": recent_rate,
            "similar_episode_starts": int(len(similar)),
            "similar_false_alarm_rate": similar_rate,
            "similar_correct_defense_rate": (
                float(similar_judgement.eq("correct_defense").mean())
                if not similar_judgement.empty
                else None
            ),
            "similar_median_avoided_drawdown": _median_avoided_drawdown(similar),
            "similar_median_missed_upside": _median_missed_upside(similar),
            "similar_avg_strategy_excess_vs_benchmark": (
                float(
                    pd.to_numeric(
                        similar.get("strategy_excess_vs_benchmark"),
                        errors="coerce",
                    ).mean()
                )
                if not similar.empty
                else None
            ),
            "similar_similarity_score": (
                float(pd.to_numeric(similar.get("setup_similarity_score"), errors="coerce").mean())
                if not similar.empty and "setup_similarity_score" in similar
                else None
            ),
            "similarity_basis": similarity_basis,
            "posterior_false_alarm_rate": float(posterior_rate),
            "posterior_false_alarm_low": low,
            "posterior_false_alarm_high": high,
            "sniff_test_label": label,
            "sniff_test_readout": _false_alarm_sniff_readout(
                label=label,
                historical_rate=historical_false_rate,
                recent_rate=recent_rate,
                similar_rate=similar_rate,
                posterior_rate=posterior_rate,
                recent_count=len(recent),
                similar_count=len(similar),
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
    focus_strategy: str | None = None,
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
    days = _concat_or_empty(days_frames)
    events = _concat_or_empty(event_frames)
    summary = _concat_or_empty(summary_frames)
    scorecards = pd.DataFrame(scorecard_rows)
    focus = focus_strategy if focus_strategy in results else (selected_names[0] if selected_names else None)
    exposures = _current_defensive_exposures(results, selected_names, focus_strategy=focus)
    scenario_summary = _scenario_context_summary(events)
    recent_focus = (
        events[
            events["strategy"].astype(str).eq(str(focus))
            & pd.to_numeric(events["threshold"], errors="coerce").eq(0.65)
            & events["horizon"].astype(str).eq("1m")
        ]
        .sort_values("date")
        .tail(24)
        if focus is not None and not events.empty
        else pd.DataFrame()
    )
    outputs = {
        "days": output_dir / "defensive_signal_days.csv",
        "events": output_dir / "defensive_episode_outcomes.csv",
        "summary": output_dir / "defensive_signal_summary.csv",
        "scorecards": output_dir / "defensive_signal_scorecards.csv",
        "current_exposure": output_dir / "current_defensive_exposure.csv",
        "scenario_context": output_dir / "defensive_signal_by_scenario_context.csv",
        "recent_focus": output_dir / "focus_strategy_recent_65pct_1m_episodes.csv",
        "readout": output_dir / "summary.md",
    }
    days.to_csv(outputs["days"], index=False)
    events.to_csv(outputs["events"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    scorecards.to_csv(outputs["scorecards"], index=False)
    exposures.to_csv(outputs["current_exposure"], index=False)
    scenario_summary.to_csv(outputs["scenario_context"], index=False)
    recent_focus.to_csv(outputs["recent_focus"], index=False)
    outputs["readout"].write_text(
        _defensive_judgement_markdown(
            summary,
            exposures,
            focus_strategy=focus,
            market_date=prices.index.max() if not prices.empty else None,
        ),
        encoding="utf-8",
    )
    return outputs


def _current_defensive_exposures(
    results: dict[str, BacktestResult],
    selected_names: list[str],
    *,
    focus_strategy: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = sorted(selected_names, key=lambda name: (name != focus_strategy, name))
    for name in ordered:
        result = results.get(name)
        if result is None or result.weights.empty:
            continue
        effective = effective_defensive_weight(result)
        latest = result.weights.iloc[-1].astype(float)
        explicit = float(latest.get("BIL", 0.0))
        residual = max(0.0, 1.0 - float(latest.sum()))
        metrics = calculate_metrics(
            name=name,
            returns=result.returns,
            equity=result.equity,
            turnover=result.turnover,
            transaction_costs=result.transaction_costs,
        )
        defensive = float(effective.iloc[-1])
        rows.append(
            {
                "strategy": name,
                "cagr": metrics.cagr,
                "max_drawdown": metrics.max_drawdown,
                "current_defensive_weight": defensive,
                "current_risk_weight": max(0.0, 1.0 - defensive),
                "current_explicit_bil_weight": explicit,
                "current_residual_cash_weight": residual,
            }
        )
    return pd.DataFrame(rows)


def _scenario_context_summary(events: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "benchmark_ticker", "threshold", "horizon", "scenario_context", "judgement"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame()
    frame = events[events["judgement"].astype(str).ne("insufficient_forward_data")].copy()
    if frame.empty:
        return pd.DataFrame()
    group_columns = ["strategy", "benchmark_ticker", "threshold", "horizon", "scenario_context"]
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    output = grouped.size().rename("episode_starts").reset_index()
    for label, column in (("correct_defense", "correct_defense_rate"), ("false_alarm", "false_alarm_rate")):
        rates = grouped["judgement"].apply(lambda values, expected=label: values.astype(str).eq(expected).mean())
        output[column] = rates.to_numpy()
    return output


def _defensive_judgement_markdown(
    summary: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    focus_strategy: str | None,
    market_date: object,
) -> str:
    lines = [
        "# Defensive Signal False-Alarm Audit",
        "",
        f"Generated from the latest local snapshot through {pd.Timestamp(market_date).date() if market_date is not None else 'n/a'}.",
        "",
        "This is the strategy-native defense audit. It does not treat news or scenario probabilities as sizing inputs.",
        "",
    ]
    exposure = exposures[exposures["strategy"].astype(str).eq(str(focus_strategy))]
    if not exposure.empty:
        row = exposure.iloc[0]
        lines.extend(
            [
                "## Current Focus Strategy",
                "",
                f"- Strategy: `{focus_strategy}`",
                f"- Current effective defensive weight: {float(row['current_defensive_weight']):.1%}",
                f"- Current risk weight: {float(row['current_risk_weight']):.1%}",
                "",
            ]
        )
    focus_rows = summary[
        summary["strategy"].astype(str).eq(str(focus_strategy))
        & summary["benchmark_ticker"].astype(str).eq("SPY")
        & pd.to_numeric(summary["threshold"], errors="coerce").eq(0.65)
    ] if not summary.empty else pd.DataFrame()
    lines.extend(["## Focus Strategy 65% Defensive Episode Read", ""])
    if focus_rows.empty:
        lines.append("No eligible completed episodes were available.")
    else:
        lines.extend(
            [
                "| Horizon | Episodes | Correct defense | False alarm | Mixed | Median forward SPY DD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        order = {"1w": 0, "1m": 1, "3m": 2}
        for _, row in focus_rows.sort_values("horizon", key=lambda values: values.map(order)).iterrows():
            lines.append(
                f"| {row['horizon']} | {int(row['episode_starts'])} | "
                f"{float(row['correct_defense_rate']):.1%} | "
                f"{float(row['false_alarm_rate']):.1%} | "
                f"{float(row['mixed_rate']):.1%} | "
                f"{float(row['median_benchmark_forward_max_drawdown']):.1%} |"
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Strategy-native defensive exposure is distinct from the final trade-decision overlay.",
            "- Retrospective episode frequencies are not prospective probabilities.",
            "- Scenario context is descriptive only and does not receive sizing authority.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _episode_lifecycle_metrics(
    signal: pd.Series,
    benchmark: pd.Series,
    cash: pd.Series,
    returns: pd.Series,
    defensive_weight: pd.Series,
    index: pd.Index,
    position: int,
    horizon_days: int,
) -> dict[str, object]:
    rerisk_position = _rerisk_position(signal, position)
    days_to_rerisk = None if rerisk_position is None else int(rerisk_position - position)
    benchmark_recovery_position = _benchmark_recovery_position(
        benchmark,
        position,
        horizon_days,
    )
    recovery_days = (
        None
        if benchmark_recovery_position is None
        else int(benchmark_recovery_position - position)
    )
    rerisk_timing = _rerisk_timing_label(
        rerisk_position=rerisk_position,
        benchmark_recovery_position=benchmark_recovery_position,
        episode_position=position,
    )
    days = days_to_rerisk if days_to_rerisk is not None else horizon_days
    evaluation_days = min(days, max(0, len(index) - position - 1))
    rerisk_date = index[rerisk_position] if rerisk_position is not None else pd.NaT
    return {
        "episode_end_date": rerisk_date,
        "days_to_rerisk": days_to_rerisk,
        "rerisked_within_horizon": (
            bool(days_to_rerisk <= horizon_days) if days_to_rerisk is not None else False
        ),
        "defensive_weight_at_rerisk": (
            float(defensive_weight.iloc[rerisk_position])
            if rerisk_position is not None
            else pd.NA
        ),
        "benchmark_return_to_rerisk": _forward_return(benchmark, position, evaluation_days),
        "cash_return_to_rerisk": _forward_return(cash, position, evaluation_days),
        "strategy_return_to_rerisk": _forward_strategy_return(
            returns,
            position,
            evaluation_days,
        ),
        "benchmark_drawdown_to_rerisk": _forward_max_drawdown(
            benchmark,
            position,
            evaluation_days,
        ),
        "benchmark_recovery_days": recovery_days,
        "rerisk_timing": rerisk_timing,
    }


def _rerisk_position(signal: pd.Series, position: int) -> int | None:
    for next_position in range(position + 1, len(signal)):
        if not bool(signal.iloc[next_position]):
            return next_position
    return None


def _benchmark_recovery_position(
    benchmark: pd.Series,
    position: int,
    horizon_days: int,
    *,
    material_drawdown: float = -0.02,
) -> int | None:
    end = min(position + horizon_days, len(benchmark) - 1)
    if position < 0 or end <= position:
        return None
    path = benchmark.iloc[position : end + 1].dropna()
    if path.empty:
        return None
    start = float(path.iloc[0])
    if start == 0.0:
        return None
    relative = path / start - 1.0
    trough_label = relative.idxmin()
    if float(relative.loc[trough_label]) > material_drawdown:
        return position
    trough_position = benchmark.index.get_loc(trough_label)
    if isinstance(trough_position, slice) or not isinstance(trough_position, int):
        return None
    recovery = benchmark.iloc[trough_position : end + 1]
    recovered = recovery[recovery >= start]
    if recovered.empty:
        return None
    recovery_position = benchmark.index.get_loc(recovered.index[0])
    return recovery_position if isinstance(recovery_position, int) else None


def _rerisk_timing_label(
    *,
    rerisk_position: int | None,
    benchmark_recovery_position: int | None,
    episode_position: int,
) -> str:
    if benchmark_recovery_position is None:
        return "no_benchmark_recovery_observed" if rerisk_position is not None else "not_rerisked"
    if rerisk_position is None:
        return "not_rerisked"
    if benchmark_recovery_position == episode_position:
        return "no_material_drawdown"
    if rerisk_position < benchmark_recovery_position:
        return "before_recovery"
    if rerisk_position == benchmark_recovery_position:
        return "at_recovery"
    return "after_recovery"


def _episode_setup_features(
    benchmark: pd.Series,
    defensive_weight: pd.Series,
    position: int,
) -> dict[str, float | pd.NA]:
    return {
        "benchmark_trailing_21d_return": _trailing_return(benchmark, position, 21),
        "benchmark_trailing_63d_return": _trailing_return(benchmark, position, 63),
        "benchmark_trailing_21d_vol": _trailing_volatility(benchmark, position, 21),
        "benchmark_drawdown_from_63d_high": _drawdown_from_high(benchmark, position, 63),
        "defensive_weight_change_21d": _trailing_change(defensive_weight, position, 21),
    }


def _trailing_return(series: pd.Series, position: int, days: int) -> float | pd.NA:
    start = position - days
    if start < 0 or position >= len(series):
        return pd.NA
    start_value = series.iloc[start]
    end_value = series.iloc[position]
    if pd.isna(start_value) or pd.isna(end_value) or float(start_value) == 0.0:
        return pd.NA
    return float(end_value / start_value - 1.0)


def _trailing_change(series: pd.Series, position: int, days: int) -> float | pd.NA:
    start = position - days
    if start < 0 or position >= len(series):
        return pd.NA
    start_value = series.iloc[start]
    end_value = series.iloc[position]
    if pd.isna(start_value) or pd.isna(end_value):
        return pd.NA
    return float(end_value - start_value)


def _trailing_volatility(series: pd.Series, position: int, days: int) -> float | pd.NA:
    start = position - days
    if start < 0 or position >= len(series):
        return pd.NA
    returns = series.iloc[start : position + 1].pct_change().dropna()
    if returns.empty:
        return pd.NA
    return float(returns.std(ddof=0) * math.sqrt(252.0))


def _drawdown_from_high(series: pd.Series, position: int, days: int) -> float | pd.NA:
    start = max(0, position - days)
    if position >= len(series):
        return pd.NA
    path = series.iloc[start : position + 1].dropna()
    if path.empty:
        return pd.NA
    high = float(path.max())
    current = float(path.iloc[-1])
    if high == 0.0:
        return pd.NA
    return float(current / high - 1.0)


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


def _latest_scenario_context_row(
    date: pd.Timestamp | pd.NaT,
    scenario_context: pd.DataFrame | None,
) -> dict[str, float | str | None]:
    if scenario_context is None or scenario_context.empty or "date" not in scenario_context:
        return {}
    timestamp = pd.to_datetime(date, errors="coerce")
    if pd.isna(timestamp):
        return {}
    frame = scenario_context.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if frame.empty:
        return {}
    eligible = frame[frame["date"] <= timestamp]
    if eligible.empty:
        return {}
    row = eligible.iloc[-1]
    if timestamp - row["date"] > pd.Timedelta(days=65):
        return {}
    output: dict[str, float | str | None] = {}
    for column in [
        "risk_off",
        "transition",
        "risk_on_fragile",
        "risk_on",
        "defensive_or_transition_probability",
        "constructive_probability",
    ]:
        value = _safe_float(row.get(column))
        if value is not None:
            output[column] = value
    if "scenario_context" in row:
        output["scenario_context"] = str(row.get("scenario_context"))
    return output


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


def _similar_setup_events(
    eligible: pd.DataFrame,
    *,
    current_setup: dict[str, object] | None,
    current_defensive_weight: float | None,
    similar_band: float,
    max_events: int = 24,
) -> tuple[pd.DataFrame, str]:
    if eligible.empty:
        return pd.DataFrame(), "none"
    setup = dict(current_setup or {})
    if current_defensive_weight is not None and "defensive_weight" not in setup:
        setup["defensive_weight"] = current_defensive_weight
    feature_scales = {
        "defensive_weight": 0.10,
        "defensive_weight_change_21d": 0.20,
        "risk_off": 0.10,
        "transition": 0.12,
        "defensive_or_transition_probability": 0.15,
        "constructive_probability": 0.15,
        "benchmark_trailing_21d_return": 0.08,
        "benchmark_trailing_63d_return": 0.14,
        "benchmark_trailing_21d_vol": 0.16,
        "benchmark_drawdown_from_63d_high": 0.10,
    }
    usable_features = [
        feature
        for feature in feature_scales
        if feature in eligible and _safe_float(setup.get(feature)) is not None
    ]
    if len(usable_features) >= 2:
        scored = eligible.copy()
        distances: list[pd.Series] = []
        for feature in usable_features:
            current = _safe_float(setup.get(feature))
            if current is None:
                continue
            values = pd.to_numeric(scored[feature], errors="coerce")
            distance = (values - current).abs() / feature_scales[feature]
            distances.append(distance)
        if distances:
            scored["setup_similarity_score"] = pd.concat(distances, axis=1).mean(axis=1)
            scored = scored.dropna(subset=["setup_similarity_score"]).sort_values(
                ["setup_similarity_score", "date"],
            )
            close = scored[scored["setup_similarity_score"] <= 1.0]
            if len(close) >= 5:
                return close.head(max_events), "multi_feature_context"
            if not scored.empty:
                return scored.head(min(max_events, max(5, min(8, len(scored))))), (
                    "nearest_multi_feature_context"
                )
    if current_defensive_weight is None or "defensive_weight" not in eligible:
        return pd.DataFrame(), "none"
    defensive = pd.to_numeric(eligible["defensive_weight"], errors="coerce")
    similar = eligible[
        defensive.between(
            max(0.0, current_defensive_weight - similar_band),
            min(1.0, current_defensive_weight + similar_band),
        )
    ].copy()
    if not similar.empty:
        similar["setup_similarity_score"] = (defensive.loc[similar.index] - current_defensive_weight).abs()
    return similar, "defensive_weight_only"


def _median_avoided_drawdown(events: pd.DataFrame) -> float | None:
    if events.empty or "judgement" not in events:
        return None
    correct = events[events["judgement"].astype(str).eq("correct_defense")]
    if correct.empty or "benchmark_forward_max_drawdown" not in correct:
        return None
    drawdown = pd.to_numeric(correct["benchmark_forward_max_drawdown"], errors="coerce")
    drawdown = drawdown[drawdown < 0.0]
    if drawdown.empty:
        return None
    return float((-drawdown).median())


def _median_missed_upside(events: pd.DataFrame) -> float | None:
    if events.empty or "judgement" not in events:
        return None
    false_alarms = events[events["judgement"].astype(str).eq("false_alarm")]
    if false_alarms.empty or "benchmark_excess_vs_cash" not in false_alarms:
        return None
    missed = pd.to_numeric(false_alarms["benchmark_excess_vs_cash"], errors="coerce")
    missed = missed[missed > 0.0]
    if missed.empty:
        return None
    return float(missed.median())


def _bool_rate(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _rerisk_timing_rate(frame: pd.DataFrame, *, timing: str) -> float | None:
    if frame.empty or "rerisk_timing" not in frame:
        return None
    rerisked = frame[frame["days_to_rerisk"].notna()] if "days_to_rerisk" in frame else frame
    if rerisked.empty:
        return None
    return float(rerisked["rerisk_timing"].astype(str).eq(timing).mean())


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
    similar_rate: float | None,
    posterior_rate: float,
    recent_count: int,
    similar_count: int,
) -> str:
    if similar_count >= 5 and similar_rate is not None:
        if similar_rate >= historical_rate + 0.12 and similar_rate >= 0.35:
            return "similar_setups_false_alarm_prone"
        if similar_rate <= historical_rate - 0.12 and similar_rate <= 0.30:
            return "similar_setups_support_defense"
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
    similar_rate: float | None,
    posterior_rate: float,
    recent_count: int,
    similar_count: int,
) -> str:
    recent_text = "not enough recent episodes" if recent_rate is None else f"{recent_rate:.1%}"
    similar_text = (
        "not enough similar setups" if similar_rate is None else f"{similar_rate:.1%}"
    )
    if label == "similar_setups_false_alarm_prone":
        prefix = "Historically similar defensive setups have false-alarmed more often than the long-run base rate."
    elif label == "similar_setups_support_defense":
        prefix = "Historically similar defensive setups have supported staying defensive."
    elif label == "recent_false_alarms_elevated":
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
        f"{recent_text} across {recent_count} recent episodes; similar-setup rate is "
        f"{similar_text} across {similar_count} episodes; Bayesian-updated rate is {posterior_rate:.1%}."
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
