from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_bot.config import BotConfig
from trade_bot.features.indicators import bounded_forward_fill
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.defensive_judgement import (
    DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS,
    DefensiveJudgementHorizon,
    classify_defensive_judgement,
)

DEFAULT_SCENARIO_CALIBRATION_DIR = Path("reports/scenario_probability_calibration")
DEFAULT_SCENARIO_CALIBRATION_BINS = tuple(np.linspace(0.0, 1.0, 11))
DEFAULT_SCENARIO_CALIBRATION_MIN_OBSERVATIONS = 104
DEFAULT_SCENARIO_CALIBRATION_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_SCENARIO_CALIBRATION_SEED = 42


@dataclass(frozen=True)
class ScenarioProbabilityCalibrationRun:
    outcomes: pd.DataFrame
    reliability: pd.DataFrame
    metrics: pd.DataFrame
    walk_forward: pd.DataFrame
    latest_authority: pd.DataFrame
    output_paths: dict[str, Path]


def run_scenario_probability_calibration(
    origin_states: pd.DataFrame,
    prices: pd.DataFrame,
    config: BotConfig,
    *,
    output_dir: Path = DEFAULT_SCENARIO_CALIBRATION_DIR,
    horizons: tuple[DefensiveJudgementHorizon, ...] = (
        DEFAULT_DEFENSIVE_JUDGEMENT_HORIZONS
    ),
    probability_column: str = "risk_off_probability",
    benchmark_ticker: str = "SPY",
    cash_ticker: str = "BIL",
    min_observations: int = DEFAULT_SCENARIO_CALIBRATION_MIN_OBSERVATIONS,
    bootstrap_samples: int = DEFAULT_SCENARIO_CALIBRATION_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SCENARIO_CALIBRATION_SEED,
) -> ScenarioProbabilityCalibrationRun:
    """Calibrate scenario risk-off probabilities against matured market outcomes.

    The realized label is intentionally inherited from the existing defensive-
    judgement contract: defense is correct when the benchmark underperforms cash
    or breaches the horizon-specific drawdown threshold. This keeps the target
    independent from the scenario model and avoids inventing a favorable label.
    """

    outcomes = build_scenario_outcomes(
        origin_states,
        prices,
        horizons=horizons,
        probability_column=probability_column,
        benchmark_ticker=benchmark_ticker,
        cash_ticker=cash_ticker,
    )
    reliability = build_reliability_table(outcomes)
    metrics = build_calibration_metrics(
        outcomes,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    walk_forward = build_walk_forward_authority(
        outcomes,
        min_observations=min_observations,
    )
    latest_authority = summarize_latest_authority(
        walk_forward,
        metrics,
        min_observations=min_observations,
    )
    output_paths = write_scenario_calibration_outputs(
        output_dir=output_dir,
        outcomes=outcomes,
        reliability=reliability,
        metrics=metrics,
        walk_forward=walk_forward,
        latest_authority=latest_authority,
        prices=prices,
        config=config,
        parameters={
            "probability_column": probability_column,
            "benchmark_ticker": benchmark_ticker,
            "cash_ticker": cash_ticker,
            "min_observations": min_observations,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "horizons": [horizon.__dict__ for horizon in horizons],
            "authority_formula": (
                "sqrt(max(0,brier_skill)*max(0,2*(auc-0.5)))*sample_quality"
            ),
            "trial_roster": ["raw_scenario_probability_calibration"],
        },
    )
    return ScenarioProbabilityCalibrationRun(
        outcomes=outcomes,
        reliability=reliability,
        metrics=metrics,
        walk_forward=walk_forward,
        latest_authority=latest_authority,
        output_paths=output_paths,
    )


def build_scenario_outcomes(
    origin_states: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: tuple[DefensiveJudgementHorizon, ...],
    probability_column: str,
    benchmark_ticker: str,
    cash_ticker: str,
) -> pd.DataFrame:
    required = {"origin_date", probability_column}
    missing = required.difference(origin_states.columns)
    if missing:
        raise ValueError(f"Origin states are missing required columns: {sorted(missing)}")
    if benchmark_ticker not in prices or cash_ticker not in prices:
        raise ValueError("Benchmark and cash price columns are required for calibration.")
    clean_prices = prices.sort_index()
    positions = {
        pd.Timestamp(date): position for position, date in enumerate(clean_prices.index)
    }
    rows: list[dict[str, Any]] = []
    states = origin_states.copy()
    states["origin_date"] = pd.to_datetime(states["origin_date"], errors="coerce")
    states[probability_column] = pd.to_numeric(
        states[probability_column], errors="coerce"
    )
    for _, state in states.dropna(subset=["origin_date", probability_column]).iterrows():
        origin = pd.Timestamp(state["origin_date"])
        position = positions.get(origin)
        if position is None:
            continue
        for horizon in horizons:
            benchmark_return = _forward_return(
                clean_prices[benchmark_ticker], position, horizon.trading_days
            )
            cash_return = _forward_return(
                clean_prices[cash_ticker], position, horizon.trading_days
            )
            benchmark_drawdown = _forward_max_drawdown(
                clean_prices[benchmark_ticker], position, horizon.trading_days
            )
            judgement = classify_defensive_judgement(
                benchmark_forward_return=benchmark_return,
                cash_forward_return=cash_return,
                benchmark_forward_max_drawdown=benchmark_drawdown,
                horizon=horizon,
            )
            if judgement == "insufficient_forward_data":
                continue
            rows.append(
                {
                    "origin_date": origin,
                    "horizon": horizon.label,
                    "forward_days": horizon.trading_days,
                    "predicted_risk_off_probability": float(
                        np.clip(float(state[probability_column]), 0.0, 1.0)
                    ),
                    "realized_risk_off": int(judgement == "correct_defense"),
                    "realized_false_alarm": int(judgement == "false_alarm"),
                    "judgement": judgement,
                    "benchmark_forward_return": benchmark_return,
                    "cash_forward_return": cash_return,
                    "benchmark_forward_max_drawdown": benchmark_drawdown,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "origin_date"]).reset_index(
        drop=True
    )


def build_reliability_table(
    outcomes: pd.DataFrame,
    *,
    bins: tuple[float, ...] = DEFAULT_SCENARIO_CALIBRATION_BINS,
) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for horizon, group in outcomes.groupby("horizon", sort=False):
        probabilities = pd.to_numeric(
            group["predicted_risk_off_probability"], errors="coerce"
        )
        labels = pd.to_numeric(group["realized_risk_off"], errors="coerce")
        bucket = pd.cut(
            probabilities,
            bins=bins,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )
        for interval, positions in group.groupby(bucket, observed=False).groups.items():
            scoped_probability = probabilities.loc[positions].dropna()
            scoped_labels = labels.loc[positions].dropna()
            observations = int(len(scoped_labels))
            realized = float(scoped_labels.mean()) if observations else np.nan
            predicted = (
                float(scoped_probability.mean()) if not scoped_probability.empty else np.nan
            )
            low, high = _wilson_interval(int(scoped_labels.sum()), observations)
            rows.append(
                {
                    "horizon": horizon,
                    "probability_bin": str(interval),
                    "bin_lower": float(interval.left),
                    "bin_upper": float(interval.right),
                    "observations": observations,
                    "mean_predicted_probability": predicted,
                    "realized_risk_off_frequency": realized,
                    "realized_frequency_ci_low": low,
                    "realized_frequency_ci_high": high,
                    "calibration_gap": (
                        realized - predicted
                        if pd.notna(realized) and pd.notna(predicted)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_calibration_metrics(
    outcomes: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows = []
    for horizon, group in outcomes.groupby("horizon", sort=False):
        metrics = _metric_values(group)
        block_length = max(1, int(math.ceil(float(group["forward_days"].iloc[0]) / 5.0)))
        intervals = _block_bootstrap_intervals(
            group,
            block_length=block_length,
            samples=bootstrap_samples,
            seed=seed + int(group["forward_days"].iloc[0]),
        )
        rows.append(
            {
                "horizon": horizon,
                "observations": int(len(group)),
                "positive_rate": float(group["realized_risk_off"].mean()),
                "mean_predicted_probability": float(
                    group["predicted_risk_off_probability"].mean()
                ),
                "block_length_origins": block_length,
                **metrics,
                **intervals,
            }
        )
    return pd.DataFrame(rows)


def build_walk_forward_authority(
    outcomes: pd.DataFrame,
    *,
    min_observations: int,
) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for horizon, group in outcomes.groupby("horizon", sort=False):
        ordered = group.sort_values("origin_date").reset_index(drop=True)
        forward_days = int(ordered["forward_days"].iloc[0])
        for position in range(min_observations, len(ordered)):
            evaluation_date = pd.Timestamp(ordered.loc[position, "origin_date"])
            matured_cutoff = evaluation_date - pd.offsets.BDay(forward_days)
            training = ordered[ordered["origin_date"] <= matured_cutoff]
            if len(training) < min_observations:
                continue
            metrics = _metric_values(training)
            authority = calibration_authority(
                brier_skill=float(metrics["brier_skill"]),
                auc=float(metrics["auc"]),
                observations=len(training),
                min_observations=min_observations,
            )
            rows.append(
                {
                    "evaluation_date": evaluation_date,
                    "horizon": horizon,
                    "training_through": pd.Timestamp(training["origin_date"].max()),
                    "training_observations": int(len(training)),
                    "brier_score": metrics["brier_score"],
                    "brier_skill": metrics["brier_skill"],
                    "auc": metrics["auc"],
                    "ece": metrics["ece"],
                    "calibration_authority": authority,
                }
            )
    return pd.DataFrame(rows)


def summarize_latest_authority(
    walk_forward: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    min_observations: int,
    trailing_estimates: int = 26,
) -> pd.DataFrame:
    rows = []
    if not walk_forward.empty:
        for horizon, group in walk_forward.groupby("horizon", sort=False):
            recent = group.sort_values("evaluation_date").tail(trailing_estimates)
            latest = group.sort_values("evaluation_date").iloc[-1]
            rows.append(
                {
                    "horizon": horizon,
                    "calibration_authority": float(
                        recent["calibration_authority"].median()
                    ),
                    "latest_point_authority": float(latest["calibration_authority"]),
                    "authority_recent_low": float(
                        recent["calibration_authority"].quantile(0.10)
                    ),
                    "authority_recent_high": float(
                        recent["calibration_authority"].quantile(0.90)
                    ),
                    "training_observations": int(latest["training_observations"]),
                    "training_through": latest["training_through"],
                    "authority_status": _authority_status(
                        float(recent["calibration_authority"].median())
                    ),
                    "sizing_authority": "calibration_gated",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    for _, metric in metrics.iterrows():
        authority = calibration_authority(
            brier_skill=float(metric["brier_skill"]),
            auc=float(metric["auc"]),
            observations=int(metric["observations"]),
            min_observations=min_observations,
        )
        rows.append(
            {
                "horizon": metric["horizon"],
                "calibration_authority": authority,
                "latest_point_authority": authority,
                "authority_recent_low": np.nan,
                "authority_recent_high": np.nan,
                "training_observations": int(metric["observations"]),
                "training_through": pd.NaT,
                "authority_status": _authority_status(authority),
                "sizing_authority": "retrospective_fallback",
            }
        )
    return pd.DataFrame(rows)


def calibration_authority(
    *,
    brier_skill: float,
    auc: float,
    observations: int,
    min_observations: int,
) -> float:
    """Convert calibration and discrimination into bounded sizing authority."""

    if not np.isfinite(brier_skill) or not np.isfinite(auc):
        return 0.0
    calibration_skill = float(np.clip(brier_skill, 0.0, 1.0))
    discrimination_skill = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
    sample_quality = float(
        np.clip(observations / max(2.0 * float(min_observations), 1.0), 0.0, 1.0)
    )
    return float(
        np.clip(
            math.sqrt(calibration_skill * discrimination_skill) * sample_quality,
            0.0,
            1.0,
        )
    )


def effective_scenario_multiplier(raw_multiplier: float, authority: float) -> float:
    """Shrink an uncalibrated defensive multiplier toward no adjustment."""

    raw = float(np.clip(raw_multiplier, 0.0, 1.0))
    calibrated_authority = float(np.clip(authority, 0.0, 1.0))
    return float(np.clip(1.0 - calibrated_authority * (1.0 - raw), raw, 1.0))


def _metric_values(group: pd.DataFrame) -> dict[str, float]:
    probabilities = pd.to_numeric(
        group["predicted_risk_off_probability"], errors="coerce"
    ).clip(1e-6, 1.0 - 1e-6)
    labels = pd.to_numeric(group["realized_risk_off"], errors="coerce")
    valid = probabilities.notna() & labels.notna()
    probabilities = probabilities[valid].astype(float)
    labels = labels[valid].astype(float)
    if probabilities.empty:
        return {
            "brier_score": np.nan,
            "climatology_brier_score": np.nan,
            "brier_skill": np.nan,
            "log_loss": np.nan,
            "auc": np.nan,
            "ece": np.nan,
            "sharpness": np.nan,
        }
    brier = float(np.mean((probabilities - labels) ** 2))
    base_rate = float(labels.mean())
    climatology_brier = float(np.mean((base_rate - labels) ** 2))
    brier_skill = (
        float(1.0 - brier / climatology_brier) if climatology_brier > 0 else 0.0
    )
    log_loss = float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )
    return {
        "brier_score": brier,
        "climatology_brier_score": climatology_brier,
        "brier_skill": brier_skill,
        "log_loss": log_loss,
        "auc": _auc(labels, probabilities),
        "ece": _expected_calibration_error(probabilities, labels),
        "sharpness": float(probabilities.std(ddof=0)),
    }


def _auc(labels: pd.Series, probabilities: pd.Series) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = probabilities.rank(method="average")
    positive_rank_sum = float(ranks[labels.eq(1.0)].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _expected_calibration_error(
    probabilities: pd.Series,
    labels: pd.Series,
    *,
    bins: int = 10,
) -> float:
    bucket = pd.cut(
        probabilities,
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    total = len(labels)
    error = 0.0
    for _, positions in labels.groupby(bucket, observed=True).groups.items():
        if len(positions) == 0:
            continue
        error += len(positions) / total * abs(
            float(labels.loc[positions].mean())
            - float(probabilities.loc[positions].mean())
        )
    return float(error)


def _block_bootstrap_intervals(
    group: pd.DataFrame,
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if group.empty or samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    ordered = group.sort_values("origin_date").reset_index(drop=True)
    observations = len(ordered)
    starts = np.arange(max(1, observations - block_length + 1))
    draws: dict[str, list[float]] = {"brier_skill": [], "auc": [], "ece": []}
    blocks_needed = int(math.ceil(observations / block_length))
    for _ in range(samples):
        selected: list[int] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            selected.extend(range(int(start), min(int(start) + block_length, observations)))
        sample = ordered.iloc[selected[:observations]]
        values = _metric_values(sample)
        for metric in draws:
            value = float(values[metric])
            if np.isfinite(value):
                draws[metric].append(value)
    intervals: dict[str, float] = {}
    for metric, values in draws.items():
        intervals[f"{metric}_ci_low"] = (
            float(np.quantile(values, 0.025)) if values else np.nan
        )
        intervals[f"{metric}_ci_high"] = (
            float(np.quantile(values, 0.975)) if values else np.nan
        )
    return intervals


def _authority_status(authority: float) -> str:
    if authority >= 0.70:
        return "high"
    if authority >= 0.40:
        return "moderate"
    if authority >= 0.15:
        return "low"
    return "insufficient"


def _forward_return(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    path = bounded_forward_fill(series.iloc[position : end + 1])
    if path.isna().any() or float(path.iloc[0]) == 0.0:
        return pd.NA
    return float(path.iloc[-1] / path.iloc[0] - 1.0)


def _forward_max_drawdown(series: pd.Series, position: int, days: int) -> float | pd.NA:
    end = position + days
    if position < 0 or end >= len(series):
        return pd.NA
    path = bounded_forward_fill(series.iloc[position : end + 1]).dropna()
    if path.empty:
        return pd.NA
    relative = path / path.iloc[0]
    return float((relative / relative.cummax() - 1.0).min())


def _wilson_interval(successes: int, observations: int, z: float = 1.96) -> tuple[float, float]:
    if observations <= 0:
        return np.nan, np.nan
    probability = successes / observations
    denominator = 1.0 + z**2 / observations
    center = (probability + z**2 / (2.0 * observations)) / denominator
    margin = z * math.sqrt(
        (probability * (1.0 - probability) + z**2 / (4.0 * observations))
        / observations
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def write_scenario_calibration_outputs(
    *,
    output_dir: Path,
    outcomes: pd.DataFrame,
    reliability: pd.DataFrame,
    metrics: pd.DataFrame,
    walk_forward: pd.DataFrame,
    latest_authority: pd.DataFrame,
    prices: pd.DataFrame,
    config: BotConfig,
    parameters: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "outcomes": output_dir / "scenario_outcomes.csv",
        "reliability": output_dir / "reliability.csv",
        "metrics": output_dir / "calibration_metrics.csv",
        "walk_forward": output_dir / "walk_forward_authority.csv",
        "latest_authority": output_dir / "latest_authority.csv",
        "summary": output_dir / "summary.md",
    }
    outcomes.to_csv(paths["outcomes"], index=False)
    reliability.to_csv(paths["reliability"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    walk_forward.to_csv(paths["walk_forward"], index=False)
    latest_authority.to_csv(paths["latest_authority"], index=False)
    paths["summary"].write_text(
        _markdown_summary(metrics, reliability, latest_authority),
        encoding="utf-8",
    )
    paths["authority_json"] = output_dir / "latest_authority.json"
    paths["authority_json"].write_text(
        json.dumps(
            {
                str(row["horizon"]): {
                    key: _json_value(value)
                    for key, value in row.items()
                    if key != "horizon"
                }
                for row in latest_authority.to_dict("records")
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = write_research_manifest(
        output_dir,
        study="scenario_probability_calibration",
        config=config,
        prices=prices,
        parameters=parameters,
        artifacts=[path.name for path in paths.values()],
    )
    return paths


def _markdown_summary(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    latest_authority: pd.DataFrame,
) -> str:
    lines = [
        "# Scenario Probability Calibration",
        "",
        "Risk-off probabilities are evaluated against matured SPY-versus-BIL and drawdown outcomes.",
        "Sizing authority is earned from expanding-history Brier skill and AUC; weak evidence shrinks authority toward zero.",
        "",
        "## Calibration metrics",
        "",
        *_markdown_table(metrics),
        "",
        "## Latest calibration-gated authority",
        "",
        *_markdown_table(latest_authority),
        "",
        "## Reliability curve",
        "",
        *_markdown_table(reliability[reliability["observations"].gt(0)]),
        "",
        "## Evidence limits",
        "",
        "- Retrospective research only; this does not prove prospective calibration.",
        "- Weekly origins overlap at longer horizons; block bootstrap intervals address dependence only approximately.",
        "- Current-universe survivorship and pre-inception proxy limitations remain unresolved.",
        "- Authority is a governance gate, not a claim that the probability model is optimal.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No eligible observations."]
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _json_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    return value
