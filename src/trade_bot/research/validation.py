from __future__ import annotations

import math
from typing import cast

import pandas as pd


def add_overfit_diagnostics(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Add selection-adjusted validation diagnostics to experiment scorecards.

    These diagnostics are deliberately conservative. They do not replace true
    forward paper trading, but they make optimizer pressure visible by combining
    candidate count, walk-forward decay, left-tail behavior, and drawdown severity.
    """
    if scorecard.empty:
        return scorecard.copy()

    frame = scorecard.copy()
    for column in [
        "cagr",
        "promotion_score",
        "walk_forward_median_cagr",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "max_drawdown",
    ]:
        if column not in frame:
            frame[column] = float("nan")

    if "iteration" in frame:
        candidate_counts = frame.groupby("iteration")["strategy"].transform("count")
    else:
        candidate_counts = pd.Series(len(frame), index=frame.index)
    max_count = max(float(candidate_counts.max()), 1.0)
    selection_pressure = (candidate_counts.astype(float).map(_log1p) / _log1p(max_count)).clip(
        0.0,
        1.0,
    )

    cagr = pd.to_numeric(frame["cagr"].map(_optional_float), errors="coerce").fillna(0.0)
    walk_forward_cagr = pd.to_numeric(
        frame["walk_forward_median_cagr"].map(_optional_float),
        errors="coerce",
    ).fillna(cagr)
    walk_forward_positive_rate = (
        pd.to_numeric(frame["walk_forward_positive_rate"].map(_optional_float), errors="coerce")
        .fillna(0.50)
    )
    left_tail_return = pd.to_numeric(
        frame["left_tail_regime_return"].map(_optional_float),
        errors="coerce",
    ).fillna(-0.10)
    max_drawdown = pd.to_numeric(
        frame["max_drawdown"].map(_optional_float),
        errors="coerce",
    ).fillna(-0.20)

    holdout_decay = ((cagr - walk_forward_cagr).clip(lower=0.0) / 0.12).clip(0.0, 1.0)
    holdout_fragility = ((0.75 - walk_forward_positive_rate).clip(lower=0.0) / 0.35).clip(
        0.0,
        1.0,
    )
    left_tail_penalty = ((-0.08 - left_tail_return).clip(lower=0.0) / 0.25).clip(0.0, 1.0)
    drawdown_penalty = ((max_drawdown.abs() - 0.18).clip(lower=0.0) / 0.25).clip(0.0, 1.0)

    overfit_risk_score = (
        0.20 * selection_pressure
        + 0.30 * holdout_decay
        + 0.25 * holdout_fragility
        + 0.15 * left_tail_penalty
        + 0.10 * drawdown_penalty
    ).clip(0.0, 1.0)
    frame["selection_pressure"] = selection_pressure
    frame["holdout_decay"] = holdout_decay
    frame["holdout_fragility"] = holdout_fragility
    frame["left_tail_penalty"] = left_tail_penalty
    frame["drawdown_penalty"] = drawdown_penalty
    frame["overfit_risk_score"] = overfit_risk_score
    frame["overfit_risk_label"] = overfit_risk_score.map(_overfit_label)
    frame["selection_adjusted_promotion_score"] = pd.to_numeric(
        frame["promotion_score"].map(_optional_float),
        errors="coerce",
    ).fillna(0.0) * (1.0 - 0.60 * overfit_risk_score)
    frame["validation_tier"] = frame.apply(_validation_tier, axis=1)
    return frame


def _validation_tier(row: pd.Series) -> str:
    overfit = _optional_float(row.get("overfit_risk_score")) or 0.0
    walk_rate = _optional_float(row.get("walk_forward_positive_rate")) or 0.0
    left_tail = _optional_float(row.get("left_tail_regime_return")) or -1.0
    if overfit <= 0.30 and walk_rate >= 0.80 and left_tail >= -0.15:
        return "paper_champion_candidate"
    if overfit <= 0.45 and walk_rate >= 0.70 and left_tail >= -0.20:
        return "paper_challenger_candidate"
    if overfit >= 0.65 or walk_rate < 0.55 or left_tail < -0.30:
        return "reject_or_redesign"
    return "needs_more_holdout_evidence"


def _overfit_label(value: float) -> str:
    if value >= 0.75:
        return "critical"
    if value >= 0.55:
        return "high"
    if value >= 0.35:
        return "moderate"
    return "low"


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(object, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _log1p(value: float) -> float:
    return math.log1p(value)
