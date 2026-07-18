from __future__ import annotations

from typing import Literal

import pandas as pd

CardTone = Literal["critical", "warning", "success", "neutral"]

_VALID_TONES: set[str] = {"critical", "warning", "success", "neutral"}


def normalize_tone(tone: object | None) -> CardTone:
    value = str(tone or "neutral").strip().lower()
    if value in _VALID_TONES:
        return value  # type: ignore[return-value]
    return "neutral"


def risk_status_tone(risk_status: object) -> CardTone:
    status = str(risk_status).strip().lower()
    if status in {"red", "orange"}:
        return "critical"
    if status == "yellow":
        return "warning"
    if status == "green":
        return "success"
    return "neutral"


def risk_score_tone(score: object) -> CardTone:
    value = _as_float(score)
    if value is None:
        return "neutral"
    if value >= 0.45:
        return "critical"
    if value >= 0.25:
        return "warning"
    return "success"


def risk_budget_tone(multiplier: object) -> CardTone:
    value = _as_float(multiplier)
    if value is None:
        return "neutral"
    if value < 0.70:
        return "critical"
    if value < 0.95:
        return "warning"
    if value <= 1.05:
        return "success"
    return "warning"


def probability_pressure_tone(probability: object, *, warning_at: float, critical_at: float) -> CardTone:
    value = _as_float(probability)
    if value is None:
        return "neutral"
    if value >= critical_at:
        return "critical"
    if value >= warning_at:
        return "warning"
    return "success"


def portfolio_risk_tone(level: object) -> CardTone:
    text = str(level).strip().lower()
    if text in {"constraint_breach", "breach", "critical"}:
        return "critical"
    if text in {"risk_reduced", "watch_correlation_shift"} or "watch" in text:
        return "warning"
    if text in {"within_limits", "ok", "clear"}:
        return "success"
    return "neutral"


def instability_tone(state_or_score: object) -> CardTone:
    score = _as_float(state_or_score)
    if score is not None:
        if score >= 0.55:
            return "critical"
        if score >= 0.35:
            return "warning"
        return "success"
    state = str(state_or_score).strip().lower()
    if state in {"stressed", "unstable", "high"}:
        return "critical"
    if state in {"elevated", "watch"}:
        return "warning"
    if state in {"calm", "contained", "low"}:
        return "success"
    return "neutral"


def expected_shortfall_tone(value: object) -> CardTone:
    return _threshold_tone(value, warning_at=0.02, critical_at=0.05, lower_is_better=True)


def stress_loss_tone(value: object) -> CardTone:
    return _threshold_tone(value, warning_at=0.10, critical_at=0.20, lower_is_better=True)


def beta_tone(value: object, *, warning_at: float, critical_at: float) -> CardTone:
    return _threshold_tone(value, warning_at=warning_at, critical_at=critical_at, lower_is_better=True)


def beta_delta_tone(value: object) -> CardTone:
    return _threshold_tone(value, warning_at=0.60, critical_at=0.85, lower_is_better=True)


def sleeve_exposure_tone(sleeve: str, percent_of_max: object) -> CardTone:
    value = _as_float(percent_of_max)
    if value is None:
        return "neutral"
    sleeve_key = sleeve.strip().lower()
    if sleeve_key == "defensive":
        if value >= 0.90:
            return "critical"
        if value >= 0.65:
            return "warning"
        return "neutral"
    if sleeve_key == "stocks":
        if value >= 0.95:
            return "critical"
        if value >= 0.80:
            return "warning"
        if value >= 0.30:
            return "success"
        return "neutral"
    if value <= 0.01:
        return "neutral"
    if value >= 0.80:
        return "warning"
    return "success"


def target_defensive_tone(defensive_weight: object) -> CardTone:
    value = _as_float(defensive_weight)
    if value is None:
        return "neutral"
    if value >= 0.90:
        return "critical"
    if value >= 0.65:
        return "warning"
    if value <= 0.35:
        return "success"
    return "neutral"


def decision_sanity_tone(signal: object) -> CardTone:
    text = str(signal).strip().lower()
    if any(token in text for token in ("cap not binding", "no sanity cap", "not binding")):
        return "success"
    if any(token in text for token in ("capped", "warning", "review")):
        return "warning"
    return "neutral"


def posture_calibration_tone(signal: object) -> CardTone:
    text = str(signal).strip().lower()
    if any(token in text for token in ("under-risk", "opportunity-cost", "event-driven")):
        return "warning"
    if any(token in text for token in ("supported", "balanced", "no bearish-bias")):
        return "success"
    if any(token in text for token in ("over-risk", "breach")):
        return "critical"
    return "neutral"


def _threshold_tone(
    value: object,
    *,
    warning_at: float,
    critical_at: float,
    lower_is_better: bool,
) -> CardTone:
    number = _as_float(value)
    if number is None:
        return "neutral"
    if lower_is_better:
        if number >= critical_at:
            return "critical"
        if number >= warning_at:
            return "warning"
        return "success"
    if number <= critical_at:
        return "critical"
    if number <= warning_at:
        return "warning"
    return "success"


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if cleaned.lower() in {"", "n/a", "nan", "none"}:
                return None
            if cleaned.endswith("%"):
                return float(cleaned[:-1]) / 100.0
            return float(cleaned)
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
