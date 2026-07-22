from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskTimingAssessment:
    """Point-in-time escalation state for quantitative allocation sizing.

    The broad risk score remains a fragility measure.  This assessment answers a
    narrower question: has deterioration been confirmed strongly enough, across
    independent market groups, to reduce the allocation budget now?
    """

    state: str
    multiplier: float
    confirmation_breaks: tuple[str, ...]
    recovery_confirmations: tuple[str, ...]
    evidence: pd.DataFrame


def assess_risk_timing(
    risk_score: float,
    confirmation_matrix: pd.DataFrame,
    market_health: pd.DataFrame,
) -> RiskTimingAssessment:
    signals = _signal_states(confirmation_matrix)
    spy_1m = _health_value(market_health, "SPY", "return_1m")
    spy_3m = _health_value(market_health, "SPY", "return_3m")
    spy_dd = _health_value(market_health, "SPY", "drawdown")
    qqq_1m = _health_value(market_health, "QQQ", "return_1m")
    qqq_3m = _health_value(market_health, "QQQ", "return_3m")
    qqq_dd = _health_value(market_health, "QQQ", "drawdown")
    rsp_1m = _health_value(market_health, "RSP", "return_1m")
    hyg_1m = _health_value(market_health, "HYG", "return_1m")
    vixy_1m = _health_value(market_health, "VIXY", "return_1m")

    credit_break = _is_negative(signals.get("High Yield vs IG Credit")) and hyg_1m <= -0.01
    volatility_break = _is_negative(signals.get("Volatility ETF Pressure")) and vixy_1m >= 0.05
    breadth_break = _is_negative(signals.get("Equal Weight vs Cap Weight")) and (
        rsp_1m <= spy_1m - 0.01
    )
    trend_break = (
        (spy_1m <= -0.02 and spy_3m <= 0.0)
        or (qqq_1m <= -0.025 and qqq_3m <= 0.0)
        or (
            _is_negative(signals.get("SPY Trend"))
            and _is_negative(signals.get("QQQ Trend"))
        )
    )
    break_flags = {
        "credit": credit_break,
        "volatility": volatility_break,
        "breadth": breadth_break,
        "trend": trend_break,
    }
    breaks = tuple(name for name, active in break_flags.items() if active)

    recovery_flags = {
        "credit": hyg_1m >= 0.01,
        "volatility": vixy_1m <= -0.05,
        "breadth": rsp_1m >= spy_1m + 0.01,
        "trend": spy_1m >= 0.02 and qqq_1m >= 0.025,
    }
    recoveries = tuple(name for name, active in recovery_flags.items() if active)
    worst_drawdown = min(_finite_or_zero(spy_dd), _finite_or_zero(qqq_dd))

    # Escalation needs agreement across market groups.  Recovery needs broad,
    # short-horizon improvement before a still-elevated slow risk score is relaxed.
    if len(breaks) >= 3 and worst_drawdown <= -0.10:
        state, multiplier = "severe_break", 0.40
    elif len(breaks) >= 2:
        state, multiplier = "confirmed_break", 0.65
    elif len(recoveries) >= 3 and risk_score >= 0.45:
        state, multiplier = "stabilizing", 1.00
    elif len(breaks) == 1 and risk_score >= 0.45:
        state, multiplier = "warning", 0.90
    elif risk_score >= 0.45:
        state, multiplier = "fragile_intact", 1.00
    elif risk_score >= 0.25:
        state, multiplier = "watch", 1.00
    else:
        state, multiplier = "normal", 1.00

    evidence = pd.DataFrame(
        [
            {
                "theme": theme,
                "break_confirmed": bool(break_flags[theme]),
                "recovery_confirmed": bool(recovery_flags[theme]),
            }
            for theme in ("credit", "volatility", "breadth", "trend")
        ]
    )
    return RiskTimingAssessment(
        state=state,
        multiplier=multiplier,
        confirmation_breaks=breaks,
        recovery_confirmations=recoveries,
        evidence=evidence,
    )


def _signal_states(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty or "name" not in frame or "status" not in frame:
        return {}
    return dict(zip(frame["name"].astype(str), frame["status"].astype(str), strict=True))


def _is_negative(value: str | None) -> bool:
    return str(value).lower() in {"bearish", "risk_off", "risk-pressure", "risk_pressure"}


def _health_value(frame: pd.DataFrame, ticker: str, column: str) -> float:
    if frame.empty or ticker not in frame.index or column not in frame:
        return np.nan
    value = pd.to_numeric(pd.Series([frame.loc[ticker, column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _finite_or_zero(value: float) -> float:
    return value if np.isfinite(value) else 0.0
