from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_RISK_AI_BETA_TICKERS,
    DEFAULT_RISK_BROAD_EQUITY_TICKERS,
    DEFAULT_RISK_COMMODITY_TICKERS,
    DEFAULT_RISK_CREDIT_TICKERS,
    DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS,
    DEFAULT_RISK_DEFENSIVE_TICKERS,
    DEFAULT_RISK_DURATION_TICKERS,
    DEFAULT_RISK_GOLD_TICKERS,
    DEFAULT_RISK_HIGH_BETA_TICKERS,
    DEFAULT_RISK_INTERNATIONAL_TICKERS,
    DEFAULT_RISK_SECTOR_TICKERS,
    TRADING_DAYS_PER_YEAR,
)

PHASES = (
    "normal_cycle",
    "acceleration",
    "pre_break",
    "early_unwind",
    "liquidation",
    "bottoming",
    "recovery",
    "post_unwind_compounding",
)
DEFAULT_PHASE_HORIZONS = (0, 21, 63, 126, 252)
DEFAULT_PHASE_VALIDATION_STEP_DAYS = 63
DEFAULT_PHASE_MIN_TRAIN_DAYS = 756
DEFAULT_PHASE_MAX_CANDIDATES = 60
CORE_BENCHMARKS = ("SPY", "QQQ", "BIL")


@dataclass(frozen=True)
class CycleTrackerRun:
    output_dir: Path
    artifacts: dict[str, Path]
    phase_probabilities: pd.DataFrame
    transition_forecast: pd.DataFrame
    evidence: pd.DataFrame
    candidate_scores: pd.DataFrame
    phase_candidate_frontier: pd.DataFrame
    validation_metrics: pd.DataFrame
    validation_observations: pd.DataFrame
    readout: str


def run_cycle_tracker(
    *,
    prices: pd.DataFrame,
    scenario_lattice: pd.DataFrame | None = None,
    output_dir: str | Path = "reports/cycle_tracker",
    candidate_tickers: Iterable[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_PHASE_HORIZONS,
    min_train_days: int = DEFAULT_PHASE_MIN_TRAIN_DAYS,
    origin_step_days: int = DEFAULT_PHASE_VALIDATION_STEP_DAYS,
) -> CycleTrackerRun:
    """Build a speculative-cycle tracker with prior-only validation.

    The tracker is an explanatory research layer. Feature rows at each historical
    origin are built only from prices available through that origin. Forward
    validation starts on the next trading session.
    """

    clean = _clean_prices(prices)
    if clean.empty:
        raise ValueError("Cycle tracker requires non-empty price history.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    horizons = tuple(dict.fromkeys(int(horizon) for horizon in horizons if int(horizon) >= 0))
    if not horizons:
        horizons = DEFAULT_PHASE_HORIZONS
    validation_horizons = tuple(horizon for horizon in horizons if horizon > 0)

    current_feature = build_cycle_feature_snapshot(clean)
    phase_probabilities = _phase_probability_frame(current_feature, horizon_days=0)
    transition_forecast = build_phase_transition_forecast(
        phase_probabilities,
        scenario_lattice=scenario_lattice,
        horizons=horizons,
    )
    evidence = _evidence_frame(current_feature)
    tickers = _candidate_ticker_list(clean, candidate_tickers)
    validation_observations = build_cycle_validation_observations(
        clean,
        tickers=tickers,
        horizons=validation_horizons,
        min_train_days=min_train_days,
        origin_step_days=origin_step_days,
    )
    validation_metrics = summarize_cycle_validation(validation_observations)
    candidate_scores = build_cycle_candidate_scores(
        clean,
        phase_probabilities,
        validation_metrics,
        tickers=tickers,
        horizon_days=_default_candidate_horizon(validation_horizons or tuple(h for h in DEFAULT_PHASE_HORIZONS if h > 0)),
    )
    phase_candidate_frontier = build_phase_candidate_frontier(
        clean,
        phase_probabilities,
        transition_forecast,
        validation_metrics,
        tickers=tickers,
    )
    readout = _build_readout(
        phase_probabilities=phase_probabilities,
        transition_forecast=transition_forecast,
        candidate_scores=candidate_scores,
        phase_candidate_frontier=phase_candidate_frontier,
        validation_metrics=validation_metrics,
    )

    artifacts = {
        "phase_probabilities": output / "cycle_phase_probabilities.csv",
        "transition_forecast": output / "cycle_transition_forecast.csv",
        "evidence": output / "cycle_evidence_components.csv",
        "candidate_scores": output / "cycle_candidate_scores.csv",
        "phase_candidate_frontier": output / "cycle_phase_candidate_frontier.csv",
        "validation_metrics": output / "cycle_validation_metrics.csv",
        "validation_observations": output / "cycle_validation_observations.csv",
        "summary": output / "summary.md",
    }
    phase_probabilities.to_csv(artifacts["phase_probabilities"], index=False)
    transition_forecast.to_csv(artifacts["transition_forecast"], index=False)
    evidence.to_csv(artifacts["evidence"], index=False)
    candidate_scores.to_csv(artifacts["candidate_scores"], index=False)
    phase_candidate_frontier.to_csv(artifacts["phase_candidate_frontier"], index=False)
    validation_metrics.to_csv(artifacts["validation_metrics"], index=False)
    validation_observations.to_csv(artifacts["validation_observations"], index=False)
    artifacts["summary"].write_text(readout, encoding="utf-8")

    return CycleTrackerRun(
        output_dir=output,
        artifacts=artifacts,
        phase_probabilities=phase_probabilities,
        transition_forecast=transition_forecast,
        evidence=evidence,
        candidate_scores=candidate_scores,
        phase_candidate_frontier=phase_candidate_frontier,
        validation_metrics=validation_metrics,
        validation_observations=validation_observations,
        readout=readout,
    )


def build_cycle_feature_snapshot(prices: pd.DataFrame) -> dict[str, object]:
    clean = _clean_prices(prices)
    if clean.empty:
        return _empty_feature_snapshot()
    returns = clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    qqq_spy_126 = _relative_return(clean, "QQQ", "SPY", 126)
    smh_spy_126 = _relative_return(clean, "SMH", "SPY", 126)
    qqq_rsp_126 = _relative_return(clean, "QQQ", "RSP", 126)
    rsp_spy_63 = _relative_return(clean, "RSP", "SPY", 63)
    iwm_spy_63 = _relative_return(clean, "IWM", "SPY", 63)
    hyg_lqd_63 = _relative_return(clean, "HYG", "LQD", 63)
    hyg_lqd_21 = _relative_return(clean, "HYG", "LQD", 21)
    qqq_21 = _absolute_return(clean, "QQQ", 21)
    qqq_63 = _absolute_return(clean, "QQQ", 63)
    qqq_252 = _absolute_return(clean, "QQQ", 252)
    smh_63 = _absolute_return(clean, "SMH", 63)
    spy_63 = _absolute_return(clean, "SPY", 63)
    vixy_21 = _absolute_return(clean, "VIXY", 21)
    vixy_63 = _absolute_return(clean, "VIXY", 63)
    qqq_drawdown = _drawdown(clean, "QQQ", 252)
    smh_drawdown = _drawdown(clean, "SMH", 252)
    spy_drawdown = _drawdown(clean, "SPY", 252)
    large_move_share = _large_move_share(returns, "SPY", 21)
    realized_vol_21 = _realized_vol(returns, "SPY", 21)
    above_200d = _above_moving_average(clean, "SPY", 200)

    leadership_acceleration = _average_scores(
        _threshold_score(qqq_spy_126, calm=-0.04, stressed=0.18),
        _threshold_score(smh_spy_126, calm=-0.06, stressed=0.30),
        _threshold_score(qqq_252, calm=0.04, stressed=0.35),
    )
    concentration_pressure = _average_scores(
        _threshold_score(qqq_rsp_126, calm=-0.04, stressed=0.18),
        _threshold_score(smh_spy_126, calm=-0.06, stressed=0.30),
        _threshold_score(-rsp_spy_63, calm=-0.02, stressed=0.10),
    )
    breadth_improvement = _average_scores(
        _threshold_score(rsp_spy_63, calm=-0.05, stressed=0.08),
        _threshold_score(iwm_spy_63, calm=-0.08, stressed=0.10),
    )
    credit_pressure = _threshold_score(-hyg_lqd_63, calm=-0.02, stressed=0.08)
    credit_improvement = _threshold_score(hyg_lqd_21, calm=-0.02, stressed=0.05)
    volatility_pressure = _average_scores(
        _threshold_score(vixy_21, calm=-0.10, stressed=0.35),
        _threshold_score(vixy_63, calm=-0.05, stressed=0.45),
        _threshold_score(realized_vol_21, calm=0.12, stressed=0.35),
    )
    volatility_easing = _threshold_score(-vixy_21, calm=-0.05, stressed=0.30)
    large_move_pressure = _threshold_score(large_move_share, calm=0.08, stressed=0.45)
    qqq_unwind = _average_scores(
        _threshold_score(-qqq_21, calm=-0.02, stressed=0.10),
        _threshold_score(-qqq_63, calm=-0.03, stressed=0.18),
        _threshold_score(-qqq_drawdown, calm=0.03, stressed=0.18),
    )
    smh_unwind = _average_scores(
        _threshold_score(-smh_63, calm=-0.04, stressed=0.25),
        _threshold_score(-smh_drawdown, calm=0.06, stressed=0.25),
    )
    market_liquidation = _average_scores(
        _threshold_score(-spy_63, calm=-0.03, stressed=0.16),
        _threshold_score(-spy_drawdown, calm=0.03, stressed=0.18),
        volatility_pressure,
        credit_pressure,
        large_move_pressure,
    )
    deep_drawdown = _average_scores(
        _threshold_score(-qqq_drawdown, calm=0.10, stressed=0.35),
        _threshold_score(-spy_drawdown, calm=0.06, stressed=0.25),
    )
    short_reversal = _average_scores(
        _threshold_score(qqq_21, calm=-0.03, stressed=0.08),
        _threshold_score(smh_63, calm=-0.08, stressed=0.10),
    )
    recovery_momentum = _average_scores(
        _threshold_score(qqq_63, calm=-0.03, stressed=0.12),
        _threshold_score(spy_63, calm=-0.02, stressed=0.10),
        breadth_improvement,
        credit_improvement,
    )
    low_vol = 1.0 - volatility_pressure
    broad_trend = _average_scores(
        _threshold_score(spy_63, calm=-0.03, stressed=0.12),
        _threshold_score(1.0 if above_200d else 0.0, calm=0.0, stressed=1.0),
        breadth_improvement,
    )

    raw_scores = {
        "acceleration": _weighted_mean(
            (leadership_acceleration, 0.45),
            (_threshold_score(qqq_63, calm=-0.02, stressed=0.16), 0.25),
            (low_vol, 0.15),
            (credit_improvement, 0.15),
        ),
        "pre_break": _weighted_mean(
            (leadership_acceleration, 0.30),
            (concentration_pressure, 0.30),
            (volatility_pressure, 0.15),
            (1.0 - breadth_improvement, 0.15),
            (large_move_pressure, 0.10),
        ),
        "early_unwind": _weighted_mean(
            (qqq_unwind, 0.35),
            (smh_unwind, 0.20),
            (volatility_pressure, 0.20),
            (credit_pressure, 0.15),
            (1.0 - breadth_improvement, 0.10),
        ),
        "liquidation": market_liquidation,
        "bottoming": _weighted_mean(
            (deep_drawdown, 0.25),
            (short_reversal, 0.30),
            (volatility_easing, 0.25),
            (credit_improvement, 0.20),
        ),
        "recovery": _weighted_mean(
            (recovery_momentum, 0.45),
            (volatility_easing, 0.20),
            (credit_improvement, 0.20),
            (1.0 - deep_drawdown, 0.15),
        ),
        "post_unwind_compounding": _weighted_mean(
            (broad_trend, 0.40),
            (breadth_improvement, 0.25),
            (low_vol, 0.20),
            (credit_improvement, 0.15),
        ),
    }
    extreme = max(raw_scores.values())
    raw_scores["normal_cycle"] = float(np.clip(0.75 - 0.45 * extreme, 0.08, 0.75))
    probabilities = _normalize_scores(raw_scores)
    dominant_phase = max(probabilities, key=probabilities.get)

    return {
        "as_of_date": str(clean.index.max().date()),
        "dominant_phase": dominant_phase,
        "dominant_phase_probability": float(probabilities[dominant_phase]),
        "probabilities": probabilities,
        "components": {
            "leadership_acceleration": leadership_acceleration,
            "concentration_pressure": concentration_pressure,
            "breadth_improvement": breadth_improvement,
            "credit_pressure": credit_pressure,
            "credit_improvement": credit_improvement,
            "volatility_pressure": volatility_pressure,
            "volatility_easing": volatility_easing,
            "large_move_pressure": large_move_pressure,
            "qqq_unwind": qqq_unwind,
            "smh_unwind": smh_unwind,
            "market_liquidation": market_liquidation,
            "deep_drawdown": deep_drawdown,
            "short_reversal": short_reversal,
            "recovery_momentum": recovery_momentum,
            "low_volatility": low_vol,
            "broad_trend": broad_trend,
        },
        "raw_values": {
            "qqq_vs_spy_126d": qqq_spy_126,
            "smh_vs_spy_126d": smh_spy_126,
            "qqq_vs_rsp_126d": qqq_rsp_126,
            "rsp_vs_spy_63d": rsp_spy_63,
            "iwm_vs_spy_63d": iwm_spy_63,
            "hyg_vs_lqd_63d": hyg_lqd_63,
            "hyg_vs_lqd_21d": hyg_lqd_21,
            "qqq_21d": qqq_21,
            "qqq_63d": qqq_63,
            "qqq_252d": qqq_252,
            "smh_63d": smh_63,
            "spy_63d": spy_63,
            "vixy_21d": vixy_21,
            "vixy_63d": vixy_63,
            "qqq_drawdown_252d": qqq_drawdown,
            "smh_drawdown_252d": smh_drawdown,
            "spy_drawdown_252d": spy_drawdown,
            "spy_large_move_share_21d": large_move_share,
            "spy_realized_vol_21d": realized_vol_21,
            "spy_above_200d": float(above_200d),
        },
    }


def build_phase_transition_forecast(
    phase_probabilities: pd.DataFrame,
    *,
    scenario_lattice: pd.DataFrame | None,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    latest = {
        str(row["phase"]): float(row["probability"])
        for _, row in phase_probabilities.iterrows()
        if str(row.get("phase", "")) in PHASES
    }
    if not latest:
        latest = {phase: 1.0 / len(PHASES) for phase in PHASES}
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if int(horizon) == 0:
            dominant_phase = max(latest, key=latest.get)
            for phase in PHASES:
                rows.append(
                    {
                        "horizon": _horizon_label(horizon),
                        "horizon_days": 0,
                        "phase": phase,
                        "probability": float(latest.get(phase, 0.0)),
                        "dominant_phase": dominant_phase,
                        "source": "current_feature_nowcast",
                    }
                )
            continue
        scenario_mix = _scenario_phase_prior(scenario_lattice, horizon_days=horizon)
        scenario_weight = _scenario_weight_for_horizon(horizon)
        combined = {
            phase: (1.0 - scenario_weight) * latest.get(phase, 0.0)
            + scenario_weight * scenario_mix.get(phase, 0.0)
            for phase in PHASES
        }
        combined = _normalize_scores(combined)
        label = _horizon_label(horizon)
        dominant_phase = max(combined, key=combined.get)
        for phase in PHASES:
            rows.append(
                {
                    "horizon": label,
                    "horizon_days": int(horizon),
                    "phase": phase,
                    "probability": float(combined[phase]),
                    "dominant_phase": dominant_phase,
                    "source": "current_feature_scenario_blend",
                }
            )
    return pd.DataFrame(rows)


def build_cycle_validation_observations(
    prices: pd.DataFrame,
    *,
    tickers: Iterable[str],
    horizons: tuple[int, ...],
    min_train_days: int,
    origin_step_days: int,
) -> pd.DataFrame:
    clean = _clean_prices(prices)
    tickers = [ticker for ticker in tickers if ticker in clean.columns]
    if clean.empty or not tickers:
        return pd.DataFrame()
    horizons = tuple(int(horizon) for horizon in horizons if int(horizon) > 0)
    if not horizons:
        return pd.DataFrame()
    max_horizon = max(horizons)
    rows: list[dict[str, object]] = []
    start = max(int(min_train_days), 252)
    stop = len(clean.index) - max_horizon - 2
    if stop <= start:
        return pd.DataFrame()
    for origin_pos in range(start, stop + 1, max(1, int(origin_step_days))):
        origin_date = clean.index[origin_pos]
        feature = build_cycle_feature_snapshot(clean.iloc[: origin_pos + 1])
        phase = str(feature["dominant_phase"])
        phase_probability = float(feature["dominant_phase_probability"])
        for horizon in horizons:
            start_pos = origin_pos + 1
            end_pos = min(start_pos + int(horizon), len(clean.index) - 1)
            if end_pos <= start_pos:
                continue
            spy_return = _forward_return(clean, "SPY", start_pos, end_pos)
            qqq_return = _forward_return(clean, "QQQ", start_pos, end_pos)
            bil_return = _forward_return(clean, "BIL", start_pos, end_pos)
            for ticker in tickers:
                forward_return = _forward_return(clean, ticker, start_pos, end_pos)
                if np.isnan(forward_return):
                    continue
                forward_drawdown = _forward_max_drawdown(clean, ticker, start_pos, end_pos)
                rows.append(
                    {
                        "origin_date": str(origin_date.date()),
                        "entry_date": str(clean.index[start_pos].date()),
                        "end_date": str(clean.index[end_pos].date()),
                        "dominant_phase": phase,
                        "phase_probability": phase_probability,
                        "horizon": _horizon_label(horizon),
                        "horizon_days": int(horizon),
                        "ticker": ticker,
                        "asset_role": _asset_role(ticker),
                        "forward_return": forward_return,
                        "forward_max_drawdown": forward_drawdown,
                        "spy_forward_return": spy_return,
                        "qqq_forward_return": qqq_return,
                        "bil_forward_return": bil_return,
                        "excess_vs_spy": forward_return - spy_return,
                        "excess_vs_qqq": forward_return - qqq_return,
                        "excess_vs_bil": forward_return - bil_return,
                        "beats_spy": bool(forward_return > spy_return),
                        "beats_qqq": bool(forward_return > qqq_return),
                        "beats_bil": bool(forward_return > bil_return),
                        "severe_drawdown": bool(forward_drawdown <= _severe_drawdown_cutoff(ticker, horizon)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_cycle_validation(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    frame = observations.copy()
    numeric_columns = [
        "forward_return",
        "forward_max_drawdown",
        "excess_vs_spy",
        "excess_vs_qqq",
        "excess_vs_bil",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(["dominant_phase", "horizon", "horizon_days", "ticker", "asset_role"])
    summary = grouped.agg(
        origins=("origin_date", "nunique"),
        median_forward_return=("forward_return", "median"),
        mean_forward_return=("forward_return", "mean"),
        median_forward_drawdown=("forward_max_drawdown", "median"),
        worst_forward_drawdown=("forward_max_drawdown", "min"),
        median_excess_vs_spy=("excess_vs_spy", "median"),
        median_excess_vs_qqq=("excess_vs_qqq", "median"),
        median_excess_vs_bil=("excess_vs_bil", "median"),
        hit_rate_vs_spy=("beats_spy", "mean"),
        hit_rate_vs_qqq=("beats_qqq", "mean"),
        hit_rate_vs_bil=("beats_bil", "mean"),
        severe_drawdown_rate=("severe_drawdown", "mean"),
    ).reset_index()
    summary["phase_rank_score"] = (
        pd.to_numeric(summary["median_excess_vs_qqq"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(summary["median_excess_vs_spy"], errors="coerce").fillna(0.0)
        - 0.5 * pd.to_numeric(summary["severe_drawdown_rate"], errors="coerce").fillna(0.0)
        + 0.1 * pd.to_numeric(summary["hit_rate_vs_qqq"], errors="coerce").fillna(0.0)
    )
    return summary.sort_values(
        ["horizon_days", "dominant_phase", "phase_rank_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_cycle_candidate_scores(
    prices: pd.DataFrame,
    phase_probabilities: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    *,
    tickers: Iterable[str],
    horizon_days: int,
) -> pd.DataFrame:
    clean = _clean_prices(prices)
    dominant_phase = _dominant_phase_from_probabilities(phase_probabilities)
    latest = clean.iloc[-1]
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker not in clean.columns or pd.isna(latest.get(ticker)):
            continue
        metrics = _matching_validation_metric(
            validation_metrics,
            phase=dominant_phase,
            ticker=ticker,
            horizon_days=horizon_days,
        )
        momentum_21 = _absolute_return(clean, ticker, 21)
        momentum_63 = _absolute_return(clean, ticker, 63)
        drawdown_252 = _drawdown(clean, ticker, 252)
        validation_score = (
            _safe_float(metrics.get("phase_rank_score"), default=0.0)
            if metrics is not None
            else 0.0
        )
        momentum_score = _average_scores(
            _threshold_score(momentum_21, calm=-0.05, stressed=0.08),
            _threshold_score(momentum_63, calm=-0.08, stressed=0.15),
        )
        drawdown_penalty = _threshold_score(-drawdown_252, calm=0.05, stressed=0.30)
        role = _asset_role(ticker)
        phase_fit = _phase_role_fit(dominant_phase, role)
        candidate_score = (
            0.45 * validation_score
            + 0.25 * momentum_score
            + 0.20 * phase_fit
            - 0.10 * drawdown_penalty
        )
        rows.append(
            {
                "ticker": ticker,
                "asset_role": role,
                "current_phase": dominant_phase,
                "horizon": _horizon_label(horizon_days),
                "horizon_days": int(horizon_days),
                "candidate_score": float(candidate_score),
                "candidate_role": _candidate_role(candidate_score, dominant_phase, role),
                "current_momentum_21d": momentum_21,
                "current_momentum_63d": momentum_63,
                "current_drawdown_252d": drawdown_252,
                "phase_forward_median_return": (
                    _safe_float(metrics.get("median_forward_return"), default=np.nan)
                    if metrics is not None
                    else np.nan
                ),
                "phase_median_excess_vs_spy": (
                    _safe_float(metrics.get("median_excess_vs_spy"), default=np.nan)
                    if metrics is not None
                    else np.nan
                ),
                "phase_median_excess_vs_qqq": (
                    _safe_float(metrics.get("median_excess_vs_qqq"), default=np.nan)
                    if metrics is not None
                    else np.nan
                ),
                "phase_hit_rate_vs_qqq": (
                    _safe_float(metrics.get("hit_rate_vs_qqq"), default=np.nan)
                    if metrics is not None
                    else np.nan
                ),
                "phase_median_forward_drawdown": (
                    _safe_float(metrics.get("median_forward_drawdown"), default=np.nan)
                    if metrics is not None
                    else np.nan
                ),
                "phase_origins": (
                    int(metrics.get("origins", 0)) if metrics is not None else 0
                ),
                "interpretation": _candidate_interpretation(ticker, dominant_phase, role),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values("candidate_score", ascending=False).reset_index(drop=True)


def build_phase_candidate_frontier(
    prices: pd.DataFrame,
    phase_probabilities: pd.DataFrame,
    transition_forecast: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    *,
    tickers: Iterable[str],
    top_n_per_phase: int = 8,
) -> pd.DataFrame:
    """Score conditional winners for every phase and horizon in the frontier.

    This is a current research shelf, not a historical training step. Historical
    performance inputs come from prior-only validation metrics, then current
    momentum/drawdown and the current phase probabilities are used to decide
    which assets deserve inspection if a given phase dominates.
    """

    clean = _clean_prices(prices)
    if clean.empty or transition_forecast.empty or validation_metrics.empty:
        return pd.DataFrame()
    tickers = [ticker for ticker in tickers if ticker in clean.columns]
    if not tickers:
        return pd.DataFrame()

    as_of_date = ""
    if not phase_probabilities.empty and "as_of_date" in phase_probabilities:
        as_of_values = phase_probabilities["as_of_date"].dropna().astype(str)
        if not as_of_values.empty:
            as_of_date = str(as_of_values.iloc[0])

    rows: list[dict[str, object]] = []
    forecast = transition_forecast.copy()
    forecast["horizon_days"] = pd.to_numeric(forecast["horizon_days"], errors="coerce")
    forecast["probability"] = pd.to_numeric(forecast["probability"], errors="coerce").fillna(0.0)
    for _, phase_row in forecast.iterrows():
        phase = str(phase_row.get("phase", ""))
        horizon_days = int(phase_row.get("horizon_days", 0))
        if phase not in PHASES or horizon_days <= 0:
            continue
        phase_probability = _safe_float(phase_row.get("probability"), default=0.0)
        phase_metrics = validation_metrics[
            validation_metrics["dominant_phase"].astype(str).eq(phase)
            & (
                pd.to_numeric(validation_metrics["horizon_days"], errors="coerce")
                == horizon_days
            )
        ].copy()
        if phase_metrics.empty:
            continue
        for ticker in tickers:
            matches = phase_metrics[phase_metrics["ticker"].astype(str).eq(ticker)]
            if matches.empty:
                continue
            metrics = matches.sort_values("phase_rank_score", ascending=False).iloc[0].to_dict()
            role = _asset_role(ticker)
            momentum_21 = _absolute_return(clean, ticker, 21)
            momentum_63 = _absolute_return(clean, ticker, 63)
            drawdown_252 = _drawdown(clean, ticker, 252)
            rank_score = _safe_float(metrics.get("phase_rank_score"), default=0.0)
            validation_score = float(np.clip(0.5 + rank_score, 0.0, 1.0))
            momentum_score = _average_scores(
                _threshold_score(momentum_21, calm=-0.05, stressed=0.08),
                _threshold_score(momentum_63, calm=-0.08, stressed=0.15),
            )
            drawdown_penalty = _threshold_score(-drawdown_252, calm=0.05, stressed=0.30)
            phase_fit = _phase_role_fit(phase, role)
            frontier_score = (
                0.40 * validation_score
                + 0.20 * phase_fit
                + 0.15 * momentum_score
                + 0.15 * phase_probability
                - 0.10 * drawdown_penalty
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "horizon": str(phase_row.get("horizon", _horizon_label(horizon_days))),
                    "horizon_days": horizon_days,
                    "phase": phase,
                    "phase_probability": phase_probability,
                    "ticker": ticker,
                    "asset_role": role,
                    "frontier_score": float(frontier_score),
                    "frontier_role": _candidate_role(frontier_score, phase, role),
                    "current_momentum_21d": momentum_21,
                    "current_momentum_63d": momentum_63,
                    "current_drawdown_252d": drawdown_252,
                    "median_forward_return": _safe_float(
                        metrics.get("median_forward_return"),
                        default=np.nan,
                    ),
                    "median_excess_vs_spy": _safe_float(
                        metrics.get("median_excess_vs_spy"),
                        default=np.nan,
                    ),
                    "median_excess_vs_qqq": _safe_float(
                        metrics.get("median_excess_vs_qqq"),
                        default=np.nan,
                    ),
                    "hit_rate_vs_qqq": _safe_float(
                        metrics.get("hit_rate_vs_qqq"),
                        default=np.nan,
                    ),
                    "median_forward_drawdown": _safe_float(
                        metrics.get("median_forward_drawdown"),
                        default=np.nan,
                    ),
                    "origins": int(_safe_float(metrics.get("origins"), default=0.0)),
                    "interpretation": _phase_candidate_interpretation(ticker, phase, role),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["horizon_days", "phase", "frontier_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    frame["rank"] = (
        frame.groupby(["horizon_days", "phase"])["frontier_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    frame = frame[frame["rank"] <= int(top_n_per_phase)].copy()
    return frame.sort_values(
        ["horizon_days", "phase_probability", "phase", "rank"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _phase_probability_frame(feature: dict[str, object], *, horizon_days: int) -> pd.DataFrame:
    probabilities = dict(feature.get("probabilities", {}))
    dominant = str(feature.get("dominant_phase", "normal_cycle"))
    rows = []
    for phase in PHASES:
        rows.append(
            {
                "as_of_date": str(feature.get("as_of_date", "")),
                "horizon": _horizon_label(horizon_days),
                "horizon_days": int(horizon_days),
                "phase": phase,
                "probability": float(probabilities.get(phase, 0.0)),
                "dominant_phase": dominant,
                "source": "current_feature_nowcast",
            }
        )
    return pd.DataFrame(rows)


def _evidence_frame(feature: dict[str, object]) -> pd.DataFrame:
    components = dict(feature.get("components", {}))
    raw_values = dict(feature.get("raw_values", {}))
    rows = []
    for component, score in components.items():
        rows.append(
            {
                "as_of_date": str(feature.get("as_of_date", "")),
                "component": component,
                "component_score": float(score),
                "state": _component_state(float(score)),
                "latest_value": np.nan,
                "interpretation": _component_interpretation(component),
            }
        )
    for name, value in raw_values.items():
        rows.append(
            {
                "as_of_date": str(feature.get("as_of_date", "")),
                "component": name,
                "component_score": np.nan,
                "state": "raw_value",
                "latest_value": _safe_float(value, default=np.nan),
                "interpretation": _raw_value_interpretation(name),
            }
        )
    return pd.DataFrame(rows)


def _scenario_phase_prior(
    scenario_lattice: pd.DataFrame | None,
    *,
    horizon_days: int,
) -> dict[str, float]:
    base = {phase: 1.0 / len(PHASES) for phase in PHASES}
    if scenario_lattice is None or scenario_lattice.empty:
        return base
    if not {"risk_bucket", "probability"}.issubset(scenario_lattice.columns):
        return base
    frame = scenario_lattice.copy()
    if "horizon" in frame:
        label = _scenario_horizon_label(horizon_days)
        matching = frame[frame["horizon"].astype(str).eq(label)]
        if matching.empty:
            matching = frame
        frame = matching
    probabilities = pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0)
    if probabilities.sum() <= 0:
        return base
    phase_scores = dict.fromkeys(PHASES, 0.0)
    for risk_bucket, probability in zip(frame["risk_bucket"].astype(str), probabilities, strict=False):
        mapped = _risk_bucket_phase_weights(risk_bucket)
        for phase, weight in mapped.items():
            phase_scores[phase] += float(probability) * weight
    return _normalize_scores(phase_scores)


def _risk_bucket_phase_weights(risk_bucket: str) -> dict[str, float]:
    bucket = risk_bucket.lower()
    if "risk_off" in bucket:
        return {
            "early_unwind": 0.30,
            "liquidation": 0.35,
            "bottoming": 0.15,
            "normal_cycle": 0.20,
        }
    if "transition" in bucket:
        return {
            "pre_break": 0.25,
            "early_unwind": 0.25,
            "bottoming": 0.20,
            "recovery": 0.15,
            "normal_cycle": 0.15,
        }
    if "fragile" in bucket:
        return {
            "acceleration": 0.25,
            "pre_break": 0.40,
            "early_unwind": 0.15,
            "normal_cycle": 0.20,
        }
    if "risk_on" in bucket:
        return {
            "acceleration": 0.35,
            "post_unwind_compounding": 0.30,
            "recovery": 0.15,
            "normal_cycle": 0.20,
        }
    return {"normal_cycle": 1.0}


def _candidate_ticker_list(prices: pd.DataFrame, candidate_tickers: Iterable[str] | None) -> list[str]:
    if candidate_tickers is None:
        raw = (
            list(CORE_BENCHMARKS)
            + list(DEFAULT_RISK_BROAD_EQUITY_TICKERS)
            + list(DEFAULT_RISK_HIGH_BETA_TICKERS)
            + list(DEFAULT_RISK_AI_BETA_TICKERS)
            + list(DEFAULT_RISK_INTERNATIONAL_TICKERS)
            + list(DEFAULT_RISK_DEFENSIVE_TICKERS)
            + list(DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS)
            + list(DEFAULT_RISK_DURATION_TICKERS)
            + list(DEFAULT_RISK_CREDIT_TICKERS)
            + list(DEFAULT_RISK_COMMODITY_TICKERS)
            + list(DEFAULT_RISK_GOLD_TICKERS)
            + list(DEFAULT_RISK_SECTOR_TICKERS)
        )
    else:
        raw = list(candidate_tickers)
    deduped: list[str] = []
    for ticker in raw:
        normalized = str(ticker).upper().strip()
        if normalized and normalized in prices.columns and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:DEFAULT_PHASE_MAX_CANDIDATES]


def _asset_role(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in DEFAULT_RISK_DEFENSIVE_TICKERS:
        return "cash_defensive"
    if ticker in DEFAULT_RISK_AI_BETA_TICKERS or ticker in {"QQQ", "SMH", "SOXX", "IGV", "XLK"}:
        return "ai_growth"
    if ticker in DEFAULT_RISK_INTERNATIONAL_TICKERS:
        return "international"
    if ticker in DEFAULT_RISK_GOLD_TICKERS:
        return "gold"
    if ticker in DEFAULT_RISK_COMMODITY_TICKERS:
        return "commodity"
    if ticker in DEFAULT_RISK_DURATION_TICKERS:
        return "duration"
    if ticker in DEFAULT_RISK_CREDIT_TICKERS:
        return "credit"
    if ticker in DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS:
        return "defensive_equity"
    if ticker in DEFAULT_RISK_SECTOR_TICKERS:
        return "sector"
    if ticker in DEFAULT_RISK_BROAD_EQUITY_TICKERS:
        return "broad_equity"
    return "other"


def _phase_role_fit(phase: str, role: str) -> float:
    matrix = {
        "acceleration": {
            "ai_growth": 0.95,
            "sector": 0.75,
            "broad_equity": 0.65,
            "international": 0.45,
            "cash_defensive": 0.10,
        },
        "pre_break": {
            "ai_growth": 0.55,
            "sector": 0.50,
            "defensive_equity": 0.65,
            "cash_defensive": 0.55,
            "gold": 0.55,
        },
        "early_unwind": {
            "cash_defensive": 0.95,
            "gold": 0.70,
            "duration": 0.65,
            "defensive_equity": 0.55,
            "ai_growth": 0.20,
        },
        "liquidation": {
            "cash_defensive": 1.00,
            "duration": 0.65,
            "gold": 0.55,
            "credit": 0.15,
            "ai_growth": 0.10,
        },
        "bottoming": {
            "broad_equity": 0.65,
            "ai_growth": 0.70,
            "international": 0.60,
            "cash_defensive": 0.35,
            "credit": 0.45,
        },
        "recovery": {
            "ai_growth": 0.85,
            "broad_equity": 0.75,
            "international": 0.70,
            "sector": 0.65,
            "cash_defensive": 0.15,
        },
        "post_unwind_compounding": {
            "broad_equity": 0.80,
            "international": 0.75,
            "ai_growth": 0.65,
            "defensive_equity": 0.55,
            "cash_defensive": 0.10,
        },
        "normal_cycle": {
            "broad_equity": 0.70,
            "defensive_equity": 0.60,
            "international": 0.60,
            "ai_growth": 0.55,
            "cash_defensive": 0.25,
        },
    }
    return float(matrix.get(phase, {}).get(role, 0.45))


def _candidate_role(score: float, phase: str, role: str) -> str:
    if phase in {"early_unwind", "liquidation"} and role == "cash_defensive":
        return "defend"
    if score >= 0.55:
        return "scale_candidate"
    if score >= 0.35:
        return "starter_reentry"
    if score >= 0.15:
        return "watch"
    return "avoid"


def _candidate_interpretation(ticker: str, phase: str, role: str) -> str:
    if phase in {"early_unwind", "liquidation"} and role == "cash_defensive":
        return f"{ticker} is a defensive candidate for unwind/liquidation phases."
    if phase in {"bottoming", "recovery"} and role in {"ai_growth", "broad_equity", "international"}:
        return f"{ticker} is a possible post-unwind re-entry candidate if recovery evidence holds."
    if phase in {"acceleration", "pre_break"} and role == "ai_growth":
        return f"{ticker} can still participate in leadership, but watch concentration and unwind evidence."
    return f"{ticker} is scored as a {role} candidate for the current phase."


def _phase_candidate_interpretation(ticker: str, phase: str, role: str) -> str:
    if phase in {"early_unwind", "liquidation"}:
        if role == "cash_defensive":
            return f"{ticker} is a defensive winner candidate if {phase} dominates."
        if role in {"gold", "duration", "defensive_equity"}:
            return f"{ticker} is a possible ballast candidate if {phase} dominates."
        return f"{ticker} should be treated cautiously if {phase} dominates."
    if phase == "bottoming":
        return f"{ticker} is ranked for bottoming/re-entry evidence after an unwind."
    if phase in {"recovery", "post_unwind_compounding"}:
        return f"{ticker} is ranked for post-unwind compounding if recovery broadens."
    if phase in {"acceleration", "pre_break"} and role == "ai_growth":
        return f"{ticker} is ranked for leadership participation, with fragility risk monitored."
    return f"{ticker} is ranked as a {role} candidate if {phase} dominates."


def _matching_validation_metric(
    validation_metrics: pd.DataFrame,
    *,
    phase: str,
    ticker: str,
    horizon_days: int,
) -> dict[str, object] | None:
    if validation_metrics.empty:
        return None
    matches = validation_metrics[
        validation_metrics["dominant_phase"].astype(str).eq(phase)
        & validation_metrics["ticker"].astype(str).eq(ticker)
        & (pd.to_numeric(validation_metrics["horizon_days"], errors="coerce") == horizon_days)
    ]
    if matches.empty:
        matches = validation_metrics[
            validation_metrics["ticker"].astype(str).eq(ticker)
            & (pd.to_numeric(validation_metrics["horizon_days"], errors="coerce") == horizon_days)
        ]
    if matches.empty:
        return None
    return matches.sort_values("phase_rank_score", ascending=False).iloc[0].to_dict()


def _default_candidate_horizon(horizons: tuple[int, ...]) -> int:
    if 63 in horizons:
        return 63
    return sorted(horizons)[min(len(horizons) - 1, 0)]


def _dominant_phase_from_probabilities(phase_probabilities: pd.DataFrame) -> str:
    if phase_probabilities.empty:
        return "normal_cycle"
    ordered = phase_probabilities.sort_values("probability", ascending=False)
    return str(ordered.iloc[0]["phase"])


def _component_interpretation(component: str) -> str:
    text = {
        "leadership_acceleration": "Growth/AI leadership is moving faster than the broad market.",
        "concentration_pressure": "Leadership is narrow enough to raise fragility risk.",
        "breadth_improvement": "Equal-weight and small-cap participation are improving.",
        "credit_pressure": "Credit risk appetite is deteriorating versus safer bonds.",
        "credit_improvement": "Credit risk appetite is stabilizing or improving.",
        "volatility_pressure": "Volatility and realized movement are rising.",
        "volatility_easing": "Volatility pressure is fading.",
        "large_move_pressure": "Large daily index moves are becoming more common.",
        "qqq_unwind": "Nasdaq/growth leadership is actively unwinding.",
        "smh_unwind": "Semiconductor/AI leadership is actively unwinding.",
        "market_liquidation": "Broad equity, volatility, and credit stress resemble liquidation pressure.",
        "deep_drawdown": "Core risk assets have already experienced a meaningful drawdown.",
        "short_reversal": "Short-term reversal evidence is improving from a drawdown.",
        "recovery_momentum": "Risk assets, breadth, and credit are recovering together.",
        "low_volatility": "Volatility pressure is low.",
        "broad_trend": "Broad market trend and participation are constructive.",
    }
    return text.get(component, component.replace("_", " "))


def _raw_value_interpretation(name: str) -> str:
    return name.replace("_", " ")


def _component_state(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _build_readout(
    *,
    phase_probabilities: pd.DataFrame,
    transition_forecast: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    phase_candidate_frontier: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> str:
    dominant = _dominant_phase_from_probabilities(phase_probabilities)
    top_probability = float(phase_probabilities["probability"].max()) if not phase_probabilities.empty else float("nan")
    top_candidates = (
        ", ".join(candidate_scores.head(5)["ticker"].astype(str).tolist())
        if not candidate_scores.empty
        else "none"
    )
    validation_rows = len(validation_metrics)
    frontier_rows = len(phase_candidate_frontier)
    horizon_read = ""
    if not transition_forecast.empty:
        pieces = []
        for horizon, group in transition_forecast.groupby("horizon", sort=False):
            row = group.sort_values("probability", ascending=False).iloc[0]
            pieces.append(f"{horizon}: {row['phase']} {float(row['probability']):.1%}")
        horizon_read = "; ".join(pieces)
    return (
        "# Speculative Cycle Tracker\n\n"
        f"Dominant nowcast phase: **{dominant}** ({top_probability:.1%}).\n\n"
        f"Phase frontier by horizon: {horizon_read or 'not available'}.\n\n"
        f"Top conditional candidates for the current phase: {top_candidates}.\n\n"
        f"Phase/horizon winner frontier rows: {frontier_rows:,}.\n\n"
        f"Validation metric rows: {validation_rows:,}.\n\n"
        "Use this as a research/watch layer. It identifies which speculative-cycle "
        "phase the market most resembles, which phases are plausible across horizons, "
        "and which assets historically behaved better in similar prior states. It is "
        "not an allocation override and does not claim to time a bubble peak."
    )


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.DataFrame()
    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.loc[:, ~clean.columns.duplicated()]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    return clean.dropna(how="all").ffill()


def _empty_feature_snapshot() -> dict[str, object]:
    probabilities = {phase: 1.0 / len(PHASES) for phase in PHASES}
    return {
        "as_of_date": "",
        "dominant_phase": "normal_cycle",
        "dominant_phase_probability": probabilities["normal_cycle"],
        "probabilities": probabilities,
        "components": {},
        "raw_values": {},
    }


def _absolute_return(prices: pd.DataFrame, ticker: str, lookback: int) -> float:
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].dropna()
    if len(series) <= lookback:
        return np.nan
    start = float(series.iloc[-lookback - 1])
    end = float(series.iloc[-1])
    if start <= 0:
        return np.nan
    return end / start - 1.0


def _relative_return(prices: pd.DataFrame, ticker: str, benchmark: str, lookback: int) -> float:
    first = _absolute_return(prices, ticker, lookback)
    second = _absolute_return(prices, benchmark, lookback)
    if np.isnan(first) or np.isnan(second):
        return np.nan
    return first - second


def _drawdown(prices: pd.DataFrame, ticker: str, lookback: int) -> float:
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].dropna().tail(lookback)
    if series.empty:
        return np.nan
    peak = float(series.max())
    latest = float(series.iloc[-1])
    if peak <= 0:
        return np.nan
    return latest / peak - 1.0


def _realized_vol(returns: pd.DataFrame, ticker: str, lookback: int) -> float:
    if ticker not in returns.columns:
        return np.nan
    series = returns[ticker].dropna().tail(lookback)
    if len(series) < max(5, lookback // 3):
        return np.nan
    return float(series.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _large_move_share(returns: pd.DataFrame, ticker: str, lookback: int) -> float:
    if ticker not in returns.columns:
        return np.nan
    series = returns[ticker].dropna().tail(lookback)
    if series.empty:
        return np.nan
    return float(series.abs().ge(0.01).mean())


def _above_moving_average(prices: pd.DataFrame, ticker: str, lookback: int) -> bool:
    if ticker not in prices.columns:
        return False
    series = prices[ticker].dropna()
    if len(series) < max(20, lookback // 2):
        return False
    ma = float(series.tail(lookback).mean())
    latest = float(series.iloc[-1])
    return latest >= ma if ma > 0 else False


def _forward_return(prices: pd.DataFrame, ticker: str, start_pos: int, end_pos: int) -> float:
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].iloc[start_pos : end_pos + 1].dropna()
    if len(series) < 2:
        return np.nan
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0:
        return np.nan
    return end / start - 1.0


def _forward_max_drawdown(prices: pd.DataFrame, ticker: str, start_pos: int, end_pos: int) -> float:
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].iloc[start_pos : end_pos + 1].dropna()
    if len(series) < 2:
        return np.nan
    wealth = series.astype(float) / float(series.iloc[0])
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _threshold_score(value: float, *, calm: float, stressed: float) -> float:
    if value is None or np.isnan(value):
        return 0.5
    if stressed == calm:
        return 0.5
    raw = (float(value) - calm) / (stressed - calm)
    return float(np.clip(raw, 0.0, 1.0))


def _average_scores(*values: float) -> float:
    usable = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not usable:
        return 0.5
    return float(np.clip(np.mean(usable), 0.0, 1.0))


def _weighted_mean(*items: tuple[float, float]) -> float:
    values = [(float(value), float(weight)) for value, weight in items if not np.isnan(value)]
    if not values:
        return 0.5
    numerator = sum(value * weight for value, weight in values)
    denominator = sum(weight for _value, weight in values)
    if denominator <= 0:
        return 0.5
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {phase: max(0.0, float(scores.get(phase, 0.0))) for phase in PHASES}
    total = sum(cleaned.values())
    if total <= 0:
        return {phase: 1.0 / len(PHASES) for phase in PHASES}
    return {phase: value / total for phase, value in cleaned.items()}


def _scenario_weight_for_horizon(horizon_days: int) -> float:
    if horizon_days <= 0:
        return 0.0
    if horizon_days <= 21:
        return 0.30
    if horizon_days <= 63:
        return 0.45
    if horizon_days <= 126:
        return 0.60
    return 0.70


def _horizon_label(horizon_days: int) -> str:
    mapping = {0: "0m", 5: "1w", 21: "1m", 42: "2m", 63: "3m", 126: "6m", 252: "1y"}
    return mapping.get(int(horizon_days), f"{int(horizon_days)}d")


def _scenario_horizon_label(horizon_days: int) -> str:
    if horizon_days <= 0:
        return "0m"
    if horizon_days <= 21:
        return "1m"
    if horizon_days <= 63:
        return "3m"
    if horizon_days <= 126:
        return "6m"
    return "1y"


def _severe_drawdown_cutoff(ticker: str, horizon_days: int) -> float:
    role = _asset_role(ticker)
    base = -0.04 if role == "cash_defensive" else -0.12
    if horizon_days >= 126:
        return base * 1.5
    if horizon_days <= 21:
        return base * 0.6
    return base


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default
