from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS,
    DEFAULT_FORWARD_SIMULATION_FALLBACK_PROBABILITIES,
    DEFAULT_FORWARD_SIMULATION_INITIAL_SCENARIO_WEIGHT,
    DEFAULT_FORWARD_SIMULATION_MIN_REGIME_OBSERVATIONS,
    DEFAULT_FORWARD_SIMULATION_PATHS,
    DEFAULT_FORWARD_SIMULATION_RANDOM_SEED,
    DEFAULT_FORWARD_SIMULATION_TRANSITION_SCENARIO_WEIGHT,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_MIN_TRAIN_DAYS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_ORIGIN_FREQUENCY,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_RAMP_DRAWDOWN_LIMIT,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_RAMP_SEVERE_PROBABILITY,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_WAIT_SEVERE_PROBABILITY,
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
    DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR,
)
from trade_bot.research.strategy_outcome_utility import (
    annual_contribution_schedule,
    contribution_amount_for_day,
)

REGIME_BUCKETS = ("risk_off", "transition", "risk_on_fragile", "risk_on")


@dataclass(frozen=True)
class ForwardSimulationConfig:
    """Configuration for regime-conditioned forward path simulation.

    The engine is intentionally empirical: it samples realized historical return
    blocks from regime-labeled strategy history, then uses current scenario
    probabilities to bias the starting state and future transition probabilities.
    """

    horizon_years: float = DEFAULT_OUTCOME_HORIZON_YEARS
    starting_account_value: float = DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE
    annual_contribution: float = DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION
    contribution_timing: str = DEFAULT_OUTCOME_CONTRIBUTION_TIMING
    trading_days_per_year: int = DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR
    paths: int = DEFAULT_FORWARD_SIMULATION_PATHS
    block_days: int = DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS
    random_seed: int = DEFAULT_FORWARD_SIMULATION_RANDOM_SEED
    initial_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_INITIAL_SCENARIO_WEIGHT
    transition_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_TRANSITION_SCENARIO_WEIGHT
    min_regime_observations: int = DEFAULT_FORWARD_SIMULATION_MIN_REGIME_OBSERVATIONS
    hard_drawdown_limit: float = DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT


@dataclass(frozen=True)
class ForwardSimulationValidationConfig:
    """Rolling-origin calibration settings for the forward simulation engine."""

    origin_frequency: str = DEFAULT_FORWARD_SIMULATION_VALIDATION_ORIGIN_FREQUENCY
    horizons: tuple[tuple[str, int], ...] = tuple(
        DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS.items()
    )
    min_train_days: int = DEFAULT_FORWARD_SIMULATION_VALIDATION_MIN_TRAIN_DAYS
    trading_days_per_year: int = DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR
    paths: int = DEFAULT_FORWARD_SIMULATION_PATHS
    block_days: int = DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS
    random_seed: int = DEFAULT_FORWARD_SIMULATION_RANDOM_SEED
    initial_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_INITIAL_SCENARIO_WEIGHT
    transition_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_TRANSITION_SCENARIO_WEIGHT
    min_regime_observations: int = DEFAULT_FORWARD_SIMULATION_MIN_REGIME_OBSERVATIONS
    interval_low: float = DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW
    interval_high: float = DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH
    severe_drawdown_limit: float = DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT
    ramp_drawdown_limit: float = DEFAULT_FORWARD_SIMULATION_VALIDATION_RAMP_DRAWDOWN_LIMIT
    ramp_severe_probability: float = DEFAULT_FORWARD_SIMULATION_VALIDATION_RAMP_SEVERE_PROBABILITY
    wait_severe_probability: float = DEFAULT_FORWARD_SIMULATION_VALIDATION_WAIT_SEVERE_PROBABILITY


def scenario_bucket_probabilities(scenario_outlook: pd.DataFrame | None) -> pd.Series:
    """Return normalized 1M scenario probabilities by broad risk bucket."""

    base = pd.Series(DEFAULT_FORWARD_SIMULATION_FALLBACK_PROBABILITIES, dtype=float)
    base = base.reindex(REGIME_BUCKETS).fillna(0.0)
    if scenario_outlook is None or scenario_outlook.empty:
        return _normalize_probability_vector(base)
    if "risk_bucket" not in scenario_outlook or "probability" not in scenario_outlook:
        return _normalize_probability_vector(base)

    frame = scenario_outlook.copy()
    if "horizon" in frame:
        one_month = frame[frame["horizon"].astype(str).eq("1m")]
        if not one_month.empty:
            frame = one_month

    probabilities = pd.Series(0.0, index=REGIME_BUCKETS, dtype=float)
    for _, row in frame.iterrows():
        bucket = _normalize_bucket(row.get("risk_bucket"))
        if bucket not in probabilities.index:
            continue
        value = _safe_float(row.get("probability"))
        if value is None:
            continue
        probabilities.loc[bucket] += max(value, 0.0)

    if float(probabilities.sum()) <= 0.0:
        return _normalize_probability_vector(base)
    return _normalize_probability_vector(probabilities)


def build_regime_return_library(
    daily_returns: pd.Series | np.ndarray,
    *,
    config: ForwardSimulationConfig | None = None,
) -> pd.DataFrame:
    """Label historical strategy returns into coarse simulation regimes."""

    cfg = config or ForwardSimulationConfig()
    returns = _clean_returns_series(daily_returns)
    if returns.empty:
        return pd.DataFrame(columns=["return", "regime"])

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    rolling_21 = _rolling_compound_return(returns, 21)
    rolling_63 = _rolling_compound_return(returns, 63)
    vol_21 = returns.rolling(21, min_periods=5).std() * np.sqrt(cfg.trading_days_per_year)
    vol_21 = vol_21.fillna(vol_21.median()).fillna(0.0)

    daily_q10 = _quantile(returns, 0.10)
    rolling_21_q20 = _quantile(rolling_21, 0.20)
    rolling_63_q50 = _quantile(rolling_63, 0.50)
    rolling_63_q60 = _quantile(rolling_63, 0.60)
    vol_q60 = _quantile(vol_21, 0.60)
    vol_q70 = _quantile(vol_21, 0.70)
    vol_q75 = _quantile(vol_21, 0.75)

    labels = pd.Series("transition", index=returns.index, dtype=object)
    risk_off = (
        (drawdown <= -0.10)
        | (returns <= daily_q10)
        | ((rolling_21 <= rolling_21_q20) & (vol_21 >= vol_q60))
    )
    risk_on_fragile = (rolling_63 >= rolling_63_q60) & (vol_21 >= vol_q70) & ~risk_off
    risk_on = (rolling_63 >= rolling_63_q50) & (vol_21 <= vol_q75) & ~risk_off

    labels.loc[risk_on] = "risk_on"
    labels.loc[risk_on_fragile] = "risk_on_fragile"
    labels.loc[risk_off] = "risk_off"

    library = pd.DataFrame({"return": returns, "regime": labels})
    return _repair_sparse_regime_labels(library, min_observations=cfg.min_regime_observations)


def simulate_regime_conditioned_paths(
    daily_returns: pd.Series | np.ndarray,
    *,
    scenario_outlook: pd.DataFrame | None = None,
    config: ForwardSimulationConfig | None = None,
) -> pd.DataFrame:
    """Simulate forward account paths using current scenarios and historical regimes."""

    cfg = config or ForwardSimulationConfig()
    library = build_regime_return_library(daily_returns, config=cfg)
    if library.empty or cfg.paths <= 0 or cfg.horizon_years <= 0:
        return _empty_paths_frame()

    scenario_probs = scenario_bucket_probabilities(scenario_outlook)
    historical_probs = _historical_regime_probabilities(library)
    start_probs = _blend_probabilities(
        historical_probs,
        scenario_probs,
        scenario_weight=cfg.initial_scenario_weight,
    )
    transition_matrix = _blend_transition_matrix(
        _empirical_transition_matrix(library, block_days=cfg.block_days),
        scenario_probs,
        scenario_weight=cfg.transition_scenario_weight,
    )
    block_library = _build_block_library(library, block_days=cfg.block_days)
    contribution_schedule = annual_contribution_schedule(
        annual_contribution=cfg.annual_contribution,
        trading_days_per_year=cfg.trading_days_per_year,
        contribution_timing=cfg.contribution_timing,
    )

    total_days = int(cfg.horizon_years * cfg.trading_days_per_year)
    if total_days <= 0:
        return _empty_paths_frame()

    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, float | int]] = []
    for path_id in range(cfg.paths):
        wealth = float(cfg.starting_account_value)
        peak = wealth
        max_drawdown = 0.0
        drawdown_square_sum = 0.0
        regime_counts = dict.fromkeys(REGIME_BUCKETS, 0)
        regime = _sample_bucket(rng, start_probs)
        day_idx = 0
        while day_idx < total_days:
            block = _sample_block(rng, block_library, regime=regime)
            if block.size == 0:
                break
            usable_days = min(int(block.size), total_days - day_idx)
            for day_return in block[:usable_days]:
                regime_counts[regime] += 1
                wealth *= 1.0 + float(day_return)
                day_idx += 1
                contribution = contribution_amount_for_day(
                    day_idx,
                    contribution_schedule,
                    trading_days_per_year=cfg.trading_days_per_year,
                )
                if contribution:
                    wealth += contribution
                peak = max(peak, wealth)
                current_drawdown = min(wealth / peak - 1.0, 0.0) if peak else 0.0
                max_drawdown = min(max_drawdown, current_drawdown)
                drawdown_square_sum += current_drawdown**2
                if day_idx >= total_days:
                    break
            regime = _sample_bucket(rng, transition_matrix.loc[regime])

        row: dict[str, float | int] = {
            "path": path_id,
            "terminal_wealth": wealth,
            "max_drawdown": max_drawdown,
            "ulcer_index": float(np.sqrt(drawdown_square_sum / max(day_idx, 1))),
        }
        for bucket in REGIME_BUCKETS:
            row[f"share_{bucket}"] = regime_counts[bucket] / max(day_idx, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_forward_simulation(
    paths: pd.DataFrame,
    *,
    config: ForwardSimulationConfig | None = None,
) -> dict[str, float | int | None]:
    cfg = config or ForwardSimulationConfig()
    if paths.empty:
        return {
            "paths": 0,
            "terminal_wealth_p10": None,
            "terminal_wealth_p50": None,
            "terminal_wealth_p90": None,
            "max_drawdown_p10": None,
            "max_drawdown_p50": None,
            "ulcer_index_p50": None,
            "severe_drawdown_probability": None,
            "capital_impairment_probability": None,
            "mean_risk_off_share": None,
            "mean_transition_share": None,
        }
    terminal = pd.to_numeric(paths["terminal_wealth"], errors="coerce").dropna()
    drawdown = pd.to_numeric(paths["max_drawdown"], errors="coerce").dropna()
    ulcer = pd.to_numeric(paths["ulcer_index"], errors="coerce").dropna()
    total_contributions = cfg.starting_account_value + cfg.annual_contribution * cfg.horizon_years
    return {
        "paths": int(paths.shape[0]),
        "terminal_wealth_p10": _quantile_or_none(terminal, 0.10),
        "terminal_wealth_p50": _quantile_or_none(terminal, 0.50),
        "terminal_wealth_p90": _quantile_or_none(terminal, 0.90),
        "max_drawdown_p10": _quantile_or_none(drawdown, 0.10),
        "max_drawdown_p50": _quantile_or_none(drawdown, 0.50),
        "ulcer_index_p50": _quantile_or_none(ulcer, 0.50),
        "severe_drawdown_probability": _mean_or_none(drawdown <= cfg.hard_drawdown_limit),
        "capital_impairment_probability": _mean_or_none(terminal < total_contributions),
        "mean_risk_off_share": _column_mean_or_none(paths, "share_risk_off"),
        "mean_transition_share": _column_mean_or_none(paths, "share_transition"),
    }


def regime_mix_frame(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame(
            columns=["regime", "mean_path_share", "p10_path_share", "p90_path_share"]
        )
    rows = []
    for bucket in REGIME_BUCKETS:
        column = f"share_{bucket}"
        if column not in paths:
            continue
        values = pd.to_numeric(paths[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "regime": bucket,
                "mean_path_share": float(values.mean()),
                "p10_path_share": float(values.quantile(0.10)),
                "p90_path_share": float(values.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def simulation_settings_frame(config: ForwardSimulationConfig | None = None) -> pd.DataFrame:
    cfg = config or ForwardSimulationConfig()
    return pd.DataFrame(
        [
            {"setting": "paths", "value": f"{cfg.paths:,}", "meaning": "Forward paths simulated."},
            {
                "setting": "block_days",
                "value": f"{cfg.block_days}",
                "meaning": "Historical return-block length sampled inside each regime.",
            },
            {
                "setting": "annual_contribution",
                "value": f"${cfg.annual_contribution:,.0f}",
                "meaning": "Annual contribution total injected according to contribution timing.",
            },
            {
                "setting": "contribution_timing",
                "value": cfg.contribution_timing,
                "meaning": "Contribution cadence; monthly is split into 12 period-end deposits.",
            },
            {
                "setting": "initial_scenario_weight",
                "value": f"{cfg.initial_scenario_weight:.0%}",
                "meaning": "How strongly today's scenario map biases the starting regime.",
            },
            {
                "setting": "transition_scenario_weight",
                "value": f"{cfg.transition_scenario_weight:.0%}",
                "meaning": "How strongly today's scenario map biases future regime transitions.",
            },
            {
                "setting": "min_regime_observations",
                "value": f"{cfg.min_regime_observations}",
                "meaning": "Minimum historical observations before a regime gets its own return library.",
            },
        ]
    )


def scenario_probability_frame(scenario_outlook: pd.DataFrame | None) -> pd.DataFrame:
    probabilities = scenario_bucket_probabilities(scenario_outlook)
    return (
        probabilities.rename("probability")
        .reset_index()
        .rename(columns={"index": "regime"})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )


def rolling_origin_simulation_backtest(
    daily_returns: pd.Series | np.ndarray,
    *,
    scenario_history: pd.DataFrame | None = None,
    config: ForwardSimulationValidationConfig | None = None,
) -> pd.DataFrame:
    """Backtest simulation calibration through historical rolling origins.

    Each origin trains only on returns available up to that date, simulates the
    configured forward horizons, then compares realized future returns and
    drawdowns with the simulated P10/P50/P90 bands.
    """

    cfg = config or ForwardSimulationValidationConfig()
    returns = _clean_returns_series(daily_returns).sort_index()
    if returns.empty:
        return _empty_validation_frame()

    rows: list[dict[str, object]] = []
    origins = _validation_origin_dates(returns, config=cfg)
    for origin_position, origin_date in enumerate(origins):
        train = _returns_through_origin(returns, origin_date)
        if len(train) < cfg.min_train_days:
            continue
        scenario_outlook = _scenario_history_for_origin(scenario_history, origin_date)
        for horizon_label, horizon_days in cfg.horizons:
            realized = _returns_after_origin(returns, origin_date, horizon_days)
            if len(realized) < horizon_days:
                continue
            simulation_config = _validation_simulation_config(
                cfg,
                horizon_days=horizon_days,
                origin_position=origin_position,
            )
            paths = simulate_regime_conditioned_paths(
                train,
                scenario_outlook=scenario_outlook,
                config=simulation_config,
            )
            if paths.empty:
                continue
            rows.append(
                _simulation_validation_origin_row(
                    origin_date=origin_date,
                    horizon_label=horizon_label,
                    horizon_days=horizon_days,
                    train_days=len(train),
                    realized=realized,
                    paths=paths,
                    config=cfg,
                )
            )
    if not rows:
        return _empty_validation_frame()
    return pd.DataFrame(rows)


def summarize_simulation_validation(validation: pd.DataFrame) -> dict[str, object]:
    """Summarize rolling-origin simulation calibration quality."""

    if validation.empty:
        return {
            "rows": 0,
            "origins": 0,
            "horizons": 0,
            "interval_coverage": None,
            "target_coverage": None,
            "coverage_error": None,
            "median_error_mean": None,
            "median_abs_error": None,
            "too_bullish_share": None,
            "too_bearish_share": None,
            "severe_drawdown_brier": None,
            "realized_severe_drawdown_rate": None,
            "simulated_severe_drawdown_probability_mean": None,
            "launch_decision_accuracy": None,
            "validity_read": "insufficient_history",
        }

    target_coverage = _validation_target_coverage(validation)
    coverage = _mean_or_none(validation["realized_in_interval"])
    median_error = pd.to_numeric(validation["p50_error"], errors="coerce").dropna()
    severe_probability = pd.to_numeric(
        validation["simulated_severe_drawdown_probability"],
        errors="coerce",
    )
    severe_event = pd.to_numeric(validation["realized_severe_drawdown"], errors="coerce")
    launch_match = (
        validation["simulated_launch_decision"]
        .astype(str)
        .eq(validation["realized_launch_decision"].astype(str))
    )
    severe_brier = _mean_or_none((severe_probability - severe_event) ** 2)
    median_error_mean = _mean_or_none(median_error)
    median_abs_error = _mean_or_none(median_error.abs())
    summary = {
        "rows": int(validation.shape[0]),
        "origins": int(validation["origin_date"].nunique()),
        "horizons": int(validation["horizon"].nunique()),
        "interval_coverage": coverage,
        "target_coverage": target_coverage,
        "coverage_error": coverage - target_coverage if coverage is not None else None,
        "median_error_mean": median_error_mean,
        "median_abs_error": median_abs_error,
        "too_bullish_share": _mean_or_none(median_error > 0.0),
        "too_bearish_share": _mean_or_none(median_error < 0.0),
        "severe_drawdown_brier": severe_brier,
        "realized_severe_drawdown_rate": _mean_or_none(severe_event),
        "simulated_severe_drawdown_probability_mean": _mean_or_none(severe_probability),
        "launch_decision_accuracy": _mean_or_none(launch_match),
    }
    summary["validity_read"] = _simulation_validation_read(summary)
    return summary


def rolling_origin_strategy_rank_validation(
    strategy_returns: dict[str, pd.Series | np.ndarray],
    *,
    scenario_history: pd.DataFrame | None = None,
    config: ForwardSimulationValidationConfig | None = None,
) -> pd.DataFrame:
    """Validate whether simulated median rankings predict realized rankings."""

    frames: list[pd.DataFrame] = []
    for strategy_name, returns in strategy_returns.items():
        validation = rolling_origin_simulation_backtest(
            returns,
            scenario_history=scenario_history,
            config=config,
        )
        if validation.empty:
            continue
        validation = validation.copy()
        validation.insert(0, "strategy", strategy_name)
        frames.append(validation)
    if not frames:
        return _empty_rank_validation_frame()

    ranked = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (origin_date, horizon), group in ranked.groupby(["origin_date", "horizon"]):
        if group["strategy"].nunique() < 2:
            continue
        group = group.copy()
        group["simulated_rank"] = group["simulated_p50_return"].rank(
            ascending=False,
            method="min",
        )
        group["realized_rank"] = group["realized_return"].rank(
            ascending=False,
            method="min",
        )
        predicted_top = str(group.sort_values("simulated_rank").iloc[0]["strategy"])
        realized_top = str(group.sort_values("realized_rank").iloc[0]["strategy"])
        rank_correlation = group["simulated_rank"].corr(group["realized_rank"], method="spearman")
        for _, row in group.iterrows():
            rows.append(
                {
                    "origin_date": origin_date,
                    "horizon": horizon,
                    "strategy": row["strategy"],
                    "simulated_p50_return": row["simulated_p50_return"],
                    "realized_return": row["realized_return"],
                    "simulated_rank": int(row["simulated_rank"]),
                    "realized_rank": int(row["realized_rank"]),
                    "rank_error": int(row["simulated_rank"] - row["realized_rank"]),
                    "predicted_top_strategy": predicted_top,
                    "realized_top_strategy": realized_top,
                    "top_strategy_hit": predicted_top == realized_top,
                    "rank_correlation": rank_correlation,
                }
            )
    if not rows:
        return _empty_rank_validation_frame()
    return pd.DataFrame(rows)


def summarize_strategy_rank_validation(rank_validation: pd.DataFrame) -> dict[str, object]:
    if rank_validation.empty:
        return {
            "rows": 0,
            "origin_horizons": 0,
            "top_strategy_hit_rate": None,
            "mean_rank_correlation": None,
            "mean_abs_rank_error": None,
            "ranking_read": "insufficient_history",
        }
    origin_horizon = rank_validation[["origin_date", "horizon"]].drop_duplicates()
    correlations = rank_validation[["origin_date", "horizon", "rank_correlation"]].drop_duplicates()
    hit_rows = rank_validation[["origin_date", "horizon", "top_strategy_hit"]].drop_duplicates()
    mean_abs_rank_error = _mean_or_none(
        pd.to_numeric(rank_validation["rank_error"], errors="coerce").abs()
    )
    top_hit = _mean_or_none(hit_rows["top_strategy_hit"])
    rank_corr = _mean_or_none(pd.to_numeric(correlations["rank_correlation"], errors="coerce"))
    summary = {
        "rows": int(rank_validation.shape[0]),
        "origin_horizons": int(origin_horizon.shape[0]),
        "top_strategy_hit_rate": top_hit,
        "mean_rank_correlation": rank_corr,
        "mean_abs_rank_error": mean_abs_rank_error,
    }
    summary["ranking_read"] = _strategy_ranking_read(summary)
    return summary


def _clean_returns_series(daily_returns: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(daily_returns, pd.Series):
        values = daily_returns.copy()
    else:
        values = pd.Series(np.asarray(daily_returns, dtype=float).reshape(-1))
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values.clip(lower=-0.95, upper=1.0)
    return values.astype(float)


def _validation_origin_dates(
    returns: pd.Series,
    *,
    config: ForwardSimulationValidationConfig,
) -> list[object]:
    if returns.empty:
        return []
    latest_origin_position = len(returns) - 2
    candidate_index = returns.index[config.min_train_days : max(latest_origin_position + 1, 0)]
    if len(candidate_index) == 0:
        return []

    if isinstance(returns.index, pd.DatetimeIndex):
        candidates = pd.Series(candidate_index, index=pd.DatetimeIndex(candidate_index))
        rule = _origin_resample_rule(config.origin_frequency)
        origins = candidates.resample(rule).last().dropna().tolist()
        return [origin for origin in origins if origin in returns.index]

    step = 21 if config.origin_frequency == "monthly" else 63
    if config.origin_frequency not in {"monthly", "quarterly"}:
        step = max(1, int(config.origin_frequency))
    return candidate_index[::step].tolist()


def _origin_resample_rule(origin_frequency: str) -> str:
    frequency = origin_frequency.strip().lower()
    if frequency in {"monthly", "month", "m"}:
        return "ME"
    if frequency in {"quarterly", "quarter", "q"}:
        return "QE"
    return origin_frequency


def _returns_through_origin(returns: pd.Series, origin_date: object) -> pd.Series:
    position = _index_position(returns.index, origin_date)
    if position is None:
        return pd.Series(dtype=float)
    return returns.iloc[: position + 1].copy()


def _returns_after_origin(
    returns: pd.Series,
    origin_date: object,
    horizon_days: int,
) -> pd.Series:
    position = _index_position(returns.index, origin_date)
    if position is None:
        return pd.Series(dtype=float)
    return returns.iloc[position + 1 : position + 1 + horizon_days].copy()


def _index_position(index: pd.Index, value: object) -> int | None:
    try:
        location = index.get_loc(value)
    except KeyError:
        return None
    if isinstance(location, slice):
        return int(location.stop - 1)
    if isinstance(location, np.ndarray):
        matches = np.flatnonzero(location)
        return int(matches[-1]) if matches.size else None
    return int(location)


def _scenario_history_for_origin(
    scenario_history: pd.DataFrame | None,
    origin_date: object,
) -> pd.DataFrame | None:
    if scenario_history is None or scenario_history.empty:
        return None
    frame = scenario_history.copy()
    date_column = next(
        (
            column
            for column in ("origin_date", "as_of_date", "date", "created_at_utc", "created_at")
            if column in frame
        ),
        None,
    )
    if date_column is None:
        return None

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    origin_timestamp = pd.to_datetime(origin_date, errors="coerce")
    if pd.isna(origin_timestamp):
        return None
    eligible = frame[frame[date_column].le(origin_timestamp)].copy()
    if eligible.empty:
        return None
    latest_date = eligible[date_column].max()
    return eligible[eligible[date_column].eq(latest_date)].copy()


def _validation_simulation_config(
    validation_config: ForwardSimulationValidationConfig,
    *,
    horizon_days: int,
    origin_position: int,
) -> ForwardSimulationConfig:
    horizon_years = horizon_days / max(float(validation_config.trading_days_per_year), 1.0)
    return ForwardSimulationConfig(
        horizon_years=horizon_years,
        starting_account_value=1.0,
        annual_contribution=0.0,
        contribution_timing=DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
        trading_days_per_year=validation_config.trading_days_per_year,
        paths=validation_config.paths,
        block_days=validation_config.block_days,
        random_seed=validation_config.random_seed + origin_position * 1009 + horizon_days,
        initial_scenario_weight=validation_config.initial_scenario_weight,
        transition_scenario_weight=validation_config.transition_scenario_weight,
        min_regime_observations=validation_config.min_regime_observations,
        hard_drawdown_limit=validation_config.severe_drawdown_limit,
    )


def _simulation_validation_origin_row(
    *,
    origin_date: object,
    horizon_label: str,
    horizon_days: int,
    train_days: int,
    realized: pd.Series,
    paths: pd.DataFrame,
    config: ForwardSimulationValidationConfig,
) -> dict[str, object]:
    terminal_returns = _path_terminal_returns(paths)
    simulated_p10 = _quantile_or_none(terminal_returns, config.interval_low)
    simulated_p50 = _quantile_or_none(terminal_returns, 0.50)
    simulated_p90 = _quantile_or_none(terminal_returns, config.interval_high)
    drawdowns = _numeric_column(paths, "max_drawdown")
    severe_probability = _mean_or_none(drawdowns <= config.severe_drawdown_limit)
    realized_return = _realized_terminal_return(realized)
    realized_drawdown = _realized_max_drawdown(realized)
    realized_severe = bool(realized_drawdown <= config.severe_drawdown_limit)
    return {
        "origin_date": origin_date,
        "horizon": horizon_label,
        "horizon_days": int(horizon_days),
        "train_days": int(train_days),
        "paths": int(paths.shape[0]),
        "realized_return": realized_return,
        "realized_max_drawdown": realized_drawdown,
        "realized_severe_drawdown": realized_severe,
        "simulated_p10_return": simulated_p10,
        "simulated_p50_return": simulated_p50,
        "simulated_p90_return": simulated_p90,
        "target_interval_coverage": config.interval_high - config.interval_low,
        "realized_in_interval": (
            simulated_p10 is not None
            and simulated_p90 is not None
            and simulated_p10 <= realized_return <= simulated_p90
        ),
        "p50_error": simulated_p50 - realized_return if simulated_p50 is not None else None,
        "p50_abs_error": (
            abs(simulated_p50 - realized_return) if simulated_p50 is not None else None
        ),
        "simulated_severe_drawdown_probability": severe_probability,
        "severe_drawdown_probability_error": (
            severe_probability - float(realized_severe) if severe_probability is not None else None
        ),
        "simulated_launch_decision": _simulated_launch_decision(
            simulated_p10=simulated_p10,
            simulated_p50=simulated_p50,
            severe_probability=severe_probability,
            config=config,
        ),
        "realized_launch_decision": _realized_launch_decision(
            realized_return=realized_return,
            realized_drawdown=realized_drawdown,
            config=config,
        ),
    }


def _path_terminal_returns(paths: pd.DataFrame) -> pd.Series:
    terminal = _numeric_column(paths, "terminal_wealth")
    return terminal - 1.0


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)


def _realized_terminal_return(returns: pd.Series) -> float:
    clean = _clean_returns_series(returns)
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _realized_max_drawdown(returns: pd.Series) -> float:
    clean = _clean_returns_series(returns)
    if clean.empty:
        return 0.0
    equity = pd.concat(
        [
            pd.Series([1.0], index=[clean.index[0]]),
            (1.0 + clean).cumprod(),
        ]
    )
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _simulated_launch_decision(
    *,
    simulated_p10: float | None,
    simulated_p50: float | None,
    severe_probability: float | None,
    config: ForwardSimulationValidationConfig,
) -> str:
    severe = severe_probability if severe_probability is not None else 0.0
    median = simulated_p50 if simulated_p50 is not None else 0.0
    lower = simulated_p10 if simulated_p10 is not None else 0.0
    if median <= 0.0 or severe >= config.wait_severe_probability:
        return "wait"
    if lower < 0.0 or severe >= config.ramp_severe_probability:
        return "ramp_in"
    return "full_launch"


def _realized_launch_decision(
    *,
    realized_return: float,
    realized_drawdown: float,
    config: ForwardSimulationValidationConfig,
) -> str:
    if realized_return <= 0.0 or realized_drawdown <= config.severe_drawdown_limit:
        return "wait"
    if realized_drawdown <= config.ramp_drawdown_limit:
        return "ramp_in"
    return "full_launch"


def _validation_target_coverage(validation: pd.DataFrame) -> float:
    if "target_interval_coverage" not in validation:
        return 0.80
    values = pd.to_numeric(validation["target_interval_coverage"], errors="coerce").dropna()
    if values.empty:
        return 0.80
    return float(values.mean())


def _simulation_validation_read(summary: dict[str, object]) -> str:
    rows = int(summary.get("rows") or 0)
    if rows < 10:
        return "limited_sample"
    coverage_error = _safe_float(summary.get("coverage_error"))
    median_error = _safe_float(summary.get("median_error_mean"))
    severe_brier = _safe_float(summary.get("severe_drawdown_brier"))
    if coverage_error is not None and coverage_error <= -0.15:
        return "interval_too_narrow"
    if coverage_error is not None and coverage_error >= 0.15:
        return "interval_too_wide"
    if median_error is not None and median_error >= 0.03:
        return "too_bullish"
    if median_error is not None and median_error <= -0.03:
        return "too_bearish"
    if severe_brier is not None and severe_brier >= 0.20:
        return "drawdown_miscalibrated"
    return "calibrated_enough_for_research"


def _strategy_ranking_read(summary: dict[str, object]) -> str:
    origin_horizons = int(summary.get("origin_horizons") or 0)
    if origin_horizons < 5:
        return "limited_sample"
    hit_rate = _safe_float(summary.get("top_strategy_hit_rate"))
    rank_correlation = _safe_float(summary.get("mean_rank_correlation"))
    if (
        hit_rate is not None
        and hit_rate >= 0.65
        and (rank_correlation is None or rank_correlation >= 0.35)
    ):
        return "ranking_signal_useful"
    if hit_rate is not None and hit_rate <= 0.40:
        return "ranking_signal_weak"
    if rank_correlation is not None and rank_correlation < 0.0:
        return "ranking_signal_inverted"
    return "ranking_signal_mixed"


def _empty_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "origin_date",
            "horizon",
            "horizon_days",
            "train_days",
            "paths",
            "realized_return",
            "realized_max_drawdown",
            "realized_severe_drawdown",
            "simulated_p10_return",
            "simulated_p50_return",
            "simulated_p90_return",
            "target_interval_coverage",
            "realized_in_interval",
            "p50_error",
            "p50_abs_error",
            "simulated_severe_drawdown_probability",
            "severe_drawdown_probability_error",
            "simulated_launch_decision",
            "realized_launch_decision",
        ]
    )


def _empty_rank_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "origin_date",
            "horizon",
            "strategy",
            "simulated_p50_return",
            "realized_return",
            "simulated_rank",
            "realized_rank",
            "rank_error",
            "predicted_top_strategy",
            "realized_top_strategy",
            "top_strategy_hit",
            "rank_correlation",
        ]
    )


def _normalize_bucket(value: object) -> str:
    bucket = str(value or "").strip().lower()
    if "risk_off" in bucket:
        return "risk_off"
    if bucket == "risk_on_fragile" or "fragile" in bucket:
        return "risk_on_fragile"
    if "transition" in bucket or "choppy" in bucket:
        return "transition"
    if bucket == "risk_on" or "risk_on" in bucket:
        return "risk_on"
    return bucket


def _normalize_probability_vector(values: pd.Series) -> pd.Series:
    vector = pd.to_numeric(values.reindex(REGIME_BUCKETS), errors="coerce").fillna(0.0)
    vector = vector.clip(lower=0.0)
    total = float(vector.sum())
    if total <= 0.0:
        vector = pd.Series(DEFAULT_FORWARD_SIMULATION_FALLBACK_PROBABILITIES, dtype=float)
        vector = vector.reindex(REGIME_BUCKETS).fillna(0.0)
        total = float(vector.sum())
    return vector / max(total, 1e-12)


def _rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window, min_periods=max(5, min(window, 21))).apply(
        np.prod,
        raw=True,
    ) - 1.0


def _quantile(values: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(quantile))


def _repair_sparse_regime_labels(library: pd.DataFrame, *, min_observations: int) -> pd.DataFrame:
    if library.empty:
        return library
    output = library.copy()
    counts = output["regime"].value_counts()
    sparse_regimes = [
        regime for regime in REGIME_BUCKETS if counts.get(regime, 0) < min_observations
    ]
    if sparse_regimes:
        output.loc[output["regime"].isin(sparse_regimes), "regime"] = "transition"
    return output


def _historical_regime_probabilities(library: pd.DataFrame) -> pd.Series:
    probabilities = library["regime"].value_counts(normalize=True)
    return _normalize_probability_vector(probabilities.reindex(REGIME_BUCKETS).fillna(0.0))


def _blend_probabilities(
    historical: pd.Series,
    scenario: pd.Series,
    *,
    scenario_weight: float,
) -> pd.Series:
    weight = float(np.clip(scenario_weight, 0.0, 1.0))
    return _normalize_probability_vector((1.0 - weight) * historical + weight * scenario)


def _empirical_transition_matrix(library: pd.DataFrame, *, block_days: int) -> pd.DataFrame:
    labels = library["regime"].astype(str).reset_index(drop=True)
    block = max(1, int(block_days))
    block_labels = labels.iloc[::block].reset_index(drop=True)
    matrix = pd.DataFrame(1.0, index=REGIME_BUCKETS, columns=REGIME_BUCKETS)
    for current, following in zip(block_labels.iloc[:-1], block_labels.iloc[1:], strict=False):
        if current in matrix.index and following in matrix.columns:
            matrix.loc[current, following] += 1.0
    return matrix.div(matrix.sum(axis=1), axis=0)


def _blend_transition_matrix(
    empirical: pd.DataFrame,
    scenario: pd.Series,
    *,
    scenario_weight: float,
) -> pd.DataFrame:
    weight = float(np.clip(scenario_weight, 0.0, 1.0))
    scenario_row = _normalize_probability_vector(scenario)
    matrix = empirical.reindex(index=REGIME_BUCKETS, columns=REGIME_BUCKETS).fillna(0.0)
    for regime in REGIME_BUCKETS:
        matrix.loc[regime] = _normalize_probability_vector(
            (1.0 - weight) * matrix.loc[regime] + weight * scenario_row
        )
    return matrix


def _build_block_library(library: pd.DataFrame, *, block_days: int) -> dict[str, list[np.ndarray]]:
    block = max(1, int(block_days))
    returns = library["return"].to_numpy(dtype=float)
    labels = library["regime"].astype(str).to_numpy()
    output: dict[str, list[np.ndarray]] = {bucket: [] for bucket in REGIME_BUCKETS}
    if returns.size == 0:
        return output
    max_start = max(returns.size - block, 0)
    for start in range(max_start + 1):
        regime = labels[start]
        if regime in output:
            output[regime].append(returns[start : start + block])
    fallback_blocks = [returns[start : start + block] for start in range(max_start + 1)]
    for regime in REGIME_BUCKETS:
        if not output[regime]:
            output[regime] = fallback_blocks
    return output


def _sample_bucket(rng: np.random.Generator, probabilities: pd.Series) -> str:
    vector = _normalize_probability_vector(probabilities)
    return str(rng.choice(vector.index.to_numpy(), p=vector.to_numpy(dtype=float)))


def _sample_block(
    rng: np.random.Generator,
    block_library: dict[str, list[np.ndarray]],
    *,
    regime: str,
) -> np.ndarray:
    blocks = block_library.get(regime) or []
    if not blocks:
        return np.array([], dtype=float)
    return blocks[int(rng.integers(0, len(blocks)))]


def _empty_paths_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "path",
            "terminal_wealth",
            "max_drawdown",
            "ulcer_index",
            "share_risk_off",
            "share_transition",
            "share_risk_on_fragile",
            "share_risk_on",
        ]
    )


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _quantile_or_none(values: pd.Series, quantile: float) -> float | None:
    if values.empty:
        return None
    return float(values.quantile(quantile))


def _mean_or_none(values: pd.Series | np.ndarray) -> float | None:
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return float(series.mean())


def _column_mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())
