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
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
    DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR,
)

REGIME_BUCKETS = ("risk_off", "transition", "risk_on_fragile", "risk_on")


@dataclass(frozen=True)
class ForwardSimulationConfig:
    """Configuration for regime-conditioned forward path simulation.

    The engine is intentionally empirical: it samples realized historical return
    blocks from regime-labeled strategy history, then uses current scenario
    probabilities to bias the starting state and future transition probabilities.
    """

    horizon_years: int = DEFAULT_OUTCOME_HORIZON_YEARS
    starting_account_value: float = DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE
    annual_contribution: float = DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION
    trading_days_per_year: int = DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR
    paths: int = DEFAULT_FORWARD_SIMULATION_PATHS
    block_days: int = DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS
    random_seed: int = DEFAULT_FORWARD_SIMULATION_RANDOM_SEED
    initial_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_INITIAL_SCENARIO_WEIGHT
    transition_scenario_weight: float = DEFAULT_FORWARD_SIMULATION_TRANSITION_SCENARIO_WEIGHT
    min_regime_observations: int = DEFAULT_FORWARD_SIMULATION_MIN_REGIME_OBSERVATIONS
    hard_drawdown_limit: float = DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT


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
                if day_idx % cfg.trading_days_per_year == 0:
                    wealth += float(cfg.annual_contribution)
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
        return pd.DataFrame(columns=["regime", "mean_path_share", "p10_path_share", "p90_path_share"])
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


def _clean_returns_series(daily_returns: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(daily_returns, pd.Series):
        values = daily_returns.copy()
    else:
        values = pd.Series(np.asarray(daily_returns, dtype=float).reshape(-1))
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values.clip(lower=-0.95, upper=1.0)
    return values.astype(float)


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
    sparse_regimes = [regime for regime in REGIME_BUCKETS if counts.get(regime, 0) < min_observations]
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
