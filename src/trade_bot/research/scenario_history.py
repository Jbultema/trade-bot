from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.research.current_state import (
    _risk_score,
    _risk_status,
    build_confirmation_matrix,
    build_market_health,
    momentum_state_table,
)
from trade_bot.research.future_scenarios import build_scenario_lattice
from trade_bot.storage.run_store import RunStore

SCENARIO_HISTORY_COLUMNS = [
    "snapshot_time",
    "market_date",
    "horizon",
    "scenario",
    "risk_bucket",
    "probability",
    "rank",
    "run_id",
]


def scenario_history_from_snapshots(
    store: RunStore,
    *,
    limit: int = 250,
) -> pd.DataFrame:
    """Build date-stamped scenario probabilities from saved run snapshots."""

    snapshots = store.list_snapshots(limit=limit)
    if snapshots.empty or "run_id" not in snapshots:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    frames: list[pd.DataFrame] = []
    for _, row in snapshots.iloc[::-1].iterrows():
        run_id = str(row["run_id"])
        try:
            run, manifest = store.load_snapshot(run_id)
        except (FileNotFoundError, TypeError, OSError, AttributeError, ValueError):
            continue
        frame = scenario_history_from_lattice(
            getattr(run.current_state, "scenario_lattice", pd.DataFrame()),
            market_date=manifest.market_date,
            created_at_utc=manifest.created_at_utc,
            run_id=manifest.run_id,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    return clean_scenario_history(pd.concat(frames, ignore_index=True))


def reconstruct_scenario_history_from_prices(
    prices: pd.DataFrame,
    *,
    origin_frequency: str = "quarterly",
    min_train_days: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Rebuild scenario probabilities from historical price state at each origin.

    This is point-in-time safe for the price-derived scenario engine because each
    origin is computed from data truncated through that date.
    """

    clean_prices = prices.dropna(how="all").sort_index()
    if clean_prices.empty:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    origins = _reconstruction_origin_dates(
        clean_prices.index,
        origin_frequency=origin_frequency,
        min_train_days=min_train_days,
        start_date=start_date,
        end_date=end_date,
    )
    frames: list[pd.DataFrame] = []
    for origin in origins:
        as_of_prices = clean_prices.loc[:origin].dropna(how="all")
        if len(as_of_prices) < min_train_days:
            continue
        momentum_state = momentum_state_table(as_of_prices)
        confirmation_matrix = build_confirmation_matrix(as_of_prices, momentum_state)
        market_health = build_market_health(as_of_prices, momentum_state)
        risk_score = _risk_score(confirmation_matrix, market_health)
        risk_status = _risk_status(risk_score)
        scenario_lattice, _scenario_drivers = build_scenario_lattice(
            confirmation_matrix,
            market_health,
            momentum_state,
            risk_score,
            risk_status,
        )
        timestamp = pd.to_datetime(origin, errors="coerce")
        if pd.isna(timestamp):
            continue
        frame = scenario_history_from_lattice(
            scenario_lattice,
            market_date=str(timestamp.date()),
            created_at_utc=(
                timestamp.tz_localize("UTC").isoformat()
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC").isoformat()
            ),
            run_id=f"reconstructed_{timestamp.strftime('%Y%m%d')}",
        )
        if not frame.empty:
            frame["source"] = "reconstructed_price_state"
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*SCENARIO_HISTORY_COLUMNS, "source"])
    history = clean_scenario_history(pd.concat(frames, ignore_index=True))
    history["source"] = "reconstructed_price_state"
    return history


def scenario_history_from_lattice(
    scenario_lattice: pd.DataFrame,
    *,
    market_date: str,
    created_at_utc: str,
    run_id: str,
) -> pd.DataFrame:
    """Convert one snapshot scenario lattice to validation-ready history rows."""

    if scenario_lattice.empty or not {"horizon", "risk_bucket", "probability"}.issubset(
        scenario_lattice.columns
    ):
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    frame = scenario_lattice.copy()
    if "scenario" not in frame:
        frame["scenario"] = frame["risk_bucket"].astype(str)
    if "rank" not in frame:
        frame["rank"] = (
            frame.groupby("horizon")["probability"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    frame["snapshot_time"] = pd.to_datetime(created_at_utc, errors="coerce", utc=True)
    if frame["snapshot_time"].isna().all():
        frame["snapshot_time"] = pd.to_datetime(market_date, errors="coerce", utc=True)
    market_timestamp = pd.to_datetime(market_date, errors="coerce")
    frame["market_date"] = market_timestamp.date() if pd.notna(market_timestamp) else pd.NaT
    frame["run_id"] = str(run_id)
    frame = frame.dropna(subset=["probability", "snapshot_time"])
    if frame.empty:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    return frame[SCENARIO_HISTORY_COLUMNS].copy()


def _reconstruction_origin_dates(
    index: pd.Index,
    *,
    origin_frequency: str,
    min_train_days: int,
    start_date: str | None,
    end_date: str | None,
) -> list[object]:
    if len(index) <= min_train_days:
        return []
    candidate_index = index[min_train_days:]
    if start_date is not None:
        start_timestamp = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_timestamp):
            candidate_index = candidate_index[candidate_index >= start_timestamp]
    if end_date is not None:
        end_timestamp = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_timestamp):
            candidate_index = candidate_index[candidate_index <= end_timestamp]
    if len(candidate_index) == 0:
        return []
    frequency = origin_frequency.strip().lower()
    if isinstance(candidate_index, pd.DatetimeIndex):
        candidates = pd.Series(candidate_index, index=pd.DatetimeIndex(candidate_index))
        if frequency in {"monthly", "month", "m"}:
            return candidates.resample("ME").last().dropna().tolist()
        if frequency in {"quarterly", "quarter", "q"}:
            return candidates.resample("QE").last().dropna().tolist()
    step = 21 if frequency == "monthly" else 63
    if frequency not in {"monthly", "month", "m", "quarterly", "quarter", "q"}:
        try:
            step = max(1, int(origin_frequency))
        except ValueError:
            step = 63
    return candidate_index[::step].tolist()


def clean_scenario_history(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize scenario history loaded from snapshots, CSV, or parquet."""

    if history.empty:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    frame = history.copy()
    if "snapshot_time" not in frame:
        for date_column in ("origin_date", "as_of_date", "date", "created_at_utc", "created_at"):
            if date_column in frame:
                frame["snapshot_time"] = frame[date_column]
                break
    if "market_date" not in frame:
        for date_column in ("origin_date", "as_of_date", "date", "snapshot_time"):
            if date_column in frame:
                frame["market_date"] = frame[date_column]
                break
    for column in SCENARIO_HISTORY_COLUMNS:
        if column not in frame:
            frame[column] = "" if column not in {"probability", "rank"} else None
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce", utc=True)
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce").dt.date
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_time", "market_date", "probability"])
    if frame.empty:
        return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)
    return (
        frame[SCENARIO_HISTORY_COLUMNS]
        .sort_values(
            ["snapshot_time", "horizon", "rank"],
        )
        .reset_index(drop=True)
    )


def write_scenario_history(history: pd.DataFrame, path: str | Path) -> Path:
    """Write scenario history as CSV or parquet based on file suffix."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        history.to_parquet(output_path, index=False)
    else:
        history.to_csv(output_path, index=False)
    return output_path
