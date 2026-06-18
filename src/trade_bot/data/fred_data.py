from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from warnings import warn

import pandas as pd
import requests
import yaml

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    name: str
    category: str
    risk_polarity: str


def load_fred_catalog(path: str | Path | None) -> tuple[FredSeries, ...]:
    if path is None:
        return ()
    catalog_path = Path(path)
    if not catalog_path.exists():
        return ()

    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return tuple(_series_from_mapping(item) for item in raw.get("series", []))


def load_or_fetch_fred_data(
    catalog: tuple[FredSeries, ...],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    series_ids = sorted({series.series_id for series in catalog})
    cache_path = Path(cache_dir) / "fred_series.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: pd.DataFrame | None = None

    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        if set(series_ids).issubset(cached.columns):
            return _filter_data(cached[series_ids], start=start, end=end)

    if cached is not None and not refresh:
        missing = sorted(set(series_ids) - set(cached.columns))
        fetched = fetch_fred_data(missing, start=start, end=end)
        data = cached.combine_first(fetched).reindex(
            columns=sorted(set(cached.columns) | set(fetched.columns))
        )
        data.to_parquet(cache_path)
        return _filter_data(data.reindex(columns=series_ids), start=start, end=end)

    data = fetch_fred_data(series_ids, start=start, end=end)
    data.to_parquet(cache_path)
    return _filter_data(data.reindex(columns=series_ids), start=start, end=end)


def fetch_fred_data(series_ids: list[str], start: str, end: str | None) -> pd.DataFrame:
    frames = []
    for series_id in series_ids:
        series = fetch_fred_series(series_id, start=start, end=end)
        if not series.empty:
            frames.append(series)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().dropna(how="all")


def fetch_fred_series(series_id: str, start: str, end: str | None) -> pd.Series:
    response = requests.get(
        FRED_GRAPH_URL,
        params={"id": series_id},
        timeout=30,
    )
    if response.status_code >= 400:
        warn(f"FRED request failed for {series_id}: HTTP {response.status_code}", stacklevel=2)
        return pd.Series(dtype=float, name=series_id)

    try:
        series = parse_fred_csv(response.text, series_id)
    except (KeyError, ValueError) as error:
        warn(f"FRED response could not be parsed for {series_id}: {error}", stacklevel=2)
        return pd.Series(dtype=float, name=series_id)

    filtered = series.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    return filtered.rename(series_id)


def parse_fred_csv(csv_text: str, series_id: str) -> pd.Series:
    frame = pd.read_csv(StringIO(csv_text))
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        msg = f"Expected observation_date and {series_id} columns."
        raise KeyError(msg)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    values = pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["observation_date"], name=series_id)
    return series.dropna().sort_index()


def _series_from_mapping(raw: dict[str, Any]) -> FredSeries:
    return FredSeries(
        series_id=str(raw["id"]),
        name=str(raw["name"]),
        category=str(raw["category"]),
        risk_polarity=str(raw.get("risk_polarity", "neutral")),
    )


def _filter_data(data: pd.DataFrame, *, start: str, end: str | None) -> pd.DataFrame:
    filtered = data.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    return filtered.dropna(how="all")
