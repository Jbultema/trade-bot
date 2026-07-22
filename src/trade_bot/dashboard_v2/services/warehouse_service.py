from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from trade_bot.DEFAULTS import DEFAULT_RUN_STORE_DB_PATH
from trade_bot.storage.warehouse import TradingWarehouse


def _read_from_warehouse[T](
    warehouse_path: str | Path,
    reader: Callable[[TradingWarehouse], T],
    *,
    empty: T,
) -> T:
    path = Path(warehouse_path)
    if not path.is_file():
        return empty
    try:
        return reader(TradingWarehouse(path, read_only=True))
    except (OSError, duckdb.Error):
        return empty


@st.cache_data(show_spinner=False, ttl=60)
def read_warehouse_table(
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    table_name: str = "",
    limit: int | None = None,
) -> pd.DataFrame:
    if not table_name:
        return pd.DataFrame()
    return _read_from_warehouse(
        warehouse_path,
        lambda warehouse: warehouse.read_table(table_name, limit=limit),
        empty=pd.DataFrame(),
    )


@st.cache_data(show_spinner=False, ttl=60)
def champion_challenger_frame(
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
) -> pd.DataFrame:
    return _read_from_warehouse(
        warehouse_path,
        lambda warehouse: warehouse.champion_challenger_frame(),
        empty=pd.DataFrame(),
    )


@st.cache_data(show_spinner=False, ttl=60)
def monitoring_windows(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> pd.DataFrame:
    return _read_from_warehouse(
        warehouse_path,
        lambda warehouse: warehouse.list_monitoring_windows(status=None),
        empty=pd.DataFrame(),
    )


@st.cache_data(show_spinner=False, ttl=60)
def warehouse_counts(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> pd.DataFrame:
    return _read_from_warehouse(
        warehouse_path,
        lambda warehouse: warehouse.table_counts(),
        empty=pd.DataFrame(),
    )


@st.cache_data(show_spinner=False, ttl=60)
def simulation_validation_summary(
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_from_warehouse(
        warehouse_path,
        lambda warehouse: (
            warehouse.simulation_validation_runs(limit=25),
            warehouse.simulation_validation_metrics(limit=2_500),
        ),
        empty=(pd.DataFrame(), pd.DataFrame()),
    )
