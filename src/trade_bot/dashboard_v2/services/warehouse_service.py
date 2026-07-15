from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.DEFAULTS import DEFAULT_RUN_STORE_DB_PATH
from trade_bot.storage.warehouse import TradingWarehouse


@st.cache_data(show_spinner=False, ttl=60)
def read_warehouse_table(
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    table_name: str = "",
    limit: int | None = None,
) -> pd.DataFrame:
    if not table_name:
        return pd.DataFrame()
    return TradingWarehouse(warehouse_path).read_table(table_name, limit=limit)


@st.cache_data(show_spinner=False, ttl=60)
def champion_challenger_frame(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> pd.DataFrame:
    return TradingWarehouse(warehouse_path).champion_challenger_frame()


@st.cache_data(show_spinner=False, ttl=60)
def monitoring_windows(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> pd.DataFrame:
    return TradingWarehouse(warehouse_path).list_monitoring_windows(status=None)


@st.cache_data(show_spinner=False, ttl=60)
def warehouse_counts(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> pd.DataFrame:
    return TradingWarehouse(warehouse_path).table_counts()


@st.cache_data(show_spinner=False, ttl=60)
def simulation_validation_summary(warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    warehouse = TradingWarehouse(warehouse_path)
    return warehouse.simulation_validation_runs(limit=25), warehouse.simulation_validation_metrics(limit=2_500)

