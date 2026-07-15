from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.dashboard.loaders import (
    load_experiment_dashboard_frames,
    load_experiment_scorecards_frame,
)
from trade_bot.DEFAULTS import DEFAULT_EXPERIMENTS_DIR


@st.cache_data(show_spinner=False, ttl=60)
def scorecards(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    return load_experiment_scorecards_frame(root)


@st.cache_data(show_spinner=False, ttl=60)
def dashboard_frames(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_experiment_dashboard_frames(root)


def top_scorecards(frame: pd.DataFrame, *, limit: int = 25) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in [
            "growth_constrained_utility_score",
            "selection_adjusted_promotion_score",
            "promotion_score",
            "cagr",
        ]
        if column in frame
    ]
    if not sort_columns:
        return frame.head(limit)
    return frame.sort_values(sort_columns, ascending=False).head(limit).reset_index(drop=True)

