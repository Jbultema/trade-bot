from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, ttl=60)
def read_csv_artifact(path: str | Path) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return pd.DataFrame()
    return pd.read_csv(artifact_path)


def pbo_frames(report_dir: str | Path = "reports/pbo_diagnostics") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": read_csv_artifact(root / "pbo_summary.csv"),
        "selection": read_csv_artifact(root / "pbo_strategy_selection.csv"),
        "stats": read_csv_artifact(root / "pbo_strategy_stats.csv"),
    }


def leadership_frames(report_dir: str | Path = "reports/leadership_diagnostics") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": read_csv_artifact(root / "leadership_summary.csv"),
        "impairment": read_csv_artifact(root / "leadership_impairment.csv"),
        "router": read_csv_artifact(root / "walk_forward_router_comparison.csv"),
    }

