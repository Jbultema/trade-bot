from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.config import load_config
from trade_bot.DEFAULT import (
    DEFAULT_EXPERIMENT_CACHE_TTL_SECONDS,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS,
)
from trade_bot.research.baselines import BaselineRun, run_configured_baselines
from trade_bot.research.experiment_monitor import (
    load_experiment_candidates,
    load_experiment_regime_metrics,
    load_experiment_scorecards,
    load_experiment_walk_forward,
)
from trade_bot.storage.run_store import RunStore, SnapshotManifest, build_snapshot_fingerprint


@st.cache_data(show_spinner=False, ttl=DEFAULT_EXPERIMENT_CACHE_TTL_SECONDS)
def load_experiment_dashboard_frames(
    root: str | Path = DEFAULT_EXPERIMENTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_experiment_scorecards(root),
        load_experiment_regime_metrics(root),
        load_experiment_walk_forward(root),
        load_experiment_candidates(root),
    )


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_snapshot_dashboard_run(
    config_path_string: str,
    events_path_string: str,
    macro_path_string: str,
    news_path_string: str,
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
) -> tuple[BaselineRun, SnapshotManifest] | None:
    run_store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    fingerprint = build_snapshot_fingerprint(
        config_path_string,
        events_path_string,
        macro_path_string,
        news_path_string,
    )
    return run_store.load_latest_snapshot(
        fingerprint=fingerprint,
        require_matching_config=True,
    )


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_previous_snapshot_dashboard_run(
    config_path_string: str,
    events_path_string: str,
    macro_path_string: str,
    news_path_string: str,
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
    current_run_id: str | None = None,
) -> tuple[BaselineRun, SnapshotManifest] | None:
    run_store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    fingerprint = build_snapshot_fingerprint(
        config_path_string,
        events_path_string,
        macro_path_string,
        news_path_string,
    )
    snapshots = run_store.list_snapshots(limit=50)
    if snapshots.empty or "combined_config_hash" not in snapshots:
        return None
    snapshots = snapshots[snapshots["combined_config_hash"] == fingerprint.combined_hash]
    if current_run_id is not None and "run_id" in snapshots:
        snapshots = snapshots[snapshots["run_id"] != current_run_id]
    if snapshots.empty:
        return None
    return run_store.load_snapshot(str(snapshots.iloc[0]["run_id"]))


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def load_snapshot_jobs_frame(
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
) -> pd.DataFrame:
    run_store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    return run_store.list_jobs(limit=8)


@st.cache_data(show_spinner="Running backtests...")
def load_live_run(
    config_path_string: str,
    events_path_string: str,
    macro_path_string: str,
    news_path_string: str,
    refresh: bool,
    refresh_macro_data: bool,
    refresh_news_data: bool,
) -> BaselineRun:
    config = load_config(config_path_string)
    return run_configured_baselines(
        config,
        refresh_data=refresh,
        refresh_macro=refresh_macro_data,
        refresh_news=refresh_news_data,
        event_config_path=events_path_string,
        macro_config_path=macro_path_string,
        news_config_path=news_path_string,
    )
