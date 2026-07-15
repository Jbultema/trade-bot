from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from trade_bot.dashboard_v2.services.runtime import DashboardPaths
from trade_bot.DEFAULTS import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_FORWARD_TEST_ACCOUNT,
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_REPORT_PATH,
)
from trade_bot.storage.run_store import RunStore


@dataclass(frozen=True)
class QueuedJob:
    job_id: str
    label: str


def _store(paths: DashboardPaths) -> RunStore:
    return RunStore(paths.run_store_path, artifact_dir=paths.artifact_dir, job_log_dir=paths.job_log_dir)


def queue_daily_update(paths: DashboardPaths) -> QueuedJob:
    job = _store(paths).start_daily_update_job(
        config_path=paths.config_path,
        events_path=paths.events_path,
        macro_path=paths.macro_path,
        news_path=paths.news_path,
        report_path=DEFAULT_REPORT_PATH,
        experiment_dir=DEFAULT_EXPERIMENTS_DIR,
        journal_path=paths.journal_path,
        refresh_data=True,
        refresh_macro=True,
        refresh_news=True,
        migrate_warehouse=True,
        paper_valuation=True,
    )
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="daily update")


def queue_snapshot(paths: DashboardPaths, *, refresh_data: bool, refresh_macro: bool, refresh_news: bool) -> QueuedJob:
    job = _store(paths).start_snapshot_build_job(
        config_path=paths.config_path,
        events_path=paths.events_path,
        macro_path=paths.macro_path,
        news_path=paths.news_path,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
    )
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="snapshot build")


def queue_warehouse_migration(paths: DashboardPaths) -> QueuedJob:
    job = _store(paths).start_warehouse_migration_job(
        experiment_dir=DEFAULT_EXPERIMENTS_DIR,
        journal_path=paths.journal_path,
    )
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="warehouse migration")


def queue_paper_valuation(paths: DashboardPaths) -> QueuedJob:
    job = _store(paths).start_paper_valuation_job(config_path=paths.config_path)
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="paper valuation")


def queue_monitoring_seed(
    paths: DashboardPaths,
    *,
    capital_base: float,
    top_n: int = DEFAULT_MONITORING_TOP_N,
    start_date: date,
) -> QueuedJob:
    job = _store(paths).start_monitoring_seed_job(
        mode="paper",
        account=DEFAULT_FORWARD_TEST_ACCOUNT,
        capital_base=capital_base,
        top_n=top_n,
        start_date=start_date.isoformat(),
    )
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="monitoring seed")


def queue_ml_diagnostics(paths: DashboardPaths, *, profile: str, refresh_data: bool) -> QueuedJob:
    job = _store(paths).start_ml_diagnostics_job(
        config_path=paths.config_path,
        output_dir=DEFAULT_ML_DIAGNOSTICS_DIR,
        profile=profile,
        refresh_data=refresh_data,
    )
    st.cache_data.clear()
    return QueuedJob(job_id=job.job_id, label="ML diagnostics")

