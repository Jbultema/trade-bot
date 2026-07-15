from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from trade_bot.config import load_config
from trade_bot.dashboard.loaders import (
    load_live_run,
    load_snapshot_dashboard_run,
    load_snapshot_dashboard_run_by_id,
    load_snapshot_jobs_frame,
)
from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_FORWARD_TEST_ACCOUNT,
    DEFAULT_FORWARD_TEST_STRATEGY,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_NEWS_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.action_headline import ActionHeadline, build_action_headline
from trade_bot.research.baselines import BaselineRun
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.trading.book_alignment import (
    BookAlignmentRun,
    build_book_alignment,
    latest_book_account_value,
)
from trade_bot.trading.journal import TradeJournal

RunSource = Literal["Latest snapshot (fast)", "Selected snapshot", "Live pipeline"]


@dataclass(frozen=True)
class DashboardPaths:
    config_path: Path = DEFAULT_CONFIG_PATH
    events_path: Path = DEFAULT_EVENTS_PATH
    macro_path: Path = DEFAULT_MACRO_PATH
    news_path: Path = DEFAULT_NEWS_PATH
    journal_path: Path = DEFAULT_JOURNAL_PATH
    run_store_path: Path = DEFAULT_RUN_STORE_DB_PATH
    artifact_dir: Path = DEFAULT_RUN_STORE_ARTIFACT_DIR
    job_log_dir: Path = DEFAULT_RUN_STORE_JOB_LOG_DIR


@dataclass(frozen=True)
class DashboardRuntime:
    paths: DashboardPaths
    run_source: RunSource
    bot_config: object
    baseline_run: BaselineRun
    snapshot_manifest: SnapshotManifest | None
    snapshot_loaded: bool
    journal: TradeJournal
    open_ticket_count: int
    book_alignment: BookAlignmentRun
    execution_book_alignment: BookAlignmentRun | None
    action_headline: ActionHeadline


def render_path_controls(defaults: DashboardPaths | None = None) -> DashboardPaths:
    defaults = defaults or DashboardPaths()
    with st.sidebar.expander("Local paths", expanded=False):
        return DashboardPaths(
            config_path=Path(st.text_input("Config", str(defaults.config_path))),
            events_path=Path(st.text_input("Events", str(defaults.events_path))),
            macro_path=Path(st.text_input("Macro", str(defaults.macro_path))),
            news_path=Path(st.text_input("News", str(defaults.news_path))),
            journal_path=Path(st.text_input("Trade journal", str(defaults.journal_path))),
            run_store_path=Path(st.text_input("Run store", str(defaults.run_store_path))),
            artifact_dir=Path(st.text_input("Snapshot artifacts", str(defaults.artifact_dir))),
            job_log_dir=Path(st.text_input("Snapshot job logs", str(defaults.job_log_dir))),
        )


def snapshot_choices(paths: DashboardPaths, *, limit: int = 100) -> pd.DataFrame:
    store = RunStore(paths.run_store_path, artifact_dir=paths.artifact_dir, job_log_dir=paths.job_log_dir)
    return store.list_snapshots(limit=limit)


def snapshot_option_label(frame: pd.DataFrame, run_id: str) -> str:
    row = frame[frame["run_id"].astype(str) == str(run_id)].iloc[0]
    return f"{row['market_date']} | {row['risk_status']} | {row['created_at_utc']} | {str(run_id)[:18]}"


def load_runtime(
    *,
    paths: DashboardPaths,
    run_source: RunSource,
    selected_snapshot_run_id: str | None,
    refresh_data: bool = False,
    refresh_macro: bool = False,
    refresh_news: bool = False,
) -> DashboardRuntime:
    bot_config = load_config(paths.config_path)
    snapshot_manifest: SnapshotManifest | None = None
    snapshot_loaded = False

    if run_source == "Latest snapshot (fast)":
        snapshot_payload = load_snapshot_dashboard_run(
            str(paths.config_path),
            str(paths.events_path),
            str(paths.macro_path),
            str(paths.news_path),
            str(paths.run_store_path),
            str(paths.artifact_dir),
            str(paths.job_log_dir),
        )
        if snapshot_payload is None:
            baseline_run = load_live_run(
                str(paths.config_path),
                str(paths.events_path),
                str(paths.macro_path),
                str(paths.news_path),
                refresh_data,
                refresh_macro,
                refresh_news,
            )
        else:
            baseline_run, snapshot_manifest = snapshot_payload
            snapshot_loaded = True
    elif run_source == "Selected snapshot" and selected_snapshot_run_id:
        baseline_run, snapshot_manifest = load_snapshot_dashboard_run_by_id(
            str(paths.run_store_path),
            str(paths.artifact_dir),
            str(paths.job_log_dir),
            selected_snapshot_run_id,
        )
        snapshot_loaded = True
    else:
        baseline_run = load_live_run(
            str(paths.config_path),
            str(paths.events_path),
            str(paths.macro_path),
            str(paths.news_path),
            refresh_data,
            refresh_macro,
            refresh_news,
        )

    journal = TradeJournal(paths.journal_path)
    open_tickets = journal.load_recommendation_tickets(status="open")
    book_alignment = build_book_alignment(
        journal=journal,
        trade_decision=baseline_run.trade_decision,
        prices=baseline_run.prices,
        mode="paper",
        account=DEFAULT_FORWARD_TEST_ACCOUNT,
        strategy_name=DEFAULT_FORWARD_TEST_STRATEGY,
        account_value=latest_book_account_value(
            journal,
            mode="paper",
            account=DEFAULT_FORWARD_TEST_ACCOUNT,
            strategy_name=DEFAULT_FORWARD_TEST_STRATEGY,
            default=10_000.0,
        ),
    )
    execution_book_alignment = (
        book_alignment if not book_alignment.position_plan.empty else None
    )
    action_headline = build_action_headline(
        current_state=baseline_run.current_state,
        trade_decision=baseline_run.trade_decision,
        news_monitor=baseline_run.news_monitor,
        open_ticket_count=len(open_tickets),
        position_plan=book_alignment.position_plan,
    )
    return DashboardRuntime(
        paths=paths,
        run_source=run_source,
        bot_config=bot_config,
        baseline_run=baseline_run,
        snapshot_manifest=snapshot_manifest,
        snapshot_loaded=snapshot_loaded,
        journal=journal,
        open_ticket_count=len(open_tickets),
        book_alignment=book_alignment,
        execution_book_alignment=execution_book_alignment,
        action_headline=action_headline,
    )


def freshness_label(runtime: DashboardRuntime) -> str:
    manifest = runtime.snapshot_manifest
    if manifest is None:
        return f"{runtime.run_source} live at {datetime.now(UTC).replace(microsecond=0).isoformat()}"
    return f"{manifest.market_date} | {manifest.risk_status.upper()} | {manifest.created_at_utc}"


def load_job_frame(paths: DashboardPaths) -> pd.DataFrame:
    return load_snapshot_jobs_frame(
        str(paths.run_store_path),
        str(paths.artifact_dir),
        str(paths.job_log_dir),
    )
