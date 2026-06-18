from __future__ import annotations

from pathlib import Path

import streamlit as st

from trade_bot.config import load_config
from trade_bot.dashboard.book_alignment import _render_book_alignment
from trade_bot.dashboard.briefs import _render_decision_brief, _render_operating_brief
from trade_bot.dashboard.components import (
    _render_action_headline,
    _render_metric_guide,
)
from trade_bot.dashboard.loaders import (
    load_experiment_dashboard_frames,
    load_live_run,
    load_previous_snapshot_dashboard_run,
    load_snapshot_dashboard_run,
    load_snapshot_jobs_frame,
)
from trade_bot.dashboard.macro_minute import _render_macro_minute
from trade_bot.dashboard.sections import _render_dashboard_section
from trade_bot.dashboard.styles import _install_dashboard_styles
from trade_bot.DEFAULT import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_FORWARD_TEST_ACCOUNT,
    DEFAULT_FORWARD_TEST_STRATEGY,
    DEFAULT_MACRO_PATH,
    DEFAULT_NEWS_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.action_headline import build_action_headline
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.trading.book_alignment import build_book_alignment, latest_book_account_value
from trade_bot.trading.journal import DEFAULT_JOURNAL_PATH, TradeJournal

st.set_page_config(page_title="Trade Bot Dashboard", layout="wide")
_install_dashboard_styles()
st.title("Trade Bot Operations")
st.markdown(
    """
    <div class="dashboard-hero-panel">
        <div class="dashboard-hero-copy">
            <p class="dashboard-kicker">Local decision cockpit</p>
            <p class="dashboard-subtitle">
                Macro-aware swing strategy research, scenario sizing, paper monitoring,
                and human-reviewed trade decisions in one local workspace.
            </p>
            <div class="dashboard-pill-row">
                <span class="dashboard-pill">Local only</span>
                <span class="dashboard-pill">Long only</span>
                <span class="dashboard-pill">Paper first</span>
                <span class="dashboard-pill">Human reviewed</span>
            </div>
        </div>
        <div class="dashboard-hero-rail" aria-label="Operating constraints">
            <div class="dashboard-rail-item">
                <span class="dashboard-rail-label">Execution</span>
                <span class="dashboard-rail-value">Manual review</span>
            </div>
            <div class="dashboard-rail-item">
                <span class="dashboard-rail-label">Cadence</span>
                <span class="dashboard-rail-value">Snapshot first</span>
            </div>
            <div class="dashboard-rail-item">
                <span class="dashboard-rail-label">Evidence</span>
                <span class="dashboard-rail-value">Paper monitored</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

config_path = Path(st.sidebar.text_input("Config", str(DEFAULT_CONFIG_PATH)))
events_path = Path(st.sidebar.text_input("Events", str(DEFAULT_EVENTS_PATH)))
macro_path = Path(st.sidebar.text_input("Macro", str(DEFAULT_MACRO_PATH)))
news_path = Path(st.sidebar.text_input("News", str(DEFAULT_NEWS_PATH)))
journal_path = Path(st.sidebar.text_input("Trade journal", str(DEFAULT_JOURNAL_PATH)))
run_store_path = Path(st.sidebar.text_input("Run store", str(DEFAULT_RUN_STORE_DB_PATH)))
artifact_dir = Path(
    st.sidebar.text_input("Snapshot artifacts", str(DEFAULT_RUN_STORE_ARTIFACT_DIR))
)
job_log_dir = Path(st.sidebar.text_input("Snapshot job logs", str(DEFAULT_RUN_STORE_JOB_LOG_DIR)))
run_source = st.sidebar.radio("Run source", ["Latest snapshot (fast)", "Live pipeline"], index=0)
refresh_data = st.sidebar.checkbox("Refresh market data", value=False)
refresh_macro = st.sidebar.checkbox("Refresh macro data", value=False)
refresh_news = st.sidebar.checkbox("Refresh news", value=False)
st.sidebar.caption(
    "Fast mode reads the latest precomputed snapshot. Live mode runs the full pipeline."
)
bot_config = load_config(config_path)

run_store = RunStore(run_store_path, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
if st.sidebar.button("Start Background Snapshot Refresh"):
    job = run_store.start_snapshot_build_job(
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
    )
    st.sidebar.success(f"Queued snapshot job: {job.job_id}")

snapshot_jobs = load_snapshot_jobs_frame(str(run_store_path), str(artifact_dir), str(job_log_dir))
if not snapshot_jobs.empty:
    with st.sidebar.expander("Snapshot jobs", expanded=False):
        job_columns = [
            "created_at_utc",
            "status",
            "run_id",
            "completed_at_utc",
            "log_path",
            "error_message",
        ]
        st.dataframe(snapshot_jobs[job_columns], use_container_width=True)

snapshot_manifest: SnapshotManifest | None = None
snapshot_loaded = False
if run_source == "Latest snapshot (fast)":
    snapshot_payload = load_snapshot_dashboard_run(
        str(config_path),
        str(events_path),
        str(macro_path),
        str(news_path),
        str(run_store_path),
        str(artifact_dir),
        str(job_log_dir),
    )
    if snapshot_payload is None:
        st.warning(
            "No completed snapshot matches the current config files. "
            "Falling back to a live run for this session; build a snapshot to make cold opens fast."
        )
        baseline_run = load_live_run(
            str(config_path),
            str(events_path),
            str(macro_path),
            str(news_path),
            refresh_data,
            refresh_macro,
            refresh_news,
        )
    else:
        baseline_run, snapshot_manifest = snapshot_payload
        snapshot_loaded = True
else:
    baseline_run = load_live_run(
        str(config_path),
        str(events_path),
        str(macro_path),
        str(news_path),
        refresh_data,
        refresh_macro,
        refresh_news,
    )

if snapshot_manifest is not None:
    st.sidebar.success(
        "Snapshot loaded: "
        f"{snapshot_manifest.market_date} | {snapshot_manifest.risk_status.upper()} | "
        f"{snapshot_manifest.created_at_utc}"
    )
elif not snapshot_loaded:
    st.sidebar.info("Dashboard is using a live pipeline run.")

previous_snapshot_payload = load_previous_snapshot_dashboard_run(
    str(config_path),
    str(events_path),
    str(macro_path),
    str(news_path),
    str(run_store_path),
    str(artifact_dir),
    str(job_log_dir),
    current_run_id=snapshot_manifest.run_id if snapshot_manifest is not None else None,
)
previous_baseline_run = previous_snapshot_payload[0] if previous_snapshot_payload else None

journal = TradeJournal(journal_path)
headline_open_tickets = journal.load_recommendation_tickets(status="open")
action_headline = build_action_headline(
    current_state=baseline_run.current_state,
    trade_decision=baseline_run.trade_decision,
    news_monitor=baseline_run.news_monitor,
    open_ticket_count=len(headline_open_tickets),
)
_render_macro_minute(
    baseline_run=baseline_run,
    headline=action_headline,
    open_ticket_count=len(headline_open_tickets),
    previous_run=previous_baseline_run,
)
_render_action_headline(action_headline)
default_book_alignment = build_book_alignment(
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
_render_book_alignment(
    default_book_alignment,
    heading="Default Paper Book Alignment",
    show_position_plan=False,
)
(
    experiment_scorecards,
    experiment_regimes,
    experiment_walk_forward,
    experiment_candidates,
) = load_experiment_dashboard_frames()
_render_operating_brief(
    baseline_run=baseline_run,
    headline=action_headline,
)
_render_decision_brief(
    baseline_run=baseline_run,
    headline=action_headline,
    open_ticket_count=len(headline_open_tickets),
    experiment_scorecards=experiment_scorecards,
)
_render_metric_guide()
st.divider()

DASHBOARD_SECTIONS = (
    "Command Center",
    "Risk & Scenarios",
    "Research Lab",
    "Monitoring",
    "News & Macro",
    "Performance",
    "Forward Test",
)
selected_section = st.radio(
    "Dashboard section",
    DASHBOARD_SECTIONS,
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(
    "Sections render one at a time so the operating view stays focused and dense research tables do not dominate the page."
)
_render_dashboard_section(
    selected_section,
    bot_config=bot_config,
    baseline_run=baseline_run,
    journal=journal,
    experiment_scorecards=experiment_scorecards,
    experiment_regimes=experiment_regimes,
    experiment_walk_forward=experiment_walk_forward,
    experiment_candidates=experiment_candidates,
    warehouse_path=str(run_store_path),
)
