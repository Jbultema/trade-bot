from __future__ import annotations

import html
from datetime import UTC, date, datetime
from pathlib import Path

import streamlit as st

from trade_bot.config import load_config
from trade_bot.dashboard.components import _render_metric_info_rail
from trade_bot.dashboard.formatting import _safe_timezone
from trade_bot.dashboard.loaders import (
    load_experiment_dashboard_frames,
    load_experiment_scorecards_frame,
    load_live_run,
    load_snapshot_dashboard_run,
    load_snapshot_dashboard_run_by_id,
    load_snapshot_jobs_frame,
)
from trade_bot.dashboard.navigation import (
    render_dashboard_workbench_selector,
    render_selected_section_guide,
)
from trade_bot.dashboard.overview import (
    execution_book_alignment_or_none,
    headline_position_plan,
    render_operating_overview,
)
from trade_bot.dashboard.sections import _render_dashboard_section
from trade_bot.dashboard.styles import (
    _install_dashboard_styles,
    _install_quick_reference_rail_layout,
)
from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_FORWARD_TEST_ACCOUNT,
    DEFAULT_FORWARD_TEST_STRATEGY,
    DEFAULT_MACRO_PATH,
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_MONITORING_COHORT_START_DATE,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_NEWS_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.action_headline import build_action_headline
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.trading.book_alignment import build_book_alignment, latest_book_account_value
from trade_bot.trading.journal import DEFAULT_JOURNAL_PATH, TradeJournal


def _render_freshness_strip(
    *,
    snapshot_manifest: SnapshotManifest | None,
    baseline_run: object,
    run_source: str,
) -> None:
    market_date = html.escape(str(getattr(baseline_run.current_state, "market_date", "n/a")))
    risk_status = html.escape(
        str(getattr(baseline_run.current_state, "risk_status", "n/a")).upper()
    )
    local_timezone = _safe_timezone("America/Denver")
    if snapshot_manifest is not None:
        local_time = _local_time_label(snapshot_manifest.created_at_utc, local_timezone)
        utc_time = html.escape(snapshot_manifest.created_at_utc)
        snapshot_label = (
            "Selected snapshot" if run_source == "Selected snapshot" else "Latest snapshot"
        )
        freshness_text = f"{snapshot_label}: {html.escape(local_time)}"
        detail_text = f"UTC {utc_time}"
    else:
        local_time = (
            datetime.now(local_timezone).replace(microsecond=0).strftime("%Y-%m-%d %I:%M %p %Z")
        )
        freshness_text = f"{html.escape(run_source)} loaded live: {html.escape(local_time)}"
        detail_text = "No saved snapshot timestamp for this run"
    st.markdown(
        f"""
        <div class="freshness-strip">
            <span class="freshness-kicker">Latest update</span>
            <span class="freshness-main">{freshness_text}</span>
            <span class="freshness-chip">Market date {market_date}</span>
            <span class="freshness-chip">{risk_status} risk</span>
            <span class="freshness-detail">{detail_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _local_time_label(timestamp_utc: str, timezone: object) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except ValueError:
        return timestamp_utc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(timezone).strftime("%Y-%m-%d %I:%M %p %Z")


st.set_page_config(page_title="Trade Bot Dashboard", layout="wide")
_install_dashboard_styles()
st.markdown(
    """
    <div class="brand-masthead">
        <div class="brand-lockup">
            <div class="brand-mark" aria-label="Trade Bot mark">
                <span class="brand-mark-text">TB</span>
                <span class="brand-mark-line"></span>
            </div>
            <div class="brand-copy">
                <p class="brand-eyebrow">Regime Research Lab</p>
                <h1 class="brand-title">Trade Bot Operations</h1>
                <p class="brand-subtitle">
                    Local decision support for macro-aware swing research, scenario sizing,
                    and paper-monitored strategy evidence.
                </p>
            </div>
        </div>
        <div class="brand-proof-row">
            <span class="brand-proof">Local only</span>
            <span class="brand-proof">Long only</span>
            <span class="brand-proof">Paper first</span>
            <span class="brand-proof">Human reviewed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <div class="sidebar-kicker">Local Runtime</div>
        <div class="sidebar-title">Dashboard Controls</div>
    </div>
    """,
    unsafe_allow_html=True,
)
run_source = st.sidebar.radio(
    "Run source",
    ["Latest snapshot (fast)", "Selected snapshot", "Live pipeline"],
    index=0,
)
show_quick_reference = st.sidebar.toggle(
    "Show quick reference rail",
    value=False,
    help="Show the fixed right-side lookup rail for metrics, ticket fields, workflows, and ticker symbols.",
)
if show_quick_reference:
    _install_quick_reference_rail_layout()

with st.sidebar.expander("Local paths", expanded=False):
    config_path = Path(st.text_input("Config", str(DEFAULT_CONFIG_PATH)))
    events_path = Path(st.text_input("Events", str(DEFAULT_EVENTS_PATH)))
    macro_path = Path(st.text_input("Macro", str(DEFAULT_MACRO_PATH)))
    news_path = Path(st.text_input("News", str(DEFAULT_NEWS_PATH)))
    journal_path = Path(st.text_input("Trade journal", str(DEFAULT_JOURNAL_PATH)))
    run_store_path = Path(st.text_input("Run store", str(DEFAULT_RUN_STORE_DB_PATH)))
    artifact_dir = Path(st.text_input("Snapshot artifacts", str(DEFAULT_RUN_STORE_ARTIFACT_DIR)))
    job_log_dir = Path(st.text_input("Snapshot job logs", str(DEFAULT_RUN_STORE_JOB_LOG_DIR)))

run_store = RunStore(run_store_path, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
st.sidebar.caption(
    "Fast mode reads a precomputed snapshot. Selected snapshot lets you time-travel "
    "through recent saved dashboard states."
)
st.sidebar.caption(
    "Recommended: run the full update, then refresh the browser after the job completes."
)
selected_snapshot_run_id: str | None = None
if run_source == "Selected snapshot":
    snapshot_choices = run_store.list_snapshots(limit=100)
    if snapshot_choices.empty:
        st.sidebar.warning("No completed snapshots are available to select.")
    else:
        snapshot_options = snapshot_choices["run_id"].astype(str).tolist()

        def _snapshot_option_label(run_id: str) -> str:
            row = snapshot_choices[snapshot_choices["run_id"] == run_id].iloc[0]
            return (
                f"{row['market_date']} | {row['risk_status']} | "
                f"{row['created_at_utc']} | {run_id[:18]}"
            )

        selected_snapshot_run_id = st.sidebar.selectbox(
            "Snapshot",
            snapshot_options,
            format_func=_snapshot_option_label,
            help="Load the saved dashboard state from a specific run snapshot.",
        )
if st.sidebar.button("Run Full Daily Update", type="primary"):
    job = run_store.start_daily_update_job(
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        report_path=DEFAULT_REPORT_PATH,
        experiment_dir=DEFAULT_EXPERIMENTS_DIR,
        journal_path=journal_path,
        refresh_data=True,
        refresh_macro=True,
        refresh_news=True,
        migrate_warehouse=True,
        paper_valuation=True,
    )
    st.cache_data.clear()
    st.sidebar.success(f"Queued daily update job: {job.job_id}")
    st.sidebar.caption("Refresh this page after the job completes to load the new snapshot.")

with st.sidebar.expander("Targeted update jobs", expanded=False):
    st.caption(
        "Use these when the daily snapshot is already current and you only need one downstream "
        "piece refreshed."
    )
    if st.button("Migrate Warehouse", key="sidebar_migrate_warehouse"):
        job = run_store.start_warehouse_migration_job(
            experiment_dir=DEFAULT_EXPERIMENTS_DIR,
            journal_path=journal_path,
        )
        st.cache_data.clear()
        st.success(f"Queued warehouse migration: {job.job_id}")
    if st.button("Run Paper Valuation", key="sidebar_run_paper_valuation"):
        job = run_store.start_paper_valuation_job(config_path=config_path)
        st.cache_data.clear()
        st.success(f"Queued paper valuation: {job.job_id}")
    seed_top_n = st.number_input(
        "Seed top N paper windows",
        min_value=1,
        max_value=25,
        value=DEFAULT_MONITORING_TOP_N,
        step=1,
    )
    seed_capital = st.number_input(
        "Seed capital per window",
        min_value=100.0,
        max_value=1_000_000.0,
        value=10_000.0,
        step=1_000.0,
    )
    monitoring_start_date = st.date_input(
        "Monitoring cohort start",
        value=date.fromisoformat(DEFAULT_MONITORING_COHORT_START_DATE),
        help=(
            "Used when seeding new paper windows or resetting active paper windows. "
            "Use the same date for fair champion/challenger YTD comparisons."
        ),
    )
    if st.button("Seed Monitoring Windows", key="sidebar_seed_monitoring"):
        job = run_store.start_monitoring_seed_job(
            mode="paper",
            account=DEFAULT_FORWARD_TEST_ACCOUNT,
            capital_base=float(seed_capital),
            top_n=int(seed_top_n),
            start_date=monitoring_start_date.isoformat(),
        )
        st.cache_data.clear()
        st.success(f"Queued monitoring-window seed: {job.job_id}")
    if st.button("Reset Active Paper Windows To Cohort Start", key="sidebar_reset_monitoring"):
        job = run_store.start_monitoring_start_reset_job(
            config_path=config_path,
            start_date=monitoring_start_date.isoformat(),
            mode="paper",
            account=DEFAULT_FORWARD_TEST_ACCOUNT,
            status="active",
            value_after_reset=True,
        )
        st.cache_data.clear()
        st.success(f"Queued monitoring reset + valuation: {job.job_id}")
    ml_profile = st.selectbox("ML diagnostics profile", ["standard", "research"], index=0)
    ml_refresh_data = st.checkbox("Refresh prices for ML diagnostics", value=False)
    if st.button("Run ML Diagnostics", key="sidebar_run_ml_diagnostics"):
        job = run_store.start_ml_diagnostics_job(
            config_path=config_path,
            output_dir=DEFAULT_ML_DIAGNOSTICS_DIR,
            profile=ml_profile,
            refresh_data=ml_refresh_data,
        )
        st.cache_data.clear()
        st.success(f"Queued ML diagnostics: {job.job_id}")
    st.caption(
        "Large research sweeps remain CLI/Codex-driven because they are long-running, "
        "parameterized, and can create many artifacts."
    )

with st.sidebar.expander("Advanced snapshot options", expanded=False):
    refresh_data = st.checkbox("Refresh market data", value=False)
    refresh_macro = st.checkbox("Refresh macro data", value=False)
    refresh_news = st.checkbox("Refresh news", value=False)
    if st.button("Build Snapshot Only"):
        job = run_store.start_snapshot_build_job(
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
        )
        st.cache_data.clear()
        st.success(f"Queued snapshot job: {job.job_id}")
bot_config = load_config(config_path)

snapshot_jobs = load_snapshot_jobs_frame(str(run_store_path), str(artifact_dir), str(job_log_dir))
if not snapshot_jobs.empty:
    with st.sidebar.expander("Update jobs", expanded=False):
        job_columns = [
            "created_at_utc",
            "status",
            "run_id",
            "completed_at_utc",
            "log_path",
            "error_message",
        ]
        st.dataframe(snapshot_jobs[job_columns], width="stretch")

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
elif run_source == "Selected snapshot" and selected_snapshot_run_id is not None:
    snapshot_payload = load_snapshot_dashboard_run_by_id(
        str(run_store_path),
        str(artifact_dir),
        str(job_log_dir),
        selected_snapshot_run_id,
    )
    if snapshot_payload is None:
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

journal = TradeJournal(journal_path)
headline_open_tickets = journal.load_recommendation_tickets(status="open")
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
execution_book_alignment = execution_book_alignment_or_none(default_book_alignment)
action_headline = build_action_headline(
    current_state=baseline_run.current_state,
    trade_decision=baseline_run.trade_decision,
    news_monitor=baseline_run.news_monitor,
    open_ticket_count=len(headline_open_tickets),
    position_plan=headline_position_plan(
        baseline_run=baseline_run,
        default_book_alignment=default_book_alignment,
    ),
)
experiment_scorecards = load_experiment_scorecards_frame()
if show_quick_reference:
    _render_metric_info_rail()

_render_freshness_strip(
    snapshot_manifest=snapshot_manifest,
    baseline_run=baseline_run,
    run_source=run_source,
)
render_operating_overview(
    baseline_run=baseline_run,
    headline=action_headline,
    open_ticket_count=len(headline_open_tickets),
    experiment_scorecards=experiment_scorecards,
    default_book_alignment=default_book_alignment,
    previous_run=None,
    execution_book_alignment=execution_book_alignment,
)
selected_section = render_dashboard_workbench_selector()
if selected_section == "Research Lab":
    (
        experiment_scorecards,
        experiment_regimes,
        experiment_walk_forward,
        experiment_candidates,
        decision_sanity_impacts,
    ) = load_experiment_dashboard_frames()
else:
    empty_experiment_frame = experiment_scorecards.iloc[0:0].copy()
    experiment_regimes = empty_experiment_frame
    experiment_walk_forward = empty_experiment_frame
    experiment_candidates = empty_experiment_frame
    decision_sanity_impacts = empty_experiment_frame
render_selected_section_guide(selected_section)
st.markdown(
    '<div class="dashboard-workbench-divider" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)
section_slot = st.empty()
with section_slot.container(), st.spinner(f"Loading {selected_section}..."):
    _render_dashboard_section(
        selected_section,
        bot_config=bot_config,
        baseline_run=baseline_run,
        journal=journal,
        experiment_scorecards=experiment_scorecards,
        experiment_regimes=experiment_regimes,
        experiment_walk_forward=experiment_walk_forward,
        experiment_candidates=experiment_candidates,
        decision_sanity_impacts=decision_sanity_impacts,
        warehouse_path=str(run_store_path),
        artifact_dir=str(artifact_dir),
        job_log_dir=str(job_log_dir),
        book_alignment=execution_book_alignment,
    )
