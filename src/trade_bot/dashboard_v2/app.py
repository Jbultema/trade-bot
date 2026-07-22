from __future__ import annotations

import html
from datetime import date

import streamlit as st

from trade_bot.dashboard.components import _render_metric_info_rail
from trade_bot.dashboard.styles import (
    _install_dashboard_styles,
    _install_quick_reference_rail_layout,
)
from trade_bot.dashboard_v2.perf import render_perf_footer, timed
from trade_bot.dashboard_v2.routes import route_by_key, routes
from trade_bot.dashboard_v2.services.job_service import (
    queue_daily_update,
    queue_ml_diagnostics,
    queue_monitoring_seed,
    queue_paper_valuation,
    queue_snapshot,
    queue_warehouse_migration,
)
from trade_bot.dashboard_v2.services.runtime import (
    HISTORICAL_SNAPSHOT_NOTICE,
    DashboardPaths,
    freshness_label,
    load_job_frame,
    load_runtime,
    render_path_controls,
    snapshot_choices,
    snapshot_option_label,
)
from trade_bot.dashboard_v2.session_state import DEFAULT_ROUTE_KEY, route_state_key
from trade_bot.dashboard_v2.styles import install_v2_styles
from trade_bot.DEFAULTS import DEFAULT_MONITORING_COHORT_START_DATE, DEFAULT_MONITORING_TOP_N

st.set_page_config(page_title="Trade Bot Dashboard", layout="wide")
_install_dashboard_styles()
install_v2_styles()
st.session_state["dashboard_v2_perf_samples"] = []

st.markdown(
    """
    <div class="v2-masthead">
        <div>
            <p class="v2-kicker">Trade Bot V2.2</p>
            <h1 class="v2-title">Fast Operations Workbench</h1>
            <p class="v2-subtitle">
                Summary-first dashboard over the same local snapshots, DuckDB warehouse,
                and research artifacts. Heavy diagnostics load only when requested.
            </p>
        </div>
        <div class="v2-chip-row">
            <span class="v2-chip">Snapshot first</span>
            <span class="v2-chip">DuckDB backed</span>
            <span class="v2-chip">Heavy work gated</span>
            <span class="v2-chip">Primary UI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Dashboard")
run_source = st.sidebar.radio(
    "Run source",
    ["Latest snapshot (fast)", "Selected snapshot", "Live pipeline"],
    index=0,
)
show_quick_reference = st.sidebar.toggle(
    "Show quick reference rail",
    value=False,
    help=(
        "Show the fixed right-side lookup rail for metrics, ticket fields, workflows, "
        "and ticker symbols."
    ),
)
if show_quick_reference:
    _install_quick_reference_rail_layout()
paths = render_path_controls(DashboardPaths())

selected_snapshot_run_id: str | None = None
if run_source == "Selected snapshot":
    choices = snapshot_choices(paths)
    if choices.empty:
        st.sidebar.warning("No completed snapshots are available.")
    else:
        options = choices["run_id"].astype(str).tolist()
        selected_snapshot_run_id = st.sidebar.selectbox(
            "Snapshot",
            options,
            format_func=lambda run_id: snapshot_option_label(choices, str(run_id)),
        )

with st.sidebar.expander("Refresh jobs", expanded=False):
    if st.button("Run Full Daily Update", type="primary"):
        job = queue_daily_update(paths)
        st.success(f"Queued {job.label}: {job.job_id}")
    refresh_data = st.checkbox("Refresh market data", value=False)
    refresh_macro = st.checkbox("Refresh macro data", value=False)
    refresh_news = st.checkbox("Refresh news", value=False)
    if st.button("Build Snapshot Only"):
        job = queue_snapshot(
            paths,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
        )
        st.success(f"Queued {job.label}: {job.job_id}")
    if st.button("Migrate Warehouse"):
        job = queue_warehouse_migration(paths)
        st.success(f"Queued {job.label}: {job.job_id}")
    if st.button("Run Paper Valuation"):
        job = queue_paper_valuation(paths)
        st.success(f"Queued {job.label}: {job.job_id}")

with st.sidebar.expander("Monitoring / diagnostics jobs", expanded=False):
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
    )
    if st.button("Seed Monitoring Windows"):
        job = queue_monitoring_seed(
            paths,
            capital_base=float(seed_capital),
            top_n=int(seed_top_n),
            start_date=monitoring_start_date,
        )
        st.success(f"Queued {job.label}: {job.job_id}")
    ml_profile = st.selectbox("ML diagnostics profile", ["standard", "research"], index=0)
    ml_refresh_data = st.checkbox("Refresh prices for ML diagnostics", value=False)
    if st.button("Run ML Diagnostics"):
        job = queue_ml_diagnostics(
            paths,
            profile=str(ml_profile),
            refresh_data=bool(ml_refresh_data),
        )
        st.success(f"Queued {job.label}: {job.job_id}")

with timed("runtime.load"):
    runtime = load_runtime(
        paths=paths,
        run_source=run_source,  # type: ignore[arg-type]
        selected_snapshot_run_id=selected_snapshot_run_id,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
    )

if runtime.snapshot_manifest is not None:
    st.sidebar.success(f"Snapshot loaded: {freshness_label(runtime)}")
else:
    st.sidebar.info("Using a live pipeline run.")
if runtime.operating_strategy_error and not runtime.is_historical_snapshot_mode:
    st.error("Promoted operating book is unavailable: " + runtime.operating_strategy_error)
if run_source == "Selected snapshot":
    st.sidebar.warning(
        "Point-in-time snapshot mode pins the baseline run only and is display-only. Current "
        "action and risk surfaces are suppressed; persisted research artifacts retain their "
        "own source dates."
    )
    st.warning(HISTORICAL_SNAPSHOT_NOTICE)

jobs = load_job_frame(paths)
if not jobs.empty:
    with st.sidebar.expander("Recent jobs", expanded=False):
        job_columns = [
            column
            for column in [
                "created_at_utc",
                "status",
                "run_id",
                "completed_at_utc",
                "log_path",
                "error_message",
            ]
            if column in jobs
        ]
        st.dataframe(jobs[job_columns], width="stretch", hide_index=True)

st.markdown(
    f"""
    <div class="freshness-strip">
        <span class="freshness-kicker">Latest update</span>
        <span class="freshness-main">{html.escape(freshness_label(runtime))}</span>
        <span class="freshness-chip">Market date {html.escape(str(runtime.baseline_run.current_state.market_date))}</span>
        <span class="freshness-chip">{html.escape(str(runtime.baseline_run.current_state.risk_status).upper())} risk</span>
    </div>
    """,
    unsafe_allow_html=True,
)

route_options = routes()
label_to_key = {route.label: route.key for route in route_options}
default_key = st.session_state.get(route_state_key(), DEFAULT_ROUTE_KEY)
default_label = next(
    (route.label for route in route_options if route.key == default_key),
    route_options[0].label,
)
selected_label = st.pills(
    "V2.2 workbench",
    [route.label for route in route_options],
    selection_mode="single",
    default=default_label,
    key="dashboard_v2_route_label",
)
selected_route_key = label_to_key.get(str(selected_label or default_label), DEFAULT_ROUTE_KEY)
st.session_state[route_state_key()] = selected_route_key
route = route_by_key(selected_route_key)

st.markdown(
    f"""
    <div class="v2-route-card">
        <h2>{html.escape(route.title)}</h2>
        <p><strong>{html.escape(route.lane)}</strong> / {html.escape(route.question)}</p>
        <span class="v2-runtime">{html.escape(route.runtime)}: {html.escape(route.runtime_note)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with timed(f"page.{route.key}"), st.spinner(f"Loading {route.title}..."):
    route.render(runtime)

if show_quick_reference:
    _render_metric_info_rail()

render_perf_footer()
