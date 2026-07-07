from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.DEFAULTS import (
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
    DEFAULT_SCENARIO_EXPLANATION_TOP_SCENARIOS,
    DEFAULT_SCENARIO_HISTORY_SNAPSHOT_LIMIT,
    DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.operating_exposure import (
    aggregate_beta_adjusted_spy_delta,
    build_beta_adjusted_delta_table,
    build_sleeve_exposure_table,
    build_tactical_matrix,
    weights_from_position_plan,
)
from trade_bot.storage.run_store import RunStore

_SCENARIO_HISTORY_COLUMNS = [
    "snapshot_time",
    "market_date",
    "horizon",
    "scenario",
    "risk_bucket",
    "probability",
    "rank",
    "run_id",
]
_SCENARIO_BUCKET_COLOR_MAP = {
    "risk_on": "#0f766e",
    "risk_on_fragile": "#84cc16",
    "fragile_upside": "#84cc16",
    "transition": "#b7791f",
    "risk_off_then_relief": "#7c3aed",
    "risk_off": "#b91c1c",
    "shock": "#7f1d1d",
}


def _render_risk_and_scenarios(
    baseline_run: BaselineRun,
    *,
    run_store_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
) -> None:
    current_state = baseline_run.current_state
    trade_decision = baseline_run.trade_decision
    regime_instability = getattr(current_state, "regime_instability", pd.DataFrame())
    regime_instability_components = getattr(
        current_state,
        "regime_instability_components",
        pd.DataFrame(),
    )

    st.subheader("Portfolio Risk Engine")
    st.caption(
        "Sizing guardrails. These rows explain whether scenarios, stress loss, beta, or concentration changed the target position."
    )
    portfolio_risk = baseline_run.portfolio_risk or trade_decision.portfolio_risk
    if portfolio_risk is None or portfolio_risk.summary.empty:
        st.write("No portfolio risk diagnostics available.")
    else:
        risk_summary = portfolio_risk.summary.iloc[0]
        risk_cols = st.columns(6)
        _helped_metric(risk_cols[0], "Risk Level", str(risk_summary["portfolio_risk_level"]))
        _helped_metric(
            risk_cols[1],
            "Risk Multiplier",
            f"{float(risk_summary['portfolio_risk_multiplier']):.2f}",
            key="portfolio_risk_multiplier",
        )
        _helped_metric(
            risk_cols[2],
            "ES 95",
            f"{float(risk_summary['post_expected_shortfall_95']):.2%}",
            key="post_expected_shortfall_95",
        )
        _helped_metric(
            risk_cols[3],
            "Max Stress Loss",
            f"{float(risk_summary['post_max_stress_loss']):.2%}",
            key="post_max_stress_loss",
        )
        _helped_metric(
            risk_cols[4],
            "Equity Beta",
            f"{float(risk_summary['post_equity_beta']):.2f}",
            key="post_equity_beta",
        )
        _helped_metric(
            risk_cols[5],
            "AI Beta",
            f"{float(risk_summary['post_ai_beta']):.2f}",
            key="post_ai_beta",
        )

        (
            risk_constraints_tab,
            risk_scenario_tab,
            risk_factor_tab,
            risk_tail_tab,
            risk_correlation_tab,
        ) = st.tabs(["Constraints", "Scenarios", "Factors / Betas", "Tail / Stress", "Correlation"])
        with risk_constraints_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.summary))
            _render_metric_dataframe(_display_metrics(portfolio_risk.constraint_report))
            st.caption("Risk-engine sizing bridge")
            _render_metric_dataframe(_display_metrics(portfolio_risk.sizing_adjustments))
        with risk_scenario_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.scenario_risk_budget))
        with risk_factor_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.factor_exposures))
            st.caption("Pre- versus post-risk beta decomposition")
            _render_metric_dataframe(_display_metrics(portfolio_risk.beta_decomposition))
        with risk_tail_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.tail_risk))
            _render_metric_dataframe(_display_metrics(portfolio_risk.stress_tests))
        with risk_correlation_tab:
            _render_metric_dataframe(_display_metrics(portfolio_risk.correlation_regime))
            _render_metric_dataframe(_display_metrics(portfolio_risk.marginal_risk_contribution))

    st.subheader("Operating Exposure")
    st.caption(
        "Current target posture translated into operating sleeves, percent of maximum sleeve exposure, "
        "and beta-adjusted S&P 500 delta."
    )
    current_weights = weights_from_position_plan(trade_decision.position_plan)
    if current_weights.empty:
        st.write("No current target weights are available for exposure diagnostics.")
    else:
        sleeve_exposure = build_sleeve_exposure_table(current_weights, baseline_run.prices)
        beta_delta = aggregate_beta_adjusted_spy_delta(baseline_run.prices, current_weights)
        exposure_cols = st.columns(6)
        _helped_metric(
            exposure_cols[0],
            "Beta-Adjusted S&P Delta",
            f"{beta_delta:.1%}" if pd.notna(beta_delta) else "n/a",
            key="beta_adjusted_spy_delta",
        )
        for column_index, sleeve in enumerate(["stocks", "defensive", "gold", "crypto", "credit"], start=1):
            sleeve_row = sleeve_exposure[sleeve_exposure["sleeve"].astype(str) == sleeve]
            percent_of_max = (
                float(sleeve_row["percent_of_max_sleeve"].iloc[0])
                if not sleeve_row.empty
                else float("nan")
            )
            _helped_metric(
                exposure_cols[column_index],
                f"{sleeve.title()} % of Max",
                f"{percent_of_max:.0%}" if pd.notna(percent_of_max) else "n/a",
                key="percent_of_max_sleeve",
            )

        sleeve_tab, beta_tab, tactical_tab = st.tabs(
            ["Sleeve Exposure", "Beta Delta", "Tactical Matrix"]
        )
        with sleeve_tab:
            _render_metric_dataframe(_display_metrics(sleeve_exposure), hide_index=True)
        with beta_tab:
            beta_table = build_beta_adjusted_delta_table(baseline_run.prices, current_weights)
            _render_metric_dataframe(_display_metrics(beta_table), hide_index=True)
        with tactical_tab:
            tactical_matrix = build_tactical_matrix(
                baseline_run.prices,
                current_weights=current_weights,
                risk_status=str(getattr(current_state, "risk_status", "")),
                regime=_lead_regime_label(current_state),
            )
            _render_metric_dataframe(_display_metrics(tactical_matrix), hide_index=True)

    st.subheader("Regime Instability Index")
    st.caption(
        "Watch-only transition-risk diagnostic. This summarizes realized volatility, +/-1% SPY days, "
        "dispersion, correlation shift, breadth/concentration, volatility pressure, and credit stress. "
        "It does not alter sizing until we backtest it as an overlay."
    )
    if regime_instability.empty:
        st.write("No regime-instability diagnostics are available.")
    else:
        instability = regime_instability.iloc[0]
        instability_cols = st.columns(5)
        _helped_metric(
            instability_cols[0],
            "Instability",
            str(instability.get("regime_instability_state", "n/a")).upper(),
        )
        _helped_metric(
            instability_cols[1],
            "Score",
            f"{float(instability.get('regime_instability_score', 0.0)):.2f}",
        )
        _helped_metric(
            instability_cols[2],
            "SPY +/-1% YTD",
            f"{float(instability.get('spy_ytd_large_move_share', 0.0)):.1%}",
        )
        _helped_metric(
            instability_cols[3],
            "Large Move Days",
            (
                f"{int(instability.get('spy_ytd_large_move_days', 0))}/"
                f"{int(instability.get('spy_ytd_trading_days', 0))}"
            ),
        )
        _helped_metric(
            instability_cols[4],
            "Use",
            "Watch Only",
        )
        st.write(str(instability.get("regime_instability_read", "")))
        _render_metric_dataframe(_display_metrics(regime_instability_components))

    st.subheader("Future-State Scenario Lattice")
    scenario_lattice = current_state.scenario_lattice
    _render_scenario_probability_explanation(
        scenario_lattice,
        current_state.scenario_drivers,
    )
    _render_scenario_probability_history(
        baseline_run,
        run_store_path=run_store_path,
        artifact_dir=artifact_dir,
        job_log_dir=job_log_dir,
    )
    _render_metric_dataframe(_display_metrics(current_state.scenario_drivers))

    scenario_horizon = st.radio(
        "Scenario horizon",
        ["1w", "1m", "3m", "6m"],
        index=1,
        horizontal=True,
    )
    scenario_bucket_options = ["all", *sorted(scenario_lattice["risk_bucket"].unique())]
    scenario_bucket = st.selectbox("Risk bucket", scenario_bucket_options)
    scenario_limit = st.slider("Scenarios shown", min_value=5, max_value=20, value=12, step=1)
    scenario_view = scenario_lattice[scenario_lattice["horizon"] == scenario_horizon]
    if scenario_bucket != "all":
        scenario_view = scenario_view[scenario_view["risk_bucket"] == scenario_bucket]
    _render_metric_dataframe(
        _display_metrics(scenario_view.sort_values("rank").head(scenario_limit))
    )

    st.subheader("Risk Confirmation Matrix")
    st.dataframe(current_state.confirmation_matrix, use_container_width=True)

    st.subheader("Market Health")
    _render_metric_dataframe(_display_metrics(current_state.market_health))

    st.subheader("Vol-Adjusted Momentum Signal Table")
    momentum_filter = st.radio(
        "Momentum filter",
        ["all", "bullish", "neutral", "bearish"],
        horizontal=True,
    )
    momentum_state_table = current_state.momentum_state.copy()
    if momentum_filter != "all":
        momentum_state_table = momentum_state_table[momentum_state_table["momentum_state_label"] == momentum_filter]
    _render_metric_dataframe(_display_metrics(momentum_state_table.head(75)))


def _render_scenario_probability_explanation(
    scenario_lattice: pd.DataFrame,
    scenario_drivers: pd.DataFrame,
) -> None:
    if scenario_lattice.empty and scenario_drivers.empty:
        st.write("No scenario-probability explanation is available.")
        return
    st.markdown("**Scenario Probability Explanation**")
    st.caption(
        "This is the scenario layer's evidence bridge: probabilities are summarized by horizon, "
        "then driver scores show what is pushing the distribution. Use this to understand why the "
        "risk budget changed before inspecting the full lattice."
    )
    chart_cols = st.columns(2)
    with chart_cols[0]:
        probability_figure = _scenario_probability_stack_figure(scenario_lattice)
        if probability_figure.data:
            st.plotly_chart(probability_figure, use_container_width=True)
        else:
            st.write("No probability stack is available.")
    with chart_cols[1]:
        driver_figure = _scenario_driver_score_figure(scenario_drivers)
        if driver_figure.data:
            st.plotly_chart(driver_figure, use_container_width=True)
        else:
            st.write("No driver-score chart is available.")


def _render_scenario_probability_history(
    baseline_run: BaselineRun,
    *,
    run_store_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
) -> None:
    st.markdown("**Scenario Probability Evolution**")
    st.caption(
        "Saved daily snapshots let this page show whether risk-off, transition, fragile upside, "
        "and risk-on probabilities are rising, fading, or staying stable. Use this before "
        "reacting to a single current-day probability."
    )
    history = _load_scenario_history_frame(
        str(run_store_path),
        str(artifact_dir),
        str(job_log_dir),
        DEFAULT_SCENARIO_HISTORY_SNAPSHOT_LIMIT,
    )
    if history.empty:
        history = _scenario_history_from_lattice(
            baseline_run.current_state.scenario_lattice,
            market_date=getattr(baseline_run.current_state, "market_date", ""),
            created_at_utc=getattr(baseline_run.current_state, "market_date", ""),
            run_id="current_session",
        )
    if history.empty:
        st.write("No scenario history is available yet.")
        return

    available_horizons = _ordered_horizons(history["horizon"].dropna().astype(str).unique())
    if not available_horizons:
        st.write("No scenario horizons are available in saved snapshots.")
        return
    control_cols = st.columns([1, 1, 2])
    selected_horizon = control_cols[0].selectbox(
        "Scenario history horizon",
        available_horizons,
        index=available_horizons.index("1m") if "1m" in available_horizons else 0,
        key="risk_scenario_history_horizon",
    )
    distinct_market_dates = history["market_date"].dropna().nunique()
    default_granularity = (
        "Latest per market date" if distinct_market_dates > 1 else "Every saved refresh"
    )
    granularity_options = ["Latest per market date", "Every saved refresh"]
    granularity = control_cols[1].radio(
        "History granularity",
        granularity_options,
        index=granularity_options.index(default_granularity),
        horizontal=True,
        key="risk_scenario_history_granularity",
    )
    control_cols[2].caption(
        "Latest-per-date is cleaner for daily trends. Every saved refresh is useful when you "
        "have multiple same-day updates or only one market date loaded."
    )

    scoped_history = _scenario_history_scope(history, granularity)
    scoped_history = scoped_history[scoped_history["horizon"].astype(str) == selected_horizon]
    if scoped_history.empty:
        st.write("No scenario history is available for the selected horizon.")
        return
    insights = _scenario_history_insights(scoped_history, selected_horizon)
    metric_cols = st.columns(4)
    summary = _scenario_history_metric_summary(scoped_history)
    _helped_metric(metric_cols[0], "Latest Risk-Off", summary["latest_risk_off"])
    _helped_metric(metric_cols[1], "Latest Transition", summary["latest_transition"])
    _helped_metric(metric_cols[2], "Dominant Bucket", summary["dominant_bucket"])
    _helped_metric(metric_cols[3], "History Points", summary["history_points"])
    _render_metric_dataframe(insights, hide_index=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        bucket_figure = _scenario_bucket_history_figure(scoped_history)
        if bucket_figure.data:
            st.plotly_chart(bucket_figure, use_container_width=True)
        else:
            st.write("No risk-bucket history chart is available.")
    with chart_cols[1]:
        named_figure = _scenario_named_history_figure(scoped_history)
        if named_figure.data:
            st.plotly_chart(named_figure, use_container_width=True)
        else:
            st.write("No named-scenario history chart is available.")

    with st.expander("Scenario history detail", expanded=False):
        detail = (
            scoped_history[
                [
                    "snapshot_time",
                    "market_date",
                    "horizon",
                    "scenario",
                    "risk_bucket",
                    "probability",
                    "rank",
                    "run_id",
                ]
            ]
            .sort_values(["snapshot_time", "rank", "probability"], ascending=[False, True, False])
            .head(200)
        )
        _render_metric_dataframe(_display_metrics(detail), hide_index=True)


@st.cache_data(show_spinner=False, ttl=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS)
def _load_scenario_history_frame(
    store_path_string: str,
    artifact_dir_string: str,
    job_log_dir_string: str,
    limit: int,
) -> pd.DataFrame:
    store = RunStore(
        store_path_string,
        artifact_dir=artifact_dir_string,
        job_log_dir=job_log_dir_string,
    )
    snapshots = store.list_snapshots(limit=limit)
    if snapshots.empty or "run_id" not in snapshots:
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    frames: list[pd.DataFrame] = []
    for _, row in snapshots.iloc[::-1].iterrows():
        run_id = str(row["run_id"])
        try:
            run, manifest = store.load_snapshot(run_id)
        except (FileNotFoundError, TypeError, OSError, AttributeError, ValueError):
            continue
        frame = _scenario_history_from_lattice(
            getattr(run.current_state, "scenario_lattice", pd.DataFrame()),
            market_date=manifest.market_date,
            created_at_utc=manifest.created_at_utc,
            run_id=manifest.run_id,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    history = pd.concat(frames, ignore_index=True)
    return _clean_scenario_history(history)


def _scenario_history_from_lattice(
    scenario_lattice: pd.DataFrame,
    *,
    market_date: str,
    created_at_utc: str,
    run_id: str,
) -> pd.DataFrame:
    if scenario_lattice.empty or not {"horizon", "risk_bucket", "probability"}.issubset(
        scenario_lattice.columns
    ):
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    frame = scenario_lattice.copy()
    if "scenario" not in frame:
        frame["scenario"] = frame["risk_bucket"].astype(str)
    if "rank" not in frame:
        frame["rank"] = (
            frame.groupby("horizon")["probability"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    frame["snapshot_time"] = pd.to_datetime(created_at_utc, errors="coerce", utc=True)
    if frame["snapshot_time"].isna().all():
        frame["snapshot_time"] = pd.to_datetime(market_date, errors="coerce", utc=True)
    market_timestamp = pd.to_datetime(market_date, errors="coerce")
    frame["market_date"] = market_timestamp.date() if pd.notna(market_timestamp) else pd.NaT
    frame["run_id"] = str(run_id)
    frame = frame.dropna(subset=["probability", "snapshot_time"])
    if frame.empty:
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    return frame[_SCENARIO_HISTORY_COLUMNS].copy()


def _clean_scenario_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    frame = history.copy()
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce", utc=True)
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce").dt.date
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_time", "market_date", "probability"])
    if frame.empty:
        return pd.DataFrame(columns=_SCENARIO_HISTORY_COLUMNS)
    return frame.sort_values(["snapshot_time", "horizon", "rank"]).reset_index(drop=True)


def _scenario_history_scope(history: pd.DataFrame, granularity: str) -> pd.DataFrame:
    frame = _clean_scenario_history(history)
    if frame.empty:
        return frame
    if granularity == "Latest per market date":
        frame = (
            frame.sort_values("snapshot_time")
            .drop_duplicates(["market_date", "horizon", "scenario"], keep="last")
            .copy()
        )
        frame["history_time"] = pd.to_datetime(frame["market_date"], errors="coerce")
    else:
        frame["history_time"] = frame["snapshot_time"]
    return frame.sort_values(["history_time", "horizon", "rank"]).reset_index(drop=True)


def _scenario_history_metric_summary(history: pd.DataFrame) -> dict[str, str]:
    bucket_pivot = _scenario_bucket_history_pivot(history)
    if bucket_pivot.empty:
        return {
            "latest_risk_off": "n/a",
            "latest_transition": "n/a",
            "dominant_bucket": "n/a",
            "history_points": "0",
        }
    latest = bucket_pivot.iloc[-1]
    dominant_bucket = str(latest.idxmax()).replace("_", " ").title()
    latest_risk_off = float(
        latest[[column for column in latest.index if "risk_off" in str(column)]].sum()
    )
    latest_transition = float(latest.get("transition", 0.0))
    return {
        "latest_risk_off": f"{latest_risk_off:.1%}",
        "latest_transition": f"{latest_transition:.1%}",
        "dominant_bucket": dominant_bucket,
        "history_points": f"{bucket_pivot.shape[0]}",
    }


def _scenario_history_insights(history: pd.DataFrame, horizon: str) -> pd.DataFrame:
    bucket_pivot = _scenario_bucket_history_pivot(history)
    if bucket_pivot.empty:
        return pd.DataFrame(columns=["insight", "read", "detail"])
    latest_time = bucket_pivot.index[-1]
    latest = bucket_pivot.iloc[-1]
    prior = bucket_pivot.iloc[-2] if len(bucket_pivot) > 1 else None
    latest_risk_off = float(
        latest[[column for column in latest.index if "risk_off" in str(column)]].sum()
    )
    latest_transition = float(latest.get("transition", 0.0))
    dominant_bucket = str(latest.idxmax()).replace("_", " ").title()
    rows = [
        {
            "insight": "Current regime mix",
            "read": f"{dominant_bucket} leads",
            "detail": (
                f"{horizon} risk-off is {latest_risk_off:.1%} and transition is "
                f"{latest_transition:.1%} as of {_format_history_time(latest_time)}."
            ),
        }
    ]
    if prior is None:
        rows.append(
            {
                "insight": "Trend evidence",
                "read": "More history needed",
                "detail": "Only one saved history point is available for this horizon.",
            }
        )
    else:
        prior_risk_off = float(
            prior[[column for column in prior.index if "risk_off" in str(column)]].sum()
        )
        prior_transition = float(prior.get("transition", 0.0))
        delta_risk_off = latest_risk_off - prior_risk_off
        delta_transition = latest_transition - prior_transition
        rows.append(
            {
                "insight": "Risk pressure trend",
                "read": _risk_pressure_read(delta_risk_off, delta_transition),
                "detail": (
                    f"Since the prior saved point, risk-off changed {delta_risk_off:+.1%} "
                    f"and transition changed {delta_transition:+.1%}."
                ),
            }
        )
        scenario_delta = _largest_named_scenario_delta(history, latest_time)
        if scenario_delta:
            rows.append(scenario_delta)
    rows.append(
        {
            "insight": "Coverage",
            "read": f"{len(bucket_pivot)} saved point(s)",
            "detail": (
                f"History spans {_format_history_time(bucket_pivot.index[0])} to "
                f"{_format_history_time(bucket_pivot.index[-1])}."
            ),
        }
    )
    return pd.DataFrame(rows)


def _scenario_bucket_history_figure(history: pd.DataFrame) -> go.Figure:
    pivot = _scenario_bucket_history_pivot(history)
    if pivot.empty:
        return go.Figure()
    figure = go.Figure()
    bucket_order = _ordered_buckets(pivot.columns)
    if len(pivot.index) == 1:
        latest = pivot.iloc[-1]
        figure.add_trace(
            go.Bar(
                x=[str(bucket).replace("_", " ").title() for bucket in bucket_order],
                y=[float(latest.get(bucket, 0.0)) for bucket in bucket_order],
                marker_color=[
                    _SCENARIO_BUCKET_COLOR_MAP.get(str(bucket), "#64748b")
                    for bucket in bucket_order
                ],
                hovertemplate="%{x}<br>Probability: %{y:.1%}<extra></extra>",
                name="Latest probability",
            )
        )
        xaxis = {"title": "Risk bucket"}
    else:
        for bucket in bucket_order:
            figure.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[bucket],
                    name=str(bucket).replace("_", " ").title(),
                    mode="lines",
                    stackgroup="scenario_probability",
                    line={
                        "width": 1.8,
                        "color": _SCENARIO_BUCKET_COLOR_MAP.get(str(bucket), "#64748b"),
                    },
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
                )
            )
        xaxis = {"title": "Snapshot"}
    figure.update_layout(
        title="Risk-Bucket Probability Over Time",
        template="plotly_white",
        yaxis={"title": "Probability", "tickformat": ".0%", "range": [0, 1]},
        xaxis=xaxis,
        height=380,
        margin={"l": 20, "r": 20, "t": 60, "b": 60},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
    )
    return figure


def _scenario_named_history_figure(
    history: pd.DataFrame,
    *,
    top_n: int = DEFAULT_SCENARIO_EXPLANATION_TOP_SCENARIOS,
) -> go.Figure:
    if history.empty or "history_time" not in history:
        return go.Figure()
    frame = history.copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    latest_time = frame["history_time"].max()
    latest = (
        frame[frame["history_time"] == latest_time]
        .sort_values("probability", ascending=False)
        .head(top_n)
    )
    top_scenarios = list(latest["scenario"].astype(str))
    if not top_scenarios:
        return go.Figure()
    scoped = frame[frame["scenario"].astype(str).isin(top_scenarios)]
    pivot = (
        scoped.groupby(["history_time", "scenario"], dropna=False)["probability"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    figure = go.Figure()
    if len(pivot.index) == 1:
        values = pivot.iloc[-1].sort_values(ascending=False)
        figure.add_trace(
            go.Bar(
                x=values.index.astype(str),
                y=values.values,
                marker_color="#0f766e",
                hovertemplate="%{x}<br>Probability: %{y:.1%}<extra></extra>",
                name="Latest probability",
            )
        )
        xaxis = {"title": "Scenario"}
    else:
        for scenario in top_scenarios:
            if scenario in pivot:
                figure.add_trace(
                    go.Scatter(
                        x=pivot.index,
                        y=pivot[scenario],
                        mode="lines+markers",
                        name=scenario,
                        hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
                    )
                )
        xaxis = {"title": "Snapshot"}
    figure.update_layout(
        title="Top Named Scenarios Over Time",
        template="plotly_white",
        yaxis={"title": "Probability", "tickformat": ".0%", "range": [0, 1]},
        xaxis=xaxis,
        height=380,
        margin={"l": 20, "r": 20, "t": 60, "b": 60},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
    )
    return figure


def _scenario_bucket_history_pivot(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "history_time" not in history:
        return pd.DataFrame()
    frame = history.copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    pivot = (
        frame.dropna(subset=["probability", "history_time"])
        .groupby(["history_time", "risk_bucket"], dropna=False)["probability"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    if pivot.empty:
        return pivot
    return pivot[_ordered_buckets(pivot.columns)]


def _largest_named_scenario_delta(history: pd.DataFrame, latest_time: object) -> dict[str, str] | None:
    frame = history.copy()
    times = sorted(frame["history_time"].dropna().unique())
    if len(times) < 2:
        return None
    prior_time = times[-2]
    latest = (
        frame[frame["history_time"] == latest_time]
        .groupby("scenario", dropna=False)["probability"]
        .sum()
    )
    prior = (
        frame[frame["history_time"] == prior_time]
        .groupby("scenario", dropna=False)["probability"]
        .sum()
    )
    delta = latest.subtract(prior, fill_value=0.0)
    if delta.empty:
        return None
    scenario = str(delta.abs().idxmax())
    value = float(delta.loc[scenario])
    direction = "rose" if value > 0 else "fell"
    return {
        "insight": "Largest scenario move",
        "read": f"{scenario} {direction}",
        "detail": f"{scenario} {direction} {abs(value):.1%} from the prior saved point.",
    }


def _risk_pressure_read(delta_risk_off: float, delta_transition: float) -> str:
    combined_delta = delta_risk_off + delta_transition
    if combined_delta >= 0.03:
        return "Risk pressure rising"
    if combined_delta <= -0.03:
        return "Risk pressure fading"
    return "Mostly stable"


def _ordered_horizons(horizons: list[str] | pd.Index | pd.Series) -> list[str]:
    values = [str(horizon) for horizon in horizons if pd.notna(horizon)]
    order = ["1w", "1m", "3m", "6m"]
    ordered = [horizon for horizon in order if horizon in values]
    ordered.extend(sorted(horizon for horizon in values if horizon not in order))
    return ordered


def _ordered_buckets(buckets: pd.Index | list[object]) -> list[object]:
    order = [
        "risk_off",
        "shock",
        "risk_off_then_relief",
        "transition",
        "risk_on_fragile",
        "fragile_upside",
        "risk_on",
    ]
    values = list(buckets)
    ordered = [bucket for bucket in order if bucket in values]
    ordered.extend(sorted(bucket for bucket in values if bucket not in ordered))
    return ordered


def _format_history_time(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return "n/a"
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        return timestamp.date().isoformat()
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _scenario_probability_stack_figure(scenario_lattice: pd.DataFrame) -> go.Figure:
    if scenario_lattice.empty or not {"horizon", "risk_bucket", "probability"}.issubset(
        scenario_lattice.columns
    ):
        return go.Figure()
    scoped = scenario_lattice.copy()
    scoped["probability"] = pd.to_numeric(scoped["probability"], errors="coerce")
    pivot = (
        scoped.dropna(subset=["probability"])
        .groupby(["horizon", "risk_bucket"], dropna=False)["probability"]
        .sum()
        .unstack(fill_value=0.0)
    )
    if pivot.empty:
        return go.Figure()
    horizon_order = [horizon for horizon in ["1w", "1m", "3m", "6m"] if horizon in pivot.index]
    horizon_order.extend([horizon for horizon in pivot.index if horizon not in horizon_order])
    pivot = pivot.reindex(horizon_order)
    color_map = {
        "risk_on": "#0f766e",
        "fragile_upside": "#65a30d",
        "transition": "#b7791f",
        "risk_off": "#b91c1c",
        "shock": "#7f1d1d",
    }
    figure = go.Figure()
    for bucket in pivot.columns:
        figure.add_trace(
            go.Bar(
                x=pivot.index.astype(str),
                y=pivot[bucket],
                name=str(bucket).replace("_", " ").title(),
                marker_color=color_map.get(str(bucket), "#64748b"),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Risk-Bucket Probability by Horizon",
        template="plotly_white",
        barmode="stack",
        yaxis={"title": "Probability", "tickformat": ".0%", "range": [0, 1]},
        xaxis={"title": "Horizon"},
        height=360,
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return figure


def _scenario_driver_score_figure(scenario_drivers: pd.DataFrame) -> go.Figure:
    if scenario_drivers.empty:
        return go.Figure()
    label_column = _first_existing_column(
        scenario_drivers,
        ["driver", "factor", "signal", "scenario_driver"],
    )
    score_column = _first_existing_column(
        scenario_drivers,
        ["score", "driver_score", "contribution", "value"],
    )
    if label_column is None or score_column is None:
        return go.Figure()
    drivers = scenario_drivers.copy()
    drivers[score_column] = pd.to_numeric(drivers[score_column], errors="coerce")
    drivers = drivers.dropna(subset=[score_column])
    if drivers.empty:
        return go.Figure()
    drivers = drivers.reindex(drivers[score_column].abs().sort_values(ascending=False).index).head(
        DEFAULT_SCENARIO_EXPLANATION_TOP_SCENARIOS
    )
    colors = drivers[score_column].map(lambda value: "#0f766e" if value >= 0 else "#b91c1c")
    hover_columns = [column for column in ["evidence", "read", "state"] if column in drivers]
    customdata = drivers[hover_columns] if hover_columns else None
    figure = go.Figure(
        go.Bar(
            x=drivers[score_column],
            y=drivers[label_column].astype(str),
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=_scenario_driver_hover_template(hover_columns),
        )
    )
    figure.update_layout(
        title="Top Scenario Drivers",
        template="plotly_white",
        xaxis={"title": "Driver score"},
        yaxis={"title": "", "autorange": "reversed"},
        height=360,
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
    )
    return figure


def _scenario_driver_hover_template(hover_columns: list[str]) -> str:
    template = "<b>%{y}</b><br>Driver score: %{x:.2f}"
    for index, column in enumerate(hover_columns):
        template += f"<br>{column}: %{{customdata[{index}]}}"
    return template + "<extra></extra>"


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame:
            return column
    return None


def _lead_regime_label(current_state: object) -> str:
    growth_inflation_map = getattr(current_state, "growth_inflation_map", pd.DataFrame())
    if isinstance(growth_inflation_map, pd.DataFrame) and not growth_inflation_map.empty:
        for column in ("regime", "market_regime", "dominant_regime", "cycle"):
            if column in growth_inflation_map:
                return str(growth_inflation_map[column].iloc[0])
    scenario_lattice = getattr(current_state, "scenario_lattice", pd.DataFrame())
    if isinstance(scenario_lattice, pd.DataFrame) and not scenario_lattice.empty:
        one_month = (
            scenario_lattice[scenario_lattice["horizon"].astype(str) == "1m"]
            if "horizon" in scenario_lattice
            else pd.DataFrame()
        )
        if not one_month.empty and "scenario" in one_month:
            return str(one_month.sort_values("probability", ascending=False)["scenario"].iloc[0])
    return ""
