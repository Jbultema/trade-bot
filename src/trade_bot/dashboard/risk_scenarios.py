from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.DEFAULTS import DEFAULT_SCENARIO_EXPLANATION_TOP_SCENARIOS
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.operating_exposure import (
    aggregate_beta_adjusted_spy_delta,
    build_beta_adjusted_delta_table,
    build_sleeve_exposure_table,
    build_tactical_matrix,
    weights_from_position_plan,
)


def _render_risk_and_scenarios(baseline_run: BaselineRun) -> None:
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
