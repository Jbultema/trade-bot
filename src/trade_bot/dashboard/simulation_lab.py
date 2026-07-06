from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import _helped_metric, _render_metric_dataframe
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _format_currency,
    _format_decimal,
    _format_percent,
)
from trade_bot.DEFAULTS import (
    DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
    DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
    DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_HORIZON_YEARS,
    DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT,
    DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
)
from trade_bot.research.approach_explorer import (
    build_approach_backtest_result,
    build_approach_catalog,
    decision_sanity_from_catalog_row,
    execution_for_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.forward_simulation import (
    ForwardSimulationConfig,
    build_regime_return_library,
    regime_mix_frame,
    scenario_probability_frame,
    simulate_regime_conditioned_paths,
    simulation_settings_frame,
    summarize_forward_simulation,
)
from trade_bot.research.strategy_outcome_utility import (
    OutcomeBootstrapConfig,
    add_outcome_frontier_flags,
    bootstrap_outcome_paths,
    contribution_periods_per_year,
    enrich_strategy_outcome_utility,
    summarize_bootstrap_outcomes,
    terminal_wealth_from_cagr,
)


def _render_simulation_lab(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> None:
    st.subheader("Simulation Lab")
    st.caption(
        "Forward-looking planning workbench. This section separates deterministic CAGR math, "
        "historical sequence risk, and scenario-conditioned forward paths so the future-state "
        "engine has its own inspection surface."
    )

    _render_simulation_planning_cards()
    tabs = st.tabs(
        [
            "Future-State Map",
            "Strategy Simulations",
            "Interpretability",
        ]
    )
    scenario_source = _scenario_source_frame(baseline_run)
    probabilities = scenario_probability_frame(scenario_source)

    with tabs[0]:
        _render_future_state_map(baseline_run, scenario_source, probabilities)

    selected_strategy, selected_scorecard, selected_result = _selected_simulation_strategy(
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    with tabs[1]:
        _render_strategy_simulations(
            selected_strategy=selected_strategy,
            selected_scorecard=selected_scorecard,
            selected_result=selected_result,
            scenario_source=scenario_source,
        )
    with tabs[2]:
        _render_simulation_interpretability(
            selected_strategy=selected_strategy,
            selected_result=selected_result,
            scenario_source=scenario_source,
            probabilities=probabilities,
        )


def _render_simulation_planning_cards() -> None:
    contribution_periods = contribution_periods_per_year(DEFAULT_OUTCOME_CONTRIBUTION_TIMING)
    contribution_amount = DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION / contribution_periods
    cadence = DEFAULT_OUTCOME_CONTRIBUTION_TIMING.replace("_", " ").title()
    cols = st.columns(5)
    _helped_metric(cols[0], "Starting Account", _format_currency(DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE))
    _helped_metric(cols[1], "Annual Contribution", f"{_format_currency(DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION)} / yr")
    _helped_metric(cols[2], "Contribution Cadence", f"{_format_currency(contribution_amount)} {cadence}")
    _helped_metric(cols[3], "Horizon", f"{DEFAULT_OUTCOME_HORIZON_YEARS} years")
    _helped_metric(
        cols[4],
        "Soft / Hard DD",
        (
            f"{abs(DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT):.0%} / "
            f"{abs(DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT):.0%}"
        ),
    )


def _render_future_state_map(
    baseline_run: BaselineRun,
    scenario_source: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> None:
    st.markdown("**Future-State Simulation Map**")
    st.caption(
        "This is the aggregate state input to the forward engine. Detailed current scenarios are "
        "mapped into broad empirical return buckets, then used to bias simulated starting states "
        "and future regime transitions."
    )
    risk_off = _probability_for_bucket(probabilities, "risk_off")
    transition = _probability_for_bucket(probabilities, "transition")
    fragile = _probability_for_bucket(probabilities, "risk_on_fragile")
    risk_on = _probability_for_bucket(probabilities, "risk_on")
    top = _top_scenario_row(scenario_source)
    cols = st.columns(5)
    _helped_metric(cols[0], "Risk-Off", _format_percent(risk_off))
    _helped_metric(cols[1], "Transition", _format_percent(transition))
    _helped_metric(cols[2], "Fragile Risk-On", _format_percent(fragile))
    _helped_metric(cols[3], "Broad Risk-On", _format_percent(risk_on))
    _helped_metric(cols[4], "Top Scenario", str(top.get("scenario", "n/a"))[:38])

    chart_cols = st.columns([1.05, 1.0])
    with chart_cols[0]:
        st.plotly_chart(_scenario_probability_figure(probabilities), use_container_width=True)
    with chart_cols[1]:
        st.caption("Current scenario records")
        scenario_columns = [
            "horizon",
            "rank",
            "scenario",
            "probability",
            "risk_bucket",
            "expected_bot_posture",
            "preferred_exposure",
            "confirmation",
            "off_ramp",
        ]
        available = [column for column in scenario_columns if column in scenario_source]
        if available:
            _render_metric_dataframe(
                _display_metrics(scenario_source[available].head(12)),
                hide_index=True,
            )
        else:
            st.write("No detailed scenario rows are available in the loaded run.")

    with st.expander("Simulation settings", expanded=False):
        st.caption("Regime-conditioned forward simulation settings")
        _render_metric_dataframe(simulation_settings_frame(ForwardSimulationConfig()), hide_index=True)
        st.caption("Current state context")
        _render_metric_dataframe(
            pd.DataFrame(
                [
                    {
                        "input": "market_date",
                        "value": baseline_run.current_state.market_date,
                        "meaning": "Date of the market snapshot used by the current scenario map.",
                    },
                    {
                        "input": "risk_status",
                        "value": str(baseline_run.current_state.risk_status).upper(),
                        "meaning": "Current risk posture that informs scenario interpretation.",
                    },
                    {
                        "input": "risk_score",
                        "value": _format_decimal(baseline_run.current_state.risk_score),
                        "meaning": "Current composite risk score from the operating run.",
                    },
                ]
            ),
            hide_index=True,
        )


def _selected_simulation_strategy(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
) -> tuple[str | None, pd.Series | None, BacktestResult | None]:
    options = _strategy_option_frame(bot_config, experiment_scorecards)
    if options.empty:
        st.warning("No reconstructable strategies are available for simulation.")
        return None, None, None

    labels = options["simulation_label"].tolist()
    selected_label = st.selectbox(
        "Strategy to simulate",
        labels,
        key="simulation_lab_selected_strategy",
    )
    row = options[options["simulation_label"] == selected_label].iloc[0]
    strategy_name = str(row["strategy"])
    result = _result_for_strategy(
        strategy_name,
        bot_config=bot_config,
        baseline_run=baseline_run,
    )
    return strategy_name, row, result


def _render_strategy_simulations(
    *,
    selected_strategy: str | None,
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult | None,
    scenario_source: pd.DataFrame,
) -> None:
    st.markdown("**Selected-Strategy Simulation Readout**")
    st.caption(
        "Use this to ask whether a candidate's wealth range and drawdown range are tolerable, "
        "not just whether its historical point estimate looked good."
    )
    if selected_strategy is None or selected_result is None:
        st.info("Pick a strategy with reconstructable history to see simulation paths.")
        return

    daily_returns = _daily_returns(selected_result)
    if daily_returns.empty:
        st.warning("The selected strategy has no usable daily return history.")
        return
    if len(daily_returns) < 30:
        st.warning(
            "The selected strategy has fewer than 30 daily observations. "
            "Simulation paths are suppressed until the candidate has enough history "
            "to produce a useful distribution."
        )
        return

    deterministic_wealth = _deterministic_wealth(selected_scorecard, selected_result)
    bootstrap_paths = _cached_bootstrap_paths(_returns_tuple(selected_result))
    bootstrap_summary = summarize_bootstrap_outcomes(bootstrap_paths)
    forward_paths = _cached_regime_forward_paths(
        _returns_tuple(selected_result),
        _scenario_records(scenario_source),
    )
    forward_summary = summarize_forward_simulation(
        forward_paths,
        config=ForwardSimulationConfig(),
    )

    cols = st.columns(6)
    _helped_metric(cols[0], "Deterministic 15Y", _format_currency(deterministic_wealth))
    _helped_metric(
        cols[1],
        "Bootstrap Median",
        _format_currency(bootstrap_summary.get("terminal_wealth_p50")),
    )
    _helped_metric(
        cols[2],
        "Forward Median",
        _format_currency(forward_summary.get("terminal_wealth_p50")),
    )
    _helped_metric(
        cols[3],
        "Forward P10",
        _format_currency(forward_summary.get("terminal_wealth_p10")),
    )
    _helped_metric(
        cols[4],
        "Median Forward DD",
        _format_percent(forward_summary.get("max_drawdown_p50")),
    )
    _helped_metric(
        cols[5],
        "Severe DD Prob",
        _format_percent(forward_summary.get("severe_drawdown_probability")),
    )

    st.info(_simulation_plain_english_read(selected_strategy, bootstrap_summary, forward_summary))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            _simulation_histogram(
                bootstrap_paths,
                column="terminal_wealth",
                title="Historical sequence bootstrap",
                xaxis_title="Terminal wealth",
                color="#0f766e",
                hovertemplate="Terminal wealth %{x:$,.0f}<br>Paths %{y}<extra></extra>",
            ),
            use_container_width=True,
        )
    with chart_cols[1]:
        st.plotly_chart(
            _simulation_histogram(
                forward_paths,
                column="terminal_wealth",
                title="Regime-conditioned forward paths",
                xaxis_title="Terminal wealth",
                color="#2563eb",
                hovertemplate="Terminal wealth %{x:$,.0f}<br>Paths %{y}<extra></extra>",
            ),
            use_container_width=True,
        )

    st.caption("Outcome distribution summary")
    _render_metric_dataframe(
        _display_metrics(
            pd.DataFrame(
                [
                    _summary_row("Deterministic CAGR", deterministic_wealth, None),
                    _summary_row("Historical bootstrap", None, bootstrap_summary),
                    _summary_row("Regime-conditioned", None, forward_summary),
                ]
            )
        ),
        hide_index=True,
    )


def _render_simulation_interpretability(
    *,
    selected_strategy: str | None,
    selected_result: BacktestResult | None,
    scenario_source: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> None:
    st.markdown("**Simulation Interpretability**")
    st.caption(
        "This is the audit layer for why the simulation says what it says: scenario probabilities, "
        "historical regime labels, sampled return libraries, and model limitations."
    )
    model_rows = [
        {
            "layer": "Deterministic CAGR",
            "what_it_uses": "One historical CAGR point estimate plus configured contributions.",
            "best_for": "Fast frontier ranking and simple benchmark comparison.",
            "main_failure_mode": "Ignores sequencing, volatility, and drawdown timing.",
        },
        {
            "layer": "Historical block bootstrap",
            "what_it_uses": "Resampled blocks of the selected strategy's realized daily returns.",
            "best_for": "Sequence risk, drawdown persistence, and terminal-wealth range.",
            "main_failure_mode": "Assumes the historical return mix remains representative.",
        },
        {
            "layer": "Regime-conditioned forward paths",
            "what_it_uses": "Current scenario probabilities plus historical regime-labeled return blocks.",
            "best_for": "Current-state-aware planning ranges and downside/tail inspection.",
            "main_failure_mode": "Regime labels are coarse and scenario probabilities are not forecasts.",
        },
    ]
    _render_metric_dataframe(pd.DataFrame(model_rows), hide_index=True)

    cols = st.columns(2)
    with cols[0]:
        st.caption("Current scenario probability bridge")
        _render_metric_dataframe(_display_metrics(probabilities), hide_index=True)
    with cols[1]:
        st.caption("Scenario records feeding the bridge")
        available = [
            column
            for column in [
                "horizon",
                "scenario",
                "probability",
                "risk_bucket",
                "expected_bot_posture",
            ]
            if column in scenario_source
        ]
        if available:
            _render_metric_dataframe(_display_metrics(scenario_source[available].head(10)), hide_index=True)
        else:
            st.write("No detailed scenario records available.")

    if selected_strategy is None or selected_result is None:
        return

    daily_returns = _daily_returns(selected_result)
    if len(daily_returns) < 30:
        st.info("Selected strategy history is too short for a meaningful regime-return library.")
        return
    library = build_regime_return_library(
        daily_returns,
        config=ForwardSimulationConfig(),
    )
    if library.empty:
        return
    regime_summary = (
        library.groupby("regime")["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_daily_return", "std": "daily_volatility"})
    )
    regime_summary["annualized_mean_return"] = regime_summary["mean_daily_return"] * 252
    regime_summary["annualized_volatility"] = regime_summary["daily_volatility"] * (252**0.5)
    st.caption(f"Historical return library for {selected_strategy}")
    _render_metric_dataframe(_display_metrics(regime_summary), hide_index=True)

    paths = _cached_regime_forward_paths(_returns_tuple(selected_result), _scenario_records(scenario_source))
    mix = regime_mix_frame(paths)
    if not mix.empty:
        st.caption("Average simulated regime mix")
        _render_metric_dataframe(_display_metrics(mix), hide_index=True)


def _strategy_option_frame(bot_config: Any, experiment_scorecards: pd.DataFrame) -> pd.DataFrame:
    if not experiment_scorecards.empty and {"strategy", "cagr", "max_drawdown"}.issubset(
        experiment_scorecards.columns
    ):
        frame = add_outcome_frontier_flags(
            enrich_strategy_outcome_utility(experiment_scorecards)
        ).copy()
        if "research_status" in frame:
            active = frame[~frame["research_status"].astype(str).eq("pruned_dead_end")].copy()
            if not active.empty:
                frame = active
        frame = frame.sort_values("growth_constrained_utility_score", ascending=False).head(80)
        frame["simulation_label"] = frame.apply(_scorecard_option_label, axis=1)
        return frame

    catalog = build_approach_catalog(bot_config)
    if catalog.empty or "strategy" not in catalog:
        return pd.DataFrame()
    output = catalog.copy()
    output = output[output.get("source", pd.Series("", index=output.index)).astype(str).eq("baseline")]
    if output.empty:
        output = catalog.head(25).copy()
    output["simulation_label"] = output.apply(
        lambda row: f"{row.get('display_name', row.get('strategy', 'Strategy'))} | configured",
        axis=1,
    )
    return output.reset_index(drop=True)


def _scorecard_option_label(row: pd.Series) -> str:
    return (
        f"{row.get('display_name', row.get('strategy', 'Strategy'))} | "
        f"utility {_format_decimal(row.get('growth_constrained_utility_score'))} | "
        f"CAGR {_format_percent(row.get('cagr'))} | "
        f"DD {_format_percent(row.get('max_drawdown'))}"
    )


def _result_for_strategy(
    strategy_name: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
) -> BacktestResult | None:
    catalog = build_approach_catalog(bot_config)
    if catalog.empty or "strategy" not in catalog:
        return baseline_run.results.get(strategy_name)
    matches = catalog[catalog["strategy"].astype(str).eq(strategy_name)]
    if matches.empty:
        return baseline_run.results.get(strategy_name)
    row = matches.iloc[0]
    try:
        strategy = strategy_from_catalog_row(row)
        execution = execution_for_catalog_row(row, bot_config.execution)
        result, _ = build_approach_backtest_result(
            baseline_run.prices,
            strategy,
            execution,
            scenario_sizing=scenario_sizing_from_catalog_row(row),
            future_state_model=future_state_model_from_catalog_row(row),
            strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
            decision_sanity=decision_sanity_from_catalog_row(row),
            name=strategy_name,
        )
    except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return baseline_run.results.get(strategy_name)
    return result


def _scenario_source_frame(baseline_run: BaselineRun) -> pd.DataFrame:
    lattice = baseline_run.current_state.scenario_lattice.copy()
    if not lattice.empty and {"risk_bucket", "probability"}.issubset(lattice.columns):
        one_month = lattice[lattice.get("horizon", pd.Series("", index=lattice.index)).astype(str).eq("1m")]
        return one_month.copy() if not one_month.empty else lattice
    outlook = baseline_run.current_state.scenario_outlook.copy()
    if not outlook.empty:
        return outlook
    return pd.DataFrame()


def _probability_for_bucket(probabilities: pd.DataFrame, bucket: str) -> float | None:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return None
    matches = probabilities[probabilities["regime"].astype(str).eq(bucket)]
    if matches.empty:
        return 0.0
    return _safe_float(matches.iloc[0]["probability"])


def _top_scenario_row(scenario_source: pd.DataFrame) -> pd.Series:
    if scenario_source.empty:
        return pd.Series(dtype=object)
    if "probability" in scenario_source:
        values = pd.to_numeric(scenario_source["probability"], errors="coerce").fillna(-1.0)
        return scenario_source.loc[values.idxmax()]
    return scenario_source.iloc[0]


def _scenario_probability_figure(probabilities: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=probabilities["regime"],
            y=probabilities["probability"],
            marker={"color": ["#b91c1c", "#f59e0b", "#2563eb", "#0f766e"][: len(probabilities)]},
            hovertemplate="%{x}<br>Probability %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        yaxis_tickformat=".0%",
        yaxis_title="Probability",
        xaxis_title="Simulation bucket",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


def _deterministic_wealth(
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult,
) -> float | None:
    cagr = _safe_float(selected_scorecard.get("cagr")) if selected_scorecard is not None else None
    if cagr is None:
        cagr = _cagr_from_result(selected_result)
    if cagr is None:
        return None
    return float(
        terminal_wealth_from_cagr(
            cagr,
            years=DEFAULT_OUTCOME_HORIZON_YEARS,
            starting_account_value=DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE,
            annual_contribution=DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION,
            contribution_timing=DEFAULT_OUTCOME_CONTRIBUTION_TIMING,
        ).iloc[0]
    )


def _daily_returns(result: BacktestResult) -> pd.Series:
    return (
        result.equity.pct_change()
        .replace([float("inf"), float("-inf")], pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
    )


def _returns_tuple(result: BacktestResult) -> tuple[float, ...]:
    return tuple(float(value) for value in _daily_returns(result).to_numpy(dtype=float))


def _scenario_records(scenario_source: pd.DataFrame) -> tuple[tuple[str, float], ...]:
    probabilities = scenario_probability_frame(scenario_source)
    return tuple(
        (str(row["regime"]), float(row["probability"])) for _, row in probabilities.iterrows()
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_bootstrap_paths(return_values: tuple[float, ...]) -> pd.DataFrame:
    return bootstrap_outcome_paths(pd.Series(return_values, dtype=float), config=OutcomeBootstrapConfig())


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_regime_forward_paths(
    return_values: tuple[float, ...],
    scenario_records: tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    scenario_frame = pd.DataFrame(
        [{"risk_bucket": regime, "probability": probability} for regime, probability in scenario_records]
    )
    return simulate_regime_conditioned_paths(
        pd.Series(return_values, dtype=float),
        scenario_outlook=scenario_frame,
        config=ForwardSimulationConfig(),
    )


def _simulation_plain_english_read(
    selected_strategy: str,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
) -> str:
    forward_p10 = _format_currency(forward_summary.get("terminal_wealth_p10"))
    forward_p50 = _format_currency(forward_summary.get("terminal_wealth_p50"))
    forward_p90 = _format_currency(forward_summary.get("terminal_wealth_p90"))
    severe = _format_percent(forward_summary.get("severe_drawdown_probability"))
    bootstrap_median = _format_currency(bootstrap_summary.get("terminal_wealth_p50"))
    return (
        f"{selected_strategy}: the historical bootstrap median is {bootstrap_median}. "
        f"The scenario-conditioned central range is {forward_p10} to {forward_p90}, "
        f"with median {forward_p50} and severe-drawdown probability {severe}. "
        "Read this as a planning distribution, not as a point forecast."
    )


def _summary_row(
    label: str,
    deterministic_wealth: float | None,
    summary: dict[str, object] | None,
) -> dict[str, object]:
    if summary is None:
        return {
            "model": label,
            "terminal_wealth_p10": deterministic_wealth,
            "terminal_wealth_p50": deterministic_wealth,
            "terminal_wealth_p90": deterministic_wealth,
            "max_drawdown_p50": None,
            "ulcer_index_p50": None,
            "severe_drawdown_probability": None,
            "capital_impairment_probability": None,
        }
    return {
        "model": label,
        "terminal_wealth_p10": summary.get("terminal_wealth_p10"),
        "terminal_wealth_p50": summary.get("terminal_wealth_p50"),
        "terminal_wealth_p90": summary.get("terminal_wealth_p90"),
        "max_drawdown_p50": summary.get("max_drawdown_p50"),
        "ulcer_index_p50": summary.get("ulcer_index_p50"),
        "severe_drawdown_probability": summary.get("severe_drawdown_probability"),
        "capital_impairment_probability": summary.get("capital_impairment_probability"),
    }


def _simulation_histogram(
    paths: pd.DataFrame,
    *,
    column: str,
    title: str,
    xaxis_title: str,
    color: str,
    hovertemplate: str,
) -> go.Figure:
    fig = go.Figure()
    if column in paths and not paths.empty:
        fig.add_trace(
            go.Histogram(
                x=paths[column],
                nbinsx=40,
                marker={"color": color, "line": {"color": "#0f172a", "width": 0.5}},
                hovertemplate=hovertemplate,
            )
        )
    fig.update_layout(
        height=300,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Path count",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _cagr_from_result(result: BacktestResult) -> float | None:
    if result.equity.empty or len(result.equity) < 2:
        return None
    start = _safe_float(result.equity.iloc[0])
    end = _safe_float(result.equity.iloc[-1])
    if start is None or end is None or start <= 0.0:
        return None
    years = max(len(result.equity) / 252.0, 1.0 / 252.0)
    return (end / start) ** (1.0 / years) - 1.0


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
