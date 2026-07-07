from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import (
    _clearable_selectbox,
    _helped_metric,
    _render_metric_dataframe,
)
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _escape_markdown_dollars,
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
    DEFAULT_SIMULATION_REFERENCE_STRATEGIES,
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
    REGIME_BUCKETS,
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
    _render_simulation_method_guide()
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
            baseline_run=baseline_run,
            scenario_source=scenario_source,
        )
    with tabs[2]:
        _render_simulation_interpretability(
            selected_strategy=selected_strategy,
            selected_scorecard=selected_scorecard,
            selected_result=selected_result,
            baseline_run=baseline_run,
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


def _render_simulation_method_guide() -> None:
    with st.expander("How to read the simulation forms", expanded=False):
        _render_metric_dataframe(
            pd.DataFrame(
                [
                    {
                        "form": "Deterministic 15Y",
                        "what_it_does": "Applies historical CAGR to the configured starting balance and monthly contributions.",
                        "best_use": "Fast reference point for the growth frontier.",
                        "do_not_use_for": "Drawdown, path risk, or confidence.",
                    },
                    {
                        "form": "Historical bootstrap",
                        "what_it_does": "Resamples historical daily-return blocks from the selected strategy.",
                        "best_use": "Sequence risk and drawdown ranges if the future looks like the strategy's own past.",
                        "do_not_use_for": "A current-state-aware forecast.",
                    },
                    {
                        "form": "Regime-conditioned forward paths",
                        "what_it_does": "Samples historical return blocks by regime, then biases starting and transition states using today's scenario map.",
                        "best_use": "Forward planning ranges when current risk-off/transition/risk-on probabilities matter.",
                        "do_not_use_for": "A promise that the specific regime sequence will happen.",
                    },
                    {
                        "form": "Reference overlays",
                        "what_it_does": "Runs the same simulation logic on SPY/QQQ reference portfolios when available.",
                        "best_use": "Judging whether the selected strategy earns its complexity versus doing nothing.",
                        "do_not_use_for": "Declaring a strategy superior without Monitoring and out-of-sample evidence.",
                    },
                ]
            ),
            hide_index=True,
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
    selected_label = _clearable_selectbox(
        "Strategy to simulate",
        labels,
        key="simulation_lab_selected_strategy",
        placeholder="Search simulation strategies...",
    )
    if selected_label is None:
        st.info("Choose a strategy to run forward simulations.")
        return None, None, None
    row = options[options["simulation_label"] == selected_label].iloc[0]
    strategy_name = str(row["strategy"])
    result = _result_for_strategy(
        strategy_name,
        bot_config=bot_config,
        baseline_run=baseline_run,
    )
    return strategy_name, row, result


def _reference_option_frame(
    baseline_run: BaselineRun,
    selected_strategy: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for strategy_name, label in DEFAULT_SIMULATION_REFERENCE_STRATEGIES:
        if strategy_name == selected_strategy:
            continue
        if strategy_name in baseline_run.results:
            rows.append({"strategy": strategy_name, "label": label})
    return pd.DataFrame(rows)


def _reference_simulations(
    *,
    baseline_run: BaselineRun,
    reference_options: pd.DataFrame,
    selected_reference_labels: list[str],
    scenario_source: pd.DataFrame,
) -> list[dict[str, object]]:
    if reference_options.empty or not selected_reference_labels:
        return []
    selected_options = reference_options[
        reference_options["label"].astype(str).isin(selected_reference_labels)
    ].copy()
    colors = ("#0f766e", "#f97316", "#7c3aed", "#64748b")
    simulations: list[dict[str, object]] = []
    for position, (_, row) in enumerate(selected_options.iterrows()):
        strategy_name = str(row["strategy"])
        result = baseline_run.results.get(strategy_name)
        if result is None:
            continue
        returns = _returns_tuple(result)
        if len(returns) < 30:
            continue
        bootstrap_paths = _cached_bootstrap_paths(returns)
        forward_paths = _cached_regime_forward_paths(returns, _scenario_records(scenario_source))
        simulations.append(
            {
                "strategy": strategy_name,
                "label": str(row["label"]),
                "deterministic_wealth": _deterministic_wealth(None, result),
                "bootstrap_paths": bootstrap_paths,
                "bootstrap_summary": summarize_bootstrap_outcomes(bootstrap_paths),
                "forward_paths": forward_paths,
                "forward_summary": summarize_forward_simulation(
                    forward_paths,
                    config=ForwardSimulationConfig(),
                ),
                "result": result,
                "color": colors[position % len(colors)],
            }
        )
    return simulations


def _render_strategy_simulations(
    *,
    selected_strategy: str | None,
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult | None,
    baseline_run: BaselineRun,
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
    reference_options = _reference_option_frame(baseline_run, selected_strategy)
    selected_reference_labels: list[str] = []
    if not reference_options.empty:
        selected_reference_labels = st.multiselect(
            "Reference overlays",
            reference_options["label"].tolist(),
            default=reference_options["label"].head(2).tolist(),
            help=(
                "Compare this strategy against major do-nothing references using the same "
                "bootstrap and regime-conditioned simulation settings."
            ),
            key="simulation_lab_reference_overlays",
        )
    reference_simulations = _reference_simulations(
        baseline_run=baseline_run,
        reference_options=reference_options,
        selected_reference_labels=selected_reference_labels,
        scenario_source=scenario_source,
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

    st.info(
        _escape_markdown_dollars(
            _simulation_plain_english_read(
                selected_strategy,
                bootstrap_summary,
                forward_summary,
                reference_simulations=reference_simulations,
            )
        )
    )

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
            _simulation_overlay_histogram(
                [
                    {
                        "label": selected_strategy,
                        "paths": forward_paths,
                        "color": "#2563eb",
                    },
                    *[
                        {
                            "label": str(reference["label"]),
                            "paths": reference["forward_paths"],
                            "color": reference["color"],
                        }
                        for reference in reference_simulations
                    ],
                ],
                column="terminal_wealth",
                title="Regime-conditioned forward paths vs references",
                xaxis_title="Terminal wealth",
                value_label="Terminal wealth",
                value_format="currency",
            ),
            use_container_width=True,
        )

    drawdown_cols = st.columns(2)
    with drawdown_cols[0]:
        st.plotly_chart(
            _simulation_histogram(
                forward_paths,
                column="max_drawdown",
                title="Selected strategy forward drawdown distribution",
                xaxis_title="Max drawdown",
                color="#dc2626",
                hovertemplate="Max drawdown %{x:.1%}<br>Paths %{y}<extra></extra>",
            ),
            use_container_width=True,
        )
    with drawdown_cols[1]:
        st.plotly_chart(
            _simulation_overlay_histogram(
                [
                    {
                        "label": selected_strategy,
                        "paths": forward_paths,
                        "color": "#dc2626",
                    },
                    *[
                        {
                            "label": str(reference["label"]),
                            "paths": reference["forward_paths"],
                            "color": reference["color"],
                        }
                        for reference in reference_simulations
                    ],
                ],
                column="max_drawdown",
                title="Forward drawdown paths vs references",
                xaxis_title="Max drawdown",
                value_label="Max drawdown",
                value_format="percent",
            ),
            use_container_width=True,
        )

    if reference_simulations:
        st.caption("Reference portfolio comparison")
        selected_forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
        comparison_rows = [
            _simulation_comparison_row(
                label="Selected strategy",
                strategy=selected_strategy,
                deterministic_wealth=deterministic_wealth,
                bootstrap_summary=bootstrap_summary,
                forward_summary=forward_summary,
                selected_forward_median=selected_forward_median,
            )
        ]
        comparison_rows.extend(
            _simulation_comparison_row(
                label=str(reference["label"]),
                strategy=str(reference["strategy"]),
                deterministic_wealth=reference["deterministic_wealth"],
                bootstrap_summary=reference["bootstrap_summary"],
                forward_summary=reference["forward_summary"],
                selected_forward_median=selected_forward_median,
            )
            for reference in reference_simulations
        )
        _render_metric_dataframe(_display_metrics(pd.DataFrame(comparison_rows)), hide_index=True)
    elif reference_options.empty:
        st.caption(
            "SPY/QQQ reference overlays are unavailable in this loaded run because the snapshot "
            "does not include buy-and-hold reference results."
        )

    st.caption("Drawdown distribution detail")
    drawdown_rows = [
        _drawdown_distribution_row(
            label="Selected bootstrap",
            strategy=selected_strategy,
            paths=bootstrap_paths,
        ),
        _drawdown_distribution_row(
            label="Selected forward",
            strategy=selected_strategy,
            paths=forward_paths,
        ),
    ]
    for reference in reference_simulations:
        drawdown_rows.append(
            _drawdown_distribution_row(
                label=f"{reference['label']} forward",
                strategy=str(reference["strategy"]),
                paths=reference["forward_paths"],
            )
        )
    _render_metric_dataframe(_display_metrics(pd.DataFrame(drawdown_rows)), hide_index=True)

    st.caption("Simulation calibration and historical resemblance")
    validation_rows = [
        _simulation_validation_row(
            label="Selected strategy",
            strategy=selected_strategy,
            result=selected_result,
            deterministic_wealth=deterministic_wealth,
            bootstrap_summary=bootstrap_summary,
            forward_summary=forward_summary,
        )
    ]
    for reference in reference_simulations:
        result = reference.get("result")
        if isinstance(result, BacktestResult):
            validation_rows.append(
                _simulation_validation_row(
                    label=str(reference["label"]),
                    strategy=str(reference["strategy"]),
                    result=result,
                    deterministic_wealth=reference["deterministic_wealth"],
                    bootstrap_summary=reference["bootstrap_summary"],
                    forward_summary=reference["forward_summary"],
                )
            )
    _render_metric_dataframe(_display_metrics(pd.DataFrame(validation_rows)), hide_index=True)

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


def _drawdown_distribution_row(
    *,
    label: str,
    strategy: str,
    paths: pd.DataFrame,
) -> dict[str, object]:
    drawdowns = _numeric_path_column(paths, "max_drawdown")
    ulcers = _numeric_path_column(paths, "ulcer_index")
    return {
        "simulation": label,
        "strategy": strategy,
        "paths": int(paths.shape[0]),
        "max_drawdown_p10": _series_quantile(drawdowns, 0.10),
        "max_drawdown_p25": _series_quantile(drawdowns, 0.25),
        "max_drawdown_p50": _series_quantile(drawdowns, 0.50),
        "max_drawdown_p75": _series_quantile(drawdowns, 0.75),
        "max_drawdown_p90": _series_quantile(drawdowns, 0.90),
        "ulcer_index_p50": _series_quantile(ulcers, 0.50),
        "breach_soft_band": _series_mean(drawdowns <= DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT),
        "breach_hard_band": _series_mean(drawdowns <= DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT),
    }


def _simulation_validation_row(
    *,
    label: str,
    strategy: str,
    result: BacktestResult,
    deterministic_wealth: float | None,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
) -> dict[str, object]:
    historical = _historical_result_metrics(result)
    bootstrap_median = _safe_float(bootstrap_summary.get("terminal_wealth_p50"))
    forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    forward_dd = _safe_float(forward_summary.get("max_drawdown_p50"))
    forward_ulcer = _safe_float(forward_summary.get("ulcer_index_p50"))
    deterministic_delta = _relative_delta(forward_median, deterministic_wealth)
    drawdown_delta = (
        forward_dd - historical["max_drawdown"]
        if forward_dd is not None and historical["max_drawdown"] is not None
        else None
    )
    ulcer_delta = (
        forward_ulcer - historical["ulcer_index"]
        if forward_ulcer is not None and historical["ulcer_index"] is not None
        else None
    )
    return {
        "portfolio": label,
        "strategy": strategy,
        "observed_years": historical["observed_years"],
        "historical_cagr": historical["cagr"],
        "historical_max_drawdown": historical["max_drawdown"],
        "historical_ulcer_index": historical["ulcer_index"],
        "deterministic_15y": deterministic_wealth,
        "bootstrap_median_15y": bootstrap_median,
        "forward_median_15y": forward_median,
        "forward_vs_deterministic": deterministic_delta,
        "forward_median_drawdown": forward_dd,
        "forward_minus_historical_dd": drawdown_delta,
        "forward_median_ulcer": forward_ulcer,
        "forward_minus_historical_ulcer": ulcer_delta,
        "calibration_read": _simulation_calibration_read(
            deterministic_delta=deterministic_delta,
            drawdown_delta=drawdown_delta,
            ulcer_delta=ulcer_delta,
        ),
    }


def _render_simulation_insight_cards(cards: list[dict[str, str]]) -> None:
    st.markdown(
        '<div class="operating-grid">'
        + "".join(_simulation_insight_card_html(card) for card in cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _simulation_insight_card_html(card: dict[str, str]) -> str:
    tone = card.get("tone", "")
    tone_class = f" operating-card-{html.escape(tone)}" if tone else ""
    return (
        f'<div class="operating-card{tone_class}">'
        f'<p class="operating-label">{html.escape(card["label"])}</p>'
        f'<p class="operating-answer">{html.escape(card["answer"])}</p>'
        f'<p class="operating-detail">{html.escape(card["detail"])}</p>'
        "</div>"
    )


def _simulation_interpretability_cards(
    *,
    strategy: str,
    validation_row: dict[str, object],
    drawdown_row: dict[str, object],
    forward_summary: dict[str, object],
    reference_edges: list[dict[str, object]],
    probabilities: pd.DataFrame,
) -> list[dict[str, str]]:
    calibration_read = str(validation_row.get("calibration_read", "insufficient_history"))
    reference_edge = _best_reference_edge(reference_edges)
    p10_wealth = _safe_float(forward_summary.get("terminal_wealth_p10"))
    p50_wealth = _safe_float(forward_summary.get("terminal_wealth_p50"))
    p90_wealth = _safe_float(forward_summary.get("terminal_wealth_p90"))
    median_dd = _safe_float(drawdown_row.get("max_drawdown_p50"))
    severe_prob = _safe_float(forward_summary.get("severe_drawdown_probability"))
    hard_breach = _safe_float(drawdown_row.get("breach_hard_band"))
    risk_off = _probability_sum(probabilities, ("risk_off",))
    transition = _probability_sum(probabilities, ("transition",))
    fragile = _probability_sum(probabilities, ("risk_on_fragile",))
    top_regime, top_probability = _top_probability_label(probabilities)
    return [
        {
            "tone": _calibration_tone(calibration_read),
            "label": "Past Resemblance",
            "answer": _calibration_answer(calibration_read),
            "detail": (
                f"Forward median is {_format_percent(validation_row.get('forward_vs_deterministic'))} "
                "versus deterministic CAGR math; median drawdown delta is "
                f"{_format_percent(validation_row.get('forward_minus_historical_dd'))}; "
                f"ulcer delta is {_format_percent(validation_row.get('forward_minus_historical_ulcer'))}."
            ),
        },
        {
            "tone": _reference_edge_tone(reference_edge),
            "label": "Reference Edge",
            "answer": _reference_edge_answer(reference_edge),
            "detail": (
                f"{strategy} forward range is {_format_currency(p10_wealth)} to "
                f"{_format_currency(p90_wealth)}, with median {_format_currency(p50_wealth)}. "
                f"{_reference_edge_detail(reference_edge)}"
            ),
        },
        {
            "tone": _drawdown_tone(median_dd, severe_prob, hard_breach),
            "label": "Drawdown Pain",
            "answer": f"Median {_format_percent(median_dd)}; hard breach {_format_percent(hard_breach)}",
            "detail": (
                f"The simulated median max drawdown is {_format_percent(median_dd)}. "
                f"Severe drawdown probability is {_format_percent(severe_prob)}; "
                f"hard-band breach share is {_format_percent(hard_breach)}."
            ),
        },
        {
            "tone": "warning" if risk_off + transition >= 0.55 else "success",
            "label": "Scenario Tilt",
            "answer": f"{top_regime} leads at {_format_percent(top_probability)}",
            "detail": (
                f"Current scenario bridge gives risk-off {_format_percent(risk_off)}, "
                f"transition {_format_percent(transition)}, fragile risk-on {_format_percent(fragile)}. "
                "The forward engine uses this to bias starting and transition regimes."
            ),
        },
    ]


def _simulation_verdict_rows(
    *,
    selected_strategy: str,
    validation_row: dict[str, object],
    drawdown_row: dict[str, object],
    forward_summary: dict[str, object],
    reference_edges: list[dict[str, object]],
    probabilities: pd.DataFrame,
) -> list[dict[str, object]]:
    top_regime, top_probability = _top_probability_label(probabilities)
    best_edge = _best_reference_edge(reference_edges)
    return [
        {
            "question": "Do simulated futures resemble the tested past?",
            "read": _calibration_answer(str(validation_row.get("calibration_read", "insufficient_history"))),
            "evidence": (
                "Forward vs deterministic "
                f"{_format_percent(validation_row.get('forward_vs_deterministic'))}; "
                "forward DD minus historical DD "
                f"{_format_percent(validation_row.get('forward_minus_historical_dd'))}; "
                "forward ulcer minus historical ulcer "
                f"{_format_percent(validation_row.get('forward_minus_historical_ulcer'))}."
            ),
            "implication": "Use this to judge whether the simulation is roughly past-like or stress-shifted.",
        },
        {
            "question": "Does the strategy earn its complexity versus references?",
            "read": _reference_edge_answer(best_edge),
            "evidence": _reference_edge_detail(best_edge),
            "implication": (
                f"{selected_strategy} should be compared against simple references before it is "
                "treated as an operational improvement."
            ),
        },
        {
            "question": "Where is the drawdown pain?",
            "read": (
                f"Median max drawdown {_format_percent(drawdown_row.get('max_drawdown_p50'))}; "
                f"p10 max drawdown {_format_percent(drawdown_row.get('max_drawdown_p10'))}."
            ),
            "evidence": (
                f"Severe drawdown probability {_format_percent(forward_summary.get('severe_drawdown_probability'))}; "
                f"hard-band breach share {_format_percent(drawdown_row.get('breach_hard_band'))}; "
                f"median ulcer {_format_percent(drawdown_row.get('ulcer_index_p50'))}."
            ),
            "implication": "This is the behavioral tolerance check, not just a return forecast.",
        },
        {
            "question": "What current-state assumption is shaping the paths?",
            "read": f"{top_regime} has the largest scenario share at {_format_percent(top_probability)}.",
            "evidence": (
                f"Risk-off {_format_percent(_probability_sum(probabilities, ('risk_off',)))}; "
                f"transition {_format_percent(_probability_sum(probabilities, ('transition',)))}; "
                f"risk-on {_format_percent(_probability_sum(probabilities, ('risk_on',)))}; "
                f"fragile risk-on {_format_percent(_probability_sum(probabilities, ('risk_on_fragile',)))}."
            ),
            "implication": "If the scenario map is wrong, the regime-conditioned simulation will be wrong in that direction.",
        },
        {
            "question": "What should not be over-read?",
            "read": "This is empirical path planning, not an oracle.",
            "evidence": "The model samples historical regime-return blocks and cannot invent unseen AI, credit, policy, or liquidity events.",
            "implication": "Use the output as a distribution and stress lens alongside paper monitoring.",
        },
    ]


def _regime_resemblance_frame(
    *,
    probabilities: pd.DataFrame,
    library: pd.DataFrame,
    forward_paths: pd.DataFrame,
) -> pd.DataFrame:
    historical = (
        library["regime"].astype(str).value_counts(normalize=True)
        if not library.empty and "regime" in library
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for bucket in REGIME_BUCKETS:
        historical_share = _safe_float(historical.get(bucket, 0.0)) or 0.0
        scenario_share = _probability_sum(probabilities, (bucket,))
        simulated_share = _series_mean(_numeric_path_column(forward_paths, f"share_{bucket}")) or 0.0
        scenario_delta = scenario_share - historical_share
        simulation_delta = simulated_share - historical_share
        rows.append(
            {
                "regime": bucket,
                "historical_return_library_share": historical_share,
                "current_scenario_probability": scenario_share,
                "mean_simulated_path_share": simulated_share,
                "scenario_minus_history": scenario_delta,
                "simulation_minus_history": simulation_delta,
                "read": _regime_resemblance_read(simulation_delta),
            }
        )
    return pd.DataFrame(rows)


def _regime_resemblance_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    bars = [
        ("historical_return_library_share", "Historical library", "#64748b"),
        ("current_scenario_probability", "Current scenario", "#f59e0b"),
        ("mean_simulated_path_share", "Simulated paths", "#0f766e"),
    ]
    for column, label, color in bars:
        fig.add_trace(
            go.Bar(
                x=frame["regime"],
                y=frame[column],
                name=label,
                marker={"color": color},
                hovertemplate=f"{label}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=360,
        title="Regime resemblance: past vs current scenario vs simulated paths",
        yaxis={"title": "Share", "tickformat": ".0%"},
        xaxis={"title": "Regime bucket"},
        margin={"l": 20, "r": 20, "t": 48, "b": 84},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _simulation_missing_rows(
    validation_row: dict[str, object],
    probabilities: pd.DataFrame,
    library: pd.DataFrame,
) -> list[dict[str, object]]:
    regime_counts = (
        library["regime"].astype(str).value_counts().reindex(REGIME_BUCKETS).fillna(0).astype(int)
        if not library.empty and "regime" in library
        else pd.Series(0, index=REGIME_BUCKETS, dtype=int)
    )
    sparse_regimes = [regime for regime, count in regime_counts.items() if int(count) < 50]
    return [
        {
            "gap": "Novel regime risk",
            "current_read": (
                f"Risk-off plus transition probability is "
                f"{_format_percent(_probability_sum(probabilities, ('risk_off', 'transition')))}."
            ),
            "why_it_matters": "The simulator can sample only historical analogues, so a new AI/credit/policy break may not be fully represented.",
            "mitigation": "Compare simulation output with Daily Brief drivers, paper monitoring, and explicit stress tests.",
        },
        {
            "gap": "Sparse regime evidence",
            "current_read": (
                f"Sparse buckets: {', '.join(sparse_regimes) if sparse_regimes else 'none'}; "
                f"smallest bucket count {int(regime_counts.min()) if not regime_counts.empty else 0}."
            ),
            "why_it_matters": "A regime with little history produces less reliable forward-path behavior.",
            "mitigation": "Treat affected regimes as lower-confidence and inspect reference overlays.",
        },
        {
            "gap": "Execution and taxes",
            "current_read": "Path simulations use strategy returns before actual fills, missed trades, and taxable-account frictions.",
            "why_it_matters": "A strategy can look good in ideal paths and still disappoint when execution timing or taxes drag returns.",
            "mitigation": "Use Monitoring, Forward Test, implementation shortfall, and taxable-impact views before live use.",
        },
        {
            "gap": "Point-estimate confidence",
            "current_read": str(validation_row.get("calibration_read", "insufficient_history")),
            "why_it_matters": "Large gaps between deterministic, bootstrap, and regime-conditioned output mean the CAGR point estimate is not enough.",
            "mitigation": "Prioritize candidates that retain an edge across the distribution, not only in deterministic CAGR math.",
        },
    ]


def _simulation_method_rows() -> list[dict[str, str]]:
    return [
        {
            "layer": "Deterministic CAGR",
            "what_it_uses": "One historical CAGR point estimate plus configured starting account and monthly contributions.",
            "best_for": "Fast frontier ranking and simple benchmark comparison.",
            "main_failure_mode": "Ignores sequencing, volatility, and drawdown path pain.",
        },
        {
            "layer": "Historical block bootstrap",
            "what_it_uses": "Resampled blocks of the selected strategy's realized daily returns.",
            "best_for": "Sequence risk, drawdown persistence, and terminal-wealth range.",
            "main_failure_mode": "Assumes the future is drawn from the same realized return mix.",
        },
        {
            "layer": "Regime-conditioned forward paths",
            "what_it_uses": "Current scenario probabilities plus historical regime-labeled return blocks.",
            "best_for": "Current-state-aware planning ranges and downside/tail inspection.",
            "main_failure_mode": "Regime labels are coarse and scenario probabilities can be wrong.",
        },
    ]


def _reference_edge_rows(
    forward_summary: dict[str, object],
    reference_simulations: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    rows: list[dict[str, object]] = []
    for reference in reference_simulations:
        summary = reference.get("forward_summary")
        if not isinstance(summary, dict):
            continue
        reference_median = _safe_float(summary.get("terminal_wealth_p50"))
        edge = selected_median - reference_median if selected_median is not None and reference_median is not None else None
        rows.append(
            {
                "reference": reference.get("label"),
                "reference_strategy": reference.get("strategy"),
                "selected_forward_median": selected_median,
                "reference_forward_median": reference_median,
                "selected_minus_reference": edge,
                "selected_edge_pct": _relative_delta(selected_median, reference_median),
                "reference_median_forward_drawdown": summary.get("max_drawdown_p50"),
                "reference_severe_drawdown_probability": summary.get("severe_drawdown_probability"),
            }
        )
    return rows


def _simulation_comparison_row(
    *,
    label: str,
    strategy: str,
    deterministic_wealth: float | None,
    bootstrap_summary: dict[str, object],
    forward_summary: dict[str, object],
    selected_forward_median: float | None,
) -> dict[str, object]:
    forward_median = _safe_float(forward_summary.get("terminal_wealth_p50"))
    selected_delta = (
        selected_forward_median - forward_median
        if selected_forward_median is not None and forward_median is not None
        else None
    )
    return {
        "portfolio": label,
        "strategy": strategy,
        "deterministic_15y": deterministic_wealth,
        "bootstrap_p10": bootstrap_summary.get("terminal_wealth_p10"),
        "bootstrap_median": bootstrap_summary.get("terminal_wealth_p50"),
        "bootstrap_p90": bootstrap_summary.get("terminal_wealth_p90"),
        "forward_p10": forward_summary.get("terminal_wealth_p10"),
        "forward_median": forward_summary.get("terminal_wealth_p50"),
        "forward_p90": forward_summary.get("terminal_wealth_p90"),
        "selected_minus_row_forward_median": selected_delta,
        "median_forward_drawdown": forward_summary.get("max_drawdown_p50"),
        "severe_drawdown_probability": forward_summary.get("severe_drawdown_probability"),
    }


def _render_simulation_interpretability(
    *,
    selected_strategy: str | None,
    selected_scorecard: pd.Series | None,
    selected_result: BacktestResult | None,
    baseline_run: BaselineRun,
    scenario_source: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> None:
    st.markdown("**Simulation Interpretability**")
    st.caption(
        "This turns simulation paths into decision intelligence: whether the simulated future "
        "looks like the tested past, whether the selected strategy earns its complexity versus "
        "references, and what can still go wrong."
    )
    if selected_strategy is None or selected_result is None:
        st.info("Pick a strategy with reconstructable history to see simulation interpretation.")
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
        st.info("The selected strategy did not produce a usable historical return library.")
        return

    deterministic_wealth = _deterministic_wealth(selected_scorecard, selected_result)
    return_values = _returns_tuple(selected_result)
    bootstrap_paths = _cached_bootstrap_paths(return_values)
    bootstrap_summary = summarize_bootstrap_outcomes(bootstrap_paths)
    forward_paths = _cached_regime_forward_paths(return_values, _scenario_records(scenario_source))
    forward_summary = summarize_forward_simulation(forward_paths, config=ForwardSimulationConfig())
    reference_options = _reference_option_frame(baseline_run, selected_strategy)
    reference_simulations = _reference_simulations(
        baseline_run=baseline_run,
        reference_options=reference_options,
        selected_reference_labels=reference_options["label"].head(2).tolist()
        if not reference_options.empty
        else [],
        scenario_source=scenario_source,
    )
    validation_row = _simulation_validation_row(
        label="Selected strategy",
        strategy=selected_strategy,
        result=selected_result,
        deterministic_wealth=deterministic_wealth,
        bootstrap_summary=bootstrap_summary,
        forward_summary=forward_summary,
    )
    drawdown_row = _drawdown_distribution_row(
        label="Selected forward",
        strategy=selected_strategy,
        paths=forward_paths,
    )
    reference_edges = _reference_edge_rows(forward_summary, reference_simulations)

    _render_simulation_insight_cards(
        _simulation_interpretability_cards(
            strategy=selected_strategy,
            validation_row=validation_row,
            drawdown_row=drawdown_row,
            forward_summary=forward_summary,
            reference_edges=reference_edges,
            probabilities=probabilities,
        )
    )

    st.caption("Simulation verdict")
    _render_metric_dataframe(
        _display_metrics(
            pd.DataFrame(
                _simulation_verdict_rows(
                    selected_strategy=selected_strategy,
                    validation_row=validation_row,
                    drawdown_row=drawdown_row,
                    forward_summary=forward_summary,
                    reference_edges=reference_edges,
                    probabilities=probabilities,
                )
            )
        ),
        hide_index=True,
    )

    resemblance = _regime_resemblance_frame(
        probabilities=probabilities,
        library=library,
        forward_paths=forward_paths,
    )
    if not resemblance.empty:
        st.caption("How much do the simulated futures resemble the tested past?")
        st.plotly_chart(_regime_resemblance_figure(resemblance), use_container_width=True)
        _render_metric_dataframe(_display_metrics(resemblance), hide_index=True)

    st.caption("What the simulation may be missing")
    _render_metric_dataframe(
        pd.DataFrame(_simulation_missing_rows(validation_row, probabilities, library)),
        hide_index=True,
    )

    regime_summary = (
        library.groupby("regime")["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_daily_return", "std": "daily_volatility"})
    )
    regime_summary["annualized_mean_return"] = regime_summary["mean_daily_return"] * 252
    regime_summary["annualized_volatility"] = regime_summary["daily_volatility"] * (252**0.5)
    with st.expander("Audit details: model inputs, scenario records, and sampled return libraries", expanded=False):
        _render_metric_dataframe(pd.DataFrame(_simulation_method_rows()), hide_index=True)
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

        st.caption(f"Historical return library for {selected_strategy}")
        _render_metric_dataframe(_display_metrics(regime_summary), hide_index=True)
        mix = regime_mix_frame(forward_paths)
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
    *,
    reference_simulations: list[dict[str, object]] | None = None,
) -> str:
    forward_p10 = _format_currency(forward_summary.get("terminal_wealth_p10"))
    forward_p50 = _format_currency(forward_summary.get("terminal_wealth_p50"))
    forward_p90 = _format_currency(forward_summary.get("terminal_wealth_p90"))
    severe = _format_percent(forward_summary.get("severe_drawdown_probability"))
    bootstrap_median = _format_currency(bootstrap_summary.get("terminal_wealth_p50"))
    reference_sentence = _reference_plain_english_read(forward_summary, reference_simulations or [])
    return (
        f"{selected_strategy}: the historical bootstrap median is {bootstrap_median}. "
        f"The scenario-conditioned central range is {forward_p10} to {forward_p90}, "
        f"with median {forward_p50} and severe-drawdown probability {severe}. "
        f"{reference_sentence}"
        "Read this as a planning distribution, not as a point forecast."
    )


def _reference_plain_english_read(
    selected_forward_summary: dict[str, object],
    reference_simulations: list[dict[str, object]],
) -> str:
    selected_median = _safe_float(selected_forward_summary.get("terminal_wealth_p50"))
    if selected_median is None or not reference_simulations:
        return ""
    parts: list[str] = []
    for reference in reference_simulations:
        summary = reference.get("forward_summary")
        if not isinstance(summary, dict):
            continue
        reference_median = _safe_float(summary.get("terminal_wealth_p50"))
        if reference_median is None:
            continue
        delta = selected_median - reference_median
        relation = "above" if delta >= 0 else "below"
        parts.append(
            f"{reference['label']} median {_format_currency(reference_median)} "
            f"({_format_currency(abs(delta))} {relation})"
        )
    if not parts:
        return ""
    return f"Against references: {'; '.join(parts)}. "


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


def _simulation_overlay_histogram(
    items: list[dict[str, object]],
    *,
    column: str,
    title: str,
    xaxis_title: str,
    value_label: str = "Value",
    value_format: str = "decimal",
) -> go.Figure:
    fig = go.Figure()
    for item in items:
        paths = item.get("paths")
        if not isinstance(paths, pd.DataFrame) or column not in paths or paths.empty:
            continue
        label = str(item.get("label", "portfolio"))
        hovertemplate = _overlay_hover_template(label, value_label, value_format)
        fig.add_trace(
            go.Histogram(
                x=paths[column],
                nbinsx=40,
                name=label,
                opacity=0.55,
                marker={
                    "color": str(item.get("color", "#2563eb")),
                    "line": {"color": "#0f172a", "width": 0.5},
                },
                hovertemplate=hovertemplate,
            )
        )
    xaxis: dict[str, object] = {"title": xaxis_title}
    if value_format == "percent":
        xaxis["tickformat"] = ".0%"
    fig.update_layout(
        barmode="overlay",
        height=340,
        title=title,
        xaxis=xaxis,
        yaxis_title="Path count",
        margin={"l": 20, "r": 20, "t": 44, "b": 76},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _overlay_hover_template(label: str, value_label: str, value_format: str) -> str:
    if value_format == "currency":
        value = "%{x:$,.0f}"
    elif value_format == "percent":
        value = "%{x:.1%}"
    else:
        value = "%{x}"
    return f"{label}<br>{value_label} {value}<br>Paths %{{y}}<extra></extra>"


def _cagr_from_result(result: BacktestResult) -> float | None:
    if result.equity.empty or len(result.equity) < 2:
        return None
    start = _safe_float(result.equity.iloc[0])
    end = _safe_float(result.equity.iloc[-1])
    if start is None or end is None or start <= 0.0:
        return None
    years = max(len(result.equity) / 252.0, 1.0 / 252.0)
    return (end / start) ** (1.0 / years) - 1.0


def _historical_result_metrics(result: BacktestResult) -> dict[str, float | None]:
    equity = pd.to_numeric(result.equity, errors="coerce").dropna()
    if equity.empty:
        return {
            "observed_years": None,
            "cagr": None,
            "max_drawdown": None,
            "ulcer_index": None,
        }
    drawdown = equity / equity.cummax() - 1.0
    return {
        "observed_years": len(equity) / 252.0,
        "cagr": _cagr_from_result(result),
        "max_drawdown": _series_min(drawdown),
        "ulcer_index": _series_ulcer_index(drawdown),
    }


def _simulation_calibration_read(
    *,
    deterministic_delta: float | None,
    drawdown_delta: float | None,
    ulcer_delta: float | None,
) -> str:
    if deterministic_delta is None and drawdown_delta is None and ulcer_delta is None:
        return "insufficient_history"
    stressed = (
        (deterministic_delta is not None and deterministic_delta <= -0.15)
        or (drawdown_delta is not None and drawdown_delta <= -0.05)
        or (ulcer_delta is not None and ulcer_delta >= 0.03)
    )
    optimistic = (
        deterministic_delta is not None
        and deterministic_delta >= 0.15
        and (drawdown_delta is None or drawdown_delta >= 0.03)
    )
    if stressed:
        return "simulation_more_stressed_than_history"
    if optimistic:
        return "simulation_more_optimistic_than_history"
    return "broadly_past_like"


def _calibration_answer(calibration_read: str) -> str:
    labels = {
        "simulation_more_stressed_than_history": "Forward paths are more stressed than history",
        "simulation_more_optimistic_than_history": "Forward paths look easier than history",
        "broadly_past_like": "Forward paths broadly resemble the tested past",
        "insufficient_history": "Not enough history to judge resemblance",
    }
    return labels.get(calibration_read, calibration_read.replace("_", " ").title())


def _calibration_tone(calibration_read: str) -> str:
    if calibration_read == "simulation_more_stressed_than_history":
        return "warning"
    if calibration_read == "simulation_more_optimistic_than_history":
        return "critical"
    if calibration_read == "broadly_past_like":
        return "success"
    return "warning"


def _best_reference_edge(reference_edges: list[dict[str, object]]) -> dict[str, object] | None:
    scored = [
        row
        for row in reference_edges
        if _safe_float(row.get("selected_minus_reference")) is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda row: _safe_float(row.get("selected_minus_reference")) or 0.0)


def _reference_edge_answer(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "Reference comparison unavailable"
    delta = _safe_float(edge.get("selected_minus_reference"))
    label = str(edge.get("reference", "reference"))
    if delta is None:
        return f"Reference comparison to {label} unavailable"
    if delta >= 0:
        return f"Beats {label} by {_format_currency(delta)}"
    return f"Trails {label} by {_format_currency(abs(delta))}"


def _reference_edge_detail(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "Reference overlays were unavailable for this loaded run."
    return (
        f"Selected median {_format_currency(edge.get('selected_forward_median'))}; "
        f"{edge.get('reference')} median {_format_currency(edge.get('reference_forward_median'))}; "
        f"edge {_format_percent(edge.get('selected_edge_pct'))}; reference median drawdown "
        f"{_format_percent(edge.get('reference_median_forward_drawdown'))}."
    )


def _reference_edge_tone(edge: dict[str, object] | None) -> str:
    if edge is None:
        return "warning"
    delta = _safe_float(edge.get("selected_minus_reference"))
    if delta is None:
        return "warning"
    return "success" if delta >= 0 else "critical"


def _drawdown_tone(
    median_drawdown: float | None,
    severe_probability: float | None,
    hard_breach: float | None,
) -> str:
    if (hard_breach is not None and hard_breach >= 0.10) or (
        severe_probability is not None and severe_probability >= 0.10
    ):
        return "critical"
    if median_drawdown is not None and median_drawdown <= DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT:
        return "warning"
    return "success"


def _probability_sum(probabilities: pd.DataFrame, buckets: tuple[str, ...]) -> float:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return 0.0
    frame = probabilities[probabilities["regime"].astype(str).isin(buckets)]
    if frame.empty:
        return 0.0
    return float(pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0).sum())


def _top_probability_label(probabilities: pd.DataFrame) -> tuple[str, float | None]:
    if probabilities.empty or "regime" not in probabilities or "probability" not in probabilities:
        return "n/a", None
    values = pd.to_numeric(probabilities["probability"], errors="coerce")
    if values.dropna().empty:
        return "n/a", None
    idx = values.idxmax()
    return str(probabilities.loc[idx, "regime"]), _safe_float(values.loc[idx])


def _regime_resemblance_read(simulation_delta: float) -> str:
    if simulation_delta >= 0.05:
        return "future_overweights_past"
    if simulation_delta <= -0.05:
        return "future_underweights_past"
    return "past_like"


def _numeric_path_column(paths: pd.DataFrame, column: str) -> pd.Series:
    if paths.empty or column not in paths:
        return pd.Series(dtype=float)
    return pd.to_numeric(paths[column], errors="coerce").dropna()


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(quantile))


def _series_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.mean())


def _series_min(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.min())


def _series_ulcer_index(drawdown: pd.Series) -> float | None:
    clean = pd.to_numeric(drawdown, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean.pow(2).mean()) ** 0.5)


def _relative_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or abs(reference) <= 1e-12:
        return None
    return value / reference - 1.0


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
