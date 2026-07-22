from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.risk_scenarios import (
    _render_portfolio_risk_history,
    _render_regime_instability_history,
    _render_risk_and_scenarios,
    _render_scenario_probability_explanation,
    _render_scenario_probability_history,
)
from trade_bot.dashboard_v2.components.cards import (
    render_callout,
    render_card_grid,
    render_section_header,
)
from trade_bot.dashboard_v2.components.tones import (
    beta_delta_tone,
    beta_tone,
    expected_shortfall_tone,
    instability_tone,
    portfolio_risk_tone,
    risk_budget_tone,
    sleeve_exposure_tone,
    stress_loss_tone,
)
from trade_bot.dashboard_v2.services.runtime import (
    HISTORICAL_SNAPSHOT_NOTICE,
    DashboardRuntime,
)
from trade_bot.research.operating_exposure import (
    aggregate_beta_adjusted_spy_delta,
    build_beta_adjusted_delta_table,
    build_sleeve_exposure_table,
    build_tactical_matrix,
    weights_from_position_plan,
)


def render_risk_page(runtime: DashboardRuntime) -> None:
    if runtime.is_historical_snapshot_mode:
        st.warning(HISTORICAL_SNAPSHOT_NOTICE)
        st.caption(
            "Risk is an operating surface tied to the current promoted book. Switch to Latest "
            "snapshot or Live pipeline before interpreting exposure caps, stress loss, beta, "
            "or scenario sizing as current."
        )
        return
    operating_error = getattr(runtime, "operating_strategy_error", None)
    if operating_error or runtime.operating_trade_decision is None:
        st.error(
            "Operating risk unavailable for promoted book "
            f"'{runtime.promoted_book.book_name}': {operating_error or 'unknown resolution error'}"
        )
        st.caption(
            "No promoted-book exposure or risk claim is shown because the named strategy "
            "could not be resolved exactly and safely."
        )
        return
    current_state = runtime.baseline_run.current_state
    portfolio_risk = _portfolio_risk(runtime)
    risk_summary = _first_row(getattr(portfolio_risk, "summary", pd.DataFrame()))
    current_weights = weights_from_position_plan(runtime.operating_trade_decision.position_plan)
    sleeve_exposure = (
        build_sleeve_exposure_table(current_weights, runtime.baseline_run.prices)
        if not current_weights.empty
        else pd.DataFrame()
    )
    beta_delta = (
        aggregate_beta_adjusted_spy_delta(runtime.baseline_run.prices, current_weights)
        if not current_weights.empty
        else float("nan")
    )
    instability = _first_row(getattr(current_state, "regime_instability", pd.DataFrame()))
    risk_level = _value(risk_summary, "portfolio_risk_level", "n/a")
    risk_multiplier = _value(risk_summary, "portfolio_risk_multiplier")
    es95 = _value(risk_summary, "post_expected_shortfall_95")
    max_stress_loss = _value(risk_summary, "post_max_stress_loss")
    equity_beta = _value(risk_summary, "post_equity_beta")
    ai_beta = _value(risk_summary, "post_ai_beta")
    defensive_fraction = _sleeve_fraction(sleeve_exposure, "defensive")
    instability_state = str(_value(instability, "regime_instability_state", "n/a")).upper()

    render_card_grid(
        [
            ("Risk Level", risk_level, None, portfolio_risk_tone(risk_level)),
            (
                "Risk Multiplier",
                _fmt_float(risk_multiplier),
                None,
                risk_budget_tone(risk_multiplier),
            ),
            ("ES 95", _fmt_pct(es95), None, expected_shortfall_tone(es95)),
            ("Max Stress Loss", _fmt_pct(max_stress_loss), None, stress_loss_tone(max_stress_loss)),
            (
                "Equity Beta",
                _fmt_float(equity_beta),
                None,
                beta_tone(equity_beta, warning_at=0.75, critical_at=1.00),
            ),
            (
                "AI Beta",
                _fmt_float(ai_beta),
                None,
                beta_tone(ai_beta, warning_at=0.60, critical_at=0.90),
            ),
            ("Beta-Adjusted S&P Delta", _fmt_pct(beta_delta), None, beta_delta_tone(beta_delta)),
            (
                "Defensive % of Max",
                _fmt_pct(defensive_fraction),
                None,
                sleeve_exposure_tone("defensive", defensive_fraction),
            ),
            ("Instability", instability_state, None, instability_tone(instability_state)),
        ]
    )

    view = st.pills(
        "Risk view",
        [
            "Overview",
            "Portfolio Risk",
            "Operating Exposure",
            "Instability",
            "Scenarios",
            "Confirmation",
            "Momentum",
            "Full Workbench",
        ],
        default="Overview",
        selection_mode="single",
        key="dashboard_v2_risk_view",
    )
    selected_view = view or "Overview"
    if selected_view == "Overview":
        _render_overview(runtime, risk_summary, instability, sleeve_exposure, beta_delta)
    elif selected_view == "Portfolio Risk":
        _render_portfolio_risk(runtime, portfolio_risk)
    elif selected_view == "Operating Exposure":
        _render_operating_exposure(runtime, current_weights, sleeve_exposure, beta_delta)
    elif selected_view == "Instability":
        _render_instability(runtime)
    elif selected_view == "Scenarios":
        _render_scenarios(runtime)
    elif selected_view == "Confirmation":
        _render_confirmation(runtime)
    elif selected_view == "Momentum":
        _render_momentum(runtime)
    else:
        render_callout(
            "This loads the complete Risk & Scenarios workbench for the original dense table set.",
            heavy=True,
        )
        _render_risk_and_scenarios(
            runtime.baseline_run,
            run_store_path=runtime.paths.run_store_path,
            artifact_dir=runtime.paths.artifact_dir,
            job_log_dir=runtime.paths.job_log_dir,
        )


def _render_overview(
    runtime: DashboardRuntime,
    risk_summary: pd.Series,
    instability: pd.Series,
    sleeve_exposure: pd.DataFrame,
    beta_delta: float,
) -> None:
    render_section_header("Risk Operating Read")
    scenario_lattice = getattr(
        runtime.baseline_run.current_state, "scenario_lattice", pd.DataFrame()
    )
    top_scenario = _top_scenario_read(scenario_lattice)
    risk_level = _value(risk_summary, "portfolio_risk_level", "n/a")
    risk_multiplier = _fmt_float(_value(risk_summary, "portfolio_risk_multiplier"))
    defensive = _sleeve_percent(sleeve_exposure, "defensive")
    instability_state = str(_value(instability, "regime_instability_state", "n/a")).lower()
    render_callout(
        f"Current risk engine read is {risk_level} with risk multiplier {risk_multiplier}. "
        f"Operating exposure for promoted book '{runtime.promoted_book.book_name}' "
        f"({runtime.promoted_book.strategy_name}) is {defensive} defensive and beta-adjusted "
        f"S&P delta is {_fmt_pct(beta_delta)}. Scenario pressure is led by {top_scenario}; "
        f"instability is {instability_state}."
    )
    _render_portfolio_risk_history(
        run_store_path=str(runtime.paths.run_store_path),
        artifact_dir=str(runtime.paths.artifact_dir),
        job_log_dir=str(runtime.paths.job_log_dir),
    )
    _render_regime_instability_history(
        run_store_path=str(runtime.paths.run_store_path),
        artifact_dir=str(runtime.paths.artifact_dir),
        job_log_dir=str(runtime.paths.job_log_dir),
    )


def _render_portfolio_risk(runtime: DashboardRuntime, portfolio_risk: Any | None) -> None:
    render_section_header("Portfolio Risk Detail")
    if portfolio_risk is None or getattr(portfolio_risk, "summary", pd.DataFrame()).empty:
        st.info("No portfolio risk diagnostics are available.")
        return
    render_callout(
        "Use this view to see which guardrail changed target size: scenario stress, expected shortfall, factor beta, tail tests, or correlation."
    )
    detail_view = (
        st.pills(
            "Portfolio risk detail",
            ["Constraints", "Scenarios", "Factors / Betas", "Tail / Stress", "Correlation"],
            default="Constraints",
            selection_mode="single",
            key="dashboard_v2_portfolio_risk_detail",
        )
        or "Constraints"
    )
    if detail_view == "Constraints":
        _render_metric_dataframe(_display_metrics(portfolio_risk.summary))
        _render_metric_dataframe(_display_metrics(portfolio_risk.constraint_report))
        _render_metric_dataframe(_display_metrics(portfolio_risk.sizing_adjustments))
    elif detail_view == "Scenarios":
        _render_metric_dataframe(_display_metrics(portfolio_risk.scenario_risk_budget))
    elif detail_view == "Factors / Betas":
        _render_metric_dataframe(_display_metrics(portfolio_risk.factor_exposures))
        _render_metric_dataframe(_display_metrics(portfolio_risk.beta_decomposition))
    elif detail_view == "Tail / Stress":
        _render_metric_dataframe(_display_metrics(portfolio_risk.tail_risk))
        _render_metric_dataframe(_display_metrics(portfolio_risk.stress_tests))
    else:
        _render_metric_dataframe(_display_metrics(portfolio_risk.correlation_regime))
        _render_metric_dataframe(_display_metrics(portfolio_risk.marginal_risk_contribution))


def _render_operating_exposure(
    runtime: DashboardRuntime,
    current_weights: pd.Series,
    sleeve_exposure: pd.DataFrame,
    beta_delta: float,
) -> None:
    render_section_header("Operating Exposure")
    render_callout(
        f"Exposure is calculated from promoted book '{runtime.promoted_book.book_name}' "
        f"and its resolved strategy target ({runtime.promoted_book.strategy_name}), not from "
        "the configured default baseline when those differ."
    )
    if current_weights.empty:
        st.info("No current target weights are available for exposure diagnostics.")
        return
    stocks_fraction = _sleeve_fraction(sleeve_exposure, "stocks")
    defensive_fraction = _sleeve_fraction(sleeve_exposure, "defensive")
    gold_fraction = _sleeve_fraction(sleeve_exposure, "gold")
    crypto_fraction = _sleeve_fraction(sleeve_exposure, "crypto")
    credit_fraction = _sleeve_fraction(sleeve_exposure, "credit")
    render_card_grid(
        [
            ("Beta-Adjusted S&P Delta", _fmt_pct(beta_delta), None, beta_delta_tone(beta_delta)),
            (
                "Stocks % of Max",
                _fmt_pct(stocks_fraction),
                None,
                sleeve_exposure_tone("stocks", stocks_fraction),
            ),
            (
                "Defensive % of Max",
                _fmt_pct(defensive_fraction),
                None,
                sleeve_exposure_tone("defensive", defensive_fraction),
            ),
            (
                "Gold % of Max",
                _fmt_pct(gold_fraction),
                None,
                sleeve_exposure_tone("gold", gold_fraction),
            ),
            (
                "Crypto % of Max",
                _fmt_pct(crypto_fraction),
                None,
                sleeve_exposure_tone("crypto", crypto_fraction),
            ),
            (
                "Credit % of Max",
                _fmt_pct(credit_fraction),
                None,
                sleeve_exposure_tone("credit", credit_fraction),
            ),
        ]
    )
    exposure_view = (
        st.pills(
            "Exposure detail",
            ["Sleeve Exposure", "Beta Delta", "Tactical Matrix"],
            default="Sleeve Exposure",
            selection_mode="single",
            key="dashboard_v2_operating_exposure_detail",
        )
        or "Sleeve Exposure"
    )
    if exposure_view == "Sleeve Exposure":
        _render_metric_dataframe(_display_metrics(sleeve_exposure), hide_index=True)
    elif exposure_view == "Beta Delta":
        beta_table = build_beta_adjusted_delta_table(runtime.baseline_run.prices, current_weights)
        _render_metric_dataframe(_display_metrics(beta_table), hide_index=True)
    else:
        tactical_matrix = build_tactical_matrix(
            runtime.baseline_run.prices,
            current_weights=current_weights,
            risk_status=str(getattr(runtime.baseline_run.current_state, "risk_status", "")),
            regime=_lead_regime_label(runtime.baseline_run.current_state),
        )
        _render_metric_dataframe(_display_metrics(tactical_matrix), hide_index=True)


def _render_instability(runtime: DashboardRuntime) -> None:
    render_section_header("Regime Instability")
    current_state = runtime.baseline_run.current_state
    regime_instability = getattr(current_state, "regime_instability", pd.DataFrame())
    components = getattr(current_state, "regime_instability_components", pd.DataFrame())
    if regime_instability.empty:
        st.info("No regime-instability diagnostics are available.")
        return
    row = regime_instability.iloc[0]
    instability_state = str(row.get("regime_instability_state", "n/a")).upper()
    instability_score = row.get("regime_instability_score")
    large_move_share = row.get("spy_ytd_large_move_share")
    render_card_grid(
        [
            ("Instability", instability_state, None, instability_tone(instability_state)),
            ("Score", _fmt_float(instability_score), None, instability_tone(instability_score)),
            (
                "SPY +/-1% YTD",
                _fmt_pct(large_move_share),
                None,
                instability_tone(large_move_share),
            ),
            (
                "Large Move Days",
                f"{int(row.get('spy_ytd_large_move_days', 0))}/{int(row.get('spy_ytd_trading_days', 0))}",
            ),
            ("Use", "Watch Only", None, "warning"),
        ]
    )
    read = str(row.get("regime_instability_read", "")).strip()
    if read:
        render_callout(read)
    _render_metric_dataframe(_display_metrics(components))
    _render_regime_instability_history(
        run_store_path=str(runtime.paths.run_store_path),
        artifact_dir=str(runtime.paths.artifact_dir),
        job_log_dir=str(runtime.paths.job_log_dir),
    )


def _render_scenarios(runtime: DashboardRuntime) -> None:
    render_section_header("Future-State Scenario Lattice")
    current_state = runtime.baseline_run.current_state
    scenario_lattice = getattr(current_state, "scenario_lattice", pd.DataFrame())
    _render_scenario_probability_explanation(
        scenario_lattice,
        getattr(current_state, "scenario_drivers", pd.DataFrame()),
    )
    _render_scenario_probability_history(
        runtime.baseline_run,
        run_store_path=runtime.paths.run_store_path,
        artifact_dir=runtime.paths.artifact_dir,
        job_log_dir=runtime.paths.job_log_dir,
    )
    drivers = getattr(current_state, "scenario_drivers", pd.DataFrame())
    if not drivers.empty:
        render_section_header("Scenario Drivers")
        _render_metric_dataframe(_display_metrics(drivers))
    if scenario_lattice.empty:
        return
    scenario_horizon = st.radio(
        "Scenario horizon",
        ["1w", "1m", "3m", "6m"],
        index=1,
        horizontal=True,
        key="dashboard_v2_risk_scenario_horizon",
    )
    scenario_bucket_options = ["all", *sorted(scenario_lattice["risk_bucket"].dropna().unique())]
    scenario_bucket = st.selectbox(
        "Risk bucket",
        scenario_bucket_options,
        key="dashboard_v2_risk_scenario_bucket",
    )
    scenario_limit = st.slider(
        "Scenarios shown",
        min_value=5,
        max_value=20,
        value=12,
        step=1,
        key="dashboard_v2_risk_scenario_limit",
    )
    scenario_view = scenario_lattice[scenario_lattice["horizon"] == scenario_horizon]
    if scenario_bucket != "all":
        scenario_view = scenario_view[scenario_view["risk_bucket"] == scenario_bucket]
    _render_metric_dataframe(
        _display_metrics(scenario_view.sort_values("rank").head(scenario_limit))
    )


def _render_confirmation(runtime: DashboardRuntime) -> None:
    render_section_header("Confirmation and Health")
    current_state = runtime.baseline_run.current_state
    confirmation = getattr(current_state, "confirmation_matrix", pd.DataFrame())
    market_health = getattr(current_state, "market_health", pd.DataFrame())
    if confirmation.empty and market_health.empty:
        st.info("No confirmation or market-health diagnostics are available.")
        return
    render_callout(
        "Confirmation checks whether price, breadth, credit, volatility, and leadership proxies agree with the operating posture."
    )
    if not confirmation.empty:
        render_section_header("Risk Confirmation Matrix")
        _render_metric_dataframe(_display_metrics(confirmation))
    if not market_health.empty:
        render_section_header("Market Health")
        display_health = _market_health_for_display(market_health)
        if "VIXY" in display_health.index:
            render_callout(
                "VIXY all-history drawdown is intentionally not reported. This short-term "
                "VIX-futures proxy is evaluated through its 1-month/3-month returns and "
                "momentum state instead."
            )
        _render_metric_dataframe(_display_metrics(display_health))


def _render_momentum(runtime: DashboardRuntime) -> None:
    render_section_header("Vol-Adjusted Momentum")
    momentum = getattr(runtime.baseline_run.current_state, "momentum_state", pd.DataFrame())
    if momentum.empty:
        st.info("No momentum-state table is available.")
        return
    momentum_filter = st.radio(
        "Momentum filter",
        ["all", "bullish", "neutral", "bearish"],
        horizontal=True,
        key="dashboard_v2_risk_momentum_filter",
    )
    if momentum_filter != "all" and "momentum_state_label" in momentum:
        momentum = momentum[momentum["momentum_state_label"] == momentum_filter]
    _render_metric_dataframe(_display_metrics(momentum.head(75)))


def _portfolio_risk(runtime: DashboardRuntime) -> Any | None:
    operating_decision = runtime.operating_trade_decision
    return (
        (operating_decision.portfolio_risk if operating_decision is not None else None)
        or runtime.baseline_run.portfolio_risk
        or runtime.baseline_run.trade_decision.portfolio_risk
    )


def _market_health_for_display(market_health: pd.DataFrame) -> pd.DataFrame:
    """Normalize older snapshots to the current VIXY display contract."""

    output = market_health.copy()
    if "VIXY" not in output.index:
        return output
    if "drawdown" in output:
        output.loc["VIXY", "drawdown"] = float("nan")
    if "drawdown_basis" not in output:
        output["drawdown_basis"] = "drawdown_from_available_history_peak"
    output.loc["VIXY", "drawdown_basis"] = "not_applicable_short_term_volatility_proxy"
    return output


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.iloc[0]


def _value(row: pd.Series, column: str, default: object = float("nan")) -> object:
    if row.empty or column not in row:
        return default
    return row.get(column, default)


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.2%}"


def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.2f}"


def _sleeve_percent(sleeve_exposure: pd.DataFrame, sleeve: str) -> str:
    return _fmt_pct(_sleeve_fraction(sleeve_exposure, sleeve))


def _sleeve_fraction(sleeve_exposure: pd.DataFrame, sleeve: str) -> float:
    if (
        sleeve_exposure.empty
        or "sleeve" not in sleeve_exposure
        or "percent_of_max_sleeve" not in sleeve_exposure
    ):
        return float("nan")
    row = sleeve_exposure[sleeve_exposure["sleeve"].astype(str) == sleeve]
    if row.empty:
        return float("nan")
    return float(pd.to_numeric(row["percent_of_max_sleeve"].iloc[0], errors="coerce"))


def _lead_regime_label(current_state: object) -> str:
    growth_map = getattr(current_state, "growth_inflation_map", pd.DataFrame())
    if growth_map.empty:
        return "missing"
    row = growth_map.iloc[0]
    regime = str(row.get("regime", "missing"))
    probability = row.get("probability")
    try:
        return f"{regime} {float(probability):.0%}"
    except (TypeError, ValueError):
        return regime


def _top_scenario_read(scenario_lattice: pd.DataFrame) -> str:
    if scenario_lattice.empty:
        return "no scenario read"
    frame = scenario_lattice.copy()
    if "horizon" in frame and frame["horizon"].astype(str).eq("1m").any():
        frame = frame[frame["horizon"].astype(str).eq("1m")]
    if "rank" in frame:
        frame = frame.sort_values("rank")
    row = frame.iloc[0]
    scenario = str(row.get("scenario", "unknown scenario"))
    probability = row.get("probability")
    try:
        return f"{scenario} ({float(probability):.1%})"
    except (TypeError, ValueError):
        return scenario
