from __future__ import annotations

from datetime import date
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

CATEGORY_LABEL_COLUMNS = {
    "operability_label",
    "monitoring_readiness_label",
    "growth_utility_tier",
    "after_tax_growth_utility_tier",
}

CATEGORY_LABELS = {
    "paper_operable": "Paper operable",
    "weekly_cadence": "Weekly cadence",
    "weekly_large_moves": "Weekly cadence, large moves",
    "review_churn": "Review churn",
    "review_large_moves": "Review large moves",
    "too_twitchy": "Too twitchy",
    "paper_ready": "Paper ready",
    "paper_candidate": "Paper candidate",
    "blocked": "Blocked",
    "growth_champion_candidate": "Growth champion candidate",
    "growth_challenger_candidate": "Growth challenger candidate",
    "growth_watchlist": "Growth watchlist",
    "growth_research_only": "Growth research only",
    "growth_reject_hard_drawdown": "Growth reject: hard drawdown",
}


def _format_category_label(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    normalized = value.strip()
    if not normalized:
        return value
    return CATEGORY_LABELS.get(normalized, normalized.replace("_", " ").title())


def _format_category_columns(display: pd.DataFrame) -> pd.DataFrame:
    for column in CATEGORY_LABEL_COLUMNS:
        if column in display:
            display[column] = display[column].map(_format_category_label)
    return display


def _display_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    display = metrics.copy()
    currency_columns = [
        "terminal_wealth_15y",
        "terminal_wealth_with_contributions_15y",
        "after_tax_final_equity",
        "after_tax_terminal_wealth_15y",
        "after_tax_terminal_wealth_with_contributions_15y",
        "total_tax_liability",
        "total_tax_benefit",
        "net_estimated_tax_paid",
        "realized_short_term_gain",
        "realized_long_term_gain",
        "realized_loss_harvested",
        "wash_sale_disallowed_loss",
        "loss_carryforward_end",
        "ideal_final_equity",
        "actual_final_equity",
        "shortfall_dollars",
    ]
    for column in currency_columns:
        if column in display:
            display[column] = display[column].map(_format_currency)
    percent_columns = [
        "cagr",
        "median_cagr",
        "median_max_drawdown",
        "median_walk_forward_positive_rate",
        "median_left_tail_regime_return",
        "median_turnover",
        "worst_cagr",
        "best_cagr",
        "momentum_6m_skip_1w",
        "annualized_volatility",
        "realized_vol_3m",
        "max_drawdown",
        "worst_drawdown",
        "best_day",
        "worst_day",
        "return_1d",
        "return_1w",
        "return_1m",
        "return_3m",
        "drawdown",
        "total_return",
        "current_drawdown",
        "coverage",
        "average_turnover",
        "median_material_turnover",
        "max_single_day_turnover",
        "total_transaction_cost",
        "positive_window_rate",
        "return",
        "daily_return",
        "cumulative_return",
        "benchmark_return",
        "benchmark_cumulative_return",
        "excess_return",
        "gross_exposure",
        "net_exposure",
        "snapshot_cagr",
        "snapshot_max_drawdown",
        "snapshot_average_turnover",
        "risk_asset_return",
        "defensive_return",
        "oil_complex_return",
        "credit_relative_return",
        "vixy_return",
        "spy_return",
        "qqq_return",
        "primary_strategy_return",
        "best_strategy_return",
        "probability",
        "percentile_5y",
        "pct_change_1w",
        "pct_change_2w",
        "pct_change_1m",
        "pct_change_3m",
        "pct_change_12m",
        "range_position_1y",
        "usable_share",
        "best_cagr",
        "best_max_drawdown",
        "active_day_rate",
        "base_cagr",
        "overlay_cagr",
        "raw_cagr",
        "capped_cagr",
        "delta_cagr",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "raw_max_drawdown",
        "capped_max_drawdown",
        "delta_max_drawdown",
        "max_drawdown_improvement",
        "delta_worst_1y_cagr",
        "delta_worst_3y_cagr",
        "delta_worst_5y_cagr",
        "raw_left_tail_regime_return",
        "capped_left_tail_regime_return",
        "delta_left_tail_regime_return",
        "proven_relevance",
        "current_activation",
        "previous_30d_activation",
        "previous_90d_activation",
        "change_30d",
        "change_90d",
        "best_cagr",
        "best_max_drawdown",
        "median_cagr",
        "median_max_drawdown",
        "median_reentry_score",
        "median_average_turnover",
        "median_delta_promotion_score",
        "median_delta_cagr",
        "median_delta_max_drawdown",
        "median_delta_calmar",
        "median_delta_reentry_score",
        "median_delta_average_turnover",
        "cagr_win_rate",
        "drawdown_win_rate",
        "reentry_win_rate",
        "churn_win_rate",
        "promotion_win_rate",
        "calmar_win_rate",
        "net_evidence_score",
        "delta_reentry_score",
        "mean_delta_cagr",
        "mean_delta_max_drawdown",
        "mean_delta_turnover",
        "mean_delta_walk_forward_positive_rate",
        "mean_delta_left_tail_regime_return",
        "delta_positive_1y_window_rate",
        "delta_average_turnover",
        "raw_average_turnover",
        "capped_average_turnover",
        "raw_walk_forward_positive_rate",
        "capped_walk_forward_positive_rate",
        "delta_walk_forward_positive_rate",
        "promotion_win_rate",
        "drawdown_win_rate",
        "calmar_win_rate",
        "current_weight",
        "scenario_adjusted_weight",
        "delta_weight",
        "target_weight",
        "weight",
        "one_month_risk_off_probability",
        "one_month_transition_probability",
        "one_month_fragile_upside_probability",
        "one_month_risk_on_probability",
        "constructive_scenario_probability",
        "current_risk_asset_weight",
        "target_risk_asset_weight",
        "target_defensive_weight",
        "opportunity_pressure",
        "event_pressure",
        "macro_pressure",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "worst_regime_return",
        "worst_regime_cagr",
        "median_regime_return",
        "median_regime_cagr",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "transition_regime_return",
        "regime_positive_rate",
        "transition_regime_hit_rate",
        "walk_forward_median_cagr",
        "walk_forward_worst_cagr",
        "walk_forward_positive_rate",
        "walk_forward_worst_drawdown",
        "risk_weight",
        "risk_weight_before_1m",
        "risk_weight_at_event",
        "risk_weight_after_1m",
        "risk_weight_change",
        "defensive_weight",
        "drawdown_at_event",
        "forward_return_3m",
        "total_change",
        "average_weight",
        "max_weight",
        "average_risk_weight",
        "min_risk_weight",
        "latest_risk_weight",
        "low_risk_day_rate",
        "spy_ytd_large_move_share",
        "latest_percentile",
        "drawdown_recovery_return",
        "after_tax_cagr",
        "after_tax_max_drawdown",
        "after_tax_drawdown_recovery_return",
        "after_tax_drawdown_soft_penalty",
        "after_tax_drawdown_hard_penalty",
        "short_term_gain_share",
        "annualized_factor_volatility",
        "return_contribution",
        "absolute_contribution_share",
        "risk_contribution_pct",
        "strategy_arithmetic_return",
        "factor_model_r_squared",
        "residual_return_contribution",
        "residual_contribution_share",
        "residual_annualized_volatility",
        "residual_variance_share",
        "dominant_factor_share",
        "full_r_squared",
        "recent_r_squared",
        "r_squared_drop",
        "full_residual_volatility",
        "recent_residual_volatility",
        "shortfall_return",
        "ideal_cumulative_return",
        "actual_cumulative_return",
        "tracking_error",
        "price_slippage_pct",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(_format_percent)
    for column in [
        "sharpe",
        "sortino",
        "calmar",
        "median_sharpe",
        "median_calmar",
        "snapshot_sharpe",
        "snapshot_calmar",
        "years",
        "final_equity",
        "windows",
        "score",
        "z_score_5y",
        "change_1w",
        "change_2w",
        "change_1m",
        "change_3m",
        "change_12m",
        "short_move_z_1m",
        "change_acceleration_1m_vs_3m",
        "slope_1m",
        "slope_3m",
        "realized_vol_1m",
        "realized_vol_3m",
        "reversal_pressure",
        "risk_score",
        "mean_risk_score",
        "best_calmar",
        "iteration_rank",
        "promotion_score",
        "monitoring_readiness_score",
        "confidence_score",
        "benchmark_knockout_score",
        "raw_promotion_score",
        "capped_promotion_score",
        "delta_promotion_score",
        "mean_delta_promotion_score",
        "selection_adjusted_promotion_score",
        "overfit_risk_score",
        "selection_pressure",
        "holdout_decay",
        "holdout_fragility",
        "left_tail_penalty",
        "drawdown_penalty",
        "best_score",
        "best_confidence",
        "median_score",
        "median_benchmark_score",
        "median_readiness_score",
        "urgency_score",
        "confidence",
        "source_priority",
        "latest_pressure",
        "pressure_threshold",
        "risk_multiplier",
        "base_sharpe",
        "overlay_sharpe",
        "delta_sharpe",
        "base_calmar",
        "overlay_calmar",
        "raw_calmar",
        "capped_calmar",
        "delta_calmar",
        "mean_delta_calmar",
        "usable_days",
        "active_days",
        "risk_budget_multiplier",
        "scenario_event_macro_multiplier",
        "portfolio_risk_multiplier",
        "robustness_score",
        "growth_constrained_utility_score",
        "after_tax_calmar",
        "after_tax_growth_constrained_utility_score",
        "after_tax_wealth_multiple_vs_spy",
        "after_tax_wealth_multiple_vs_qqq",
        "tax_drag_bps_per_year",
        "wealth_multiple_vs_spy",
        "wealth_multiple_vs_qqq",
        "operability_score",
        "reentry_score",
        "material_trade_days_per_year",
        "mean_days_between_material_trades",
        "median_reentry_days",
        "reentry_cycles",
        "holdout_folds",
        "tested_regimes",
        "portfolio_equity_beta",
        "portfolio_ai_beta",
        "pre_equity_beta",
        "post_equity_beta",
        "max_equity_beta",
        "pre_ai_beta",
        "post_ai_beta",
        "max_ai_beta",
        "beta",
        "pre_beta",
        "post_beta",
        "full_beta",
        "recent_beta",
        "beta_change",
        "beta_drift",
        "abs_beta_drift",
        "residual_volatility_ratio",
        "correlation",
        "correlation_shift",
        "average_correlation_short",
        "average_correlation_long",
        "correlation_regime_shift",
        "scenario_risk_multiplier",
        "regime_instability_score",
        "component_score",
        "weight",
    ]:
        if column in display:
            display[column] = display[column].map(_format_decimal)
    for column in [
        "pre_risk_target_weight",
        "risk_adjusted_weight",
        "risk_engine_delta",
        "portfolio_expected_shortfall_95",
        "portfolio_max_stress_loss",
        "pre_expected_shortfall_95",
        "post_expected_shortfall_95",
        "max_expected_shortfall_95",
        "pre_max_stress_loss",
        "post_max_stress_loss",
        "max_stress_loss",
        "pre_scenario_weighted_stress_loss",
        "post_scenario_weighted_stress_loss",
        "max_scenario_weighted_stress_loss",
        "scenario_probability_weight",
        "pre_shock_return",
        "pre_loss",
        "post_shock_return",
        "post_loss",
        "risk_engine_delta_loss",
        "confidence_level",
        "value_at_risk",
        "expected_shortfall",
        "worst_day",
        "portfolio_annualized_volatility",
        "factor_annualized_volatility",
        "realized_volatility",
        "risk_contribution_pct",
        "annualized_vol_contribution",
        "post_absolute_beta_share",
        "risk_off_probability",
        "transition_probability",
        "fragile_upside_probability",
        "risk_on_probability",
        "ai_unwind_probability",
        "credit_stress_probability",
        "inflation_oil_probability",
        "max_single_asset_weight",
        "max_concentration_hhi",
        "max_expected_shortfall_95",
        "max_stress_loss",
        "max_scenario_weighted_stress_loss",
        "min_defensive_weight",
        "post_defensive_weight",
        "drawdown_soft_penalty",
        "drawdown_hard_penalty",
    ]:
        if column in display:
            display[column] = display[column].map(_format_percent)
    display = _format_category_columns(display)
    return display


def _display_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    currency_columns = [
        "reference_price",
        "actual_price",
        "limit_low",
        "limit_high",
        "target_notional",
        "actual_notional",
        "notional_gap",
        "min_notional",
        "max_notional",
        "price",
        "notional",
        "fees",
        "net_cash_deployed",
        "current_notional",
        "delta_notional",
        "account_value",
        "largest_delta_notional",
        "cost_basis_per_share",
        "total_cost_basis",
        "proceeds",
        "cost_basis",
        "realized_gain_loss",
        "taxable_gain_loss",
        "unrealized_gain_loss",
        "current_value",
        "current_price",
        "wash_sale_adjustment",
    ]
    share_columns = ["min_shares", "max_shares", "quantity", "net_quantity", "remaining_quantity"]
    percent_columns = [
        "current_weight",
        "scenario_adjusted_weight",
        "target_weight",
        "delta_weight",
        "current_cash_weight",
        "target_cash_weight",
        "max_abs_delta",
        "largest_delta_weight",
        "min_trade_weight",
        "unrealized_loss_pct",
        "price_slippage_pct",
    ]
    for column in currency_columns:
        if column in display:
            display[column] = display[column].map(_format_currency)
    for column in share_columns:
        if column in display:
            display[column] = display[column].map(_format_shares)
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(_format_percent)
    return display


def _format_percent(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.2%}"


def _format_decimal(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:,.2f}"


def _format_currency(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"${numeric:,.2f}"


def _format_shares(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:,.4f}"


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _safe_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("America/Denver")


def _result_date_bounds(results: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for result in results.values():
        equity = result.equity.dropna()
        if equity.empty:
            continue
        starts.append(pd.Timestamp(equity.index.min()))
        ends.append(pd.Timestamp(equity.index.max()))
    if not starts or not ends:
        today = pd.Timestamp(date.today())
        return today, today
    return min(starts), max(ends)


def _default_strategy_selection(strategy_names: list[str]) -> list[str]:
    preferred = [
        "drawdown_managed_dual_momentum",
        "vol_target_dual_momentum",
        "dual_momentum_core",
        "buy_hold_spy",
        "buy_hold_qqq",
    ]
    selected = [name for name in preferred if name in strategy_names]
    return selected or strategy_names[: min(4, len(strategy_names))]


def _window_start_from_preset(
    preset: str,
    *,
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    custom_start: date | None = None,
) -> pd.Timestamp:
    if preset == "30 days":
        start = latest - pd.DateOffset(days=30)
    elif preset == "90 days":
        start = latest - pd.DateOffset(days=90)
    elif preset == "6 months":
        start = latest - pd.DateOffset(months=6)
    elif preset == "1 year":
        start = latest - pd.DateOffset(years=1)
    elif preset == "3 years":
        start = latest - pd.DateOffset(years=3)
    elif preset == "5 years":
        start = latest - pd.DateOffset(years=5)
    elif preset == "YTD":
        start = pd.Timestamp(year=latest.year, month=1, day=1)
    elif preset == "Custom" and custom_start is not None:
        start = pd.Timestamp(custom_start)
    else:
        start = earliest
    return max(earliest, min(start, latest))
