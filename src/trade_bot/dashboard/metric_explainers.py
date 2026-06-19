from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricExplainer:
    metric: str
    category: str
    plain_english: str
    calculation: str
    how_to_read: str
    caution: str
    aliases: tuple[str, ...] = ()


METRIC_EXPLAINERS: tuple[MetricExplainer, ...] = (
    MetricExplainer(
        metric="CAGR",
        category="Performance",
        plain_english="Compounded annual growth rate: the annual return that would compound to the same ending value.",
        calculation="Ending equity divided by starting equity, annualized over the length of the test.",
        how_to_read="Higher is better, but only after checking drawdown, turnover, and robustness.",
        caution="A high CAGR can come from one lucky period or one concentrated theme.",
        aliases=("cagr", "median_cagr", "base_cagr", "overlay_cagr", "delta_cagr"),
    ),
    MetricExplainer(
        metric="Sharpe",
        category="Performance",
        plain_english="Return per unit of total volatility.",
        calculation="Average daily net return times 252 divided by annualized daily-return volatility.",
        how_to_read="Above 1 is useful for simple daily strategies; compare it across similar approaches.",
        caution="This implementation does not subtract a risk-free rate; Sharpe also penalizes upside volatility and can miss crash risk.",
        aliases=("sharpe", "median_sharpe", "base_sharpe", "overlay_sharpe", "delta_sharpe"),
    ),
    MetricExplainer(
        metric="Sortino",
        category="Performance",
        plain_english="Return per unit of downside volatility.",
        calculation="Average daily net return times 252 divided by annualized standard deviation of returns clipped at zero.",
        how_to_read="Useful when a strategy has upside jumps but controlled downside.",
        caution="It still depends on the historical sample and does not guarantee future tail control.",
        aliases=("sortino",),
    ),
    MetricExplainer(
        metric="Max Drawdown",
        category="Performance",
        plain_english="The worst peak-to-trough loss during the tested period.",
        calculation="Lowest percentage loss from a prior equity-curve high.",
        how_to_read="Less negative is better. This is the pain metric to compare against buy-and-hold.",
        caution="Future drawdowns can be worse than the historical maximum.",
        aliases=(
            "max_drawdown",
            "worst_drawdown",
            "current_drawdown",
            "base_max_drawdown",
            "overlay_max_drawdown",
            "best_max_drawdown",
        ),
    ),
    MetricExplainer(
        metric="Calmar",
        category="Performance",
        plain_english="Annual return divided by worst drawdown.",
        calculation="CAGR divided by absolute max drawdown.",
        how_to_read="Higher means more return per unit of drawdown pain.",
        caution="A high Calmar can be unstable when drawdown history is short.",
        aliases=("calmar", "median_calmar", "base_calmar", "overlay_calmar", "delta_calmar"),
    ),
    MetricExplainer(
        metric="Average Turnover",
        category="Trading",
        plain_english="How much of the portfolio changes on an average rebalance.",
        calculation="Average absolute change in portfolio weights.",
        how_to_read="Lower usually means fewer trades, less friction, and better account usability.",
        caution="Too-low turnover can also mean the system is slow to exit bad regimes.",
        aliases=("average_turnover", "delta_average_turnover"),
    ),
    MetricExplainer(
        metric="Growth of $1",
        category="Performance",
        plain_english="Window-rebased equity curve showing how one dollar would have grown in the selected period.",
        calculation="Each selected strategy is divided by its value at the window start.",
        how_to_read="Use it to compare recent windows without the long-history chart dominating the view.",
        caution="Short windows are diagnostic, not proof that a strategy is structurally better.",
        aliases=("growth_of_1", "growth_of_$1", "windowed_performance"),
    ),
    MetricExplainer(
        metric="Worst 1Y/3Y/5Y CAGR",
        category="Robustness",
        plain_english="The worst rolling-window annual return for that horizon.",
        calculation="Calculate rolling 1, 3, or 5 year windows and report the weakest CAGR.",
        how_to_read="This tells you how bad the strategy looked when started at unlucky times.",
        caution="It still depends on the historical regimes available in the data.",
        aliases=("worst_1y_cagr", "worst_3y_cagr", "worst_5y_cagr", "worst_cagr"),
    ),
    MetricExplainer(
        metric="Positive Window Rate",
        category="Robustness",
        plain_english="Share of rolling windows with a positive return.",
        calculation="Count positive rolling windows divided by all tested rolling windows.",
        how_to_read="Higher means the strategy worked across more starting dates.",
        caution="A strategy can have many small wins and a few unacceptable losses.",
        aliases=("positive_window_rate", "delta_positive_1y_window_rate"),
    ),
    MetricExplainer(
        metric="Walk-Forward Median CAGR",
        category="Robustness",
        plain_english="Median return in sequential holdout windows after an initial burn-in span.",
        calculation="Evaluate the fixed strategy in repeated test windows and take the median holdout CAGR.",
        how_to_read="Better regime-breadth evidence than a single full-history backtest.",
        caution="Current folds do not retrain or re-optimize parameters, so this is a walk-forward holdout diagnostic rather than a full model-selection engine.",
        aliases=("walk_forward_median_cagr",),
    ),
    MetricExplainer(
        metric="Walk-Forward Worst CAGR",
        category="Robustness",
        plain_english="Worst annualized return across walk-forward holdout folds.",
        calculation="Minimum CAGR among all sequential holdout test windows.",
        how_to_read="This is a forward-style failure-mode metric.",
        caution="Annualizing short holdouts can make losses look larger or smaller than the lived path; folds do not refit strategy parameters.",
        aliases=("walk_forward_worst_cagr",),
    ),
    MetricExplainer(
        metric="Walk-Forward Positive Rate",
        category="Robustness",
        plain_english="Share of walk-forward holdout folds with positive returns.",
        calculation="Positive holdout folds divided by all holdout folds.",
        how_to_read="Higher means the approach held up across more unseen test periods.",
        caution="It does not say how large the losing folds were, and it does not prove future adaptability.",
        aliases=("walk_forward_positive_rate",),
    ),
    MetricExplainer(
        metric="Promotion Decision",
        category="Research",
        plain_english="Research triage label for whether an experiment deserves more attention.",
        calculation="Rule-based classification from promotion score, robustness, drawdown, and fragility checks.",
        how_to_read="Promote means monitor and paper test; it is not live-trading approval.",
        caution="A promoted candidate can still be rejected later after forward testing.",
        aliases=("promotion_decision", "decision"),
    ),
    MetricExplainer(
        metric="Promotion Score",
        category="Research",
        plain_english="Composite experiment score used to rank candidates.",
        calculation="Weighted ranks of return, drawdown, Calmar, walk-forward strength, and regime behavior.",
        how_to_read="Useful for sorting experiments quickly before reading details.",
        caution="A composite score can hide the reason a strategy is fragile.",
        aliases=("promotion_score", "best_score", "median_score"),
    ),
    MetricExplainer(
        metric="Robustness Score",
        category="Research",
        plain_english="Composite score focused on stability across windows, walk-forward folds, and regimes.",
        calculation="Weighted ranks of rolling-window, walk-forward, and regime-resilience metrics.",
        how_to_read="Higher means less evidence of one-period overfitting.",
        caution="Robustness is historical breadth, not certainty.",
        aliases=("robustness_score",),
    ),
    MetricExplainer(
        metric="Role",
        category="Research",
        plain_english="How a strategy is intended to be used in the portfolio.",
        calculation="Manual experiment metadata: core, satellite, overlay, or operating-system candidate.",
        how_to_read="Core can carry more capital; satellite is opportunistic; overlay modifies sizing or risk.",
        caution="Role is a design label, not a guarantee of safety.",
        aliases=("role", "research_role"),
    ),
    MetricExplainer(
        metric="Left-Tail Regime Return",
        category="Regime Tests",
        plain_english="Performance during named stress or crisis windows.",
        calculation="Total return inside the historical windows tagged as left-tail regimes.",
        how_to_read="Less negative is better. This is directly tied to avoiding retirement-damaging drawdowns.",
        caution="Named regimes are sparse and may not match the next crisis.",
        aliases=("left_tail_regime_return", "left_tail_regime_cagr"),
    ),
    MetricExplainer(
        metric="Transition Regime Hit Rate",
        category="Regime Tests",
        plain_english="Share of market-transition windows where the strategy made money.",
        calculation="Positive transition-regime windows divided by all transition-regime windows.",
        how_to_read="Important for detecting whether the strategy survives leadership changes and policy whipsaw.",
        caution="A low value means the strategy may be late when market character changes.",
        aliases=("transition_regime_hit_rate", "transition_regime_return"),
    ),
    MetricExplainer(
        metric="Regime Positive Rate",
        category="Regime Tests",
        plain_english="Share of named regime windows with positive returns.",
        calculation="Positive named-regime windows divided by all named-regime windows.",
        how_to_read="Shows whether the strategy wins across varied market environments.",
        caution="A strategy can be good overall even with a mediocre regime hit rate if losses are controlled.",
        aliases=("regime_positive_rate",),
    ),
    MetricExplainer(
        metric="Risk Status",
        category="Current State",
        plain_english="Dashboard color/state summary of current market risk.",
        calculation="Derived from the price/ratio confirmation matrix plus SPY, QQQ, HYG, and VIXY stress checks.",
        how_to_read="Green allows normal risk, yellow asks for caution, red asks for capital preservation.",
        caution="Macro, event, scenario, and portfolio-risk layers affect the later trade-decision risk budget rather than this base risk-status color.",
        aliases=("risk_status", "risk"),
    ),
    MetricExplainer(
        metric="Risk Score",
        category="Current State",
        plain_english="Numerical estimate of current market risk pressure.",
        calculation="Weighted current-state signals scaled roughly from 0 to 1.",
        how_to_read="Higher means more evidence for reducing risk or requiring confirmation.",
        caution="Small changes are less important than direction, drivers, and threshold crossings.",
        aliases=("risk_score", "mean_risk_score"),
    ),
    MetricExplainer(
        metric="Severity",
        category="Current State",
        plain_english="How urgent the dashboard thinks today's action review is.",
        calculation="Points from risk state, scenario risk, news pressure, trade size, and open tickets.",
        how_to_read="Use it to separate do-nothing days from small-action or critical-action days.",
        caution="High severity means review; it does not force execution.",
        aliases=("severity",),
    ),
    MetricExplainer(
        metric="Max Change",
        category="Trade Decision",
        plain_english="Largest target-weight change in the current recommendation.",
        calculation="Maximum absolute difference between current and target weights.",
        how_to_read="Large values mean the recommendation deserves extra human review.",
        caution="Large changes can be caused by stale positions, not just a new signal.",
        aliases=("max_change", "max_position_change", "delta_weight"),
    ),
    MetricExplainer(
        metric="Risk Budget",
        category="Trade Decision",
        plain_english="How much of normal strategy risk is currently allowed.",
        calculation="Minimum of risk-status, scenario, event, and validated-macro multipliers, then multiplied by any portfolio-risk multiplier.",
        how_to_read="1.00 means normal size; 0.50 means half-size; near 0 means mostly defensive.",
        caution="It controls sizing, not whether the underlying forecast is right.",
        aliases=(
            "risk_budget",
            "risk_budget_multiplier",
            "portfolio_risk_multiplier",
            "scenario_event_macro_multiplier",
        ),
    ),
    MetricExplainer(
        metric="Posture Calibration",
        category="Trade Decision",
        plain_english="Bias check for whether the recommendation may be too defensive given constructive scenario evidence.",
        calculation="Rule-based status from risk-off, risk-on, fragile-upside, transition, event pressure, macro pressure, and proposed risk reduction.",
        how_to_read="Use warnings as a prompt to review upside participation before accepting a de-risking trade.",
        caution="This is a governance check, not an override; it does not force the system to add risk.",
        aliases=(
            "posture_calibration_status",
            "posture_calibration_signal",
            "posture_calibration_note",
        ),
    ),
    MetricExplainer(
        metric="Opportunity Pressure",
        category="Trade Decision",
        plain_english="How much constructive medium-term evidence is pushing against defensive sizing.",
        calculation="risk_on + fragile_upside + 0.5 * transition - risk_off - event_pressure - macro_pressure, clipped from 0 to 1.",
        how_to_read="Higher values mean defensive recommendations deserve extra upside-participation review.",
        caution="It is not a forecast return and should be read with price, breadth, credit, and drawdown evidence.",
        aliases=(
            "opportunity_pressure",
            "one_month_risk_on_probability",
            "constructive_scenario_probability",
            "current_risk_asset_weight",
            "target_risk_asset_weight",
            "target_defensive_weight",
        ),
    ),
    MetricExplainer(
        metric="1M Risk-Off Probability",
        category="Scenarios",
        plain_english="Scenario model probability assigned to one-month risk-off outcomes.",
        calculation="Sum of 1-month scenario probabilities whose bucket is risk-off.",
        how_to_read="Higher values should tighten position sizing and off-ramp rules.",
        caution="This is a model-implied probability, not an option-market probability.",
        aliases=("1m_risk_off", "one_month_risk_off_probability", "risk_off_probability"),
    ),
    MetricExplainer(
        metric="Scenario Probability",
        category="Scenarios",
        plain_english="Relative probability assigned to a future-state scenario.",
        calculation="Softmax-normalized score from scenario drivers such as trend, credit, breadth, and oil.",
        how_to_read="Use it to understand why the risk engine is sizing up or down.",
        caution="Scenario probabilities are directional rankings, not calibrated odds yet.",
        aliases=("probability",),
    ),
    MetricExplainer(
        metric="Risk Bucket",
        category="Scenarios",
        plain_english="Scenario grouping used by the risk engine.",
        calculation="Template metadata such as risk_on, transition, risk_off, or risk_on_fragile.",
        how_to_read="Risk-off and transition buckets reduce allowed risk more than constructive buckets.",
        caution="The bucket is only as good as the scenario drivers feeding it.",
        aliases=("risk_bucket",),
    ),
    MetricExplainer(
        metric="Event Pressure",
        category="News And Events",
        plain_english="How much current news/event risk is pressuring the risk budget.",
        calculation="Active event-risk items converted into a sizing multiplier input.",
        how_to_read="Higher values mean the system sees market-relevant event risk that needs confirmation.",
        caution="News can lead, lag, or be ignored by markets, so price confirmation matters.",
        aliases=("event_pressure",),
    ),
    MetricExplainer(
        metric="Urgency Score",
        category="News And Events",
        plain_english="How important and time-sensitive a news item appears to be.",
        calculation="Classification confidence, source priority, recency, category, and phase combined.",
        how_to_read="High urgency items can become active event-risk context.",
        caution="Urgency is not the same as trade direction.",
        aliases=("urgency_score",),
    ),
    MetricExplainer(
        metric="News Phase",
        category="News And Events",
        plain_english="Whether news is likely leading markets, confirming prices, or explaining past moves.",
        calculation="Text classification into leading_warning, coincident_confirmation, lagging_explanation, or uncertain.",
        how_to_read="Leading warnings should be watched before price confirms; lagging items carry less signal.",
        caution="Phase can be wrong when information is partially priced already.",
        aliases=("phase", "event_phase"),
    ),
    MetricExplainer(
        metric="Expected Shortfall 95",
        category="Risk Engine",
        plain_english="Average loss on days worse than the 95th percentile loss threshold.",
        calculation="Positive loss magnitude of the mean return at or below the 5th percentile in the tail window.",
        how_to_read="Lower is better. It is stricter than value-at-risk because it averages tail losses.",
        caution="It is backward-looking and can understate unseen crash behavior.",
        aliases=(
            "es_95",
            "expected_shortfall",
            "portfolio_expected_shortfall_95",
            "post_expected_shortfall_95",
            "pre_expected_shortfall_95",
            "max_expected_shortfall_95",
        ),
    ),
    MetricExplainer(
        metric="Max Stress Loss",
        category="Risk Engine",
        plain_english="Estimated portfolio loss under the worst configured stress scenario.",
        calculation="Apply configured group-level shock returns to current weights and report the largest positive loss magnitude.",
        how_to_read="Use it to see if the target portfolio violates drawdown guardrails before trading.",
        caution="Stress scenarios are incomplete by design; the next shock may be different.",
        aliases=("max_stress_loss", "post_max_stress_loss", "pre_max_stress_loss"),
    ),
    MetricExplainer(
        metric="Scenario-Weighted Stress Loss",
        category="Risk Engine",
        plain_english="Stress loss adjusted by current scenario probabilities.",
        calculation="Stress losses weighted by scenario-probability inputs and summed.",
        how_to_read="Useful when multiple future states are plausible and not all are equally likely.",
        caution="Bad probability calibration can make this too lenient or too strict.",
        aliases=(
            "scenario_weighted_stress_loss",
            "post_scenario_weighted_stress_loss",
            "pre_scenario_weighted_stress_loss",
            "max_scenario_weighted_stress_loss",
        ),
    ),
    MetricExplainer(
        metric="Equity Beta",
        category="Risk Engine",
        plain_english="Sensitivity of the portfolio to broad equity-market moves.",
        calculation="Regression-style exposure to the broad equity proxy, usually SPY.",
        how_to_read="1.00 behaves like full market beta; below 1 is less equity-sensitive.",
        caution="Beta can change quickly during crises when correlations rise.",
        aliases=("equity_beta", "portfolio_equity_beta", "post_equity_beta", "pre_equity_beta"),
    ),
    MetricExplainer(
        metric="AI Beta",
        category="Risk Engine",
        plain_english="Sensitivity to AI/semiconductor/mega-cap growth exposure.",
        calculation="Exposure to AI proxy assets such as SMH, SOXX, QQQ, and related names.",
        how_to_read="High AI beta is acceptable only when AI leadership and risk budget both confirm.",
        caution="This is the key concentration risk if AI capex or mega-cap leadership reprices.",
        aliases=("ai_beta", "portfolio_ai_beta", "post_ai_beta", "pre_ai_beta", "max_ai_beta"),
    ),
    MetricExplainer(
        metric="Correlation Shift",
        category="Risk Engine",
        plain_english="How much recent asset correlation differs from longer-term correlation.",
        calculation="Short-lookback average correlation minus long-lookback average correlation.",
        how_to_read="Positive shifts can mean diversification is failing when it is needed most.",
        caution="Correlation can spike after losses have already started.",
        aliases=(
            "correlation_shift",
            "average_correlation_short",
            "average_correlation_long",
            "correlation_regime_shift",
        ),
    ),
    MetricExplainer(
        metric="Marginal Risk Contribution",
        category="Risk Engine",
        plain_english="How much each holding contributes to total portfolio risk.",
        calculation="Weight times covariance-with-portfolio divided by portfolio variance, with volatility contribution annualized.",
        how_to_read="Use it to find positions that dominate risk even if their dollar weights look modest.",
        caution="Contribution estimates depend on the covariance lookback.",
        aliases=(
            "risk_contribution_pct",
            "annualized_vol_contribution",
            "marginal_risk_contribution",
        ),
    ),
    MetricExplainer(
        metric="HHI Concentration",
        category="Risk Engine",
        plain_english="Portfolio concentration score based on squared position weights.",
        calculation="Sum of squared weights across holdings.",
        how_to_read="Higher means fewer positions dominate the portfolio.",
        caution="Low HHI does not guarantee true diversification if positions share the same factor beta.",
        aliases=("max_concentration_hhi", "concentration_hhi"),
    ),
    MetricExplainer(
        metric="Vol-Adjusted Momentum Score",
        category="Signals",
        plain_english="Momentum score normalized by realized volatility for a tradable asset.",
        calculation="Lookback return with skip-period logic divided by realized volatility.",
        how_to_read="Positive is supportive, negative is adverse, near zero is mixed.",
        caution="Fast shocks can outrun vol-adjusted momentum trend signals.",
        aliases=("momentum_state_score", "momentum_state_label"),
    ),
    MetricExplainer(
        metric="Signal Inclusion Delta",
        category="Signals",
        plain_english="Backtest impact of adding a candidate signal or macro group.",
        calculation="Overlay metric minus base metric, such as delta CAGR or delta Sharpe.",
        how_to_read="Positive deltas with drawdown improvement are most interesting.",
        caution="A useful diagnostic signal may still be too noisy for direct sizing.",
        aliases=(
            "delta_cagr",
            "delta_sharpe",
            "delta_calmar",
            "max_drawdown_improvement",
            "active_day_rate",
        ),
    ),
)


def metric_help(metric_name: str) -> str | None:
    explainer = metric_detail(metric_name)
    if explainer is None:
        return None
    return (
        f"{explainer.plain_english}\n\n"
        f"How to read: {explainer.how_to_read}\n\n"
        f"Watch out: {explainer.caution}"
    )


def metric_detail(metric_name: str) -> MetricExplainer | None:
    return _EXPLAINER_BY_KEY.get(_normalize_key(metric_name))


def metric_categories() -> tuple[str, ...]:
    return tuple(dict.fromkeys(explainer.category for explainer in METRIC_EXPLAINERS))


def metric_guide_frame(
    *,
    category: str | None = None,
    search: str = "",
) -> pd.DataFrame:
    rows = [
        {
            "metric": explainer.metric,
            "category": explainer.category,
            "plain_english": explainer.plain_english,
            "calculation": explainer.calculation,
            "how_to_read": explainer.how_to_read,
            "caution": explainer.caution,
            "aliases": ", ".join(explainer.aliases),
        }
        for explainer in METRIC_EXPLAINERS
    ]
    frame = pd.DataFrame(rows)
    if category:
        frame = frame[frame["category"] == category]
    query = search.strip().lower()
    if query:
        haystack = frame.astype(str).agg(" ".join, axis=1).str.lower()
        frame = frame[haystack.str.contains(re.escape(query), na=False)]
    return frame.reset_index(drop=True)


def _build_explainer_lookup() -> dict[str, MetricExplainer]:
    lookup: dict[str, MetricExplainer] = {}
    for explainer in METRIC_EXPLAINERS:
        for key in (explainer.metric, *explainer.aliases):
            lookup[_normalize_key(key)] = explainer
    return lookup


def _normalize_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("$", " dollar ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


_EXPLAINER_BY_KEY = _build_explainer_lookup()
