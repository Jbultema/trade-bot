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
        metric="Time Underwater",
        category="Performance",
        plain_english=(
            "Share of evaluated days where the equity curve is below its prior high-water mark."
        ),
        calculation="Mean of days where drawdown is below zero.",
        how_to_read=(
            "High values mean the strategy often spends time below a prior peak; this can happen even "
            "when the drawdowns are shallow."
        ),
        caution=(
            "This is not the same as time spent in a severe drawdown. A strategy can have high time "
            "underwater and still have acceptable max drawdown."
        ),
        aliases=("Days below prior peak", "time_underwater", "underwater_rate"),
    ),
    MetricExplainer(
        metric="Ulcer Index",
        category="Performance",
        plain_english=(
            "Drawdown pain metric that combines depth and persistence of drawdowns."
        ),
        calculation="Square root of the average squared drawdown from prior equity highs.",
        how_to_read=(
            "Lower is better. Tiny below-peak days barely move it; deep and persistent drawdowns move it a lot."
        ),
        caution=(
            "It is still historical and sample-dependent, but it is usually more useful than raw days below peak."
        ),
        aliases=("ulcer_index", "pain_index"),
    ),
    MetricExplainer(
        metric="Variance Contribution",
        category="Attribution",
        plain_english="How much each proxy factor contributes to strategy return variance.",
        calculation="Covariance of the fitted factor return contribution with strategy returns, divided by strategy-return variance.",
        how_to_read=(
            "Positive values mean the factor explains strategy variance in the same direction; negative values "
            "mean it offsets strategy variance."
        ),
        caution=(
            "This can look similar to return contribution when the same factors both earned returns and drove volatility."
        ),
        aliases=("risk_contribution_pct", "variance_contribution", "risk contribution"),
    ),
    MetricExplainer(
        metric="Beta-Adjusted S&P Delta",
        category="Risk",
        plain_english=(
            "Approximate S&P 500-equivalent exposure after accounting for each holding's rolling beta."
        ),
        calculation="Sum of position weight times rolling beta to SPY over the configured lookback.",
        how_to_read="A 65% reading means the book behaves roughly like a 65% SPY allocation for broad-market moves.",
        caution="Rolling beta can change quickly in stress regimes and is only a linear approximation.",
        aliases=("beta_adjusted_spy_delta",),
    ),
    MetricExplainer(
        metric="Percent of Max Sleeve",
        category="Risk",
        plain_english="How much of a configured sleeve limit is currently being used.",
        calculation="Current sleeve weight divided by that sleeve's configured maximum exposure.",
        how_to_read="A 100% reading means the sleeve is at its current operating maximum.",
        caution="Sleeve limits are policy constraints, not proof that the exposure is attractive.",
        aliases=(
            "percent_of_max_sleeve",
            "stocks_percent_of_max_sleeve",
            "defensive_percent_of_max_sleeve",
            "gold_percent_of_max_sleeve",
            "crypto_percent_of_max_sleeve",
            "credit_percent_of_max_sleeve",
        ),
    ),
    MetricExplainer(
        metric="Factor R2",
        category="Attribution",
        plain_english="Share of the strategy's daily return variance explained by the proxy factor model.",
        calculation="1 minus residual variance divided by strategy-return variance.",
        how_to_read="High values mean the strategy is mostly explained by broad proxy factors; low values mean more residual strategy-specific behavior.",
        caution="A high R2 is not bad by itself, but it can reveal that several strategies are the same disguised factor bet.",
        aliases=("factor_model_r_squared", "factor_r2"),
    ),
    MetricExplainer(
        metric="Residual Share",
        category="Attribution",
        plain_english="Portion of absolute return contribution not explained by the proxy factor set.",
        calculation="Absolute residual strategy contribution divided by total absolute factor plus residual contribution.",
        how_to_read="Higher values suggest more unique behavior after broad beta, AI/growth, rates, credit, commodities, and volatility are considered.",
        caution="Residual behavior can be true skill, missing factors, noise, or overfit. Inspect robustness before trusting it.",
        aliases=("residual_contribution_share", "residual_share"),
    ),
    MetricExplainer(
        metric="Factor Decay",
        category="Attribution",
        plain_english="Whether recent factor exposure no longer resembles the full-history factor profile.",
        calculation="Compares recent beta, R2, and residual volatility against full-history attribution.",
        how_to_read="A flag means the strategy may be drifting, crowded, broken, or entering a new regime.",
        caution="Short windows can produce false alarms. Treat decay flags as review triggers, not automatic exits.",
        aliases=("drift_flag", "model_decay_flag", "beta_drift", "abs_beta_drift"),
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
        metric="Growth-Constrained Utility Score",
        category="Research",
        plain_english="Outcome score for accumulation accounts that value high terminal wealth while keeping drawdowns inside a survivable band.",
        calculation="Log-scaled 15-year wealth with contributions, adjusted for validation quality, soft/hard drawdown penalties, overfit risk, left-tail behavior, and churn.",
        how_to_read="Use it to find candidates where extra CAGR appears worth the additional drawdown burden.",
        caution="This is a research-selection score, not live trade approval; weak walk-forward or regime evidence can still disqualify a high-growth backtest.",
        aliases=("growth_constrained_utility_score", "outcome_utility", "growth_utility"),
    ),
    MetricExplainer(
        metric="Growth Utility Tier",
        category="Research",
        plain_english="Plain-language tier derived from growth-constrained utility and drawdown eligibility.",
        calculation="Rule-based label from utility score and the hard drawdown gate.",
        how_to_read="Champion and challenger tiers deserve paper-monitoring review before lower tiers.",
        caution="A tier can change when new experiments, benchmarks, or validation diagnostics are added.",
        aliases=("growth_utility_tier",),
    ),
    MetricExplainer(
        metric="15-Year Terminal Wealth With Contributions",
        category="Performance",
        plain_english="Projected account value after applying the strategy CAGR to the configured starting balance and annual contributions.",
        calculation="Starting value compounded by CAGR for 15 years plus end-of-year annual contributions grown at the same CAGR.",
        how_to_read="This is the practical retirement-accumulation outcome to compare across high-growth and lower-drawdown strategies.",
        caution="It assumes the historical CAGR repeats for planning math only; it is not a forecast.",
        aliases=(
            "15Y Wealth",
            "15 Year Wealth",
            "15-year wealth",
            "terminal_wealth_with_contributions_15y",
            "terminal_wealth_15y",
            "projected_wealth",
        ),
    ),
    MetricExplainer(
        metric="Outcome Planning Assumptions",
        category="Performance",
        plain_english="The account, contribution, horizon, and drawdown policy used by the Outcome Frontier.",
        calculation="Defaults come from src/trade_bot/DEFAULTS.py and are applied consistently to deterministic and simulated outcome views.",
        how_to_read="Use these settings to confirm the wealth math matches the real planning problem before comparing strategies.",
        caution="Changing assumptions changes the wealth ranking; rerun the refresh stack after changing defaults so stored scorecards stay aligned.",
        aliases=(
            "Starting Account",
            "Annual Contribution",
            "Horizon",
            "Soft / Hard DD",
            "Projection Mode",
            "planning_assumptions",
            "outcome_assumptions",
        ),
    ),
    MetricExplainer(
        metric="Bootstrap Wealth Range",
        category="Performance",
        plain_english="Sequence-aware wealth distribution from resampling the selected strategy's historical daily returns.",
        calculation="Block bootstrap daily returns for the configured horizon, add end-of-year contributions, and report terminal-wealth quantiles.",
        how_to_read="P10 is a worse-but-plausible sampled path, median is the center sampled path, and P90 is a better sampled path.",
        caution="This is historical return-sequence resampling, not a regime-conditioned forecast and not proof the future distribution is known.",
        aliases=(
            "Bootstrap P10 Wealth",
            "Bootstrap Median",
            "Bootstrap P90 Wealth",
            "terminal_wealth_p10",
            "terminal_wealth_p50",
            "terminal_wealth_p90",
            "sequence-aware outcome simulation",
        ),
    ),
    MetricExplainer(
        metric="Bootstrap Simulated Drawdown",
        category="Performance",
        plain_english="Path-level drawdown pain from the historical block-bootstrap simulations.",
        calculation="For each sampled path, compute max drawdown and Ulcer Index; the dashboard shows median simulated values.",
        how_to_read="Use this to see whether the terminal-wealth result depends on paths that would be hard to sit through.",
        caution="The simulation only resamples history and can miss unseen future drawdowns or changed regime behavior.",
        aliases=(
            "Median Sim DD",
            "Median Sim Ulcer",
            "bootstrap_max_drawdown",
            "bootstrap_ulcer_index",
            "simulated_drawdown",
        ),
    ),
    MetricExplainer(
        metric="Extra Wealth Versus Benchmark",
        category="Performance",
        plain_english="Projected wealth difference versus a benchmark under the same 15-year contribution assumptions.",
        calculation="Selected strategy projected wealth minus benchmark projected wealth.",
        how_to_read="Positive means the selected strategy projects more wealth than the benchmark under the planning assumptions; negative means it trails.",
        caution="This is scenario math from historical CAGR, not a forecast. It can swing sharply when CAGR estimates are close.",
        aliases=("Extra vs SPY", "Extra vs QQQ", "extra_wealth_vs_spy", "extra_wealth_vs_qqq"),
    ),
    MetricExplainer(
        metric="Drawdown Recovery Return",
        category="Performance",
        plain_english="Gain required to recover from the reported max drawdown.",
        calculation="1 divided by 1 minus drawdown depth, minus 1. A 20 percent drawdown requires a 25 percent rebound.",
        how_to_read="Use it to translate drawdown into the recovery burden you would need to sit through.",
        caution="The time required to recover can matter more than the percentage recovery alone.",
        aliases=("Recovery Needed", "drawdown_recovery_return", "recovery_return_required"),
    ),
    MetricExplainer(
        metric="After-Tax CAGR",
        category="Tax",
        plain_english="Estimated CAGR after applying configured taxable-account rates to realized gains, losses, wash-sale adjustments, and loss carryforwards.",
        calculation="Taxable backtest equity after year-end estimated tax cash flows, then the standard CAGR formula.",
        how_to_read="Compare this with pre-tax CAGR to see whether active trading survives taxable-account drag.",
        caution="This is an estimate from configured assumptions and reconstructed lots; broker lots and real tax advice still govern live decisions.",
        aliases=("after_tax_cagr",),
    ),
    MetricExplainer(
        metric="Tax Drag",
        category="Tax",
        plain_english="Annualized return lost to estimated taxable-account effects.",
        calculation="Pre-tax CAGR minus after-tax CAGR, shown in basis points per year.",
        how_to_read="Lower is better when returns are comparable; high drag means turnover or short-term realizations are eating the edge.",
        caution="A strategy can have high tax drag and still be worth monitoring if the after-tax terminal wealth remains strong.",
        aliases=("tax_drag_bps_per_year", "tax_drag"),
    ),
    MetricExplainer(
        metric="Wash-Sale Disallowed Loss",
        category="Tax",
        plain_english="Estimated losses that cannot currently reduce taxable gains because a similar replacement position was bought inside the wash-sale window.",
        calculation="Losses on sold lots multiplied by replacement quantity inside the configured wash-sale window, capped at the realized loss.",
        how_to_read="Large values mean the strategy is trying to harvest losses but re-entering too quickly or too similarly.",
        caution="The model uses configured substitute maps and is intentionally conservative; actual wash-sale treatment needs tax review.",
        aliases=("wash_sale_disallowed_loss",),
    ),
    MetricExplainer(
        metric="After-Tax Growth-Constrained Utility Score",
        category="Tax",
        plain_english="Growth-constrained outcome score recomputed with estimated after-tax CAGR and drawdown.",
        calculation="Same utility formula as the pre-tax score, but using after-tax CAGR and after-tax max drawdown.",
        how_to_read="Use this when comparing strategies for a taxable brokerage account rather than an IRA-like account.",
        caution="Do not compare taxable and pre-tax scores without checking the account profile and tax model status.",
        aliases=("after_tax_growth_constrained_utility_score", "after_tax_growth_utility"),
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
        metric="Regime Instability Score",
        category="Current State",
        plain_english="Watch-only estimate of whether market internals look statistically unstable.",
        calculation="Weighted blend of SPY +/-1% day share, realized volatility, cross-sectional dispersion, VIXY pressure, correlation shift, breadth/concentration, and credit stress.",
        how_to_read="Higher means the market may be in a transition regime even if headline trend has not broken.",
        caution="This is not currently allowed to change trade sizing; it needs ablation testing before becoming an allocation signal.",
        aliases=(
            "regime_instability_score",
            "regime_instability_state",
            "component_score",
            "spy_ytd_large_move_share",
        ),
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
