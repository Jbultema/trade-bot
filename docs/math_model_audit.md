# Math, Formula, And Model Audit

This document locks the current math and model semantics used by the app. It is
meant to prevent code drift: if a future change alters one of these formulas or
interpretations, update this document and the formula-contract tests in the same
change. It is a formula and semantics contract, not a daily market-status report.

Original audit date: 2026-06-17. Last docs review: 2026-06-21.

## Audit Result

Status: usable for research and paper trading, with explicit caveats.

The core backtest, performance, strategy, risk, scenario, and trade-decision math is internally consistent for research and paper trading. Selected-window best/worst-day stats come from the rebased equity window and exclude the prior close-to-close return on the first displayed date.

The main risk is not arithmetic inversion. The main risk is semantic overclaiming. Several app concepts are heuristic policy rules, not calibrated probability models:

- Scenario probabilities are softmax-normalized model scores, not empirically calibrated odds.
- News urgency is deterministic triage, not a forecast of event returns.
- Walk-forward metrics are fixed-strategy holdout diagnostics, not full retrain-and-reselect walk-forward optimization.
- Macro inclusion tests are exploratory and are not revision-safe yet.
- Sharpe and Sortino use raw net strategy returns and do not subtract a risk-free rate.

## Locked Assumptions

- Long-only only. Negative weights are clipped to zero.
- No shorting and no derivatives in the default operating system.
- Cash or T-bill residual is implicit unless a defensive ticker such as `BIL` is explicitly present.
- Signals are shifted by `signal_lag_days`, default 1, before returns are calculated.
- Daily trading-day annualization uses 252 trading days.
- Transaction costs are modeled as portfolio turnover times basis points.
- Strategy recommendations are review targets, not automatic execution instructions.

## Backtest Engine

Source: `src/trade_bot/backtest/engine.py`

Daily asset return:

```text
asset_return[t] = price[t] / price[t-1] - 1
```

Implementation fills forward prices, uses `pct_change(fill_method=None)`, and fills the first return with 0.

Execution weights:

```text
execution_weight[t] = normalized_target_weight[t - signal_lag_days]
```

Long-only normalization:

```text
clean_weight = max(raw_weight, 0)
if sum(clean_weight) > 1:
    clean_weight = clean_weight / sum(clean_weight)
else:
    residual is cash or defensive exposure if another layer explicitly assigns it
```

Portfolio return:

```text
gross_return[t] = sum_i execution_weight_i[t] * asset_return_i[t]
turnover[t] = sum_i abs(execution_weight_i[t] - execution_weight_i[t-1])
transaction_cost[t] = turnover[t] * transaction_cost_bps / 10000
net_return[t] = gross_return[t] - transaction_cost[t]
equity[t] = initial_capital * product(1 + net_return)
```

Volatility targeting:

```text
realized_portfolio_vol[t] = std(portfolio_return over lookback) * sqrt(252)
scale[t] = target_annual_vol / realized_portfolio_vol[t]
scale[t] = clip(scale[t], 0, max_leverage)
execution_scale[t] = scale[t-1]
```

The one-day shift is required to avoid using same-day realized returns.

Drawdown control:

```text
shadow_equity[t] = product(1 + pre_control_strategy_return)
rolling_drawdown[t] = shadow_equity[t] / rolling_max(shadow_equity, lookback) - 1
scale[t] = risk_multiplier if rolling_drawdown[t] <= max_drawdown else 1
execution_scale[t] = scale[t-1]
```

The one-day shift is required to avoid avoiding the trigger-day loss.

## Performance Metrics

Source: `src/trade_bot/backtest/metrics.py`

Let `r[t]` be daily net returns and `E[t]` be equity.

Years:

```text
years = max((last_date - first_date).days / 365.25, 1 / 365.25)
```

Initial equity is reconstructed to avoid losing the first return:

```text
initial_equity = E[first] / (1 + r[first])
```

CAGR:

```text
CAGR = (E[last] / initial_equity) ** (1 / years) - 1
```

Annualized volatility:

```text
annualized_volatility = std(r) * sqrt(252)
```

Sharpe:

```text
Sharpe = mean(r) * 252 / annualized_volatility
```

Current implementation does not subtract a risk-free rate.

Sortino:

```text
downside_return[t] = min(r[t], 0)
downside_volatility = std(downside_return) * sqrt(252)
Sortino = mean(r) * 252 / downside_volatility
```

This is a zero-threshold downside-volatility implementation.

Drawdown:

```text
drawdown[t] = E[t] / cumulative_max(E)[t] - 1
max_drawdown = min(drawdown)
```

Calmar:

```text
Calmar = CAGR / abs(max_drawdown)
```

Average turnover and total transaction cost:

```text
average_turnover = mean(turnover)
total_transaction_cost = sum(transaction_cost)
```

Dashboard selected-window performance uses the selected equity window itself:

```text
growth_of_1[t] = equity[t] / equity[first_window_date]
window_return = equity[last_window_date] / equity[first_window_date] - 1
window_years = max((last_window_date - first_window_date).days / 365.25, 1 / 365.25)
window_cagr = growth_of_1[last_window_date] ** (1 / window_years) - 1
window_daily_return[t] = pct_change(growth_of_1 within selected window)
window_daily_return[first_window_date] = 0
window_annualized_volatility = std(window_daily_return) * sqrt(252)
window_sharpe = mean(window_daily_return) * 252 / window_annualized_volatility
window_calmar = window_cagr / abs(window_max_drawdown)
```

Selected-window metrics intentionally rebase on the first selected date and do not include the
pre-window close-to-close return. Single-observation windows report zero volatility, Sharpe, and
Calmar rather than `NaN`.

## Strategy Signals

Source: `src/trade_bot/strategies/momentum.py` and `src/trade_bot/features/indicators.py`

Lookback return with skip:

```text
shifted_price[t] = price[t - skip_days]
lookback_return[t] = shifted_price[t] / shifted_price[t - lookback_days] - 1
```

Moving-average trend:

```text
moving_average[t] = mean(price over moving_average_days)
absolute_trend_active[t] = price[t] > moving_average[t]
```

Realized volatility:

```text
realized_volatility[t] = std(daily_return over lookback) * sqrt(252)
```

Relative momentum selects the top `top_n` assets by one of:

```text
return = lookback_return
risk_adjusted_return = lookback_return / realized_volatility
return_trend_quality = lookback_return + 0.25 * (price / moving_average - 1)
```

Dual momentum adds:

```text
eligible = lookback_return > min_return
```

Weighting options:

```text
equal: selected assets receive equal raw weight
inverse_volatility: raw_weight_i = 1 / realized_volatility_i
momentum_score: raw_weight_i = max(lookback_return_i, 0)
risk_adjusted_score: raw_weight_i = max(ranking_value_i, 0)
```

If no asset qualifies and a defensive ticker is configured, the strategy assigns 100 percent to the defensive ticker.

## Current-State Signals

Source: `src/trade_bot/research/current_state.py`

vol-adjusted momentum score:

```text
momentum = lookback_return(price, lookback_days=126, skip_days=5)
volatility = realized_volatility(daily_returns, lookback=63)
momentum_state_score = momentum / volatility
```

Vol-Adjusted Momentum state:

```text
bullish if momentum_state_score >= 0.60
bearish if momentum_state_score <= -0.40
neutral otherwise
```

Confirmation matrix:

- Absolute signals map `bullish` to 1, `bearish` to -1, and neutral/insufficient to 0.
- Inverse pressure signals, such as `VIXY` and `UUP`, multiply that score by -1.
- Relative signals run Vol-Adjusted Momentum on ratios such as `HYG / LQD`, `RSP / SPY`, and `SMH / SPY`.

Current risk score:

```text
risk_on_score = mean(confirmation_scores)
raw_risk = 0.5 - risk_on_score / 2
```

Additive stress checks:

```text
+0.10 if SPY drawdown < -8 percent
+0.10 if QQQ drawdown < -10 percent
+0.10 if HYG is bearish
+0.15 if VIXY is bullish
risk_score = clip(raw_risk, 0, 1)
```

Risk status:

```text
green if risk_score < 0.25
yellow if risk_score < 0.45
orange if risk_score < 0.65
red otherwise
```

Macro, event, scenario, and portfolio-risk layers do not set this base color directly. They affect the later trade-decision risk budget.

## Scenario Lattice

Source: `src/trade_bot/research/future_scenarios.py`

Scenario drivers are clipped to `[-1, 1]` and summarize market trend, breadth, credit, AI leadership, concentration, volatility/liquidity, energy/inflation relief, defensive pressure, duration support, drawdown resilience, and style rotation.

Each scenario has a template score:

```text
score = base_score + horizon_bias[horizon] + risk_tilt * (risk_score - 0.5)
score += sum(driver_weight_j * driver_score_j)
```

Within each horizon, scenario probabilities are softmax-normalized:

```text
probability_i = exp(score_i / temperature) / sum_j exp(score_j / temperature)
temperature = 0.70
```

These are model-implied relative probabilities for ranking and sizing. They are not calibrated market odds.

## Macro State And Inclusion Tests

Source: `src/trade_bot/research/macro_state.py` and `src/trade_bot/research/signal_inclusion.py`

Macro signal table:

```text
z_score_5y = (latest_value - mean(history)) / std(history)
percentile_5y = share(history <= latest_value)
change_N = latest_value - prior_value
pct_change_N = latest_value / prior_value - 1, only when signs match and prior != 0
short_move_z = z-score of latest 21-day level move against rolling move history
change_acceleration = change_1m - change_3m / 3
```

Macro risk score:

```text
direction = 1 if risk_off_when_rising else -1 if risk_on_when_rising else 0
scaled_z = tanh(z_score / 2)
scaled_change = tanh(change_component)
scaled_short_move = tanh(short_move_z / 2)
scaled_acceleration = tanh(change_acceleration)
risk_score = clip(direction * (
    0.45 * scaled_z
  + 0.25 * scaled_change
  + 0.20 * scaled_short_move
  + 0.10 * scaled_acceleration
), -1, 1)
```

Signal inclusion pressure uses publication lag discipline:

```text
available_date = observation_date + publication_lag_days
z_score = (aligned_value - rolling_mean) / rolling_std
if risk_on_when_rising: z_score = -z_score
if neutral: z_score = 0
pressure = clip(z_score, -2, 2) / 2
```

Risk-reduction-only overlay:

```text
active = pressure > pressure_threshold
risk_weights[active] *= risk_multiplier
freed_weight moves to defensive_ticker
```

Current caveat: FRED data is not revision-safe. Inclusion tests are evidence for paper candidates, not final allocation authority.

## News And Event Risk

Source: `src/trade_bot/research/news_monitor.py` and `src/trade_bot/research/event_risk.py`

News classification is deterministic keyword/channel triage. It maps text into:

- category
- direction: escalation, deescalation, or uncertain
- confidence
- risk channels
- candidate proxies
- phase: leading_warning, coincident_confirmation, lagging_explanation, or phase_uncertain

News urgency:

```text
urgency = 0.35 * confidence
        + 0.15 * clipped_source_priority
        + category_weight
        + phase_weight
        + direction_weight
        + recency_weight
urgency = clip(urgency, 0, 1)
```

Activation:

```text
event_risk_generated if category != unclassified
    and urgency >= activation_threshold
    and source_priority >= 3
```

Historical event windows:

```text
anchor = first trading date at or after event date
window_return = value[end_position] / value[start_position] - 1
```

Current caveat: news urgency is not a directional trade forecast. It controls whether an item becomes event-risk context that needs market confirmation.

## Portfolio Risk Engine

Source: `src/trade_bot/portfolio/risk.py`

Scenario budget uses the one-month scenario lattice when available:

```text
risk_off = sum(probability where risk_bucket contains risk_off)
transition = sum(probability where risk_bucket == transition)
fragile = sum(probability where risk_bucket == risk_on_fragile)
risk_on = sum(probability where risk_bucket == risk_on)
```

Scenario risk multiplier:

```text
scenario_multiplier = 1 - 0.55 * risk_off - 0.20 * transition - 0.15 * fragile
scenario_multiplier = clip(scenario_multiplier, DEFAULT_SCENARIO_MIN_MULTIPLIER, 1.0)
```

Scenario-adjusted constraints:

```text
max_equity_beta = max(0.35, base_max_equity_beta * (1 - 0.35 * risk_off - 0.15 * transition))
max_ai_beta = max(0.20, base_max_ai_beta * (1 - 0.45 * ai_unwind - 0.30 * risk_off - 0.15 * fragile))
max_expected_shortfall_95 = max(0.0125, base_max_es95 * (1 - 0.35 * risk_off - 0.15 * transition))
max_stress_loss = max(0.06, base_max_stress_loss * (1 - 0.35 * risk_off - 0.15 * transition))
min_defensive_weight = clip(base_min_defensive + 0.40 * risk_off + 0.20 * transition + 0.10 * fragile + 0.10 * ai_unwind, 0, 0.65)
max_single_asset_weight = max(0.25, base_max_single * (1 - 0.25 * risk_off - 0.10 * transition))
max_concentration_hhi = max(0.22, base_max_hhi * (1 - 0.25 * risk_off - 0.10 * transition))
```

Factor beta:

```text
beta_factor = cov(portfolio_return, factor_proxy_return) / var(factor_proxy_return)
correlation = corr(portfolio_return, factor_proxy_return)
```

AI beta proxy:

```text
ai_beta = max(
    beta_to_ai_semiconductor_proxy,
    0.75 * beta_to_nasdaq_growth_proxy,
    1.25 * direct_weight_in_ai_beta_group
)
```

Tail risk:

```text
VaR_95 = abs(min(0, 5th_percentile_return))
ES_95 = abs(min(0, mean(returns <= 5th_percentile_return)))
```

Both are reported as positive loss magnitudes.

Stress tests:

```text
stress_return = sum_i weight_i * configured_group_shock_i
stress_loss = abs(min(0, stress_return))
scenario_weighted_stress_loss = sum(stress_loss_j * scenario_probability_weight_j)
```

Correlation regime:

```text
short_average_correlation = mean(pairwise correlations over short lookback)
long_average_correlation = mean(pairwise correlations over long lookback)
correlation_shift = short_average_correlation - long_average_correlation
```

Marginal risk contribution:

```text
portfolio_variance = w.T * covariance_matrix * w
marginal = covariance_matrix * w
risk_contribution_i = w_i * marginal_i / portfolio_variance
annualized_vol_contribution_i = risk_contribution_i * sqrt(portfolio_variance) * sqrt(252)
```

## Factor Attribution And Shortfall

Source: `src/trade_bot/research/factor_attribution.py`

This layer explains selected strategies and monitoring behavior. It is not a
trade generator and does not replace the portfolio risk engine. It uses ETF proxy
factors because this local project does not have institutional factor-model
holdings, Barra-style exposures, or Bloomberg-grade company fundamentals.

Proxy factor model:

```text
strategy_return_t = alpha + sum_j beta_j * factor_return_j,t + residual_t
```

The coefficients are estimated with ordinary least squares over the overlapping
strategy/factor daily return window.

Return contribution:

```text
factor_return_contribution_j = sum_t beta_j * factor_return_j,t
residual_strategy_contribution = sum_t (alpha + residual_t)
absolute_contribution_share_j =
    abs(factor_return_contribution_j) /
    sum(abs(all factor contributions) + abs(residual contribution))
```

Risk contribution:

```text
factor_component_j,t = beta_j * factor_return_j,t
risk_contribution_j = cov(factor_component_j, strategy_return) / var(strategy_return)
residual_risk_contribution =
    cov(alpha + residual, strategy_return) / var(strategy_return)
```

Model fit and residual behavior:

```text
factor_model_r_squared = 1 - var(strategy_return - predicted_return) / var(strategy_return)
residual_annualized_volatility = std(alpha + residual) * sqrt(252)
residual_variance_share = var(alpha + residual) / var(strategy_return)
```

Factor decay monitoring compares the full-history attribution to a recent
lookback:

```text
beta_drift_j = recent_beta_j - full_beta_j
drift_flag_j = abs(beta_drift_j) >= configured_beta_drift_threshold
r_squared_drop = full_r_squared - recent_r_squared
residual_volatility_ratio = recent_residual_volatility / full_residual_volatility
model_decay_flag =
    r_squared_drop >= configured_r2_drop_threshold
    or residual_volatility_ratio >= configured_residual_vol_ratio_threshold
```

Implementation shortfall has two V1 forms:

1. Ticket/execution audit: join recommendation tickets to logged executions and
   flag unexecuted tickets, price-band breaks, and size-band breaks.
2. Equity shortfall when actual account valuation is available:

```text
ideal_equity_rebased_t = ideal_equity_t / ideal_equity_start * actual_equity_start
shortfall_dollars = actual_final_equity - ideal_final_equity_rebased
shortfall_return = actual_cumulative_return - ideal_cumulative_return
tracking_error = std(actual_return_t - ideal_return_t) * sqrt(252)
```

Current caveat: the shortfall tab audits ticket/execution discipline, but it
does not ingest broker-grade daily account valuation.

Constraint application:

1. Cap non-defensive single-asset weights and move freed weight to the defensive ticker.
2. Raise defensive weight to the scenario minimum by scaling risk assets.
3. For up to four iterations, compute beta, AI beta, ES95, max stress loss, and scenario-weighted stress loss. Apply the most binding scaler to risk assets and move freed weight to the defensive ticker.
4. Reapply single-asset cap and normalize.

## Trade Decision And Sizing

Source: `src/trade_bot/research/trade_decision.py`

Base strategy weights come from the selected primary strategy. The decision engine then computes scenario, event, macro, and portfolio-risk context.

Risk-status multiplier:

```text
green: 1.00
yellow: 0.90
orange: 0.65
red: 0.40
fallback: 0.85
```

Scenario context multiplier:

```text
scenario_multiplier = 1 - 0.55 * risk_off_probability
                        - 0.20 * transition_probability
                        - 0.15 * fragile_upside_probability
scenario_multiplier = clip(scenario_multiplier, 0.40, 1.00)
```

Event pressure:

```text
event_pressure = min(
    0.25,
    0.07 * leading_escalation_events
  + 0.04 * other_escalation_events
  + 0.02 * uncertain_events
)
event_multiplier = clip(1 - event_pressure, 0.75, 1.00)
```

Macro pressure:

```text
macro_pressure = min(0.15, 0.05 * count(active paper-candidate macro pressure groups))
macro_multiplier = clip(1 - macro_pressure, 0.85, 1.00)
```

Pre-portfolio risk multiplier:

```text
scenario_event_macro_multiplier = min(
    risk_status_multiplier,
    scenario_multiplier,
    event_multiplier,
    macro_multiplier
)
```

Risk assets are scaled by this multiplier and freed weight is moved to the defensive ticker.

Portfolio-risk multiplier:

```text
portfolio_risk_multiplier = post_risk_asset_weight / pre_risk_asset_weight
```

Pre-sanity risk budget:

```text
pre_sanity_risk_budget_multiplier = clip(
    scenario_event_macro_multiplier * portfolio_risk_multiplier,
    0,
    1
)
```

Decision-sanity cap:

```text
confirmation_break_count = count(negative gates among credit, volatility, breadth, trend)
left_tail_confirmed = risk_status in {orange, red} or risk_off_probability >= 35 percent
cap_eligible = event_pressure > 0
               and confirmation_break_count < 2
               and not left_tail_confirmed
               and macro_pressure < 10 percent

max_defensive_weight = current_defensive_weight + 25 percentage points
if cap_eligible and pre_sanity_defensive_weight > max_defensive_weight:
    final_defensive_weight = max_defensive_weight
    freed weight is redistributed to non-defensive holdings pro rata
else:
    final_weights = pre_sanity_weights
```

Displayed final risk budget is the actual final risk-asset ratio after portfolio-risk and decision-sanity sizing:

```text
risk_budget_multiplier = final_non_defensive_weight / current_non_defensive_weight
risk_budget_multiplier = clip(risk_budget_multiplier, 0, 1)
```

The historical experiment analogue is in `src/trade_bot/research/experiments.py`. It cannot use current curated news labels across history, so it tests the same governing idea with price-observable gates: credit, volatility/liquidity, breadth, and trend. Paired raw-versus-capped experiments write `decision_sanity_impact.csv` so adoption can be evaluated from backtests rather than dashboard preference.

Posture calibration is an anti-over-bearish governance check. It reports context to the dashboard and evidence table without overriding sizing.

```text
constructive_probability = risk_on_probability + 0.50 * fragile_upside_probability
constructive_probability = clip(constructive_probability, 0, 1)

opportunity_pressure = risk_on_probability
                     + fragile_upside_probability
                     + 0.50 * transition_probability
                     - risk_off_probability
                     - event_pressure
                     - macro_pressure
opportunity_pressure = clip(opportunity_pressure, 0, 1)
```

Posture status rules:

```text
defense_justified if risk_status is orange/red or risk_off_probability >= 35 percent
event_defense_review if event_pressure >= 12 percent
under_risk_review if risk_budget <= 75 percent
    and opportunity_pressure >= 45 percent
    and constructive_probability >= risk_off_probability + 15 percent
opportunity_cost_watch if risk_reduction >= 10 percent
    and opportunity_pressure >= 35 percent
    and risk_status is green/yellow
upside_participation_ok if constructive_probability >= 45 percent
    and target_risk_asset_weight >= current_risk_asset_weight - 5 percent
balanced otherwise
```

Position-plan action:

```text
delta_weight = scenario_adjusted_weight - current_weight
ADD if delta_weight >= min_trade_weight
REDUCE if delta_weight <= -min_trade_weight
HOLD otherwise
```

Recommendation action:

```text
REDUCE_RISK if risk_status is orange/red and any delta <= -5 percent
REVIEW_REDUCE_RISK if any delta <= -5 percent
REVIEW_ADD_RISK if any delta >= 5 percent
HOLD otherwise
```

## Action Headline

Source: `src/trade_bot/research/action_headline.py`

Severity points come from risk state, trade decision, scenario risk, event pressure, macro pressure, high-urgency news, active news, strategy alerts, and open tickets.

Critical action if any of:

```text
risk_status == red
recommended_action == REDUCE_RISK
max_abs_delta >= 20 percent
risk_off_probability >= 35 percent
severity >= 10
```

Small action if any of:

```text
risk_status in yellow/orange
recommended_action != HOLD
max_abs_delta >= 2 percent
severity >= 3
```

Otherwise the dashboard is a do-nothing day.

## Experiment Scoring

Source: `src/trade_bot/research/experiments.py`

Experiment scoring is an exploration ranking system, not a capital allocation model.

Rolling-window metrics evaluate fixed strategy returns across 1-, 3-, and 5-year windows.

Regime metrics evaluate fixed strategy returns in named historical regimes.

Walk-forward holdout metrics use a train-span/test-span calendar split to evaluate fixed strategy returns in sequential test windows. They do not retrain or re-optimize parameters inside each fold.

Robustness score:

```text
0.20 * rank(positive_1y_window_rate)
+0.15 * rank(worst_3y_cagr)
+0.20 * rank(walk_forward_positive_rate)
+0.15 * rank(walk_forward_worst_cagr)
+0.12 * rank(worst_regime_return)
+0.10 * rank(left_tail_regime_return)
+0.08 * rank(transition_regime_hit_rate)
```

Promotion score:

```text
0.18 * rank(calmar)
+0.14 * rank(sharpe)
+0.10 * rank(cagr)
+0.12 * rank(max_drawdown)
+0.10 * rank(worst_3y_cagr)
+0.08 * rank(positive_1y_window_rate)
+0.10 * rank(walk_forward_positive_rate)
+0.08 * rank(walk_forward_worst_cagr)
+0.06 * rank(worst_regime_return)
+0.04 * rank(left_tail_regime_return)
```

Growth-constrained outcome utility:

```text
terminal_wealth_15y = starting_account_value * (1 + CAGR) ^ 15

terminal_wealth_with_contributions_15y
    = starting_account_value * (1 + CAGR) ^ 15
    + periodic_contribution * (((1 + periodic_return) ^ total_periods - 1)
                               / periodic_return)

contribution_periods_per_year = 12 for monthly default, 4 for quarterly,
                                and 1 for end-of-year
periodic_contribution = annual_contribution / contribution_periods_per_year
periodic_return = (1 + CAGR) ^ (1 / contribution_periods_per_year) - 1
total_periods = years * contribution_periods_per_year

If periodic_return is effectively zero, the contribution term uses
periodic_contribution * total_periods.

wealth_multiple_vs_spy = strategy_terminal_wealth_with_contributions
                         / spy_terminal_wealth_with_contributions
wealth_multiple_vs_qqq = strategy_terminal_wealth_with_contributions
                         / qqq_terminal_wealth_with_contributions

drawdown_recovery_return = 1 / (1 - absolute_drawdown_depth) - 1

soft_drawdown_penalty = clip((absolute_drawdown_depth - 22 percent)
                             / (30 percent - 22 percent), 0, 1)
hard_drawdown_penalty = 1 if absolute_drawdown_depth >= 30 percent else 0

wealth_score = log-scaled terminal wealth between the configured floor CAGR
               and target CAGR baselines.
validation_score starts at 1.0 and subtracts penalties for:
    weak walk-forward positive rate,
    worst 3Y CAGR below the configured floor,
    left-tail regime return below the configured floor,
    high overfit risk,
    excessive churn.

growth_constrained_utility_score
    = 0.78 * wealth_score
    + 0.22 * validation_score
    - 0.18 * soft_drawdown_penalty
    - 0.35 * hard_drawdown_penalty
    - churn_penalty_weight * churn_penalty
```

Default planning assumptions are a 15-year horizon, 220,000 dollar starting
account, 65,000 dollars of annual contributions split into monthly period-end
deposits, a soft drawdown band beginning at -22 percent, and hard drawdown
rejection at -30 percent. This score is an experiment-selection and
paper-monitoring priority layer. It does not replace promotion score,
robustness score, walk-forward diagnostics, regime diagnostics, or human review.

Sequence-aware outcome simulation:

```text
For each selected strategy:
1. compute daily strategy returns from the reconstructed equity curve,
2. sample historical daily returns in fixed-size blocks,
3. compound each sampled path for the configured planning horizon,
4. add scheduled contributions according to the configured cadence,
5. compute terminal wealth, max drawdown, and Ulcer Index for each path,
6. report distribution summaries such as P10, median, and P90 terminal wealth.
```

The block bootstrap is a stronger planning diagnostic than deterministic CAGR
because it exposes sequence risk: two strategies with similar CAGR can produce
different lived paths if one has deeper or more persistent drawdowns. It
resamples historical strategy returns and can miss future regimes that are not
represented in the historical path.

Regime-conditioned forward simulation:

```text
For each selected strategy:
1. compute daily returns from the reconstructed equity curve,
2. label historical days into risk_off, transition, risk_on_fragile, or risk_on,
3. aggregate today's scenario rollup into the same broad regime buckets,
4. blend historical regime frequencies with today's scenario probabilities
   for the starting state,
5. blend empirical regime transition frequencies with today's scenario
   probabilities for forward transitions,
6. sample historical return blocks from the active simulated regime,
7. add scheduled contributions according to the configured cadence,
8. compute terminal wealth, max drawdown, Ulcer Index, hard-drawdown breach
   probability, capital-shortfall probability, and average regime mix.
```

The intended modeling ladder is:

```text
deterministic CAGR projection
-> historical block-bootstrap sequence-risk projection
-> regime-conditioned forward simulation using scenario probabilities,
   regime transition assumptions, and strategy allocation rules.
```

The regime-conditioned layer is implemented as a planning and research lens. It
should stay out of direct trade automation until calibration, walk-forward
behavior, and paper-forward monitoring show that it improves selection, sizing,
re-entry, or drawdown control.

Promotion decisions:

```text
reject_left_tail if max_drawdown <= -30 percent
reject_regime_fragility if left_tail_regime_return < -20 percent
reject_regime_fragility if worst_regime_return < -25 percent
reject_regime_fragility if worst_3y_cagr < -5 percent
reject_walk_forward_fragility if walk_forward_positive_rate < 45 percent
promote_candidate if promotion_score >= 0.75 and calmar >= 0.45 and robustness_score >= 0.55
evolve_next_iteration if promotion_score >= 0.55
reject_or_hold_for_reference otherwise
```

Scenario position sizing inside experiments:

```text
risk_off_pressure = 0.30 * adverse_market
                  + 0.25 * adverse_credit
                  + 0.20 * liquidity_pressure
                  + 0.15 * drawdown_pressure
                  + 0.10 * oil_inflation_pressure

transition_pressure = 0.35 * adverse_breadth
                    + 0.25 * adverse_credit
                    + 0.20 * liquidity_pressure
                    + 0.20 * oil_inflation_pressure

fragile_upside_pressure = ai_concentration * adverse_breadth * (1 - 0.50 * risk_off_pressure)

risk_multiplier = risk_on_multiplier
                - risk_off_pressure * (risk_on_multiplier - stress_multiplier)
                - transition_pressure * (risk_on_multiplier - transition_multiplier)
                - fragile_upside_pressure * (risk_on_multiplier - fragile_upside_multiplier)
risk_multiplier = clip(risk_multiplier, min_multiplier, max_multiplier)
```

## Trade Tickets

Source: `src/trade_bot/trading/journal.py`

Trade tickets convert target-weight deltas into auditable paper/live suggestions.

```text
target_notional = delta_weight * account_value
side = BUY if target_notional > 0 else SELL
min_notional = abs(target_notional) * (1 - size_band_pct)
max_notional = abs(target_notional) * (1 + size_band_pct)
limit_low = reference_price * (1 - price_band_pct)
limit_high = reference_price * (1 + price_band_pct)
shares = notional / reference_price
```

Whole-share mode floors share counts to integers.

## Research Pruning and Current Model Read

Source: `src/trade_bot/research/curation.py`, `src/trade_bot/research/future_state_ml.py`, and `src/trade_bot/research/experiments.py`.

The experiment system separates historical evidence from current operating candidates. `research_status` does not delete results; it classifies them for default dashboard curation. Low-return ML probes, failed left-tail/regime tests, and reactive drawdown-control hybrids are marked as `pruned_dead_end` so they remain auditable without crowding paper-monitoring decisions.

The ML conclusion is empirical, not theoretical: bounded ML overlays preserve the high-CAGR AI escape engine better than unconstrained future-state allocation, while the tested strategy-specific drawdown models do not materially reduce max drawdown. Reactive rolling drawdown controls are especially suspect for this engine because they often cut after damage is already visible and can impair reentry.

Future model work should optimize for high-CAGR drawdown mitigation, reentry, and live drift confidence. A low-CAGR defensive model can be retained as a reference sleeve, but it should not be treated as a successful answer to the primary growth problem.

## Taxable Account Status

Current base formulas remain pre-tax unless a field is explicitly named `after_tax`, `tax_`, `realized_`, `wash_sale`, or `loss_carryforward`. Transaction costs are modeled through turnover in the base engine. The taxable layer estimates tax drag by reconstructing implied executions, deriving tax lots, classifying realized short-term/long-term gains and losses, applying wash-sale disallowance, carrying losses by calendar year, and recomputing after-tax metrics and after-tax growth utility.

## Known Limitations

- No risk-free rate in Sharpe or Sortino.
- No calibrated scenario-probability model yet.
- No true walk-forward parameter re-optimization yet.
- No slippage model beyond turnover cost.
- No market-impact model.
- Taxable-account outputs are estimated and not broker-grade. They do not model dividends, broker-lot imports, exact estimated-tax timing, or full wash-sale replacement-basis chains. Pre-tax / IRA-like fields remain the default unless explicitly labeled after-tax.
- FRED macro histories are not revision-safe.
- Yahoo Finance data is acceptable for early research but not institutional-grade.
- News classification is keyword/rule based and will miss stories outside configured channels.
- Stress tests are configured shocks, not exhaustive historical simulation.
- Human execution remains required; tickets are not broker orders.

## Drift Control Rules

Any future change to a formula in this document should include:

1. Code change.
2. Documentation update in this file.
3. A test update or new test in `tests/test_math_contracts.py`, or the closest domain-specific test file.
4. A short note in the final change summary explaining whether historical scorecards are still comparable.

The highest-priority locked tests are:

- `tests/test_math_contracts.py`
- `tests/test_backtest_engine.py`
- `tests/test_reporting.py`
- `tests/test_portfolio_risk.py`
- `tests/test_trade_decision.py`
- `tests/test_experiments.py`
- `tests/test_signal_inclusion.py`
- `tests/test_current_state.py`
- `tests/test_event_risk.py`
- `tests/test_news_monitor.py`
