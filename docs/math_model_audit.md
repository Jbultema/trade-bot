# Math, Formula, And Model Audit

This document locks the current math and model semantics used by the app. It is meant to prevent code drift: if a future change alters one of these formulas or interpretations, update this document and the formula-contract tests in the same change.

Audit date: 2026-06-17

## Audit Result

Status: usable for research and paper trading, with explicit caveats.

The core backtest, performance, strategy, risk, scenario, and trade-decision math is internally consistent after this audit. One dashboard-window boundary issue was fixed: selected-window best/worst-day stats now come from the rebased equity window and no longer include the prior close-to-close return on the first displayed date.

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

Dashboard selected-window performance now uses the selected equity window itself:

```text
growth_of_1[t] = equity[t] / equity[first_window_date]
window_return = equity[last_window_date] / equity[first_window_date] - 1
window_daily_return[t] = pct_change(equity within selected window)
window_daily_return[first_window_date] = 0
```

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

VAMS score:

```text
momentum = lookback_return(price, lookback_days=126, skip_days=5)
volatility = realized_volatility(daily_returns, lookback=63)
vams_score = momentum / volatility
```

VAMS state:

```text
bullish if vams_score >= 0.60
bearish if vams_score <= -0.40
neutral otherwise
```

Confirmation matrix:

- Absolute signals map `bullish` to 1, `bearish` to -1, and neutral/insufficient to 0.
- Inverse pressure signals, such as `VIXY` and `UUP`, multiply that score by -1.
- Relative signals run VAMS on ratios such as `HYG / LQD`, `RSP / SPY`, and `SMH / SPY`.

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

Final risk budget:

```text
risk_budget_multiplier = clip(
    scenario_event_macro_multiplier * portfolio_risk_multiplier,
    0,
    1
)
```

Posture calibration is an anti-over-bearish governance check. It reports context to the dashboard and evidence table, but it does not currently override sizing.

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

Promotion decisions:

```text
reject_left_tail if max_drawdown <= -35 percent
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

## Known Limitations

- No risk-free rate in Sharpe or Sortino.
- No calibrated scenario-probability model yet.
- No true walk-forward parameter re-optimization yet.
- No slippage model beyond turnover cost.
- No market-impact model.
- No tax model.
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
