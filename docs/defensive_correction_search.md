# Defensive Correction Architecture Search

Status: pre-registered retrospective research. Live allocation authority is zero.

## Objective and stopping rule

Trade Bot appears too defensive in many historical periods. The objective is to
find whether a low-degree-of-freedom, point-in-time mechanism can materially
reduce that opportunity cost without simply restoring the drawdowns the base
strategy was designed to avoid.

Continue until one architecture clears the full retrospective gate and merits
prospective shadow monitoring, more than 10 genuinely distinct architectures
fail, or 30 distinct architectures have been tested. Threshold sensitivities do
not count as separate architectures.

## Common contract

- Base paths: stored executed weights for all 11 dynamic risk-managed strategies.
- Inputs: only each date's already-executed base weights and market information
  available by the prior close.
- Primary focus: configured i111 strategy.
- Costs: 5 basis points per unit of turnover; survivors are retested at 10 and
  20 basis points.
- Defensive weight: explicit BIL plus unallocated residual cash.
- Material correction: at least 2.5 percentage points on active days and active
  on at least 5% of focus-strategy days.
- Maximum one-day correction: 20 percentage points.
- Hard portfolio-risk constraints: outside the search and never overridden.
- News, events, scenarios, and revised macro: excluded.
- Evaluation: 5,422 daily sessions from 2005-01-03 through 2026-07-23, 11
  strategies, four fixed calendar eras, a crisis-excluded ordinary-market path,
  quarterly-sampled one- and three-year rolling windows, eight named stress
  windows, current allocation effect, turnover, and drawdown.
- A missing or unavailable signal fails closed to no correction.

## First-wave architecture roster

These are different causal rules, not parameter variants:

1. `fixed_existing_sleeve_relief`: transfer five points from defense to the
   strategy's existing risk sleeve whenever defense is at least 60%.
2. `dual_trend_intact_relief`: relieve only when both SPY and QQQ are above their
   200-day moving averages.
3. `credit_volatility_intact_relief`: relieve only when credit is not broken and
   volatility pressure is not confirmed.
4. `breadth_intact_relief`: relieve only when the RSP/SPY relative trend is
   intact.
5. `positive_momentum_relief`: relieve only when both SPY and QQQ have positive
   one-month returns.
6. `low_volatility_relief`: relieve only when realized SPY volatility is below
   its trailing three-year 60th percentile.
7. `shallow_drawdown_relief`: relieve only while SPY's one-year drawdown is
   shallower than 5%.
8. `defense_duration_decay`: after 20 consecutive high-defense sessions without
   a confirmed break, release 10 points.
9. `rapid_ramp_damper`: reverse the portion of a five-session defensive increase
   exceeding 10 points unless a break is confirmed.
10. `recovery_cross_accelerator`: release 10 points for 20 sessions after SPY
    recovers above its 50-day average.
11. `intact_risk_floor`: while dual trend is intact, maintain at least 45% in the
    strategy's existing risk sleeve.
12. `spy_beta_bridge`: while dual trend is intact, bridge up to 10 defensive
    points into SPY rather than the existing concentrated sleeve.
13. `splv_low_beta_bridge`: under the same condition, bridge into SPLV.
14. `rsp_breadth_bridge`: when cap-weight trend is intact and breadth is not
    broken, bridge into RSP.
15. `family_disagreement_relief`: relieve 10 points when one strategy is highly
    defensive but its strategy-family median is below 50% defense.
16. `family_gap_proportional_relief`: release the strategy/family defensive gap,
    capped at 15 points, when the market is not in a confirmed break.
17. `opportunity_cost_feedback`: after a high-defense episode has allowed SPY to
    gain at least 5% without a confirmed break, release 10 points.
18. `native_reentry_accelerator`: when native defense is already falling by at
    least five points over five sessions, accelerate that re-entry by another
    five points.
19. `health_score_proportional_relief`: combine trend, credit, breadth,
    volatility, and momentum as equally weighted prior-close confirmations and
    release two points per intact confirmation above two, capped at 10.
20. `ramp_duration_composite`: combine the rapid-ramp damper with duration decay,
    taking the larger correction while preserving the confirmed-break veto.

The earlier confirmation-fixed and hierarchical policies remain comparators and
do not count as new first-wave architectures.

## Second-wave conditional combinations

The first-wave read identified `breadth_intact_relief` as the only architecture
that cleared the focus-return, focus-drawdown, cross-strategy return, era,
crisis, materiality, and higher-cost gates while missing the strict
cross-strategy non-worse-drawdown gate. Before testing second-wave results, ten
breadth-centered combinations are declared:

21. `breadth_break_veto_relief`: breadth relief only without a multi-group
    confirmed break.
22. `breadth_dual_trend_relief`: require intact breadth plus intact SPY/QQQ
    200-day trends.
23. `breadth_shallow_drawdown_relief`: require intact breadth and a sub-5% SPY
    drawdown.
24. `breadth_low_volatility_relief`: require intact breadth and low realized
    volatility.
25. `breadth_positive_momentum_relief`: require intact breadth and positive
    SPY/QQQ one-month momentum.
26. `breadth_credit_volatility_relief`: require intact breadth, credit, and
    volatility.
27. `breadth_duration_confirmed_relief`: require intact breadth, no confirmed
    break, and at least 20 consecutive high-defense sessions before releasing
    10 points.
28. `breadth_splv_bridge`: direct five points to SPLV instead of the native
    concentrated risk sleeve.
29. `breadth_split_existing_splv`: split five points equally between the native
    risk sleeve and SPLV.
30. `breadth_state_adaptive_bridge`: direct five points to the native sleeve
    with zero broken confirmation groups, to SPLV with one broken group, and
    apply no relief with two or more.

These combinations are a conditional second wave chosen after the first-wave
screen. They are not independent confirmatory evidence. Any survivor therefore
needs stricter holdouts and prospective monitoring.

## Retrospective gate

An architecture must satisfy every item:

1. Focus CAGR delta is positive.
2. Focus maximum drawdown is not worse by more than 1 percentage point.
3. At least 60% of strategies have positive CAGR delta.
4. At least 60% of strategies have non-worse maximum drawdown.
5. At least three of four eras have positive cross-strategy median annualized
   return delta.
6. At least 75% of strategy/crisis tests have non-worse drawdown.
7. The focus correction is materially active under the definition above.
8. The result remains directionally intact at 10 and 20 basis-point costs.

Passing this gate permits prospective shadow monitoring only. It does not permit
automatic changes to current allocation logic.

## Explicit ordinary-market and rolling diagnostics

The full-path and era results are the primary tests; named crises are a separate
adversarial gate. To prevent a result from looking useful only because of stress
episodes, the study also removes every session in the eight named crisis windows
and recomputes an ordinary-market path. That leaves 3,592 sessions. Rolling
consistency is measured in 83 one-year and 75 three-year windows per strategy,
sampled every 63 sessions. These are overlapping descriptive diagnostics, not
independent trials and not additional post-hoc promotion gates.

For the closest `breadth_intact_relief` rule:

- crisis-excluded focus annualized-return delta is +0.42 percentage points;
- 54.5% of strategies have positive crisis-excluded annualized-return delta,
  while all 11 have non-worse crisis-excluded maximum drawdown;
- 54.2% of focus one-year windows and 73.3% of focus three-year windows have
  higher return than base;
- 69.9% of focus one-year windows and 50.7% of focus three-year windows have
  non-worse maximum drawdown.

The opportunity-cost improvement therefore exists outside crises, but it is not
uniform across strategies or shorter rolling windows. The full-path drawdown
damage is specifically a stress-period problem, which is why the rule remains a
near-miss rather than a live correction.
