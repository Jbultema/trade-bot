# Native Timing Hazard Research

Status: pre-registered retrospective research. Allocation authority is zero.

## Question

Can Trade Bot replace binary high-defense relief rules with a point-in-time,
continuous risk-budget curve that reduces costly early defense without
materially weakening protection when warnings progress into genuine breaks?

This study is a research proxy for a native timing redesign. It does not modify
the operating risk engine, configured strategies, or current target weights.

## Unit of evidence and label

- Sampling: the configured weekly decision cadence, using information available
  by the prior close.
- Market-break label: the worst SPY or QQQ close-to-close path from an origin
  through the next 63 trading sessions reaches -10% or worse.
- Label maturity: an observation may train a model only after all 63 forward
  sessions have elapsed.
- Episode accounting: adjacent positive weeks belong to one event cluster.
  Overlapping weekly rows and multiple strategy variants are not treated as
  independent break events.
- Current/unmatured rows may receive a probability but never enter training or
  retrospective scoring.

The 10%/63-session definition is fixed before model results are inspected.
Eight- and twelve-percent labels are sensitivity diagnostics, not alternate
trials eligible to rescue a failed result.

## Point-in-time feature families

### Existing market-confirmation features

- SPY and QQQ 1-, 3-, and 6-month returns and trailing drawdowns;
- cap-weight/equal-weight breadth;
- high-yield versus investment-grade credit;
- volatility pressure and realized volatility;
- trend, credit, breadth, and volatility break count.

### Incremental features not present in the binary timing rule

- cross-sectional return dispersion and negative-return share;
- leadership spread between the strongest assets and the median asset;
- rate/duration relative behavior;
- volatility acceleration;
- family-median native defense, defense velocity, and defense dispersion;
- family-median portfolio concentration, turnover, and strategy drawdown;
- warning age and time spent without additional deterioration.

All rolling market features are shifted one session. Family features use
already-executed weights and equity available at the origin.

## Model roster

1. `market_core_global`: regularized logistic hazard using only existing
   market-confirmation features.
2. `market_augmented_global`: the global model plus cross-sectional, leadership,
   rate, and volatility-dynamics features.
3. `family_partial_pool`: a family model using augmented market and family
   state, shrunk toward the augmented global probability. The shrinkage weight
   is `n / (n + prior_strength)`.

Regularization strength and family prior strength are chosen only inside each
outer training window. Model quality is measured with Brier score, log loss,
ROC AUC when both classes exist, calibration error, and event-cluster counts.

## Native continuous-budget policy roster

The continuous hazard target is:

`clip(defense_floor + hazard_slope * probability, defense_floor, defense_ceiling)`

It is blended with the native defensive weight. Candidate variants are:

1. `constant_continuous_existing`: historical break base-rate placebo; apply the
   same continuous sizing machinery without hazard ranking.
2. `global_continuous_existing`: global hazard; scale the existing risky sleeve.
3. `family_continuous_existing`: family-pooled hazard; scale the existing sleeve.
4. `family_confirm_accel_existing`: add defense continuously as independent
   break groups accumulate.
5. `family_confirm_age_existing`: add confirmation acceleration and progressively
   release defense after four warning weeks without additional deterioration.
6. `family_confirm_age_spy_bridge`: the same curve, but use SPY when native
   defense relief has no existing risky sleeve to scale.

The constant-probability placebo was added as a falsification diagnostic after
the first hazard-driven run. It is not an independently pre-registered
promotion candidate. If it matches the hazard policy, any allocation benefit
must be attributed to generic continuous exposure smoothing rather than hazard
forecast skill.

The first run selected the lowest hazard slope, lowest floor, and highest native
blend in every outer fold. A post-result boundary-extension diagnostic therefore
adds `constant_mild_continuous_existing` and
`global_mild_continuous_existing`, using 0%/5% floors, 0.6/0.9 slopes,
65%/80% ceilings, and 75%/90%/95% native blends. This directly tests whether
the original grid was still too defensive. Because the extension was motivated
by observed results, it cannot pass the retrospective promotion gate on the
same history; it requires a new holdout or prospective shadow record.

This is not a 60% trigger. The target is defined for the full zero-to-100%
defensive range. Early low-hazard warnings remain mildly defensive; confirmed
deterioration accelerates defense; stale warnings decay gradually.

The inner policy grid is fixed to:

- defense floor: 10% or 20%;
- hazard slope: 1.2 or 1.6;
- defense ceiling: 75% or 90%;
- native-defense blend: 25%, 50%, or 75%;
- confirmation acceleration: 5 or 10 percentage points per break group;
- stale-warning decay: 1 or 2 points per week after week four, capped at 15
  points.

Grid values are parameter alternatives, not distinct architectures.

## Utility and drawdown budget

The objective is not unchanged drawdown everywhere. Inner selection maximizes
family-median net annualized return improvement after:

- penalizing additional maximum drawdown;
- penalizing upside regret when SPY is positive;
- including configured transaction costs;
- rejecting a focus-strategy drawdown degradation worse than one percentage
  point;
- rejecting any focus path beyond the configured 30% hard drawdown boundary.

Final reporting includes CAGR, maximum drawdown, Calmar, turnover, transaction
cost, 15-year deterministic wealth using the configured $220,000 starting
account and $4,000 annual contribution, up- and down-market return deltas,
up-market regret, avoided loss, and current allocation effect.

## Validation

- Nested expanding walk-forward tests begin in 2015 and use four chronological
  outer folds.
- Every inner validation label must mature before its outer test begins.
- Performance is reported only on concatenated outer-test predictions.
- Results are split by strategy, family, full-history outer tests, calendar
  era, and named crises; the full-history tests are not crisis-only samples.
- Each named crisis receives a leave-crisis-out model refit with overlapping
  training labels removed.
- Eight- and twelve-percent break labels, 10/20 basis-point costs, and policy
  parameter stability are diagnostics.
- Focus results receive zero-, one-, and two-session extra execution-lag checks
  plus a paired 63-session block bootstrap of CAGR and drawdown deltas.
- The final selected architecture, if any, is frozen into prospective shadow
  monitoring with zero allocation authority.

## Research gate

A candidate may enter prospective shadow monitoring only if:

1. Hazard Brier score improves on the expanding historical base-rate forecast.
2. Augmented or family features improve out-of-sample Brier score over the core
   global model.
3. The result is directionally stable under the 8%, 10%, and 12% break labels.
4. A majority of leave-crisis-cluster-out tests beat their base-rate forecasts.
5. Focus-strategy out-of-sample CAGR improves.
6. Focus maximum drawdown is not worse by more than one percentage point and
   remains inside the 30% hard boundary.
7. Focus 15-year deterministic wealth improves.
8. At least three of four outer folds have positive focus return delta.
9. The i111 family has positive median CAGR and utility deltas.
10. At least 75% of strategy/crisis checks stay within a 1.5-point drawdown
   damage budget.
11. The effect remains directionally positive at 10 and 20 basis-point costs.
12. The candidate changes defensive allocation by at least 2.5 points on at
    least 5% of focus-strategy outer-test sessions.

Passing permits shadow monitoring only. It does not authorize a broad
implementation or live sizing.

## Empirical result: 2026-07-23

No pre-registered architecture passed all gates, and no operating allocation
was changed.

The hazard forecast itself failed:

- the augmented global and family-pooled models had mean -10% label Brier
  scores of 0.2434 and 0.2435 versus 0.2348 for the expanding base rate;
- neither model beat the base rate on average at the -8%, -10%, or -12% break
  definition;
- mean leave-crisis-cluster-out Brier improvement was -0.0496 for the augmented
  model and -0.0473 for the family model;
- confirmation acceleration, warning-age decay, and the SPY bridge all reduced
  focus-strategy CAGR.

The original simple continuous curves were positive but modest. The global
curve added 0.53 CAGR points and improved full-period maximum drawdown by 1.75
points for the focus strategy. A constant-probability placebo still added 0.24
CAGR points, proving that part of the benefit came from continuous exposure
regularization rather than hazard ranking.

The post-hoc mild boundary extension was materially stronger:

The following table is **only the 2015-2026 nested outer-test window**. It must
not be compared with the roughly 20.67% full-history 2005-2026 configured-path
CAGR. The valid same-window improvement is 3.03 percentage points, not a
roughly 10-point jump from 20% to 30%.

| 2015-2026 OOS focus result | Native base | Mild constant shadow | Delta |
| --- | ---: | ---: | ---: |
| CAGR | 29.86% | 32.89% | +3.03 pts |
| Maximum drawdown | -25.80% | -26.37% | -0.57 pts |
| 15-year deterministic wealth | — | — | +$4.81 million |
| Current defensive weight | 57.68% | 47.92% | -9.75 pts |

All four chronological folds improved CAGR by 1.84, 3.09, 0.93, and 7.14
points. Their maximum-drawdown deltas were -0.20, -2.61, -0.57, and -4.35
points. This is the trade the earlier gate obscured: the allocation gains are
real in the sample, but they are purchased with occasional materially worse
drawdown.

All six i111 variants improved CAGR under the mild constant curve, with a
3.55-point family-median gain. The same architecture had a -0.15-point median
CAGR effect in the dynamic-risk-managed family. The evidence therefore supports
family-specific calibration and rejects a universal overlay.

At 20 basis-point costs the focus CAGR delta remained +2.53 points. With one
and two additional execution sessions it remained +1.74 and +1.72 points.
Across 1,000 paired 63-session block resamples, the CAGR-delta 5th/50th/95th
percentiles were +1.48/+2.87/+4.56 points, while 27% of resamples worsened
maximum drawdown by more than one point.

The economic mechanism is transparent: on positive-SPY sessions the mild
constant curve added 10.78 annualized return points, while on non-positive-SPY
sessions it lost 7.58 points. It is a deliberate reduction of the engine's
defensive bias, not a free alpha source or a better crash predictor.

The frozen `i111_continuous_defense_calibration_v1` shadow candidate uses the
family-selected modal curve: 0% floor, 0.6 slope, 65% ceiling, 75% native blend,
and one extra execution session. It has zero allocation authority, may not be
retuned on the same history, and requires new prospective evidence before any
promotion decision.
