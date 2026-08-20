# Native Timing Progression Research

Status: pre-registered retrospective second-wave research. Allocation authority
is zero.

## Question

The frozen mild continuous-defense curve materially improved i111-family CAGR,
but it did so by carrying more risk rather than by forecasting breaks. Can a
conditional warning-progression layer add defense back only when an existing
warning is deteriorating, improving the mild curve's drawdown behavior without
surrendering most of its return gain?

This study does not retune the frozen mild curve. It treats that curve as the
required benchmark and tests only protection-restoration mechanisms above it.

## Frozen base curve

- defense floor: 0%;
- probability slope: 0.60;
- defense ceiling: 65%;
- native-defense blend: 75%;
- destination: the existing risky sleeve;
- one additional execution session, matching the frozen shadow;
- allocation authority: 0%.

## Conditional warning population

A family-week is eligible when at least one condition is true using information
available by the prior close:

- family median defense is at least 25%;
- family median defense rose at least 10 points over 21 sessions;
- at least one independent market break group is active.

Adjacent eligible weeks form one warning episode. Episode weights sum to one so
long warnings do not multiply their apparent evidence. The progression label
remains the previously fixed event: the worst SPY or QQQ path over the next 63
sessions reaches -10% or worse. Labels must mature before training.

The conditional model is a fixed L2 logistic regression using:

- independent break count;
- SPY and QQQ trailing drawdowns;
- breadth and credit 63-session changes;
- realized volatility and its acceleration;
- family defense level and 21-session change;
- family drawdown;
- warning age and non-deterioration age.

The regularization value is fixed at 0.20. It is not selected on second-wave
results. Probability improvement is measured against the expanding
warning-episode base rate, not the unconditional market-break base rate.

## Fixed architecture roster

1. `mild_frozen`: unchanged frozen mild curve.
2. `progression_restore_10`: restore up to 10 defense points in proportion to
   conditional probability above its warning base rate.
3. `progression_restore_20`: the same with a 20-point cap.
4. `confirmation_restore_10`: restore up to 10 points only as independent break
   groups accumulate; no learned probability.
5. `hybrid_restore_15`: use the larger of a 15-point probability restoration
   and a 15-point confirmation restoration.
6. `native_severe_veto`: use the mild curve normally, but never remain below
   native defense when two or more independent break groups are active.

These are architecture tests, not a threshold sweep. No additional candidate
may be added after inspecting their results in this wave.

## Evaluation

- i111 family only, because the first wave was positive for all six i111
  variants and negative for the unrelated dynamic-risk-managed family;
- the same four expanding outer folds from 2015 through 2026;
- full daily paths within each outer fold, not crisis-only samples;
- configured transaction costs;
- fold, era, and named-crisis return/drawdown diagnostics;
- conditional Brier score and episode count;
- current defensive-weight effect;
- comparison both with native Trade-bot and with `mild_frozen`.

## Second-wave gate

A restoration challenger may replace the frozen mild shadow only if:

1. conditional Brier score beats the warning-episode base rate;
2. focus CAGR retains at least 75% of `mild_frozen`'s gain over native;
3. focus maximum drawdown improves versus `mild_frozen`;
4. focus maximum drawdown remains within one point of native and above the
   configured -30% hard boundary;
5. at least three of four folds retain positive CAGR versus native;
6. at least three of four folds have non-worse drawdown versus `mild_frozen`;
7. i111-family median utility improves versus `mild_frozen`;
8. at least 75% of named strategy/crisis checks stay within a 1.5-point
   drawdown-damage budget versus native;
9. the result remains positive at 20 basis-point costs;
10. the allocation effect is material.

Passing this retrospective gate permits a new frozen prospective shadow only.
It cannot authorize live sizing because this architecture was motivated by
results already observed through July 2026.

## Second-wave result

The warning-progression model did not beat its conditional episode base rate:
mean Brier improvement was -0.0006. Its highest-probability quintile did not
consistently concentrate realized breaks across outer folds.

`progression_restore_20` was the closest sizing policy:

- focus CAGR delta versus native: +2.88 percentage points;
- focus CAGR delta versus the executable mild curve: -0.44 points;
- focus maximum-drawdown delta versus mild: +0.11 points;
- gates passed: 8/10;
- failed gates: conditional model skill and i111-family utility versus mild.

The trade is not attractive enough: it gives up 0.44 CAGR points to recover
only 0.11 drawdown points, and the learned probability is not independently
validated.

## Pre-registered third wave: treatment-risk model

This section was added after the second-wave results and therefore defines a
new retrospective experiment, not a rescue of the failed progression model.

The target changes from "will the market break?" to "will applying the frozen
mild curve cause at least one additional percentage point of worst 63-session
drawdown versus native Trade Bot?" This is a policy-specific treatment-risk
label.

- The full-history mild path is generated point in time using only break labels
  that matured before each weekly decision.
- The harm label is one only when mild forward maximum drawdown is at least one
  point worse than native over the next 63 sessions.
- Training remains restricted to warning-eligible episodes with equal episode
  weights.
- The feature roster and fixed `C=0.20` regularization are unchanged.
- The model must beat the expanding treatment-harm base rate out of sample.

Fixed candidate roster:

1. `treatment_restore_10`: restore up to 10 points in proportion to treatment
   harm probability above its training base rate.
2. `treatment_restore_20`: restore up to 20 points.
3. `treatment_native_veto`: revert to native defense when predicted treatment
   harm is at least 50%; this threshold is fixed before results.

The same return, drawdown, fold, family, crisis, cost, and materiality gates
apply. No further architecture or threshold may be added after this run.
