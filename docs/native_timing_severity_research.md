# Native Timing Joint-Severity Research

Status: pre-registered third-wave exploratory research. Allocation authority is
zero. This wave was motivated by inspected second-wave results and therefore
cannot provide confirmatory evidence on the same history.

## Motivation

The conditional logistic progression model did not beat the warning-episode
base rate. Simple point-in-time severity dimensions were more monotonic:
independent confirmation count, trailing SPY drawdown, and realized
volatility. This wave tests whether requiring those dimensions jointly can
restore useful protection above the frozen mild curve without paying the large
return cost of broad confirmation restoration.

## Point-in-time state

- confirmation: at least two independent break groups;
- extreme confirmation: at least three groups;
- drawdown: SPY is at least 5% below its trailing one-year high;
- deep drawdown: SPY is at least 10% below its trailing one-year high;
- high volatility: 21-session realized volatility exceeds the 75th percentile
  estimated only from data available before each outer fold.

## Fixed architecture roster

1. `mild_frozen`: unchanged benchmark.
2. `joint_severe_restore_10`: restore 10 defense points only when confirmation,
   drawdown, and high volatility are all present.
3. `joint_severe_restore_15`: the same with 15 points.
4. `drawdown_vol_restore_10`: restore 10 points during a deep drawdown or when
   both a 5% drawdown and high volatility are present.
5. `graduated_joint_restore`: restore 5 points when two of confirmation,
   drawdown, and high volatility are present, and 10 points when all three are.
6. `extreme_native_veto`: never remain below native defense when at least three
   confirmation groups and a deep drawdown are present.

No thresholds or candidates may be changed after inspecting this wave.

## Gate

A candidate must retain 75% of the mild curve's focus CAGR gain, improve focus
drawdown versus mild, remain within one drawdown point of native, retain
positive focus CAGR in at least three folds, improve drawdown versus mild in at
least three folds, improve median i111 utility versus mild, respect the named
crisis drawdown budget, remain positive at 20 basis-point costs, and have a
material allocation effect.

Even a full pass permits only a separately frozen prospective shadow. It cannot
replace the existing shadow or alter live sizing on retrospective evidence.

## Empirical result: 2026-07-24

`joint_severe_restore_10` was the strongest challenger:

- focus CAGR improved 2.47 percentage points versus native and was 0.85 points
  below the full mild curve;
- focus maximum drawdown improved 1.50 points versus native and 0.70 points
  versus the mild curve;
- eight of nine retrospective gates passed;
- the 75%-of-mild return-retention gate missed by approximately 0.02
  percentage points;
- i111-family median utility improved versus mild;
- the rule was active in 70 weekly observations, or 11.9% of the OOS sample;
- at 20 basis-point costs the focus CAGR improvement remained 1.80 points.

In 1,000 paired 63-session block resamples, 99.8% retained positive CAGR versus
native. Drawdown improved versus mild in 64.7% of resamples, while 27.3%
worsened native drawdown by more than one point. The protection improvement is
therefore meaningful but not stable enough to replace native or the existing
frozen mild shadow.

The fixed rule also retained positive CAGR improvement versus native with zero,
one, and two additional execution sessions: +2.53, +2.47, and +1.80 percentage
points, respectively. Activation was distributed across 18 episodes from 2015
through 2025; the largest (the 2020 shock) contributed 21.4% of active weeks.
This makes a single-crisis explanation unlikely, but it does not turn the
post-hoc history into prospective evidence.

The rule is retained only as
`i111_joint_severity_restore_v2_observation`, a zero-authority prospective
observation. It may not be retuned on the same history.

At the freeze date the severe-state restoration was inactive: the independent
break count was zero, SPY was 2.57% below its trailing high, and 21-session
realized volatility was 11.37% versus an 18.97% historical 75th-percentile
threshold. The observation therefore matched the frozen mild target: 44.98%
defensive versus 53.75% under native timing. These are a dated research
snapshot, not a standing current allocation recommendation.

## Pre-registered adversarial extension

The initial run identified `joint_severe_restore_10` as an 8/9 near-pass. It
retained 74.3% of the mild curve's focus CAGR gain versus the fixed 75% gate,
while improving focus drawdown by 0.70 points versus mild. The gate is not
changed.

Before deciding whether the near-pass is useful enough for zero-authority
prospective observation, the following diagnostics are fixed:

- zero-, one-, and two-session additional execution lags;
- 1,000 paired 63-session block resamples versus native and versus mild;
- activation dates grouped into distinct episodes;
- the share of activation contributed by the largest episode;
- focus and i111-family utility under the same fixed utility function.

No policy parameter or threshold changes are permitted. A diagnostic shadow
may be recorded only if CAGR remains positive versus native at every lag, the
bootstrap 5th-percentile CAGR delta versus native is positive, and activation
is not wholly explained by one episode. Such a record remains a near-miss
research shadow, not a promoted candidate.
