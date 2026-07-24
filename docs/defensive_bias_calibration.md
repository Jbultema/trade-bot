# Defensive Bias Calibration Research

Status: retrospective research only. No allocation authority.

## Question

Trade-bot often becomes defensive well before a selloff. This study asks whether that
timing error is stable enough to support a small, point-in-time correction across
ordinary markets as well as crises. It does not assume that every early defense is a
mistake or that defensive and risk-on errors are symmetric.

## Pre-registered design

- Population: every eligible month-end from the stored full-history price panel after
  756 market days, not a hand-picked crash sample.
- Strategies: dynamic risk-managed strategies with interpretable variation in their
  defensive weights. Static allocations and buy-and-hold references are excluded.
- Families: an `i111` family and a broader `dynamic_risk_managed` family. Evidence is
  partially pooled from global to family to strategy; global and family samples are
  averaged by origin so near-clone strategies cannot multiply the apparent sample.
- Actions:
  - `defense_relief` when defense is at least 60%: transfer at most 5 percentage
    points from BIL/residual defense to the strategy's existing risk sleeve.
  - `risk_restraint` when defense is at most 20%: transfer at most 5 percentage
    points from the existing risk sleeve to BIL.
- Feasibility: the correction cannot create a new risky asset. If no risk sleeve
  exists, defense relief is zero.
- Confirmation gate: defense relief is prohibited in `confirmed_break` or
  `severe_break` states. Risk restraint is not prohibited.
- Learning horizon: 21 trading days, with 63 trading days retained as a secondary
  outcome. An outcome may enter an estimate only after its maturity date.
- Utility: candidate excess return plus 0.75 times drawdown improvement; drawdown
  deterioration is penalized at 1.50 times its magnitude. Incremental trading costs
  are included.
- Pooling: global, family, and strategy means use prior strengths 24, 18, and 12.
  A hierarchical action requires at least 24 global origins, 12 family origins, and
  8 strategy observations, and all three posterior means must exceed 5 basis points.
- Candidate roster:
  1. fixed symmetric 5-point correction;
  2. confirmation-gated fixed symmetric correction;
  3. confirmation-gated hierarchical symmetric correction;
  4. confirmation-gated hierarchical defense-relief-only correction.
- Sensitivity: 2.5/5/7.5-point caps, 55/60/65% defensive triggers, and
  15/20/25% low-defense triggers.
- Evaluation: whole-history policy metrics, calendar-era slices, ordinary-market
  versus stress regimes, named crisis windows, and leave-one-crisis-window-out
  hierarchical estimates. Named windows are evaluation slices, not the training
  population.

## Promotion boundary

Retrospective evidence can only make a candidate eligible for prospective shadow
monitoring. It cannot modify the live allocation policy. A candidate must improve
focus-strategy CAGR, avoid worsening focus maximum drawdown by more than one point,
show positive CAGR deltas across at least 60% of strategies and eras, and avoid worse
drawdown in at least 80% of leave-one-crisis-out tests. Even a retrospective pass
remains non-authoritative until prospective evidence exists.

Hard portfolio-risk constraints are excluded from the correction. They encode loss
tolerance and portfolio composition rather than a forecast that can be "bias
corrected."
