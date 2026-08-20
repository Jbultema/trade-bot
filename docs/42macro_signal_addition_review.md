# Candidate Signal Additions From the August 2026 Review

Status: research decision note. Evidence current through 2026-08-18.

## Decision

The new 42 Macro material identifies one useful transmission mechanism and two
slow-moving vulnerability diagnostics. None has earned allocation authority.
They should enter as research/context signals first and only affect sizing after
point-in-time historical validation and a frozen out-of-sample test.

## 1. Capital competition and long-duration supply

**Finding:** This is the most important new mechanism. Exceptional corporate AI
financing is arriving while sovereign duration supply is already large. The
mechanism can hurt long-duration equities before AI revenue visibly disappoints.

**What Trade Bot already captures:** 30-year and 10-year nominal yields, 5-year
and 10-year real yields, Treasury-curve changes, TLT/IEF relative behavior,
credit spreads, and long-duration growth performance. On August 18,
`duration_support` was adverse (-0.26) while credit (+0.46) and market trend
(+0.72) remained supportive. That is the correct partial-confirmation read.

**Gap:** Trade Bot does not directly measure corporate issuance volume, maturity
distribution, corporate-versus-sovereign portfolio share, auction absorption, or
30-year real yield.

**Recommendation:** Add a research-only `capital_competition` panel with 30-year
nominal/real yields, curve steepening, Treasury auction/term-premium proxies, and
public corporate issuance data when a reproducible point-in-time source is
available. It should have zero sizing authority until it demonstrates incremental
information beyond existing duration, credit, and growth-relative-price inputs.

## 2. FINRA customer margin debt

**Finding:** [FINRA's July customer margin debits](https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics)
were $1.417 trillion, down 5.6% from the June peak but up 38.6% year over year.
This is meaningful vulnerability and positioning context. Four selected
historical episodes do not justify crash probabilities such as 25%, 25%, and
50%.

**Recommendation:** Track level, year-over-year growth, month-over-month change,
drawdown from the trailing peak, and margin debt relative to broad equity market
capitalization. Label it `vulnerability_only`; do not treat it as a crash timer.
FINRA publishes monthly with a lag and warns that reporting-method changes can
affect month-to-month variation.

## 3. Earnings-expectation stretch

**Finding:** The useful idea is expectation asymmetry: good realized earnings can
still disappoint if the priced forward path is unusually demanding.

**Gap:** Trade Bot explicitly lacks point-in-time forward EPS, revision, margin,
and valuation history. A current consensus number copied from commentary is not
safe for historical testing because estimates are revised and vendor definitions
differ.

**Recommendation:** Add only after obtaining a vintage-preserved estimates feed.
The initial diagnostic should compare three-year forward consensus EPS CAGR with
trailing and long-run EPS growth, plus breadth and direction of revisions. Keep it
informational until walk-forward tests establish incremental value.

## Rejected or already covered

- Concentration and AI leadership are already first-class Trade Bot inputs.
- Household equity exposure is useful structural context but too slow and
  definition-sensitive for tactical sizing.
- Bitcoin treasury leverage belongs in a separate crypto balance-sheet risk
  study; cold-wallet hacks and quantum risk are not established market-timing
  signals.
- The `Paradigm E` total-war extrapolation is a narrative hypothesis, not a
  measurable addition to the risk engine.

## Promotion gate

Any new diagnostic must show incremental information after controlling for the
existing duration, trend, breadth, volatility, credit, and AI-leadership inputs.
Evaluation must use publication-lagged vintages, nested walk-forward selection,
an untouched holdout, and explicit utility/drawdown trade-offs. Until then the
diagnostics may explain risk but may not size the portfolio.
