# Chapter 13 research pass — Cash Waterfalls and Return on Invested Capital

Status date: 2026-07-25.  This is a staging-only evidence-to-model package, not
chapter prose and not a completed model.  It applies the economic-model governance
contract to accepted canonical claims whose `review_status` is `reviewed`. The
integration acceptance controls `integration/lane_1_claim_acceptance.csv` and
`integration/lane_2_claim_acceptance.csv` were reviewed alongside the canonical claim ledger; this packet retains their source and nonadditivity boundaries.

## Decision

No project or company can support a publication-standard project IRR, accounting AI
ROIC, incremental AI ROIC, economic return, implied return, or reverse-DCF output
today.  This is an evidence result, not a conclusion that returns are inadequate.

River Bend and CCAC VIII support the most complete **priority maps**, not a return
calculation.  River Bend has debt principal/rate/maturity, a reported revenue/NOI
forecast, reserve requirement, and an indenture waterfall, but lacks a reconciled
sources-and-uses, rent cash timing, operating-cost schedule, amortization schedule,
equity at risk, and terminal value.  CCAC VIII has a documented controlled-account
waterfall, commitment/draw, rate margins, coverage tests and reserves, but its customer
cash flow, equipment cost, acceptance schedule, power-cost formula, amortization and
residual are redacted.

Hyperion, Saline, Helios I and the Broadcom/Apollo racks provide partial funding or
control mechanics.  Missing debt/equity, rent, operating cost, recovery and/or asset
title terms keep them unresolved.  CoreWeave and all five hyperscalers provide
company-level stock, flow and obligation observations, but no project-level AI revenue,
cost, useful-life, tax, working-capital or capital-allocation bridge.  Their five
attribution methods remain blocked rather than yielding a single claimed “AI ROIC.”

## Observed inputs retained

- River Bend: $3.25bn funded senior secured notes at 6.192%, maturity 2042, construction
  reserve requirement, priority waterfall, $7bn 15-year base lease, and disclosed
  management NOI forecast.  Project cost inferred from approximate loan-to-cost is a
  Grade-C reconciliation, not audited source-and-use.
- CCAC VIII: $8.5bn DDTL commitment, $1.26bn draw, floating/fixed pricing, DSCR tests,
  liquidity/power reserves and nine-step cash control.  Customer economics are redacted.
- Hyperion: disclosed opening contributions/distribution, estimated development cost,
  four-year initial lease commitment and conditional residual support.  These measures
  overlap and are not debt, equity or expected loss totals.
- Saline: $16bn announced campus is separate from DTE storage; customer-funded utility
  revenue requirements and termination recovery are formula-based but dollar cost is
  not public.
- Hyperscaler cash capex, noncash additions, CIP, leases, commitments, debt, RPO,
  prepayments and depreciation are retained as distinct clocks.  They are not AI-only
  return denominators.

## Model rule applications

1. The `cash_waterfall_inputs.csv` rows preserve priority without fabricating cash
   amount, timing, or claimant return.
2. `funding_clock_reconciliation.csv` keeps cash, noncash, recognized stock, contract
   commitment, and contingent support separate.  No row is summed into a capital total.
3. `return_denominator_inputs.csv` contains an explicit “unresolved” row for every
   missing material denominator component.  A return model inherits that state.
4. `required_revenue_models.csv` records equations and the inputs that would be needed;
   it does not emit a required-revenue point estimate when price, cost, term or terminal
   value is open-ended.
5. `first_loss_thresholds.csv` identifies disclosed triggers only.  A trigger is not a
   solved threshold or proof of a first-loss bearer.

## Falsification and update path

The package would change if a filed project budget, rent schedule, debt-service table,
asset acceptance certificate, customer order form, reserve report, tax/working-capital
schedule, useful-life record, asset-level cloud P&L, borrowing-base certificate, or
intercreditor/guarantee waterfall is released.  Each is queued with the exact model
output it would unblock.

## Handoff

CH12 receives only observed/contract-implied operating-cost and power-priority inputs;
it cannot infer a unit-economic break-even.  CH14 receives an explicit block on
productive utilization, realized price, asset life and residual.  CH17 receives
documented cash-control/default triggers, not a loss-severity model.  Chapter 13 should
publish the unresolved perimeter alongside the priority maps rather than a false return
table.
