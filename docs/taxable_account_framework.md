# Taxable Account Framework

Status: planned design, not implemented in current backtests. Last reviewed: 2026-06-22.

This document defines how taxable brokerage support should be added without
confusing pre-tax research results with after-tax operating evidence. The current
system is still best interpreted as IRA-like or pre-tax research unless a result
is explicitly labeled after-tax.

This is not tax advice. It is an engineering design for research and paper
monitoring. Before using taxable-account outputs for real trades, review the
model assumptions with a qualified tax professional.

## Why Taxable Accounts Need A Separate Model

A taxable account is path-dependent at the tax-lot level. A trade is not only a
weight change; it can realize short-term gains, long-term gains, losses, wash-sale
adjustments, and loss carryforwards. Two strategies with the same pre-tax CAGR can
produce different after-tax wealth if one churns winners weekly and the other lets
positions cross long-term holding periods.

The current backtest engine models next-session execution, long-only weights,
turnover, and explicit transaction costs. It does not yet model realized taxes,
tax lots, wash sales, estimated taxes, dividend taxation, or account-level loss
carryforwards.

## Tax Rules The Model Must Respect

Use official IRS sources as the baseline reference. At minimum, the model must
track:

- Short-term versus long-term classification. IRS Topic 409 classifies gains and
  losses by holding period; generally, assets held more than one year are
  long-term, while assets held one year or less are short-term.
- Preferential long-term capital-gain rates and ordinary-income treatment for
  net short-term capital gains. Tax brackets change over time and must be
  configurable rather than hardcoded.
- Capital loss deduction limits and carryforwards. Excess capital losses can
  offset capital gains, then only a limited amount of ordinary income each year,
  with remaining losses carried forward.
- Wash-sale rules. IRS Publication 550 describes loss disallowance when a
  substantially identical security is acquired within the relevant 30-day before
  or after window, including certain IRA/Roth IRA replacement purchases.
- Net investment income tax. IRS Topic 559 describes the 3.8 percent NIIT that
  can apply to investment income above MAGI thresholds.
- State taxes. The first model may allow a flat state capital-gain rate, but it
  should remain configurable and optional.

Reference links:

- IRS Topic 409, Capital gains and losses: https://www.irs.gov/taxtopics/tc409
- IRS Publication 550, Investment Income and Expenses: https://www.irs.gov/publications/p550
- IRS Topic 559, Net investment income tax: https://www.irs.gov/taxtopics/tc559

## Account Profile

Add an explicit account profile rather than hiding taxable behavior inside a
strategy config.

Suggested fields:

```text
account_type: ira | roth | taxable
federal_short_term_tax_rate
federal_long_term_tax_rate
state_short_term_tax_rate
state_long_term_tax_rate
niit_rate
niit_applies
capital_loss_carryforward_short
capital_loss_carryforward_long
annual_loss_deduction_limit
lot_selection_method: fifo | specific_id_tax_min | highest_cost | lowest_gain
wash_sale_enforcement: off | warn | block_loss_harvest | strict
```

Defaults belong in `src/trade_bot/DEFAULTS.py`. User-specific tax rates should
belong in local config or local secrets-free profile files, not in source code.

## Tax-Lot Ledger

Taxable support requires a real lot ledger. Portfolio weights are not enough.

Each buy lot should store:

```text
account_id
ticker
acquired_at
quantity
price
cost_basis
source_trade_id
wash_sale_adjustment
holding_period_start
```

Each sell should produce realized-lot records:

```text
account_id
ticker
sold_at
quantity
sale_price
cost_basis
realized_gain_loss
short_or_long_term
wash_sale_disallowed_loss
replacement_lot_id
strategy_id
recommendation_ticket_id
```

The paper/live journal can remain the execution source of truth, but tax lots
need their own derived tables because one execution can consume multiple lots.

## After-Tax Backtest Mechanics

The taxable backtest should run parallel to the existing pre-tax engine:

1. Generate target weights and next-session executions as today.
2. Convert executions into buys/sells by account profile.
3. Select lots for sells using the configured lot-selection method.
4. Classify realized gains/losses as short-term or long-term.
5. Detect wash-sale windows and adjust loss treatment.
6. Apply tax liabilities or deferred tax assets to an after-tax cash/equity path.
7. Carry losses forward across calendar years.

Required output metrics:

```text
pre_tax_cagr
after_tax_cagr
pre_tax_terminal_wealth
after_tax_terminal_wealth
tax_drag_bps_per_year
realized_short_term_gain
realized_long_term_gain
realized_loss_harvested
wash_sale_disallowed_loss
loss_carryforward_end
short_term_gain_share
after_tax_growth_constrained_utility_score
```

The dashboard must label these clearly. Never mix pre-tax and after-tax scorecards
without an `account_profile` or `tax_model_status` field.

## Tax-Aware Strategy Behavior

Taxable accounts should not simply reject active strategies. Instead, they should
score whether the return edge survives taxes.

Likely favorable taxable behavior:

- lower churn when signals are only marginally different
- using new cash contributions before selling appreciated lots
- avoiding small trims that realize short-term gains
- preferring specific-ID lot selection where available
- letting winners cross long-term holding periods when risk does not require exit
- tax-loss harvesting during drawdowns while maintaining similar-but-not-
  substantially-identical exposure

Tax constraints should not override true left-tail exits. A taxable model can
warn that a trade realizes gains, but it should not hold a collapsing position
only to avoid taxes.

## Tax-Loss Harvesting Design

Tax-loss harvesting is a separate overlay, not a default behavior of every
strategy. It should require:

- unrealized loss threshold
- minimum notional loss threshold
- substitute asset map
- wash-sale lookback/lookforward guard
- cross-account warning for IRA/Roth replacement exposure
- minimum days before returning to the original ticker
- tracking of harvested losses and deferred/disallowed losses

Substitute maps should be conservative. For example, swapping two ETFs that track
the same index may be too close for an automated recommendation. The app should
show the exposure-preserving intent and require human review.

## Dashboard Requirements

Add an account-mode selector to research and monitoring views:

```text
Pre-tax / IRA-like
Taxable estimated
```

Taxable mode should add:

- after-tax performance table
- tax drag by strategy
- realized gain/loss summary
- short-term versus long-term realization mix
- wash-sale warnings
- tax-loss harvesting candidates
- near-long-term holding-period warnings
- projected estimated-tax cash need

The operating brief should say whether a trade is being recommended despite tax
cost because risk evidence is strong, or delayed because the signal is marginal
and taxable drag is high.

## Implementation Plan

Phase T0: Documentation and flags.

- Keep all current scorecards labeled pre-tax.
- Add `account_type` and `tax_model_status` fields where scorecards or dashboard
  views could become account-aware later.

Phase T1: Simplified taxable simulator.

- Add account profile defaults.
- Add tax-lot ledger for backtests.
- Add FIFO and specific-ID tax-min lot selection.
- Add short/long-term gain classification.
- Add federal/state/NIIT tax-rate inputs.
- Add after-tax metrics and dashboard comparison.

Phase T2: Wash-sale and tax-loss harvesting.

- Add wash-sale detection.
- Add substitute asset maps.
- Add loss carryforward accounting.
- Add TLH candidate generation with human-review warnings.

Phase T3: Paper/live taxable monitoring.

- Import or manually maintain starting lots.
- Tie recommendation tickets to executed lots.
- Track realized gains/losses forward.
- Reconcile broker-reported lots manually before tax-sensitive live use.

## Research Rule

A strategy can be promoted for an IRA and rejected for taxable, or vice versa.
Curation should eventually expose separate rankings:

```text
pre_tax_growth_constrained_utility_score
after_tax_growth_constrained_utility_score
ira_monitoring_readiness
taxable_monitoring_readiness
```

Until the taxable simulator exists, taxable-account conclusions should be framed
as design hypotheses, not measured results.
