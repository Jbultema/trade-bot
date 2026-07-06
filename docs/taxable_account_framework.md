# Taxable Account Framework

Status: V1/V2 estimated taxable-account support implemented for research and paper monitoring. Last reviewed: 2026-06-22.

This document defines how taxable brokerage support is modeled without
confusing pre-tax research results with after-tax operating evidence. The base
backtest remains IRA-like/pre-tax. Estimated taxable outputs are only the fields
explicitly labeled `after_tax`, `tax_`, `realized_`, `wash_sale`, or
`loss_carryforward`.

This is not tax advice. It is an engineering design for research and paper
monitoring. Before using taxable-account outputs for real trades, review the
model assumptions with a qualified tax professional.

## Why Taxable Accounts Need A Separate Model

A taxable account is path-dependent at the tax-lot level. A trade is not only a
weight change; it can realize short-term gains, long-term gains, losses, wash-sale
adjustments, and loss carryforwards. Two strategies with the same pre-tax CAGR can
produce different after-tax wealth if one churns winners weekly and the other lets
positions cross long-term holding periods.

The backtest engine models next-session execution, long-only weights,
turnover, and explicit transaction costs. The tax layer runs in parallel: it
reconstructs implied executions from strategy weights, creates derived tax lots,
classifies realized gains/losses, estimates wash-sale disallowance, applies
calendar-year tax cash flows, and emits after-tax scorecard fields. It still does
not model dividend taxation, broker-specific lot reconciliation, estimated-tax
payment timing, or full replacement-lot basis adjustments.

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


## Current Dashboard Surfaces

The dashboard exposes taxable support in two main places:

- **Research Lab -> Experiment Monitor -> Taxable Impact** is the portfolio-level
  taxable research lens. It shows configured account assumptions, pre-tax versus
  estimated after-tax CAGR, tax drag, after-tax growth utility, top after-tax
  candidates, and a tax-drag watchlist.
- **Research Lab -> Experiment Monitor -> Candidate Details workbench** shows an
  estimated taxable-account readout for the selected strategy before the full
  scorecard. Use this when inspecting a single strategy's mechanics, allocation
  behavior, and robustness.
- **Forward Test -> Estimated taxable lots** rebuilds and displays derived open
  and realized tax-lot tables for the selected mode/account from the local
  execution journal.

Taxable fields are deliberately explicit: `after_tax_*`, `tax_*`, `realized_*`,
`wash_sale_*`, and `loss_carryforward_*`. If those fields are missing, the
scorecard was probably generated before the taxable layer existed or the
warehouse has not been migrated after rerunning experiments.

The dashboard does not render tax-loss harvesting candidates from live price
updates. The helper exists in `trade_bot.tax.harvesting`; operational dashboard
TLH cards should be added only after broker-lot import/reconciliation is defined
or the user explicitly accepts paper-only estimates.

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

## Implementation Status

Implemented V1: account profile and simplified taxable simulator.

- `TaxAccountProfile` and `TaxAccountConfig` define taxable/IRA/Roth mode,
  short-term and long-term federal/state/NIIT rates, loss carryforward inputs,
  lot-selection method, and wash-sale behavior.
- `TaxLotLedger` creates derived lots from backtest or journal executions.
- FIFO, highest-cost/specific-ID tax-min, and lowest-gain lot selection are
  supported.
- Realized lots are classified short-term versus long-term using the configured
  holding-period threshold.
- Estimated after-tax CAGR, max drawdown, Calmar, tax drag, realized gain/loss,
  wash-sale, loss-carryforward, and after-tax growth-utility fields are added to
  experiment scorecards.

Implemented V2: wash-sale and tax-loss harvesting research support.

- Wash-sale detection scans same/substitute tickers inside the configured window
  and marks disallowed loss estimates.
- Substitute maps are supported by the ledger and loss-harvesting helper.
- Calendar-year loss carryforward accounting is included in the estimated tax
  cash-flow model.
- `find_tax_loss_harvest_candidates` surfaces lots with sufficiently large
  unrealized losses and labels them `human_review_required`.
- `TradeJournal.rebuild_tax_lots()` rebuilds derived open and realized tax-lot
  tables from paper/live execution history.

Still future work / not broker-grade yet:

- Import or manually seed broker-reported opening lots.
- Model dividends, qualified dividends, and distributions.
- Model exact estimated-tax payment timing.
- Carry disallowed wash-sale basis onto specific replacement lots.
- Reconcile cross-account wash-sale exposure against IRA/Roth activity.
- Reconcile all derived lots against broker statements before tax-sensitive live
  use.

## Research Rule

A strategy can be promoted for an IRA and rejected for taxable, or vice versa.
Curation should eventually expose separate rankings:

```text
pre_tax_growth_constrained_utility_score
after_tax_growth_constrained_utility_score
ira_monitoring_readiness
taxable_monitoring_readiness
```

Taxable-account conclusions are measured estimates, not design-only
hypotheses. Treat them as research support until broker lots,
actual account settings, and qualified tax review are reconciled.
