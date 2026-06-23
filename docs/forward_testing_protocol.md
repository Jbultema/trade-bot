# Forward Testing Protocol

This project should not move from research to real money until the strategy, logging, and review
workflow are locked enough that results can be audited.

## Stages

1. Research only
   - Run experiments and backtests.
   - Reject candidates with obvious leakage, left-tail fragility, or benchmark-relative weakness.
   - No trade tickets are treated as actionable.

2. Locked shadow testing
   - Pick one to three candidate operating systems.
   - Generate recommendation tickets from the dashboard.
   - Lock each recommendation set before the decision window passes.
   - Log paper executions with timestamp, ticker, side, quantity, price, fees, and notes.
   - Review paper results versus the locked recommendation, not versus revised later signals.

3. Very small live test
   - Use the same ticket and execution workflow, but set mode to `live`.
   - Keep trade sizes intentionally small.
   - Require manual execution and manual price entry.
   - Compare live fills against the suggested price and size ranges.

4. Scale review
   - Scale only after enough forward observations show that recommendations are operationally
     usable, not just backtest-good.
   - Review missed trades, skipped trades, stale signals, slippage, overtrading, and regime behavior.

## What Gets Locked

Each locked recommendation snapshot stores:

- timestamp in UTC
- paper/live mode
- account label
- account value used for sizing
- recommended action
- risk status and risk-budget multiplier
- base position and scenario-adjusted position
- human explanation
- full position bridge
- evidence and scenario links
- generated trade tickets with ticker, side, reference price, price range, notional range, and share range

## Trade Ticket Rules

Trade tickets are not broker orders. They are auditable decision records.

- `paper` mode is the default for shadow testing.
- `live` mode is only for manually executed real trades.
- The dashboard uses latest cached market prices as reference prices.
- Price bands define the intended acceptable execution range.
- Size bands define the intended notional/share range.
- Whole-share sizing can be toggled depending on the actual brokerage/account constraint.


## Taxable Account Forward Testing

Taxable-account monitoring is a parallel evidence track. It should not replace
pre-tax/IRA-like strategy research, and it should not override true risk exits.

Before paper-monitoring a taxable brokerage version of a strategy:

- Review **Research Lab -> Experiment Monitor -> Taxable Impact**.
- Confirm the strategy still has acceptable `after_tax_cagr`,
  `after_tax_max_drawdown`, and `after_tax_growth_constrained_utility_score`.
- Check `tax_drag_bps_per_year`, `short_term_gain_share`, realized short/long
  gain mix, wash-sale disallowed-loss estimates, and loss carryforward.
- Confirm the tax assumptions in `tax_account` match the intended planning
  scenario closely enough for research use.

When logging paper or live executions for taxable accounts:

- Keep account labels explicit, for example `paper_taxable_core` rather than a
  generic account name.
- Record fees and exact execution timestamps because tax lots are rebuilt from
  execution history.
- Rebuild derived lots with `TradeJournal.rebuild_tax_lots()` after executions
  are logged if you need current open/realized lot tables.
- Treat the local tax-lot tables as estimated audit support until broker lots are
  reconciled.

Before real taxable-account trades, add one more gate: broker-reported opening
lots, wash-sale exposure, and personal tax assumptions need manual review. The
bot can show estimated drag and TLH candidates; it cannot decide that a taxable
trade is appropriate.

## Minimum Gate Before Real Money

Before real trades, the system should have:

- a locked candidate strategy or small ensemble
- at least several weeks of paper tickets and paper executions
- no unresolved leakage issues in the backtest engine
- benchmark-relative scorecards for the candidate set
- a written reason for why the strategy is expected to work now
- a written off-ramp for when the strategy stops working

