# Macro Coverage Gap And Data Roadmap

Status: maintained capability-gap map. Last reviewed: 2026-06-21.

This project is not trying to clone any commercial macro product. The useful target is functional
parity with the operating pattern: macro regime, positioning, market signal,
risk sizing, and human-readable action translation.

## Current Implemented Coverage

The local public-data system currently approximates several institutional-style
macro concepts, but it should be treated as a research proxy stack rather than a
commercial data terminal. These signals are useful only after they survive
backtests, walk-forward checks, and paper-monitoring review.

- Regime Pulse Lite:
  - growth
  - inflation
  - monetary policy
  - fiscal policy
  - liquidity
  - positioning
- Growth-Inflation Map:
  - Growth-disinflation
  - Reflation
  - Inflation
  - Deflation
- Positioning / crowding proxy:
  - 3-month return z-score
  - 14-day RSI
  - crowded upside and washed-out re-entry states
- Dashboard surface:
  - News & Macro now shows cycle, asset, Growth-Inflation Map, and positioning tables.

## Remaining Gaps

- True ETF and mutual-fund flow data.
- AAII sentiment and allocation history.
- NAAIM exposure history.
- CFTC COT futures positioning by asset class.
- Options surface, implied correlation, skew, and term structure.
- Consensus forecast comparison for GDP, inflation, EPS, and policy.
- Revision-safe macro vintages.
- Global central-bank balance sheets, broad money, FX reserves, PMIs, and
  country-level inflation/policy grids.
- Secular inflation model with structural drivers.
- Long-only asset-class state backtests.
- Direct earnings-revision and analyst-estimate revision history.
- IPO calendar and post-IPO performance monitoring for large market-structure
  events.
- Source-quality scoring for news/event inputs.

## Intended Use

The new regime-pulse layer should not automatically override strategy signals.
It should answer four operating questions:

1. Is the status quo macro backdrop risk-on, mixed, or risk-off?
2. Are positioning and crowding making the same trade fragile?
3. Is the likely growth-inflation regime changing?
4. Should the risk engine cut, hold, or re-risk exposure?
