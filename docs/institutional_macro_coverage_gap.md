# Macro Coverage Gap And Data Roadmap

Status: maintained capability-gap map. Last reviewed: 2026-06-29.

This project is not trying to clone any commercial macro product. The useful target is functional
parity with the operating pattern: macro regime, positioning, market signal,
risk sizing, and human-readable action translation.

## Current Implemented Coverage

The local public-data system approximates several institutional-style
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
  - News & Macro shows cycle, asset, Growth-Inflation Map, and positioning tables.
- Cross-Source Insight Diagnostics:
  - AI supplier / hyperscaler divergence.
  - Hyperscaler capex / free-cash-flow pressure.
  - AI capex inflation pass-through.
  - Concentration versus broadening.
  - Oil / inflation shock.
  - Private-credit / liquidity stress.
  - Fed-put / policy uncertainty.
  - Speculative leverage proxy.
  - IPO / equity-supply pressure.
  - Positive catalyst absorption.
  - International chip concentration.
  - Sector valuation / policy proxy.
  - Easy-bubble versus hard risk-off.

## Data-Support Discipline

The source-informed diagnostic layer exists because investor commentary can point
to useful questions before they become obvious in broad indexes. It is not a
license to invent precision. Each diagnostic row carries a `data_support` label:

- `direct`: the project has a reasonably direct public-data measure.
- `proxy`: the project has a tradable public-market proxy that can be tested.
- `thin_proxy`: the project has only partial public-market or news-derived
  evidence. These rows can guide research, but they should not directly drive
  trade sizing until the proxy improves backtests or paper monitoring.
- `unsupported_watchlist`: the idea is potentially important, but the project
  lacks the institutional feed needed to measure it. These rows are explicitly
  non-trading rows.

The dashboard should surface these labels so a future user or agent can tell the
difference between "the bot sees this in the data" and "a smart external source
raised this, but we do not have the data to verify it."

## Promotion Standard For New Diagnostics

Cross-source diagnostics start as explainers. They are allowed to shape research
questions and dashboard interpretation, but they should not change portfolio
weights until they pass a promotion test. A diagnostic can become an allocation
input only after a dedicated experiment answers these questions:

1. Does adding the signal improve the target objective versus a matched strategy
   without the signal?
2. Does the improvement survive walk-forward windows, drawdown regimes,
   transition regimes, transaction costs, and turnover/churn checks?
3. Does the signal improve at least one economically meaningful outcome such as
   CAGR, max drawdown, Calmar, re-entry timing, left-tail regime loss, or
   paper-monitoring drift?
4. Does the signal have enough historical availability to avoid post-hoc
   narrative fitting?
5. Is the signal independent enough to add information beyond trend, credit,
   volatility, breadth, macro pressure, and existing event-risk inputs?

Until those tests pass, the UI should label the diagnostic as
`explainer_research_only` with `no_direct_sizing_authority`.

### AI Earnings-Quality / Reflexivity Boundary

AI accounting-quality, circular-financing, and private-mark stories can be
tracked as current news/event context, but they are not model drivers.

Direct private AI mark-to-market gains, circular private financing, and
company-specific private investment revaluations are not consistently available
as free, long-history data. Until the project adds SEC/XBRL extraction and
shows out-of-sample value in ablation, walk-forward, and paper-monitoring tests,
these items should remain `watch_context` events with `sizing_authority: false`.
They should not create a scored proxy, dashboard promotion, or allocation
change.

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
- Direct hyperscaler capex, free-cash-flow, depreciation, order-book, and
  consensus-revision feeds.
- Dealer gamma, CTA exposure, option-flow, short-interest, and margin-debt data
  with useful timeliness.
- Bloomberg-style sector, country, and constituent-contribution monitors.
- Reliable IPO, lockup, secondary-offering, convertible-issuance, and free-float
  calendars.
- Private-credit fund marks, redemption queues, covenant-level loan data, and
  dealer balance-sheet measures.

## Intended Use

The new regime-pulse layer should not automatically override strategy signals.
It should answer four operating questions:

1. Is the status quo macro backdrop risk-on, mixed, or risk-off?
2. Are positioning and crowding making the same trade fragile?
3. Is the likely growth-inflation regime changing?
4. Should the risk engine cut, hold, or re-risk exposure?

The cross-source diagnostic layer should answer a separate set of questions:

1. Which recurring external-source themes are supported by our public
   data proxies?
2. Which themes are only weakly proxied and therefore useful mainly for research
   design?
3. Which themes require paid or unavailable data and must stay out of trade
   sizing?
4. Are multiple source-informed pressures pointing in the same direction, or are
   they contradictory?
