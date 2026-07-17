# Trade Bot Technical Explainer

Status: canonical technical reference. Last reviewed: 2026-07-17.

This document explains how the major pieces of Trade Bot work behind the scenes.
It is intended for developers, reviewers, and technical users who need to answer:
"Where does this number come from?" or "Does the system use this approach?"

## Architecture Summary

Trade Bot is a local Python application with these main layers:

| Layer | Primary Modules | Role |
| --- | --- | --- |
| Config/defaults | `config.py`, `DEFAULTS.py`, `configs/*.yaml` | Defines universes, strategies, execution assumptions, thresholds, and file paths. |
| Data | `data/market_data.py`, `data/fred_data.py` | Loads Yahoo-compatible prices and FRED macro data into local caches. |
| Features | `features/indicators.py`, `features/valuation.py` | Computes returns, drawdowns, momentum, volatility, and valuation helpers. |
| Backtest | `backtest/engine.py`, `backtest/metrics.py`, `backtest/windows.py` | Applies target weights, transaction costs, lag, turnover, equity curves, and metrics. |
| Strategy construction | `strategies/momentum.py`, `research/experiments.py` | Builds baseline weights and research candidate weights. |
| Current state | `research/current_state.py` | Builds market health, momentum state, confirmation matrix, macro state, scenarios, and regime diagnostics. |
| Risk and sizing | `portfolio/risk.py`, `research/trade_decision.py` | Converts scenarios and constraints into target posture and position deltas. |
| Research scoring | `research/validation.py`, `research/strategy_outcome_utility.py`, `research/curation.py` | Scores candidates for robustness, outcome utility, and monitoring readiness. |
| Cycle tracking | `research/cycle_tracker.py` | Builds speculative-cycle phase nowcasts, horizon phase frontiers, conditional candidate scores, and prior-only validation artifacts. |
| Monitoring/journal | `storage/warehouse.py`, `trading/journal.py`, `trading/book_alignment.py` | Stores paper windows, valuations, tickets, executions, and book alignment. |
| Dashboard | `dashboard/*.py` | Streamlit UI over snapshots, warehouse tables, and research artifacts. |
| CLI | `cli.py` | Operational commands for daily updates, snapshots, experiments, ML, monitoring, and valuation. |

## Data Flow

The normal daily flow is:

```text
configs -> data load -> baseline run -> snapshot -> warehouse migration -> paper valuation -> dashboard
```

The main command is:

```bash
poetry run trade-bot run-daily-update
```

Internally it:

1. loads `configs/baseline.yaml`,
2. loads market prices,
3. loads macro series,
4. runs configured baseline strategies,
5. builds current state,
6. runs news monitor and event activation,
7. builds event-risk study,
8. runs signal-inclusion tests,
9. builds the trade decision,
10. saves a snapshot to DuckDB and pickle artifact,
11. writes the static HTML report,
12. refreshes the strategy registry,
13. migrates experiment and journal outputs,
14. writes paper valuations for active windows.

## Configuration And Defaults

Defaults belong in `src/trade_bot/DEFAULTS.py`. Configurable user-level choices
belong in `configs/*.yaml`.

Use `DEFAULTS.py` for:

- file paths,
- threshold constants,
- default tax assumptions,
- dashboard display defaults,
- driver-rotation thresholds,
- monitoring defaults,
- fallback ticker metadata.

Use `configs/baseline.yaml` for:

- data start/end,
- execution assumptions,
- ticker universe,
- baseline strategies,
- tax-account config overrides.

Avoid hardcoding thresholds at the top of feature modules. That creates drift.

## Price And Macro Data

Market data is pulled by `load_or_fetch_yahoo_prices`. It uses local cache files
unless refresh is requested. Macro data is configured in `configs/macro_fred.yaml`
and loaded through `load_or_fetch_fred_data`.

Important limitations:

- Yahoo-compatible market data can revise or occasionally fail.
- FRED data has release timing and revision issues.
- Public proxies are not equivalent to proprietary terminal data.
- Private credit, AI capex, and liquidity plumbing often use imperfect proxies.

## Backtest Engine

The backtest engine takes:

- adjusted price data,
- target weights,
- execution assumptions,
- optional volatility target,
- optional drawdown control.

It applies:

- configured signal lag,
- rebalance cadence,
- transaction costs,
- turnover tracking,
- equity curve compounding.

The default execution assumptions live in config:

```yaml
execution:
  initial_capital: 100000.0
  transaction_cost_bps: 5.0
  rebalance: "W-WED"
  signal_lag_days: 1
```

This matters because the system is not allowed to look ahead or assume same-day
perfect execution.

## Core Metrics

Key metrics are calculated in `backtest/metrics.py`.

| Metric | Meaning |
| --- | --- |
| CAGR | Annualized compounded return. |
| Volatility | Annualized standard deviation of returns. |
| Sharpe | Excess return per unit volatility, using configured simple assumptions. |
| Sortino | Return per unit downside volatility. |
| Max drawdown | Worst equity peak-to-trough decline. |
| Calmar | CAGR divided by absolute max drawdown. |
| Average turnover | Average absolute portfolio weight change per rebalance. |
| Ulcer Index | Depth and persistence of drawdowns. |

Formula contracts are documented in `docs/math_model_audit.md` and tested in
`tests/test_math_contracts.py`.

## Current-State Engine

`research/current_state.py` builds the daily market read. It calculates:

- market date,
- momentum state,
- data quality,
- macro signals,
- macro category summary,
- positioning/crowding,
- regime instability,
- regime pulse,
- signal coverage,
- confirmation matrix,
- market health,
- risk score and status,
- strategy alerts,
- scenario lattice and rollup.

The risk score is intentionally a compact current-state measure. It is not the
same thing as a strategy score.

## Scenario Lattice

`research/future_scenarios.py` builds scenario probabilities from current market
and macro state. The scenario buckets include states such as:

- risk-off,
- transition,
- broad risk-on,
- narrow AI-led melt-up,
- risk-off then relief.

The scenario output is used by `trade_decision.py` to shape risk budget and
target posture. It does not directly forecast exact prices.

## Scenario / Phase Frontier

`research/cycle_tracker.py` builds the Speculative Cycle Tracker. It is a
batch-backed research/watch layer for markets where speculative leadership,
liquidity, and unwind risk matter. It translates price-observable evidence into
0M nowcast phase probabilities, then blends the current phase read with the
current scenario lattice to create forward horizon phase frontiers.

The phase taxonomy is:

- normal cycle,
- acceleration,
- pre-break,
- early unwind,
- liquidation,
- bottoming,
- recovery,
- post-unwind compounding.

The tracker uses price-available features such as growth leadership
acceleration, narrow leadership pressure, breadth improvement, credit pressure,
volatility pressure, large-move share, QQQ/semiconductor unwind pressure, deep
drawdown, short reversal, and broad trend. The 0M row is the current phase
classification only; current scenario probabilities can affect today's 1-month,
3-month, 6-month, and 1-year phase frontier. Historical validation does not use
current scenario probabilities at old origins.

The candidate layer has two surfaces. `cycle_candidate_scores.csv` ranks assets
for the current dominant phase. `cycle_phase_candidate_frontier.csv` ranks
assets for every available phase/horizon pair so a user can ask, "if
liquidation, bottoming, recovery, or renewed acceleration dominates this
horizon, what historically worked better in similar prior states?" Those
frontier scores combine prior-only validation metrics with current momentum,
drawdown, phase fit, and current phase probability. They are an inspection
shelf, not optimized portfolio weights.

The leakage rule is explicit: at each historical origin the feature snapshot is
built from prices through that origin only, and the evaluated return window
starts on the next trading session. This prevents the tracker from using future
price data to classify the phase being validated.

Run:

```bash
poetry run trade-bot run-cycle-tracker
```

The command writes artifact CSVs under `reports/cycle_tracker/` and persists
summary tables to DuckDB:

- `cycle_tracker_runs`,
- `cycle_tracker_phase_probabilities`,
- `cycle_tracker_transition_forecast`,
- `cycle_tracker_evidence`,
- `cycle_tracker_candidate_scores`,
- `cycle_tracker_phase_candidate_frontier`,
- `cycle_tracker_validation_metrics`.

Dashboard V2 reads these persisted outputs in Research -> Cycle Tracker. The
dashboard should not run the cycle tracker validation directly on cold start.
Use the output to understand current phase risk, plausible next phases, and
conditional winner candidates under similar historical phase reads. Do not use
it as an exact bubble-top timer or a standalone allocation override.

## Trade Decision

`research/trade_decision.py` converts current state, events, news, signal
inclusion, and scenario outlook into a target allocation bridge.

Outputs include:

- recommended action,
- current posture,
- scenario-adjusted posture,
- ticker-level target weights,
- delta weights,
- risk adjustment reasons,
- human explanation.

The system distinguishes:

- model target,
- risk-adjusted target,
- decision-sanity capped target,
- logged book alignment.

## Portfolio Risk Engine

`portfolio/risk.py` applies risk controls after the scenario target. It includes:

- beta/factor checks,
- expected shortfall,
- stress loss,
- concentration constraints,
- scenario minimum defensive allocation,
- scenario-weighted stress constraints,
- portfolio risk multiplier.

The risk engine is a guardrail. It should not be treated as a complete
institutional risk system, but it is more than a simple momentum toggle.

## Decision-Sanity Overlay

Decision sanity prevents context-only or event-only pressure from forcing huge
defensive moves without market confirmation. Larger defensive moves should
generally require confirmation from at least some combination of:

- credit,
- volatility,
- breadth,
- trend,
- price action.

This overlay is tested in paired experiments. It is not simply hand-waved into
production.

## News And Events

News monitoring is handled by `research/news_monitor.py`, event study logic by
`research/event_risk.py`, and dashboard display by `dashboard/news_macro.py`.

News/event items can be:

- activated events,
- context only,
- high urgency,
- leading warning,
- coincident,
- lagging,
- phase uncertain.

The important design rule is that news generally informs context and risk review.
It should not become a direct trade driver unless validated or confirmed by
market data.

## Driver Rotation

`research/driver_rotation.py` creates a table that separates:

- normally important,
- currently active,
- emerging importance,
- fading importance.

Each driver has:

- historical relevance,
- current activation,
- short/long change,
- model role,
- data support,
- evidence.

The dashboard visualizes this as a quadrant/heatmap/table. It helps users see
what is moving the market narrative versus what has actually mattered in tests.

## Signal Evidence And Ablations

`research/signal_evidence.py` tags strategy families and compares parent/control
pairs to estimate marginal contribution.

Signal recommendations include:

- validated contributor,
- promising mixed,
- not proven,
- context only,
- research gap.

This is the preferred path for deciding whether a signal deserves more operating
surface or should be pruned.

## Experiment Engine

`research/experiments.py` generates candidate strategies for a given iteration.
Each iteration creates scorecards, candidates, regime metrics, walk-forward
summaries, window summaries, and manifests.

Run:

```bash
poetry run trade-bot run-experiment-iteration --config configs/baseline.yaml --iteration 161 --output-dir data/experiments_reset_v2
```

The system is designed for iterative research:

1. test several candidates,
2. score them,
3. promote/evolve/reject,
4. inspect results,
5. design the next batch.

## Validation And Scoring

Candidate scoring uses multiple layers:

- raw performance,
- promotion score,
- robustness score,
- monitoring readiness,
- walk-forward diagnostics,
- regime metrics,
- overfit risk,
- left-tail behavior,
- outcome utility,
- taxable impact where available.

No single metric should dominate selection.

## Outcome Utility

`research/strategy_outcome_utility.py` adds a growth-constrained objective. It
models an accumulation account with configurable horizon, starting value,
contributions, drawdown soft limit, and hard limit.

It computes:

- terminal wealth,
- terminal wealth with contributions,
- benchmark wealth deltas,
- recovery return needed,
- drawdown penalties,
- growth-constrained utility score,
- growth utility tier.

This explicitly encodes the insight that higher CAGR with tolerable drawdown may
be better than low-drawdown undergrowth for a long accumulation horizon.

The dashboard exposes sequence-aware simulations in the top-level
**Simulation Lab** workbench. The historical sequence model uses a block
bootstrap over daily strategy returns, then applies the same starting account,
annual contribution, and horizon settings, with the annual contribution split
across the configured cadence. The default cadence is monthly. It reports
P10/median/P90 terminal wealth plus simulated drawdown and Ulcer Index
summaries. This is stronger than the deterministic CAGR card because it shows
path risk, but it remains a historical-resampling diagnostic.

The forward simulation engine in `research/forward_simulation.py` adds the
regime-conditioned layer. It labels the selected strategy's historical daily
returns into `risk_off`, `transition`, `risk_on_fragile`, and `risk_on`, blends
today's scenario probabilities with empirical regime-transition paths, then
samples forward return blocks with scheduled contributions. The dashboard
reports P10/median/P90 terminal wealth, median simulated drawdown,
severe-drawdown probability, capital-shortfall probability, and the average
regime mix across paths. This is the strongest planning layer in the app, but it
is still scenario-conditioned historical simulation rather than a guarantee or
automatic trading rule.

The same module now includes a rolling-origin validation harness for the
simulation engine. For historical month-end or quarter-end origins, it trains
only on returns available through that origin, simulates configured forward
horizons such as 3 months, 6 months, 1 year, 3 years, and 5 years, then compares
realized future returns and drawdowns with the simulated P10/P50/P90 bands. The
summary scores interval coverage, bullish/bearish median bias, severe-drawdown
probability calibration, hindsight launch stance, and multi-strategy ranking
usefulness. This is the calibration layer the forward simulator needs before it
can influence sizing or launch decisions.

The forward simulator now has three advanced controls layered on top of the
original regime-block sampler:

- **Duration-aware regime transitions:** simulated regimes track how long they
  have persisted and adjust transition odds against historical regime-duration
  distributions. Young regimes get more persistence; overstretched regimes get
  more exit pressure.
- **Covariate-matched return blocks:** historical blocks carry trend,
  volatility, drawdown, shock, and optional external numeric covariates. The
  sampler can prefer blocks that resemble the latest state instead of sampling
  every broad regime bucket uniformly.
- **Factor-proxy paths:** when factor proxy returns are available, the engine
  can fit strategy returns to transparent factor betas, sample factor blocks,
  and reconstruct strategy paths from factor returns plus residual behavior.
  This is a proxy stress lens, not a full synthetic-price rerun of every
  strategy rule.

The same validation harness is available from the CLI:

```bash
poetry run trade-bot validate-simulation-engine
```

By default it uses the latest stored snapshot strategy returns and writes CSV
evidence under `reports/simulation_validation/`. A `--scenario-history` file may
be supplied, but it must include a date column such as `origin_date`,
`as_of_date`, `date`, `created_at_utc`, or `created_at`; undated scenario rows
are ignored to avoid contaminating old origins with current scenario
probabilities.

Pass `--ablation` to also write a model-ablation CSV for the selected strategy.
That file compares baseline regime blocks, duration-aware transitions, duration
plus covariate matching, and factor-proxy paths when enough factor proxy prices
exist in the latest snapshot. Use it as the first check before trusting the more
complex simulation machinery: if the enhanced variants do not improve
calibration or drawdown probability scores, keep them as stress lenses rather
than decision inputs. The ablation is opt-in because it repeats rolling-origin
validation several times and is materially slower than the default validation
run.

Simulation Lab can also run the same bootstrap and regime-conditioned path
machinery for configured reference portfolios such as Hold SPY and Hold QQQ.
Those references are not a separate benchmark shortcut; they use the same
terminal-wealth, contribution, and drawdown simulation settings as the selected
strategy. The comparison table reports selected-strategy median forward wealth
minus each reference median so the user can evaluate whether a candidate's
extra complexity is earning its keep versus doing nothing.

The Strategy Simulations view includes an advanced diagnostics table and a
Current-path resemblance section. The advanced table reports duration/covariate
regime paths and, when possible, factor-proxy paths. Use the factor-model
R-squared and covariate-match distance as confidence checks: weak factor fit or
large match distance means the simulation is more of a stress test than a
high-confidence planning distribution. Current-path resemblance is not the same
as rolling-origin validation; it compares the selected live simulation output
with the strategy's own historical profile.

Simulation Lab keeps this forward modeling separate from Research Lab's
empirical evidence surfaces. Research Lab answers "which strategies worked and
why?" while Simulation Lab answers "what future range could this selected
strategy experience under deterministic, bootstrapped, and current-scenario
conditioned assumptions?"

## ML Diagnostics

ML lives under `src/trade_bot/ml` and `research/future_state_ml.py`.

Current ML use is diagnostic and research-oriented:

- future-state classification,
- Brier score,
- calibration error,
- balanced accuracy,
- feature importance,
- drift diagnostics.

Run:

```bash
poetry run trade-bot run-ml-diagnostics --config configs/baseline.yaml --profile standard
```

Use `--profile research` for heavier sweeps. The dashboard should not train
models on cold start.

## Factor Attribution

`research/factor_attribution.py` decomposes strategy behavior into transparent ETF
proxy factors:

- market beta,
- QQQ/growth beta,
- AI/semis beta,
- breadth,
- cyclicals,
- rates/duration,
- credit,
- commodities,
- volatility,
- residual behavior.

This helps answer whether strategies are genuinely different or mostly disguised
versions of the same AI/growth bet.

## Monitoring

Monitoring is handled by `storage/warehouse.py`. It stores:

- strategy registry,
- monitoring windows,
- daily valuations,
- snapshot metrics,
- experiment scorecards,
- journal data.

The key table is `strategy_daily_valuations`, which tracks forward paper results
from the first available trading point on or after the monitoring-window start
date rather than importing full-history backtest gains. This is idealized
strategy monitoring; actual timing, quantities, fees, and missed executions stay
in the Forward Test journal and book-alignment layer.

## Forward Test And Journal

`trading/journal.py` stores tickets and executions in SQLite. Forward Test uses
this to log:

- recommendation tickets,
- execution mode,
- account label,
- ticker,
- side,
- quantity,
- price,
- fees,
- timestamp,
- notes.

Book Alignment uses journal executions to estimate the current logged book.

## Book Alignment

`trading/book_alignment.py` compares the latest target posture against the local
book derived from logged executions.

It is not a broker import. It answers whether the locally tracked paper/live
book is close enough to the latest model target.

The source-of-truth boundary matters:

- recommendation tickets record suggested actions and do not change holdings;
- logged executions change the locally tracked paper/live book;
- Forward Test can recalculate alignment from the freshest SQLite journal
  state;
- DuckDB journal and monitoring tables are warehouse mirrors and can lag until
  the migration/valuation jobs are rerun.

After logging executions, the user should refresh/recalculate Book Alignment
before treating a dashboard rebalance warning as still actionable. If the
warehouse-backed monitoring tables disagree with Forward Test immediately after
logging, prefer Forward Test for current book state and rerun the warehouse
migration or paper valuation job before using monitoring/audit tables.

## Taxable Layer

Tax modules live under `src/trade_bot/tax`.

They estimate:

- open lots,
- realized lots,
- short-term and long-term gains,
- wash-sale warnings,
- tax drag,
- loss carryforward,
- TLH candidates.

This is a research layer. Broker lots and professional tax review remain required
for real taxable decisions.

## Storage

The system uses local storage:

| Store | Path | Purpose |
| --- | --- | --- |
| Cache | `data/cache/` | Market, macro, and news input caches. |
| Run store | `data/run_store/trade_bot.duckdb` | Snapshot metadata and warehouse tables. |
| Snapshot artifacts | `data/run_store/snapshots/` | Pickled `BaselineRun` objects for fast dashboard loads. |
| Journal | `data/trading_journal.sqlite` | Tickets, executions, and derived tax lots. |
| Reports | `reports/` | Static report and experiment outputs. |

## Dashboard Structure

The archived V1 dashboard is split into:

- app shell and sidebar: `dashboard/app.py`,
- top overview: `dashboard/overview.py`,
- operating cards: `dashboard/briefs.py`,
- section navigation: `dashboard/navigation.py`,
- section modules: command center, risk scenarios, research lab, monitoring,
  news/macro, performance, forward test,
- styling: `dashboard/styles.py`,
- explanations: `dashboard/metric_explainers.py` and
  `dashboard/ticket_explainers.py`.

The primary dashboard is the summary-first shell under `dashboard_v2/`. It does
not introduce a second storage system. Instead, it puts service wrappers around
the existing snapshot store, DuckDB warehouse, and artifact directories, then
renders pages from small view-specific summaries. The important performance
rule is that Research, Simulation, and Monitoring overview pages should not
hydrate deep diagnostics, raw split files, full candidate workbenches, or path
engines until the user selects that subview.

Start the primary dashboard with:

```bash
poetry run trade-bot run-dashboard
```

The archived V1 fallback remains available with `run-dashboard-v1` for
comparison/debugging only.

## Testing Strategy

Tests cover:

- data loaders,
- config parsing,
- backtest formulas,
- current-state logic,
- trade decision logic,
- risk engine,
- experiments,
- curation,
- dashboard rendering,
- monitoring,
- journal,
- tax tracking,
- ML diagnostics.

Run focused tests during development and full tests before larger handoff:

```bash
poetry run ruff check src tests
poetry run pytest -q
```

For formula-sensitive changes:

```bash
poetry run pytest tests/test_math_contracts.py tests/test_backtest_engine.py tests/test_metrics.py -q
```

For dashboard changes:

```bash
poetry run pytest tests/test_dashboard_app.py tests/test_dashboard_navigation.py tests/test_dashboard_explainability.py -q
```

## Extension Guidelines

When adding a new feature:

1. Put defaults in `DEFAULTS.py`.
2. Put user choices in config.
3. Keep dashboard cold-start fast.
4. Add research artifacts before operating claims.
5. Separate model drivers from explanatory context.
6. Add tests.
7. Update docs.

When adding a new signal:

1. Ask whether data exists.
2. Ask whether historical/proxy analogs exist.
3. Add it as context first.
4. Build ablation or paired tests.
5. Promote only if it improves outcomes.

When adding a new strategy:

1. Define thesis.
2. Define universe.
3. Define risk-on and defensive sleeves.
4. Define re-entry/off-ramp logic.
5. Backtest with costs and lag.
6. Score across regimes.
7. Compare against references.
8. Add paper monitoring only if operational.

## What The System Does Not Do

Trade Bot does not:

- place trades,
- guarantee performance,
- import broker positions automatically,
- provide legal or tax advice,
- solve intraday execution,
- use proprietary institutional datasets,
- turn every news item into a trade,
- make LLM-generated claims authoritative.

These boundaries are intentional.
