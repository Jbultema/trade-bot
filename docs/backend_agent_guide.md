# Backend Agent Guide

This document is the canonical backend onboarding guide for a future engineer or AI agent working in this repository. It explains what this project is trying to accomplish, why the system is structured this way, where the major code lives, and what standards should govern future changes.

This guide is intentionally separate from `README.md`. The README is for running the app. This file is for understanding and extending the backend without losing the project intent.

## Executive Summary

Trade Bot is a local-only, Python-first research and decision-support system for long-only swing and momentum trading. It is not an automated execution bot. The system studies stocks, ETFs, macro time series, news/events, market internals, and strategy experiments, then produces human-reviewed trade recommendations and paper-monitoring evidence.

The core goal is to pursue high long-term returns while reducing left-tail risk enough that the strategy is viable for real money after a long paper-monitoring period. The project owner is not trying to build an intraday trading desk, a shorting strategy, an options system, or a black-box LLM trading agent. The intended operating cadence is human-triggered trades a few times per day at most, with a preference for holding periods that are measured in days, weeks, or months rather than minutes.

The system is built around three ideas:

1. Price and macro data should drive most decisions through auditable Python code.
2. Risk management and position sizing are first-class, not afterthoughts.
3. Every attractive strategy must survive rolling-window, walk-forward, regime, churn, and paper-monitoring checks before it can influence live decisions.

## Project Boundaries

Hard boundaries:

- Local machine first. Do not assume cloud services, hosted databases, Databricks, or work infrastructure.
- Long-only by default. Do not introduce shorts, leverage, options, futures, or derivatives unless the project owner explicitly asks for a separate research track.
- Human-reviewed execution. The system can suggest trades, sizing ranges, price bands, and rationale, but should not place orders.
- At-least-about-one-day trading horizon. The system can refresh intraday data or news, but it should not require ultra-low latency or frequent micro-adjustments.
- No investment advice claims. The app is a research and paper-monitoring system; recommendations are decision-support outputs requiring human review.

Soft design principles:

- Prefer simple, testable, lag-safe rules over clever but untestable heuristics.
- Prefer broad, robust evidence over one historical winner.
- Prefer strategy families and operating systems over a zoo of tiny variants.
- Treat dashboard explanations as part of the product, not decoration. The project owner needs to understand what to do, why, and what changed.

## Why This Is Not A Pure Forecasting Project

The project owner is a forecasting expert, but the trading problem is broader than forecasting. A useful system has to answer at least five questions:

1. What can be bought in a normal brokerage account?
2. Which assets or strategy families should be favored now?
3. How large should positions be after risk constraints?
4. What would make the system reduce risk or re-enter risk?
5. Is the strategy behaving in paper monitoring like it behaved in backtests?

Forecasting enters the system through scenario probabilities, regime classification, re-risking classifiers, off-ramp classifiers, signal inclusion tests, and ML overlays. Forecasting does not replace portfolio construction or risk constraints. A forecast that cannot be translated into a testable allocation rule is not operationally useful here.

## Current Operating Model

The core workflow is:

1. Load prices, macro data, configured events, and news-source metadata.
2. Backtest configured baseline strategies with lagged execution and transaction costs.
3. Build the current-state object from market, macro, scenario, positioning, news, event, and risk diagnostics.
4. Convert current state plus primary strategy weights into a trade decision.
5. Apply scenario, event, macro, decision-sanity, and portfolio-risk constraints.
6. Save a snapshot to the local run store.
7. Render dashboard sections and reports from that snapshot.
8. Paper-monitor selected strategies through journal and monitoring-window tables.
9. Iterate research candidates, curate the useful set, and promote only what survives validation.

The main backend orchestration entry point is `src/trade_bot/research/baselines.py`:

```text
run_configured_baselines(config)
  -> load_or_fetch_yahoo_prices
  -> load_or_fetch_fred_data
  -> build_strategy_weights
  -> run_backtest
  -> calculate metrics, rolling windows, calendar metrics
  -> build_current_state
  -> run_news_monitor and activate_news_events
  -> run_event_risk_study
  -> run_signal_inclusion_tests
  -> build_trade_decision
  -> BaselineRun
```

`BaselineRun` is the object most of the dashboard consumes. It contains prices, macro data, strategy backtests, metrics, current-state diagnostics, event/news runs, signal inclusion results, trade decision output, and portfolio-risk output.

## Important Runtime Commands

Use Poetry and the local virtual environment. The project targets Python 3.12.

```bash
poetry install --sync
poetry run trade-bot fetch-prices
poetry run trade-bot run-baselines
poetry run trade-bot build-snapshot
poetry run trade-bot run-daily-update
poetry run trade-bot run-dashboard
```

`run-daily-update` is the canonical one-command operating refresh. It refreshes market data, macro data, and news by default; builds and stores a fresh snapshot; writes the baseline report; refreshes the local strategy registry and warehouse tables; migrates experiment/journal outputs; and writes daily paper-monitoring valuations for active monitoring windows. Use cached inputs only when intentionally doing a faster local check:

```bash
poetry run trade-bot run-daily-update --cached-data --cached-macro --cached-news
```

The dashboard exposes the same workflow as the primary sidebar button labeled `Run Full Daily Update`. That button queues the job in the local run store so the app can keep serving the latest completed snapshot while the refresh runs in the background. The sidebar also exposes targeted background jobs for warehouse migration, paper valuation, monitoring-window seeding, and standard/research ML diagnostics. Those buttons should call `RunStore` job helpers rather than launching untracked subprocesses from Streamlit.

The primary dashboard lives under `src/trade_bot/dashboard_v2/` and can be
started with:

```bash
poetry run trade-bot run-dashboard
```

The old Streamlit app under `src/trade_bot/dashboard/` is archived for
comparison/debugging through `poetry run trade-bot run-dashboard-v1`. New
dashboard work should target V2 unless there is a specific fallback bug to
inspect. V2 uses a small service layer over the same storage surfaces:

- `services/runtime.py` for config paths, snapshots, live fallback, book
  alignment, and action headline construction,
- `services/warehouse_service.py` for DuckDB tables and monitoring/simulation
  summaries,
- `services/experiment_service.py` for scorecard and aggregate experiment
  artifacts,
- `services/artifact_service.py` for PBO, leadership, router, Cycle Tracker,
  and other report artifacts,
- `services/job_service.py` for sidebar-triggered background jobs.

The V2 rule is summary-first rendering. Pages should load cards and compact
tables first, then put expensive charts, full workbenches, raw artifacts, and
path engines behind explicit view selectors or buttons. Avoid `st.tabs()` for
heavyweight areas because Streamlit can execute hidden tab contents.

Keep broad experiment sweeps, dependency management, Git operations, and live-broker execution out of one-click dashboard buttons. They are long-running, parameterized, environment-sensitive, or intentionally require explicit review.

Other useful commands:

```bash
poetry run trade-bot list-snapshots
poetry run trade-bot run-experiment-iteration
poetry run trade-bot run-ml-diagnostics
poetry run trade-bot seed-monitoring-windows
poetry run trade-bot run-paper-valuation
poetry run trade-bot list-champion-challenger
poetry run pytest -q
poetry run ruff check src tests
```

When the dashboard feels slow, prefer `build-snapshot` before app launch. The intended dashboard path is to load a recent snapshot, not recompute all research on every page open.

## Repository Map

Documentation navigation starts in `docs/doc_index.md`. Keep dated plans in
`docs/archive/` and keep current operating behavior in the canonical docs listed
there. Do not let dated experiment notes masquerade as live system status.

High-level code ownership:

- `src/trade_bot/DEFAULTS.py`: centralized default constants. New defaults should usually go here instead of being scattered through modules.
- `src/trade_bot/config.py`: typed config model and ticker universe assembly.
- `src/trade_bot/data/`: market and macro data loaders/caches.
- `src/trade_bot/features/`: reusable feature engineering and valuation helpers.
- `src/trade_bot/backtest/`: lag-safe backtest engine, performance metrics, rolling/calendar windows.
- `src/trade_bot/strategies/`: strategy weight generators anchored around momentum and related allocation logic.
- `src/trade_bot/research/`: current-state engine, scenario logic, experiment machinery, event/news monitors, ML diagnostics, curation, and validation.
- `src/trade_bot/portfolio/`: portfolio risk engine and constraints.
- `src/trade_bot/trading/`: paper/live journal concepts, tickets, executions, monitoring windows, and book alignment.
- `src/trade_bot/storage/`: local DuckDB run store and warehouse support.
- `src/trade_bot/dashboard/`: Streamlit app and section-level renderers.
- `src/trade_bot/reporting/`: static report generation.
- `configs/`: baseline strategy, event, macro, and news-source configuration.
- `docs/`: project design, math audit, protocols, and research notes.
- `tests/`: contract tests for formulas, UI helpers, strategy behavior, storage, ML, and research views.

## Data And Storage Architecture

The system is deliberately local and file-backed.

Primary data paths:

- `data/cache/`: cached price, macro, and news data.
- `data/run_store/trade_bot.duckdb`: local run-store metadata, snapshot manifests, jobs, and related tables.
- `data/run_store/snapshots/`: pickled `BaselineRun` snapshots for fast dashboard loading.
- `data/trading_journal.sqlite`: local trading journal and recommendation-ticket state.
- `data/experiments_reset_v2/`: active reset-era experiment outputs when present locally.
- `reports/experiments/`: historical experiment outputs and scorecards from earlier runs.
- `reports/`: static reports, app logs, and runtime files.

Storage design rationale:

- Cached market data avoids unnecessary network calls and makes local work repeatable.
- Snapshot artifacts keep the dashboard snappy and preserve exact recommendation states.
- DuckDB is used for local analytical metadata because it is fast, simple, and file-based.
- SQLite remains suitable for journal-style transactional paper/live records.
- The project intentionally avoids a remote database until there is a concrete need.

Future agents should avoid creating parallel storage patterns unless there is a clear reason. If a result needs to be queried by the dashboard or used across runs, prefer extending the run store or warehouse instead of writing another ad hoc CSV.

The dashboard treats the reset-era experiment root as the active
research root when it exists. Older `reports/experiments/` artifacts remain
auditable evidence, but they are historical unless a workflow explicitly merges
or selects that root.

## Account And Tax Model Status

Base backtests and scorecards are pre-tax / IRA-like unless a field is
explicitly labeled after-tax. The account-aware taxable layer runs as a parallel
evaluation path: `TaxAccountProfile`, `TaxLotLedger`, taxable backtest
enrichment, wash-sale estimates, loss-harvesting candidates, and journal-derived
tax-lot tables live under `src/trade_bot/tax/` and
`src/trade_bot/trading/journal.py`. Keep the existing pre-tax/IRA-style rankings
intact and add taxable-specific fields rather than silently penalizing strategy
returns.

Do not let tax optimization override real risk exits. Tax-aware logic can delay
marginal trades, prefer new cash, or harvest losses, but left-tail risk control
remains the higher-priority guardrail.


Dashboard surfaces:

- `Research Lab -> Experiment Monitor -> Taxable Impact` is the cross-strategy
  after-tax comparison surface.
- `Research Lab -> Experiment Monitor -> Candidate Details workbench` includes a
  compact selected-strategy estimated taxable readout when scorecard fields are
  available.
- `Forward Test` remains the execution journal surface. Derived tax-lot tables
  are rebuilt from that journal; do not treat them as broker-confirmed lots.

When adding taxable features, keep labels explicit and avoid silently mixing
pre-tax and after-tax rankings. Any score that uses taxable assumptions should
include `after_tax`, `tax`, `realized`, `wash_sale`, or `loss_carryforward` in
its column name unless there is a very strong reason not to.

## Config And Defaults

The project owner strongly prefers defaults to be centralized. Use `src/trade_bot/DEFAULTS.py` for reusable default values, then pass them into modules through function signatures or config models. Do not add a parallel singular `DEFAULT.py` shim or scatter defaults across individual implementation modules.

Config files:

- `configs/baseline.yaml`: strategy universe, execution assumptions, configured approaches.
- `configs/events.yaml`: curated event-risk scenarios and historical/current event definitions.
- `configs/macro_fred.yaml`: macro time-series catalog.
- `configs/news_sources.yaml`: news-source coverage map.

Be careful with config semantics:

- A configured strategy is not automatically an operational candidate.
- A news/event item is not automatically a trading command.
- A macro signal is not automatically included in allocation. Inclusion must be tested.
- Any new config field should have a documented default, type validation, and at least one test.

## Strategy Framework

The strategy universe has expanded over time, but the core families are:

1. Baselines and reference portfolios.
   - Examples: broad equity, Nasdaq/growth, balanced references, defensive/cash references.
   - Purpose: keep sophisticated systems honest against simple buy-and-hold alternatives.

2. Momentum and dual momentum.
   - Use lookback returns, skip periods, trend filters, risk-adjusted momentum, and defensive fallback.
   - Purpose: capture persistent medium-term trends without intraday trading.

3. AI-beta escape variants.
   - Target the high-CAGR AI/growth sleeve but add off-ramps into defensive assets when risk breaks.
   - Purpose: keep high upside potential while avoiding the worst left-tail drawdowns.

4. Sector/factor rotation.
   - Rotate among investable ETFs and factors based on momentum, regime, macro, and risk state.
   - Purpose: avoid reducing the problem to only `QQQ` versus cash.

5. Defensive credit/rates systems.
   - Use credit, duration, T-bill, and lower-risk sleeves when market risk is not being paid well.
   - Purpose: provide left-tail protection and carry-like alternatives.

6. Dip-buying and re-risking systems.
   - Add risk back in metered steps after drawdowns when repair conditions improve.
   - Purpose: avoid the prior failure mode of getting risk-off correctly but staying risk-off too long.

7. Operating-system candidates.
   - Combine allocation, risk-off, re-entry, churn, and scenario overlays into paper-monitorable systems.
   - Purpose: reduce the final live/paper set to a small number of understandable systems.

The desired end state is not hundreds of live strategies. The research engine may test hundreds of variants, but the operating layer should eventually run a small number of champions/challengers plus references.

## Current-State Engine

Source: `src/trade_bot/research/current_state.py`.

The current-state engine is the daily diagnostic spine. It does not directly place trades. It builds the information that later modules use to explain the market, size risk, and decide whether a recommendation deserves action.

Important outputs include:

- `strategy_alerts`: current signals and alert conditions for configured approaches.
- `momentum_state`: vol-adjusted momentum state by ticker.
- `confirmation_matrix`: bullish/neutral/bearish confirmation evidence.
- `market_health`: breadth, trend, credit, volatility, and risk-health summaries.
- `scenario_outlook`, `scenario_drivers`, `scenario_lattice`: future-state scenario probabilities and rankings.
- `macro_signals` and `macro_category_summary`: macro pressure groups and signal states.
- `regime_pulse_cycles`, `regime_pulse_assets`, `growth_inflation_map`: macro-weather and asset-regime views.
- `positioning_summary` and `positioning_crowding`: proxy positioning/crowding diagnostics.
- `regime_instability` and `regime_instability_components`: watch-only transition-risk diagnostic.
- `risk_status`, `risk_score`, `risk_summary`: human-readable current risk posture.

Current-state outputs should be lag-safe where they affect backtests or trade decisions. Dashboard-only diagnostics can be same-day observations, but their role should be clear.

## Future-State And Scenario Modeling

The system predicts future states, not exact prices. Scenario probabilities are used to shape risk budgets and explanations. They are treated as calibrated probabilities only when a specific ML diagnostic proves calibration.

Scenario horizons include:

- `1w`
- `1m`
- `3m`
- `6m`

Risk buckets include concepts like risk-off, transition, fragile risk-on, broad risk-on, AI-led melt-up, inflation/energy shock, credit stress, and similar state families.

The scenario layer answers questions like:

- Is the next month more likely to be a clean risk-on setup or a transition zone?
- Are AI/growth signals strong but concentrated?
- Are macro, credit, breadth, or volatility signals arguing for smaller position size?
- What would need to change before re-risking is justified?

Important caution: scenario probabilities are useful only if they improve allocation outcomes after risk constraints and transaction costs. Do not promote a scenario model because its narrative sounds good.

## News And Event Layer

Sources:

- `src/trade_bot/research/news_monitor.py`
- `src/trade_bot/research/event_risk.py`
- `configs/news_sources.yaml`
- `configs/events.yaml`

The news/event layer is designed to reduce blind spots, not to chase headlines. It watches categories such as monetary policy, AI infrastructure, private credit, energy/oil, geopolitical risk, earnings/capex signals, macro releases, and sector-specific catalysts.

The implementation uses lightweight deterministic and metadata-based processing rather than LLM-first reading. News items receive category, urgency, phase, activation, risk-channel, candidate-proxy, and confirmation-window metadata. Activated news can create event-risk context, but event/news signals are intentionally constrained by decision-sanity logic.

Key principle: news-only de-risking should be capped unless market confirmation also deteriorates. Decision sanity prevents narrative/event pressure from overpowering market confirmation.

## Regime Instability Index

Source: `src/trade_bot/research/regime_instability.py`.

The regime instability index is a watch-only signal. It should appear in the dashboard and research outputs, but it must not directly alter trade sizing until an overlay is backtested.

The index blends:

- SPY large-move share over 21 and 63 trading days.
- SPY realized volatility over 21 and 63 trading days.
- Cross-sectional dispersion across liquid equity proxies.
- VIXY pressure as a tradable volatility proxy.
- Short-run versus long-run correlation shift.
- Breadth/concentration pressure from ratios such as RSP/SPY, QQQ/RSP, and SMH/SPY.
- Credit stress pressure from HYG/LQD.

Why it exists:

- Index trend can remain constructive while internals become unstable.
- Sustained high-volatility regimes can be dangerous even if a percentile-only signal looks normal relative to its own unstable history.
- A separate instability read helps identify market-transition zones for research into sizing, off-ramp, and re-entry overlays.

Do not treat this as a trade command until tests show that it improves net CAGR, drawdown, Calmar, re-entry timing, or churn behavior.

## Trade Decision Layer

Source: `src/trade_bot/research/trade_decision.py`.

The trade decision layer translates the primary strategy's latest target weights into an actionable recommendation after risk overlays.

Inputs:

- Primary strategy weights from the latest backtest.
- Current-state risk status and scenario lattice.
- Current event-risk and news context.
- Macro signal inclusion pressure.
- Portfolio risk constraints.
- Decision-sanity checks.

Important concepts:

- `base_position`: what the primary strategy wants before overlays.
- `pre_risk_target_position`: scenario/event/macro adjusted position.
- `pre_sanity_target_position`: position after portfolio-risk constraints.
- `scenario_adjusted_position`: final target after decision-sanity cap.
- `risk_budget_multiplier`: effective final risk budget implied by target weights.
- `position_plan`: ticker-level target, current, delta, and sizing bridge.
- `evidence`: human-readable justification rows.
- `scenario_links`: which future scenarios are influencing the recommendation.

Action labels are decision-support labels such as `DO_NOTHING`, `REVIEW_INCREASE_RISK`, `REVIEW_REDUCE_RISK`, and related variants. They should not be interpreted as automatic order instructions.

## Portfolio Risk Engine

Source: `src/trade_bot/portfolio/risk.py`.

The risk engine exists because raw momentum output is not enough. It evaluates the candidate portfolio against constraints and risk diagnostics before the dashboard presents target weights.

Risk concepts include:

- Factor/beta exposures.
- Equity beta and AI beta.
- Concentration and max-single-asset limits.
- Correlation regime shifts.
- Expected shortfall and tail risk.
- Stress tests and scenario-weighted stress.
- Marginal risk contribution.
- Defensive minimums and risk-asset multipliers.
- Turnover caps and sizing adjustments.

The risk engine is a guardrail layer. It should be conservative about permitting dangerous concentration, but it should not permanently suppress risk without evidence. The project owner explicitly wants high enough CAGR to matter, so future risk work should optimize the return/drawdown tradeoff rather than defaulting to cash.

## Decision Sanity Overlay

The decision-sanity overlay guards against large defensive moves based only on news/event pressure. Its core rule is that large cash/T-bill moves require confirmation from at least two market-confirmation groups such as credit, volatility, breadth, or trend.

Purpose:

- Avoid narrative-only overreaction.
- Preserve the ability to cut risk when market internals confirm deterioration.
- Keep the bot from reinforcing the project owner's natural bearish bias.
- Make recommendations more stable and operationally realistic.

Future changes to this layer must be backtested. Do not hand-tune it from vibes.

## Backtesting Standards

See also: `docs/math_model_audit.md`, `docs/iteration_protocol.md`, `docs/forward_testing_protocol.md`, and `docs/experiment_plan.md`.

The backtest engine should remain lag-safe:

- Strategy signals are shifted by `signal_lag_days` before returns are applied.
  The operating default is 2 for close-derived features: lag 1 implicitly uses
  the feature close as the return-interval boundary and is research-only.
- Volatility targeting and drawdown controls use shifted scaling.
- Transaction costs are charged through turnover.
- Long-only weights are clipped and normalized.
- Residual uninvested weight behaves like cash unless explicitly assigned.

Evaluation must not rely only on full-history CAGR. Required views include:

- Full-history metrics for context.
- Rolling 1-year, 3-year, and 5-year windows.
- Calendar-year metrics.
- Walk-forward train/test splits.
- Regime holdout and left-tail regime behavior.
- Entry-date sensitivity.
- Custom dashboard windows, including 30 days, 90 days, YTD, and custom start/end.
- Turnover, trade count, and practical operability.

The strongest strategy is not necessarily the one with the top CAGR. The desired candidates should preserve high upside while improving survivability, drawdown, Calmar, and human-operable churn.

## Experimentation And Curation

Research has intentionally explored many strategies. That is acceptable in the research layer, but the operational layer should be curated.

Experiment principles:

- Start broad enough to avoid missing strategy families.
- Go deep only where evidence is promising.
- Force diversity in the curated shelf so one family does not crowd out every other failure mode.
- Track parentage and rationale so the research path is explainable.
- Promote candidates based on validation, not only raw score.

The key dashboard area is:

```text
Research Lab -> Experiment Monitor
```

The upper aggregate area is for leaderboard, curated shelf, outcome frontier,
signal evidence, family map, taxable impact, validation/QC, and manifests. The
lower Candidate Details workbench is the canonical strategy drill-down. It should
include explanation, performance-over-time charts, allocation behavior, decision
timeline, factor attribution, mechanics, robustness diagnostics, and manifest/risk
notes. Avoid creating parallel strategy explorers unless there is a clear reason.

Promotion is not deployment. A promoted candidate can be suitable for paper monitoring, but live-money trust requires forward evidence.

Signal evidence is the pruning and expansion layer for monitored inputs. It asks
whether signal families have actually improved candidate behavior in historical
experiments instead of assuming every interesting macro, news, or narrative
diagnostic should influence allocation.

```bash
poetry run trade-bot run-signal-evidence --experiment-dir data/experiments_reset_v2
```

The command writes:

- `reports/signal_evidence/signal_family_evidence.csv`: family-level evidence tier,
  counts, win rates, marginal medians, recommendation, and data caveat.
- `reports/signal_evidence/signal_marginal_tests.csv`: child-versus-parent
  deltas for CAGR, drawdown, Calmar, re-entry score, turnover, and promotion
  score. These are the preferred ablation rows.
- `reports/signal_evidence/tagged_strategy_signal_families.csv`: normalized
  scorecards with signal-family tags for auditability.

Dashboard path:

```text
Research Lab -> Experiment Monitor -> Signal Evidence
```

Default interpretation:

- `validated_contributor`: acceptable as a model-search driver.
- `promising_mixed`: keep testing; inspect failure windows.
- `not_proven`: do not promote as a default trade driver.
- `context_only`: useful for human explanation, not allocation.
- `research_gap`: backlog item; do not imply the data is already sufficient.

Factor attribution is the canonical answer to "what actually moved the
needle?" for a selected strategy. The implementation lives in
`src/trade_bot/research/factor_attribution.py` and is surfaced in:

```text
Research Lab -> Experiment Monitor -> Candidate Details workbench -> Factor Attribution
```

It uses transparent ETF proxy factors rather than a proprietary factor database:
market beta, QQQ/growth, AI/semis, breadth, sector/cyclicals, rates/duration,
credit, commodities, volatility, and residual strategy behavior. The output
includes:

- factor betas and correlations;
- return contribution by factor;
- risk contribution by factor;
- residual return and residual volatility;
- recent factor-decay flags versus the full-history factor profile.

Use this before saying that two strategies are independent. If both are mostly
QQQ/growth or AI/semis beta with similar residual behavior, they should not both
consume scarce paper-monitoring slots without a clear operational reason.

Implementation shortfall is monitored separately from ideal strategy research.
The dashboard path is:

```text
Monitoring -> Shortfall / Drift
```

V1 compares journal recommendation tickets with logged paper/live executions,
flagging missed tickets and executions outside price or size bands. It does not
yet import broker-grade daily account valuation. When that data exists, use
`build_implementation_shortfall` to compare ideal strategy equity to actual
account equity over the same dates.

## ML And Bayesian Models

See also: `docs/ml_research_framework.md`.

ML is allowed and encouraged, but only as targeted evidence machinery. It should not become an unconstrained trade generator.

High-value ML seams:

- Future-state probability estimation.
- Re-risking and dip-repair classification.
- Left-tail off-ramp warnings.
- Feature and signal inclusion.
- Strategy-family routing.
- News/event impact scoring.
- Sector rotation.
- Churn and durability filters.
- Paper/live drift monitoring.

Model validation must include:

- Clearly defined target labels and horizons.
- Lag-safe features.
- Walk-forward and regime splits.
- Calibration checks for probability outputs.
- Economic utility after sizing, transaction costs, and constraints.
- Direct comparison to rule-based controls.

Simple models are acceptable, but robust classical ML and Bayesian variants should be tested where they can improve the operating system. Expensive or opaque models are not useful if they cannot be explained, rerun, and validated locally.

## Paper Monitoring And Journal Workflow

Sources:

- `src/trade_bot/trading/journal.py`
- `src/trade_bot/dashboard/forward_test.py`
- `src/trade_bot/dashboard/monitoring.py`
- `src/trade_bot/storage/warehouse.py`

The system distinguishes research backtests from forward paper monitoring.

Important objects:

- Recommendation tickets: generated suggestions with ticker, price band, size band, rationale, and status.
- Paper executions: manually logged entries representing what would have been traded.
- Live executions: future manual logs for real trades, if ever used.
- Monitoring windows: strategy/account/time windows for champion/challenger tracking.
- Daily valuations: paper/live position performance through time.

Paper monitoring exists to answer:

- Would the system have made reasonable decisions in real time?
- Did recommendations change too often?
- Were suggested moves clear enough for a human to execute?
- Did the live/paper path resemble the backtested behavior?
- Which champions/challengers deserve more attention?

Do not skip paper monitoring before live-money promotion.

## Dashboard Mental Model

The dashboard is not meant to expose every table at once. It should guide the user from conclusion to evidence.

Top-level flow:

1. Action Headline: whether this is a do-nothing, small-action, or critical-action day.
2. Operating Brief: execution checklist with sizing translation, scenario constraints, decision sanity, and bias checks.
3. Decision Brief: collapsed research context with supporting evidence, scenario bridge, and invalidation details.
4. Book-aware recommendation: visible default paper-book alignment, with raw position and execution rows kept inside the details expander.
5. Right-side Term Lookup: compact metric/tracker explanations sourced from the metric explainer registry. This is the quick-reference surface for terms that appear anywhere in the app.
6. Insight Workbench: deeper workbenches for Command Center, Risk & Scenarios, Simulation Lab, Research Lab, Monitoring, News & Macro, Performance, and Forward Test.

Simulation Lab is the dedicated forward-path planning workbench. Keep
deterministic wealth math, historical block-bootstrap sequence risk,
regime-conditioned forward simulation, and simulation interpretability there.
Research Lab should stay focused on empirical strategy evidence, aggregate
experiment comparisons, and individual candidate diagnostics.

Design standard:

- The dashboard should explain what to do, why, what changed, and what would invalidate the conclusion.
- Avoid duplicative sections that show the same cards with different labels.
- Keep the top-of-page operating surface focused on today's action, not broad research context.
- Keep research details available, but do not bury the operating decision.
- Keep definitions close to the user through hover help and the right-side Term Lookup, rather than repeating long glossary text inside each section.
- Keep one detailed workbench visible at a time. If a section becomes a duplicate of another section, consolidate it or make one an explicit drill-down.
- Dark mode must remain readable.

## Quality And Testing Standards

Before finishing meaningful backend changes, run at least:

```bash
poetry run ruff check src tests
poetry run pytest -q
```

For narrower changes, run targeted tests first, then full tests if feasible.

Test expectations:

- New formulas need direct tests.
- New dashboard helpers need app or rendering tests where practical.
- New storage tables need migration and round-trip tests.
- New strategy behavior needs backtest or allocation-contract tests.
- Any change to risk/trade-decision semantics should update docs and tests together.

The repository uses Ruff, Black-compatible line length, mypy settings, and pytest. Keep code Pythonic, typed where practical, and close to existing patterns.

## Documentation Standards

When changing system behavior, update the relevant doc:

- Formula or interpretation change: `docs/math_model_audit.md`.
- New ML seam or validation rule: `docs/ml_research_framework.md`.
- Research workflow or iteration rule: `docs/iteration_protocol.md` or `docs/experiment_plan.md`.
- Live/paper process change: `docs/forward_testing_protocol.md`.
- Local/work boundary change: this internal agent guide.
- Overall architecture or agent onboarding change: this file.

Documentation should explain what changed, why it exists, how it is used, and what it must not be used for.

## Common Failure Modes To Avoid

1. Treating full-history CAGR as proof.
   - Always inspect rolling windows, regimes, and entry-date sensitivity.

2. Letting the bot become permanently bearish.
   - Risk management is necessary, but high-CAGR strategies need controlled re-entry.

3. Chasing one magic historical period.
   - Prefer robust families and multiple validation cuts.

4. Building dashboard tables without conclusions.
   - Every top-level section should make interpretation easier.

5. Making large news-only allocation moves.
   - Require market confirmation unless a backtested overlay proves otherwise.

6. Adding new data without inclusion tests.
   - More signals are useful only if they improve decisions or explanations.

7. Creating too many monitored strategies.
   - Monitoring should focus on a curated champion/challenger set plus baselines.

8. Breaking local-only assumptions.
   - Do not silently introduce cloud dependencies, work infra, or external auth requirements.

9. Confusing paper tickets with submitted recommendations.
   - The system needs exact state, timing, prices, size ranges, and disposition tracking.

10. Letting defaults drift across files.
    - Use `DEFAULTS.py` for reusable defaults.

## Current Limitations And Open Research Areas

Known limitations:

- Scenario probabilities are useful research probabilities, but not all are empirically calibrated odds.
- Macro data still needs better release-lag and revision discipline for production-grade historical tests.
- News/event NLP is lightweight and should be empirically tested before receiving more authority.
- Some features rely on public proxies rather than institutional datasets.
- Paper-monitoring evidence is still young relative to the desired confidence level for live capital.
- The final champion/challenger set should remain small and curated.

High-value next research areas:

- Backtest regime-instability overlays for sizing, re-entry, and off-ramp behavior.
- Improve re-risking after drawdowns without catching falling knives.
- Test sector-rotation systems that are more nuanced than risk-on versus cash.
- Build strategy-family routing that favors robust families under current conditions.
- Improve probability calibration for future-state ML and Bayesian models.
- Add drift detection for paper/live monitoring.

## Future-Agent Checklist

When starting a new backend task, do this first:

1. Confirm the active repo path and git status.
2. Read the relevant docs before editing formulas, risk logic, storage, or dashboard surfaces.
3. Identify whether the change affects research only, paper monitoring, or live decision-support semantics.
4. Preserve long-only, human-reviewed, local-only assumptions unless explicitly told otherwise.
5. Check whether a default belongs in `DEFAULTS.py`.
6. Add tests for any new behavior.
7. Run Ruff and pytest.
8. Update docs if behavior or interpretation changed.
9. If the dashboard consumes the change, consider whether snapshots need backward-compatible guards.
10. Summarize whether the change affects trading recommendations or is watch-only/research-only.

The most important rule: do not make the system look more confident than the evidence warrants. The goal is high-return decision support with defensible risk controls, not a dashboard that rationalizes whatever the market just did.
