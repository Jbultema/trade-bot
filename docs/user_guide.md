# Trade Bot User Guide

Status: canonical user-facing guide. Last reviewed: 2026-07-05.

This guide explains how to use Trade Bot as a full workflow: daily operation,
strategy research, paper monitoring, execution journaling, taxable review, and
eventual controlled live tracking. It assumes the app has already been installed;
see `docs/setup_guide.md` for first-time setup.

## Mental Model

Trade Bot has four operating layers:

1. **Daily decision layer**: what the system thinks today and whether action is
   needed.
2. **Research layer**: which strategies worked historically, why, and under what
   regimes.
3. **Monitoring layer**: which strategies are being paper-tracked from a real
   start date.
4. **Journal layer**: which recommendations were reviewed, locked, skipped,
   paper-traded, or live-traded.

The safest way to use the app is to move from left to right:

```text
Research evidence -> paper monitoring -> recommendation ticket -> execution log -> review
```

Do not skip paper monitoring. Do not treat a backtest as live proof.

## Dashboard Controls

Most normal operation can be done from the dashboard UI. The left sidebar is the
runtime control panel; the main page is the decision and research surface; the
right reference rail is the glossary and quick-help surface.

### Left Sidebar

Use the left sidebar for regular refresh and runtime controls.

| Control | Use It For | Notes |
| --- | --- | --- |
| Run source: **Latest snapshot (fast)** | Normal dashboard use | Reads the latest completed snapshot and keeps the app snappy. |
| Run source: **Live pipeline** | Debugging or ad hoc local checks | Recomputes in-session and can be slower. Do not use this as the default daily workflow. |
| **Run Full Daily Update** | Normal one-click refresh | Queues the full daily stack: market data, macro data, news, snapshot, warehouse migration, and paper valuation. |
| Local paths | Changing config/run-store inputs | Most users should leave these alone after setup. |
| Advanced refresh options | Partial snapshot rebuilds | Use only when intentionally refreshing one input layer or debugging. |
| Update jobs | Checking background job status | After a queued job finishes, refresh the browser to load the new snapshot. |
| Show quick reference rail | Showing/hiding the right help rail | Turn it off when you need more horizontal chart/table space. |

### Right Reference Rail

The right rail is a searchable quick-reference system for:

- metrics,
- ticket fields,
- workflow terms,
- ticker symbols,
- dashboard concepts.

Use it when a card, table column, or chart label is unfamiliar. It explains what
the term means, how it is calculated, how to read it, and what can go wrong.

### Main Page

The main page is organized as:

1. Header and freshness strip.
2. Action Headline.
3. Operating Brief.
4. Book Alignment.
5. Insight Workbench.

The first four sections answer "what do I need to do today?" The Insight
Workbench answers "why, what evidence supports it, and what should I inspect?"

## The Daily Operating Workflow

Use this workflow on market days or whenever you want a fresh read.

### Step 1: Run The Daily Update

Preferred dashboard workflow:

1. Open the dashboard.
2. In the left sidebar, keep **Run source** set to **Latest snapshot (fast)**.
3. Click **Run Full Daily Update**.
4. Open **Update jobs** and wait until the queued job completes.
5. Refresh the browser so the app loads the new snapshot.

The sidebar button refreshes market data, macro data, news, the snapshot,
warehouse tables, and active paper-monitoring valuations.

CLI equivalent from the repo root:

```bash
poetry run trade-bot run-daily-update
```

Use the CLI when you are automating the workflow, running the app headless, or
debugging outside Streamlit.

Use cached inputs only for a faster local smoke check:

```bash
poetry run trade-bot run-daily-update --cached-data --cached-macro --cached-news
```

### Step 2: Open Or Refresh The Dashboard

If the dashboard is not already running, start it:

```bash
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8501
```

If port 8501 is busy:

```bash
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8502
```

If the dashboard is already open and the sidebar job has completed, refresh the
browser tab.

### Step 3: Confirm Freshness

At the top of the app, read the latest update strip. Confirm:

- snapshot timestamp,
- market date,
- risk state,
- whether the loaded view is a latest snapshot or live pipeline.

If the timestamp is stale, use the sidebar **Run Full Daily Update** button or
run the CLI command again.

### Step 4: Read The Action Headline

The Action Headline is the first operating answer. It should tell you whether
today is:

- no material action,
- small action,
- review/reduce risk,
- critical action.

Do not immediately trade from the headline. It is an executive summary.

### Step 5: Read The Operating Brief

The Operating Brief converts the recommendation into a practical checklist:

- sizing translation,
- scenario constraints,
- decision sanity,
- bias checks,
- execution caveats.

If the brief says no material change, stop unless you are intentionally reviewing
research or monitoring.

### Step 6: Check Book Alignment

Book Alignment compares the latest target against the locally logged paper or
live book.

Common outcomes:

- **Aligned / Do Nothing**: the latest target is already reflected closely enough.
- **Small Drift / Small Rebalance**: the book differs, but not by enough to force
  a major action.
- **Material Drift / Review**: the book is meaningfully away from target and may
  need a ticket.

If Book Alignment says aligned, do not create duplicate tickets just because the
Action Headline still says the current posture is defensive or risk-reduced.

### Step 7: Drill Only Where Needed

Use the Insight Workbench when you need detail:

| Question | Workbench |
| --- | --- |
| What exactly is the target posture? | Command Center |
| Why is risk being reduced or increased? | Risk & Scenarios |
| What future outcome range should I expect? | Simulation Lab |
| Which strategy should be trusted? | Research Lab |
| Are paper strategies behaving? | Monitoring |
| What news or macro inputs are active? | News & Macro |
| How did performance behave in a selected window? | Performance |
| What ticket or execution should be logged? | Forward Test |

## Dashboard Sections

### Operating Overview

Use this to decide what to do today. It intentionally avoids dense research
tables.

Read:

1. latest update strip,
2. Action Headline,
3. Operating Brief,
4. Decision Brief if needed,
5. Book Alignment.

### Command Center

Use this to inspect the current recommended posture and trade-decision bridge.
This section is most useful when the headline says action may be needed.

Look for:

- target weights,
- delta weights,
- add/reduce/hold action by ticker,
- scenario-adjusted posture,
- risk-engine adjustment reason.

### Risk & Scenarios

Use this to understand why sizing changed.

Look for:

- portfolio risk multiplier,
- expected shortfall,
- max stress loss,
- beta/factor exposure,
- scenario probabilities,
- confirmation matrix,
- regime instability,
- risk constraints.

### Simulation Lab

Use this to inspect the forward-looking planning distribution for a selected
strategy. Simulation Lab is not a replacement for Research Lab evidence; it is
the place to understand how deterministic CAGR math, historical sequence risk,
and current scenario probabilities change the expected range of outcomes.

Recommended flow:

1. Future-State Map: read the current scenario probabilities and how they map
   into broad simulation buckets.
2. Strategy Simulations: choose a strategy and compare deterministic 15-year
   wealth, historical bootstrap range, and regime-conditioned forward range.
3. Interpretability: inspect the model ladder, scenario bridge, historical
   regime-return library, and simulated regime mix.

Simulation Lab uses the configured accumulation assumptions from
`src/trade_bot/DEFAULTS.py`: starting account value, annual contribution,
monthly contribution timing, planning horizon, soft/hard drawdown bands,
bootstrap path count, and regime-forward simulation settings. Treat the outputs
as planning distributions and risk ranges, not point forecasts or trade
instructions.

### Research Lab

Use this before promoting any strategy into paper monitoring.

Research Lab is split into two layers:

- **Aggregate Insights Across Experiments**: compare strategies and families.
- **Candidate Details**: inspect one strategy deeply.

Recommended flow:

1. Leaderboard: find high-scoring candidates.
2. Curated Shelf: see candidates chosen for diversity and operability.
3. Outcome Frontier: compare CAGR versus drawdown and projected terminal wealth.
4. Family Map: understand whether many strategies are the same bet.
5. Signal Evidence: see which signal families helped historically.
6. Candidate Details: inspect one strategy before monitoring.

Outcome Frontier uses the configured accumulation assumptions from
`src/trade_bot/DEFAULTS.py`: starting account value, annual contribution,
contribution timing, planning horizon, and soft/hard drawdown bands. The default
contribution timing splits the annual total into monthly period-end deposits.
The headline 15-year wealth metric is deterministic CAGR planning math. Use
Simulation Lab when you need sequence-aware simulation, current-scenario
simulation, path-risk ranges, or interpretability for a selected strategy.

### Monitoring

Use this to evaluate forward paper evidence.

Look for:

- active windows,
- champion/challenger/reference roles,
- valuation date,
- cumulative return,
- benchmark cumulative return,
- excess return,
- drawdown,
- warehouse health.

### News & Macro

Use this to understand the current context and possible blind spots.

Important distinction:

- allocation drivers can affect tested model/risk layers,
- validated context helps interpretation,
- explainer-only items should not drive trades,
- unsupported watchlist items are reminders, not signals.

### Performance

Use this for selected-window performance review. Rebase growth of $1 to custom
windows when asking "what if I started here?"

### Forward Test

Use this to lock recommendations and log executions.

Use Forward Test when:

- the daily read says action is warranted,
- you want to paper-trade the current recommendation,
- you need to log a live execution,
- you need an audit trail for price, size, and timing.

## Adding A New Strategy

There are two paths: config-level baseline strategies and research-generated
experiment strategies.

### Path A: Add A Baseline Strategy

Use this for a simple strategy that should always appear in the baseline run.

1. Edit `configs/baseline.yaml`.
2. Add the strategy under `strategies`.
3. Keep tickers in the configured universe.
4. Run:

```bash
poetry run trade-bot run-baselines
poetry run pytest tests/test_config.py tests/test_momentum.py -q
```

5. Open the dashboard and confirm the strategy appears where expected.

### Path B: Add A Research Strategy

Use this for experimental strategy families, overlays, or variants.

1. Add the candidate generator or family logic in
   `src/trade_bot/research/experiments.py` or the appropriate research module.
2. Keep default thresholds in `src/trade_bot/DEFAULTS.py`.
3. Run a targeted iteration:

```bash
poetry run trade-bot run-experiment-iteration --config configs/baseline.yaml --iteration 161 --output-dir data/experiments_reset_v2
```

4. Refresh signal evidence if the strategy tests a signal family:

```bash
poetry run trade-bot run-signal-evidence --experiment-dir data/experiments_reset_v2
```

5. Migrate the warehouse:

```bash
poetry run trade-bot migrate-warehouse
```

6. Inspect Research Lab before starting paper monitoring.

## Testing A New Strategy

Do not judge a strategy by one backtest metric. Use this checklist:

- Compare to SPY, QQQ, 60/40, and BIL/cash where relevant.
- Check CAGR and terminal wealth.
- Check max drawdown and recovery needed.
- Check Ulcer Index.
- Check worst rolling 1Y and 3Y outcomes.
- Check walk-forward positive rate.
- Check worst walk-forward CAGR.
- Check left-tail regime return.
- Check factor attribution.
- Check turnover and action cadence.
- Check taxable impact if a taxable account is relevant.
- Check whether the strategy is reconstructable for monitoring.

If a strategy is strong historically but not reconstructable, treat it as a
research idea until runtime support is added.

## Promoting A Strategy To Paper Monitoring

### From The Dashboard

1. Go to **Research Lab**.
2. Use Leaderboard, Curated Shelf, Outcome Frontier, and Candidate Details to
   pick a candidate.
3. Go to **Monitoring -> Monitoring Controls**.
4. Choose the candidate set.
5. Select the strategy.
6. Choose role:
   - `champion` for the main candidate,
   - `challenger` for alternatives,
   - `reference` for baselines.
7. Set mode to `paper`.
8. Set account label.
9. Set capital base.
10. Click the start/update button.
11. After the next full daily update, confirm the window receives a valuation
    row. The sidebar **Run Full Daily Update** button includes paper valuation.

### From The CLI

Seed top operational candidates:

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --mode paper --account default_paper_account --capital-base 10000 --top-n 5 --start-date YYYY-MM-DD
poetry run trade-bot run-paper-valuation
```

Monitor one strategy:

```bash
poetry run trade-bot monitor-strategy STRATEGY_NAME --role challenger --mode paper --account default_paper_account --capital-base 10000 --start-date YYYY-MM-DD
```

Make it the only champion for that account:

```bash
poetry run trade-bot monitor-strategy STRATEGY_NAME --role champion --mode paper --account default_paper_account --capital-base 10000 --start-date YYYY-MM-DD --demote-other-champions
```

## Daily Paper Monitoring

After the daily update:

1. Open Monitoring.
2. Confirm active windows were valued today.
3. Compare champion, challengers, and references.
4. Check excess return and drawdown.
5. Review whether any strategy is stale, lagging, or drifting.
6. Do not promote or demote on one noisy day unless the original thesis is broken.

The normal path is the sidebar **Run Full Daily Update** button. Useful CLI
commands when you want to inspect state outside the app:

```bash
poetry run trade-bot list-monitoring-windows
poetry run trade-bot list-champion-challenger
```

## Locking And Logging Paper Trades

Use this when the current recommendation should be paper-traded.

1. Read Action Headline, Operating Brief, and Book Alignment.
2. Go to Forward Test.
3. Review recommendation tickets.
4. Lock the relevant ticket.
5. Execute the paper trade in the Forward Test form using realistic price and
   size assumptions.
6. Save the execution.
7. Re-check Book Alignment.

Do not log trades that did not happen. Missed executions are part of the evidence.

## Logging Live Trades

Live tracking uses the same journal concept, but the standard is higher.

Before live use:

- paper-monitor the strategy,
- check implementation shortfall,
- confirm exact ticker availability,
- confirm order type outside the app,
- confirm tax/account implications,
- use small position sizes first,
- log exact execution time, price, fees, quantity, and notes.

The app does not place the trade. It records the decision and execution after
human action.

## Taxable Brokerage Workflow

Use Taxable Impact before paper-monitoring a strategy in a taxable account.

1. Go to Research Lab.
2. Open Taxable Impact.
3. Compare pre-tax and after-tax CAGR.
4. Check tax drag in basis points per year.
5. Check short-term gain share.
6. Check wash-sale warnings.
7. Inspect Candidate Details for the selected strategy.
8. Avoid high-turnover strategies unless the after-tax edge survives.

Forward Test can rebuild estimated tax lots from logged executions, but broker
records remain the source of truth.

## Reviewing News And Narrative

News should answer: "What context might matter today?" It should not answer:
"What trade should I make?" unless the signal has been validated and confirmed.

Use News & Macro to classify inputs:

- **allocation_driver**: tested enough to affect model/risk layers.
- **validated_context**: useful context, but not direct sizing.
- **explainer_only**: narrative context only.
- **unsupported**: watchlist or data gap.

If a narrative sounds compelling but cannot be measured or tested, keep it in the
news/context layer.

## Pruning And Cleanup Workflow

Prune when Research Lab becomes noisy.

Hide from default views:

- failed experiments,
- low-CAGR defensive sleeves,
- failed ML routers,
- redundant variants,
- unsupported watchlists,
- strategies that cannot be valued forward,
- pruned-dead-end rows.

Keep visible:

- active champions,
- active challengers,
- core references,
- curated top candidates,
- high-growth candidates with tolerable drawdown,
- strategies with distinct factor exposure.

## Suggested Weekly Review

Once per week:

1. Run the daily update.
2. Review Monitoring.
3. Review Outcome Frontier.
4. Review Candidate Details for the champion and top challengers.
5. Check factor attribution overlap.
6. Check Strategy Family Map for redundancy.
7. Check News & Macro driver rotation.
8. Decide whether any paper window should be paused, closed, promoted, or left
   unchanged.

## Suggested Monthly Review

Once per month:

1. Run ML diagnostics if you use ML artifacts:

```bash
poetry run trade-bot run-ml-diagnostics --config configs/baseline.yaml --profile standard
```

2. Run signal evidence if new experiments were added:

```bash
poetry run trade-bot run-signal-evidence --experiment-dir data/experiments_reset_v2
```

3. Review docs and assumptions.
4. Archive stale plans.
5. Prune default views if research noise has grown.
6. Reconfirm that monitored strategies remain operationally feasible.

## Escalation Rules

Consider pausing or reducing reliance on a strategy if:

- live/paper behavior diverges sharply from backtest behavior,
- factor attribution shows hidden concentration,
- drawdown exceeds tested expectations,
- turnover becomes operationally unreasonable,
- taxable drag destroys the edge,
- the thesis depends on unsupported narrative,
- data feeds break or become stale.

Consider promoting a strategy only if:

- historical and walk-forward evidence are strong,
- paper monitoring is consistent,
- the strategy is explainable,
- action cadence is tolerable,
- implementation shortfall is acceptable,
- it improves terminal wealth or risk-adjusted outcomes versus references.
