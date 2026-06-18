# Trade Bot

Local swing/momentum trading research and decision-support system.

This project is intentionally not an automated execution bot. It collects data, runs backtests, summarizes current market context, suggests human-reviewed trades, and tracks paper/live decisions. You execute any real trades manually.

Core constraints:

- human-triggered trades only
- long-only stocks and ETFs by default
- no default derivatives or shorting
- practical holding periods are measured in trading days, not minutes
- supported-account accessibility matters for the tradable universe
- maximum-return research must stay constrained by left-tail risk and long-term capital-preservation goals

## Environment

This repo is pinned to Python 3.12.6 through `pyenv` and uses Poetry for dependency and environment management. Poetry is configured locally to keep the virtualenv in `.venv`.

```bash
pyenv install -s 3.12.6
pyenv local 3.12.6
poetry env use "$(pyenv which python)"
poetry install
```

Run project commands through Poetry so the correct interpreter and dependencies are always used.

## Quick Start

Run this from the repo root.

```bash
poetry run trade-bot fetch-prices --config configs/baseline.yaml
poetry run trade-bot build-snapshot --config configs/baseline.yaml --events configs/events.yaml --macro configs/macro_fred.yaml --news configs/news_sources.yaml
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD --top-n 25 --capital-base 10000
poetry run trade-bot run-paper-valuation
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8501
```

Use the latest snapshot market date for `YYYY-MM-DD`; check it with `poetry run trade-bot list-snapshots`.

Then open `http://localhost:8501`.

Most dashboard opens should use the sidebar default, `Latest snapshot (fast)`. Use `Live pipeline` only when you intentionally want the dashboard open to recompute the full pipeline.

## Daily Operating Loop

Use this loop when you are reviewing the bot for a new trade window.

1. Build or refresh a snapshot.

```bash
poetry run trade-bot build-snapshot --config configs/baseline.yaml --events configs/events.yaml --macro configs/macro_fred.yaml --news configs/news_sources.yaml
```

Add refresh flags only when needed:

```bash
poetry run trade-bot build-snapshot --refresh-data --refresh-macro --refresh-news
```

2. Update warehouse mirrors and paper valuations.

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot run-paper-valuation
```

3. Open or refresh the dashboard.

```bash
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8501
```

4. Read the dashboard from top to bottom:

- **Macro Minute**: one-minute market situation report.
- **Action Headline**: do-nothing, small-action, or critical-action state.
- **Operating Brief**: conclusion, recommended action, sizing translation, scenario incorporation, and bias check.
- **Decision Brief**: plain-English instructions for what to do next.
- **Selected section tabs**: deeper supporting evidence.

5. If there is a paper or live action, go to **Forward Test**, lock the recommendation set, then log execution details after you act. Do not rely on memory for price, time, quantity, or rationale.

## Dashboard Walkthrough

### Top-Level Readout

The top of the app is the operating surface. It is designed to answer three questions before you look at any tables:

- What kind of day is this: do nothing, small actions, or critical actions?
- What action is recommended and how large is the target-position change?
- Why did the system change posture: price/trend, macro, news, scenario probabilities, or portfolio-risk constraints?

The most important top-of-page cards are:

- **Macro Minute**: summarizes risk status, current news/event pressure, scenario map, and the risk-budget action.
- **Action Headline**: severity score and next action.
- **Operating Brief**: the instruction sheet. This is the first place to look for what the system thinks you should do.
- **Decision Brief**: a higher-level explanation of conclusion, evidence, and what would change the recommendation.
- **Metric Guide**: built-in explainers for metrics that are easy to misuse.

### Command Center

Use this for current-state operations. It shows the current risk state, trade decision, recommendation bridge, current positions, and key evidence. If the top-level brief says action is required, this section is where you inspect the immediate support.

Questions this section should answer:

- What is the current risk state?
- What is the target posture?
- Which tickers are being added, reduced, or held?
- Is the action driven by scenario probabilities, current risk engine constraints, or strategy signal changes?

### Risk & Scenarios

Use this when you need to understand the off-ramp. It shows portfolio-risk diagnostics, scenario lattice, factor exposure, stress tests, expected shortfall, and constraint output.

Questions this section should answer:

- What are the most likely forward regimes over 1w, 1m, 3m, and 6m?
- How much risk-off or transition probability is being assigned?
- Are factor exposure, beta, stress loss, or expected shortfall forcing lower risk?
- What would need to improve before adding risk back?

### Research Lab

Use this for strategy research, not for same-day execution. It contains the approach explorer, experiment monitor, category leaderboards, regime tests, walk-forward tests, candidate manifests, and signal-inclusion diagnostics.

Important distinction: a promoted experiment is not automatically live-operable. It means the idea deserves monitoring or implementation. A strategy becomes paper-operable only when it exists in the runtime pipeline and can be valued in snapshots.

### Monitoring

Use this for champion/challenger forward testing. It reads from the canonical DuckDB warehouse and shows active paper windows, the top 25 ranked experiment candidates, reference portfolios, valuation status, snapshot metrics, strategy registry rows, and warehouse health.

Open **Monitoring Controls** to start monitoring an experiment or change an active window. Pick a strategy, choose `champion`, `challenger`, or `reference`, set the mode/account label, and assign paper capital. Use separate account labels when the same strategy should be monitored as multiple sleeves or capital sizes. Leave `Only champion` unchecked if multiple active champions are intentional.

Current paper monitoring starts at the configured capital base. The first valuation row is intentionally `0.00%` return; subsequent rows compound from future snapshots. This avoids the mistake of treating full-history backtest growth as forward paper performance.

Questions this section should answer:

- Which system is the current champion?
- Which challengers are being paper-monitored?
- Are any reference policies beating or lagging tactical approaches?
- Are any forward windows ahead of benchmark, lagging, or in drawdown review?
- Is the warehouse seeded and healthy?

### News & Macro

Use this to inspect what the system is reading from news, event configs, macro series, and signal-inclusion tests. This is where you check whether sector-specific or geopolitical stories are being converted into event-risk context rather than ignored.

Questions this section should answer:

- Which news items are active?
- Are they treated as leading warnings, confirming signals, or lagging context?
- Which macro categories are pressuring risk?
- Are any data sources stale or missing?

### Performance

Use this for backtest and historical-window performance. The key feature is selected-window rebasing: choose 30 days, 90 days, YTD, 1 year, 3 years, 5 years, full history, or custom windows and inspect growth of `$1` and drawdown for that period.

Questions this section should answer:

- Did the approach work recently, not only over the full 2005-2026 sample?
- How did it behave around market transitions?
- Is the drawdown profile acceptable for the retirement-risk objective?

### Forward Test

Use this to lock recommendations and log paper/live executions. This is the audit trail for what the system recommended, what you did, when you did it, at what price, and why.

Default mode should be `paper`. Only use `live` after a strategy has earned trust through forward monitoring.

## Storage Model

The system intentionally keeps all storage local.

Primary locations:

- `data/cache/`: cached market, macro, and news inputs.
- `data/run_store/trade_bot.duckdb`: canonical DuckDB warehouse and snapshot metadata.
- `data/run_store/snapshots/`: pickled snapshot artifacts for fast dashboard cold starts.
- `data/trading_journal.sqlite`: local trade-journal source used by the Forward Test UI.
- `reports/experiments/`: experiment iteration CSVs and summaries.
- `reports/baseline_report.html`: static HTML report from baseline runs.

The warehouse command mirrors experiment CSVs and the SQLite journal into DuckDB:

```bash
poetry run trade-bot migrate-warehouse
```

Use this after new experiment runs, journal changes, or when you want the dashboard Monitoring section to reflect the latest local artifacts.

## Snapshot And Warehouse Commands

List recent snapshots:

```bash
poetry run trade-bot list-snapshots --limit 10
```

List background snapshot jobs:

```bash
poetry run trade-bot list-snapshot-jobs --limit 10
```

Migrate local artifacts into DuckDB:

```bash
poetry run trade-bot migrate-warehouse
```

Seed paper monitoring windows from the top ranked strategy registry entries. The default is 25 so the leading experiment candidates are visible in Monitoring; reference portfolio policies are also retained as comparison anchors. Snapshot-ready strategies receive daily paper valuations first.

```bash
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD --top-n 25 --capital-base 10000
```

Write the next daily paper valuation from the latest snapshot:

```bash
poetry run trade-bot run-paper-valuation
```

Show active monitoring windows:

```bash
poetry run trade-bot list-monitoring-windows
```

Show champion/challenger status:

```bash
poetry run trade-bot list-champion-challenger
```

## Research Loop

Strategy research runs in bounded batches. One iteration tests 3-10 candidates, scores them, and marks each as promote, evolve, or reject. Iteration 41 is reserved for reference portfolio policies such as 60/40, 80/20, Bogleheads-style global allocations, Permanent Portfolio, Golden Butterfly, and all-weather-style sizing so simple go-to portfolios stay visible beside tactical systems. Iterations 42-49 are active-trading probes run with `configs/active_trading.yaml`, which uses daily rebalancing, next-day signal lag, and higher transaction costs to test whether more responsive systems still earn their operational burden.

```bash
poetry run trade-bot run-experiment-iteration --config configs/baseline.yaml --iteration 41
poetry run trade-bot run-experiment-iteration --config configs/active_trading.yaml --iteration 42
poetry run trade-bot migrate-warehouse
```

The intended end state is 1-3 operational systems, not a dashboard full of live strategies. Research should go broad first, then deep:

- broad: many hypotheses, overlays, universes, and risk rules
- deep: walk-forward testing, regime holdouts, left-tail windows, overfit diagnostics, and forward paper monitoring
- operational: promote only strategies that can be explained, valued, monitored, and acted on with human latency

See [docs/iteration_protocol.md](docs/iteration_protocol.md), [docs/creative_strategy_backlog.md](docs/creative_strategy_backlog.md), and [docs/experiment_plan.md](docs/experiment_plan.md).

## Formula Audit

The locked math, formula definitions, model semantics, and drift-control rules live in [docs/math_model_audit.md](docs/math_model_audit.md). Any change to core backtest, risk, scenario, trade-decision, paper-valuation, or experiment formulas should update that document and the corresponding formula-contract tests.

Run validation before trusting changes:

```bash
poetry run black --check src tests
RUFF_CACHE_DIR=/private/tmp/trade-bot-ruff-cache poetry run ruff check src tests
poetry run pytest -p no:cacheprovider
```

## Sharing The Dashboard

Do not expose Streamlit directly to the public internet. For temporary demos with users, prefer a private network tool such as Tailscale or a controlled tunnel. Shared users should start as read-only viewers.

Before broader sharing, add or enable:

- lightweight app login
- read-only roles for viewers
- hidden or disabled state-changing controls for non-admin users
- redaction of exact account values and secrets
- clear `paper only / not investment advice` demo labeling

## Troubleshooting

If the dashboard is slow on cold start, check that a snapshot exists and the sidebar is set to `Latest snapshot (fast)`.

```bash
poetry run trade-bot list-snapshots
```

If the Monitoring section is empty, seed and value the warehouse.

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD
poetry run trade-bot run-paper-valuation
```

If CLI commands use the wrong Python, confirm pyenv and Poetry are aligned.

```bash
python --version
poetry run python --version
poetry env info
```

If Codex keeps requesting file approvals after a repo move, start a fresh Codex session rooted at the actual repo path.

## Project Boundary

This is a private local research project. It should stay separate from work infrastructure and data. See [docs/project_boundaries.md](docs/project_boundaries.md) for the dependency and operating rules, including how any optional `lib-aim-timeseries` reuse should be isolated behind adapters.

## Not Investment Advice

This system is research tooling and decision support. It does not guarantee returns, does not remove market risk, and should not be treated as investment advice.
