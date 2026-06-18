# Trade Bot

> Local, human-reviewed trading research for long-only swing and momentum strategies. The system collects market, macro, news, and strategy evidence; builds snapshots; recommends paper/live actions for human review; and tracks forward results. It does not place trades automatically.

| Area | Current Role |
| --- | --- |
| Execution model | Human-triggered, long-only, next-session oriented |
| Trade universe | Stocks and ETFs that are practical to trade in supported accounts |
| Research target | Higher returns with explicit left-tail and capital-preservation constraints |
| Operating mode | Local-first Python, DuckDB, Streamlit, Poetry, pyenv |
| Live-money posture | Paper first; small live trades only after forward evidence earns trust |

## Operating Principles

- Human review is mandatory before any real trade.
- Long-only stocks and ETFs are the default. No default derivatives, shorting, or automated execution.
- Holding periods are measured in trading days and weeks, not minutes.
- Backtests must be judged across full history, recent windows, regime shifts, and walk-forward holdouts.
- Current-state recommendations and future-scenario research are related but separate systems.
- Risk management, position sizing, and off-ramp behavior matter as much as return forecasts.

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

Use the latest snapshot market date for `YYYY-MM-DD`; check it with:

```bash
poetry run trade-bot list-snapshots --limit 10
```

Then open `http://localhost:8501`.

Most dashboard opens should use the sidebar default, `Latest snapshot (fast)`. Use `Live pipeline` only when you intentionally want the dashboard open to recompute the full pipeline.

## Daily Operating Loop

| Step | Command or Dashboard Area | Purpose |
| --- | --- | --- |
| 1 | `build-snapshot` | Refresh the current market, macro, news, scenario, and strategy state. |
| 2 | `migrate-warehouse` | Mirror local artifacts into the canonical DuckDB warehouse. |
| 3 | `run-paper-valuation` | Update forward paper monitoring windows from the latest snapshot. |
| 4 | Dashboard top readout | Read Macro Minute, Action Headline, Operating Brief, and Decision Brief. |
| 5 | Monitoring | Check champion/challenger forward performance and paper windows. |
| 6 | Forward Test | Lock recommendations and log paper/live executions when action is warranted. |

Daily commands:

```bash
poetry run trade-bot build-snapshot --config configs/baseline.yaml --events configs/events.yaml --macro configs/macro_fred.yaml --news configs/news_sources.yaml
poetry run trade-bot migrate-warehouse
poetry run trade-bot run-paper-valuation
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8501
```

Add refresh flags only when needed:

```bash
poetry run trade-bot build-snapshot --refresh-data --refresh-macro --refresh-news
```

## Dashboard Map

The dashboard is intentionally organized from action to evidence. Start at the top, then drill only where needed.

| Section | Use It For | Primary Questions |
| --- | --- | --- |
| Top-Level Readout | One-screen operating posture | Is today do-nothing, small-action, or critical-action? What changed? |
| Command Center | Current-state trade decision | What is the target posture, and which tickers are affected? |
| Risk & Scenarios | Off-ramp and sizing discipline | Are factor risk, stress loss, scenarios, or expected shortfall forcing lower risk? |
| Research Lab | Strategy research and diagnostics | Which approaches worked, why, and across which windows/regimes? |
| Monitoring | Champion/challenger forward paper testing | Which monitored systems are ahead, lagging, or in drawdown review? |
| News & Macro | Narrative and macro source review | What news or macro pressure is active, stale, or missing? |
| Performance | Backtest and selected-window charts | Did the approach work recently and through transitions? |
| Forward Test | Recommendation and execution journal | What was recommended, what was done, at what price, and why? |

### Top-Level Readout

The top of the app is the operating surface. It is designed to answer three questions before you look at any tables:

- What kind of day is this: do nothing, small actions, or critical actions?
- What action is recommended and how large is the target-position change?
- Why did the system change posture: price/trend, macro, news, scenario probabilities, or portfolio-risk constraints?

Key cards:

- **Macro Minute**: current market situation, scenario pressure, new/recent changes, news/event pressure, and the practical action read-through.
- **Action Headline**: severity score, risk state, largest target change, active news, and open tickets.
- **Default Paper Book Alignment**: whether the paper book reflects the latest default target posture.
- **Operating Brief**: conclusion, recommended action, sizing translation, scenario incorporation, risk constraints, and bias check.
- **Decision Brief**: plain-English explanation of what to do next and what would change the recommendation.
- **Metric Guide**: hover/table explainers for metrics that are easy to misuse.

### Research Lab

Use this for strategy research, not same-day execution. It contains the experiment monitor, approach detail, performance-over-time views, allocation behavior, mechanics, robustness diagnostics, candidate manifests, and signal-inclusion tests.

Important distinction: a promoted experiment is not automatically live-operable. It means the idea deserves monitoring or implementation. A strategy becomes paper-operable only when it exists in the runtime pipeline and can be valued in snapshots.

### Monitoring

Use this for champion/challenger forward testing. It reads from the canonical DuckDB warehouse and shows active paper windows, ranked experiment candidates, reference portfolios, valuation status, snapshot metrics, strategy registry rows, and warehouse health.

Open **Monitoring Controls** to start monitoring an experiment or change an active window. Pick a strategy, choose `champion`, `challenger`, or `reference`, set the mode/account label, and assign paper capital. Use separate account labels when the same strategy should be monitored as multiple sleeves or capital sizes. Leave `Only champion` unchecked if multiple active champions are intentional.

Current paper monitoring starts at the configured capital base. The first valuation row is intentionally `0.00%` return; subsequent rows compound from future snapshots. This avoids treating full-history backtest growth as forward paper performance.

## Paper Monitoring Commands

Seed paper monitoring windows from the top ranked strategy registry entries. The default is 25 so leading candidates are visible in Monitoring; reference portfolio policies are retained as comparison anchors.

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD --top-n 25 --capital-base 10000
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

To manually add one strategy to paper monitoring:

```bash
poetry run trade-bot monitor-strategy STRATEGY_NAME --role challenger --mode paper --account default_paper_account --capital-base 10000 --start-date YYYY-MM-DD
```

## Storage Model

The system intentionally keeps all storage local.

| Location | Purpose |
| --- | --- |
| `data/cache/` | Cached market, macro, and news inputs. |
| `data/run_store/trade_bot.duckdb` | Canonical DuckDB warehouse and snapshot metadata. |
| `data/run_store/snapshots/` | Pickled snapshot artifacts for fast dashboard cold starts. |
| `data/trading_journal.sqlite` | Local trade-journal source used by the Forward Test UI. |
| `reports/experiments/` | Experiment iteration CSVs and summaries. |
| `reports/baseline_report.html` | Static HTML report from baseline runs. |

Keep `.env`, `.venv/`, `data/`, `reports/`, DuckDB files, parquet files, CSV exports, and local caches out of Git unless there is a deliberate reason to version a small fixture.

## Snapshot And Warehouse Commands

| Task | Command |
| --- | --- |
| List snapshots | `poetry run trade-bot list-snapshots --limit 10` |
| List background jobs | `poetry run trade-bot list-snapshot-jobs --limit 10` |
| Migrate warehouse | `poetry run trade-bot migrate-warehouse` |
| Seed monitoring | `poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD --top-n 25 --capital-base 10000` |
| Run paper valuation | `poetry run trade-bot run-paper-valuation` |
| List windows | `poetry run trade-bot list-monitoring-windows` |
| Champion/challenger | `poetry run trade-bot list-champion-challenger` |

## Research Loop

Strategy research runs in bounded batches. One iteration tests 3-10 candidates, scores them, and marks each as promote, evolve, or reject.

The intended end state is 1-3 operational systems, not a dashboard full of live strategies. Research should go broad first, then deep:

| Phase | Meaning |
| --- | --- |
| Broad | Many hypotheses, overlays, universes, and risk rules. |
| Deep | Walk-forward testing, regime holdouts, left-tail windows, overfit diagnostics, and forward paper monitoring. |
| Operational | Promote only systems that can be explained, valued, monitored, and acted on with human latency. |

Reference portfolio policies are included so simple allocations stay visible beside tactical systems. Active-trading probes use `configs/active_trading.yaml`, which applies daily rebalancing checks, next-day signal lag, and higher transaction-cost assumptions.

```bash
poetry run trade-bot run-experiment-iteration --config configs/baseline.yaml --iteration 41
poetry run trade-bot run-experiment-iteration --config configs/active_trading.yaml --iteration 42
poetry run trade-bot migrate-warehouse
```

See [docs/iteration_protocol.md](docs/iteration_protocol.md), [docs/creative_strategy_backlog.md](docs/creative_strategy_backlog.md), and [docs/experiment_plan.md](docs/experiment_plan.md).

## Formula Audit

The locked math, formula definitions, model semantics, and drift-control rules live in [docs/math_model_audit.md](docs/math_model_audit.md). Any change to core backtest, risk, scenario, trade-decision, paper-valuation, or experiment formulas should update that document and the corresponding formula-contract tests.

Run validation before trusting changes:

```bash
poetry run black --check src tests
poetry run ruff check --no-cache src tests
poetry run pytest -p no:cacheprovider
```

## Personal GitHub Workflow

This repo can use a personal GitHub SSH alias without clashing with work repositories. The local remote can point at a host alias such as `github-personal`, while other repos keep using their own `github.com` work setup.

```bash
git remote -v
git remote add personal git@github-personal:<personal-user>/<repo-name>.git
git push -u personal main
```

If the remote already exists, normal pushes are:

```bash
git push personal main
```

## Sharing The Dashboard

Do not expose Streamlit directly to the public internet. For temporary demos with users, prefer a private network tool such as Tailscale or a controlled tunnel. Shared users should start as read-only viewers.

Before broader sharing, add or enable:

- lightweight app login
- read-only roles for viewers
- hidden or disabled state-changing controls for non-admin users
- redaction of exact account values and secrets
- clear paper-only and not-investment-advice demo labeling

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
