# Trade Bot

Local swing/momentum trading research and decision-support system.

This project is intentionally not an automated execution bot. It is designed to:

- collect daily market data into a local cache
- run reproducible long-only backtests
- compare strategies against broad-market benchmarks
- produce trade suggestions with next-session execution assumptions
- show risk, regime, position sizing, and drawdown context for human review
- translate major policy/geopolitical news into auditable event-risk scenarios

Core constraints:

- human-triggered trades only
- no shorting
- no derivatives as the default operating mode
- minimum practical holding period is measured in human trading days, not minutes
- Vanguard rollover IRA accessibility matters for the tradable universe
- capital preservation and left-tail control are first-class objectives

## Quick Start

```bash
poetry install
poetry run trade-bot fetch-prices --config configs/baseline.yaml
poetry run trade-bot run-baselines --config configs/baseline.yaml --events configs/events.yaml
poetry run trade-bot run-experiment-iteration --config configs/baseline.yaml --iteration 1
poetry run streamlit run src/trade_bot/dashboard/app.py
```

The generated HTML report is written to `reports/baseline_report.html`.

## First Research Surface

The first implemented strategies are deliberately simple and auditable:

- buy and hold benchmarks
- absolute momentum with moving-average filters
- relative momentum rotation
- dual momentum rotation
- volatility-targeted sizing
- drawdown kill-switch overlays

Signals are computed from information available after the market close and are shifted one trading session before portfolio returns are calculated. This keeps the default backtest aligned with human execution.

## Iterative Research Loop

Strategy research runs in bounded batches. One iteration tests 3-10 candidates, scores them, and
marks each as promote, evolve, or reject. The intended end state is 1-3 operational systems, not a
dashboard full of live strategies. The dashboard includes an experiment monitor for the full test
history. See [docs/iteration_protocol.md](docs/iteration_protocol.md) and
[docs/creative_strategy_backlog.md](docs/creative_strategy_backlog.md).

## Current Dashboard Surface

The dashboard now includes both historical and current-state views:

- current risk status
- strategy-level trading alerts
- scenario sketch
- granular scenario lattice across 1w, 1m, 3m, and 6m horizons
- scenario driver table covering breadth, credit, AI concentration, liquidity, energy, duration, and drawdown state
- event-risk monitor and historical analog windows
- risk confirmation matrix
- VAMS-style signal table
- data quality coverage
- rolling-window and calendar-year backtest diagnostics

## Data Caveats

The initial data path uses Yahoo Finance through `yfinance` for fast local research. That is good enough to bootstrap strategy testing, but it is not a final institutional-grade historical data source. The code is structured so higher-quality sources can replace or supplement it.

## Formula Audit

The locked math, formula definitions, model semantics, and drift-control rules live in [docs/math_model_audit.md](docs/math_model_audit.md). Any change to core backtest, risk, scenario, trade-decision, or experiment formulas should update that document and the corresponding formula-contract tests.

## Project Boundary

This is a private personal research project. It should stay separate from work infrastructure and data. See [docs/project_boundaries.md](docs/project_boundaries.md) for the dependency and operating rules, including how any optional `lib-aim-timeseries` reuse should be isolated behind adapters.

## Not Investment Advice

This system is research tooling and decision support. It does not guarantee returns, does not remove market risk, and should not be treated as investment advice.
