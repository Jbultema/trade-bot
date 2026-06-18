# Project Boundaries

This is a private personal research project. It should remain operationally separate from work systems.

## Operating Boundary

- Do not push this repository to GitHub by default.
- Do not use work data, work credentials, Databricks workspaces, or Hasbro-specific infrastructure.
- Keep all market data, backtest results, prompts, and dashboard outputs local unless explicitly changed later.
- Treat account-specific trading choices as personal Vanguard IRA decisions, not work artifacts.

## Dependency Boundary

The default package should remain standalone and open-market-data oriented. Private forecasting libraries can be useful, but they should not become required dependencies.

If `lib-aim-timeseries` functionality is used:

- add it behind an optional adapter layer
- keep the core strategy/backtest interfaces independent
- make the dependency opt-in through local environment configuration
- avoid copying proprietary code into this repository
- keep tests runnable without that dependency installed

This preserves the project as a clean "can this be done" research system while allowing reuse of personally maintained forecasting patterns where they clearly improve the work.

## Technical Bias

The system should be Python-first and reproducible:

- structured numerical signals drive portfolio decisions
- forecasting, risk, and backtesting logic must be auditable
- LLM-derived signals are supporting features for unstructured context
- every LLM-derived signal should retain source, timestamp, prompt/version, score, and explanation
- risk controls and sizing logic can override forecast/alpha signals
