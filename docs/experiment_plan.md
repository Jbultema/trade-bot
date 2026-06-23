# Experiment Roadmap

Status: maintained roadmap. Last reviewed: 2026-06-22.

This document defines the project phase boundary and current research direction.
It replaces the original planning note now archived at
`docs/archive/experiment_plan_2026-06-17.md`.

## Phase Boundary

### Phase 1: Current-State Trade Bot

Phase 1 is the active operating system. It answers: given the latest local
prices, macro data, news/events, scenarios, strategy evidence, and portfolio
risk constraints, what long-only action should be considered now?

Current Phase 1 capabilities:

- local data ingestion and cached reproducibility
- snapshot-based dashboard loading for fast daily review
- current market regime, risk-state, scenario, macro, news/event, and decision
  sanity diagnostics
- long-only baseline, momentum, sector-rotation, dip-reentry, risk-cycle, and
  reference-portfolio strategy tests
- risk management, scenario-aware sizing, factor/stress checks, expected
  shortfall, and portfolio constraints
- rolling-window, calendar-year, walk-forward, regime, drawdown, and selected
  custom-window diagnostics
- Research Lab curation, strategy-family maps, allocation-history review, and
  experiment monitor views
- paper-monitoring windows, recommendation tickets, execution journal, and
  forward valuation
- targeted classical ML and Bayesian overlays for future-state probabilities,
  off-ramp/re-entry diagnostics, strategy drawdown guards, and research
  diagnostics

Phase 1 remains paper-first. A dashboard recommendation is a decision-support
object, not an order.

### Phase 2: Simulated-Future-Enabled Trade Bot

Phase 2 remains future work. It should start only after Phase 1 has enough
paper-monitoring history to make behavior drift, missed-trade reviews, and
forward execution quality measurable.

Phase 2 should answer: given the current state, which future states are
plausible, how do candidate strategies behave across those paths, and which
current action has the best risk-adjusted expected utility?

Candidate Phase 2 work:

- probabilistic scenario generation and transition models
- return, volatility, correlation, and drawdown path simulation
- stress-state injection for credit, inflation, AI concentration, liquidity,
  policy, and geopolitical shocks
- strategy-policy testing across simulated futures
- decision rules based on expected return, drawdown risk, regret, and
  survivability

## Current Research Direction

The useful next work is not more one-off strategy variants. Favor work that
improves confidence, operability, and risk/re-entry behavior:

- preserve high-CAGR candidates while limiting left-tail drawdown
- improve re-risking after defensive periods without catching falling knives
- test strategy-specific failure-mode labels, not only broad market-state labels
- strengthen sector and factor rotation as alternatives to simple risk-on/cash
  switches
- evaluate whether ML/Bayesian overlays improve sizing and transitions after
  costs, churn, and walk-forward validation
- keep the default monitored set small, with baselines and only a few
  champion/challenger systems
- add taxable-account evaluation as a parallel research mode, not as a silent
  change to existing pre-tax/IRA-like scorecards

## Validation Rules

- Use next-session execution assumptions by default.
- Keep locked benchmark references: SPY, QQQ, VTI, BIL, and selected reference
  portfolios from `configs/baseline.yaml`.
- Report turnover and transaction costs for every strategy.
- Report full-history, calendar-year, rolling-window, walk-forward, regime, and
  custom-window performance. A single 2005-to-present score is never enough.
- Treat thresholds as policy-constrained hyperparameters, not values chosen
  only because one backtest won.
- Track strategy behavior through market transitions, drawdowns, off-ramps, and
  re-entry periods.
- Promote candidate operating systems only after reviewing allocation history,
  drawdown behavior, robustness diagnostics, and forward paper-monitoring
  readiness.
- Label account semantics explicitly. Until the taxable simulator exists, all
  strategy results should be treated as pre-tax / IRA-like research outputs.

## Related Docs

- `docs/iteration_protocol.md`: experiment loop, promotion rules, artifact roots,
  and curation rules.
- `docs/math_model_audit.md`: locked formulas and model semantics.
- `docs/ml_research_framework.md`: ML/Bayesian seams and validation gates.
- `docs/forward_testing_protocol.md`: paper/live monitoring and ticket workflow.
- `docs/taxable_account_framework.md`: planned taxable-account and after-tax
  research semantics.
- `docs/research_pruning_and_growth.md`: current pruning rules and growth
  direction.
