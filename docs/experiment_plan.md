# Experiment And Operating Roadmap

Status: maintained roadmap and pruning guardrail. Last reviewed: 2026-07-17.

This document is the current roadmap for Trade Bot research and operating
hardening. It replaces the old Phase 1 / Phase 2 boundary plan. The original
planning note remains archived at `docs/archive/experiment_plan_2026-06-17.md`.

The important maintenance rule is simple: do not let dated build plans remain
visible as if they describe current system behavior. When a plan is executed,
either update this roadmap to the new operating reality or archive the dated
plan.

## Current Operating Platform

Trade Bot is now a V2-first local research and monitoring system. The dashboard
loads saved snapshots, DuckDB warehouse tables, and persisted research artifacts
before hydrating expensive workbench views. V1 is retained only as an archived
fallback for comparison/debugging.

Current operating capabilities include:

- local market, macro, news/event, and snapshot ingestion;
- fast V2 dashboard pages for Today, Macro, Risk, Forward Test, Research,
  Performance, Launch, Simulation, and Monitoring;
- current-state risk, scenario, confirmation, instability, macro-driver, and
  decision-sanity diagnostics;
- long-only baseline, momentum, sector-rotation, dip-reentry, risk-cycle,
  high-growth, and reference-portfolio strategy tests;
- scenario-aware sizing, expected shortfall, stress loss, beta/factor checks,
  concentration checks, and defensive floors;
- Research Lab outcome frontier, candidate deep dive, PBO/backtest-QC,
  leadership dependence, false-alarm judgement, and walk-forward router
  diagnostics;
- Simulation Lab deterministic, bootstrap, regime-conditioned, duration-aware,
  covariate-matched, factor-proxy, and rolling-origin validation views;
- Scenario / Phase Frontier and Cycle Tracker with 0M nowcast, path-constrained
  horizon frontiers, crisis playback, historical reliability, and conditional
  winner shelves;
- Launch Lab entry testing, ramp protocols, aggregate launch reads, and
  Experiment Operator trial contracts;
- paper/live Forward Test tickets, execution journal, book alignment, and
  monitoring windows with explicit start dates;
- estimated taxable-account research support for paper monitoring and planning.

The platform remains paper-first and human-reviewed. A dashboard recommendation
is a decision-support object, not an order.

## Active Priorities

The useful next work is not unlimited strategy proliferation. Favor work that
improves confidence, operability, and interpretability.

1. **Operate and observe the strongest candidates.** Keep the default monitored
   set small. Use paper/live experiments, start-date cohorts, and book
   alignment to see whether the best candidates still behave like their
   historical evidence.
2. **Harden source-of-truth boundaries.** Keep strategy identity, runtime
   snapshot metrics, experiment scorecards, candidate manifests, tickets,
   executions, and warehouse mirrors consistent enough that all UI panes point
   at the same strategy and book state.
3. **Improve reliability labels before adding authority.** Cycle Tracker,
   Simulation Lab, Launch Lab, false-alarm judgement, and router diagnostics
   should explain confidence and sample limits before they influence operating
   posture.
4. **Keep macro and narrative inputs in the right lane.** They should explain
   and challenge the current read unless validated tests promote them into
   sizing authority.
5. **Continue UI simplification.** Prefer fast summary pages, explicit full
   workbench loads, hover help, and high-value visuals over large raw tables at
   the top of each page.

## Parked Or Higher-Bar Work

These items are not rejected, but they need stronger justification than routine
dashboard work:

- broker-linked automatic trading;
- options, leverage, shorting, or intraday execution;
- proprietary macro data replication;
- vintage macro/news reconstruction beyond point-in-time safe price-derived
  backfills;
- direct stock-price forecasting as an allocation authority;
- broad new strategy sweeps that do not improve the current champion/challenger
  evidence set;
- live taxable-account trading logic without broker lot reconciliation and tax
  review.

## Validation Rules

- Use next-session execution assumptions by default.
- Keep locked benchmark references: SPY, QQQ, VTI, BIL, and selected reference
  portfolios from `configs/baseline.yaml`.
- Report turnover and transaction costs for every strategy.
- Report full-history, calendar-year, rolling-window, walk-forward, regime, and
  custom-window performance. A single 2005-to-present score is never enough.
- Treat thresholds as policy-constrained hyperparameters, not values chosen only
  because one backtest won.
- Track strategy behavior through market transitions, drawdowns, off-ramps, and
  re-entry periods.
- Promote candidate operating systems only after reviewing allocation history,
  drawdown behavior, robustness diagnostics, and forward paper-monitoring
  readiness.
- Label account semantics explicitly. Base strategy results remain pre-tax /
  IRA-like unless fields are explicitly labeled as estimated taxable outputs.
- For cycle and scenario tools, keep the point-in-time boundary explicit:
  historical origins may use only data available through that origin, and
  forward windows start after the origin.

## Related Docs

- `docs/whitepaper.md`: canonical system narrative.
- `docs/technical_explainer.md`: implementation architecture and model
  semantics.
- `docs/cycle_tracker_design.md`: Scenario / Phase Frontier and Cycle Tracker
  design contract.
- `docs/iteration_protocol.md`: experiment loop, promotion rules, artifact
  roots, and curation rules.
- `docs/math_model_audit.md`: locked formulas and model semantics.
- `docs/ml_research_framework.md`: ML/Bayesian seams and validation gates.
- `docs/forward_testing_protocol.md`: paper/live monitoring and ticket workflow.
- `docs/taxable_account_framework.md`: estimated taxable-account and after-tax
  research semantics.
- `docs/research_pruning_and_growth.md`: pruning rules and high-value regrowth
  direction.
