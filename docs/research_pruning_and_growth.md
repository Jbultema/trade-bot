# Research Pruning and Regrowth Protocol

Status: maintained research-policy note. Last reviewed: 2026-07-22.

This project is intentionally experimental, but the dashboard should not treat every experiment as equally alive. Historical artifacts remain auditable; pruning only changes the default research queue.

## Current Empirical Read

The canonical close-safe replay library is classified into research statuses;
the reset-era archive is retained only as the definition/audit source:

- `operational_candidate`: high-growth, tolerable drawdown, and still worth paper-monitoring or close inspection.
- `needs_iteration`: promising mechanism, but not ready to monitor without a tighter next experiment.
- `research_archive`: useful context or reference, not a current operating candidate.
- `pruned_dead_end`: failed risk/return, low-growth, or validation-failed experiment. Keep for audit, hide from default curation.
- `reference`: static or configured benchmark rows.

The working read after the ML, outcome-utility, launch, simulation, and
Cycle-Tracker expansions is sharper than the original reset-era read: broad
future-state ML can easily become too conservative, while explicit off-ramp and
re-entry systems have preserved high CAGR better. Strategy-specific drawdown ML
improved CAGR and Calmar slightly in some cases, but it is not a promoted
default driver for material max-drawdown reduction versus the strongest
high-growth re-entry families. Reactive classic drawdown-control hybrids were
worse and should not be promoted for the growth engine without a new reason.

The growth-constrained outcome lens changed the curation objective. The leading
runtime snapshot candidates are persisted in DuckDB as `snapshot_strategy_metrics`
and have recently clustered near the 20-22 percent historical CAGR and roughly
-18 to -22 percent max-drawdown area. These are operable latest-snapshot rows,
not the same source as archived `experiment_scorecard` rows, which may rank
differently. Use `poetry run trade-bot audit-strategy-sources` when reconciling
"best strategy" claims across runtime snapshots, experiment scorecards, rolling
windows, and docs. That is strong research evidence, not a live-return promise.
The correct comparison is not "highest CAGR wins"; it is whether the extra
terminal wealth justifies drawdown, turnover, concentration, execution, launch,
paper-monitoring, and validation risk. The hard pruning boundary for drawdown is
the configured growth hard limit, not a blanket -25 percent cutoff. The
broad/sector growth-frontier pass was useful evidence but did not dethrone the
existing high-CAGR re-entry and growth-guardrail families.

This section is empirical, not a permanent truth. Revisit it after major
experiment batches, material data changes, or sustained forward paper evidence.

## Pruning Rules

Rows are pruned from default views when any of these are true:

- explicit left-tail, regime-fragility, or walk-forward-fragility rejection;
- CAGR below 5%;
- weak combined return and risk-adjusted profile;
- max drawdown at or beyond the configured hard growth drawdown limit;
- reactive classic drawdown-control variants lose too much growth;
- future-state ML probes remain below practical return thresholds;
- weak walk-forward positive rate;
- left-tail regime loss is too large;
- outcome utility cannot justify the drawdown burden after validation penalties.

Pruned rows are not deleted. They remain available by selecting `All approaches` or filtering Research Lab leaderboards by `pruned_dead_end`.

## Default Operating Surface

The default dashboard and monitoring surface is intentionally narrower than the
research archive:

- Unsupported watchlist items are hidden from the main action layer and shown
  only as data gaps.
- Thin-proxy narrative diagnostics are research-only unless a marginal
  contribution test proves they improve CAGR, drawdown, re-entry, or churn.
- Default reference anchors are limited to SPY, QQQ, BIL/cash if configured,
  and the U.S. 60/40 policy benchmark. Other policy portfolios remain
  inspectable in explicit all-approach views.
- Low-CAGR defensive sleeves, failed ML routers, poor sector-rotation ML, and
  `pruned_dead_end` rows are suppressed from default candidate shelves.
- Narrative/news modules remain visible as "watch this" diagnostics unless
  ablation and forward-monitoring evidence promotes them into a model driver.

This pruning is display and monitoring governance, not deletion. The archive is
still the audit trail.

Signal families have their own evidence audit. Before promoting a new
monitor, source family, or narrative diagnostic into the default operating
surface, run:

```bash
poetry run trade-bot run-signal-evidence --experiment-dir data/experiments_close_safe_v22
```

Dashboard path:

```text
Research Lab -> Experiment Monitor -> Signal Evidence
```

Use the resulting labels as pruning guidance:

- `validated_contributor`: keep visible and continue iterating.
- `promising_mixed`: keep in research views, but require failure-case review.
- `not_proven`: hide from default decision surfaces unless manually selected.
- `context_only`: explanation layer only; do not let it drive allocation.
- `research_gap`: backlog only until data and paired ablations exist.

## Regrowth Rules

New work should branch from mechanisms that passed at least one of these tests:

- preserves double-digit CAGR while improving some risk metric;
- improves reentry or left-tail behavior without becoming sticky defensive;
- creates genuinely different exposure from the AI escape family;
- improves confidence diagnostics, not just full-history CAGR.

Avoid broadening through more one-off variants that only differ by tiny thresholds. Prefer experiments that test a new mechanism, a clean ablation, or a dashboard/monitoring decision the user must actually make.

## Current Growth Direction

The most fruitful direction remains a high-CAGR operating system with bounded risk overlays. The next useful experiments should focus on:

- strategy-specific failure-mode labels rather than generic market-state labels;
- reentry after risk-off periods;
- concentrated AI/growth exposure limits that preserve upside;
- sector/factor alternatives that are not just lower-return defensive substitutes;
- calibrated confidence intervals and live drift checks for paper-monitoring;
- account-aware and taxable-account evaluation as the tax-lot simulator matures.

The bar is practical: a strategy that compounds at 3-5% is not useful for this project unless it is explicitly a defensive/reference sleeve. The operating systems should target high returns with drawdown mitigation, not cash-like returns with better optics.
