# Research Pruning and Regrowth Protocol

Status: maintained research-policy note. Last reviewed: 2026-06-22.

This project is intentionally experimental, but the dashboard should not treat every experiment as equally alive. Historical artifacts remain auditable; pruning only changes the default research queue.

## Current Empirical Read

The active reset-era experiment archive is classified into research statuses:

- `operational_candidate`: high-growth, tolerable drawdown, and still worth paper-monitoring or close inspection.
- `needs_iteration`: promising mechanism, but not ready to monitor without a tighter next experiment.
- `research_archive`: useful context or reference, not a current operating candidate.
- `pruned_dead_end`: failed risk/return, low-growth, or validation-failed experiment. Keep for audit, hide from default curation.
- `reference`: static or configured benchmark rows.

The current working read after the ML and outcome-utility expansion is sharp:
broad future-state ML can easily become too conservative, while bounded ML and
re-entry overlays preserve high CAGR better. Strategy-specific drawdown ML
improved CAGR and Calmar slightly in the best cases, but it has not yet
materially reduced max drawdown versus the best raw high-CAGR AI escape controls.
Reactive classic drawdown-control hybrids were worse and should not be promoted
for the AI escape engine without a new reason.

The growth-constrained outcome lens changed the curation objective. A strategy
with roughly 14-15 percent CAGR and a tolerable -20 to -22 percent drawdown can
rank above an 11 percent CAGR strategy with a smaller drawdown when validation is
comparable. The hard pruning boundary for drawdown is now the growth hard limit,
not a blanket -25 percent cutoff. The broad/sector growth-frontier pass was
useful evidence but did not dethrone the existing high-CAGR re-entry and AI-escape
guardrail families.

This section is empirical, not a permanent truth. Revisit it after major
experiment batches, material data changes, or sustained forward paper evidence.

## Pruning Rules

Rows are pruned from default views when any of these are true:

- explicit left-tail, regime-fragility, or walk-forward-fragility rejection;
- CAGR below 5%;
- weak combined return and risk-adjusted profile;
- max drawdown at or beyond the hard growth drawdown limit, currently -30%;
- reactive classic drawdown-control variants lose too much growth;
- future-state ML probes remain below practical return thresholds;
- weak walk-forward positive rate;
- left-tail regime loss is too large;
- outcome utility cannot justify the drawdown burden after validation penalties.

Pruned rows are not deleted. They remain available by selecting `All approaches` or filtering Research Lab leaderboards by `pruned_dead_end`.

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
