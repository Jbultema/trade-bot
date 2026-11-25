# Cleanup And Quality: November 25, 2026

Operational WIP and fixed-slate research are consolidated without changing the primary model.
Daily jobs retain stage-level diagnostics and durable snapshot IDs; writes are serialized and
snapshot retention preserves explicitly requested daily history. Alerts distinguish held and
next-target weights. Exploratory adaptation, orthogonal and public-signal studies remain
research-only; the leadership-transition candidate remains a shadow strategy.

This pass additionally protects incremental adjusted-price history from revision splicing,
checks terminal freshness per instrument, preserves caches on failed full-history refreshes,
and rejects invalid exporter targets. Package and lock files accompany bundle provenance.

The post-risk ledger is explicitly a **target replay**, not prospective execution evidence.
It does not prove capture-before-fill timing or implement the configured lag. All-cash targets
are retained; invalid weights are rejected rather than clipped into an apparently valid book.
Existing database rows were not silently rewritten or deleted as part of relabeling the UI.

Downloaded CFTC/CBOE/FINRA inputs in `research/public_signal_frontiers/raw/` remain local and
ignored. Historical reports and source inventories are preserved. They were not removed just
to obtain a clean checkout. `public_signal_frontiers.py` retains the official acquisition URLs.

New CI runs lint and the offline suite. The five-repo verifier and fuller quality assessment
live in Portfolio Lab under `scripts/verify_workspace.py` and
`docs/system_assessment_20261125.md`.

Important open findings: technology-universe selection/survivorship, incomplete lifetime trial
accounting, immature prospective evidence, and Cycle Tracker's missing direct
post-unwind-to-unwind transition despite a corresponding stress multiplier. Those model
changes need separate historical recurrence/false-alarm validation, not a cleanup-only tweak.

The first Linux CI run also exposed Mac-only publication font paths and a hidden local-store
dependency in the dashboard test. Publication rendering preserves existing Mac typography and
supports embedded DejaVu fonts on Linux (installed explicitly by CI). Read-only warehouse
queries on a fresh checkout return empty results without creating a database; dashboard tests
use an isolated store path. No tests are skipped to obtain a green portability result.
