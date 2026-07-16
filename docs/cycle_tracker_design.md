# Speculative Cycle Tracker Design

Status: current design and implementation contract.

The Speculative Cycle Tracker is a research/watch layer for markets where a
dominant theme may be moving through acceleration, fragility, unwind,
liquidation, bottoming, recovery, or post-unwind compounding. The initial use
case is the AI/growth leadership cycle, but the design is broader than "bubble
tracking." The same contract can be applied to credit, commodities, crypto,
non-US leadership, or any concentrated theme with enough price history.

## Purpose

The module answers four operating questions:

1. Which speculative-cycle phase does the current market most resemble?
2. Which phases are plausible across the 1m, 3m, 6m, and 1y horizons?
3. If a phase dominates, which assets or sleeves historically behaved best in
   similar prior states?
4. How reliable was this phase read historically when evaluated forward from
   prior-only information?

It does not attempt to call the exact top of a bubble, the exact bottom of a
drawdown, or the right day to rebalance. It is not an allocation override. It is
an explanatory and validation layer that should inform Research Lab,
Simulation Lab, Launch Lab, and human review.

## Research Basis

The design uses conservative pieces of the bubble/crisis literature:

- Kindleberger/Minsky crisis sequencing: displacement, boom/euphoria,
  distress, panic/liquidation, and recovery/revulsion. Trade Bot uses this as a
  phase taxonomy, not as a deterministic clock.
- Greenwood, Shleifer, and You, "Bubbles for Fama": sharp industry runups do
  not reliably predict poor average returns, but they do increase crash risk;
  volatility, turnover, issuance, and the path shape of the runup are useful
  attributes. Trade Bot uses observable proxies for acceleration, concentration,
  volatility pressure, breadth, and credit stress.
- Sornette-style log-periodic bubble models are treated as optional future
  diagnostics, not as the initial core. They are mathematically interesting but
  easy to overfit and should not be used before the simpler phase tracker has
  proved useful.

## Phase Taxonomy

The current phase set is:

| Phase | Meaning |
| --- | --- |
| `normal_cycle` | No dominant bubble/unwind state; broad market behavior is ordinary relative to the feature set. |
| `acceleration` | Leadership is strong, trend is constructive, and volatility/credit are not yet sending hard stress. |
| `pre_break` | Leadership remains strong but narrow, fragile, or more volatile; upside can continue, but crash risk is rising. |
| `early_unwind` | Former leaders have started to roll over while stress is rising but not yet broad liquidation. |
| `liquidation` | Broad equity, volatility, credit, and large-move evidence look like forced risk reduction. |
| `bottoming` | Drawdown is meaningful, but reversal, credit, or volatility-easing evidence is appearing. |
| `recovery` | Risk assets, breadth, and credit are recovering together. |
| `post_unwind_compounding` | The market has moved beyond acute unwind and broad participation supports compounding. |

## Data Leakage Rules

The tracker is built to avoid structural look-ahead:

- Current nowcast uses only prices available through the selected snapshot.
- Historical validation computes features from `prices.loc[:origin]`.
- Forward returns begin on the next trading session after `origin`, not on the
  feature date itself.
- Scenario probabilities are used only as current/future priors. Historical
  validation does not apply today's scenario map to past origins.
- Candidate scores are conditional reads, not optimized portfolio weights.
- Validation reports the number of prior origins so sparse phase evidence is
  visible.

## Outputs

The CLI command is:

```bash
poetry run trade-bot run-cycle-tracker
```

It writes:

| Artifact | Purpose |
| --- | --- |
| `reports/cycle_tracker/cycle_phase_probabilities.csv` | Current 0M phase nowcast probabilities. |
| `reports/cycle_tracker/cycle_transition_forecast.csv` | 0M current phase plus forward horizon phase frontier using current phase evidence and scenario-risk priors. |
| `reports/cycle_tracker/cycle_evidence_components.csv` | Feature components and raw values behind the current read. |
| `reports/cycle_tracker/cycle_candidate_scores.csv` | Conditional candidate scores for assets/sleeves in the current phase. |
| `reports/cycle_tracker/cycle_phase_candidate_frontier.csv` | Phase-by-horizon conditional winner shelf: for each plausible future phase and horizon, ranks assets using prior-only phase validation plus current momentum/drawdown context. |
| `reports/cycle_tracker/cycle_validation_metrics.csv` | Prior-only historical forward metrics by phase, horizon, and ticker. |
| `reports/cycle_tracker/cycle_validation_observations.csv` | Origin-level validation evidence for audit/debug. |
| `reports/cycle_tracker/summary.md` | Plain-language summary. |

It also persists run history and metrics in DuckDB tables:

- `cycle_tracker_runs`
- `cycle_tracker_phase_probabilities`
- `cycle_tracker_transition_forecast`
- `cycle_tracker_evidence`
- `cycle_tracker_candidate_scores`
- `cycle_tracker_phase_candidate_frontier`
- `cycle_tracker_validation_metrics`

## UI Location

The V2 Research page exposes a first-class `Cycle Tracker` tab beside Outcome
Frontier and Candidate Deep Dive. The UI reads persisted artifacts only. Heavy
phase validation remains a CLI/job step.

The V2 view should show:

- dominant 0M nowcast phase
- phase probability
- horizon phase frontier chart
- current-phase conditional winner candidate table
- scenario/phase winner frontier with horizon and phase selectors
- evidence components
- prior-only validation metrics

## Limitations

The initial implementation uses price-observable proxies. It does not yet use
true issuance, options positioning, dealer balance sheet stress, retail flow,
or proprietary 42 Macro style indicators. The candidate universe is mostly ETF
and liquid-proxy based, with individual names included only where existing data
is available.

The phase classifier is intentionally transparent and coarse. It should be
improved only when a new signal can pass the same prior-only validation rules.
