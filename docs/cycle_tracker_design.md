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

## Path-Aware State Model

The tracker produces two related reads:

- **Evidence probabilities**: simultaneous feature evidence for each phase.
  These behave like regime probabilities and can coexist.
- **Path-constrained phase state**: one decoded latent cycle path that applies
  transition rules, phase memory, and duration pressure.

The path-constrained state is the safer default for interpretation. It is a
transparent hidden semi-Markov approximation: phases have allowed transitions,
minimum/maximum duration bands, and preconditions. For example, `bottoming`
requires prior drawdown or unwind memory, `post_unwind_compounding` requires
prior unwind/recovery memory, and active severe stress can force the path from
`normal_cycle` or `acceleration` into `early_unwind` or `liquidation`.

This path layer is deliberately not a generic "phase affinity" score. It exists
because speculative-cycle phases are sequential. A market can show mixed
evidence, but it cannot be in a true post-unwind state without a prior unwind.
The UI should therefore show both views: evidence explains what signals are
firing now, while the path state explains which cycle states are currently
legal and historically plausible.

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
| `reports/cycle_tracker/cycle_path_state_history.csv` | Path-constrained decoded phase history with prior phase, duration, drawdown memory, and transition reason. |
| `reports/cycle_tracker/cycle_path_transition_forecast.csv` | 0M current path phase plus legal forward path probabilities using transition, duration, and precondition rules. |
| `reports/cycle_tracker/cycle_evidence_components.csv` | Feature components and raw values behind the current read. |
| `reports/cycle_tracker/cycle_candidate_scores.csv` | Conditional candidate scores for assets/sleeves in the current phase. |
| `reports/cycle_tracker/cycle_phase_candidate_frontier.csv` | Phase-by-horizon conditional winner shelf: for each plausible future phase and horizon, ranks assets using path-conditioned prior-only validation plus current momentum/drawdown context. |
| `reports/cycle_tracker/cycle_validation_metrics.csv` | Prior-only historical forward metrics by phase, horizon, and ticker. |
| `reports/cycle_tracker/cycle_validation_observations.csv` | Origin-level validation evidence for audit/debug. |
| `reports/cycle_tracker/cycle_phase_reliability.csv` | Phase-level classifier audit: when a historical origin was labeled with a phase, did the next horizon behave the way that phase implies? |
| `reports/cycle_tracker/cycle_path_validation_metrics.csv` | Prior-only forward metrics by decoded path phase, horizon, and ticker. This is the default evidence source for the winner frontier. |
| `reports/cycle_tracker/cycle_path_reliability.csv` | Path-phase audit. At 0M, this is same-date agreement between decoded path phase and raw evidence phase. At forward horizons, it asks whether the next horizon matched that path phase's expected behavior. |
| `reports/cycle_tracker/cycle_crisis_playback.csv` | Historical playback through named crisis windows, split into lead-up, unwind, and recovery stages. |
| `reports/cycle_tracker/summary.md` | Plain-language summary. |

It also persists run history and metrics in DuckDB tables:

- `cycle_tracker_runs`
- `cycle_tracker_phase_probabilities`
- `cycle_tracker_transition_forecast`
- `cycle_tracker_path_state_history`
- `cycle_tracker_path_transition_forecast`
- `cycle_tracker_evidence`
- `cycle_tracker_candidate_scores`
- `cycle_tracker_phase_candidate_frontier`
- `cycle_tracker_validation_metrics`
- `cycle_tracker_path_validation_metrics`
- `cycle_tracker_phase_reliability`
- `cycle_tracker_path_reliability`
- `cycle_tracker_crisis_playback`

## UI Location

The V2 Research page exposes a first-class `Cycle Tracker` tab beside Outcome
Frontier and Candidate Deep Dive. The UI reads persisted artifacts only. Heavy
phase validation remains a CLI/job step.

The V2 view should show:

- dominant 0M evidence phase and dominant 0M path-constrained phase
- phase probability
- independent horizon phase frontier chart
- path-constrained horizon phase frontier chart
- current-phase conditional winner candidate table
- scenario/phase winner frontier with horizon and phase selectors
- historical phase reliability and path-phase reliability cards/charts
- crisis playback selector for prior lead-up, unwind, and recovery periods
- evidence components
- prior-only validation metrics and path-conditioned validation metrics

## Reliability Read

The reliability read is deliberately phase-specific. It does not ask whether a
single top ticker won. It asks whether the phase label implied useful forward
behavior:

- `acceleration`: QQQ should beat SPY and cash-like exposure.
- `pre_break`: fragility should appear through drawdown, QQQ
  underperformance, or cash outperformance.
- `early_unwind`: QQQ should lag cash-like exposure or suffer a meaningful
  drawdown.
- `liquidation`: QQQ should lag cash-like exposure and suffer a severe
  drawdown.
- `bottoming`: risk assets should beat cash-like exposure without another deep
  leg down.
- `recovery`: QQQ and SPY should beat cash-like exposure with positive forward
  returns.
- `post_unwind_compounding`: broad risk assets should compound while QQQ
  remains positive without severe drawdown.
- `normal_cycle`: SPY should beat cash-like exposure without a large QQQ
  drawdown.

The UI reports fit rate, origin count, median forward benchmark behavior, and a
label such as `historically_supportive`, `mixed_but_useful`,
`weak_or_context_only`, `not_reliable`, or `thin_sample`. This makes sparse or
weak phase reads visible before users lean on the frontier.

Use the path reliability read first for sequential phases. The independent
phase reliability read remains useful for signal diagnostics, but it can
overstate states that are not sequentially legal yet.

## Frontier Interpretation

The winner frontier is conditional, not a recommendation list. It answers:
"if this phase dominates over this horizon, what historically worked from
similar prior-only path states?"

For `early_unwind` and `liquidation`, horizon matters:

- Short horizons (`1m`, `3m`) are defensive windows. Cash-like and ballast
  assets can be marked `defend` or `ballast`; speculative names are capped at
  `watch` even if a sparse historical sample happened to rebound.
- Longer horizons (`6m`, `1y`) can become reentry-after-stress windows. Growth
  or cyclicals may rank well there, but only with visible origin counts and
  drawdown context.

This prevents a one-off rebound from being misread as "buy speculative tech
during liquidation."

## Limitations

The initial implementation uses price-observable proxies. It does not yet use
true issuance, options positioning, dealer balance sheet stress, retail flow,
or proprietary 42 Macro style indicators. The candidate universe is mostly ETF
and liquid-proxy based, with individual names included only where existing data
is available.

The phase classifier is intentionally transparent and coarse. It should be
improved only when a new signal can pass the same prior-only validation rules.
