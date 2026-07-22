# V2.2 External Audit Response

Status: implemented and empirically rerun on 2026-07-22.

## Bottom line

The audit found one consequential execution-semantics defect and one
consequential historical-replacement defect.

1. The former one-row close-data shift implicitly earned the return beginning
   at the same close used to calculate the signal. The operating convention is
   now the close-safe two-row shift.
2. The old 1:1 replacement path could reuse stored strategy results while
   stamping the artifact with a changed configuration. Replacement now requires
   matching source identity to resume and recomputes every price-derived object
   invalidated by execution changes.

The corrected current paths are:

| Object | CAGR | Maximum drawdown | Interpretation |
| --- | ---: | ---: | --- |
| Configured primary, close-safe lag 2 | 20.67% | -25.80% | Operating retrospective path |
| Native i111 challenger, close-safe lag 2 | 20.84% | -24.75% | Research challenger |
| Native i111, old lag 1 | 22.18% | -19.68% | Boundary-fill approximation only |
| Five-weekday lag-1 median | 19.70% | -27.06% | Calendar-fragility comparator |
| Equal-weight weekday ensemble | 20.35% | -26.16% | More credible than best weekday, still retrospective |

The July 21 decision is now 39.98% risk assets / 60.02% defensive. Macro,
scenario, event/news, risk timing, and portfolio clamps add zero percentage
points. This is native-strategy defense, not multi-layer independent agreement.

## Disposition by audit item

| Audit item | Disposition | Evidence / control |
| --- | --- | --- |
| End-to-end execution timestamp | Fixed | `build_execution_causality_trace` exposes observation, target, first holding, return interval, fill field, and boundary approximation. Default lag is 2. |
| Weekday robustness | Fixed experiment, failed gate | Mon-Fri, daily, lag-2, lag-5, and equal-weight weekday ensemble are persisted. Wednesday fails the <=5 pp drawdown-advantage gate and lag-2 degradation narrowly fails 5 pp. |
| Risk-score orientation | Hardened | Production calculation is public and property-tested so independently more bearish inputs cannot lower the risk score. |
| Allocation authority | Hardened | Omitted allocation policy fails closed. Scenario, event/news, risk timing, and revised-history macro cannot size without explicit calibrated authority. |
| Point-in-time universe / delistings | Fail-closed, data unresolved | Every research manifest audits membership, holding eligibility, delisting treatment, and sources. All 540 indexed trials remain promotion-ineligible because verified point-in-time evidence is absent. |
| Research-wide selection | Implemented as governance blocker | 569 manifested rows, 19 manifests, 16 studies, three manifests without explicit rosters, and 113 artifact directories without manifests. Retrospective promotion status is `prospective_evidence_required`. |
| Frozen prospective evidence | Hardened | Runtime provenance records git/tree/dirty state, exact source tree, dependency lock, config, schema, and price-frame identity. Any revision starts a new source identity. |
| Revised macro vintage | Fixed authority | Macro authority defaults to zero. Nonzero authority requires provisional/validated calibration plus point-in-time or first-release vintage status. |
| Scenario false precision | UI corrected | Primary UI says raw risk-off score, displays insufficient calibration and 0% authority, and avoids treating the number as a literal forecast. |
| “Correct defense” semantics | UI/report interpretation corrected | User-facing labels are “beneficial under rule” and “costly false positive”; the rule and episode unit are stated. Internal legacy column names remain for artifact compatibility. |
| PBO scope | Qualified | Dashboard calls it within-shelf PBO and states that abandoned/unmanifested research is excluded. |
| Catastrophic-tail utility | Implemented | Bootstrap summary reports P(DD >10/20/30%), conditional loss beyond 20%, target-wealth attainment, and flow-neutral drawdown. |
| Contribution-aware planning | Implemented, bounded | Default is 15 years, $220K start, $4K annual contributions deposited monthly. Terminal wealth includes flows; drawdown does not. Retirement spending/funding-ratio modeling remains future work. |
| Canonical pickle replacement | Deferred | New manifests make pickle identity inspectable, but DuckDB/Parquet plus schema-versioned JSON has not yet replaced pickle as the canonical snapshot payload. |
| Defensive asset policy | Deferred fixed experiment | BIL remains a policy choice. No multi-asset defensive-basket search was run in this cycle to avoid expanding selection degrees of freedom. |

## Rebuilt evidence

- Replaced 223/223 operating-history dates through 2026-07-22 and pruned superseded generations.
- Replaced 397/397 pre-break-history dates and pruned superseded generations.
- The refreshed pre-break analysis contains 486 observations including
  reference controls and 42 post-break event-window snapshots.
- Sparse risk overlays lowered median CAGR from 13.09% to as low as 11.94% and
  did not improve the -22.10% median maximum drawdown.
- At 65% native defense, the 42 one-month episode starts were 45.2% beneficial
  under the stated rule, 33.3% costly false positives, and 21.4% mixed. At three
  months: 42.9%, 31.0%, and 26.2%.
- Native plus confirmation-timed defense showed some conditional discrimination
  but only 10 one-month starts; it remains insufficient for authority.

## Catastrophic-tail fixed experiment

The fixed 1,000-path, 21-session block bootstrap uses a 15-year horizon,
$220,000 initial portfolio, $4,000 annual contributions deposited monthly, and seed
`20260705`. The wealth target is the terminal value of the same cash flows at a
deterministic 10% return ($1.05M).

| Path | P(DD >25%) | P(DD >30%) | Conditional mean DD when >20% | Target success | Median terminal wealth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Configured primary | 48.5% | 22.1% | -27.2% | 98.5% | $4.22M |
| Native i111 | 47.8% | 21.7% | -27.1% | 98.6% | $4.29M |
| SPY hold | 92.3% | 73.8% | -36.9% | 60.9% | $1.25M |
| QQQ hold | 95.9% | 80.0% | -38.2% | 82.2% | $2.21M |

This is useful comparative utility evidence, not a forecast. It is still based
on contribution-aware resampling of modern-universe retrospective paths.

## Explicitly not pursued

- No new AI/news sizing features.
- No scenario-model search or fast recalibration intended to regain authority.
- No threshold grid or “perfect” risk-score search.
- No complex optimizer.
- No claim that the point-in-time universe problem has been solved without a
  source that includes historical membership and delisting returns.

The correct next promotion evidence is a frozen prospective cohort plus a real
point-in-time investable dataset. More retrospective parameter tuning cannot
repair those evidence gaps.
