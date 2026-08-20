# Chapter 12 Substantive Acceptance Record

Status date: 2026-07-25  
Packet: `staging/lane_1_chapter_12/`  
Cross-review: `staging/lane_2_review_chapter_12/cross_review_chapter_12.md`

## Decision

**Accept the corrected packet as a research-pass input map, subject to the
publication limits below.** This is not acceptance of a project IRR, AI-specific
ROIC, break-even utilization, first-loss threshold, or required-revenue model.
The packet's central finding is that no reviewed case currently contains the
matched asset-level evidence needed to calculate those outputs.

The primary validator independently passed the resealed packet:

- 14 declared files;
- 121 CSV data rows;
- manifest paths, hashes, row counts, and line counts;
- CSV shape and provisional-ID uniqueness;
- canonical source-ID and URL consistency; and
- staged source-claim referential integrity.

## Cross-review disposition

| Finding | Disposition | Primary check |
|---|---|---|
| CR12-01 | resolved | All nine required scenarios are present. Each retains the governance locator, base date, lag, affected-contract, loss-bearer, recovery-mechanism, perimeter, and execution-status fields. Unbounded scenarios remain `blocked_parameter` or `mechanism_only`; no probability is implied. |
| CR12-02 | resolved for research-pass use | The demand map contains every utilization-ladder state and the mandatory numerator, denominator, interval, basis, ownership/service, workload, aggregation, and exclusion fields. Unknown empirical states remain explicit and the gate is `partial`, not `pass`. |
| CR12-03 | resolved | River Bend debt, lease/NOI, and cash-priority facts are atomic rows with their own claims and dates. Saline contract terms and separate capital pools are distinct records. |
| CR12-04 | resolved | The OpenAI/AMD claim uses the canonical evidence as-of date and a separate announcement/effective-date field. |
| CR12-05 | resolved | The 1,106.4-MW diagnostic is consistently conditional on the 1,383-MW maximum contracted ramp. It is not current capacity, load, energy, revenue, or productive utilization. |
| CR12-06 | resolved | The River Bend lease division is displayed as approximately USD 466.7 million per year and labeled a nominal full-term aggregate average, not a period payment or cash-flow schedule. |
| CR12-07 | resolved | The River Bend coupon identity is displayed as approximately USD 201.24 million per year and labeled an annualized pre-amortization run rate conditional on full principal outstanding for a full coupon year. Actual cash timing remains unresolved. |
| CR12-08 | resolved | Diagnostic handoff readiness is `diagnostic_only`, not an evidence-state claim of `bounded`. |
| CR12-09 | resolved | The false-precision gate names the conditional labels and display rounding for all three diagnostics. |
| CR12-10 | resolved | Pipe-delimited claim lists contain complete canonical claim IDs. |

## Publication limits

1. Chapter 12 may publish the absence of a calculable bounded model as a
   discovery about disclosure and system observability.
2. It may publish the three contract-implied diagnostics only with their full
   conditions and negative definitions.
3. It must not imply that the Chapter 12 writing gate for a fully traceable
   bounded project model passed. `WG-12-002` remains blocked.
4. The 20% capex and 90% inference-price scenarios are governed
   counterfactuals, not forecasts. They are not executable against an asset
   model until a compatible bounded baseline exists.
5. Chapter 13 and Chapter 14 may consume the packet's schemas and
   diagnostic-only handoffs, but they may not promote unresolved inputs into
   bounded or observed states.

## Remaining evidence gap

The critical path remains one matched project or asset cohort containing dated
capital, productive units, realized price, variable and fixed costs,
maintenance/replacement capital, cash conversion, claimant waterfall, and
terminal or recovery evidence. Until that evidence exists, inability to
calculate a return is evidence of opacity, not evidence of either attractive or
unattractive economics.
