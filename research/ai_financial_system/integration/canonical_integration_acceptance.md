# First-Convergence Canonical Integration Acceptance

Status date: 2026-07-25

## Promoted population

The primary lane promoted the independently verified integration candidate into
the canonical data directory:

- 2 new cross-case Chapter 11 evidence-boundary claims;
- 9 evidence links for those claims;
- 12 governed unknowns;
- 34 common economic exposures;
- 49 nonadditive exposure-component links; and
- 20 accepted chapter-handoff replacements or additions.

The handoff operation updated 11 existing rows and added 9 rows. The resulting
canonical population contains:

| Table | Canonical rows |
|---|---:|
| `claims` | 427 |
| `evidence_links` | 581 |
| `unknowns` | 252 |
| `economic_exposures` | 36 |
| `exposure_components` | 51 |
| `chapter_handoffs` | 47 |

## Acceptance evidence

The six promoted CSVs are byte-identical to their independently verified
candidate versions. The rebuilt canonical SQLite database returned:

- `PRAGMA integrity_check = ok`;
- no `PRAGMA foreign_key_check` violations;
- no orphan common exposures or components;
- canonical entity, project, and source-claim resolution for all non-null
  references;
- source evidence-grade equality for all exposures;
- valid floor, ceiling, contingent, stock, flow, and unquantified amount states;
  and
- exactly one named first-loss field: NVIDIA as the accounting bearer of its
  own recognized H20 charge.

Independent verification is sealed in
`staging/lane_1_verify_exposure_candidate/verification_report.md`.

## Cross-ledger repair supplement

The first convergence exposed two valid chapter-level observations that lacked
shared common-exposure identities. The primary lane added:

- `EXP-020001`, Microsoft's $196.6 billion undiscounted uncommenced-lease
  payment population, linked nonadditively to `HOBL-070003`; and
- `EXP-020002`, NVIDIA's adjustable $0–$119 billion manufacturing-commitment
  ceiling, linked nonadditively to `SCF-080001`.

The repair also:

- retargeted River Bend's Anthropic dependency to `CLM-060014`, preserving
  Anthropic as demand anchor rather than borrower, tenant, or guarantor;
- retained Apollo's $35 billion Broadcom XPV capital solution as
  `CAP-090001`, an announced, unfunded, noncash observation that is not added
  to Broadcom's separate contingent backstop;
- corrected the Hyperion reconciliation to `CAF-030017`; and
- resolved the five associated cross-ledger handoff targets.

Independent verification is sealed in
`staging/lane_1_verify_cross_ledger_fixes/verification_report.md`. The
regenerated candidate contained 58 CSVs and 2,610 rows, passed strict width and
reference checks, returned `PRAGMA integrity_check = ok`, and had no
foreign-key violations. The two new canonical files are byte-identical to the
verified candidate versions.

## Second-half governance supplement

The accepted Chapter 12–15 packets proposed additional unknowns and analytical
handoffs. Independent review rejected nine unknown rows as semantic duplicates,
corrected 31 noncanonical IDs, repaired two blocker routes, and replaced six
overstated “bounded qualitative artifact” descriptions with
“evidence-limited qualitative artifact.” The safe promoted population was 30
unknowns and 21 handoffs.

A downstream temporary-database build then rejected pipe-delimited supporting
claim sets in the scalar `source_claim_id` foreign-key field. The integration
was rebuilt so each handoff retains one resolved principal claim in that field
and the complete ordered support set in `notes`. Independent re-verification
reconstructed all 51 references, confirmed no support loss, reproduced the
candidate byte-for-byte, and returned clean quick, integrity, and foreign-key
checks.

The rebuilt canonical database contains 252 unknowns and 47 handoffs. The
review evidence is sealed in
`staging/lane_1_verify_second_half_governance/final_reverification.md` and
`database_gate_reverification.md`.

## Interpretation controls

1. Every common component is `additive_eligible=0`. This is intentionally
   conservative until an explicit aggregation view proves compatible units,
   dates, legal perimeters, and amount states.
2. Aggregate CoreWeave debt is not added to its DDTL subset.
3. Recognized operating-lease liabilities are not added to undiscounted
   uncommenced lease payments.
4. River Bend debt, tenant lease value, and Google support remain distinct.
5. Saline's announced campus cost is not funded capital and is not added to
   utility storage or supply formulas.
6. Investments, contingent financing facilities, and cloud-service commitments
   remain separate even when they connect the same counterparties.
7. CoreWeave's provider receivable, asset purchase, and opposing lease/finance
   obligations remain gross and nonadditive; no closed cash loop is inferred.
8. A legal obligor, accounting holder, guarantor, or commercial counterparty is
   not automatically an economic first-loss bearer.

## Excluded population

Fourteen `duplicate_component` proposals and seven rejected wrappers were not
loaded as new common economic exposures. They remain subject to a separate
cross-ledger validation against the specialized ledgers. This acceptance does
not authorize their silent aggregation or imply that a referenced shared common
exposure already exists.
