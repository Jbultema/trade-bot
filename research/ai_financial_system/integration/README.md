# Canonical Integration Workspace

This directory is owned by the primary lane. It records how isolated staging
proposals were accepted, revised, merged, blocked, or rejected before canonical
database mutation.

The files here are decision evidence, not a second source of facts. Canonical facts
remain in `research/ai_financial_system/data/` after an accepted integration passes a
temporary database build.

## Current first-convergence result

- Lane 1 Chapters 2–5: sealed structural validation passed.
- Lane 2 Chapters 6–9: sealed structural validation passed.
- Reciprocal adversarial review: completed and originating-lane corrections applied.
- Proposed claim adjudication: 83 of 83 rows completed.
- New canonical reconstruction claims: two, both from Chapter 11:
  - no reviewed case presently clears the three-part systemic-consequence gate; and
  - a single circularity dollar total is invalid because unlike and overlapping
    amount states would be added.
- The other 81 proposals restate or join existing canonical atomic evidence or define
  model boundaries. Those joins remain valuable for chapter revision, but duplicate
  claims would inflate apparent corroboration and obscure provenance.
- Proposed common-exposure adjudication: 55 of 55 rows completed.
  - 34 accepted with canonical revisions or shared identities;
  - 14 retained as duplicate components in existing specialized ledgers;
  - 7 rejected as reporting/model/reconciliation/component wrappers rather than
    monetary
    exposures.
- Unknown and first-wave handoff adjudication: completed. The first-convergence
  integration candidate contains the accepted new claims, evidence links,
  governed unknowns, 20 accepted handoff replacements/additions, 34 common
  economic exposures, and 49 nonadditive component links.
- Integration candidate: independently verified and accepted. A subsequent
  cross-ledger repair added two shared common identities and two nonadditive
  component links: Microsoft's $196.6 billion uncommenced-lease population
  (`EXP-020001` / `HOBL-070003`) and NVIDIA's adjustable manufacturing-
  commitment ceiling (`EXP-020002` / `SCF-080001`).
- Canonical database: rebuilt successfully with `integrity_check=ok` and no
  foreign-key violations. It now contains 427 claims, 581 evidence links, 252
  unknowns, 36 common economic exposures, 51 exposure-component controls, and
  47 chapter handoffs.
- Second-half governance integration: 30 nonduplicate Chapter 12–15 unknowns
  and 21 governed handoffs were independently verified, database-gated, and
  promoted. Nine semantically duplicate unknown proposals were routed to
  existing questions instead of creating parallel queues. Handoff rows retain
  one scalar foreign-key claim anchor while preserving complete supporting
  claim sets in notes.
- Fourteen `duplicate_component` and seven rejected-wrapper decisions remain
  outside `economic_exposures`. Their specialized-ledger linkage passed the
  separate cross-ledger validation; they were not silently promoted by the
  common-exposure load.
- Chapter 11: corrected sealed staging research passed structural and
  substantive review. It is accepted with nonadditivity, amount-state, and
  first-loss controls. See `chapter_11_substantive_acceptance.md`.
- Chapter 12: corrected packet passed independent structural validation and
  substantive cross-review disposition. It is accepted as an input/readiness map,
  not as a return model; the fully traceable bounded-model writing gate remains
  blocked. See `chapter_12_substantive_acceptance.md`.
- Chapter 13: final corrected packet passed the primary structural validator and
  cleared the residual source-sufficiency review. It is accepted as a governed
  return-model readiness and partial claimant-priority map, not as an IRR, ROIC,
  reverse-DCF, or first-loss model. See
  `chapter_13_substantive_acceptance.md`.
- Chapter 14: corrected packet is accepted as a controlled diagnostic input
  packet, not a utilization, price, cost-curve, useful-life, or survivability
  model. See `chapter_14_substantive_acceptance.md`.
- Chapter 15: corrected packet is accepted with explicit separation of
  ownership, contractual control, accounting control, and operational
  dependence. See `chapter_15_substantive_acceptance.md`.

## Required sequence

1. Validate the sealed staging packet with
   `scripts/validate_staging_packet.py`.
2. Record claim decisions row by row.
3. Reconcile stable entity, project, instrument, exposure, and overlap identities.
4. Record exposure/component, unknown, and handoff decisions.
5. Materialize accepted rows into a temporary copy of the canonical CSV set.
6. Build a temporary SQLite database and run foreign-key, evidence, unit, status,
   overlap, and chapter-gate checks.
7. Review the database diff and affected chapter manifests.
8. Only then update canonical CSVs and rebuild publication artifacts.

No staging manifest or passing structural validator substitutes for substantive
acceptance.
