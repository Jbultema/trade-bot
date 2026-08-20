# Chapter 15 Substantive Acceptance

Status date: 2026-07-25

## Decision

**ACCEPT WITH EVIDENCE-STATE LIMITS.**

The corrected Chapter 15 packet in `staging/lane_2_chapter_15/` resolves the
critical and material findings in the independent adversarial review. It is
suitable for a writing pass that keeps ownership, contractual control,
accounting control, and operational dependence separate.

## Accepted discoveries

- OpenAI Foundation's disclosed director appointment and removal rights are
  distinct from minority economic ownership.
- Microsoft/OpenAI platform and IP rights are contract-specific and
  time-specific; the April 2026 amendment supports a nonexclusive license and
  other-cloud permission, not Microsoft governance control.
- Meta's Hyperion tenant, manager, investor, and guarantor roles do not by
  themselves establish accounting control.
- Lender account control, collateral, cure, disposition, and step-in remedies
  are conditional rights, not ordinary-course corporate control.
- Guarantees create exposure without necessarily creating board, voting, or
  operating authority.
- Platform-dependence assessments are analytical inferences even when their
  component contracts are directly observed.

## Required interpretation controls

1. No control dimension may be inferred automatically from another. Equity
   ownership, a guarantee, customer importance, collateral, and accounting
   consolidation answer different questions.
2. All eight synthetic dependency assessments are Grade-C inferences with a
   stated method, date scope, and falsifier.
3. The April 2026 OpenAI/Microsoft contractual state must not be projected
   backward onto the October 2025 ownership and governance state.
4. Saline payment support, Oracle parent support, and DTE residual regulatory
   allocation remain atomic facts. They do not establish a single controller
   or first-loss waterfall.
5. No unpublished board right, voting threshold, private collateral schedule,
   remedy enforceability, switching cost, or system-wide controller may be
   asserted.

## Revalidation

The primary integration lane reran the staging validator:

- 12 manifest-declared files;
- 100 CSV rows;
- strict CSV width and provisional-ID uniqueness checks passed;
- manifest hashes and row counts passed;
- canonical source and claim references passed.

The primary lane independently confirmed:

- every `ENT-*` token in the packet resolves to the canonical entity registry;
- the former asset-cohort and external-VIE pseudo-entities are no longer used
  as canonical identities;
- `PDM-150001` through `PDM-150008` are `inference,C`;
- `HND-150001` is `partial_observed` and `HND-150005` is
  `ready_with_limits`;
- the Saline facts are separated into payment, support, and regulatory
  allocation records; and
- unsupported Anthropic/SpaceX notice-mechanics wording was removed.

## Downstream use

Chapters 16 and 17 may use the observed right-level ledger and the explicitly
inferred dependency map. Network centrality or stress propagation must not turn
those inference rows into observed control, and must preserve the effective
date and trigger for each right.
