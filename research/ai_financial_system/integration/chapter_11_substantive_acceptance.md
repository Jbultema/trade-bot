# Chapter 11 Substantive Acceptance

Status date: 2026-07-25

## Decision

**ACCEPT WITH NONADDITIVITY AND FIRST-LOSS CONTROLS.**

The corrected Chapter 11 packet in `staging/lane_3_chapter_11/` now resolves
the critical and material findings in the independent adversarial review. It
is suitable for a writing pass that classifies reciprocal finance and
role convergence without asserting a single circularity total or a
systemically consequential loss result.

## Principal accepted findings

1. **CoreWeave's 78 MW provider arrangement is the strongest documented
   reciprocal gross-finance case in the reviewed population.** The public
   filing establishes opposing loan/receivable, asset-sale, financing-
   obligation, and lease claims, plus partial proceeds recycling. It does not
   prove a dollar-for-dollar closed cash loop.
2. **Amazon–Anthropic is the strongest performance-linked financing-
   availability case.** Amazon's compute delivery creates availability under a
   future investment facility. The evidence does not show that Anthropic's AWS
   payments return as Amazon investment.
3. **OpenAI relationships contain different mechanisms rather than one loop.**
   Funded equity, conditional equity, cloud commitments, revenue share, an LOI,
   purchase-linked warrants, prepayments, and customer-supplied hardware must
   remain separate.
4. **NVIDIA–CoreWeave is correlated multi-role exposure, not a proven closed
   loop.** Equity, supply, customer-specified hardware, residual-capacity
   support, and customer-linked debt are separate, nonadditive channels.
5. **River Bend and Saline are layered risk-transfer chains.** Project company,
   landlord, tenant, guarantor, demand anchor, utility customer, end user,
   lender, and utility roles are legally distinct.

## Required interpretation controls

- No cross-case dollar total is authorized.
- A commitment, announced ceiling, contract value, funded amount, drawn amount,
  recognized balance, guarantee, and in-kind asset are different amount states.
- Cross-ownership is not governance control; a warrant or option is not present
  ownership before its conditions are met.
- Reciprocal commercial roles are not evidence that the same cash was recycled.
- A guarantee or legal obligation does not by itself identify economic first
  loss.
- The two system-level Grade-C conclusions are evidence-boundary results:
  public evidence does not currently satisfy the specified systemic-
  consequence test, and incompatible/overlapping states prohibit a circularity
  total. Neither means systemic consequence is absent.
- The circularity taxonomy is methodology, not a factual claim.

## First-loss correction

All 30 staged exposure rows now separate legal obligor, economic beneficiary,
accounting-holder impairment risk, and economic first-loss order:

- 25 rows state `first_loss_bearer=not_determined`;
- five funded or recognized asset rows name only the accounting holder's
  security-value or receivable-impairment risk;
- every one of those five explicitly states that it is not a claimant
  waterfall; and
- no sponsor, customer, guarantor, borrower, shareholder, or package is
  assigned economic first loss without security, intercreditor, recovery, and
  claimant-order evidence.

`L3-UNK-0027` records that missing evidence as a critical blocker. Quantitative
waterfall, recovery, return, and loss-allocation handoffs remain blocked.

## Revalidation

The primary integration lane reran the staging validator:

- 17 manifest-declared files;
- 484 CSV rows;
- strict CSV width and provisional-ID uniqueness checks passed;
- manifest hashes and row counts passed;
- canonical source and claim references passed.

The primary lane also confirmed:

- case-indexed evidence coverage for the two cross-case evidence-boundary
  conclusions;
- separate funded, recognized, committed, contingent, and announced legs;
- 30 of 30 corrected first-loss fields;
- zero economic claimant-waterfall assignments;
- blocked quantitative handoffs where first loss, utilization, recovery, or
  cash schedules are absent;
- exact $16 billion announced Saline project cost with funded amount unresolved,
  rather than a false $0–$16 billion range; and
- CoreWeave and Amazon–Anthropic terminology narrowed to the actual observed
  mechanisms.

## Downstream use

Chapters 16 and 17 may use the case taxonomy, role graph, nonadditive
components, and explicit unknowns. They may not convert role convergence into
control, probability, loss magnitude, or systemic consequence without new
evidence.
