# Out-Of-Sample Model-Selection Policy

This policy is a fail-closed governance rule for all strategy and forecasting
research. A strong retrospective backtest, ordinary walk-forward score, crisis
test, bootstrap, or PBO result is not sufficient for promotion if the same
evidence influenced model architecture, feature, threshold, or parameter
selection.

## Required promotion evidence

A model may enter human promotion review only after one of these completed
evaluation designs:

1. **Sealed holdout**: the policy and primary metric are frozen before a
   chronologically later holdout is opened. The holdout may not be used for
   selection or retuning.
2. **Nested walk-forward**: all selection occurs only inside each outer fold.
   Outer-fold predictions remain untouched until the complete candidate and
   evaluation procedure are frozen.
3. **Prospective shadow**: the candidate is frozen, given zero allocation
   authority, and evaluated only on observations arriving after the freeze.

Every design must persist the selection cutoff, evaluation dates, observation
count, preregistered minimum sample, preregistered primary metric, parameter-freeze assertion, and an
assertion that evaluation outcomes were not used for model selection.

## What does not qualify

- choosing a winner after comparing ordinary walk-forward folds;
- calling the latest historical interval a holdout after inspecting it;
- leave-one-crisis-out tests used to redesign the rule;
- bootstrap resamples of the same adaptive research history;
- a prospective shadow that is still accumulating observations;
- changing a gate or threshold after viewing the candidate result.

These remain useful hypothesis-generation and robustness evidence. They cannot
authorize promotion.

## System behavior

- Missing or incomplete OOS evidence fails closed.
- Candidate scorecards retain their historical result and raw research
  decision, but their governed decision becomes `hold_for_oos_evidence`.
- A candidate cannot be labeled `operational_candidate` without a recognized,
  verified OOS status.
- Research manifests record the OOS audit and remain
  `automatic_promotion_allowed: false`; a verified audit only permits human
  review.
- Configured/reference strategies remain visible but do not inherit promotion
  evidence they have not earned.

The rule prevents adaptive research from silently converting retrospective
performance into forecasting evidence.
