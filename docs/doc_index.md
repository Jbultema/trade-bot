# Documentation Index

Status: canonical navigation. Last reviewed: 2026-07-05.

Use this file to decide which docs are current operating references and which
ones are historical research notes. When a doc captures a dated plan or
experiment result, archive it or label it as dated rather than letting it look
like live system behavior.

## Canonical Operating Docs

| Doc | Use For | Maintenance Rule |
| --- | --- | --- |
| `README.md` | Running the app, dashboard walkthrough, daily workflow | Keep user-facing and operational; avoid deep backend theory here. Update when the operating overview, Insight Workbench, Simulation Lab, Research Lab layout, or Term Lookup behavior changes. |
| `docs/setup_guide.md` | Full local setup for users who do not regularly use GitHub, VS Code, Python, Poetry, or Codex | Update when Python version, install commands, environment assumptions, dashboard launch behavior, or onboarding workflow changes. |
| `docs/user_guide.md` | Full product guide for daily operation, research, monitoring, paper tracking, live logging, taxable review, and review cadence | Update when any dashboard workflow, CLI workflow, monitoring workflow, or paper/live journal workflow changes. |
| `docs/faq.md` | Comprehensive plain-English answers for users and reviewers | Update when recurring user questions appear, metric interpretations change, or safety/governance language changes. |
| `docs/technical_explainer.md` | Behind-the-scenes architecture, data flow, model semantics, risk engine behavior, storage, ML, monitoring, and extension rules | Update when implementation architecture, module boundaries, defaults, data contracts, or model/risk semantics change. |
| `docs/learnings.md` | Maintained research summary from experiment batches and operating experience | Review after major experiment batches, after significant dashboard pruning, and before promoting new operating systems. |
| `docs/backend_agent_guide.md` | Backend onboarding for engineers and future AI agents | Update when architecture, storage, command flow, dashboard structure, or ownership changes. |
| `docs/experiment_plan.md` | Current roadmap and Phase 1/Phase 2 boundary | Keep current; archive outdated plans instead of accumulating stale queues. |
| `docs/iteration_protocol.md` | Research loop, promotion rules, curation, and artifact roots | Update when experiment scoring, roots, or promotion semantics change. |
| `docs/math_model_audit.md` | Locked formulas, model semantics, and caveats | Update with any formula or interpretation change in the same PR/change. |
| `docs/forward_testing_protocol.md` | Paper/live ticket workflow and scaling gates | Update when journal, ticket, or monitoring workflows change. |
| `docs/ml_research_framework.md` | Classical ML/Bayesian seams, cadence, validation gates | Update when ML moves from research-only to operating recommendations. |
| `docs/taxable_account_framework.md` | Estimated taxable brokerage model, after-tax research semantics, tax lots, wash-sale checks, and TLH candidates | Update when account-aware scoring, tax-lot logic, dashboard tax views, or taxable assumptions change. |

## Maintained Research Notes

| Doc | Use For | Caution |
| --- | --- | --- |
| `docs/research_pruning_and_growth.md` | Current pruning rules and high-value regrowth direction | It summarizes empirical reads that can age; review after major experiment batches. |
| `docs/institutional_macro_coverage_gap.md` | Macro/data coverage roadmap and public-data limitations | It is a capability gap map, not a claim of commercial parity. |
| `docs/creative_strategy_backlog.md` | Falsifiable future candidate ideas | Backlog items are not approved strategies until tested and promoted. |

## Archived Historical Docs

| Doc | Why Archived |
| --- | --- |
| `docs/archive/experiment_plan_2026-06-17.md` | Original phase plan and first expansion queue. Superseded by `docs/experiment_plan.md`. |
| `docs/archive/experiment_reset_rerun_plan_2026-06-18.md` | Reset-era execution plan. The reset root exists now, so current root behavior belongs in `docs/iteration_protocol.md`. |

## Cleanup Rules

- If a doc says "current," "latest," "now," or "next," add a review date or make
  it evergreen.
- If a plan has already been executed, move it under `docs/archive/` and add an
  archive header explaining what replaced it.
- If dashboard behavior changes, update `README.md` and `docs/backend_agent_guide.md`.
- If math, labels, risk semantics, or ML promotion rules change, update the
  relevant technical doc and tests together.
- Do not put personal names or private sharing details in docs; use generic
  terms like "users," "reviewers," or "project owner."
