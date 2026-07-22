# Documentation Index

Status: canonical navigation. Last reviewed: 2026-07-21.

Use this file to decide which docs are current operating references and which
ones are historical research notes. When a doc captures a dated plan or
experiment result, archive it or label it as dated rather than letting it look
like live system behavior.

## Canonical Operating Docs

| Doc | Use For | Maintenance Rule |
| --- | --- | --- |
| `README.md` | Running the app, dashboard walkthrough, daily workflow | Keep user-facing and operational; avoid deep backend theory here. Update when the operating overview, Insight Workbench, Simulation Lab, Research Lab layout, or Term Lookup behavior changes. |
| `docs/whitepaper.md` | Semi-technical system narrative for users, reviewers, and technical readers who need the full system in one document | Keep evergreen and concise relative to detailed guides; update after major architecture, strategy-family, testing, simulation, or monitoring changes. |
| `docs/ai_review_whitepaper.md` | Deep machine-oriented specification for independent LLM or technical review | Update after causal-authority, snapshot-lineage, calibration, primary-strategy, or major evidence changes. Keep explicit claims, evidence limits, and reviewer questions. |
| `docs/42macro_alignment.md` | External 42 Macro transcript comparison, outcome method, and latest interpretation | Update after transcript syncs or any posture/classification mapping change. Preserve horizon and independence caveats. |
| `docs/ai_review_whitepaper.md` | Deep machine-oriented audit packet for an independent LLM review | Refresh exact snapshot metrics, evidence cut, causal authority, known gaps, and reviewer questions after material policy or validation changes. |
| `docs/setup_guide.md` | Full local setup for users who do not regularly use GitHub, VS Code, Python, Poetry, or Codex | Update when Python version, install commands, environment assumptions, dashboard launch behavior, or onboarding workflow changes. |
| `docs/user_guide.md` | Full product guide for daily operation, research, monitoring, paper tracking, live logging, taxable review, and review cadence | Update when any dashboard workflow, CLI workflow, monitoring workflow, or paper/live journal workflow changes. |
| `docs/faq.md` | Comprehensive plain-English answers for users and reviewers | Update when recurring user questions appear, metric interpretations change, or safety/governance language changes. |
| `docs/technical_explainer.md` | Behind-the-scenes architecture, data flow, model semantics, risk engine behavior, storage, ML, monitoring, and extension rules | Update when implementation architecture, module boundaries, defaults, data contracts, or model/risk semantics change. |
| `docs/learnings.md` | Maintained research summary from experiment batches and operating experience | Review after major experiment batches, after significant dashboard pruning, and before promoting new operating systems. |
| `docs/backend_agent_guide.md` | Backend onboarding for engineers and future AI agents | Update when architecture, storage, command flow, dashboard structure, or ownership changes. |
| `docs/experiment_plan.md` | Current operating roadmap, active priorities, and pruning guardrails | Keep evergreen; do not let executed phase plans remain as apparent current work. |
| `docs/iteration_protocol.md` | Research loop, promotion rules, curation, and artifact roots | Update when experiment scoring, roots, or promotion semantics change. |
| `docs/math_model_audit.md` | Locked formulas, model semantics, and caveats | Update with any formula or interpretation change in the same PR/change. |
| `docs/forward_testing_protocol.md` | Paper/live ticket workflow and scaling gates | Update when journal, ticket, or monitoring workflows change. |
| `docs/ml_research_framework.md` | Classical ML/Bayesian seams, cadence, validation gates | Update when ML moves from research-only to operating recommendations. |
| `docs/taxable_account_framework.md` | Estimated taxable brokerage model, after-tax research semantics, tax lots, wash-sale checks, and TLH candidates | Update when account-aware scoring, tax-lot logic, dashboard tax views, or taxable assumptions change. |
| `docs/cycle_tracker_design.md` | Scenario/phase frontier and speculative-cycle tracker design, leakage controls, artifact contract, and UI placement | Update when phase taxonomy, validation rules, persistence, or V2 Cycle Tracker UI changes. |

## Maintained Research Notes

| Doc | Use For | Caution |
| --- | --- | --- |
| `docs/research_pruning_and_growth.md` | Current pruning rules and high-value regrowth direction | It summarizes empirical reads that can age; review after major experiment batches. |
| `docs/risk_repair_research.md` | i111 risk-repair architecture, current native challenger, commands, and interpretation | Maintained research note; do not treat the challenger as primary until promoted through the normal protocol. |
| `docs/new_chat_seed_i111_adversarial_research_prompt.md` | Dated handoff that initiated the 2026-07-21 cross-sectional, prospective-monitoring, and simulation-tooling pass | Historical context only; refresh before reusing because the named work is now complete. |
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
- If a doc says simulation, Cycle Tracker, V2 dashboard, Launch Lab, or
  Experiment Operator are future-only work, either update it to current behavior
  or archive it as a dated plan.
- If dashboard behavior changes, update `README.md` and `docs/backend_agent_guide.md`.
- If math, labels, risk semantics, or ML promotion rules change, update the
  relevant technical doc and tests together.
- Do not put personal names or private sharing details in docs; use generic
  terms like "users," "reviewers," or "project owner."
