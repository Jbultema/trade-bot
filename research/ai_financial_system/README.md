# AI Financial System Forensic Research

This directory contains the primary research artifact for the AI financial-system
monograph. The report is downstream of this evidence base.

## Two-pass chapter workflow

1. **Research pass:** collect, archive, extract, cross-reference, reconstruct, and
   challenge the evidence. A chapter cannot enter drafting with unresolved provenance
   or without leaving behind structured ledgers.
2. **Writing pass:** write only from the frozen chapter evidence package. Every
   material factual statement must map to a claim and one or more sources. Inferences
   and hypotheses remain visibly separate from documentary facts.

The mandatory chapter architecture, falsification standard, confidence-history rules,
economic-importance classifications, economics hierarchy and delivered-chapter
revision protocol are defined in `editorial_and_investigative_standard.md`.

## Evidence grades

- **A — Primary evidence:** regulatory filings, audited financial statements,
  prospectuses, bond documents, earnings transcripts, contracts, official company or
  government records.
- **B — High-quality reporting:** Reuters, Financial Times, Bloomberg, Wall Street
  Journal, and rating-agency research.
- **C — Strong inference:** a conclusion supported by multiple independent records but
  not explicitly disclosed.
- **D — Hypothesis:** a testable proposition that is not yet established.

Evidence grade describes the source basis. Confidence describes the research team's
assessment after triangulation. A primary source can still be incomplete, ambiguous,
or narrowly drafted.

## Non-negotiable distinctions

Every monetary assertion must distinguish, where applicable:

`announced → committed → contracted → funded → drawn → spent → recognized`

Guarantees and contingent obligations are tracked separately. No amount is silently
promoted from one status to another.

Accounting dependence and economic dependence are also separate:

- **Accounting:** recognized assets, liabilities, revenue, expense, cash flow, equity,
  consolidation, or disclosed commitments.
- **Contractual:** enforceable rights and obligations, including termination,
  exclusivity, minimum-spend, security, and guarantee provisions.
- **Funding:** who supplied capital, in what instrument, with what seniority and
  recourse.
- **Operational:** inputs whose interruption can impair delivery even without a
  recognized liability.
- **Control:** governance, voting, consent, step-in, collateral, exclusivity, and
  information rights.

## Project layout

- `schema.sql`: durable relational schema.
- `data/source_register.csv`: source-level provenance.
- `data/entities.csv`: canonical entity registry.
- `data/claims.csv`: atomic factual assertions and interpretations.
- `data/evidence_links.csv`: many-to-many links between claims and sources.
- `data/capital_bridge.csv`: common capital-channel comparison with explicit temporal,
  recognition, AI-attribution, and overlap controls.
- `data/funding_timeline.csv`: round aggregates, investor components, tranches,
  supersession, and financing sources that keep announced, committed, and funded capital distinct.
- `data/graph_selection_policy.csv`: explicit overlap-group rules used to freeze
  Chapter 2 capital-flow graph inputs without double counting aggregate and component rows.
- `data/unknowns.csv`: unresolved questions and planned resolution paths.
- `data/claim_confidence_history.csv`: append-only dated grade and confidence
  observations; absent historical grades are never invented.
- `data/claim_economic_importance.csv`: magnitude, horizon and valuation or risk
  channel for principal discoveries.
- `data/economic_exposures.csv`: common exposure state vector with stable identity,
  legal and economic parties, status, recourse, first loss, confidence and falsifier.
- `data/exposure_components.csv`: nonadditive joins from one economic exposure to its
  contract, project, lease, loan, utility, compute and other observations.
- `data/utilization_observations.csv`: installed-through-productive utilization ladder
  and separate financial, physical and economic demand states.
- `data/chapter_handoffs.csv`: governed inputs, readiness and missing evidence passed
  from the foundational chapters to the economics and stress chapters.
- `model_specifications/economic_model_governance.md`: calculation contract for
  utilization, unit economics, project IRR, ROIC, required revenue, reverse DCF,
  first-loss thresholds and scenario outputs.
- `model_specifications/chapter_12_14_research_program.md`: shared research program and
  writing gates for the economic-engine, cash-waterfall and utilization chapters.
- `staging/`: temporary isolated research packets. Superseded lanes are moved to a
  reversible quarantine after a self-contained publication release is sealed; no
  staged row becomes canonical without identity resolution and validation.
- `chapters/chapter_01/`: frozen research pass and writing source for Chapter 1.
- `chapters/chapter_07/`: frozen five-company hyperscaler balance-sheet reconstruction.
- `chapters/chapter_08/`: frozen semiconductor physical-control, capacity-finance,
  substitution, regulatory, and loss-transmission reconstruction.
- `chapters/chapter_09/`: frozen private-credit exposure, fund-liquidity, warehouse
  control, valuation-clock, measurement-reconciliation, and risk-transmission
  reconstruction.
- `chapters/chapter_10/`: frozen public-award, tax, government-equity,
  utility-cost-allocation, sovereign-capital, confidence-history and
  economic-importance reconstruction.
- `chapters/chapter_11/`: preserved circularity and cross-ownership research pass.
- `scripts/build_database.py`: reproducibly constructs the SQLite research database.
- `scripts/build_chapter_01_methodology_data.py`: freezes the revised methodology
  claims, unknowns, confidence baselines, importance classifications and writing gates.
- `scripts/build_chapter_02_graph_data.py`: freezes separate committed-capital,
  funded-cash, and non-cash dependency graph datasets for Chapter 2.
- `scripts/build_chapter_07_hyperscaler_data.py`: freezes the Chapter 7 capex,
  obligations, customer-funding, debt, uncommenced-lease and risk-transmission
  comparison artifacts.
- `scripts/build_chapter_08_semiconductor_data.py`: freezes the Chapter 8 serial
  physical-control map, capacity-finance observations, loss pathways, substitution
  constraints, regulatory states, measurement reconciliations, and writing-gate
  audit.
- `scripts/build_chapter_09_private_credit_data.py`: freezes the Chapter 9 named-loan
  look-through, nontraded-fund liquidity, warehouse-control, valuation-clock,
  capital-chain, measurement-reconciliation, and writing-gate artifacts.
- `scripts/build_chapter_10_public_capital_data.py`: freezes award-to-cash status,
  public loan, tax-credit, government-equity, utility allocation, sovereign vehicle,
  measurement-reconciliation, falsification-input and writing-gate artifacts.
- `scripts/render_chapter.py`: legacy chapter-HTML conversion retained for
  reproducibility of previously delivered files.
- `scripts/render_publication_docx.py`: deterministic professional chapter renderer
  with real Word heading styles, institutional typography, cover metadata,
  repeating table headers, callouts, headers, footers, and page fields. It
  normalizes source-only whitespace and uses paragraph pagination rather than
  explicit break glyphs.
- `scripts/inspect_publication_docx.py`: fails closed on missing package parts or
  styles, explicit page-break glyphs, missing-glyph markers, absent metadata or page
  fields, empty headings, missing table controls, and loss of the source word
  sequence.
- `scripts/inspect_publication_pdf.swift`: uses PDFKit to extract text and metadata,
  inventory links and outlines, render every page, and identify blank or duplicate
  page candidates before human visual acceptance.
- `scripts/validate_publication_package.py`: fails closed unless the executive
  summary, complete monograph, and essay each have a consistent manifest, canonical
  database hash, source tree, searchable PDF audit, and accepted page-by-page visual
  inspection.
- `scripts/build_private_investment_candidate.py`: transforms the independently
  accepted private-investment earnings packet into typed, losslessly audited
  candidate tables without mutating canonical data.
- `scripts/validate_private_investment_candidate.py`: independently recomputes
  issuer arithmetic, roll-forwards, population controls, relational integrity,
  field preservation, and two clean candidate builds.
- `scripts/build_chapters_06_09_revision_inventory.py`: freezes the complete
  Chapters 6–9 source/claim/evidence-link preservation population and reconciles
  locally present original bytes without implying semantic support.
- `scripts/preserve_public_source_set.py`: plans or performs sequential,
  rate-limited P1/P2 source preservation with versioned bytes, response metadata,
  hashes, explicit failures, and resumable manifests.
- `scripts/validate_public_source_archive.py`: independently re-hashes every
  preserved source, checks paths, byte counts, response metadata, failure
  boundaries, and inventory completeness.
- `publication_pdf_standard.md`: governs the final three-PDF publication package:
  executive summary, complete monograph, and standalone long-form essay.
- `chapter_docx_publication_standard.md`: governs the independently usable,
  deterministic, professionally formatted Chapter 1–19 DOCX packages and their
  automated and page-by-page visual acceptance.

Generated databases and chapter deliverables live under
`reports/ai_financial_system/`.

## Publication release

The publication of record is:

`reports/ai_financial_system/publication/AI_Financial_System_Research_Release_1_1/`

It contains:

- the 4-page executive summary, 138-page full monograph, and 15-page
  **Who Owns the Future Before It Exists?** investigative essay as separate PDF
  and DOCX files;
- nineteen cleaned standalone chapter DOCX files and two research appendices;
- the exact accepted publication and chapter sources;
- the canonical research database and release receipts;
- the investigative framework and reproducibility scripts; and
- compact sealed manifests plus independent normal-visual, grayscale, editorial,
  citation, primary-source, and standalone-chapter review evidence.

The three sealed PDFs are the publication-of-record visual editions. Files under
`reports/ai_financial_system/chapters/` predate the consolidated release and are
retained only as chapter-history artifacts; use the release directory for current
standalone chapters.
