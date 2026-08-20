# Chapter 1 Research Pass — Methodology

Status: frozen for first writing pass on 2026-07-25.

## Research question

What research architecture is required to reconstruct the AI financial system without
confusing publicity with obligation, accounting recognition with economic dependence,
or a forecast with a funded asset?

## Documentary systems established

| System | What it can establish | What it cannot establish alone |
|---|---|---|
| SEC EDGAR submissions | Filing history, registration statements, periodic reports, ownership forms, exhibits | Complete private-company obligations or undisclosed contract economics |
| SEC XBRL APIs | Standardized recognized financial facts across filings | Bespoke footnotes, custom tags, contract terms, economic dependence |
| SEC MD&A and off-balance-sheet framework | Expected disclosure terrain for commitments, capital resources, guarantees, covenants, and unconsolidated arrangements | That every economically relevant dependency is quantified |
| FERC eLibrary/eTariff | Public dockets, tariffs, service agreements, regulated rate and interconnection records | Redacted commercial terms, state-only matters, unregulated contracts |
| EMMA | Official statements and continuing disclosures for municipal financing | Private debt, bank facilities, or project obligations outside municipal securities |
| USAspending | Federal awards, recipients, agencies, award values and outlays | That announced or obligated award value has been fully disbursed |
| EIA | Historical energy data and explicit demand scenarios | A particular data-center contract, committed project, or funded capacity |

## Discovery from the research pass

The first system-level finding is methodological but consequential: no single official
database measures “AI investment.” Each registry observes a different legal or
accounting surface. EDGAR centers the reporting entity; FERC centers regulated energy
transactions; EMMA centers securities issued by public-sector obligors; USAspending
centers federal awards; EIA centers energy statistics and modeled scenarios. A system
reconstruction therefore requires entity resolution and transaction matching across
registries before amounts can be aggregated.

The second finding is that official status does not eliminate category risk. An SEC
filing may contain an enforceable obligation, management's estimate, or a risk factor in
the same document. A FERC docket can contain an application rather than an approved
agreement. A federal award can be obligated but not outlaid. An EIA number can be a
scenario. The database therefore records both source grade and transaction status.

## Frozen analytical controls

### Evidence is atomic

Each claim is stored separately. A sentence containing a disclosed fact and an
interpretation becomes at least two claims with different evidence grades.

### Evidence grade is not confidence

A company press release is Grade A because it is a primary statement by the company,
but it may support only the fact that the company made an announcement. It does not
automatically verify enforceability, funding, or accounting treatment.

### Status ladder

Every transaction is assigned the highest status directly established:

1. announced
2. committed
3. contracted
4. funded
5. drawn
6. spent
7. recognized

`guaranteed` and `contingent` are orthogonal flags rather than later stages.

### Amount semantics

Amounts are stored with currency, date, scope, and one of:

- exact disclosed amount
- disclosed range
- lower bound
- upper bound
- researcher estimate
- unknown

Gross project cost, contract ceiling, equity value, debt principal, cash funded, and
recognized accounting balance are never treated as interchangeable.

### Dependency classes

- accounting
- contractual
- funding
- operational
- control

The project will measure overlap among these classes rather than assume they coincide.

## Chapter acceptance gate

Chapter 1 may be written because it leaves behind:

- a relational schema covering sources, entities, claims, flows, contracts, debt,
  leases, guarantees, investments, projects, power, compute, ownership, dependencies,
  risk pathways, and unknowns;
- a source register with explicit limitations;
- atomic claims and evidence links;
- a reproducible SQLite build and validation process;
- a documented chapter workflow and unresolved-question ledger.

## Remaining methodological unknowns

See `data/unknowns.csv`. The most material are confidential contract enforceability,
attribution of aggregate purchase obligations, state utility special-contract access,
and the share of private project finance invisible to free public records.

## Publication-standard revision — 2026-07-25

The first writing pass predated the expanded economics and falsification standard.
Revision 2 preserves the original evidence claims and adds:

- systemically consequential, high, medium and low economic-importance definitions;
- an append-only confidence-history baseline for all eight Chapter 1 claims;
- explicit falsification and missing-evidence sections;
- remaining-unknown classifications and downstream blocking fields;
- accounting-observation versus model-estimate definitions;
- stock, flow, ceiling, floor, contingent and physical-quantity semantics;
- scenario-assumption versus forecast separation;
- project IRR, accounting ROIC, incremental ROIC and economic-return definitions;
- a named first-loss threshold;
- installed-through-productive utilization states;
- financial, physical and economic demand;
- a common economic-exposure identity and nonadditive component links; and
- structured chapter handoffs with ready, bounded, blocked and not-started states.

No pre-baseline confidence grades were recreated. Every confidence row states that
earlier grade history was not retained.

## Revision 2 writing gate

The generated evidence package at
`reports/ai_financial_system/chapters/chapter_01/data/` contains:

- 8 graded claims;
- 4 unresolved methodology questions;
- 8 confidence-history baselines;
- 8 economic-importance classifications;
- the status distribution for 17 chapter handoffs; and
- 8 passed methodology writing gates.

The revised DOCX and manifest validate against schema version 2 and database SHA-256
`aee9e5a4f756f757f1e92cd954f50a4a84d84d79a51bbceecfc62d3ca09a25bd`.
