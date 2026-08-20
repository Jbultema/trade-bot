# Editorial and Investigative Standard

This file governs every new chapter and every revision of an already delivered
chapter. It converts the monograph from a collection of well-sourced narratives into
a versioned analytical system whose observations, reconstructions, interpretations,
falsifiers and economic significance can be audited separately.

## 1. Mandatory chapter architecture

Every post-methodology chapter must use the following visible sequence.

### Observed facts

Only evidence-grade A and B statements appear here. Each material statement must map
to an atomic claim and source locator. A fact may describe what a party filed,
contracted, paid, recognized or announced. It may not silently convert an award
ceiling into cash, an enterprise value into purchase consideration, a target into a
commitment, or a contractual maximum into expected exposure.

### Reconstruction

This section joins observed facts across entities, instruments and dates. It contains
the capital bridge, contract chain, ownership/use separation, cash waterfall,
dependency map and measurement reconciliations. Strong inferences are marked
evidence-grade C at the point of use.

### Interpretation

This section explains economic significance without changing the evidentiary status
of the underlying claims. It must state whether an implication is immediate,
structural or optional and must distinguish present conditions from a possible
transmission mechanism.

### Remaining unknowns

Every material missing disclosure must be named, assigned materiality, linked to the
source most likely to resolve it and retained in the database until resolved. The
prose list is generated from or reconciled to the chapter unknowns ledger.

### Evidence that would overturn these conclusions

The chapter must identify observable evidence that would materially reverse or
weaken its synthesis. Falsifiers must be specific enough to test. Examples include a
previously undisclosed offsetting guarantee, utilization above the modeled break-even
level, conversion of a contingent facility into unconditional funding, or a loss-
sharing agreement that moves first-loss exposure.

### Evidence still missing

This is a prioritized disclosure queue rather than a generic limitations paragraph.
It must name missing documents or measures such as customer utilization, take-or-pay
floors, GPU depreciation schedules, asset-level internal rates of return, cloud gross
margin, collateral reports or project cash sweeps.

### Confidence evolution

Confidence history is append-only. Each tracked synthesis claim records:

- observation date;
- claim identifier;
- evidence grade and confidence;
- evidence added or removed;
- prior grade and confidence;
- reason for change.

No earlier grade may be invented. For claims whose pre-baseline history was not
preserved, the first entry must say `baseline established; earlier grade history not
retained`. Superseded claims remain visible and link to their replacements.

### Economic importance

Each principal discovery receives two classifications:

- **magnitude:** systemically consequential, high, medium or low;
- **horizon:** immediate, structural or optional.

The classification must explain the transmission channel. Interesting governance or
contract detail is not automatically valuation-relevant. High importance requires a
plausible path to cash flow, capital needs, cost of capital, control, loss allocation
or system capacity.

`Systemically consequential` is reserved for an exposure or mechanism whose impairment
can transmit across at least three independently financed parts of the reconstructed
system, cannot be replaced within the relevant stress horizon, and is large enough to
alter capital availability, system capacity or loss allocation. Centrality or a large
headline amount alone does not meet this standard.

## 2. Economics is a required reconstruction layer

The report must now reconstruct not only ownership, capital and obligations but also,
where evidence permits:

- required and observed utilization;
- unit price and effective realized price;
- useful life and depreciation;
- maintenance and replacement capital expenditure;
- power, networking, memory and other variable cost;
- fixed operating cost;
- financing cost and refinancing assumptions;
- tax and public-incentive effects;
- customer concentration and contract duration;
- required return, project IRR and return on invested capital;
- downside break-even and first-loss threshold.

Unknown inputs remain explicit ranges. The report must not solve an economic model by
silently substituting industry convention for missing contract data.

## 3. Economic reconstruction hierarchy

1. **Observed economics:** filed revenue, cash cost, capex, debt service, utilization,
   contract price, depreciation or return.
2. **Contract-implied economics:** amounts derived directly from disclosed quantities
   and enforceable terms.
3. **Bounded economics:** ranges whose endpoints are supported by separate evidence.
4. **Scenario economics:** transparent assumptions used to test a mechanism.
5. **Unresolved economics:** model cannot be calculated without inventing a material
   input.

Every model output must carry one of these labels.

## 4. Required future ledgers

The research database will add:

- `claim_confidence_history`;
- `claim_economic_importance`;
- `unit_economics_observations`;
- `utilization_observations`;
- `pricing_observations`;
- `depreciation_and_useful_life`;
- `project_return_models`;
- `cash_waterfall_steps`;
- `roic_reconciliations`;
- `counterparty_concentration`;
- `network_centrality_snapshots`;
- `stress_scenarios`;
- `stress_transmission_steps`;
- `system_feedback_loops`;
- `historical_analogue_metrics`.

Generated outputs must retain source claim identifiers, status dates and uncertainty
for every row.

## 5. Stress-test standard

Stress work is transaction-by-transaction rather than narrative. At minimum the
systemic chapter will apply a 20 percent hyperscaler capital-expenditure reduction and
trace:

`hyperscaler project cancellation or deferral → accelerator orders → foundry wafers →
HBM and networking demand → customer deposits and supplier capacity → data-center
acceptance and utilization → warehouse borrowing bases and refinancing → utility load
forecasts and recovery → private-credit marks and investor liquidity`

Each step must state:

- initiating variable and date;
- contract or accounting bridge;
- lag;
- loss-bearing entity;
- balance-sheet and cash effect;
- control or mitigation right;
- evidence grade;
- unresolved offset.

## 6. Model discipline

- A model is not an observed fact.
- A point estimate is prohibited when a material input is an unbounded unknown.
- Enterprise value is not equity paid.
- Revenue commitment is not utilization.
- Reserved capacity is not occupied capacity.
- Depreciation policy is not physical useful life.
- Accounting ROIC is not project IRR.
- A 90 percent inference-price decline is a scenario until observed.
- Network centrality measures structural dependence; it does not by itself predict
  default or bargaining power.
- Historical analogues must be reconstructed with comparable capital intensity,
  leverage, return, oversupply, consolidation and recovery metrics rather than used as
  rhetorical labels.

## 7. Core measurement definitions

These definitions govern chapter prose, database fields and models.

- **Accounting observation:** an amount or state recognized, filed or disclosed by a
  reporting entity under a defined accounting perimeter.
- **Model estimate:** a derived output whose assumptions, method, range and evidence
  basis are separately retained. It never inherits Grade A merely because an input is
  primary evidence.
- **Stock:** a quantity measured at a date.
- **Flow:** a quantity measured over a period.
- **Ceiling:** a maximum legal or policy perimeter, not evidence that the amount was
  committed, drawn or spent.
- **Floor:** an enforceable or analytically bounded minimum.
- **Contingent amount:** an amount whose existence, timing or payment depends on a
  specified trigger.
- **Physical quantity:** capacity or equipment measured in its native unit. It is not
  a monetary exposure until joined to price, funding and ownership.
- **Scenario assumption:** an explicit counterfactual input used to test a mechanism.
  It is not assigned a probability unless a separate calibration supports one.
- **Forecast:** a dated expectation about a future outcome with a stated horizon,
  method and uncertainty.
- **Project IRR:** the discount rate that sets asset- or project-level cash flows to
  zero under a stated capital and terminal-value perimeter.
- **Accounting ROIC:** accounting operating profit after tax divided by an explicitly
  defined accounting invested-capital base.
- **Incremental ROIC:** incremental after-tax operating profit divided by incremental
  capital attributable to the analyzed investment.
- **Economic return:** cash and terminal value earned by a defined capital provider
  after operating, financing, tax and replacement requirements.
- **First-loss threshold:** the utilization, price, cost, value or cash-flow point at
  which a named subordinated or residual claimant begins to absorb economic loss.

Utilization is recorded as a state ladder rather than a single percentage:

`installed → available → delivered → energized → accepted → reserved → consumed →
billable → productive`

The denominator and time window must be stated. Reserved capacity is not consumed
capacity; consumed capacity is not necessarily billable; billable use is not
necessarily productive at a price that earns an adequate return.

Compute demand is classified separately as:

1. **financial demand:** a contract, minimum or payment exists;
2. **physical demand:** equipment or capacity has been delivered and accepted; and
3. **economic demand:** capacity is productively consumed at a price capable of
   supporting the relevant required return.

## 8. Remaining-unknown classification

Every material unknown receives one principal class:

- disclosure gap;
- contractual opacity;
- measurement gap;
- entity or transaction identity ambiguity;
- overlap or double-counting risk;
- temporal-status lag;
- AI-attribution gap;
- model uncertainty; or
- structurally unknowable with current reporting.

An unknown also records which downstream chapters it blocks and the observable event
that would resolve or materially narrow it. `Not disclosed` is a status, not a
resolution plan.

## 9. Common exposure state vector

Every major economic obligation or supported exposure is eventually represented by a
stable `exposure_id` with:

- amount range and currency;
- amount type;
- instrument;
- legal obligor;
- economic beneficiary;
- linked physical project or asset;
- funding and recognition status;
- recourse and seniority;
- maturity and cancellation rights;
- linked guarantee;
- first-loss bearer;
- evidence grade;
- economic importance;
- current confidence;
- falsification condition; and
- as-of date.

`exposure_components` joins that identity to contracts, leases, loans, guarantees,
project cases, compute commitments, utility agreements, private-credit positions and
public-capital observations. Component rows are nonadditive by default. This prevents
one data-center system from being counted independently as a hyperscaler lease, a
developer project, a private-credit loan, a utility load agreement, a customer compute
commitment and a semiconductor demand forecast.

## 10. Chapter handoffs

Every chapter must freeze a structured handoff to later analysis. A handoff names the
dataset or ledger, downstream chapter, required fields, readiness state and missing
evidence. Required core handoffs include:

- Chapter 2 financing cost, duration, seniority, recourse and first-loss position to
  Chapter 13;
- Chapter 3 demand floors and utilization states to Chapter 14;
- Chapter 4 asset cost and cash-waterfall inputs to Chapter 13;
- Chapter 5 power cost, load-ramp and stranded-cost exposure to Chapters 12 and 17;
- Chapter 6 laboratory cash obligations and funding dependencies to Chapters 12, 13
  and 17;
- Chapter 7 capital stock, depreciation and funding-capacity inputs to Chapter 13;
- Chapter 8 hardware cost, useful life, substitution and obsolescence to Chapter 14;
- Chapter 9 leverage, marks, covenants and liquidity mechanics to Chapter 17; and
- Chapters 10 and 11 public support, cross-ownership and circularity controls to
  Chapters 13, 16 and 17.

`Blocked` is an acceptable handoff result. A chapter must not invent an input merely
to make the downstream model calculable.

## 11. Chapter-ending control block

The full mandatory architecture remains visible in the body. The end of every
post-methodology chapter also contains a compact control block that points to, rather
than repeats, the relevant sections:

1. highest-confidence findings;
2. economic significance;
3. evidence that would falsify the findings;
4. missing evidence;
5. confidence changes; and
6. implications and data handed to later chapters.

## 12. Revision protocol for delivered chapters

Chapters 1 through 9 remain delivered artifacts but are not publication-final. Each
will receive a numbered revision that:

1. preserves the original frozen evidence package;
2. records new or superseding claims;
3. inserts the mandatory architecture;
4. adds chapter-specific falsifiers and missing evidence;
5. establishes a confidence-history baseline;
6. classifies the economic importance of principal discoveries;
7. regenerates the DOCX and manifest;
8. records both old and new artifact hashes.

The regenerated chapter package must pass
`chapter_docx_publication_standard.md`; successful content conversion alone is not
DOCX acceptance.

The revision must not imply that later evidence was known at the original cutoff.
