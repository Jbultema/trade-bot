# Chapter 14 Substantive Acceptance

Status date: 2026-07-25

## Decision

**ACCEPT AS A CONTROLLED DIAGNOSTIC INPUT PACKET.**

The corrected Chapter 14 packet in `staging/lane_3_chapter_14/` resolves all
six findings in the independent adversarial review. It is suitable for the
Chapter 14 writing pass only if the publication preserves the boundaries below.
This decision does not accept a productive-utilization estimate, realized
inference price, economic useful life, residual value, break-even curve, or
survivability result.

## Accepted results

- A nine-state capacity vocabulary from installed through productive use.
- Four limited contract-implied diagnostics, with display precision separated
  from internal arithmetic.
- Five nonadditive 20 percent capex counterfactuals calculated against each
  company's disclosed native-period measure.
- A governed scenario schema that names lag, affected contract, loss-bearing
  entity, and recovery mechanism, using `unknown` where evidence is absent.
- A River Bend customer-default scenario design that names disclosed Google
  support but remains blocked because the cap, waterfall, remedies, and
  recovery timing are not public.
- Reproducible cost-curve equations and an explicit missing-input register.

## Publication boundaries

1. The packet finds **no defensible bounded utilization range**. Installed,
   energized, reserved, paid, and productive capacity must not be treated as
   synonyms.
2. Approximately $3,846 per stated GPU-month is contract arithmetic, not a
   realized inference price or productive-unit price.
3. The 1.30, 1.91, and 2.00 kW-per-processor figures are facility-input
   diagnostics, not chip TDP, utilization, interval energy use, or performance
   per watt.
4. River Bend's approximately 1.35 design-capacity ratio is not measured PUE.
5. Saline's 1,106.4 MW minimum billing demand is a contractual floor, not
   energized load, energy consumption, or productive compute.
6. The five capex shocks are exact scenario arithmetic in incompatible company
   measures. They are not AI-capex estimates, cancellations, delayed capacity,
   supplier-revenue shocks, or additive system totals.
7. The 90 percent inference-price shock remains blocked because no matched
   realized price, workload, cost, utilization, economic-life, and residual
   evidence exists.

## Revalidation

The primary integration lane reran the staging validator:

- 16 manifest-declared files;
- 259 CSV rows;
- strict CSV width and provisional-ID uniqueness checks passed;
- manifest hashes and row counts passed;
- canonical source and claim references passed.

The primary lane also confirmed that:

- every no-output scenario is explicitly blocked;
- every scenario has the four structured governance fields;
- the useful-life scenario contains no invented one-year shock;
- `PWR-14013` and `PWR-14014` match their Grade-C source claim;
- all five executed capex rows are labeled
  `scenario_company_measure_arithmetic`;
- all incomplete analytical handoffs are blocked; and
- the only ready handoff is a vocabulary/control artifact, not an economic
  output.

## Downstream use

Chapters 12, 13, 16, and 17 may use the schemas, definitions, limited
diagnostics, and missing-evidence register. They may not convert the packet's
blocked fields into point estimates or probabilities.
