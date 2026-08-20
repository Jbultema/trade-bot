# Chapter 14 Research Pass

## Scope and disposition

This is a research-only utilization and cost-curve packet. It does not contain
publication prose, a utilization estimate, a realized inference-price estimate, an
asset survival conclusion, or an investment conclusion.

The evidence supports a state ledger, several contract-implied diagnostics, five
company-measure 20% capex counterfactuals, and a reproducible equation set. It does not
support a bounded productive-utilization range or a completed 90% inference-price
survivability result.

## Governing distinctions

The packet keeps the following capacity states separate:

`installed -> available -> delivered -> energized -> accepted -> reserved -> consumed
-> billable -> productive`

It also separates:

- financial demand: a contract, commitment, minimum bill, prepayment, or reservation;
- physical demand: delivered and accepted capacity or measured load;
- economic demand: paid productive use at a realized price above attributable cost;
- accounting depreciation, tax depreciation, physical life, economic life, and
  residual value;
- processor counts, IT megawatts, utility megawatts, energy, service units, and money;
- provider-controlled operation from customer acceptance and productive workload use.

## Evidence result

### No bounded utilization range

The canonical utilization table contains no observations. The strongest direct
physical evidence is SpaceX's statement that three Colossus clusters were operational:
approximately 100,000 H100 processors at 130 MW, 110,000 GB200 processors at 210 MW,
and 110,000 GB300 processors at 220 MW (CLM-030002). This supports installed,
energized, and provider-available states within the disclosed operating perimeter. It
does not disclose matched accepted, consumed, billable, or productive numerator-hours.

The Anthropic service agreement names approximately 325,000 GPUs and a $1.25 billion
monthly fee through May 2029 after reduced ramp fees (CLM-060013). The specifically
enumerated operational clusters total approximately 320,000 processors, and the
canonical reconciliation prohibits a one-for-one mapping (CLM-030003). The contract
therefore supplies financial-demand and contract-price boundaries, not a utilization
denominator.

Across 24 normalized compute relationships, four were documented as fully delivered,
five partial, 12 not delivered, two unresolved, and one not project-specific
(CLM-030023). Three were documented as utilized or paid, two partial, 12 not yet at
that stage, and seven unknown (CLM-030024). These are case counts with heterogeneous
units, not capacity-weighted utilization rates.

### Contract-implied diagnostics

The following arithmetic is retained only in the applicable ledgers:

- $1.25 billion per month divided by approximately 325,000 stated GPUs equals
  about $3,846 per stated GPU-month. This is not a realized inference
  price, invoice rate, or productive-unit price.
- Colossus facility input ratios are 1.300 kW per disclosed H100 processor,
  about 1.91 kW per disclosed GB200 processor, and 2.00 kW per disclosed
  GB300 processor. These are facility-level disclosed-MW-to-processor ratios, not chip
  TDP, energy use, utilization, or performance per watt.
- River Bend's 330 MW utility design divided by 245 MW IT capacity is approximately
  1.35. It is a design-capacity ratio, not measured PUE or operating efficiency.
- Saline's 80% minimum billing demand against 1,383 MW is 1,106.4 MW
  (CLM-050028). It is a contractual billing-demand floor, not MWh, energized load, or
  productive compute.

The ledgers retain full arithmetic precision internally but carry separate rounded
display values and measurement-basis fields. Exact arithmetic does not upgrade the
precision of approximate GPU counts, stated facility scope, or design capacities.

### Product, power, life, and residual inputs remain incomplete

The semiconductor record proves a serial dependency across qualified logic, HBM,
advanced packaging, substrates, test, rack integration, networking, and powered
deployment (CLM-080013). It does not provide a matched bill of materials, delivered
unit cost, rack cost, network cost, accelerator performance, or replacement cadence.
Aggregate manufacturing commitments and supplier deposits are retained as context only.

The power record provides capacity and cost-allocation mechanisms but no matched
site-period tariff, interval MWh, fuel cost, or productive-output denominator. The
Saline full-revenue-requirement mechanism and CCAC excess-power-cost reserve establish
who can bear cost, not the cost per productive unit.

No canonical source provides a compute cohort's economic life or market residual.
Contract terms, debt maturities, accounting depreciation, the NVIDIA H20 charge, and
Hyperion's property residual-value-guarantee threshold are explicitly excluded as
substitutes.

## Required scenario tests

### 20% hyperscaler capex reduction

The 20% shock is calculable only against each observed company measure in its native
period and perimeter. The packet calculates five nonadditive company-measure scenario
arithmetic outputs:

- Microsoft nine-month cash PP&E additions: $80.146bn to $64.1168bn;
- Amazon trailing-twelve-month cash property purchases: $147.299bn to $117.8392bn;
- Alphabet first-quarter cash PP&E purchases: $35.674bn to $28.5392bn;
- Meta first-quarter company-defined capex: $19.840bn to $15.872bn; and
- Oracle fiscal-2026 capex: $55.663bn to $44.5304bn.

These outputs are exact arithmetic under an explicit counterfactual. None measures AI
capex, an order cancellation, delayed MW, supplier revenue, or capex-to-capacity
transmission. Those transmission steps remain blocked by AI allocation, project
mapping, unit cost, cancellation rights, inventory, lead time, and financing
dependencies.

### 90% inference-price decline

The required shock multiplier is 0.10. No canonical evidence supplies a realized
inference price for a defined workload, service unit, customer, and period. Therefore
the packet reports the equation and missing-variable dependency but no shocked price,
revenue, utilization, margin, asset-survival, or residual-value output. The Anthropic
headline GPU-month diagnostic is not used as an inference-price proxy.

### Scenario governance and customer default

Every scenario row now records a structured lag, affected contract, loss-bearing
entity, and recovery mechanism, using `unknown` when evidence is absent. Scenarios
without a material baseline or output remain blocked, and no probability is inferred.
The useful-life-compression row carries no shock magnitude because neither a
cohort-specific base economic life nor a justified compression parameter is available.

The customer-default design names River Bend's 245 MW triple-net take-or-pay base
lease and Google's disclosed support for lease pass-through and expected
operating-services obligations. The support cap, default amount, claimant waterfall,
remedies, recovery timing, and replacement-tenant path remain unknown. The scenario is
therefore blocked and emits no loss or recovery output.

## Handoff result

Chapter 14 can return:

- a reconciled state vocabulary and observation ledger;
- limited contract-implied diagnostics with rounded displays;
- five nonadditive company-measure scenario arithmetic outputs;
- cost-curve equations with explicit missing inputs;
- falsifiers and a prioritized primary-evidence queue.

These are qualitative or diagnostic research materials, not economic handoff
readiness. Chapter 14 cannot yet return productive utilization, realized inference
price, a performance-adjusted cost curve, break-even utilization, economic useful
life, market residual, customer-default loss allocation, or 90%-price-shock
survivability. Those fields remain blocked for Chapters 12, 13, 16, and 17.

## No-write attestation

All outputs from this assignment are confined to
`research/ai_financial_system/staging/lane_3_chapter_14/`. No canonical CSV, schema,
database, final chapter file, publication manifest, shared script, or other lane was
edited.
