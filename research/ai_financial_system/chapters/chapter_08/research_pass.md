# Chapter 8 Research Pass

## Provisional title

**Semiconductor Control — Allocation, Foundries, Memory, Packaging, Networking, and
Supplier Finance**

## Research question

Who can actually authorize, manufacture, assemble, finance, deliver, substitute or
withhold the semiconductor systems required by the AI build—and how do volume
commitments, customer incentives, capacity finance and technical lock-in determine
where a demand shortfall or supply interruption is absorbed?

## Pass 1 scope

Reconstruct the control stack rather than treating “GPU supply” as a single market:

1. accelerator architecture, software compatibility and customer qualification;
2. foundry wafer starts, process nodes, geographic concentration and allocation;
3. advanced packaging, substrates, test and assembly;
4. high-bandwidth memory and other critical memory;
5. networking, interconnect, optics, switches and rack integration;
6. manufacturing and capacity commitments, prepayments and inventory;
7. customer warrants, rebates, vendor finance, lease guarantees and residual-capacity
   support;
8. export controls, licenses and product redesign;
9. second-source and substitution limits at each layer;
10. failure and demand-revision pathways from customer to supplier, financier and
    fabrication capacity.

## Initial entity set

- NVIDIA
- Advanced Micro Devices
- Broadcom
- Taiwan Semiconductor Manufacturing Company
- SK hynix, Micron and Samsung Electronics
- ASML
- leading packaging, substrate, networking and optical counterparties where
  documentary evidence supports a material dependency

The chapter will not manufacture a false company-by-company symmetry. Entities enter a
comparison only for the layer they control or materially finance.

## Required reconstruction bridges

1. **Manufacturing bridge:** disclosed supply/capacity commitment → firm purchase
   order → wafer or component receipt → inventory → product shipment → cash payment.
2. **Physical bridge:** accelerator → HBM → package → substrate → rack → switch /
   network → powered deployment.
3. **Control bridge:** design ownership → foundry allocation → packaging allocation →
   customer qualification → software and system switching cost.
4. **Finance bridge:** supplier prepayment or commitment → capacity investment →
   customer backlog or warrant → vendor backstop → residual-value exposure.
5. **Geographic bridge:** country and site concentration → export or disruption event
   → inventory buffer → alternate process / supplier → time and performance penalty.
6. **Demand bridge:** customer cancellation or slower deployment → supplier purchase
   commitment → inventory / capacity charge → foundry and memory utilization →
   financing or guarantee loss.

## Non-negotiable measurement rules

- Foundry revenue is not AI accelerator value.
- A manufacturing commitment is not inventory and is not necessarily unavoidable.
- Wafer capacity is not finished-system capacity.
- Packaging capacity is not interchangeable across package types.
- HBM bit output is not equivalent to qualified HBM stacks delivered in a system.
- Customer backlog is not collected cash or guaranteed utilization.
- A customer warrant is not revenue and unvested shares are not customer ownership.
- A maximum lease guarantee is not expected loss.
- Market share estimates are secondary evidence unless reproduced from primary data.
- Announced fabs and capacity expansions remain separate from funded, installed,
  qualified and volume-producing capacity.
- Export-control exposure must be tied to the rule and product period in effect; it
  cannot be generalized across generations.

## Existing evidence to carry forward

The frozen database already establishes several starting facts that require deeper
supplier-layer decomposition:

- NVIDIA disclosed $119 billion of manufacturing supply and capacity commitments,
  $30 billion of primarily R&D cloud commitments, $27 billion of contingent
  investment commitments, $18.6 billion of first-quarter private-company and
  infrastructure-fund investments, and up to $3.5 billion of partner-facility lease
  guarantees at 26 April 2026.
- AMD disclosed $25.662 billion of unconditional commitments across wafers,
  substrates, components, cloud, software and licenses, plus $4.1 billion of maximum
  commercial-partner data-center lease guarantees at 28 March 2026.
- AMD's OpenAI and Meta agreements each include a warrant for up to 160 million shares,
  with initial vesting linked to a binding one-gigawatt purchase and later vesting
  linked to larger deployments and other conditions.
- Broadcom disclosed $128.11 billion of enforceable purchase commitments, $164.6
  billion of RPO, approximately 95% TSMC wafer concentration, and a subsequent
  Apollo-linked rack-finance structure with up to a $29 billion customer-lease
  backstop.

These balances are not additive. The research pass must identify overlap, maturity,
adjustability, named counterparties and physical layer before using them in a system
map.

## Primary-source queue

- latest NVIDIA, AMD and Broadcom periodic filings and material contracts;
- TSMC annual report, quarterly materials and capacity / geographic disclosures;
- Micron, SK hynix and Samsung filings or audited annual reports covering HBM,
  prepayments, customer concentration and capacity investment;
- ASML annual and interim filings covering EUV/DUV backlog, customer concentration,
  installed-base dependency and export controls;
- material supplier and customer filings that identify packaging, substrate,
  networking, optics or rack-finance counterparties;
- BIS rules, license notices and official government documents for export-control
  pathways;
- foundry, memory and equipment earnings materials only where they add measurements
  not available in audited or filed statements.

## Writing gate

**Status: PASSED on 2026-07-25.**

The generated nine-row gate audit is frozen at
`reports/ai_financial_system/chapters/chapter_08/data/writing_gate_audit.csv`.
Passing authorizes evidence-led drafting; it does not resolve the retained unknowns or
permit Grade B or C evidence to be presented as Grade A.

Chapter 8 cannot enter writing until:

- latest-source currency is verified for every core public company;
- at least one physical bill-of-dependencies map is frozen;
- manufacturing commitments are reconciled separately from inventory and capex;
- foundry, packaging and HBM control points each have primary evidence and explicit
  unknowns;
- NVIDIA, AMD and Broadcom financing or incentive channels are separated from product
  revenue;
- geographic and export-control pathways are tied to dated rules and products;
- supplier concentration is not inferred from unnamed commitments;
- at least three demand-downside pathways identify who holds inventory, capacity,
  credit or residual-value exposure;
- all material statements are atomic claims linked to sources;
- contradictions and disclosure gaps are preserved.

## Pass 1A — The first control map is serial, not hierarchical

### NVIDIA's named supplier set does not establish fungibility

NVIDIA names TSMC and Samsung as wafer foundries; SK hynix, Micron and Samsung as
memory suppliers; CoWoS as a packaging technology; and Hon Hai, Wistron and Fabrinet
among its final-product assembly, testing and packaging providers. This is documentary
evidence of multiple named suppliers, but not evidence that any one finished
architecture can be moved freely among them.

The current physical control chain is:

`architecture and software → qualified logic wafer → qualified HBM → package and
substrate → test → rack integration → network fabric → powered customer deployment`

Each arrow contains a design, yield, timing or customer-acceptance dependency. The
public record does not yet disclose supplier shares, architecture-specific bills of
materials, capacity reservations or alternate-qualification schedules.

### A real loss episode shows where demand risk lands

The most useful downside observation is not a hypothetical shortage. NVIDIA recorded a
$4.5 billion charge for H20 excess inventory and purchase obligations after the United
States imposed a license requirement and demand diminished. The event establishes a
complete transmission direction:

`export rule → customer market closes → finished and committed supply exceeds demand
→ supplier records inventory / purchase-obligation charge`

The filing does not allocate the charge among completed products, wafers, HBM,
packaging or other obligations. It nevertheless proves that fabless architecture
control does not eliminate upstream fixed commitment risk.

NVIDIA's $119 billion manufacturing supply and capacity commitment at 26 April 2026 is
therefore best understood as a large risk-bearing perimeter, not as inventory and not
as a single enforceable liability. Some supplier arrangements can be cancelled,
rescheduled or adjusted before firm orders. Chapter 8 must decompose the commitment by
physical layer and firmness before drawing exposure conclusions.

## Pass 1B — Foundry and packaging control separate

TSMC states that technology development, product preparation, capacity build and
high-volume ramp can take more than five years. It plans with both customers and its
customers' customers. That planning visibility gives TSMC information and allocation
influence beyond an ordinary contract manufacturer, but the public record does not
show what portion is protected by deposits, minimum volumes or loss sharing.

The front end and back end must be modeled separately. TSMC said in July 2026 that
advanced-packaging capacity was tight enough to limit customer growth. It also refused
to isolate packaging capex from a broad 10%–20% bucket containing packaging, testing,
mask making and other items because the active bottleneck can move. A completed wafer
therefore is not a completed accelerator, and aggregate wafer capacity can overstate
deliverable AI-system capacity.

TSMC's first-half cash property acquisitions were $26.8 billion. At 30 June, net PP&E
was $134.811 billion and payables to contractors and equipment suppliers were $9.112
billion. Those are three accounting clocks—period cash, ending productive and
construction assets, and unpaid investment—not a single capex number.

The concentration is explicit for competitors:

- AMD states that TSMC produces all of its microprocessor and GPU wafers at
  7 nanometers or smaller.
- Broadcom states that TSMC produced approximately 95% of its contract-manufactured
  wafers over the two quarters ended 3 May 2026.

Neither disclosure gives a time-to-substitute estimate. “Second supplier exists” must
not be treated as “second supplier is qualified for this product at commercial volume.”

## Pass 1C — Memory customers are becoming capacity financiers

Micron's new strategic customer agreements are structurally more important than an
ordinary memory backlog. Management described roughly $100 billion of remaining
contractual value and expected $22 billion of customer deposits and related financial
commitments, including approximately $18 billion of cash deposits. The agreements use
take-or-pay volumes and usually fixed prices or price bands.

The filed accounting perimeter is much smaller. At 28 May 2026, RPO was approximately
$5 billion and recognized contract liabilities were $422 million. RPO uses minimum
committed volume and minimum pricing; agreements without fixed prices or price bands
and contracts of one year or less are excluded. Some of the larger agreements were
also executed after quarter end.

The apparent discrepancy is not an error and the amounts are not additive:

`multi-contract economic headline ≠ period-end accounting RPO ≠ cash already received`

The reconstruction must identify agreement date, product, volume, price floor, deposit
due date, refundability and accounting inclusion.

Customer deposits now sit beside operating cash, government incentives and corporate
financing in Micron's capacity stack. Yet physical delivery remains delayed: the first
new Boise fab targets initial wafer output in mid-2027, a second Boise fab late 2028,
the first New York fab 2030 and expanded Singapore HBM packaging beginning in the
first half of 2027. Customer finance can reduce funding risk before it reduces
construction, qualification or ramp risk.

## Pass 1D — The network layer has the same commitment asymmetry as accelerators

### Arista converts revocable demand forecasts into firm supplier exposure

Arista's March-quarter filing closes an important gap in the serial control map. Its
network platforms require merchant silicon, memory, components, contract manufacturing,
software and customer acceptance. At 31 March 2026, Arista had $8.9 billion of
enforceable noncancelable purchase commitments outside its balance sheet. Of that
amount, $7.6 billion had expected receipt dates inside twelve months. It also had $95.5
million of deposits with contract manufacturers.

The downstream side is materially less firm. Customers can often cancel, delay, reduce
or modify purchase orders with little or no notice. Arista therefore forecasts demand
and commits upstream before equivalent downstream certainty exists. It records a
liability for supplier commitments above forecast demand and reimburses contract
manufacturers for excess or obsolete components purchased on its behalf.

This is not merely boilerplate. The company says the mechanism has produced and may
again produce excess- and obsolete-inventory charges. The exact current AI-network
charge is not disclosed, so it remains an historically observed but unquantified loss
mechanism rather than a discrete realized episode.

Concentration amplifies the asymmetry. Arista's two largest unnamed end customers were
16% and 26% of 2025 revenue—42% combined. Their identity, AI share, product allocation
and matching upstream commitments are undisclosed. Qualification and acceptance can be
lengthy, which gives customers another timing control even after suppliers have begun
building components.

The loss direction is therefore:

`large-customer forecast → Arista noncancelable supplier order → customer changes order
→ excess component or finished-good exposure → Arista liability / contract-manufacturer
reimbursement`

### NVIDIA is financing two adjacent network-control suppliers

The network and optics layer is not independent of the accelerator capital stack.
NVIDIA invested $2.0 billion in Marvell preferred stock while the companies aligned
Marvell custom XPUs, compatible scale-up networking and optical interconnect technology
with NVIDIA's infrastructure ecosystem. Marvell then entered agreements to pay $870
million of deposits for wafer capacity through fiscal 2030 and substrate capacity
through fiscal 2033.

No source proves that NVIDIA's dollars specifically fund those deposits. Treating them
as matched sources and uses would be a false reconstruction. The documented structure
is still important:

`accelerator vendor equity → adjacent supplier corporate capital`

and separately:

`adjacent supplier cash → long-duration wafer and substrate reservation`

Marvell's exposure is already economically concentrated: $1.833 billion, or 76%, of
first-quarter fiscal 2027 revenue came from the data-center end market. AI-related
demand spanned electro-optics, custom silicon, storage and switching, so the filed
category cannot be reduced to a single product or customer.

NVIDIA also invested $2.0 billion of common equity in Coherent and separately entered a
nonexclusive, multi-year, multi-billion-dollar purchase and capacity agreement for
advanced laser and optical-networking products. Coherent says the capacity program may
require additional equipment, labor and working capital through 2030. Its
indium-phosphide expansion in Sherman, Texas responds to an industry-wide shortage.

This creates a three-role relationship:

1. NVIDIA is an equity financier;
2. NVIDIA is a customer with future access and capacity rights;
3. NVIDIA is the architecture controller whose system roadmap influences the value of
   the capacity.

The structure can reduce supplier funding risk and improve NVIDIA's access to scarce
inputs. It also makes the apparent demand and capital channels correlated. The equity
investment is not revenue, and the purchase commitment is not equity support. Their
economic dependence must be analyzed jointly while their accounting remains separate.

## Pass 1E — A memory-price shock is not a single price

Current evidence supports three different clocks:

1. **physical cost:** newer DRAM, NAND and HBM generations require more complex
   processes and can carry higher bit costs;
2. **market price:** conventional DRAM contract prices can reprice quarter by quarter;
3. **strategic-contract price:** HBM and other products under longer agreements may
   use annual or separately negotiated product-generation pricing.

Micron stated that greenfield capacity and product transitions were expected to raise
blended DRAM cost per bit and that price premiums for future products under strategic
customer agreements remained subject to negotiation. This is Grade A evidence of
management's expected cost and contract mechanism—not a disclosed customer price.

TrendForce's Grade B market research estimated that conventional server DRAM
profitability had exceeded HBM in early 2026 because annual HBM pricing lagged rapid
quarterly increases elsewhere in memory. It separately projected conventional DRAM
contract prices up 58%–63% quarter over quarter in second-quarter 2026 while suppliers
shifted capacity toward HBM and server products.

The implication is not simply “memory became more expensive.” A supplier can face a
temporary incentive to allocate scarce wafers toward a conventional product with
faster repricing even when long-run HBM demand is stronger. The customer that reserved
HBM under a longer agreement can be protected temporarily; the next-generation price
reset can then transfer higher cost later. Chapter 8 must preserve the dated and
secondary nature of the market estimates.

## Pass 1F — Installed equipment and legal exportability are separate capacity states

### ASML sells both machines and continuing access to productive capability

ASML sold 91 lithography systems in second-quarter 2026, but system count alone does
not establish process capacity. It also generated €2.8 billion of installed-base
management revenue from service and field options. The recurring channel is nearly 30%
of the quarter's €9.326 billion total sales.

The most useful qualification example is narrow rather than aggregate. ASML and Intel
reported that High-NA EUV had been qualified on selected Intel 18A product layers. The
statement does not establish foundry-wide qualification, qualification at TSMC or
Samsung, or availability for an accelerator or HBM process. The capacity ladder is:

`tool built → tool delivered → tool installed → process integrated → selected layer
qualified → yield ramped → product qualified → commercial wafer output`

Service, field options and productivity upgrades can improve or preserve output after
installation. A shipment count therefore overstates economically usable capacity if
the installed base lacks process acceptance, service entitlement, required upgrades or
export authorization.

### Export control is a dated product-and-customer state machine

The H20 charge cannot be generalized into a timeless “China restriction.” Three
separate states are now preserved in the regulatory ledger:

1. NVIDIA was notified in April 2025 that H20 exports to China required a license; the
   resulting market impairment contributed to a $4.5 billion inventory and
   purchase-obligation charge.
2. BIS announced in January 2026 that H200, MI325X and similar applications for China
   would receive case-by-case review subject to customer controls, United States
   testing and a showing that exports would not reduce production capacity available
   to United States customers.
3. BIS clarified in May 2026 that covered advanced-computing transactions involving an
   entity headquartered or ultimately parented in Country Group D:5 or Macau could
   require a license even when the entity itself was located elsewhere.

The governing variable is therefore not just shipment destination. Product
classification, generation, customer identity, ultimate parent, use, license posture
and policy date can each determine whether inventory is commercially usable. No public
record in the current package identifies individual license decisions or shipment
volumes.

## Frozen dataset status after Pass 1F

The Chapter 8 database and generated evidence package now contain:

- 24 physical dependency observations;
- 19 capacity-finance observations;
- 15 substitution constraints;
- three dated regulatory-control states;
- eight explicit measurement reconciliations;
- one quantified realized loss episode;
- four conditional or unquantified loss pathways;
- an eight-stage serial physical-control map;
- current-source audits for eleven core public companies;
- 31 atomic claims and 17 unresolved questions.

No cross-company capital total is calculated. The finance ledger deliberately preserves
cash spending, ending assets, payables, deposits, RPO, purchase commitments, equity
proceeds and investment visions as different measurement classes.
